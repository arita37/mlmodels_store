
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

  ['model_tch.transformer_sentence', 'model_tch.matchzoo_models', 'model_tch.textcnn', 'model_tch.pytorch_vae', 'model_tch.torchhub', 'model_tch.nbeats', 'model_tch.pplm', 'model_tch.transformer_classifier', 'model_tch.mlp', 'model_tch.03_nbeats_dataloader', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_keras.namentity_crm_bilstm', 'model_keras.textcnn', 'model_keras.nbeats', 'model_keras.armdn', 'model_keras.01_deepctr', 'model_keras.textvae', 'model_keras.02_cnn', 'model_keras.charcnn_zhang', 'model_keras.charcnn', 'model_keras.Autokeras', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.keras_gan', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model'] 

  Used ['model_tch.transformer_sentence', 'model_tch.matchzoo_models', 'model_tch.textcnn', 'model_tch.pytorch_vae', 'model_tch.torchhub', 'model_tch.nbeats', 'model_tch.pplm', 'model_tch.transformer_classifier', 'model_tch.mlp', 'model_tch.03_nbeats_dataloader', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_keras.namentity_crm_bilstm', 'model_keras.textcnn', 'model_keras.nbeats', 'model_keras.armdn', 'model_keras.01_deepctr', 'model_keras.textvae', 'model_keras.02_cnn', 'model_keras.charcnn_zhang', 'model_keras.charcnn', 'model_keras.Autokeras', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.keras_gan', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model'] 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
Iteration:   0%|          | 0/29440 [00:00<?, ?it/s][A
Iteration:   0%|          | 1/29440 [00:19<157:26:21, 19.25s/it][A
Iteration:   0%|          | 2/29440 [00:39<160:35:34, 19.64s/it][A
Iteration:   0%|          | 3/29440 [00:51<140:34:58, 17.19s/it][AKilled

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
From github.com:arita37/mlmodels_store
   f486680..9e0423b  master     -> origin/master
Updating f486680..9e0423b
Fast-forward
 .../20200522/list_log_pullrequest_20200522.md      |   2 +-
 ...-09_51b64e342c7b2661e79b8abaa33db92672ae95c7.py | 617 +++++++++++++++++++++
 2 files changed, 618 insertions(+), 1 deletion(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-22-04-09_51b64e342c7b2661e79b8abaa33db92672ae95c7.py
[master 3b7f4e4] ml_store
 2 files changed, 80 insertions(+), 11046 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   9e0423b..3b7f4e4  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 23106749.41B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 287499.39B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 4177920/440473133 [00:00<00:10, 41778841.41B/s]  2%|▏         | 7916544/440473133 [00:00<00:10, 40355666.43B/s]  3%|▎         | 11634688/440473133 [00:00<00:10, 39346345.61B/s]  4%|▎         | 16277504/440473133 [00:00<00:10, 41232445.12B/s]  5%|▍         | 20684800/440473133 [00:00<00:09, 42042022.83B/s]  6%|▌         | 24480768/440473133 [00:00<00:10, 40726985.91B/s]  7%|▋         | 28792832/440473133 [00:00<00:09, 41415209.63B/s]  8%|▊         | 34023424/440473133 [00:00<00:09, 44170274.09B/s]  9%|▉         | 39231488/440473133 [00:00<00:08, 46278450.70B/s] 10%|█         | 44498944/440473133 [00:01<00:08, 48024805.52B/s] 11%|█▏        | 49796096/440473133 [00:01<00:07, 49407512.05B/s] 12%|█▏        | 54992896/440473133 [00:01<00:07, 50145597.71B/s] 14%|█▎        | 59993088/440473133 [00:01<00:07, 49938643.92B/s] 15%|█▍        | 64976896/440473133 [00:01<00:07, 49780263.53B/s] 16%|█▌        | 69974016/440473133 [00:01<00:07, 49835698.91B/s] 17%|█▋        | 75259904/440473133 [00:01<00:07, 50702383.34B/s] 18%|█▊        | 80576512/440473133 [00:01<00:06, 51414810.68B/s] 19%|█▉        | 85820416/440473133 [00:01<00:06, 51715788.81B/s] 21%|██        | 90994688/440473133 [00:01<00:06, 51397147.16B/s] 22%|██▏       | 96212992/440473133 [00:02<00:06, 51625975.59B/s] 23%|██▎       | 101533696/440473133 [00:02<00:06, 52089661.54B/s] 24%|██▍       | 106848256/440473133 [00:02<00:06, 52400843.02B/s] 25%|██▌       | 112091136/440473133 [00:02<00:06, 51981127.62B/s] 27%|██▋       | 117368832/440473133 [00:02<00:06, 52215402.14B/s] 28%|██▊       | 122592256/440473133 [00:02<00:06, 52153714.34B/s] 29%|██▉       | 127809536/440473133 [00:02<00:05, 52143457.64B/s] 30%|███       | 133135360/440473133 [00:02<00:05, 52471349.00B/s] 31%|███▏      | 138403840/440473133 [00:02<00:05, 52530796.44B/s] 33%|███▎      | 143727616/440473133 [00:02<00:05, 52739463.65B/s] 34%|███▍      | 149068800/440473133 [00:03<00:05, 52937906.17B/s] 35%|███▌      | 154402816/440473133 [00:03<00:05, 53057732.44B/s] 36%|███▋      | 159709184/440473133 [00:03<00:05, 52284285.32B/s] 37%|███▋      | 164940800/440473133 [00:03<00:05, 51829735.79B/s] 39%|███▊      | 170127360/440473133 [00:03<00:05, 51382761.89B/s] 40%|███▉      | 175268864/440473133 [00:03<00:05, 51155534.46B/s] 41%|████      | 180398080/440473133 [00:03<00:05, 51193097.86B/s] 42%|████▏     | 185533440/440473133 [00:03<00:04, 51240645.71B/s] 43%|████▎     | 190891008/440473133 [00:03<00:04, 51919227.69B/s] 45%|████▍     | 196392960/440473133 [00:03<00:04, 52811794.87B/s] 46%|████▌     | 201680896/440473133 [00:04<00:04, 52788905.36B/s] 47%|████▋     | 207022080/440473133 [00:04<00:04, 52968595.80B/s] 48%|████▊     | 212424704/440473133 [00:04<00:04, 53279763.95B/s] 49%|████▉     | 217755648/440473133 [00:04<00:04, 52277109.03B/s] 51%|█████     | 222990336/440473133 [00:04<00:04, 52004235.96B/s] 52%|█████▏    | 228195328/440473133 [00:04<00:04, 51593716.59B/s] 53%|█████▎    | 233393152/440473133 [00:04<00:04, 51708048.12B/s] 54%|█████▍    | 238741504/440473133 [00:04<00:03, 52225612.51B/s] 55%|█████▌    | 244034560/440473133 [00:04<00:03, 52433125.87B/s] 57%|█████▋    | 249308160/440473133 [00:04<00:03, 52522402.44B/s] 58%|█████▊    | 254562304/440473133 [00:05<00:03, 52333717.36B/s] 59%|█████▉    | 259798016/440473133 [00:05<00:03, 52337120.92B/s] 60%|██████    | 265032704/440473133 [00:05<00:03, 52174429.94B/s] 61%|██████▏   | 270251008/440473133 [00:05<00:03, 51342989.12B/s] 63%|██████▎   | 275389440/440473133 [00:05<00:03, 50945035.61B/s] 64%|██████▎   | 280658944/440473133 [00:05<00:03, 51454453.32B/s] 65%|██████▍   | 285932544/440473133 [00:05<00:02, 51831250.70B/s] 66%|██████▌   | 291239936/440473133 [00:05<00:02, 52195452.81B/s] 67%|██████▋   | 296462336/440473133 [00:05<00:02, 51903474.24B/s] 68%|██████▊   | 301655040/440473133 [00:05<00:02, 50702898.67B/s] 70%|██████▉   | 306823168/440473133 [00:06<00:02, 50992227.78B/s] 71%|███████   | 312105984/440473133 [00:06<00:02, 51527684.35B/s] 72%|███████▏  | 317417472/440473133 [00:06<00:02, 51987437.49B/s] 73%|███████▎  | 322737152/440473133 [00:06<00:02, 52342532.13B/s] 74%|███████▍  | 328137728/440473133 [00:06<00:02, 52829955.85B/s] 76%|███████▌  | 333531136/440473133 [00:06<00:02, 53153509.38B/s] 77%|███████▋  | 338849792/440473133 [00:06<00:01, 52973426.24B/s] 78%|███████▊  | 344155136/440473133 [00:06<00:01, 52996435.65B/s] 79%|███████▉  | 349457408/440473133 [00:06<00:01, 52561058.85B/s] 81%|████████  | 354715648/440473133 [00:06<00:01, 52562701.10B/s] 82%|████████▏ | 360165376/440473133 [00:07<00:01, 53126771.51B/s] 83%|████████▎ | 365481984/440473133 [00:07<00:01, 53137395.55B/s] 84%|████████▍ | 370797568/440473133 [00:07<00:01, 52852045.73B/s] 85%|████████▌ | 376084480/440473133 [00:07<00:01, 52849547.91B/s] 87%|████████▋ | 381442048/440473133 [00:07<00:01, 53063071.88B/s] 88%|████████▊ | 386832384/440473133 [00:07<00:01, 53311466.84B/s] 89%|████████▉ | 392184832/440473133 [00:07<00:00, 53373608.87B/s] 90%|█████████ | 397596672/440473133 [00:07<00:00, 53591407.58B/s] 91%|█████████▏| 402978816/440473133 [00:07<00:00, 53655672.64B/s] 93%|█████████▎| 408345600/440473133 [00:07<00:00, 53499450.22B/s] 94%|█████████▍| 413696000/440473133 [00:08<00:00, 53030333.50B/s] 95%|█████████▌| 419000320/440473133 [00:08<00:00, 51670913.70B/s] 96%|█████████▋| 424176640/440473133 [00:08<00:00, 51015517.60B/s] 97%|█████████▋| 429286400/440473133 [00:08<00:00, 50945675.87B/s] 99%|█████████▊| 434386944/440473133 [00:08<00:00, 50722507.49B/s]100%|█████████▉| 439463936/440473133 [00:08<00:00, 50433275.03B/s]100%|██████████| 440473133/440473133 [00:08<00:00, 51217786.85B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:  13%|█▎        | 268/2118 [00:00<00:00, 2489.42it/s]Processing text_left with encode:  32%|███▏      | 687/2118 [00:00<00:00, 2832.37it/s]Processing text_left with encode:  52%|█████▏    | 1106/2118 [00:00<00:00, 3136.51it/s]Processing text_left with encode:  72%|███████▏  | 1529/2118 [00:00<00:00, 3399.10it/s]Processing text_left with encode:  93%|█████████▎| 1963/2118 [00:00<00:00, 3634.14it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3881.52it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 166/18841 [00:00<00:11, 1653.96it/s]Processing text_right with encode:   2%|▏         | 323/18841 [00:00<00:11, 1623.93it/s]Processing text_right with encode:   3%|▎         | 474/18841 [00:00<00:11, 1586.58it/s]Processing text_right with encode:   3%|▎         | 638/18841 [00:00<00:11, 1601.71it/s]Processing text_right with encode:   4%|▍         | 778/18841 [00:00<00:11, 1534.57it/s]Processing text_right with encode:   5%|▍         | 933/18841 [00:00<00:11, 1538.70it/s]Processing text_right with encode:   6%|▌         | 1080/18841 [00:00<00:11, 1514.25it/s]Processing text_right with encode:   6%|▋         | 1221/18841 [00:00<00:11, 1481.18it/s]Processing text_right with encode:   7%|▋         | 1372/18841 [00:00<00:11, 1488.49it/s]Processing text_right with encode:   8%|▊         | 1530/18841 [00:01<00:11, 1513.69it/s]Processing text_right with encode:   9%|▉         | 1699/18841 [00:01<00:10, 1560.64it/s]Processing text_right with encode:  10%|▉         | 1853/18841 [00:01<00:10, 1550.68it/s]Processing text_right with encode:  11%|█         | 2007/18841 [00:01<00:11, 1515.41it/s]Processing text_right with encode:  11%|█▏        | 2164/18841 [00:01<00:10, 1529.64it/s]Processing text_right with encode:  12%|█▏        | 2317/18841 [00:01<00:10, 1513.77it/s]Processing text_right with encode:  13%|█▎        | 2473/18841 [00:01<00:10, 1525.12it/s]Processing text_right with encode:  14%|█▍        | 2633/18841 [00:01<00:10, 1544.80it/s]Processing text_right with encode:  15%|█▍        | 2805/18841 [00:01<00:10, 1590.38it/s]Processing text_right with encode:  16%|█▌        | 2965/18841 [00:01<00:10, 1552.16it/s]Processing text_right with encode:  17%|█▋        | 3121/18841 [00:02<00:10, 1487.05it/s]Processing text_right with encode:  17%|█▋        | 3279/18841 [00:02<00:10, 1512.32it/s]Processing text_right with encode:  18%|█▊        | 3431/18841 [00:02<00:10, 1484.56it/s]Processing text_right with encode:  19%|█▉        | 3581/18841 [00:02<00:10, 1453.76it/s]Processing text_right with encode:  20%|█▉        | 3728/18841 [00:02<00:10, 1458.36it/s]Processing text_right with encode:  21%|██        | 3889/18841 [00:02<00:09, 1498.79it/s]Processing text_right with encode:  22%|██▏       | 4058/18841 [00:02<00:09, 1549.73it/s]Processing text_right with encode:  22%|██▏       | 4214/18841 [00:02<00:09, 1514.15it/s]Processing text_right with encode:  23%|██▎       | 4374/18841 [00:02<00:09, 1538.91it/s]Processing text_right with encode:  24%|██▍       | 4529/18841 [00:02<00:09, 1540.53it/s]Processing text_right with encode:  25%|██▍       | 4692/18841 [00:03<00:09, 1564.49it/s]Processing text_right with encode:  26%|██▌       | 4859/18841 [00:03<00:08, 1594.42it/s]Processing text_right with encode:  27%|██▋       | 5019/18841 [00:03<00:08, 1595.42it/s]Processing text_right with encode:  27%|██▋       | 5179/18841 [00:03<00:08, 1591.36it/s]Processing text_right with encode:  28%|██▊       | 5339/18841 [00:03<00:08, 1588.98it/s]Processing text_right with encode:  29%|██▉       | 5499/18841 [00:03<00:08, 1576.24it/s]Processing text_right with encode:  30%|███       | 5657/18841 [00:03<00:08, 1564.95it/s]Processing text_right with encode:  31%|███       | 5814/18841 [00:03<00:08, 1552.41it/s]Processing text_right with encode:  32%|███▏      | 5972/18841 [00:03<00:08, 1558.46it/s]Processing text_right with encode:  33%|███▎      | 6128/18841 [00:03<00:08, 1539.59it/s]Processing text_right with encode:  33%|███▎      | 6283/18841 [00:04<00:08, 1533.99it/s]Processing text_right with encode:  34%|███▍      | 6449/18841 [00:04<00:07, 1569.29it/s]Processing text_right with encode:  35%|███▌      | 6617/18841 [00:04<00:07, 1598.85it/s]Processing text_right with encode:  36%|███▌      | 6778/18841 [00:04<00:07, 1518.20it/s]Processing text_right with encode:  37%|███▋      | 6931/18841 [00:04<00:07, 1518.60it/s]Processing text_right with encode:  38%|███▊      | 7090/18841 [00:04<00:07, 1533.38it/s]Processing text_right with encode:  39%|███▊      | 7261/18841 [00:04<00:07, 1579.89it/s]Processing text_right with encode:  39%|███▉      | 7429/18841 [00:04<00:07, 1605.93it/s]Processing text_right with encode:  40%|████      | 7591/18841 [00:04<00:07, 1585.95it/s]Processing text_right with encode:  41%|████      | 7751/18841 [00:05<00:06, 1590.10it/s]Processing text_right with encode:  42%|████▏     | 7911/18841 [00:05<00:06, 1587.00it/s]Processing text_right with encode:  43%|████▎     | 8070/18841 [00:05<00:06, 1556.47it/s]Processing text_right with encode:  44%|████▎     | 8233/18841 [00:05<00:06, 1575.71it/s]Processing text_right with encode:  45%|████▍     | 8398/18841 [00:05<00:06, 1595.97it/s]Processing text_right with encode:  45%|████▌     | 8558/18841 [00:05<00:06, 1555.12it/s]Processing text_right with encode:  46%|████▋     | 8718/18841 [00:05<00:06, 1568.02it/s]Processing text_right with encode:  47%|████▋     | 8883/18841 [00:05<00:06, 1590.47it/s]Processing text_right with encode:  48%|████▊     | 9043/18841 [00:05<00:06, 1541.57it/s]Processing text_right with encode:  49%|████▉     | 9204/18841 [00:05<00:06, 1561.06it/s]Processing text_right with encode:  50%|████▉     | 9361/18841 [00:06<00:06, 1547.25it/s]Processing text_right with encode:  51%|█████     | 9520/18841 [00:06<00:05, 1559.27it/s]Processing text_right with encode:  51%|█████▏    | 9677/18841 [00:06<00:05, 1555.48it/s]Processing text_right with encode:  52%|█████▏    | 9833/18841 [00:06<00:05, 1528.09it/s]Processing text_right with encode:  53%|█████▎    | 10002/18841 [00:06<00:05, 1572.80it/s]Processing text_right with encode:  54%|█████▍    | 10160/18841 [00:06<00:05, 1566.11it/s]Processing text_right with encode:  55%|█████▍    | 10317/18841 [00:06<00:05, 1538.22it/s]Processing text_right with encode:  56%|█████▌    | 10504/18841 [00:06<00:05, 1623.55it/s]Processing text_right with encode:  57%|█████▋    | 10668/18841 [00:06<00:05, 1585.43it/s]Processing text_right with encode:  57%|█████▋    | 10828/18841 [00:06<00:05, 1564.81it/s]Processing text_right with encode:  58%|█████▊    | 10986/18841 [00:07<00:05, 1537.43it/s]Processing text_right with encode:  59%|█████▉    | 11141/18841 [00:07<00:05, 1530.12it/s]Processing text_right with encode:  60%|█████▉    | 11295/18841 [00:07<00:05, 1496.46it/s]Processing text_right with encode:  61%|██████    | 11449/18841 [00:07<00:04, 1507.75it/s]Processing text_right with encode:  62%|██████▏   | 11601/18841 [00:07<00:04, 1490.19it/s]Processing text_right with encode:  62%|██████▏   | 11758/18841 [00:07<00:04, 1512.67it/s]Processing text_right with encode:  63%|██████▎   | 11910/18841 [00:07<00:04, 1511.28it/s]Processing text_right with encode:  64%|██████▍   | 12067/18841 [00:07<00:04, 1528.41it/s]Processing text_right with encode:  65%|██████▍   | 12221/18841 [00:07<00:04, 1508.86it/s]Processing text_right with encode:  66%|██████▌   | 12374/18841 [00:08<00:04, 1513.98it/s]Processing text_right with encode:  67%|██████▋   | 12531/18841 [00:08<00:04, 1530.23it/s]Processing text_right with encode:  67%|██████▋   | 12690/18841 [00:08<00:03, 1544.44it/s]Processing text_right with encode:  68%|██████▊   | 12845/18841 [00:08<00:03, 1545.10it/s]Processing text_right with encode:  69%|██████▉   | 13000/18841 [00:08<00:03, 1531.54it/s]Processing text_right with encode:  70%|██████▉   | 13154/18841 [00:08<00:03, 1529.16it/s]Processing text_right with encode:  71%|███████   | 13308/18841 [00:08<00:03, 1529.08it/s]Processing text_right with encode:  71%|███████▏  | 13461/18841 [00:08<00:03, 1519.76it/s]Processing text_right with encode:  72%|███████▏  | 13626/18841 [00:08<00:03, 1553.28it/s]Processing text_right with encode:  73%|███████▎  | 13789/18841 [00:08<00:03, 1573.19it/s]Processing text_right with encode:  74%|███████▍  | 13947/18841 [00:09<00:03, 1554.95it/s]Processing text_right with encode:  75%|███████▍  | 14103/18841 [00:09<00:03, 1540.82it/s]Processing text_right with encode:  76%|███████▌  | 14258/18841 [00:09<00:02, 1536.55it/s]Processing text_right with encode:  77%|███████▋  | 14416/18841 [00:09<00:02, 1547.03it/s]Processing text_right with encode:  77%|███████▋  | 14571/18841 [00:09<00:02, 1522.67it/s]Processing text_right with encode:  78%|███████▊  | 14740/18841 [00:09<00:02, 1568.61it/s]Processing text_right with encode:  79%|███████▉  | 14899/18841 [00:09<00:02, 1573.26it/s]Processing text_right with encode:  80%|███████▉  | 15057/18841 [00:09<00:02, 1569.95it/s]Processing text_right with encode:  81%|████████  | 15219/18841 [00:09<00:02, 1583.05it/s]Processing text_right with encode:  82%|████████▏ | 15378/18841 [00:09<00:02, 1562.33it/s]Processing text_right with encode:  82%|████████▏ | 15535/18841 [00:10<00:02, 1563.41it/s]Processing text_right with encode:  83%|████████▎ | 15692/18841 [00:10<00:02, 1549.54it/s]Processing text_right with encode:  84%|████████▍ | 15851/18841 [00:10<00:01, 1561.35it/s]Processing text_right with encode:  85%|████████▍ | 16008/18841 [00:10<00:01, 1533.02it/s]Processing text_right with encode:  86%|████████▌ | 16162/18841 [00:10<00:01, 1500.76it/s]Processing text_right with encode:  87%|████████▋ | 16313/18841 [00:10<00:01, 1470.69it/s]Processing text_right with encode:  87%|████████▋ | 16473/18841 [00:10<00:01, 1501.98it/s]Processing text_right with encode:  88%|████████▊ | 16624/18841 [00:10<00:01, 1488.65it/s]Processing text_right with encode:  89%|████████▉ | 16774/18841 [00:10<00:01, 1490.74it/s]Processing text_right with encode:  90%|████████▉ | 16924/18841 [00:10<00:01, 1491.50it/s]Processing text_right with encode:  91%|█████████ | 17076/18841 [00:11<00:01, 1497.50it/s]Processing text_right with encode:  91%|█████████▏| 17226/18841 [00:11<00:01, 1492.69it/s]Processing text_right with encode:  92%|█████████▏| 17385/18841 [00:11<00:00, 1517.05it/s]Processing text_right with encode:  93%|█████████▎| 17537/18841 [00:11<00:00, 1507.24it/s]Processing text_right with encode:  94%|█████████▍| 17697/18841 [00:11<00:00, 1532.01it/s]Processing text_right with encode:  95%|█████████▍| 17851/18841 [00:11<00:00, 1478.61it/s]Processing text_right with encode:  96%|█████████▌| 18013/18841 [00:11<00:00, 1516.76it/s]Processing text_right with encode:  96%|█████████▋| 18177/18841 [00:11<00:00, 1550.59it/s]Processing text_right with encode:  97%|█████████▋| 18333/18841 [00:11<00:00, 1501.26it/s]Processing text_right with encode:  98%|█████████▊| 18500/18841 [00:11<00:00, 1547.84it/s]Processing text_right with encode:  99%|█████████▉| 18656/18841 [00:12<00:00, 1538.50it/s]Processing text_right with encode: 100%|█████████▉| 18811/18841 [00:12<00:00, 1528.24it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:12<00:00, 1540.96it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 574428.44it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 679135.46it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  65%|██████▍   | 410/633 [00:00<00:00, 4088.43it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4032.07it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 150/5961 [00:00<00:03, 1499.06it/s]Processing text_right with encode:   5%|▌         | 304/5961 [00:00<00:03, 1510.07it/s]Processing text_right with encode:   7%|▋         | 442/5961 [00:00<00:03, 1467.62it/s]Processing text_right with encode:  10%|█         | 599/5961 [00:00<00:03, 1493.19it/s]Processing text_right with encode:  13%|█▎        | 757/5961 [00:00<00:03, 1516.04it/s]Processing text_right with encode:  15%|█▌        | 906/5961 [00:00<00:03, 1507.67it/s]Processing text_right with encode:  18%|█▊        | 1058/5961 [00:00<00:03, 1508.50it/s]Processing text_right with encode:  20%|██        | 1215/5961 [00:00<00:03, 1522.56it/s]Processing text_right with encode:  23%|██▎       | 1375/5961 [00:00<00:02, 1543.73it/s]Processing text_right with encode:  26%|██▌       | 1530/5961 [00:01<00:02, 1543.93it/s]Processing text_right with encode:  28%|██▊       | 1682/5961 [00:01<00:02, 1535.06it/s]Processing text_right with encode:  31%|███       | 1833/5961 [00:01<00:02, 1515.26it/s]Processing text_right with encode:  33%|███▎      | 1983/5961 [00:01<00:02, 1492.64it/s]Processing text_right with encode:  36%|███▌      | 2141/5961 [00:01<00:02, 1515.54it/s]Processing text_right with encode:  39%|███▊      | 2295/5961 [00:01<00:02, 1521.25it/s]Processing text_right with encode:  41%|████      | 2447/5961 [00:01<00:02, 1445.95it/s]Processing text_right with encode:  44%|████▍     | 2611/5961 [00:01<00:02, 1498.06it/s]Processing text_right with encode:  47%|████▋     | 2778/5961 [00:01<00:02, 1545.06it/s]Processing text_right with encode:  49%|████▉     | 2934/5961 [00:01<00:01, 1539.76it/s]Processing text_right with encode:  52%|█████▏    | 3101/5961 [00:02<00:01, 1576.47it/s]Processing text_right with encode:  55%|█████▍    | 3260/5961 [00:02<00:01, 1550.75it/s]Processing text_right with encode:  57%|█████▋    | 3416/5961 [00:02<00:01, 1471.95it/s]Processing text_right with encode:  60%|█████▉    | 3565/5961 [00:02<00:01, 1473.25it/s]Processing text_right with encode:  62%|██████▏   | 3714/5961 [00:02<00:01, 1475.02it/s]Processing text_right with encode:  65%|██████▍   | 3873/5961 [00:02<00:01, 1506.52it/s]Processing text_right with encode:  68%|██████▊   | 4035/5961 [00:02<00:01, 1538.78it/s]Processing text_right with encode:  70%|███████   | 4190/5961 [00:02<00:01, 1520.10it/s]Processing text_right with encode:  73%|███████▎  | 4343/5961 [00:02<00:01, 1488.73it/s]Processing text_right with encode:  75%|███████▌  | 4493/5961 [00:02<00:00, 1491.76it/s]Processing text_right with encode:  78%|███████▊  | 4643/5961 [00:03<00:00, 1475.94it/s]Processing text_right with encode:  80%|████████  | 4791/5961 [00:03<00:00, 1469.96it/s]Processing text_right with encode:  83%|████████▎ | 4939/5961 [00:03<00:00, 1451.60it/s]Processing text_right with encode:  85%|████████▌ | 5085/5961 [00:03<00:00, 1448.47it/s]Processing text_right with encode:  88%|████████▊ | 5245/5961 [00:03<00:00, 1488.88it/s]Processing text_right with encode:  91%|█████████ | 5395/5961 [00:03<00:00, 1453.47it/s]Processing text_right with encode:  93%|█████████▎| 5541/5961 [00:03<00:00, 1452.81it/s]Processing text_right with encode:  95%|█████████▌| 5687/5961 [00:03<00:00, 1428.52it/s]Processing text_right with encode:  98%|█████████▊| 5831/5961 [00:03<00:00, 1422.35it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1500.28it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 421026.71it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 725438.74it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [01:36<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [01:36<?, ?it/s, loss=0.920]Epoch 1/1:   1%|          | 1/102 [01:36<2:41:38, 96.03s/it, loss=0.920]Killed

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
   3b7f4e4..5b48ad6  master     -> origin/master
Updating 3b7f4e4..5b48ad6
Fast-forward
 .../20200522/list_log_pullrequest_20200522.md      |   2 +-
 error_list/20200522/list_log_testall_20200522.md   | 787 +--------------------
 2 files changed, 3 insertions(+), 786 deletions(-)
[master 24432c3] ml_store
 1 file changed, 65 insertions(+)
To github.com:arita37/mlmodels_store.git
   5b48ad6..24432c3  master -> master





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
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=46e0771d503cb6ac93ad8ec495898c5fb290f860152c99a77b7b8ebae301debe
  Stored in directory: /tmp/pip-ephem-wheel-cache-6utipkbm/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   24432c3..95e09d3  master     -> origin/master
Updating 24432c3..95e09d3
Fast-forward
 error_list/20200522/list_log_pullrequest_20200522.md | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
[master 91c8d5e] ml_store
 2 files changed, 114 insertions(+)
To github.com:arita37/mlmodels_store.git
   95e09d3..91c8d5e  master -> master





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
[master d01c634] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   91c8d5e..d01c634  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f58ffcb9e18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f58ffcb9e18>

  function with postional parmater data_info <function get_dataset_torch at 0x7f58ffcb9e18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:13, 134388.34it/s] 36%|███▌      | 3522560/9912422 [00:00<00:33, 191664.37it/s] 82%|████████▏ | 8134656/9912422 [00:00<00:06, 273303.20it/s]9920512it [00:00, 27384331.74it/s]                           
0it [00:00, ?it/s]32768it [00:00, 592309.71it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 142955.02it/s]1654784it [00:00, 10752255.82it/s]                         
0it [00:00, ?it/s]8192it [00:00, 218569.23it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f58ffa70b70>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f58ffa70b70>

  function with postional parmater data_info <function get_dataset_torch at 0x7f58ffa70b70> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0018734642465909322 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.011051096439361573 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.001193624198436737 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.009545514225959778 	 Accuracy: 1
model saves at 1 accuracy

  #### Predict   ##################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f58ffa70950>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f58ffa70950>

  function with postional parmater data_info <function get_dataset_torch at 0x7f58ffa70950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  #### metrics   ##################################################### 
None

  #### Plot   ######################################################## 

  #### Save  ######################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 

  #### Load   ######################################################## 
<__main__.Model object at 0x7f58ff9ada58>

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
<__main__.Model object at 0x7f58fc09d048>

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
[master 24f5440] ml_store
 1 file changed, 148 insertions(+)
To github.com:arita37/mlmodels_store.git
   d01c634..24f5440  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140136079781840
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140136079781616
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140136079780384
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140136079779936
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140136079779432
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140136079779096

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
grad_step = 000000, loss = 0.812088
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.659727
grad_step = 000002, loss = 0.499352
grad_step = 000003, loss = 0.324041
grad_step = 000004, loss = 0.150050
grad_step = 000005, loss = 0.059465
grad_step = 000006, loss = 0.149415
grad_step = 000007, loss = 0.137499
grad_step = 000008, loss = 0.070375
grad_step = 000009, loss = 0.021155
grad_step = 000010, loss = 0.010810
grad_step = 000011, loss = 0.022610
grad_step = 000012, loss = 0.036997
grad_step = 000013, loss = 0.043694
grad_step = 000014, loss = 0.040998
grad_step = 000015, loss = 0.032881
grad_step = 000016, loss = 0.024987
grad_step = 000017, loss = 0.021339
grad_step = 000018, loss = 0.021778
grad_step = 000019, loss = 0.023290
grad_step = 000020, loss = 0.022113
grad_step = 000021, loss = 0.016565
grad_step = 000022, loss = 0.009894
grad_step = 000023, loss = 0.006127
grad_step = 000024, loss = 0.006061
grad_step = 000025, loss = 0.008015
grad_step = 000026, loss = 0.010073
grad_step = 000027, loss = 0.011161
grad_step = 000028, loss = 0.011089
grad_step = 000029, loss = 0.010265
grad_step = 000030, loss = 0.009388
grad_step = 000031, loss = 0.009011
grad_step = 000032, loss = 0.009115
grad_step = 000033, loss = 0.009124
grad_step = 000034, loss = 0.008504
grad_step = 000035, loss = 0.007341
grad_step = 000036, loss = 0.006198
grad_step = 000037, loss = 0.005536
grad_step = 000038, loss = 0.005417
grad_step = 000039, loss = 0.005612
grad_step = 000040, loss = 0.005827
grad_step = 000041, loss = 0.005876
grad_step = 000042, loss = 0.005783
grad_step = 000043, loss = 0.005714
grad_step = 000044, loss = 0.005776
grad_step = 000045, loss = 0.005877
grad_step = 000046, loss = 0.005852
grad_step = 000047, loss = 0.005655
grad_step = 000048, loss = 0.005379
grad_step = 000049, loss = 0.005142
grad_step = 000050, loss = 0.005016
grad_step = 000051, loss = 0.004981
grad_step = 000052, loss = 0.004953
grad_step = 000053, loss = 0.004873
grad_step = 000054, loss = 0.004775
grad_step = 000055, loss = 0.004725
grad_step = 000056, loss = 0.004747
grad_step = 000057, loss = 0.004796
grad_step = 000058, loss = 0.004814
grad_step = 000059, loss = 0.004775
grad_step = 000060, loss = 0.004696
grad_step = 000061, loss = 0.004624
grad_step = 000062, loss = 0.004587
grad_step = 000063, loss = 0.004565
grad_step = 000064, loss = 0.004522
grad_step = 000065, loss = 0.004454
grad_step = 000066, loss = 0.004390
grad_step = 000067, loss = 0.004355
grad_step = 000068, loss = 0.004344
grad_step = 000069, loss = 0.004332
grad_step = 000070, loss = 0.004304
grad_step = 000071, loss = 0.004273
grad_step = 000072, loss = 0.004251
grad_step = 000073, loss = 0.004232
grad_step = 000074, loss = 0.004203
grad_step = 000075, loss = 0.004161
grad_step = 000076, loss = 0.004119
grad_step = 000077, loss = 0.004086
grad_step = 000078, loss = 0.004056
grad_step = 000079, loss = 0.004019
grad_step = 000080, loss = 0.003979
grad_step = 000081, loss = 0.003946
grad_step = 000082, loss = 0.003922
grad_step = 000083, loss = 0.003896
grad_step = 000084, loss = 0.003863
grad_step = 000085, loss = 0.003827
grad_step = 000086, loss = 0.003794
grad_step = 000087, loss = 0.003759
grad_step = 000088, loss = 0.003720
grad_step = 000089, loss = 0.003679
grad_step = 000090, loss = 0.003643
grad_step = 000091, loss = 0.003606
grad_step = 000092, loss = 0.003567
grad_step = 000093, loss = 0.003529
grad_step = 000094, loss = 0.003494
grad_step = 000095, loss = 0.003457
grad_step = 000096, loss = 0.003418
grad_step = 000097, loss = 0.003381
grad_step = 000098, loss = 0.003343
grad_step = 000099, loss = 0.003302
grad_step = 000100, loss = 0.003264
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003226
grad_step = 000102, loss = 0.003188
grad_step = 000103, loss = 0.003151
grad_step = 000104, loss = 0.003115
grad_step = 000105, loss = 0.003081
grad_step = 000106, loss = 0.003048
grad_step = 000107, loss = 0.003014
grad_step = 000108, loss = 0.002985
grad_step = 000109, loss = 0.002956
grad_step = 000110, loss = 0.002930
grad_step = 000111, loss = 0.002906
grad_step = 000112, loss = 0.002886
grad_step = 000113, loss = 0.002874
grad_step = 000114, loss = 0.002862
grad_step = 000115, loss = 0.002832
grad_step = 000116, loss = 0.002809
grad_step = 000117, loss = 0.002802
grad_step = 000118, loss = 0.002784
grad_step = 000119, loss = 0.002756
grad_step = 000120, loss = 0.002741
grad_step = 000121, loss = 0.002729
grad_step = 000122, loss = 0.002704
grad_step = 000123, loss = 0.002683
grad_step = 000124, loss = 0.002671
grad_step = 000125, loss = 0.002652
grad_step = 000126, loss = 0.002629
grad_step = 000127, loss = 0.002612
grad_step = 000128, loss = 0.002597
grad_step = 000129, loss = 0.002576
grad_step = 000130, loss = 0.002553
grad_step = 000131, loss = 0.002535
grad_step = 000132, loss = 0.002518
grad_step = 000133, loss = 0.002497
grad_step = 000134, loss = 0.002475
grad_step = 000135, loss = 0.002453
grad_step = 000136, loss = 0.002430
grad_step = 000137, loss = 0.002408
grad_step = 000138, loss = 0.002387
grad_step = 000139, loss = 0.002367
grad_step = 000140, loss = 0.002353
grad_step = 000141, loss = 0.002343
grad_step = 000142, loss = 0.002334
grad_step = 000143, loss = 0.002305
grad_step = 000144, loss = 0.002258
grad_step = 000145, loss = 0.002221
grad_step = 000146, loss = 0.002207
grad_step = 000147, loss = 0.002203
grad_step = 000148, loss = 0.002190
grad_step = 000149, loss = 0.002156
grad_step = 000150, loss = 0.002113
grad_step = 000151, loss = 0.002081
grad_step = 000152, loss = 0.002066
grad_step = 000153, loss = 0.002057
grad_step = 000154, loss = 0.002044
grad_step = 000155, loss = 0.002020
grad_step = 000156, loss = 0.001987
grad_step = 000157, loss = 0.001949
grad_step = 000158, loss = 0.001913
grad_step = 000159, loss = 0.001882
grad_step = 000160, loss = 0.001857
grad_step = 000161, loss = 0.001837
grad_step = 000162, loss = 0.001823
grad_step = 000163, loss = 0.001829
grad_step = 000164, loss = 0.001883
grad_step = 000165, loss = 0.002010
grad_step = 000166, loss = 0.002071
grad_step = 000167, loss = 0.001891
grad_step = 000168, loss = 0.001672
grad_step = 000169, loss = 0.001748
grad_step = 000170, loss = 0.001856
grad_step = 000171, loss = 0.001710
grad_step = 000172, loss = 0.001586
grad_step = 000173, loss = 0.001666
grad_step = 000174, loss = 0.001683
grad_step = 000175, loss = 0.001555
grad_step = 000176, loss = 0.001527
grad_step = 000177, loss = 0.001585
grad_step = 000178, loss = 0.001532
grad_step = 000179, loss = 0.001445
grad_step = 000180, loss = 0.001465
grad_step = 000181, loss = 0.001493
grad_step = 000182, loss = 0.001426
grad_step = 000183, loss = 0.001361
grad_step = 000184, loss = 0.001362
grad_step = 000185, loss = 0.001371
grad_step = 000186, loss = 0.001337
grad_step = 000187, loss = 0.001286
grad_step = 000188, loss = 0.001260
grad_step = 000189, loss = 0.001261
grad_step = 000190, loss = 0.001261
grad_step = 000191, loss = 0.001242
grad_step = 000192, loss = 0.001206
grad_step = 000193, loss = 0.001170
grad_step = 000194, loss = 0.001146
grad_step = 000195, loss = 0.001136
grad_step = 000196, loss = 0.001136
grad_step = 000197, loss = 0.001142
grad_step = 000198, loss = 0.001158
grad_step = 000199, loss = 0.001181
grad_step = 000200, loss = 0.001211
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001207
grad_step = 000202, loss = 0.001159
grad_step = 000203, loss = 0.001081
grad_step = 000204, loss = 0.001015
grad_step = 000205, loss = 0.000986
grad_step = 000206, loss = 0.000993
grad_step = 000207, loss = 0.001019
grad_step = 000208, loss = 0.001054
grad_step = 000209, loss = 0.001092
grad_step = 000210, loss = 0.001110
grad_step = 000211, loss = 0.001085
grad_step = 000212, loss = 0.001007
grad_step = 000213, loss = 0.000929
grad_step = 000214, loss = 0.000898
grad_step = 000215, loss = 0.000920
grad_step = 000216, loss = 0.000952
grad_step = 000217, loss = 0.000964
grad_step = 000218, loss = 0.000939
grad_step = 000219, loss = 0.000892
grad_step = 000220, loss = 0.000854
grad_step = 000221, loss = 0.000842
grad_step = 000222, loss = 0.000853
grad_step = 000223, loss = 0.000868
grad_step = 000224, loss = 0.000870
grad_step = 000225, loss = 0.000855
grad_step = 000226, loss = 0.000830
grad_step = 000227, loss = 0.000804
grad_step = 000228, loss = 0.000788
grad_step = 000229, loss = 0.000784
grad_step = 000230, loss = 0.000787
grad_step = 000231, loss = 0.000790
grad_step = 000232, loss = 0.000790
grad_step = 000233, loss = 0.000787
grad_step = 000234, loss = 0.000780
grad_step = 000235, loss = 0.000768
grad_step = 000236, loss = 0.000755
grad_step = 000237, loss = 0.000741
grad_step = 000238, loss = 0.000729
grad_step = 000239, loss = 0.000719
grad_step = 000240, loss = 0.000710
grad_step = 000241, loss = 0.000704
grad_step = 000242, loss = 0.000698
grad_step = 000243, loss = 0.000693
grad_step = 000244, loss = 0.000689
grad_step = 000245, loss = 0.000686
grad_step = 000246, loss = 0.000685
grad_step = 000247, loss = 0.000686
grad_step = 000248, loss = 0.000694
grad_step = 000249, loss = 0.000714
grad_step = 000250, loss = 0.000753
grad_step = 000251, loss = 0.000816
grad_step = 000252, loss = 0.000913
grad_step = 000253, loss = 0.000993
grad_step = 000254, loss = 0.001025
grad_step = 000255, loss = 0.000893
grad_step = 000256, loss = 0.000710
grad_step = 000257, loss = 0.000627
grad_step = 000258, loss = 0.000694
grad_step = 000259, loss = 0.000789
grad_step = 000260, loss = 0.000762
grad_step = 000261, loss = 0.000660
grad_step = 000262, loss = 0.000609
grad_step = 000263, loss = 0.000660
grad_step = 000264, loss = 0.000710
grad_step = 000265, loss = 0.000667
grad_step = 000266, loss = 0.000603
grad_step = 000267, loss = 0.000602
grad_step = 000268, loss = 0.000643
grad_step = 000269, loss = 0.000651
grad_step = 000270, loss = 0.000608
grad_step = 000271, loss = 0.000578
grad_step = 000272, loss = 0.000591
grad_step = 000273, loss = 0.000612
grad_step = 000274, loss = 0.000604
grad_step = 000275, loss = 0.000575
grad_step = 000276, loss = 0.000563
grad_step = 000277, loss = 0.000574
grad_step = 000278, loss = 0.000584
grad_step = 000279, loss = 0.000576
grad_step = 000280, loss = 0.000557
grad_step = 000281, loss = 0.000548
grad_step = 000282, loss = 0.000553
grad_step = 000283, loss = 0.000558
grad_step = 000284, loss = 0.000554
grad_step = 000285, loss = 0.000542
grad_step = 000286, loss = 0.000534
grad_step = 000287, loss = 0.000531
grad_step = 000288, loss = 0.000534
grad_step = 000289, loss = 0.000536
grad_step = 000290, loss = 0.000532
grad_step = 000291, loss = 0.000524
grad_step = 000292, loss = 0.000515
grad_step = 000293, loss = 0.000509
grad_step = 000294, loss = 0.000508
grad_step = 000295, loss = 0.000509
grad_step = 000296, loss = 0.000509
grad_step = 000297, loss = 0.000506
grad_step = 000298, loss = 0.000500
grad_step = 000299, loss = 0.000493
grad_step = 000300, loss = 0.000488
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000485
grad_step = 000302, loss = 0.000483
grad_step = 000303, loss = 0.000482
grad_step = 000304, loss = 0.000480
grad_step = 000305, loss = 0.000477
grad_step = 000306, loss = 0.000474
grad_step = 000307, loss = 0.000470
grad_step = 000308, loss = 0.000467
grad_step = 000309, loss = 0.000465
grad_step = 000310, loss = 0.000463
grad_step = 000311, loss = 0.000462
grad_step = 000312, loss = 0.000460
grad_step = 000313, loss = 0.000457
grad_step = 000314, loss = 0.000453
grad_step = 000315, loss = 0.000450
grad_step = 000316, loss = 0.000448
grad_step = 000317, loss = 0.000446
grad_step = 000318, loss = 0.000444
grad_step = 000319, loss = 0.000441
grad_step = 000320, loss = 0.000438
grad_step = 000321, loss = 0.000435
grad_step = 000322, loss = 0.000432
grad_step = 000323, loss = 0.000429
grad_step = 000324, loss = 0.000427
grad_step = 000325, loss = 0.000426
grad_step = 000326, loss = 0.000425
grad_step = 000327, loss = 0.000427
grad_step = 000328, loss = 0.000427
grad_step = 000329, loss = 0.000429
grad_step = 000330, loss = 0.000430
grad_step = 000331, loss = 0.000431
grad_step = 000332, loss = 0.000427
grad_step = 000333, loss = 0.000424
grad_step = 000334, loss = 0.000422
grad_step = 000335, loss = 0.000423
grad_step = 000336, loss = 0.000428
grad_step = 000337, loss = 0.000432
grad_step = 000338, loss = 0.000432
grad_step = 000339, loss = 0.000427
grad_step = 000340, loss = 0.000418
grad_step = 000341, loss = 0.000411
grad_step = 000342, loss = 0.000408
grad_step = 000343, loss = 0.000408
grad_step = 000344, loss = 0.000408
grad_step = 000345, loss = 0.000408
grad_step = 000346, loss = 0.000402
grad_step = 000347, loss = 0.000394
grad_step = 000348, loss = 0.000387
grad_step = 000349, loss = 0.000383
grad_step = 000350, loss = 0.000383
grad_step = 000351, loss = 0.000386
grad_step = 000352, loss = 0.000388
grad_step = 000353, loss = 0.000389
grad_step = 000354, loss = 0.000385
grad_step = 000355, loss = 0.000381
grad_step = 000356, loss = 0.000376
grad_step = 000357, loss = 0.000372
grad_step = 000358, loss = 0.000370
grad_step = 000359, loss = 0.000370
grad_step = 000360, loss = 0.000371
grad_step = 000361, loss = 0.000372
grad_step = 000362, loss = 0.000373
grad_step = 000363, loss = 0.000375
grad_step = 000364, loss = 0.000375
grad_step = 000365, loss = 0.000375
grad_step = 000366, loss = 0.000371
grad_step = 000367, loss = 0.000370
grad_step = 000368, loss = 0.000371
grad_step = 000369, loss = 0.000379
grad_step = 000370, loss = 0.000400
grad_step = 000371, loss = 0.000442
grad_step = 000372, loss = 0.000517
grad_step = 000373, loss = 0.000626
grad_step = 000374, loss = 0.000783
grad_step = 000375, loss = 0.000887
grad_step = 000376, loss = 0.000882
grad_step = 000377, loss = 0.000631
grad_step = 000378, loss = 0.000402
grad_step = 000379, loss = 0.000378
grad_step = 000380, loss = 0.000522
grad_step = 000381, loss = 0.000583
grad_step = 000382, loss = 0.000449
grad_step = 000383, loss = 0.000350
grad_step = 000384, loss = 0.000413
grad_step = 000385, loss = 0.000472
grad_step = 000386, loss = 0.000407
grad_step = 000387, loss = 0.000344
grad_step = 000388, loss = 0.000388
grad_step = 000389, loss = 0.000424
grad_step = 000390, loss = 0.000368
grad_step = 000391, loss = 0.000337
grad_step = 000392, loss = 0.000378
grad_step = 000393, loss = 0.000384
grad_step = 000394, loss = 0.000343
grad_step = 000395, loss = 0.000337
grad_step = 000396, loss = 0.000365
grad_step = 000397, loss = 0.000356
grad_step = 000398, loss = 0.000329
grad_step = 000399, loss = 0.000337
grad_step = 000400, loss = 0.000352
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000336
grad_step = 000402, loss = 0.000322
grad_step = 000403, loss = 0.000333
grad_step = 000404, loss = 0.000337
grad_step = 000405, loss = 0.000324
grad_step = 000406, loss = 0.000319
grad_step = 000407, loss = 0.000327
grad_step = 000408, loss = 0.000327
grad_step = 000409, loss = 0.000317
grad_step = 000410, loss = 0.000316
grad_step = 000411, loss = 0.000322
grad_step = 000412, loss = 0.000319
grad_step = 000413, loss = 0.000313
grad_step = 000414, loss = 0.000313
grad_step = 000415, loss = 0.000317
grad_step = 000416, loss = 0.000318
grad_step = 000417, loss = 0.000316
grad_step = 000418, loss = 0.000319
grad_step = 000419, loss = 0.000324
grad_step = 000420, loss = 0.000322
grad_step = 000421, loss = 0.000315
grad_step = 000422, loss = 0.000310
grad_step = 000423, loss = 0.000308
grad_step = 000424, loss = 0.000306
grad_step = 000425, loss = 0.000302
grad_step = 000426, loss = 0.000301
grad_step = 000427, loss = 0.000305
grad_step = 000428, loss = 0.000307
grad_step = 000429, loss = 0.000309
grad_step = 000430, loss = 0.000309
grad_step = 000431, loss = 0.000309
grad_step = 000432, loss = 0.000305
grad_step = 000433, loss = 0.000300
grad_step = 000434, loss = 0.000295
grad_step = 000435, loss = 0.000294
grad_step = 000436, loss = 0.000294
grad_step = 000437, loss = 0.000296
grad_step = 000438, loss = 0.000300
grad_step = 000439, loss = 0.000303
grad_step = 000440, loss = 0.000306
grad_step = 000441, loss = 0.000304
grad_step = 000442, loss = 0.000301
grad_step = 000443, loss = 0.000294
grad_step = 000444, loss = 0.000289
grad_step = 000445, loss = 0.000287
grad_step = 000446, loss = 0.000286
grad_step = 000447, loss = 0.000287
grad_step = 000448, loss = 0.000289
grad_step = 000449, loss = 0.000292
grad_step = 000450, loss = 0.000298
grad_step = 000451, loss = 0.000305
grad_step = 000452, loss = 0.000306
grad_step = 000453, loss = 0.000305
grad_step = 000454, loss = 0.000296
grad_step = 000455, loss = 0.000288
grad_step = 000456, loss = 0.000282
grad_step = 000457, loss = 0.000279
grad_step = 000458, loss = 0.000279
grad_step = 000459, loss = 0.000282
grad_step = 000460, loss = 0.000287
grad_step = 000461, loss = 0.000291
grad_step = 000462, loss = 0.000296
grad_step = 000463, loss = 0.000294
grad_step = 000464, loss = 0.000290
grad_step = 000465, loss = 0.000282
grad_step = 000466, loss = 0.000275
grad_step = 000467, loss = 0.000271
grad_step = 000468, loss = 0.000271
grad_step = 000469, loss = 0.000273
grad_step = 000470, loss = 0.000277
grad_step = 000471, loss = 0.000284
grad_step = 000472, loss = 0.000293
grad_step = 000473, loss = 0.000310
grad_step = 000474, loss = 0.000328
grad_step = 000475, loss = 0.000336
grad_step = 000476, loss = 0.000330
grad_step = 000477, loss = 0.000310
grad_step = 000478, loss = 0.000286
grad_step = 000479, loss = 0.000269
grad_step = 000480, loss = 0.000267
grad_step = 000481, loss = 0.000276
grad_step = 000482, loss = 0.000291
grad_step = 000483, loss = 0.000302
grad_step = 000484, loss = 0.000300
grad_step = 000485, loss = 0.000286
grad_step = 000486, loss = 0.000270
grad_step = 000487, loss = 0.000259
grad_step = 000488, loss = 0.000260
grad_step = 000489, loss = 0.000265
grad_step = 000490, loss = 0.000274
grad_step = 000491, loss = 0.000281
grad_step = 000492, loss = 0.000281
grad_step = 000493, loss = 0.000275
grad_step = 000494, loss = 0.000266
grad_step = 000495, loss = 0.000256
grad_step = 000496, loss = 0.000254
grad_step = 000497, loss = 0.000256
grad_step = 000498, loss = 0.000261
grad_step = 000499, loss = 0.000267
grad_step = 000500, loss = 0.000270
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000271
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
[[0.8442733  0.86893696 0.92333335 0.94327396 1.0125437 ]
 [0.8510259  0.9212811  0.9329751  1.002209   0.9799716 ]
 [0.88709205 0.9380274  0.99374515 0.9698433  0.94657683]
 [0.93548715 0.99523365 0.99599457 0.96197665 0.92300075]
 [0.9904088  0.99751294 0.95342237 0.92366123 0.85806394]
 [0.98403776 0.96170056 0.9139071  0.8648338  0.84852225]
 [0.9437113  0.91719633 0.8460415  0.8636495  0.82670224]
 [0.89931417 0.84673816 0.85258627 0.8142806  0.84670496]
 [0.8353079  0.8374817  0.8067976  0.85925233 0.8463062 ]
 [0.84011024 0.8180039  0.8302653  0.869938   0.83237594]
 [0.79786813 0.8118698  0.8486674  0.8414202  0.9395963 ]
 [0.80252457 0.83788633 0.8273846  0.93838733 0.9499197 ]
 [0.8379933  0.867585   0.9184192  0.9424169  1.0052912 ]
 [0.8517103  0.9274476  0.9403629  1.0000263  0.97042143]
 [0.9027656  0.95094794 1.000803   0.9666383  0.9340154 ]
 [0.9466628  0.9987274  0.9812207  0.9436598  0.8954643 ]
 [0.99434316 0.9952659  0.9340126  0.9030512  0.8419595 ]
 [0.9761294  0.9462855  0.8926017  0.8446613  0.8367741 ]
 [0.93542653 0.9075873  0.82957125 0.8544349  0.82062113]
 [0.9018167  0.8471061  0.8469139  0.81491685 0.8445257 ]
 [0.84585637 0.8465987  0.8047378  0.8555526  0.8570951 ]
 [0.85784996 0.8328282  0.8311457  0.8742894  0.8380139 ]
 [0.81481004 0.82251793 0.85912514 0.8488218  0.94149995]
 [0.81301963 0.8453777  0.83369595 0.9448122  0.9531789 ]
 [0.84955597 0.87230337 0.9245405  0.9469795  1.0197415 ]
 [0.85961664 0.9293602  0.9385557  1.0094898  0.9909366 ]
 [0.8952856  0.94889235 1.0012432  0.9805568  0.95812684]
 [0.94802606 1.0087826  1.0083747  0.9738951  0.93366337]
 [1.0000045  1.0103283  0.9615036  0.9341711  0.8655388 ]
 [0.9935919  0.97522426 0.92269963 0.8721726  0.8551011 ]
 [0.95141995 0.9261109  0.85195047 0.8690556  0.8348831 ]]

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
[master 744f31c] ml_store
 1 file changed, 1123 insertions(+)
To github.com:arita37/mlmodels_store.git
   24f5440..744f31c  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   744f31c..ae5dfcc  master     -> origin/master
Updating 744f31c..ae5dfcc
Fast-forward
 .../20200522/list_log_pullrequest_20200522.md      |  2 +-
 error_list/20200522/list_log_testall_20200522.md   | 36 ++++++++++++++++++++--
 2 files changed, 35 insertions(+), 3 deletions(-)
[master f9c2142] ml_store
 1 file changed, 53 insertions(+)
To github.com:arita37/mlmodels_store.git
   ae5dfcc..f9c2142  master -> master





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
[master cc996ea] ml_store
 1 file changed, 38 insertions(+)
To github.com:arita37/mlmodels_store.git
   f9c2142..cc996ea  master -> master





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
[master 9d01fb6] ml_store
 1 file changed, 32 insertions(+)
To github.com:arita37/mlmodels_store.git
   cc996ea..9d01fb6  master -> master





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
[master c4778bf] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   9d01fb6..c4778bf  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
start

  #### Loading params   ############################################## 

  ############# Data, Params preparation   ################# 

  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'} 

  #### Loading dataset   ############################################# 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]

  #### Model init  ############################################# 

  #### Model fit   ############################################# 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]
          0         1         2         3         4         5
0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000

  #### Predict   ##################################################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]
          0         1         2         3         4         5
0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
5  0.745928  0.883387  0.838176  0.904464  0.904464  0.370110
6  1.000000  0.881878  0.467996  0.486496  0.486496  1.000000
7  0.216516  0.077549  0.433808  0.329598  0.329598  0.318466
8  0.195249  0.000000  0.000000  0.000000  0.000000  0.671960
9  0.000000  0.173783  0.369041  0.411721  0.411721  0.304384

  #### metrics   ##################################################### 
{'loss': 0.42859114333987236, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-22 04:49:07.436833: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1365, in _do_call
    return fn(*args)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1350, in _run_fn
    target_list, run_metadata)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1443, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.NotFoundError: Key Variable not found in checkpoint
	 [[{{node save_1/RestoreV2}}]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1290, in restore
    {self.saver_def.filename_tensor_name: save_path})
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 956, in run
    run_metadata_ptr)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1180, in _run
    feed_dict_tensor, options, run_metadata)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1359, in _do_run
    run_metadata)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1384, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.NotFoundError: Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
    test(data_path="", pars_choice="test01", config_mode="test")
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
    saver      = tf.compat.v1.train.Saver()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 828, in __init__
    self.build()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 840, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 878, in _build
    build_restore=build_restore)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 508, in _build_internal
    restore_sequentially, reshape)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 328, in _AddRestoreOps
    restore_sequentially)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 575, in bulk_restore
    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_io_ops.py", line 1696, in restore_v2
    name=name)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
    attrs, op_def, compute_device)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
    self._traceback = tf_stack.extract_stack()


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1300, in restore
    names_to_keys = object_graph_key_mapping(save_path)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1618, in object_graph_key_mapping
    object_graph_string = reader.get_tensor(trackable.OBJECT_GRAPH_PROTO_KEY)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/pywrap_tensorflow_internal.py", line 915, in get_tensor
    return CheckpointReader_GetTensor(self, compat.as_bytes(tensor_str))
tensorflow.python.framework.errors_impl.NotFoundError: Key _CHECKPOINTABLE_OBJECT_GRAPH not found in checkpoint

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
    test(data_path="", pars_choice="test01", config_mode="test")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
    session = load(out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 477, in load_tf
    saver.restore(sess,  full_name)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1306, in restore
    err, "a Variable name or other graph key that is missing")
tensorflow.python.framework.errors_impl.NotFoundError: Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
    test(data_path="", pars_choice="test01", config_mode="test")
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
    saver      = tf.compat.v1.train.Saver()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 828, in __init__
    self.build()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 840, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 878, in _build
    build_restore=build_restore)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 508, in _build_internal
    restore_sequentially, reshape)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 328, in _AddRestoreOps
    restore_sequentially)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 575, in bulk_restore
    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_io_ops.py", line 1696, in restore_v2
    name=name)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
    attrs, op_def, compute_device)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
    self._traceback = tf_stack.extract_stack()


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
[master 7f4b721] ml_store
 1 file changed, 234 insertions(+)
To github.com:arita37/mlmodels_store.git
   c4778bf..7f4b721  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
    from mlmodels.mode_tf.raw  import temporal_fusion_google
ModuleNotFoundError: No module named 'mlmodels.mode_tf'

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
[master 738c726] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   7f4b721..738c726  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 1.02817479 -0.50845713  1.7653351   0.77741921  0.61771419 -0.11877117
   0.45015551 -0.19899818  1.86647138  0.8709698 ]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 1.07258847 -0.58652394 -1.34267579 -1.23685338  1.24328724  0.87583893
  -0.3264995   0.62336218 -0.43495668  1.11438298]
 [ 1.25704434 -1.82391985 -0.61240697  1.16707517 -0.62373281 -0.0396687
   0.81604368  0.8858258   0.18986165  0.39310924]
 [ 1.1437713   0.7278135   0.35249436  0.51507361  1.17718111 -2.78253447
  -1.94332341  0.58464661  0.32427424 -0.23643695]
 [ 1.16777676 -0.66575452 -1.23312074 -1.67419581  1.01313574  0.82502982
  -0.12046457 -0.49821356 -0.31098498 -1.18231813]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 0.98042741  1.93752881 -0.23083974  0.36633201  1.10018476 -1.04458938
  -0.34498721  2.05117344  0.585662   -2.793085  ]
 [ 1.06523311 -0.66486777  1.00806543 -1.94504696 -1.23017555 -0.91542437
   0.33722094  1.22515585 -1.05354607  0.78522692]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 1.24549398 -0.72239191  1.1181334   1.09899633  1.00277655 -0.90163449
  -0.53223402 -0.82246719  0.72171129  0.6743961 ]
 [ 0.96457205 -0.10679399  1.12232832  1.45142926  1.21828168 -0.61803685
   0.43816635 -2.03720123 -1.94258918 -0.9970198 ]
 [ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 0.72297801  0.18553562  0.91549927  0.39442803 -0.84983074  0.72552256
  -0.15050433  1.49588477  0.67545381 -0.43820027]
 [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 0.85771953  0.09811225 -0.26046606  1.06032751 -1.39003042 -1.71116766
   0.2656424   1.65712464  1.41767401  0.44509671]
 [ 1.06702918 -0.42914228  0.35016716  1.20845633  0.75148062  1.1157018
  -0.4791571   0.84086156 -0.10288722  0.01716473]
 [ 1.18559003  0.08646441  1.23289919 -2.14246673  1.033341   -0.83016886
   0.36723181  0.45161595  1.10417433 -0.42285696]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f1eed0a3d30>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f1f07422550> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.01195228 -1.88141087  1.70018815  0.4972691  -0.91766462  0.2373327
  -1.09033833 -2.14444405 -0.36956243  0.60878366]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 0.88838944  0.28299553  0.01795589  0.10803082 -0.84967187  0.02941762
  -0.50397395 -0.13479313  1.04921829 -1.27046078]
 [ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 0.88861146  0.84958685 -0.03091142 -0.12215402 -1.14722826 -0.68085157
  -0.32606131 -1.06787658 -0.07667936  0.35571726]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 1.25704434 -1.82391985 -0.61240697  1.16707517 -0.62373281 -0.0396687
   0.81604368  0.8858258   0.18986165  0.39310924]
 [ 0.78344054 -0.05118845  0.82458463 -0.72559712  0.9317172  -0.86776868
   3.03085711 -0.13597733 -0.79726979  0.65458015]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 0.6109426  -2.79099641 -1.33520272 -0.45611756 -0.94495995 -0.97989025
  -0.15699367  0.69257435 -0.47867236 -0.10646012]
 [ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
  -0.87728152 -0.78191168 -0.43750898 -1.44087602]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 0.8786438   1.03703898 -0.47712421  0.67261975 -1.04948638  2.42887697
   0.52475049  1.00568668  0.35356722 -0.03599018]
 [ 0.98042741  1.93752881 -0.23083974  0.36633201  1.10018476 -1.04458938
  -0.34498721  2.05117344  0.585662   -2.793085  ]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 0.61363671  0.3166589   1.34710546 -1.89526695 -0.76045809  0.08972912
  -0.32905155  0.41026575  0.85987097 -1.04906775]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 0.56998385 -0.53302033 -0.17545897 -1.42655542  0.60660431  1.76795995
  -0.11598519 -0.47537288  0.47761018 -0.93391466]
 [ 0.9292506  -1.10657307 -1.95816909 -0.3592241  -1.21258781  0.5053819
   0.54264529  1.2179409  -1.94068096  0.67780757]
 [ 0.93621125  0.20437739 -1.49419377  0.61223252 -0.98437725  0.74488454
   0.49434165 -0.03628129 -0.83239535 -0.4466992 ]
 [ 1.09488485 -0.06962454 -0.11644415  0.35387043 -1.44189096 -0.18695502
   1.2911889  -0.15323616 -2.43250851 -2.277298  ]]
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
[[ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 1.14809657 -0.7332716   0.26246745  0.83600472  1.17353145  1.54335911
   0.28474811  0.75880566  0.88490881  0.2764993 ]
 [ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 0.55853873 -0.51634791 -0.51814555  0.3511169   0.82550695 -0.06877046
  -0.9520621  -1.34776494  1.47073986 -1.4614036 ]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 1.03967316 -0.73153098  0.36184732 -1.56573815  0.95928819  1.01382247
  -1.78791289 -2.22711263 -1.6993336  -0.42449279]
 [ 1.64661853 -1.52568032 -0.6069984   0.79502609  1.08480038 -0.37443832
   0.42952614  0.1340482   1.20205486  0.10622272]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 0.77528533  1.47016034  1.03298378 -0.87000822  0.78655651  0.36919047
  -0.14319575  0.85328219 -0.13971173 -0.22241403]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 0.99785516 -0.6001388   0.45794708  0.14676526 -0.93355729  0.57180488
   0.57296273 -0.03681766  0.11236849 -0.01781755]
 [ 0.88861146  0.84958685 -0.03091142 -0.12215402 -1.14722826 -0.68085157
  -0.32606131 -1.06787658 -0.07667936  0.35571726]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 1.13545112  0.8616231   0.04906169 -2.08639057 -1.1146902   0.36180164
  -0.80617821  0.42592018  0.0490804  -0.59608633]
 [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]
 [ 0.35413361  0.21112476  0.92145007  0.01652757  0.90394545  0.17718772
   0.09542509 -1.11647002  0.0809271   0.0607502 ]
 [ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 1.77547698 -0.20339445 -0.19883786  0.24266944  0.96435056  0.20183018
  -0.54577417  0.66102029  1.79215821 -0.7003985 ]
 [ 1.02817479 -0.50845713  1.7653351   0.77741921  0.61771419 -0.11877117
   0.45015551 -0.19899818  1.86647138  0.8709698 ]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]
 [ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
  -1.09715436 -0.50897228 -0.16648595 -1.03918232]]
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
[master 972025b] ml_store
 1 file changed, 247 insertions(+)
To github.com:arita37/mlmodels_store.git
   738c726..972025b  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f23ff96e128> 

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
Already up to date.
[master eccd245] ml_store
 1 file changed, 109 insertions(+)
To github.com:arita37/mlmodels_store.git
   972025b..eccd245  master -> master





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
[master 102d4ef] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   eccd245..102d4ef  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2703360/17464789 [===>..........................] - ETA: 0s
 5488640/17464789 [========>.....................] - ETA: 0s
 7544832/17464789 [===========>..................] - ETA: 0s
 9723904/17464789 [===============>..............] - ETA: 0s
11558912/17464789 [==================>...........] - ETA: 0s
13819904/17464789 [======================>.......] - ETA: 0s
15777792/17464789 [==========================>...] - ETA: 0s
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
2020-05-22 04:50:12.459395: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-22 04:50:12.464177: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-22 04:50:12.464330: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f1ec2c9740 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 04:50:12.464348: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 14s - loss: 7.9580 - accuracy: 0.4810
 2000/25000 [=>............................] - ETA: 10s - loss: 7.8430 - accuracy: 0.4885
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6768 - accuracy: 0.4993 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6781 - accuracy: 0.4992
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6942 - accuracy: 0.4982
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7050 - accuracy: 0.4975
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6644 - accuracy: 0.5001
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6321 - accuracy: 0.5023
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6121 - accuracy: 0.5036
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5946 - accuracy: 0.5047
11000/25000 [============>.................] - ETA: 4s - loss: 7.6123 - accuracy: 0.5035
12000/25000 [=============>................] - ETA: 4s - loss: 7.6411 - accuracy: 0.5017
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6643 - accuracy: 0.5002
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6524 - accuracy: 0.5009
15000/25000 [=================>............] - ETA: 3s - loss: 7.6687 - accuracy: 0.4999
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6576 - accuracy: 0.5006
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6538 - accuracy: 0.5008
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6626 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6651 - accuracy: 0.5001
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6695 - accuracy: 0.4998
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6959 - accuracy: 0.4981
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6893 - accuracy: 0.4985
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6864 - accuracy: 0.4987
25000/25000 [==============================] - 10s 401us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f3d0fd63f28>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f3d1274dd30> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.4673 - accuracy: 0.5130
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7331 - accuracy: 0.4957 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7050 - accuracy: 0.4975
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6758 - accuracy: 0.4994
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6947 - accuracy: 0.4982
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6732 - accuracy: 0.4996
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7126 - accuracy: 0.4970
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6888 - accuracy: 0.4986
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6789 - accuracy: 0.4992
11000/25000 [============>.................] - ETA: 4s - loss: 7.6387 - accuracy: 0.5018
12000/25000 [=============>................] - ETA: 4s - loss: 7.6449 - accuracy: 0.5014
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6395 - accuracy: 0.5018
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6371 - accuracy: 0.5019
15000/25000 [=================>............] - ETA: 3s - loss: 7.6329 - accuracy: 0.5022
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6283 - accuracy: 0.5025
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6405 - accuracy: 0.5017
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6504 - accuracy: 0.5011
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6440 - accuracy: 0.5015
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6551 - accuracy: 0.5008
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6725 - accuracy: 0.4996
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6633 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
25000/25000 [==============================] - 10s 394us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.4060 - accuracy: 0.5170
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6002 - accuracy: 0.5043 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5440 - accuracy: 0.5080
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5348 - accuracy: 0.5086
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5465 - accuracy: 0.5078
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5461 - accuracy: 0.5079
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5957 - accuracy: 0.5046
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5900 - accuracy: 0.5050
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5930 - accuracy: 0.5048
11000/25000 [============>.................] - ETA: 4s - loss: 7.6081 - accuracy: 0.5038
12000/25000 [=============>................] - ETA: 4s - loss: 7.6181 - accuracy: 0.5032
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6277 - accuracy: 0.5025
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
15000/25000 [=================>............] - ETA: 3s - loss: 7.6411 - accuracy: 0.5017
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6628 - accuracy: 0.5002
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6874 - accuracy: 0.4986
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6913 - accuracy: 0.4984
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6763 - accuracy: 0.4994
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6958 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6878 - accuracy: 0.4986
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6938 - accuracy: 0.4982
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6780 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
25000/25000 [==============================] - 10s 400us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
From github.com:arita37/mlmodels_store
   102d4ef..e622510  master     -> origin/master
Updating 102d4ef..e622510
Fast-forward
 .../20200522/list_log_pullrequest_20200522.md      |   2 +-
 error_list/20200522/list_log_testall_20200522.md   | 222 +++++++++++++++++++++
 2 files changed, 223 insertions(+), 1 deletion(-)
[master 37bd8ef] ml_store
 1 file changed, 328 insertions(+)
To github.com:arita37/mlmodels_store.git
   e622510..37bd8ef  master -> master





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
[master ff0b451] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   37bd8ef..ff0b451  master -> master





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

13/13 [==============================] - 2s 136ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 8/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 9/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 10/10

13/13 [==============================] - 0s 4ms/step - loss: nan

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
[master 4b410e8] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   ff0b451..4b410e8  master -> master





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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-22 04:52:08.514110: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-22 04:52:08.519873: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-22 04:52:08.520054: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558a03a2fb50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 04:52:08.520071: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 253
Trainable params: 253
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 1ms/sample - loss: 0.2503 - binary_crossentropy: 0.6936 - val_loss: 0.2500 - val_binary_crossentropy: 0.6931

  #### metrics   #################################################### 
{'MSE': 0.24990951626057759}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
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
Total params: 253
Trainable params: 253
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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 433
Trainable params: 433
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2974 - binary_crossentropy: 0.8088500/500 [==============================] - 1s 2ms/sample - loss: 0.2844 - binary_crossentropy: 0.7778 - val_loss: 0.2894 - val_binary_crossentropy: 0.7874

  #### metrics   #################################################### 
{'MSE': 0.28602239512501376}

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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 433
Trainable params: 433
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 637
Trainable params: 637
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.8197500/500 [==============================] - 1s 2ms/sample - loss: 0.2521 - binary_crossentropy: 0.8508 - val_loss: 0.2631 - val_binary_crossentropy: 0.9824

  #### metrics   #################################################### 
{'MSE': 0.2574316914969835}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 637
Trainable params: 637
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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
Total params: 453
Trainable params: 453
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3409 - binary_crossentropy: 0.9922500/500 [==============================] - 1s 3ms/sample - loss: 0.3542 - binary_crossentropy: 1.0189 - val_loss: 0.3537 - val_binary_crossentropy: 0.9940

  #### metrics   #################################################### 
{'MSE': 0.3510073751534077}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
Total params: 453
Trainable params: 453
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 183
Trainable params: 183
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 3s - loss: 0.3136 - binary_crossentropy: 0.8355500/500 [==============================] - 2s 4ms/sample - loss: 0.2976 - binary_crossentropy: 0.7997 - val_loss: 0.2932 - val_binary_crossentropy: 0.8420

  #### metrics   #################################################### 
{'MSE': 0.2939899222053267}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 183
Trainable params: 183
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-22 04:53:38.860437: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:53:38.862768: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:53:38.869144: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 04:53:38.879879: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 04:53:38.881986: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 04:53:38.883762: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:53:38.885447: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2487 - val_binary_crossentropy: 0.6905
2020-05-22 04:53:40.275926: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:53:40.278071: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:53:40.282844: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 04:53:40.292465: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 04:53:40.294107: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 04:53:40.295640: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:53:40.297154: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24814578297212364}

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
2020-05-22 04:54:06.929129: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:54:06.930625: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:54:06.934980: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 04:54:06.942046: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 04:54:06.943256: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 04:54:06.944544: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:54:06.945634: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2500 - val_binary_crossentropy: 0.6931
2020-05-22 04:54:08.636481: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:54:08.637893: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:54:08.641004: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 04:54:08.646921: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 04:54:08.648020: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 04:54:08.648908: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:54:08.649958: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2498626591797546}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-22 04:54:46.841557: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:54:46.847228: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:54:46.863141: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 04:54:46.890785: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 04:54:46.895622: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 04:54:46.899779: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:54:46.904325: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.5905 - binary_crossentropy: 1.4628 - val_loss: 0.2508 - val_binary_crossentropy: 0.6948
2020-05-22 04:54:49.382851: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:54:49.388102: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:54:49.401906: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 04:54:49.430884: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 04:54:49.435916: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 04:54:49.440610: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 04:54:49.444946: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24154867986003836}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 700
Trainable params: 700
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3188 - binary_crossentropy: 1.6477500/500 [==============================] - 5s 9ms/sample - loss: 0.3279 - binary_crossentropy: 1.5603 - val_loss: 0.3552 - val_binary_crossentropy: 1.8101

  #### metrics   #################################################### 
{'MSE': 0.3404759166403287}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 700
Trainable params: 700
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 2)         10          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 2)         14          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         18          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         10          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         14          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_5[0][0]           
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
Total params: 281
Trainable params: 281
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2875 - binary_crossentropy: 0.7852500/500 [==============================] - 5s 10ms/sample - loss: 0.2884 - binary_crossentropy: 0.7854 - val_loss: 0.2877 - val_binary_crossentropy: 0.7813

  #### metrics   #################################################### 
{'MSE': 0.2859760605358896}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 2)         10          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 2)         14          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         18          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         10          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         14          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_5[0][0]           
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
Total params: 281
Trainable params: 281
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_max[0][0]               
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
Total params: 1,864
Trainable params: 1,864
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2500 - binary_crossentropy: 0.6933500/500 [==============================] - 5s 10ms/sample - loss: 0.2474 - binary_crossentropy: 0.6879 - val_loss: 0.2538 - val_binary_crossentropy: 0.7009

  #### metrics   #################################################### 
{'MSE': 0.25041179438158884}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_max[0][0]               
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
Total params: 1,864
Trainable params: 1,864
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
regionsequence_mean (InputLayer [(None, 1)]          0                                            
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
region_10sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         6           regionsequence_mean[0][0]        
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
Total params: 176
Trainable params: 176
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 10s - loss: 0.2840 - binary_crossentropy: 1.5454500/500 [==============================] - 7s 13ms/sample - loss: 0.2984 - binary_crossentropy: 1.9154 - val_loss: 0.2844 - val_binary_crossentropy: 1.5722

  #### metrics   #################################################### 
{'MSE': 0.29122927445031643}

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
regionsequence_mean (InputLayer [(None, 1)]          0                                            
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
region_10sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         6           regionsequence_mean[0][0]        
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_max[0][0]               
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
Total params: 1,437
Trainable params: 1,437
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2835 - binary_crossentropy: 0.7737500/500 [==============================] - 7s 13ms/sample - loss: 0.2779 - binary_crossentropy: 0.7595 - val_loss: 0.2741 - val_binary_crossentropy: 0.7477

  #### metrics   #################################################### 
{'MSE': 0.2729530779377926}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_max[0][0]               
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
Total params: 1,437
Trainable params: 1,437
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         28          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         16          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_11[0][0]                    
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
Total params: 3,035
Trainable params: 2,955
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2557 - binary_crossentropy: 0.7047500/500 [==============================] - 7s 14ms/sample - loss: 0.2554 - binary_crossentropy: 0.7306 - val_loss: 0.2484 - val_binary_crossentropy: 0.6899

  #### metrics   #################################################### 
{'MSE': 0.25031209688823175}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         28          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         16          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_11[0][0]                    
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
Total params: 3,035
Trainable params: 2,955
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
   4b410e8..9da0864  master     -> origin/master
Updating 4b410e8..9da0864
Fast-forward
 .../20200522/list_log_pullrequest_20200522.md      |   2 +-
 error_list/20200522/list_log_testall_20200522.md   | 375 +++++++++++++++++++++
 2 files changed, 376 insertions(+), 1 deletion(-)
[master 9cf2bb5] ml_store
 1 file changed, 4953 insertions(+)
To github.com:arita37/mlmodels_store.git
   9da0864..9cf2bb5  master -> master





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
[master 1e4627e] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   9cf2bb5..1e4627e  master -> master





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
 1867776/11490434 [===>..........................] - ETA: 0s
 7389184/11490434 [==================>...........] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:42 - loss: 2.3122 - categorical_accuracy: 0.0938
   64/60000 [..............................] - ETA: 4:52 - loss: 2.2627 - categorical_accuracy: 0.1562
   96/60000 [..............................] - ETA: 3:52 - loss: 2.2799 - categorical_accuracy: 0.1458
  128/60000 [..............................] - ETA: 3:21 - loss: 2.2572 - categorical_accuracy: 0.1562
  160/60000 [..............................] - ETA: 3:03 - loss: 2.2301 - categorical_accuracy: 0.2000
  192/60000 [..............................] - ETA: 2:50 - loss: 2.2090 - categorical_accuracy: 0.2188
  224/60000 [..............................] - ETA: 2:42 - loss: 2.1776 - categorical_accuracy: 0.2277
  256/60000 [..............................] - ETA: 2:35 - loss: 2.1313 - categorical_accuracy: 0.2383
  288/60000 [..............................] - ETA: 2:30 - loss: 2.0690 - categorical_accuracy: 0.2604
  320/60000 [..............................] - ETA: 2:27 - loss: 2.0323 - categorical_accuracy: 0.2750
  352/60000 [..............................] - ETA: 2:24 - loss: 2.0101 - categorical_accuracy: 0.2898
  384/60000 [..............................] - ETA: 2:21 - loss: 1.9621 - categorical_accuracy: 0.3125
  416/60000 [..............................] - ETA: 2:19 - loss: 1.9275 - categorical_accuracy: 0.3293
  448/60000 [..............................] - ETA: 2:16 - loss: 1.8975 - categorical_accuracy: 0.3460
  480/60000 [..............................] - ETA: 2:14 - loss: 1.8757 - categorical_accuracy: 0.3521
  512/60000 [..............................] - ETA: 2:13 - loss: 1.8187 - categorical_accuracy: 0.3711
  544/60000 [..............................] - ETA: 2:12 - loss: 1.7950 - categorical_accuracy: 0.3787
  576/60000 [..............................] - ETA: 2:11 - loss: 1.7676 - categorical_accuracy: 0.3906
  608/60000 [..............................] - ETA: 2:10 - loss: 1.7374 - categorical_accuracy: 0.3980
  640/60000 [..............................] - ETA: 2:10 - loss: 1.7144 - categorical_accuracy: 0.4078
  672/60000 [..............................] - ETA: 2:08 - loss: 1.6813 - categorical_accuracy: 0.4211
  704/60000 [..............................] - ETA: 2:08 - loss: 1.6569 - categorical_accuracy: 0.4318
  736/60000 [..............................] - ETA: 2:07 - loss: 1.6403 - categorical_accuracy: 0.4416
  768/60000 [..............................] - ETA: 2:06 - loss: 1.6155 - categorical_accuracy: 0.4518
  800/60000 [..............................] - ETA: 2:05 - loss: 1.5905 - categorical_accuracy: 0.4575
  832/60000 [..............................] - ETA: 2:04 - loss: 1.5634 - categorical_accuracy: 0.4675
  864/60000 [..............................] - ETA: 2:04 - loss: 1.5417 - categorical_accuracy: 0.4757
  896/60000 [..............................] - ETA: 2:03 - loss: 1.5140 - categorical_accuracy: 0.4877
  928/60000 [..............................] - ETA: 2:02 - loss: 1.4973 - categorical_accuracy: 0.4946
  960/60000 [..............................] - ETA: 2:02 - loss: 1.4732 - categorical_accuracy: 0.5042
  992/60000 [..............................] - ETA: 2:01 - loss: 1.4576 - categorical_accuracy: 0.5121
 1024/60000 [..............................] - ETA: 2:01 - loss: 1.4365 - categorical_accuracy: 0.5195
 1056/60000 [..............................] - ETA: 2:01 - loss: 1.4165 - categorical_accuracy: 0.5237
 1088/60000 [..............................] - ETA: 2:00 - loss: 1.3982 - categorical_accuracy: 0.5294
 1120/60000 [..............................] - ETA: 2:00 - loss: 1.3904 - categorical_accuracy: 0.5330
 1152/60000 [..............................] - ETA: 2:00 - loss: 1.3801 - categorical_accuracy: 0.5373
 1184/60000 [..............................] - ETA: 1:59 - loss: 1.3615 - categorical_accuracy: 0.5465
 1216/60000 [..............................] - ETA: 1:59 - loss: 1.3469 - categorical_accuracy: 0.5526
 1248/60000 [..............................] - ETA: 1:59 - loss: 1.3404 - categorical_accuracy: 0.5553
 1280/60000 [..............................] - ETA: 1:58 - loss: 1.3296 - categorical_accuracy: 0.5602
 1312/60000 [..............................] - ETA: 1:58 - loss: 1.3146 - categorical_accuracy: 0.5648
 1344/60000 [..............................] - ETA: 1:58 - loss: 1.3004 - categorical_accuracy: 0.5685
 1376/60000 [..............................] - ETA: 1:58 - loss: 1.2865 - categorical_accuracy: 0.5719
 1408/60000 [..............................] - ETA: 1:58 - loss: 1.2698 - categorical_accuracy: 0.5781
 1440/60000 [..............................] - ETA: 1:58 - loss: 1.2562 - categorical_accuracy: 0.5833
 1472/60000 [..............................] - ETA: 1:57 - loss: 1.2478 - categorical_accuracy: 0.5876
 1504/60000 [..............................] - ETA: 1:57 - loss: 1.2350 - categorical_accuracy: 0.5924
 1536/60000 [..............................] - ETA: 1:57 - loss: 1.2211 - categorical_accuracy: 0.5970
 1568/60000 [..............................] - ETA: 1:56 - loss: 1.2062 - categorical_accuracy: 0.6014
 1600/60000 [..............................] - ETA: 1:56 - loss: 1.1946 - categorical_accuracy: 0.6062
 1632/60000 [..............................] - ETA: 1:56 - loss: 1.1812 - categorical_accuracy: 0.6091
 1664/60000 [..............................] - ETA: 1:56 - loss: 1.1686 - categorical_accuracy: 0.6130
 1696/60000 [..............................] - ETA: 1:55 - loss: 1.1585 - categorical_accuracy: 0.6156
 1728/60000 [..............................] - ETA: 1:55 - loss: 1.1520 - categorical_accuracy: 0.6163
 1760/60000 [..............................] - ETA: 1:55 - loss: 1.1482 - categorical_accuracy: 0.6182
 1792/60000 [..............................] - ETA: 1:55 - loss: 1.1390 - categorical_accuracy: 0.6211
 1824/60000 [..............................] - ETA: 1:54 - loss: 1.1261 - categorical_accuracy: 0.6261
 1856/60000 [..............................] - ETA: 1:54 - loss: 1.1142 - categorical_accuracy: 0.6304
 1888/60000 [..............................] - ETA: 1:54 - loss: 1.1033 - categorical_accuracy: 0.6335
 1920/60000 [..............................] - ETA: 1:54 - loss: 1.0975 - categorical_accuracy: 0.6365
 1952/60000 [..............................] - ETA: 1:54 - loss: 1.0924 - categorical_accuracy: 0.6388
 1984/60000 [..............................] - ETA: 1:54 - loss: 1.0805 - categorical_accuracy: 0.6431
 2016/60000 [>.............................] - ETA: 1:53 - loss: 1.0738 - categorical_accuracy: 0.6458
 2048/60000 [>.............................] - ETA: 1:53 - loss: 1.0684 - categorical_accuracy: 0.6475
 2080/60000 [>.............................] - ETA: 1:53 - loss: 1.0590 - categorical_accuracy: 0.6505
 2112/60000 [>.............................] - ETA: 1:53 - loss: 1.0471 - categorical_accuracy: 0.6544
 2144/60000 [>.............................] - ETA: 1:53 - loss: 1.0362 - categorical_accuracy: 0.6576
 2176/60000 [>.............................] - ETA: 1:53 - loss: 1.0272 - categorical_accuracy: 0.6604
 2208/60000 [>.............................] - ETA: 1:53 - loss: 1.0240 - categorical_accuracy: 0.6626
 2240/60000 [>.............................] - ETA: 1:52 - loss: 1.0183 - categorical_accuracy: 0.6652
 2272/60000 [>.............................] - ETA: 1:52 - loss: 1.0099 - categorical_accuracy: 0.6686
 2304/60000 [>.............................] - ETA: 1:52 - loss: 1.0063 - categorical_accuracy: 0.6680
 2336/60000 [>.............................] - ETA: 1:52 - loss: 1.0005 - categorical_accuracy: 0.6691
 2368/60000 [>.............................] - ETA: 1:52 - loss: 0.9932 - categorical_accuracy: 0.6706
 2400/60000 [>.............................] - ETA: 1:52 - loss: 0.9837 - categorical_accuracy: 0.6742
 2432/60000 [>.............................] - ETA: 1:52 - loss: 0.9776 - categorical_accuracy: 0.6760
 2464/60000 [>.............................] - ETA: 1:52 - loss: 0.9688 - categorical_accuracy: 0.6794
 2496/60000 [>.............................] - ETA: 1:51 - loss: 0.9616 - categorical_accuracy: 0.6823
 2528/60000 [>.............................] - ETA: 1:51 - loss: 0.9547 - categorical_accuracy: 0.6847
 2560/60000 [>.............................] - ETA: 1:51 - loss: 0.9482 - categorical_accuracy: 0.6875
 2592/60000 [>.............................] - ETA: 1:51 - loss: 0.9442 - categorical_accuracy: 0.6887
 2624/60000 [>.............................] - ETA: 1:51 - loss: 0.9361 - categorical_accuracy: 0.6909
 2656/60000 [>.............................] - ETA: 1:51 - loss: 0.9305 - categorical_accuracy: 0.6931
 2688/60000 [>.............................] - ETA: 1:51 - loss: 0.9214 - categorical_accuracy: 0.6961
 2720/60000 [>.............................] - ETA: 1:51 - loss: 0.9197 - categorical_accuracy: 0.6956
 2752/60000 [>.............................] - ETA: 1:50 - loss: 0.9116 - categorical_accuracy: 0.6984
 2784/60000 [>.............................] - ETA: 1:50 - loss: 0.9069 - categorical_accuracy: 0.6994
 2816/60000 [>.............................] - ETA: 1:50 - loss: 0.8992 - categorical_accuracy: 0.7013
 2848/60000 [>.............................] - ETA: 1:50 - loss: 0.8933 - categorical_accuracy: 0.7033
 2880/60000 [>.............................] - ETA: 1:50 - loss: 0.8917 - categorical_accuracy: 0.7045
 2912/60000 [>.............................] - ETA: 1:50 - loss: 0.8879 - categorical_accuracy: 0.7050
 2944/60000 [>.............................] - ETA: 1:50 - loss: 0.8803 - categorical_accuracy: 0.7079
 2976/60000 [>.............................] - ETA: 1:50 - loss: 0.8760 - categorical_accuracy: 0.7093
 3008/60000 [>.............................] - ETA: 1:49 - loss: 0.8719 - categorical_accuracy: 0.7104
 3040/60000 [>.............................] - ETA: 1:49 - loss: 0.8715 - categorical_accuracy: 0.7112
 3072/60000 [>.............................] - ETA: 1:49 - loss: 0.8670 - categorical_accuracy: 0.7129
 3104/60000 [>.............................] - ETA: 1:49 - loss: 0.8628 - categorical_accuracy: 0.7149
 3136/60000 [>.............................] - ETA: 1:49 - loss: 0.8560 - categorical_accuracy: 0.7175
 3168/60000 [>.............................] - ETA: 1:49 - loss: 0.8493 - categorical_accuracy: 0.7200
 3200/60000 [>.............................] - ETA: 1:49 - loss: 0.8432 - categorical_accuracy: 0.7222
 3232/60000 [>.............................] - ETA: 1:49 - loss: 0.8379 - categorical_accuracy: 0.7240
 3264/60000 [>.............................] - ETA: 1:49 - loss: 0.8327 - categorical_accuracy: 0.7255
 3296/60000 [>.............................] - ETA: 1:49 - loss: 0.8277 - categorical_accuracy: 0.7266
 3328/60000 [>.............................] - ETA: 1:49 - loss: 0.8240 - categorical_accuracy: 0.7278
 3360/60000 [>.............................] - ETA: 1:49 - loss: 0.8195 - categorical_accuracy: 0.7295
 3392/60000 [>.............................] - ETA: 1:49 - loss: 0.8165 - categorical_accuracy: 0.7311
 3424/60000 [>.............................] - ETA: 1:48 - loss: 0.8105 - categorical_accuracy: 0.7334
 3456/60000 [>.............................] - ETA: 1:48 - loss: 0.8108 - categorical_accuracy: 0.7341
 3488/60000 [>.............................] - ETA: 1:48 - loss: 0.8070 - categorical_accuracy: 0.7345
 3520/60000 [>.............................] - ETA: 1:48 - loss: 0.8011 - categorical_accuracy: 0.7364
 3552/60000 [>.............................] - ETA: 1:48 - loss: 0.7985 - categorical_accuracy: 0.7373
 3584/60000 [>.............................] - ETA: 1:48 - loss: 0.7936 - categorical_accuracy: 0.7394
 3616/60000 [>.............................] - ETA: 1:48 - loss: 0.7889 - categorical_accuracy: 0.7409
 3648/60000 [>.............................] - ETA: 1:48 - loss: 0.7853 - categorical_accuracy: 0.7423
 3680/60000 [>.............................] - ETA: 1:48 - loss: 0.7828 - categorical_accuracy: 0.7435
 3712/60000 [>.............................] - ETA: 1:48 - loss: 0.7771 - categorical_accuracy: 0.7454
 3744/60000 [>.............................] - ETA: 1:48 - loss: 0.7761 - categorical_accuracy: 0.7452
 3776/60000 [>.............................] - ETA: 1:47 - loss: 0.7736 - categorical_accuracy: 0.7460
 3808/60000 [>.............................] - ETA: 1:47 - loss: 0.7705 - categorical_accuracy: 0.7474
 3840/60000 [>.............................] - ETA: 1:47 - loss: 0.7666 - categorical_accuracy: 0.7482
 3872/60000 [>.............................] - ETA: 1:47 - loss: 0.7626 - categorical_accuracy: 0.7497
 3904/60000 [>.............................] - ETA: 1:47 - loss: 0.7581 - categorical_accuracy: 0.7510
 3936/60000 [>.............................] - ETA: 1:47 - loss: 0.7551 - categorical_accuracy: 0.7523
 3968/60000 [>.............................] - ETA: 1:47 - loss: 0.7528 - categorical_accuracy: 0.7530
 4000/60000 [=>............................] - ETA: 1:47 - loss: 0.7501 - categorical_accuracy: 0.7538
 4032/60000 [=>............................] - ETA: 1:47 - loss: 0.7490 - categorical_accuracy: 0.7540
 4064/60000 [=>............................] - ETA: 1:47 - loss: 0.7463 - categorical_accuracy: 0.7549
 4096/60000 [=>............................] - ETA: 1:47 - loss: 0.7424 - categorical_accuracy: 0.7563
 4128/60000 [=>............................] - ETA: 1:47 - loss: 0.7386 - categorical_accuracy: 0.7575
 4160/60000 [=>............................] - ETA: 1:47 - loss: 0.7339 - categorical_accuracy: 0.7589
 4192/60000 [=>............................] - ETA: 1:47 - loss: 0.7304 - categorical_accuracy: 0.7600
 4224/60000 [=>............................] - ETA: 1:46 - loss: 0.7273 - categorical_accuracy: 0.7609
 4256/60000 [=>............................] - ETA: 1:46 - loss: 0.7259 - categorical_accuracy: 0.7613
 4288/60000 [=>............................] - ETA: 1:46 - loss: 0.7221 - categorical_accuracy: 0.7628
 4320/60000 [=>............................] - ETA: 1:46 - loss: 0.7190 - categorical_accuracy: 0.7639
 4352/60000 [=>............................] - ETA: 1:46 - loss: 0.7175 - categorical_accuracy: 0.7642
 4384/60000 [=>............................] - ETA: 1:46 - loss: 0.7140 - categorical_accuracy: 0.7653
 4416/60000 [=>............................] - ETA: 1:46 - loss: 0.7115 - categorical_accuracy: 0.7659
 4448/60000 [=>............................] - ETA: 1:46 - loss: 0.7078 - categorical_accuracy: 0.7673
 4480/60000 [=>............................] - ETA: 1:46 - loss: 0.7040 - categorical_accuracy: 0.7683
 4512/60000 [=>............................] - ETA: 1:46 - loss: 0.7007 - categorical_accuracy: 0.7693
 4544/60000 [=>............................] - ETA: 1:46 - loss: 0.6968 - categorical_accuracy: 0.7707
 4576/60000 [=>............................] - ETA: 1:46 - loss: 0.6948 - categorical_accuracy: 0.7716
 4608/60000 [=>............................] - ETA: 1:46 - loss: 0.6933 - categorical_accuracy: 0.7728
 4640/60000 [=>............................] - ETA: 1:45 - loss: 0.6900 - categorical_accuracy: 0.7739
 4672/60000 [=>............................] - ETA: 1:45 - loss: 0.6862 - categorical_accuracy: 0.7750
 4704/60000 [=>............................] - ETA: 1:45 - loss: 0.6823 - categorical_accuracy: 0.7764
 4736/60000 [=>............................] - ETA: 1:45 - loss: 0.6793 - categorical_accuracy: 0.7774
 4768/60000 [=>............................] - ETA: 1:45 - loss: 0.6768 - categorical_accuracy: 0.7785
 4800/60000 [=>............................] - ETA: 1:45 - loss: 0.6733 - categorical_accuracy: 0.7796
 4832/60000 [=>............................] - ETA: 1:45 - loss: 0.6718 - categorical_accuracy: 0.7802
 4864/60000 [=>............................] - ETA: 1:45 - loss: 0.6705 - categorical_accuracy: 0.7804
 4896/60000 [=>............................] - ETA: 1:45 - loss: 0.6684 - categorical_accuracy: 0.7812
 4928/60000 [=>............................] - ETA: 1:45 - loss: 0.6670 - categorical_accuracy: 0.7821
 4960/60000 [=>............................] - ETA: 1:45 - loss: 0.6640 - categorical_accuracy: 0.7833
 4992/60000 [=>............................] - ETA: 1:45 - loss: 0.6610 - categorical_accuracy: 0.7847
 5024/60000 [=>............................] - ETA: 1:44 - loss: 0.6579 - categorical_accuracy: 0.7858
 5056/60000 [=>............................] - ETA: 1:44 - loss: 0.6548 - categorical_accuracy: 0.7868
 5088/60000 [=>............................] - ETA: 1:44 - loss: 0.6521 - categorical_accuracy: 0.7877
 5120/60000 [=>............................] - ETA: 1:44 - loss: 0.6495 - categorical_accuracy: 0.7883
 5152/60000 [=>............................] - ETA: 1:44 - loss: 0.6470 - categorical_accuracy: 0.7892
 5184/60000 [=>............................] - ETA: 1:44 - loss: 0.6442 - categorical_accuracy: 0.7901
 5216/60000 [=>............................] - ETA: 1:44 - loss: 0.6434 - categorical_accuracy: 0.7908
 5248/60000 [=>............................] - ETA: 1:44 - loss: 0.6426 - categorical_accuracy: 0.7912
 5280/60000 [=>............................] - ETA: 1:44 - loss: 0.6400 - categorical_accuracy: 0.7917
 5312/60000 [=>............................] - ETA: 1:44 - loss: 0.6370 - categorical_accuracy: 0.7927
 5344/60000 [=>............................] - ETA: 1:44 - loss: 0.6346 - categorical_accuracy: 0.7934
 5376/60000 [=>............................] - ETA: 1:44 - loss: 0.6324 - categorical_accuracy: 0.7939
 5408/60000 [=>............................] - ETA: 1:44 - loss: 0.6293 - categorical_accuracy: 0.7951
 5440/60000 [=>............................] - ETA: 1:44 - loss: 0.6269 - categorical_accuracy: 0.7960
 5472/60000 [=>............................] - ETA: 1:44 - loss: 0.6255 - categorical_accuracy: 0.7964
 5504/60000 [=>............................] - ETA: 1:44 - loss: 0.6226 - categorical_accuracy: 0.7974
 5536/60000 [=>............................] - ETA: 1:44 - loss: 0.6203 - categorical_accuracy: 0.7982
 5568/60000 [=>............................] - ETA: 1:44 - loss: 0.6188 - categorical_accuracy: 0.7989
 5600/60000 [=>............................] - ETA: 1:43 - loss: 0.6172 - categorical_accuracy: 0.7995
 5632/60000 [=>............................] - ETA: 1:43 - loss: 0.6186 - categorical_accuracy: 0.7994
 5664/60000 [=>............................] - ETA: 1:43 - loss: 0.6167 - categorical_accuracy: 0.8000
 5696/60000 [=>............................] - ETA: 1:43 - loss: 0.6146 - categorical_accuracy: 0.8006
 5728/60000 [=>............................] - ETA: 1:43 - loss: 0.6141 - categorical_accuracy: 0.8012
 5760/60000 [=>............................] - ETA: 1:43 - loss: 0.6124 - categorical_accuracy: 0.8019
 5792/60000 [=>............................] - ETA: 1:43 - loss: 0.6115 - categorical_accuracy: 0.8025
 5824/60000 [=>............................] - ETA: 1:43 - loss: 0.6097 - categorical_accuracy: 0.8031
 5856/60000 [=>............................] - ETA: 1:43 - loss: 0.6098 - categorical_accuracy: 0.8033
 5888/60000 [=>............................] - ETA: 1:43 - loss: 0.6078 - categorical_accuracy: 0.8040
 5920/60000 [=>............................] - ETA: 1:43 - loss: 0.6053 - categorical_accuracy: 0.8051
 5952/60000 [=>............................] - ETA: 1:43 - loss: 0.6032 - categorical_accuracy: 0.8058
 5984/60000 [=>............................] - ETA: 1:43 - loss: 0.6007 - categorical_accuracy: 0.8067
 6016/60000 [==>...........................] - ETA: 1:43 - loss: 0.5993 - categorical_accuracy: 0.8072
 6048/60000 [==>...........................] - ETA: 1:42 - loss: 0.5973 - categorical_accuracy: 0.8077
 6080/60000 [==>...........................] - ETA: 1:42 - loss: 0.5979 - categorical_accuracy: 0.8081
 6112/60000 [==>...........................] - ETA: 1:42 - loss: 0.5980 - categorical_accuracy: 0.8084
 6144/60000 [==>...........................] - ETA: 1:42 - loss: 0.5957 - categorical_accuracy: 0.8089
 6176/60000 [==>...........................] - ETA: 1:42 - loss: 0.5932 - categorical_accuracy: 0.8097
 6208/60000 [==>...........................] - ETA: 1:42 - loss: 0.5906 - categorical_accuracy: 0.8107
 6240/60000 [==>...........................] - ETA: 1:42 - loss: 0.5889 - categorical_accuracy: 0.8112
 6272/60000 [==>...........................] - ETA: 1:42 - loss: 0.5867 - categorical_accuracy: 0.8119
 6304/60000 [==>...........................] - ETA: 1:42 - loss: 0.5847 - categorical_accuracy: 0.8123
 6336/60000 [==>...........................] - ETA: 1:42 - loss: 0.5822 - categorical_accuracy: 0.8133
 6368/60000 [==>...........................] - ETA: 1:42 - loss: 0.5812 - categorical_accuracy: 0.8139
 6400/60000 [==>...........................] - ETA: 1:42 - loss: 0.5820 - categorical_accuracy: 0.8139
 6432/60000 [==>...........................] - ETA: 1:42 - loss: 0.5793 - categorical_accuracy: 0.8148
 6464/60000 [==>...........................] - ETA: 1:42 - loss: 0.5791 - categorical_accuracy: 0.8150
 6496/60000 [==>...........................] - ETA: 1:42 - loss: 0.5769 - categorical_accuracy: 0.8157
 6528/60000 [==>...........................] - ETA: 1:41 - loss: 0.5752 - categorical_accuracy: 0.8163
 6560/60000 [==>...........................] - ETA: 1:41 - loss: 0.5731 - categorical_accuracy: 0.8171
 6592/60000 [==>...........................] - ETA: 1:41 - loss: 0.5712 - categorical_accuracy: 0.8177
 6624/60000 [==>...........................] - ETA: 1:41 - loss: 0.5699 - categorical_accuracy: 0.8182
 6656/60000 [==>...........................] - ETA: 1:41 - loss: 0.5674 - categorical_accuracy: 0.8191
 6688/60000 [==>...........................] - ETA: 1:41 - loss: 0.5662 - categorical_accuracy: 0.8195
 6720/60000 [==>...........................] - ETA: 1:41 - loss: 0.5644 - categorical_accuracy: 0.8201
 6752/60000 [==>...........................] - ETA: 1:41 - loss: 0.5620 - categorical_accuracy: 0.8209
 6784/60000 [==>...........................] - ETA: 1:41 - loss: 0.5599 - categorical_accuracy: 0.8215
 6816/60000 [==>...........................] - ETA: 1:41 - loss: 0.5591 - categorical_accuracy: 0.8216
 6848/60000 [==>...........................] - ETA: 1:41 - loss: 0.5585 - categorical_accuracy: 0.8217
 6880/60000 [==>...........................] - ETA: 1:41 - loss: 0.5573 - categorical_accuracy: 0.8221
 6912/60000 [==>...........................] - ETA: 1:41 - loss: 0.5561 - categorical_accuracy: 0.8225
 6944/60000 [==>...........................] - ETA: 1:41 - loss: 0.5545 - categorical_accuracy: 0.8229
 6976/60000 [==>...........................] - ETA: 1:41 - loss: 0.5526 - categorical_accuracy: 0.8234
 7008/60000 [==>...........................] - ETA: 1:40 - loss: 0.5508 - categorical_accuracy: 0.8242
 7040/60000 [==>...........................] - ETA: 1:40 - loss: 0.5495 - categorical_accuracy: 0.8246
 7072/60000 [==>...........................] - ETA: 1:40 - loss: 0.5475 - categorical_accuracy: 0.8251
 7104/60000 [==>...........................] - ETA: 1:40 - loss: 0.5454 - categorical_accuracy: 0.8257
 7136/60000 [==>...........................] - ETA: 1:40 - loss: 0.5434 - categorical_accuracy: 0.8264
 7168/60000 [==>...........................] - ETA: 1:40 - loss: 0.5424 - categorical_accuracy: 0.8269
 7200/60000 [==>...........................] - ETA: 1:40 - loss: 0.5415 - categorical_accuracy: 0.8271
 7232/60000 [==>...........................] - ETA: 1:40 - loss: 0.5405 - categorical_accuracy: 0.8273
 7264/60000 [==>...........................] - ETA: 1:40 - loss: 0.5397 - categorical_accuracy: 0.8276
 7296/60000 [==>...........................] - ETA: 1:40 - loss: 0.5391 - categorical_accuracy: 0.8281
 7328/60000 [==>...........................] - ETA: 1:40 - loss: 0.5392 - categorical_accuracy: 0.8279
 7360/60000 [==>...........................] - ETA: 1:40 - loss: 0.5395 - categorical_accuracy: 0.8285
 7392/60000 [==>...........................] - ETA: 1:40 - loss: 0.5377 - categorical_accuracy: 0.8291
 7424/60000 [==>...........................] - ETA: 1:40 - loss: 0.5360 - categorical_accuracy: 0.8296
 7456/60000 [==>...........................] - ETA: 1:40 - loss: 0.5343 - categorical_accuracy: 0.8302
 7488/60000 [==>...........................] - ETA: 1:40 - loss: 0.5333 - categorical_accuracy: 0.8305
 7520/60000 [==>...........................] - ETA: 1:40 - loss: 0.5317 - categorical_accuracy: 0.8311
 7552/60000 [==>...........................] - ETA: 1:39 - loss: 0.5299 - categorical_accuracy: 0.8318
 7584/60000 [==>...........................] - ETA: 1:39 - loss: 0.5284 - categorical_accuracy: 0.8324
 7616/60000 [==>...........................] - ETA: 1:39 - loss: 0.5265 - categorical_accuracy: 0.8331
 7648/60000 [==>...........................] - ETA: 1:39 - loss: 0.5248 - categorical_accuracy: 0.8337
 7680/60000 [==>...........................] - ETA: 1:39 - loss: 0.5232 - categorical_accuracy: 0.8341
 7712/60000 [==>...........................] - ETA: 1:39 - loss: 0.5223 - categorical_accuracy: 0.8345
 7744/60000 [==>...........................] - ETA: 1:39 - loss: 0.5215 - categorical_accuracy: 0.8350
 7776/60000 [==>...........................] - ETA: 1:39 - loss: 0.5207 - categorical_accuracy: 0.8351
 7808/60000 [==>...........................] - ETA: 1:39 - loss: 0.5191 - categorical_accuracy: 0.8358
 7840/60000 [==>...........................] - ETA: 1:39 - loss: 0.5189 - categorical_accuracy: 0.8358
 7872/60000 [==>...........................] - ETA: 1:39 - loss: 0.5175 - categorical_accuracy: 0.8361
 7904/60000 [==>...........................] - ETA: 1:39 - loss: 0.5157 - categorical_accuracy: 0.8367
 7936/60000 [==>...........................] - ETA: 1:39 - loss: 0.5145 - categorical_accuracy: 0.8369
 7968/60000 [==>...........................] - ETA: 1:38 - loss: 0.5139 - categorical_accuracy: 0.8372
 8000/60000 [===>..........................] - ETA: 1:38 - loss: 0.5130 - categorical_accuracy: 0.8376
 8032/60000 [===>..........................] - ETA: 1:38 - loss: 0.5118 - categorical_accuracy: 0.8381
 8064/60000 [===>..........................] - ETA: 1:38 - loss: 0.5103 - categorical_accuracy: 0.8387
 8096/60000 [===>..........................] - ETA: 1:38 - loss: 0.5085 - categorical_accuracy: 0.8393
 8128/60000 [===>..........................] - ETA: 1:38 - loss: 0.5069 - categorical_accuracy: 0.8398
 8160/60000 [===>..........................] - ETA: 1:38 - loss: 0.5061 - categorical_accuracy: 0.8400
 8192/60000 [===>..........................] - ETA: 1:38 - loss: 0.5048 - categorical_accuracy: 0.8405
 8224/60000 [===>..........................] - ETA: 1:38 - loss: 0.5040 - categorical_accuracy: 0.8407
 8256/60000 [===>..........................] - ETA: 1:38 - loss: 0.5036 - categorical_accuracy: 0.8408
 8288/60000 [===>..........................] - ETA: 1:38 - loss: 0.5026 - categorical_accuracy: 0.8411
 8320/60000 [===>..........................] - ETA: 1:38 - loss: 0.5010 - categorical_accuracy: 0.8416
 8352/60000 [===>..........................] - ETA: 1:38 - loss: 0.5013 - categorical_accuracy: 0.8417
 8384/60000 [===>..........................] - ETA: 1:38 - loss: 0.5014 - categorical_accuracy: 0.8418
 8416/60000 [===>..........................] - ETA: 1:37 - loss: 0.4999 - categorical_accuracy: 0.8423
 8448/60000 [===>..........................] - ETA: 1:37 - loss: 0.4990 - categorical_accuracy: 0.8426
 8480/60000 [===>..........................] - ETA: 1:37 - loss: 0.4976 - categorical_accuracy: 0.8430
 8512/60000 [===>..........................] - ETA: 1:37 - loss: 0.4968 - categorical_accuracy: 0.8432
 8544/60000 [===>..........................] - ETA: 1:37 - loss: 0.4951 - categorical_accuracy: 0.8438
 8576/60000 [===>..........................] - ETA: 1:37 - loss: 0.4945 - categorical_accuracy: 0.8441
 8608/60000 [===>..........................] - ETA: 1:37 - loss: 0.4930 - categorical_accuracy: 0.8446
 8640/60000 [===>..........................] - ETA: 1:37 - loss: 0.4923 - categorical_accuracy: 0.8447
 8672/60000 [===>..........................] - ETA: 1:37 - loss: 0.4919 - categorical_accuracy: 0.8448
 8704/60000 [===>..........................] - ETA: 1:37 - loss: 0.4904 - categorical_accuracy: 0.8454
 8736/60000 [===>..........................] - ETA: 1:37 - loss: 0.4891 - categorical_accuracy: 0.8457
 8768/60000 [===>..........................] - ETA: 1:37 - loss: 0.4874 - categorical_accuracy: 0.8463
 8800/60000 [===>..........................] - ETA: 1:37 - loss: 0.4861 - categorical_accuracy: 0.8467
 8832/60000 [===>..........................] - ETA: 1:37 - loss: 0.4850 - categorical_accuracy: 0.8471
 8864/60000 [===>..........................] - ETA: 1:37 - loss: 0.4837 - categorical_accuracy: 0.8476
 8896/60000 [===>..........................] - ETA: 1:37 - loss: 0.4823 - categorical_accuracy: 0.8480
 8928/60000 [===>..........................] - ETA: 1:37 - loss: 0.4819 - categorical_accuracy: 0.8482
 8960/60000 [===>..........................] - ETA: 1:37 - loss: 0.4806 - categorical_accuracy: 0.8487
 8992/60000 [===>..........................] - ETA: 1:36 - loss: 0.4796 - categorical_accuracy: 0.8488
 9024/60000 [===>..........................] - ETA: 1:36 - loss: 0.4784 - categorical_accuracy: 0.8491
 9056/60000 [===>..........................] - ETA: 1:36 - loss: 0.4770 - categorical_accuracy: 0.8495
 9088/60000 [===>..........................] - ETA: 1:36 - loss: 0.4756 - categorical_accuracy: 0.8499
 9120/60000 [===>..........................] - ETA: 1:36 - loss: 0.4745 - categorical_accuracy: 0.8501
 9152/60000 [===>..........................] - ETA: 1:36 - loss: 0.4735 - categorical_accuracy: 0.8502
 9184/60000 [===>..........................] - ETA: 1:36 - loss: 0.4732 - categorical_accuracy: 0.8502
 9216/60000 [===>..........................] - ETA: 1:36 - loss: 0.4725 - categorical_accuracy: 0.8504
 9248/60000 [===>..........................] - ETA: 1:36 - loss: 0.4740 - categorical_accuracy: 0.8501
 9280/60000 [===>..........................] - ETA: 1:36 - loss: 0.4729 - categorical_accuracy: 0.8504
 9312/60000 [===>..........................] - ETA: 1:36 - loss: 0.4725 - categorical_accuracy: 0.8505
 9344/60000 [===>..........................] - ETA: 1:36 - loss: 0.4712 - categorical_accuracy: 0.8510
 9376/60000 [===>..........................] - ETA: 1:36 - loss: 0.4706 - categorical_accuracy: 0.8513
 9408/60000 [===>..........................] - ETA: 1:36 - loss: 0.4698 - categorical_accuracy: 0.8515
 9440/60000 [===>..........................] - ETA: 1:36 - loss: 0.4688 - categorical_accuracy: 0.8518
 9472/60000 [===>..........................] - ETA: 1:36 - loss: 0.4686 - categorical_accuracy: 0.8518
 9504/60000 [===>..........................] - ETA: 1:36 - loss: 0.4681 - categorical_accuracy: 0.8521
 9536/60000 [===>..........................] - ETA: 1:35 - loss: 0.4667 - categorical_accuracy: 0.8526
 9568/60000 [===>..........................] - ETA: 1:35 - loss: 0.4655 - categorical_accuracy: 0.8529
 9600/60000 [===>..........................] - ETA: 1:35 - loss: 0.4650 - categorical_accuracy: 0.8532
 9632/60000 [===>..........................] - ETA: 1:35 - loss: 0.4646 - categorical_accuracy: 0.8535
 9664/60000 [===>..........................] - ETA: 1:35 - loss: 0.4642 - categorical_accuracy: 0.8537
 9696/60000 [===>..........................] - ETA: 1:35 - loss: 0.4630 - categorical_accuracy: 0.8541
 9728/60000 [===>..........................] - ETA: 1:35 - loss: 0.4623 - categorical_accuracy: 0.8542
 9760/60000 [===>..........................] - ETA: 1:35 - loss: 0.4614 - categorical_accuracy: 0.8544
 9792/60000 [===>..........................] - ETA: 1:35 - loss: 0.4608 - categorical_accuracy: 0.8545
 9824/60000 [===>..........................] - ETA: 1:35 - loss: 0.4603 - categorical_accuracy: 0.8546
 9856/60000 [===>..........................] - ETA: 1:35 - loss: 0.4594 - categorical_accuracy: 0.8548
 9888/60000 [===>..........................] - ETA: 1:35 - loss: 0.4592 - categorical_accuracy: 0.8551
 9920/60000 [===>..........................] - ETA: 1:35 - loss: 0.4587 - categorical_accuracy: 0.8553
 9952/60000 [===>..........................] - ETA: 1:35 - loss: 0.4578 - categorical_accuracy: 0.8556
 9984/60000 [===>..........................] - ETA: 1:35 - loss: 0.4570 - categorical_accuracy: 0.8559
10016/60000 [====>.........................] - ETA: 1:34 - loss: 0.4564 - categorical_accuracy: 0.8560
10048/60000 [====>.........................] - ETA: 1:34 - loss: 0.4555 - categorical_accuracy: 0.8563
10080/60000 [====>.........................] - ETA: 1:34 - loss: 0.4553 - categorical_accuracy: 0.8564
10112/60000 [====>.........................] - ETA: 1:34 - loss: 0.4544 - categorical_accuracy: 0.8566
10144/60000 [====>.........................] - ETA: 1:34 - loss: 0.4544 - categorical_accuracy: 0.8566
10176/60000 [====>.........................] - ETA: 1:34 - loss: 0.4541 - categorical_accuracy: 0.8566
10208/60000 [====>.........................] - ETA: 1:34 - loss: 0.4536 - categorical_accuracy: 0.8568
10240/60000 [====>.........................] - ETA: 1:34 - loss: 0.4529 - categorical_accuracy: 0.8570
10272/60000 [====>.........................] - ETA: 1:34 - loss: 0.4519 - categorical_accuracy: 0.8575
10304/60000 [====>.........................] - ETA: 1:34 - loss: 0.4509 - categorical_accuracy: 0.8578
10336/60000 [====>.........................] - ETA: 1:34 - loss: 0.4502 - categorical_accuracy: 0.8580
10368/60000 [====>.........................] - ETA: 1:34 - loss: 0.4490 - categorical_accuracy: 0.8584
10400/60000 [====>.........................] - ETA: 1:34 - loss: 0.4479 - categorical_accuracy: 0.8588
10432/60000 [====>.........................] - ETA: 1:34 - loss: 0.4470 - categorical_accuracy: 0.8591
10464/60000 [====>.........................] - ETA: 1:33 - loss: 0.4462 - categorical_accuracy: 0.8592
10496/60000 [====>.........................] - ETA: 1:33 - loss: 0.4458 - categorical_accuracy: 0.8593
10528/60000 [====>.........................] - ETA: 1:33 - loss: 0.4453 - categorical_accuracy: 0.8595
10560/60000 [====>.........................] - ETA: 1:33 - loss: 0.4455 - categorical_accuracy: 0.8597
10592/60000 [====>.........................] - ETA: 1:33 - loss: 0.4445 - categorical_accuracy: 0.8600
10624/60000 [====>.........................] - ETA: 1:33 - loss: 0.4434 - categorical_accuracy: 0.8604
10656/60000 [====>.........................] - ETA: 1:33 - loss: 0.4421 - categorical_accuracy: 0.8608
10688/60000 [====>.........................] - ETA: 1:33 - loss: 0.4411 - categorical_accuracy: 0.8612
10720/60000 [====>.........................] - ETA: 1:33 - loss: 0.4399 - categorical_accuracy: 0.8616
10752/60000 [====>.........................] - ETA: 1:33 - loss: 0.4390 - categorical_accuracy: 0.8618
10784/60000 [====>.........................] - ETA: 1:33 - loss: 0.4380 - categorical_accuracy: 0.8621
10816/60000 [====>.........................] - ETA: 1:33 - loss: 0.4369 - categorical_accuracy: 0.8624
10848/60000 [====>.........................] - ETA: 1:33 - loss: 0.4376 - categorical_accuracy: 0.8624
10880/60000 [====>.........................] - ETA: 1:33 - loss: 0.4368 - categorical_accuracy: 0.8626
10912/60000 [====>.........................] - ETA: 1:33 - loss: 0.4360 - categorical_accuracy: 0.8629
10944/60000 [====>.........................] - ETA: 1:33 - loss: 0.4364 - categorical_accuracy: 0.8628
10976/60000 [====>.........................] - ETA: 1:32 - loss: 0.4354 - categorical_accuracy: 0.8632
11008/60000 [====>.........................] - ETA: 1:32 - loss: 0.4347 - categorical_accuracy: 0.8634
11040/60000 [====>.........................] - ETA: 1:32 - loss: 0.4343 - categorical_accuracy: 0.8636
11072/60000 [====>.........................] - ETA: 1:32 - loss: 0.4338 - categorical_accuracy: 0.8635
11104/60000 [====>.........................] - ETA: 1:32 - loss: 0.4327 - categorical_accuracy: 0.8638
11136/60000 [====>.........................] - ETA: 1:32 - loss: 0.4319 - categorical_accuracy: 0.8640
11168/60000 [====>.........................] - ETA: 1:32 - loss: 0.4314 - categorical_accuracy: 0.8641
11200/60000 [====>.........................] - ETA: 1:32 - loss: 0.4303 - categorical_accuracy: 0.8645
11232/60000 [====>.........................] - ETA: 1:32 - loss: 0.4296 - categorical_accuracy: 0.8646
11264/60000 [====>.........................] - ETA: 1:32 - loss: 0.4294 - categorical_accuracy: 0.8647
11296/60000 [====>.........................] - ETA: 1:32 - loss: 0.4286 - categorical_accuracy: 0.8649
11328/60000 [====>.........................] - ETA: 1:32 - loss: 0.4277 - categorical_accuracy: 0.8652
11360/60000 [====>.........................] - ETA: 1:32 - loss: 0.4273 - categorical_accuracy: 0.8653
11392/60000 [====>.........................] - ETA: 1:32 - loss: 0.4264 - categorical_accuracy: 0.8656
11424/60000 [====>.........................] - ETA: 1:32 - loss: 0.4263 - categorical_accuracy: 0.8658
11456/60000 [====>.........................] - ETA: 1:32 - loss: 0.4255 - categorical_accuracy: 0.8661
11488/60000 [====>.........................] - ETA: 1:31 - loss: 0.4245 - categorical_accuracy: 0.8663
11520/60000 [====>.........................] - ETA: 1:31 - loss: 0.4243 - categorical_accuracy: 0.8665
11552/60000 [====>.........................] - ETA: 1:31 - loss: 0.4239 - categorical_accuracy: 0.8666
11584/60000 [====>.........................] - ETA: 1:31 - loss: 0.4230 - categorical_accuracy: 0.8670
11616/60000 [====>.........................] - ETA: 1:31 - loss: 0.4222 - categorical_accuracy: 0.8672
11648/60000 [====>.........................] - ETA: 1:31 - loss: 0.4214 - categorical_accuracy: 0.8673
11680/60000 [====>.........................] - ETA: 1:31 - loss: 0.4207 - categorical_accuracy: 0.8675
11712/60000 [====>.........................] - ETA: 1:31 - loss: 0.4199 - categorical_accuracy: 0.8677
11744/60000 [====>.........................] - ETA: 1:31 - loss: 0.4189 - categorical_accuracy: 0.8680
11776/60000 [====>.........................] - ETA: 1:31 - loss: 0.4182 - categorical_accuracy: 0.8683
11808/60000 [====>.........................] - ETA: 1:31 - loss: 0.4175 - categorical_accuracy: 0.8684
11840/60000 [====>.........................] - ETA: 1:31 - loss: 0.4172 - categorical_accuracy: 0.8686
11872/60000 [====>.........................] - ETA: 1:31 - loss: 0.4169 - categorical_accuracy: 0.8685
11904/60000 [====>.........................] - ETA: 1:31 - loss: 0.4163 - categorical_accuracy: 0.8687
11936/60000 [====>.........................] - ETA: 1:31 - loss: 0.4161 - categorical_accuracy: 0.8687
11968/60000 [====>.........................] - ETA: 1:30 - loss: 0.4154 - categorical_accuracy: 0.8690
12000/60000 [=====>........................] - ETA: 1:30 - loss: 0.4147 - categorical_accuracy: 0.8692
12032/60000 [=====>........................] - ETA: 1:30 - loss: 0.4142 - categorical_accuracy: 0.8694
12064/60000 [=====>........................] - ETA: 1:30 - loss: 0.4143 - categorical_accuracy: 0.8694
12096/60000 [=====>........................] - ETA: 1:30 - loss: 0.4134 - categorical_accuracy: 0.8697
12128/60000 [=====>........................] - ETA: 1:30 - loss: 0.4127 - categorical_accuracy: 0.8699
12160/60000 [=====>........................] - ETA: 1:30 - loss: 0.4121 - categorical_accuracy: 0.8700
12192/60000 [=====>........................] - ETA: 1:30 - loss: 0.4111 - categorical_accuracy: 0.8703
12224/60000 [=====>........................] - ETA: 1:30 - loss: 0.4105 - categorical_accuracy: 0.8705
12256/60000 [=====>........................] - ETA: 1:30 - loss: 0.4101 - categorical_accuracy: 0.8706
12288/60000 [=====>........................] - ETA: 1:30 - loss: 0.4098 - categorical_accuracy: 0.8707
12320/60000 [=====>........................] - ETA: 1:30 - loss: 0.4092 - categorical_accuracy: 0.8709
12352/60000 [=====>........................] - ETA: 1:30 - loss: 0.4087 - categorical_accuracy: 0.8710
12384/60000 [=====>........................] - ETA: 1:30 - loss: 0.4082 - categorical_accuracy: 0.8713
12416/60000 [=====>........................] - ETA: 1:30 - loss: 0.4078 - categorical_accuracy: 0.8713
12448/60000 [=====>........................] - ETA: 1:30 - loss: 0.4070 - categorical_accuracy: 0.8715
12480/60000 [=====>........................] - ETA: 1:29 - loss: 0.4062 - categorical_accuracy: 0.8718
12512/60000 [=====>........................] - ETA: 1:29 - loss: 0.4058 - categorical_accuracy: 0.8719
12544/60000 [=====>........................] - ETA: 1:29 - loss: 0.4049 - categorical_accuracy: 0.8722
12576/60000 [=====>........................] - ETA: 1:29 - loss: 0.4042 - categorical_accuracy: 0.8724
12608/60000 [=====>........................] - ETA: 1:29 - loss: 0.4040 - categorical_accuracy: 0.8725
12640/60000 [=====>........................] - ETA: 1:29 - loss: 0.4031 - categorical_accuracy: 0.8728
12672/60000 [=====>........................] - ETA: 1:29 - loss: 0.4028 - categorical_accuracy: 0.8729
12704/60000 [=====>........................] - ETA: 1:29 - loss: 0.4024 - categorical_accuracy: 0.8731
12736/60000 [=====>........................] - ETA: 1:29 - loss: 0.4017 - categorical_accuracy: 0.8734
12768/60000 [=====>........................] - ETA: 1:29 - loss: 0.4014 - categorical_accuracy: 0.8735
12800/60000 [=====>........................] - ETA: 1:29 - loss: 0.4010 - categorical_accuracy: 0.8737
12832/60000 [=====>........................] - ETA: 1:29 - loss: 0.4008 - categorical_accuracy: 0.8738
12864/60000 [=====>........................] - ETA: 1:29 - loss: 0.4002 - categorical_accuracy: 0.8738
12896/60000 [=====>........................] - ETA: 1:29 - loss: 0.3999 - categorical_accuracy: 0.8738
12928/60000 [=====>........................] - ETA: 1:29 - loss: 0.3997 - categorical_accuracy: 0.8740
12960/60000 [=====>........................] - ETA: 1:29 - loss: 0.3997 - categorical_accuracy: 0.8741
12992/60000 [=====>........................] - ETA: 1:28 - loss: 0.3988 - categorical_accuracy: 0.8744
13024/60000 [=====>........................] - ETA: 1:28 - loss: 0.3979 - categorical_accuracy: 0.8747
13056/60000 [=====>........................] - ETA: 1:28 - loss: 0.3973 - categorical_accuracy: 0.8749
13088/60000 [=====>........................] - ETA: 1:28 - loss: 0.3966 - categorical_accuracy: 0.8752
13120/60000 [=====>........................] - ETA: 1:28 - loss: 0.3960 - categorical_accuracy: 0.8753
13152/60000 [=====>........................] - ETA: 1:28 - loss: 0.3954 - categorical_accuracy: 0.8755
13184/60000 [=====>........................] - ETA: 1:28 - loss: 0.3952 - categorical_accuracy: 0.8756
13216/60000 [=====>........................] - ETA: 1:28 - loss: 0.3951 - categorical_accuracy: 0.8757
13248/60000 [=====>........................] - ETA: 1:28 - loss: 0.3944 - categorical_accuracy: 0.8759
13280/60000 [=====>........................] - ETA: 1:28 - loss: 0.3939 - categorical_accuracy: 0.8761
13312/60000 [=====>........................] - ETA: 1:28 - loss: 0.3936 - categorical_accuracy: 0.8761
13344/60000 [=====>........................] - ETA: 1:28 - loss: 0.3929 - categorical_accuracy: 0.8763
13376/60000 [=====>........................] - ETA: 1:28 - loss: 0.3921 - categorical_accuracy: 0.8765
13408/60000 [=====>........................] - ETA: 1:28 - loss: 0.3914 - categorical_accuracy: 0.8766
13440/60000 [=====>........................] - ETA: 1:28 - loss: 0.3906 - categorical_accuracy: 0.8769
13472/60000 [=====>........................] - ETA: 1:27 - loss: 0.3901 - categorical_accuracy: 0.8769
13504/60000 [=====>........................] - ETA: 1:27 - loss: 0.3893 - categorical_accuracy: 0.8771
13536/60000 [=====>........................] - ETA: 1:27 - loss: 0.3888 - categorical_accuracy: 0.8771
13568/60000 [=====>........................] - ETA: 1:27 - loss: 0.3882 - categorical_accuracy: 0.8773
13600/60000 [=====>........................] - ETA: 1:27 - loss: 0.3875 - categorical_accuracy: 0.8775
13632/60000 [=====>........................] - ETA: 1:27 - loss: 0.3867 - categorical_accuracy: 0.8778
13664/60000 [=====>........................] - ETA: 1:27 - loss: 0.3859 - categorical_accuracy: 0.8780
13696/60000 [=====>........................] - ETA: 1:27 - loss: 0.3853 - categorical_accuracy: 0.8781
13728/60000 [=====>........................] - ETA: 1:27 - loss: 0.3846 - categorical_accuracy: 0.8784
13760/60000 [=====>........................] - ETA: 1:27 - loss: 0.3841 - categorical_accuracy: 0.8785
13792/60000 [=====>........................] - ETA: 1:27 - loss: 0.3832 - categorical_accuracy: 0.8788
13824/60000 [=====>........................] - ETA: 1:27 - loss: 0.3824 - categorical_accuracy: 0.8791
13856/60000 [=====>........................] - ETA: 1:27 - loss: 0.3817 - categorical_accuracy: 0.8793
13888/60000 [=====>........................] - ETA: 1:27 - loss: 0.3812 - categorical_accuracy: 0.8793
13920/60000 [=====>........................] - ETA: 1:27 - loss: 0.3804 - categorical_accuracy: 0.8796
13952/60000 [=====>........................] - ETA: 1:27 - loss: 0.3796 - categorical_accuracy: 0.8799
13984/60000 [=====>........................] - ETA: 1:27 - loss: 0.3788 - categorical_accuracy: 0.8801
14016/60000 [======>.......................] - ETA: 1:26 - loss: 0.3783 - categorical_accuracy: 0.8803
14048/60000 [======>.......................] - ETA: 1:26 - loss: 0.3783 - categorical_accuracy: 0.8803
14080/60000 [======>.......................] - ETA: 1:26 - loss: 0.3779 - categorical_accuracy: 0.8804
14112/60000 [======>.......................] - ETA: 1:26 - loss: 0.3783 - categorical_accuracy: 0.8804
14144/60000 [======>.......................] - ETA: 1:26 - loss: 0.3781 - categorical_accuracy: 0.8806
14176/60000 [======>.......................] - ETA: 1:26 - loss: 0.3774 - categorical_accuracy: 0.8808
14208/60000 [======>.......................] - ETA: 1:26 - loss: 0.3767 - categorical_accuracy: 0.8810
14240/60000 [======>.......................] - ETA: 1:26 - loss: 0.3762 - categorical_accuracy: 0.8811
14272/60000 [======>.......................] - ETA: 1:26 - loss: 0.3762 - categorical_accuracy: 0.8812
14304/60000 [======>.......................] - ETA: 1:26 - loss: 0.3759 - categorical_accuracy: 0.8814
14336/60000 [======>.......................] - ETA: 1:26 - loss: 0.3752 - categorical_accuracy: 0.8816
14368/60000 [======>.......................] - ETA: 1:26 - loss: 0.3752 - categorical_accuracy: 0.8816
14400/60000 [======>.......................] - ETA: 1:26 - loss: 0.3749 - categorical_accuracy: 0.8817
14432/60000 [======>.......................] - ETA: 1:26 - loss: 0.3746 - categorical_accuracy: 0.8818
14464/60000 [======>.......................] - ETA: 1:26 - loss: 0.3742 - categorical_accuracy: 0.8819
14496/60000 [======>.......................] - ETA: 1:26 - loss: 0.3738 - categorical_accuracy: 0.8820
14528/60000 [======>.......................] - ETA: 1:25 - loss: 0.3737 - categorical_accuracy: 0.8819
14560/60000 [======>.......................] - ETA: 1:25 - loss: 0.3736 - categorical_accuracy: 0.8819
14592/60000 [======>.......................] - ETA: 1:25 - loss: 0.3731 - categorical_accuracy: 0.8821
14624/60000 [======>.......................] - ETA: 1:25 - loss: 0.3726 - categorical_accuracy: 0.8822
14656/60000 [======>.......................] - ETA: 1:25 - loss: 0.3720 - categorical_accuracy: 0.8823
14688/60000 [======>.......................] - ETA: 1:25 - loss: 0.3717 - categorical_accuracy: 0.8824
14720/60000 [======>.......................] - ETA: 1:25 - loss: 0.3717 - categorical_accuracy: 0.8825
14752/60000 [======>.......................] - ETA: 1:25 - loss: 0.3714 - categorical_accuracy: 0.8827
14784/60000 [======>.......................] - ETA: 1:25 - loss: 0.3709 - categorical_accuracy: 0.8828
14816/60000 [======>.......................] - ETA: 1:25 - loss: 0.3704 - categorical_accuracy: 0.8830
14848/60000 [======>.......................] - ETA: 1:25 - loss: 0.3696 - categorical_accuracy: 0.8832
14880/60000 [======>.......................] - ETA: 1:25 - loss: 0.3699 - categorical_accuracy: 0.8832
14912/60000 [======>.......................] - ETA: 1:25 - loss: 0.3693 - categorical_accuracy: 0.8834
14944/60000 [======>.......................] - ETA: 1:25 - loss: 0.3688 - categorical_accuracy: 0.8835
14976/60000 [======>.......................] - ETA: 1:25 - loss: 0.3684 - categorical_accuracy: 0.8836
15008/60000 [======>.......................] - ETA: 1:25 - loss: 0.3680 - categorical_accuracy: 0.8837
15040/60000 [======>.......................] - ETA: 1:24 - loss: 0.3675 - categorical_accuracy: 0.8838
15072/60000 [======>.......................] - ETA: 1:24 - loss: 0.3675 - categorical_accuracy: 0.8838
15104/60000 [======>.......................] - ETA: 1:24 - loss: 0.3674 - categorical_accuracy: 0.8839
15136/60000 [======>.......................] - ETA: 1:24 - loss: 0.3669 - categorical_accuracy: 0.8841
15168/60000 [======>.......................] - ETA: 1:24 - loss: 0.3663 - categorical_accuracy: 0.8842
15200/60000 [======>.......................] - ETA: 1:24 - loss: 0.3661 - categorical_accuracy: 0.8844
15232/60000 [======>.......................] - ETA: 1:24 - loss: 0.3656 - categorical_accuracy: 0.8845
15264/60000 [======>.......................] - ETA: 1:24 - loss: 0.3652 - categorical_accuracy: 0.8846
15296/60000 [======>.......................] - ETA: 1:24 - loss: 0.3648 - categorical_accuracy: 0.8848
15328/60000 [======>.......................] - ETA: 1:24 - loss: 0.3646 - categorical_accuracy: 0.8848
15360/60000 [======>.......................] - ETA: 1:24 - loss: 0.3640 - categorical_accuracy: 0.8850
15392/60000 [======>.......................] - ETA: 1:24 - loss: 0.3636 - categorical_accuracy: 0.8852
15424/60000 [======>.......................] - ETA: 1:24 - loss: 0.3634 - categorical_accuracy: 0.8852
15456/60000 [======>.......................] - ETA: 1:24 - loss: 0.3627 - categorical_accuracy: 0.8855
15488/60000 [======>.......................] - ETA: 1:24 - loss: 0.3626 - categorical_accuracy: 0.8855
15520/60000 [======>.......................] - ETA: 1:24 - loss: 0.3624 - categorical_accuracy: 0.8856
15552/60000 [======>.......................] - ETA: 1:23 - loss: 0.3619 - categorical_accuracy: 0.8857
15584/60000 [======>.......................] - ETA: 1:23 - loss: 0.3613 - categorical_accuracy: 0.8860
15616/60000 [======>.......................] - ETA: 1:23 - loss: 0.3609 - categorical_accuracy: 0.8861
15648/60000 [======>.......................] - ETA: 1:23 - loss: 0.3611 - categorical_accuracy: 0.8862
15680/60000 [======>.......................] - ETA: 1:23 - loss: 0.3609 - categorical_accuracy: 0.8862
15712/60000 [======>.......................] - ETA: 1:23 - loss: 0.3606 - categorical_accuracy: 0.8863
15744/60000 [======>.......................] - ETA: 1:23 - loss: 0.3605 - categorical_accuracy: 0.8863
15776/60000 [======>.......................] - ETA: 1:23 - loss: 0.3598 - categorical_accuracy: 0.8865
15808/60000 [======>.......................] - ETA: 1:23 - loss: 0.3595 - categorical_accuracy: 0.8866
15840/60000 [======>.......................] - ETA: 1:23 - loss: 0.3593 - categorical_accuracy: 0.8867
15872/60000 [======>.......................] - ETA: 1:23 - loss: 0.3590 - categorical_accuracy: 0.8868
15904/60000 [======>.......................] - ETA: 1:23 - loss: 0.3592 - categorical_accuracy: 0.8868
15936/60000 [======>.......................] - ETA: 1:23 - loss: 0.3590 - categorical_accuracy: 0.8869
15968/60000 [======>.......................] - ETA: 1:23 - loss: 0.3589 - categorical_accuracy: 0.8870
16000/60000 [=======>......................] - ETA: 1:23 - loss: 0.3588 - categorical_accuracy: 0.8870
16032/60000 [=======>......................] - ETA: 1:22 - loss: 0.3583 - categorical_accuracy: 0.8872
16064/60000 [=======>......................] - ETA: 1:22 - loss: 0.3579 - categorical_accuracy: 0.8873
16096/60000 [=======>......................] - ETA: 1:22 - loss: 0.3577 - categorical_accuracy: 0.8875
16128/60000 [=======>......................] - ETA: 1:22 - loss: 0.3572 - categorical_accuracy: 0.8876
16160/60000 [=======>......................] - ETA: 1:22 - loss: 0.3567 - categorical_accuracy: 0.8878
16192/60000 [=======>......................] - ETA: 1:22 - loss: 0.3561 - categorical_accuracy: 0.8880
16224/60000 [=======>......................] - ETA: 1:22 - loss: 0.3560 - categorical_accuracy: 0.8880
16256/60000 [=======>......................] - ETA: 1:22 - loss: 0.3554 - categorical_accuracy: 0.8882
16288/60000 [=======>......................] - ETA: 1:22 - loss: 0.3550 - categorical_accuracy: 0.8883
16320/60000 [=======>......................] - ETA: 1:22 - loss: 0.3546 - categorical_accuracy: 0.8885
16352/60000 [=======>......................] - ETA: 1:22 - loss: 0.3545 - categorical_accuracy: 0.8885
16384/60000 [=======>......................] - ETA: 1:22 - loss: 0.3542 - categorical_accuracy: 0.8887
16416/60000 [=======>......................] - ETA: 1:22 - loss: 0.3541 - categorical_accuracy: 0.8887
16448/60000 [=======>......................] - ETA: 1:22 - loss: 0.3536 - categorical_accuracy: 0.8889
16480/60000 [=======>......................] - ETA: 1:22 - loss: 0.3533 - categorical_accuracy: 0.8890
16512/60000 [=======>......................] - ETA: 1:22 - loss: 0.3528 - categorical_accuracy: 0.8891
16544/60000 [=======>......................] - ETA: 1:21 - loss: 0.3523 - categorical_accuracy: 0.8893
16576/60000 [=======>......................] - ETA: 1:21 - loss: 0.3517 - categorical_accuracy: 0.8895
16608/60000 [=======>......................] - ETA: 1:21 - loss: 0.3512 - categorical_accuracy: 0.8896
16640/60000 [=======>......................] - ETA: 1:21 - loss: 0.3513 - categorical_accuracy: 0.8896
16672/60000 [=======>......................] - ETA: 1:21 - loss: 0.3509 - categorical_accuracy: 0.8898
16704/60000 [=======>......................] - ETA: 1:21 - loss: 0.3503 - categorical_accuracy: 0.8900
16736/60000 [=======>......................] - ETA: 1:21 - loss: 0.3505 - categorical_accuracy: 0.8900
16768/60000 [=======>......................] - ETA: 1:21 - loss: 0.3506 - categorical_accuracy: 0.8899
16800/60000 [=======>......................] - ETA: 1:21 - loss: 0.3502 - categorical_accuracy: 0.8901
16832/60000 [=======>......................] - ETA: 1:21 - loss: 0.3496 - categorical_accuracy: 0.8902
16864/60000 [=======>......................] - ETA: 1:21 - loss: 0.3496 - categorical_accuracy: 0.8903
16896/60000 [=======>......................] - ETA: 1:21 - loss: 0.3495 - categorical_accuracy: 0.8902
16928/60000 [=======>......................] - ETA: 1:21 - loss: 0.3490 - categorical_accuracy: 0.8904
16960/60000 [=======>......................] - ETA: 1:21 - loss: 0.3487 - categorical_accuracy: 0.8904
16992/60000 [=======>......................] - ETA: 1:21 - loss: 0.3481 - categorical_accuracy: 0.8906
17024/60000 [=======>......................] - ETA: 1:20 - loss: 0.3477 - categorical_accuracy: 0.8907
17056/60000 [=======>......................] - ETA: 1:20 - loss: 0.3472 - categorical_accuracy: 0.8909
17088/60000 [=======>......................] - ETA: 1:20 - loss: 0.3468 - categorical_accuracy: 0.8910
17120/60000 [=======>......................] - ETA: 1:20 - loss: 0.3465 - categorical_accuracy: 0.8911
17152/60000 [=======>......................] - ETA: 1:20 - loss: 0.3465 - categorical_accuracy: 0.8912
17184/60000 [=======>......................] - ETA: 1:20 - loss: 0.3462 - categorical_accuracy: 0.8912
17216/60000 [=======>......................] - ETA: 1:20 - loss: 0.3457 - categorical_accuracy: 0.8914
17248/60000 [=======>......................] - ETA: 1:20 - loss: 0.3452 - categorical_accuracy: 0.8915
17280/60000 [=======>......................] - ETA: 1:20 - loss: 0.3449 - categorical_accuracy: 0.8916
17312/60000 [=======>......................] - ETA: 1:20 - loss: 0.3444 - categorical_accuracy: 0.8918
17344/60000 [=======>......................] - ETA: 1:20 - loss: 0.3441 - categorical_accuracy: 0.8919
17376/60000 [=======>......................] - ETA: 1:20 - loss: 0.3438 - categorical_accuracy: 0.8920
17408/60000 [=======>......................] - ETA: 1:20 - loss: 0.3434 - categorical_accuracy: 0.8921
17440/60000 [=======>......................] - ETA: 1:20 - loss: 0.3428 - categorical_accuracy: 0.8923
17472/60000 [=======>......................] - ETA: 1:20 - loss: 0.3424 - categorical_accuracy: 0.8925
17504/60000 [=======>......................] - ETA: 1:20 - loss: 0.3422 - categorical_accuracy: 0.8925
17536/60000 [=======>......................] - ETA: 1:19 - loss: 0.3418 - categorical_accuracy: 0.8927
17568/60000 [=======>......................] - ETA: 1:19 - loss: 0.3413 - categorical_accuracy: 0.8928
17600/60000 [=======>......................] - ETA: 1:19 - loss: 0.3407 - categorical_accuracy: 0.8930
17632/60000 [=======>......................] - ETA: 1:19 - loss: 0.3404 - categorical_accuracy: 0.8931
17664/60000 [=======>......................] - ETA: 1:19 - loss: 0.3402 - categorical_accuracy: 0.8931
17696/60000 [=======>......................] - ETA: 1:19 - loss: 0.3397 - categorical_accuracy: 0.8933
17728/60000 [=======>......................] - ETA: 1:19 - loss: 0.3395 - categorical_accuracy: 0.8933
17760/60000 [=======>......................] - ETA: 1:19 - loss: 0.3396 - categorical_accuracy: 0.8934
17792/60000 [=======>......................] - ETA: 1:19 - loss: 0.3392 - categorical_accuracy: 0.8934
17824/60000 [=======>......................] - ETA: 1:19 - loss: 0.3391 - categorical_accuracy: 0.8936
17856/60000 [=======>......................] - ETA: 1:19 - loss: 0.3394 - categorical_accuracy: 0.8935
17888/60000 [=======>......................] - ETA: 1:19 - loss: 0.3391 - categorical_accuracy: 0.8937
17920/60000 [=======>......................] - ETA: 1:19 - loss: 0.3387 - categorical_accuracy: 0.8938
17952/60000 [=======>......................] - ETA: 1:19 - loss: 0.3384 - categorical_accuracy: 0.8939
17984/60000 [=======>......................] - ETA: 1:19 - loss: 0.3382 - categorical_accuracy: 0.8940
18016/60000 [========>.....................] - ETA: 1:19 - loss: 0.3379 - categorical_accuracy: 0.8941
18048/60000 [========>.....................] - ETA: 1:19 - loss: 0.3375 - categorical_accuracy: 0.8942
18080/60000 [========>.....................] - ETA: 1:18 - loss: 0.3370 - categorical_accuracy: 0.8944
18112/60000 [========>.....................] - ETA: 1:18 - loss: 0.3372 - categorical_accuracy: 0.8945
18144/60000 [========>.....................] - ETA: 1:18 - loss: 0.3367 - categorical_accuracy: 0.8947
18176/60000 [========>.....................] - ETA: 1:18 - loss: 0.3366 - categorical_accuracy: 0.8947
18208/60000 [========>.....................] - ETA: 1:18 - loss: 0.3361 - categorical_accuracy: 0.8949
18240/60000 [========>.....................] - ETA: 1:18 - loss: 0.3358 - categorical_accuracy: 0.8950
18272/60000 [========>.....................] - ETA: 1:18 - loss: 0.3355 - categorical_accuracy: 0.8950
18304/60000 [========>.....................] - ETA: 1:18 - loss: 0.3352 - categorical_accuracy: 0.8950
18336/60000 [========>.....................] - ETA: 1:18 - loss: 0.3348 - categorical_accuracy: 0.8951
18368/60000 [========>.....................] - ETA: 1:18 - loss: 0.3344 - categorical_accuracy: 0.8953
18400/60000 [========>.....................] - ETA: 1:18 - loss: 0.3339 - categorical_accuracy: 0.8954
18432/60000 [========>.....................] - ETA: 1:18 - loss: 0.3334 - categorical_accuracy: 0.8956
18464/60000 [========>.....................] - ETA: 1:18 - loss: 0.3332 - categorical_accuracy: 0.8957
18496/60000 [========>.....................] - ETA: 1:18 - loss: 0.3331 - categorical_accuracy: 0.8958
18528/60000 [========>.....................] - ETA: 1:18 - loss: 0.3326 - categorical_accuracy: 0.8959
18560/60000 [========>.....................] - ETA: 1:18 - loss: 0.3324 - categorical_accuracy: 0.8960
18592/60000 [========>.....................] - ETA: 1:17 - loss: 0.3320 - categorical_accuracy: 0.8961
18624/60000 [========>.....................] - ETA: 1:17 - loss: 0.3316 - categorical_accuracy: 0.8963
18656/60000 [========>.....................] - ETA: 1:17 - loss: 0.3317 - categorical_accuracy: 0.8962
18688/60000 [========>.....................] - ETA: 1:17 - loss: 0.3317 - categorical_accuracy: 0.8961
18720/60000 [========>.....................] - ETA: 1:17 - loss: 0.3314 - categorical_accuracy: 0.8962
18752/60000 [========>.....................] - ETA: 1:17 - loss: 0.3310 - categorical_accuracy: 0.8964
18784/60000 [========>.....................] - ETA: 1:17 - loss: 0.3307 - categorical_accuracy: 0.8965
18816/60000 [========>.....................] - ETA: 1:17 - loss: 0.3306 - categorical_accuracy: 0.8965
18848/60000 [========>.....................] - ETA: 1:17 - loss: 0.3306 - categorical_accuracy: 0.8965
18880/60000 [========>.....................] - ETA: 1:17 - loss: 0.3301 - categorical_accuracy: 0.8967
18912/60000 [========>.....................] - ETA: 1:17 - loss: 0.3296 - categorical_accuracy: 0.8969
18944/60000 [========>.....................] - ETA: 1:17 - loss: 0.3293 - categorical_accuracy: 0.8970
18976/60000 [========>.....................] - ETA: 1:17 - loss: 0.3295 - categorical_accuracy: 0.8969
19008/60000 [========>.....................] - ETA: 1:17 - loss: 0.3292 - categorical_accuracy: 0.8970
19040/60000 [========>.....................] - ETA: 1:17 - loss: 0.3291 - categorical_accuracy: 0.8971
19072/60000 [========>.....................] - ETA: 1:17 - loss: 0.3286 - categorical_accuracy: 0.8973
19104/60000 [========>.....................] - ETA: 1:17 - loss: 0.3285 - categorical_accuracy: 0.8974
19136/60000 [========>.....................] - ETA: 1:17 - loss: 0.3285 - categorical_accuracy: 0.8974
19168/60000 [========>.....................] - ETA: 1:16 - loss: 0.3284 - categorical_accuracy: 0.8974
19200/60000 [========>.....................] - ETA: 1:16 - loss: 0.3279 - categorical_accuracy: 0.8976
19232/60000 [========>.....................] - ETA: 1:16 - loss: 0.3276 - categorical_accuracy: 0.8976
19264/60000 [========>.....................] - ETA: 1:16 - loss: 0.3274 - categorical_accuracy: 0.8976
19296/60000 [========>.....................] - ETA: 1:16 - loss: 0.3270 - categorical_accuracy: 0.8978
19328/60000 [========>.....................] - ETA: 1:16 - loss: 0.3266 - categorical_accuracy: 0.8979
19360/60000 [========>.....................] - ETA: 1:16 - loss: 0.3267 - categorical_accuracy: 0.8978
19392/60000 [========>.....................] - ETA: 1:16 - loss: 0.3264 - categorical_accuracy: 0.8979
19424/60000 [========>.....................] - ETA: 1:16 - loss: 0.3261 - categorical_accuracy: 0.8980
19456/60000 [========>.....................] - ETA: 1:16 - loss: 0.3256 - categorical_accuracy: 0.8982
19488/60000 [========>.....................] - ETA: 1:16 - loss: 0.3258 - categorical_accuracy: 0.8981
19520/60000 [========>.....................] - ETA: 1:16 - loss: 0.3254 - categorical_accuracy: 0.8983
19552/60000 [========>.....................] - ETA: 1:16 - loss: 0.3250 - categorical_accuracy: 0.8984
19584/60000 [========>.....................] - ETA: 1:16 - loss: 0.3246 - categorical_accuracy: 0.8985
19616/60000 [========>.....................] - ETA: 1:16 - loss: 0.3242 - categorical_accuracy: 0.8986
19648/60000 [========>.....................] - ETA: 1:16 - loss: 0.3239 - categorical_accuracy: 0.8987
19680/60000 [========>.....................] - ETA: 1:15 - loss: 0.3236 - categorical_accuracy: 0.8988
19712/60000 [========>.....................] - ETA: 1:15 - loss: 0.3231 - categorical_accuracy: 0.8989
19744/60000 [========>.....................] - ETA: 1:15 - loss: 0.3228 - categorical_accuracy: 0.8990
19776/60000 [========>.....................] - ETA: 1:15 - loss: 0.3225 - categorical_accuracy: 0.8991
19808/60000 [========>.....................] - ETA: 1:15 - loss: 0.3220 - categorical_accuracy: 0.8993
19840/60000 [========>.....................] - ETA: 1:15 - loss: 0.3221 - categorical_accuracy: 0.8993
19872/60000 [========>.....................] - ETA: 1:15 - loss: 0.3217 - categorical_accuracy: 0.8994
19904/60000 [========>.....................] - ETA: 1:15 - loss: 0.3213 - categorical_accuracy: 0.8995
19936/60000 [========>.....................] - ETA: 1:15 - loss: 0.3209 - categorical_accuracy: 0.8996
19968/60000 [========>.....................] - ETA: 1:15 - loss: 0.3206 - categorical_accuracy: 0.8997
20000/60000 [=========>....................] - ETA: 1:15 - loss: 0.3204 - categorical_accuracy: 0.8997
20032/60000 [=========>....................] - ETA: 1:15 - loss: 0.3199 - categorical_accuracy: 0.8999
20064/60000 [=========>....................] - ETA: 1:15 - loss: 0.3194 - categorical_accuracy: 0.9000
20096/60000 [=========>....................] - ETA: 1:15 - loss: 0.3192 - categorical_accuracy: 0.9001
20128/60000 [=========>....................] - ETA: 1:15 - loss: 0.3189 - categorical_accuracy: 0.9002
20160/60000 [=========>....................] - ETA: 1:15 - loss: 0.3186 - categorical_accuracy: 0.9003
20192/60000 [=========>....................] - ETA: 1:15 - loss: 0.3184 - categorical_accuracy: 0.9003
20224/60000 [=========>....................] - ETA: 1:14 - loss: 0.3181 - categorical_accuracy: 0.9003
20256/60000 [=========>....................] - ETA: 1:14 - loss: 0.3178 - categorical_accuracy: 0.9004
20288/60000 [=========>....................] - ETA: 1:14 - loss: 0.3174 - categorical_accuracy: 0.9005
20320/60000 [=========>....................] - ETA: 1:14 - loss: 0.3171 - categorical_accuracy: 0.9006
20352/60000 [=========>....................] - ETA: 1:14 - loss: 0.3168 - categorical_accuracy: 0.9007
20384/60000 [=========>....................] - ETA: 1:14 - loss: 0.3167 - categorical_accuracy: 0.9009
20416/60000 [=========>....................] - ETA: 1:14 - loss: 0.3168 - categorical_accuracy: 0.9008
20448/60000 [=========>....................] - ETA: 1:14 - loss: 0.3169 - categorical_accuracy: 0.9007
20480/60000 [=========>....................] - ETA: 1:14 - loss: 0.3169 - categorical_accuracy: 0.9007
20512/60000 [=========>....................] - ETA: 1:14 - loss: 0.3166 - categorical_accuracy: 0.9008
20544/60000 [=========>....................] - ETA: 1:14 - loss: 0.3162 - categorical_accuracy: 0.9010
20576/60000 [=========>....................] - ETA: 1:14 - loss: 0.3159 - categorical_accuracy: 0.9010
20608/60000 [=========>....................] - ETA: 1:14 - loss: 0.3160 - categorical_accuracy: 0.9010
20640/60000 [=========>....................] - ETA: 1:14 - loss: 0.3159 - categorical_accuracy: 0.9010
20672/60000 [=========>....................] - ETA: 1:14 - loss: 0.3154 - categorical_accuracy: 0.9012
20704/60000 [=========>....................] - ETA: 1:14 - loss: 0.3152 - categorical_accuracy: 0.9013
20736/60000 [=========>....................] - ETA: 1:14 - loss: 0.3153 - categorical_accuracy: 0.9013
20768/60000 [=========>....................] - ETA: 1:13 - loss: 0.3150 - categorical_accuracy: 0.9014
20800/60000 [=========>....................] - ETA: 1:13 - loss: 0.3147 - categorical_accuracy: 0.9015
20832/60000 [=========>....................] - ETA: 1:13 - loss: 0.3145 - categorical_accuracy: 0.9016
20864/60000 [=========>....................] - ETA: 1:13 - loss: 0.3141 - categorical_accuracy: 0.9017
20896/60000 [=========>....................] - ETA: 1:13 - loss: 0.3140 - categorical_accuracy: 0.9018
20928/60000 [=========>....................] - ETA: 1:13 - loss: 0.3136 - categorical_accuracy: 0.9020
20960/60000 [=========>....................] - ETA: 1:13 - loss: 0.3133 - categorical_accuracy: 0.9021
20992/60000 [=========>....................] - ETA: 1:13 - loss: 0.3130 - categorical_accuracy: 0.9021
21024/60000 [=========>....................] - ETA: 1:13 - loss: 0.3126 - categorical_accuracy: 0.9023
21056/60000 [=========>....................] - ETA: 1:13 - loss: 0.3125 - categorical_accuracy: 0.9023
21088/60000 [=========>....................] - ETA: 1:13 - loss: 0.3121 - categorical_accuracy: 0.9024
21120/60000 [=========>....................] - ETA: 1:13 - loss: 0.3118 - categorical_accuracy: 0.9025
21152/60000 [=========>....................] - ETA: 1:13 - loss: 0.3119 - categorical_accuracy: 0.9025
21184/60000 [=========>....................] - ETA: 1:13 - loss: 0.3116 - categorical_accuracy: 0.9026
21216/60000 [=========>....................] - ETA: 1:13 - loss: 0.3112 - categorical_accuracy: 0.9027
21248/60000 [=========>....................] - ETA: 1:13 - loss: 0.3112 - categorical_accuracy: 0.9027
21280/60000 [=========>....................] - ETA: 1:13 - loss: 0.3111 - categorical_accuracy: 0.9027
21312/60000 [=========>....................] - ETA: 1:12 - loss: 0.3107 - categorical_accuracy: 0.9028
21344/60000 [=========>....................] - ETA: 1:12 - loss: 0.3108 - categorical_accuracy: 0.9028
21376/60000 [=========>....................] - ETA: 1:12 - loss: 0.3104 - categorical_accuracy: 0.9029
21408/60000 [=========>....................] - ETA: 1:12 - loss: 0.3102 - categorical_accuracy: 0.9030
21440/60000 [=========>....................] - ETA: 1:12 - loss: 0.3100 - categorical_accuracy: 0.9030
21472/60000 [=========>....................] - ETA: 1:12 - loss: 0.3096 - categorical_accuracy: 0.9031
21504/60000 [=========>....................] - ETA: 1:12 - loss: 0.3093 - categorical_accuracy: 0.9032
21536/60000 [=========>....................] - ETA: 1:12 - loss: 0.3090 - categorical_accuracy: 0.9033
21568/60000 [=========>....................] - ETA: 1:12 - loss: 0.3088 - categorical_accuracy: 0.9034
21600/60000 [=========>....................] - ETA: 1:12 - loss: 0.3087 - categorical_accuracy: 0.9034
21632/60000 [=========>....................] - ETA: 1:12 - loss: 0.3085 - categorical_accuracy: 0.9035
21664/60000 [=========>....................] - ETA: 1:12 - loss: 0.3082 - categorical_accuracy: 0.9036
21696/60000 [=========>....................] - ETA: 1:12 - loss: 0.3080 - categorical_accuracy: 0.9037
21728/60000 [=========>....................] - ETA: 1:12 - loss: 0.3081 - categorical_accuracy: 0.9037
21760/60000 [=========>....................] - ETA: 1:12 - loss: 0.3079 - categorical_accuracy: 0.9038
21792/60000 [=========>....................] - ETA: 1:12 - loss: 0.3075 - categorical_accuracy: 0.9039
21824/60000 [=========>....................] - ETA: 1:11 - loss: 0.3071 - categorical_accuracy: 0.9040
21856/60000 [=========>....................] - ETA: 1:11 - loss: 0.3069 - categorical_accuracy: 0.9041
21888/60000 [=========>....................] - ETA: 1:11 - loss: 0.3068 - categorical_accuracy: 0.9041
21920/60000 [=========>....................] - ETA: 1:11 - loss: 0.3067 - categorical_accuracy: 0.9042
21952/60000 [=========>....................] - ETA: 1:11 - loss: 0.3066 - categorical_accuracy: 0.9042
21984/60000 [=========>....................] - ETA: 1:11 - loss: 0.3062 - categorical_accuracy: 0.9043
22016/60000 [==========>...................] - ETA: 1:11 - loss: 0.3060 - categorical_accuracy: 0.9044
22048/60000 [==========>...................] - ETA: 1:11 - loss: 0.3056 - categorical_accuracy: 0.9045
22080/60000 [==========>...................] - ETA: 1:11 - loss: 0.3053 - categorical_accuracy: 0.9046
22112/60000 [==========>...................] - ETA: 1:11 - loss: 0.3052 - categorical_accuracy: 0.9047
22144/60000 [==========>...................] - ETA: 1:11 - loss: 0.3050 - categorical_accuracy: 0.9048
22176/60000 [==========>...................] - ETA: 1:11 - loss: 0.3047 - categorical_accuracy: 0.9049
22208/60000 [==========>...................] - ETA: 1:11 - loss: 0.3043 - categorical_accuracy: 0.9049
22240/60000 [==========>...................] - ETA: 1:11 - loss: 0.3042 - categorical_accuracy: 0.9050
22272/60000 [==========>...................] - ETA: 1:11 - loss: 0.3038 - categorical_accuracy: 0.9051
22304/60000 [==========>...................] - ETA: 1:11 - loss: 0.3034 - categorical_accuracy: 0.9053
22336/60000 [==========>...................] - ETA: 1:11 - loss: 0.3034 - categorical_accuracy: 0.9053
22368/60000 [==========>...................] - ETA: 1:10 - loss: 0.3033 - categorical_accuracy: 0.9053
22400/60000 [==========>...................] - ETA: 1:10 - loss: 0.3030 - categorical_accuracy: 0.9054
22432/60000 [==========>...................] - ETA: 1:10 - loss: 0.3027 - categorical_accuracy: 0.9055
22464/60000 [==========>...................] - ETA: 1:10 - loss: 0.3024 - categorical_accuracy: 0.9056
22496/60000 [==========>...................] - ETA: 1:10 - loss: 0.3021 - categorical_accuracy: 0.9058
22528/60000 [==========>...................] - ETA: 1:10 - loss: 0.3017 - categorical_accuracy: 0.9059
22560/60000 [==========>...................] - ETA: 1:10 - loss: 0.3015 - categorical_accuracy: 0.9059
22592/60000 [==========>...................] - ETA: 1:10 - loss: 0.3014 - categorical_accuracy: 0.9060
22624/60000 [==========>...................] - ETA: 1:10 - loss: 0.3013 - categorical_accuracy: 0.9061
22656/60000 [==========>...................] - ETA: 1:10 - loss: 0.3010 - categorical_accuracy: 0.9062
22688/60000 [==========>...................] - ETA: 1:10 - loss: 0.3006 - categorical_accuracy: 0.9063
22720/60000 [==========>...................] - ETA: 1:10 - loss: 0.3006 - categorical_accuracy: 0.9063
22752/60000 [==========>...................] - ETA: 1:10 - loss: 0.3003 - categorical_accuracy: 0.9065
22784/60000 [==========>...................] - ETA: 1:10 - loss: 0.3000 - categorical_accuracy: 0.9066
22816/60000 [==========>...................] - ETA: 1:10 - loss: 0.2998 - categorical_accuracy: 0.9066
22848/60000 [==========>...................] - ETA: 1:10 - loss: 0.2996 - categorical_accuracy: 0.9066
22880/60000 [==========>...................] - ETA: 1:09 - loss: 0.2992 - categorical_accuracy: 0.9067
22912/60000 [==========>...................] - ETA: 1:09 - loss: 0.2990 - categorical_accuracy: 0.9068
22944/60000 [==========>...................] - ETA: 1:09 - loss: 0.2987 - categorical_accuracy: 0.9069
22976/60000 [==========>...................] - ETA: 1:09 - loss: 0.2984 - categorical_accuracy: 0.9069
23008/60000 [==========>...................] - ETA: 1:09 - loss: 0.2981 - categorical_accuracy: 0.9071
23040/60000 [==========>...................] - ETA: 1:09 - loss: 0.2980 - categorical_accuracy: 0.9071
23072/60000 [==========>...................] - ETA: 1:09 - loss: 0.2981 - categorical_accuracy: 0.9072
23104/60000 [==========>...................] - ETA: 1:09 - loss: 0.2983 - categorical_accuracy: 0.9071
23136/60000 [==========>...................] - ETA: 1:09 - loss: 0.2981 - categorical_accuracy: 0.9071
23168/60000 [==========>...................] - ETA: 1:09 - loss: 0.2978 - categorical_accuracy: 0.9072
23200/60000 [==========>...................] - ETA: 1:09 - loss: 0.2976 - categorical_accuracy: 0.9073
23232/60000 [==========>...................] - ETA: 1:09 - loss: 0.2973 - categorical_accuracy: 0.9074
23264/60000 [==========>...................] - ETA: 1:09 - loss: 0.2975 - categorical_accuracy: 0.9073
23296/60000 [==========>...................] - ETA: 1:09 - loss: 0.2972 - categorical_accuracy: 0.9074
23328/60000 [==========>...................] - ETA: 1:09 - loss: 0.2968 - categorical_accuracy: 0.9075
23360/60000 [==========>...................] - ETA: 1:09 - loss: 0.2965 - categorical_accuracy: 0.9077
23392/60000 [==========>...................] - ETA: 1:08 - loss: 0.2962 - categorical_accuracy: 0.9077
23424/60000 [==========>...................] - ETA: 1:08 - loss: 0.2961 - categorical_accuracy: 0.9078
23456/60000 [==========>...................] - ETA: 1:08 - loss: 0.2958 - categorical_accuracy: 0.9079
23488/60000 [==========>...................] - ETA: 1:08 - loss: 0.2956 - categorical_accuracy: 0.9078
23520/60000 [==========>...................] - ETA: 1:08 - loss: 0.2954 - categorical_accuracy: 0.9079
23552/60000 [==========>...................] - ETA: 1:08 - loss: 0.2950 - categorical_accuracy: 0.9080
23584/60000 [==========>...................] - ETA: 1:08 - loss: 0.2947 - categorical_accuracy: 0.9081
23616/60000 [==========>...................] - ETA: 1:08 - loss: 0.2943 - categorical_accuracy: 0.9082
23648/60000 [==========>...................] - ETA: 1:08 - loss: 0.2942 - categorical_accuracy: 0.9083
23680/60000 [==========>...................] - ETA: 1:08 - loss: 0.2939 - categorical_accuracy: 0.9084
23712/60000 [==========>...................] - ETA: 1:08 - loss: 0.2938 - categorical_accuracy: 0.9084
23744/60000 [==========>...................] - ETA: 1:08 - loss: 0.2935 - categorical_accuracy: 0.9085
23776/60000 [==========>...................] - ETA: 1:08 - loss: 0.2931 - categorical_accuracy: 0.9086
23808/60000 [==========>...................] - ETA: 1:08 - loss: 0.2929 - categorical_accuracy: 0.9087
23840/60000 [==========>...................] - ETA: 1:08 - loss: 0.2928 - categorical_accuracy: 0.9087
23872/60000 [==========>...................] - ETA: 1:08 - loss: 0.2926 - categorical_accuracy: 0.9088
23904/60000 [==========>...................] - ETA: 1:08 - loss: 0.2922 - categorical_accuracy: 0.9089
23936/60000 [==========>...................] - ETA: 1:07 - loss: 0.2921 - categorical_accuracy: 0.9090
23968/60000 [==========>...................] - ETA: 1:07 - loss: 0.2919 - categorical_accuracy: 0.9090
24000/60000 [===========>..................] - ETA: 1:07 - loss: 0.2918 - categorical_accuracy: 0.9090
24032/60000 [===========>..................] - ETA: 1:07 - loss: 0.2915 - categorical_accuracy: 0.9091
24064/60000 [===========>..................] - ETA: 1:07 - loss: 0.2914 - categorical_accuracy: 0.9091
24096/60000 [===========>..................] - ETA: 1:07 - loss: 0.2911 - categorical_accuracy: 0.9092
24128/60000 [===========>..................] - ETA: 1:07 - loss: 0.2907 - categorical_accuracy: 0.9093
24160/60000 [===========>..................] - ETA: 1:07 - loss: 0.2905 - categorical_accuracy: 0.9094
24192/60000 [===========>..................] - ETA: 1:07 - loss: 0.2904 - categorical_accuracy: 0.9094
24224/60000 [===========>..................] - ETA: 1:07 - loss: 0.2900 - categorical_accuracy: 0.9096
24256/60000 [===========>..................] - ETA: 1:07 - loss: 0.2899 - categorical_accuracy: 0.9096
24288/60000 [===========>..................] - ETA: 1:07 - loss: 0.2896 - categorical_accuracy: 0.9097
24320/60000 [===========>..................] - ETA: 1:07 - loss: 0.2892 - categorical_accuracy: 0.9098
24352/60000 [===========>..................] - ETA: 1:07 - loss: 0.2889 - categorical_accuracy: 0.9099
24384/60000 [===========>..................] - ETA: 1:07 - loss: 0.2888 - categorical_accuracy: 0.9099
24416/60000 [===========>..................] - ETA: 1:07 - loss: 0.2887 - categorical_accuracy: 0.9099
24448/60000 [===========>..................] - ETA: 1:07 - loss: 0.2885 - categorical_accuracy: 0.9100
24480/60000 [===========>..................] - ETA: 1:06 - loss: 0.2883 - categorical_accuracy: 0.9101
24512/60000 [===========>..................] - ETA: 1:06 - loss: 0.2880 - categorical_accuracy: 0.9102
24544/60000 [===========>..................] - ETA: 1:06 - loss: 0.2877 - categorical_accuracy: 0.9102
24576/60000 [===========>..................] - ETA: 1:06 - loss: 0.2878 - categorical_accuracy: 0.9102
24608/60000 [===========>..................] - ETA: 1:06 - loss: 0.2875 - categorical_accuracy: 0.9103
24640/60000 [===========>..................] - ETA: 1:06 - loss: 0.2872 - categorical_accuracy: 0.9104
24672/60000 [===========>..................] - ETA: 1:06 - loss: 0.2869 - categorical_accuracy: 0.9105
24704/60000 [===========>..................] - ETA: 1:06 - loss: 0.2868 - categorical_accuracy: 0.9105
24736/60000 [===========>..................] - ETA: 1:06 - loss: 0.2867 - categorical_accuracy: 0.9105
24768/60000 [===========>..................] - ETA: 1:06 - loss: 0.2866 - categorical_accuracy: 0.9106
24800/60000 [===========>..................] - ETA: 1:06 - loss: 0.2863 - categorical_accuracy: 0.9107
24832/60000 [===========>..................] - ETA: 1:06 - loss: 0.2861 - categorical_accuracy: 0.9107
24864/60000 [===========>..................] - ETA: 1:06 - loss: 0.2858 - categorical_accuracy: 0.9108
24896/60000 [===========>..................] - ETA: 1:06 - loss: 0.2855 - categorical_accuracy: 0.9109
24928/60000 [===========>..................] - ETA: 1:06 - loss: 0.2852 - categorical_accuracy: 0.9110
24960/60000 [===========>..................] - ETA: 1:06 - loss: 0.2851 - categorical_accuracy: 0.9111
24992/60000 [===========>..................] - ETA: 1:05 - loss: 0.2849 - categorical_accuracy: 0.9111
25024/60000 [===========>..................] - ETA: 1:05 - loss: 0.2845 - categorical_accuracy: 0.9112
25056/60000 [===========>..................] - ETA: 1:05 - loss: 0.2845 - categorical_accuracy: 0.9112
25088/60000 [===========>..................] - ETA: 1:05 - loss: 0.2842 - categorical_accuracy: 0.9113
25120/60000 [===========>..................] - ETA: 1:05 - loss: 0.2845 - categorical_accuracy: 0.9113
25152/60000 [===========>..................] - ETA: 1:05 - loss: 0.2841 - categorical_accuracy: 0.9114
25184/60000 [===========>..................] - ETA: 1:05 - loss: 0.2839 - categorical_accuracy: 0.9115
25216/60000 [===========>..................] - ETA: 1:05 - loss: 0.2838 - categorical_accuracy: 0.9115
25248/60000 [===========>..................] - ETA: 1:05 - loss: 0.2837 - categorical_accuracy: 0.9116
25280/60000 [===========>..................] - ETA: 1:05 - loss: 0.2836 - categorical_accuracy: 0.9116
25312/60000 [===========>..................] - ETA: 1:05 - loss: 0.2839 - categorical_accuracy: 0.9116
25344/60000 [===========>..................] - ETA: 1:05 - loss: 0.2837 - categorical_accuracy: 0.9116
25376/60000 [===========>..................] - ETA: 1:05 - loss: 0.2840 - categorical_accuracy: 0.9116
25408/60000 [===========>..................] - ETA: 1:05 - loss: 0.2837 - categorical_accuracy: 0.9117
25440/60000 [===========>..................] - ETA: 1:05 - loss: 0.2836 - categorical_accuracy: 0.9118
25472/60000 [===========>..................] - ETA: 1:05 - loss: 0.2834 - categorical_accuracy: 0.9117
25504/60000 [===========>..................] - ETA: 1:05 - loss: 0.2832 - categorical_accuracy: 0.9119
25536/60000 [===========>..................] - ETA: 1:05 - loss: 0.2831 - categorical_accuracy: 0.9119
25568/60000 [===========>..................] - ETA: 1:04 - loss: 0.2828 - categorical_accuracy: 0.9120
25600/60000 [===========>..................] - ETA: 1:04 - loss: 0.2826 - categorical_accuracy: 0.9121
25632/60000 [===========>..................] - ETA: 1:04 - loss: 0.2825 - categorical_accuracy: 0.9121
25664/60000 [===========>..................] - ETA: 1:04 - loss: 0.2824 - categorical_accuracy: 0.9120
25696/60000 [===========>..................] - ETA: 1:04 - loss: 0.2821 - categorical_accuracy: 0.9121
25728/60000 [===========>..................] - ETA: 1:04 - loss: 0.2818 - categorical_accuracy: 0.9122
25760/60000 [===========>..................] - ETA: 1:04 - loss: 0.2816 - categorical_accuracy: 0.9123
25792/60000 [===========>..................] - ETA: 1:04 - loss: 0.2813 - categorical_accuracy: 0.9124
25824/60000 [===========>..................] - ETA: 1:04 - loss: 0.2810 - categorical_accuracy: 0.9125
25856/60000 [===========>..................] - ETA: 1:04 - loss: 0.2807 - categorical_accuracy: 0.9126
25888/60000 [===========>..................] - ETA: 1:04 - loss: 0.2804 - categorical_accuracy: 0.9127
25920/60000 [===========>..................] - ETA: 1:04 - loss: 0.2800 - categorical_accuracy: 0.9128
25952/60000 [===========>..................] - ETA: 1:04 - loss: 0.2799 - categorical_accuracy: 0.9129
25984/60000 [===========>..................] - ETA: 1:04 - loss: 0.2797 - categorical_accuracy: 0.9129
26016/60000 [============>.................] - ETA: 1:04 - loss: 0.2797 - categorical_accuracy: 0.9129
26048/60000 [============>.................] - ETA: 1:04 - loss: 0.2796 - categorical_accuracy: 0.9130
26080/60000 [============>.................] - ETA: 1:03 - loss: 0.2795 - categorical_accuracy: 0.9130
26112/60000 [============>.................] - ETA: 1:03 - loss: 0.2792 - categorical_accuracy: 0.9131
26144/60000 [============>.................] - ETA: 1:03 - loss: 0.2789 - categorical_accuracy: 0.9132
26176/60000 [============>.................] - ETA: 1:03 - loss: 0.2786 - categorical_accuracy: 0.9133
26208/60000 [============>.................] - ETA: 1:03 - loss: 0.2783 - categorical_accuracy: 0.9134
26240/60000 [============>.................] - ETA: 1:03 - loss: 0.2783 - categorical_accuracy: 0.9134
26272/60000 [============>.................] - ETA: 1:03 - loss: 0.2780 - categorical_accuracy: 0.9135
26304/60000 [============>.................] - ETA: 1:03 - loss: 0.2777 - categorical_accuracy: 0.9136
26336/60000 [============>.................] - ETA: 1:03 - loss: 0.2777 - categorical_accuracy: 0.9137
26368/60000 [============>.................] - ETA: 1:03 - loss: 0.2775 - categorical_accuracy: 0.9138
26400/60000 [============>.................] - ETA: 1:03 - loss: 0.2772 - categorical_accuracy: 0.9139
26432/60000 [============>.................] - ETA: 1:03 - loss: 0.2769 - categorical_accuracy: 0.9140
26464/60000 [============>.................] - ETA: 1:03 - loss: 0.2766 - categorical_accuracy: 0.9141
26496/60000 [============>.................] - ETA: 1:03 - loss: 0.2767 - categorical_accuracy: 0.9141
26528/60000 [============>.................] - ETA: 1:03 - loss: 0.2767 - categorical_accuracy: 0.9141
26560/60000 [============>.................] - ETA: 1:03 - loss: 0.2764 - categorical_accuracy: 0.9142
26592/60000 [============>.................] - ETA: 1:02 - loss: 0.2761 - categorical_accuracy: 0.9143
26624/60000 [============>.................] - ETA: 1:02 - loss: 0.2759 - categorical_accuracy: 0.9144
26656/60000 [============>.................] - ETA: 1:02 - loss: 0.2762 - categorical_accuracy: 0.9144
26688/60000 [============>.................] - ETA: 1:02 - loss: 0.2760 - categorical_accuracy: 0.9144
26720/60000 [============>.................] - ETA: 1:02 - loss: 0.2757 - categorical_accuracy: 0.9145
26752/60000 [============>.................] - ETA: 1:02 - loss: 0.2756 - categorical_accuracy: 0.9145
26784/60000 [============>.................] - ETA: 1:02 - loss: 0.2753 - categorical_accuracy: 0.9146
26816/60000 [============>.................] - ETA: 1:02 - loss: 0.2753 - categorical_accuracy: 0.9147
26848/60000 [============>.................] - ETA: 1:02 - loss: 0.2752 - categorical_accuracy: 0.9147
26880/60000 [============>.................] - ETA: 1:02 - loss: 0.2749 - categorical_accuracy: 0.9148
26912/60000 [============>.................] - ETA: 1:02 - loss: 0.2748 - categorical_accuracy: 0.9148
26944/60000 [============>.................] - ETA: 1:02 - loss: 0.2746 - categorical_accuracy: 0.9149
26976/60000 [============>.................] - ETA: 1:02 - loss: 0.2746 - categorical_accuracy: 0.9149
27008/60000 [============>.................] - ETA: 1:02 - loss: 0.2745 - categorical_accuracy: 0.9149
27040/60000 [============>.................] - ETA: 1:02 - loss: 0.2742 - categorical_accuracy: 0.9150
27072/60000 [============>.................] - ETA: 1:02 - loss: 0.2743 - categorical_accuracy: 0.9150
27104/60000 [============>.................] - ETA: 1:02 - loss: 0.2740 - categorical_accuracy: 0.9151
27136/60000 [============>.................] - ETA: 1:01 - loss: 0.2739 - categorical_accuracy: 0.9151
27168/60000 [============>.................] - ETA: 1:01 - loss: 0.2744 - categorical_accuracy: 0.9151
27200/60000 [============>.................] - ETA: 1:01 - loss: 0.2742 - categorical_accuracy: 0.9151
27232/60000 [============>.................] - ETA: 1:01 - loss: 0.2739 - categorical_accuracy: 0.9152
27264/60000 [============>.................] - ETA: 1:01 - loss: 0.2737 - categorical_accuracy: 0.9153
27296/60000 [============>.................] - ETA: 1:01 - loss: 0.2737 - categorical_accuracy: 0.9153
27328/60000 [============>.................] - ETA: 1:01 - loss: 0.2736 - categorical_accuracy: 0.9153
27360/60000 [============>.................] - ETA: 1:01 - loss: 0.2738 - categorical_accuracy: 0.9152
27392/60000 [============>.................] - ETA: 1:01 - loss: 0.2736 - categorical_accuracy: 0.9153
27424/60000 [============>.................] - ETA: 1:01 - loss: 0.2734 - categorical_accuracy: 0.9153
27456/60000 [============>.................] - ETA: 1:01 - loss: 0.2732 - categorical_accuracy: 0.9154
27488/60000 [============>.................] - ETA: 1:01 - loss: 0.2729 - categorical_accuracy: 0.9155
27520/60000 [============>.................] - ETA: 1:01 - loss: 0.2727 - categorical_accuracy: 0.9155
27552/60000 [============>.................] - ETA: 1:01 - loss: 0.2724 - categorical_accuracy: 0.9156
27584/60000 [============>.................] - ETA: 1:01 - loss: 0.2722 - categorical_accuracy: 0.9157
27616/60000 [============>.................] - ETA: 1:01 - loss: 0.2720 - categorical_accuracy: 0.9157
27648/60000 [============>.................] - ETA: 1:00 - loss: 0.2718 - categorical_accuracy: 0.9158
27680/60000 [============>.................] - ETA: 1:00 - loss: 0.2717 - categorical_accuracy: 0.9158
27712/60000 [============>.................] - ETA: 1:00 - loss: 0.2720 - categorical_accuracy: 0.9158
27744/60000 [============>.................] - ETA: 1:00 - loss: 0.2717 - categorical_accuracy: 0.9159
27776/60000 [============>.................] - ETA: 1:00 - loss: 0.2715 - categorical_accuracy: 0.9160
27808/60000 [============>.................] - ETA: 1:00 - loss: 0.2714 - categorical_accuracy: 0.9160
27840/60000 [============>.................] - ETA: 1:00 - loss: 0.2712 - categorical_accuracy: 0.9161
27872/60000 [============>.................] - ETA: 1:00 - loss: 0.2710 - categorical_accuracy: 0.9161
27904/60000 [============>.................] - ETA: 1:00 - loss: 0.2707 - categorical_accuracy: 0.9162
27936/60000 [============>.................] - ETA: 1:00 - loss: 0.2705 - categorical_accuracy: 0.9163
27968/60000 [============>.................] - ETA: 1:00 - loss: 0.2705 - categorical_accuracy: 0.9163
28000/60000 [=============>................] - ETA: 1:00 - loss: 0.2703 - categorical_accuracy: 0.9164
28032/60000 [=============>................] - ETA: 1:00 - loss: 0.2701 - categorical_accuracy: 0.9164
28064/60000 [=============>................] - ETA: 1:00 - loss: 0.2700 - categorical_accuracy: 0.9164
28096/60000 [=============>................] - ETA: 1:00 - loss: 0.2697 - categorical_accuracy: 0.9165
28128/60000 [=============>................] - ETA: 1:00 - loss: 0.2696 - categorical_accuracy: 0.9166
28160/60000 [=============>................] - ETA: 1:00 - loss: 0.2694 - categorical_accuracy: 0.9167
28192/60000 [=============>................] - ETA: 59s - loss: 0.2692 - categorical_accuracy: 0.9167 
28224/60000 [=============>................] - ETA: 59s - loss: 0.2691 - categorical_accuracy: 0.9168
28256/60000 [=============>................] - ETA: 59s - loss: 0.2688 - categorical_accuracy: 0.9168
28288/60000 [=============>................] - ETA: 59s - loss: 0.2689 - categorical_accuracy: 0.9168
28320/60000 [=============>................] - ETA: 59s - loss: 0.2688 - categorical_accuracy: 0.9168
28352/60000 [=============>................] - ETA: 59s - loss: 0.2687 - categorical_accuracy: 0.9169
28384/60000 [=============>................] - ETA: 59s - loss: 0.2684 - categorical_accuracy: 0.9170
28416/60000 [=============>................] - ETA: 59s - loss: 0.2683 - categorical_accuracy: 0.9170
28448/60000 [=============>................] - ETA: 59s - loss: 0.2681 - categorical_accuracy: 0.9171
28480/60000 [=============>................] - ETA: 59s - loss: 0.2681 - categorical_accuracy: 0.9170
28512/60000 [=============>................] - ETA: 59s - loss: 0.2679 - categorical_accuracy: 0.9171
28544/60000 [=============>................] - ETA: 59s - loss: 0.2677 - categorical_accuracy: 0.9172
28576/60000 [=============>................] - ETA: 59s - loss: 0.2676 - categorical_accuracy: 0.9172
28608/60000 [=============>................] - ETA: 59s - loss: 0.2674 - categorical_accuracy: 0.9173
28640/60000 [=============>................] - ETA: 59s - loss: 0.2672 - categorical_accuracy: 0.9173
28672/60000 [=============>................] - ETA: 59s - loss: 0.2671 - categorical_accuracy: 0.9174
28704/60000 [=============>................] - ETA: 58s - loss: 0.2671 - categorical_accuracy: 0.9174
28736/60000 [=============>................] - ETA: 58s - loss: 0.2669 - categorical_accuracy: 0.9174
28768/60000 [=============>................] - ETA: 58s - loss: 0.2666 - categorical_accuracy: 0.9175
28800/60000 [=============>................] - ETA: 58s - loss: 0.2664 - categorical_accuracy: 0.9176
28832/60000 [=============>................] - ETA: 58s - loss: 0.2665 - categorical_accuracy: 0.9176
28864/60000 [=============>................] - ETA: 58s - loss: 0.2664 - categorical_accuracy: 0.9176
28896/60000 [=============>................] - ETA: 58s - loss: 0.2661 - categorical_accuracy: 0.9177
28928/60000 [=============>................] - ETA: 58s - loss: 0.2658 - categorical_accuracy: 0.9178
28960/60000 [=============>................] - ETA: 58s - loss: 0.2657 - categorical_accuracy: 0.9179
28992/60000 [=============>................] - ETA: 58s - loss: 0.2656 - categorical_accuracy: 0.9179
29024/60000 [=============>................] - ETA: 58s - loss: 0.2653 - categorical_accuracy: 0.9180
29056/60000 [=============>................] - ETA: 58s - loss: 0.2652 - categorical_accuracy: 0.9180
29088/60000 [=============>................] - ETA: 58s - loss: 0.2652 - categorical_accuracy: 0.9180
29120/60000 [=============>................] - ETA: 58s - loss: 0.2650 - categorical_accuracy: 0.9181
29152/60000 [=============>................] - ETA: 58s - loss: 0.2651 - categorical_accuracy: 0.9180
29184/60000 [=============>................] - ETA: 58s - loss: 0.2651 - categorical_accuracy: 0.9180
29216/60000 [=============>................] - ETA: 58s - loss: 0.2648 - categorical_accuracy: 0.9181
29248/60000 [=============>................] - ETA: 57s - loss: 0.2646 - categorical_accuracy: 0.9182
29280/60000 [=============>................] - ETA: 57s - loss: 0.2645 - categorical_accuracy: 0.9182
29312/60000 [=============>................] - ETA: 57s - loss: 0.2643 - categorical_accuracy: 0.9183
29344/60000 [=============>................] - ETA: 57s - loss: 0.2641 - categorical_accuracy: 0.9183
29376/60000 [=============>................] - ETA: 57s - loss: 0.2642 - categorical_accuracy: 0.9184
29408/60000 [=============>................] - ETA: 57s - loss: 0.2641 - categorical_accuracy: 0.9184
29440/60000 [=============>................] - ETA: 57s - loss: 0.2639 - categorical_accuracy: 0.9185
29472/60000 [=============>................] - ETA: 57s - loss: 0.2638 - categorical_accuracy: 0.9185
29504/60000 [=============>................] - ETA: 57s - loss: 0.2638 - categorical_accuracy: 0.9185
29536/60000 [=============>................] - ETA: 57s - loss: 0.2637 - categorical_accuracy: 0.9185
29568/60000 [=============>................] - ETA: 57s - loss: 0.2637 - categorical_accuracy: 0.9185
29600/60000 [=============>................] - ETA: 57s - loss: 0.2635 - categorical_accuracy: 0.9185
29632/60000 [=============>................] - ETA: 57s - loss: 0.2633 - categorical_accuracy: 0.9186
29664/60000 [=============>................] - ETA: 57s - loss: 0.2630 - categorical_accuracy: 0.9187
29696/60000 [=============>................] - ETA: 57s - loss: 0.2632 - categorical_accuracy: 0.9187
29728/60000 [=============>................] - ETA: 57s - loss: 0.2631 - categorical_accuracy: 0.9187
29760/60000 [=============>................] - ETA: 56s - loss: 0.2630 - categorical_accuracy: 0.9187
29792/60000 [=============>................] - ETA: 56s - loss: 0.2628 - categorical_accuracy: 0.9188
29824/60000 [=============>................] - ETA: 56s - loss: 0.2626 - categorical_accuracy: 0.9189
29856/60000 [=============>................] - ETA: 56s - loss: 0.2624 - categorical_accuracy: 0.9189
29888/60000 [=============>................] - ETA: 56s - loss: 0.2622 - categorical_accuracy: 0.9190
29920/60000 [=============>................] - ETA: 56s - loss: 0.2620 - categorical_accuracy: 0.9191
29952/60000 [=============>................] - ETA: 56s - loss: 0.2618 - categorical_accuracy: 0.9191
29984/60000 [=============>................] - ETA: 56s - loss: 0.2615 - categorical_accuracy: 0.9192
30016/60000 [==============>...............] - ETA: 56s - loss: 0.2614 - categorical_accuracy: 0.9192
30048/60000 [==============>...............] - ETA: 56s - loss: 0.2615 - categorical_accuracy: 0.9193
30080/60000 [==============>...............] - ETA: 56s - loss: 0.2614 - categorical_accuracy: 0.9193
30112/60000 [==============>...............] - ETA: 56s - loss: 0.2613 - categorical_accuracy: 0.9193
30144/60000 [==============>...............] - ETA: 56s - loss: 0.2611 - categorical_accuracy: 0.9194
30176/60000 [==============>...............] - ETA: 56s - loss: 0.2609 - categorical_accuracy: 0.9195
30208/60000 [==============>...............] - ETA: 56s - loss: 0.2606 - categorical_accuracy: 0.9196
30240/60000 [==============>...............] - ETA: 56s - loss: 0.2607 - categorical_accuracy: 0.9196
30272/60000 [==============>...............] - ETA: 56s - loss: 0.2605 - categorical_accuracy: 0.9196
30304/60000 [==============>...............] - ETA: 55s - loss: 0.2603 - categorical_accuracy: 0.9197
30336/60000 [==============>...............] - ETA: 55s - loss: 0.2602 - categorical_accuracy: 0.9197
30368/60000 [==============>...............] - ETA: 55s - loss: 0.2599 - categorical_accuracy: 0.9198
30400/60000 [==============>...............] - ETA: 55s - loss: 0.2598 - categorical_accuracy: 0.9197
30432/60000 [==============>...............] - ETA: 55s - loss: 0.2596 - categorical_accuracy: 0.9198
30464/60000 [==============>...............] - ETA: 55s - loss: 0.2596 - categorical_accuracy: 0.9198
30496/60000 [==============>...............] - ETA: 55s - loss: 0.2594 - categorical_accuracy: 0.9199
30528/60000 [==============>...............] - ETA: 55s - loss: 0.2593 - categorical_accuracy: 0.9199
30560/60000 [==============>...............] - ETA: 55s - loss: 0.2591 - categorical_accuracy: 0.9200
30592/60000 [==============>...............] - ETA: 55s - loss: 0.2589 - categorical_accuracy: 0.9200
30624/60000 [==============>...............] - ETA: 55s - loss: 0.2587 - categorical_accuracy: 0.9201
30656/60000 [==============>...............] - ETA: 55s - loss: 0.2585 - categorical_accuracy: 0.9202
30688/60000 [==============>...............] - ETA: 55s - loss: 0.2584 - categorical_accuracy: 0.9202
30720/60000 [==============>...............] - ETA: 55s - loss: 0.2582 - categorical_accuracy: 0.9203
30752/60000 [==============>...............] - ETA: 55s - loss: 0.2579 - categorical_accuracy: 0.9204
30784/60000 [==============>...............] - ETA: 55s - loss: 0.2577 - categorical_accuracy: 0.9204
30816/60000 [==============>...............] - ETA: 54s - loss: 0.2576 - categorical_accuracy: 0.9205
30848/60000 [==============>...............] - ETA: 54s - loss: 0.2574 - categorical_accuracy: 0.9205
30880/60000 [==============>...............] - ETA: 54s - loss: 0.2572 - categorical_accuracy: 0.9206
30912/60000 [==============>...............] - ETA: 54s - loss: 0.2572 - categorical_accuracy: 0.9206
30944/60000 [==============>...............] - ETA: 54s - loss: 0.2571 - categorical_accuracy: 0.9206
30976/60000 [==============>...............] - ETA: 54s - loss: 0.2570 - categorical_accuracy: 0.9206
31008/60000 [==============>...............] - ETA: 54s - loss: 0.2569 - categorical_accuracy: 0.9206
31040/60000 [==============>...............] - ETA: 54s - loss: 0.2567 - categorical_accuracy: 0.9207
31072/60000 [==============>...............] - ETA: 54s - loss: 0.2567 - categorical_accuracy: 0.9206
31104/60000 [==============>...............] - ETA: 54s - loss: 0.2566 - categorical_accuracy: 0.9207
31136/60000 [==============>...............] - ETA: 54s - loss: 0.2564 - categorical_accuracy: 0.9207
31168/60000 [==============>...............] - ETA: 54s - loss: 0.2565 - categorical_accuracy: 0.9208
31200/60000 [==============>...............] - ETA: 54s - loss: 0.2563 - categorical_accuracy: 0.9208
31232/60000 [==============>...............] - ETA: 54s - loss: 0.2565 - categorical_accuracy: 0.9208
31264/60000 [==============>...............] - ETA: 54s - loss: 0.2563 - categorical_accuracy: 0.9209
31296/60000 [==============>...............] - ETA: 54s - loss: 0.2562 - categorical_accuracy: 0.9209
31328/60000 [==============>...............] - ETA: 53s - loss: 0.2561 - categorical_accuracy: 0.9210
31360/60000 [==============>...............] - ETA: 53s - loss: 0.2558 - categorical_accuracy: 0.9210
31392/60000 [==============>...............] - ETA: 53s - loss: 0.2557 - categorical_accuracy: 0.9211
31424/60000 [==============>...............] - ETA: 53s - loss: 0.2556 - categorical_accuracy: 0.9212
31456/60000 [==============>...............] - ETA: 53s - loss: 0.2553 - categorical_accuracy: 0.9213
31488/60000 [==============>...............] - ETA: 53s - loss: 0.2553 - categorical_accuracy: 0.9212
31520/60000 [==============>...............] - ETA: 53s - loss: 0.2552 - categorical_accuracy: 0.9213
31552/60000 [==============>...............] - ETA: 53s - loss: 0.2550 - categorical_accuracy: 0.9213
31584/60000 [==============>...............] - ETA: 53s - loss: 0.2549 - categorical_accuracy: 0.9214
31616/60000 [==============>...............] - ETA: 53s - loss: 0.2551 - categorical_accuracy: 0.9214
31648/60000 [==============>...............] - ETA: 53s - loss: 0.2550 - categorical_accuracy: 0.9214
31680/60000 [==============>...............] - ETA: 53s - loss: 0.2548 - categorical_accuracy: 0.9214
31712/60000 [==============>...............] - ETA: 53s - loss: 0.2546 - categorical_accuracy: 0.9215
31744/60000 [==============>...............] - ETA: 53s - loss: 0.2545 - categorical_accuracy: 0.9216
31776/60000 [==============>...............] - ETA: 53s - loss: 0.2543 - categorical_accuracy: 0.9216
31808/60000 [==============>...............] - ETA: 53s - loss: 0.2542 - categorical_accuracy: 0.9216
31840/60000 [==============>...............] - ETA: 53s - loss: 0.2541 - categorical_accuracy: 0.9215
31872/60000 [==============>...............] - ETA: 52s - loss: 0.2540 - categorical_accuracy: 0.9216
31904/60000 [==============>...............] - ETA: 52s - loss: 0.2541 - categorical_accuracy: 0.9216
31936/60000 [==============>...............] - ETA: 52s - loss: 0.2540 - categorical_accuracy: 0.9216
31968/60000 [==============>...............] - ETA: 52s - loss: 0.2538 - categorical_accuracy: 0.9216
32000/60000 [===============>..............] - ETA: 52s - loss: 0.2538 - categorical_accuracy: 0.9217
32032/60000 [===============>..............] - ETA: 52s - loss: 0.2536 - categorical_accuracy: 0.9217
32064/60000 [===============>..............] - ETA: 52s - loss: 0.2534 - categorical_accuracy: 0.9218
32096/60000 [===============>..............] - ETA: 52s - loss: 0.2532 - categorical_accuracy: 0.9218
32128/60000 [===============>..............] - ETA: 52s - loss: 0.2530 - categorical_accuracy: 0.9219
32160/60000 [===============>..............] - ETA: 52s - loss: 0.2528 - categorical_accuracy: 0.9219
32192/60000 [===============>..............] - ETA: 52s - loss: 0.2527 - categorical_accuracy: 0.9219
32224/60000 [===============>..............] - ETA: 52s - loss: 0.2525 - categorical_accuracy: 0.9220
32256/60000 [===============>..............] - ETA: 52s - loss: 0.2525 - categorical_accuracy: 0.9220
32288/60000 [===============>..............] - ETA: 52s - loss: 0.2524 - categorical_accuracy: 0.9220
32320/60000 [===============>..............] - ETA: 52s - loss: 0.2522 - categorical_accuracy: 0.9221
32352/60000 [===============>..............] - ETA: 52s - loss: 0.2520 - categorical_accuracy: 0.9221
32384/60000 [===============>..............] - ETA: 51s - loss: 0.2518 - categorical_accuracy: 0.9222
32416/60000 [===============>..............] - ETA: 51s - loss: 0.2517 - categorical_accuracy: 0.9222
32448/60000 [===============>..............] - ETA: 51s - loss: 0.2517 - categorical_accuracy: 0.9222
32480/60000 [===============>..............] - ETA: 51s - loss: 0.2515 - categorical_accuracy: 0.9223
32512/60000 [===============>..............] - ETA: 51s - loss: 0.2513 - categorical_accuracy: 0.9224
32544/60000 [===============>..............] - ETA: 51s - loss: 0.2511 - categorical_accuracy: 0.9224
32576/60000 [===============>..............] - ETA: 51s - loss: 0.2509 - categorical_accuracy: 0.9225
32608/60000 [===============>..............] - ETA: 51s - loss: 0.2507 - categorical_accuracy: 0.9226
32640/60000 [===============>..............] - ETA: 51s - loss: 0.2507 - categorical_accuracy: 0.9226
32672/60000 [===============>..............] - ETA: 51s - loss: 0.2505 - categorical_accuracy: 0.9226
32704/60000 [===============>..............] - ETA: 51s - loss: 0.2504 - categorical_accuracy: 0.9227
32736/60000 [===============>..............] - ETA: 51s - loss: 0.2502 - categorical_accuracy: 0.9227
32768/60000 [===============>..............] - ETA: 51s - loss: 0.2499 - categorical_accuracy: 0.9228
32800/60000 [===============>..............] - ETA: 51s - loss: 0.2497 - categorical_accuracy: 0.9229
32832/60000 [===============>..............] - ETA: 51s - loss: 0.2495 - categorical_accuracy: 0.9229
32864/60000 [===============>..............] - ETA: 51s - loss: 0.2493 - categorical_accuracy: 0.9230
32896/60000 [===============>..............] - ETA: 51s - loss: 0.2492 - categorical_accuracy: 0.9231
32928/60000 [===============>..............] - ETA: 50s - loss: 0.2491 - categorical_accuracy: 0.9231
32960/60000 [===============>..............] - ETA: 50s - loss: 0.2492 - categorical_accuracy: 0.9230
32992/60000 [===============>..............] - ETA: 50s - loss: 0.2491 - categorical_accuracy: 0.9230
33024/60000 [===============>..............] - ETA: 50s - loss: 0.2489 - categorical_accuracy: 0.9231
33056/60000 [===============>..............] - ETA: 50s - loss: 0.2487 - categorical_accuracy: 0.9232
33088/60000 [===============>..............] - ETA: 50s - loss: 0.2487 - categorical_accuracy: 0.9232
33120/60000 [===============>..............] - ETA: 50s - loss: 0.2485 - categorical_accuracy: 0.9232
33152/60000 [===============>..............] - ETA: 50s - loss: 0.2483 - categorical_accuracy: 0.9233
33184/60000 [===============>..............] - ETA: 50s - loss: 0.2481 - categorical_accuracy: 0.9233
33216/60000 [===============>..............] - ETA: 50s - loss: 0.2480 - categorical_accuracy: 0.9234
33248/60000 [===============>..............] - ETA: 50s - loss: 0.2481 - categorical_accuracy: 0.9233
33280/60000 [===============>..............] - ETA: 50s - loss: 0.2480 - categorical_accuracy: 0.9233
33312/60000 [===============>..............] - ETA: 50s - loss: 0.2478 - categorical_accuracy: 0.9234
33344/60000 [===============>..............] - ETA: 50s - loss: 0.2477 - categorical_accuracy: 0.9234
33376/60000 [===============>..............] - ETA: 50s - loss: 0.2476 - categorical_accuracy: 0.9235
33408/60000 [===============>..............] - ETA: 50s - loss: 0.2474 - categorical_accuracy: 0.9235
33440/60000 [===============>..............] - ETA: 49s - loss: 0.2472 - categorical_accuracy: 0.9236
33472/60000 [===============>..............] - ETA: 49s - loss: 0.2473 - categorical_accuracy: 0.9235
33504/60000 [===============>..............] - ETA: 49s - loss: 0.2471 - categorical_accuracy: 0.9236
33536/60000 [===============>..............] - ETA: 49s - loss: 0.2470 - categorical_accuracy: 0.9237
33568/60000 [===============>..............] - ETA: 49s - loss: 0.2468 - categorical_accuracy: 0.9237
33600/60000 [===============>..............] - ETA: 49s - loss: 0.2466 - categorical_accuracy: 0.9238
33632/60000 [===============>..............] - ETA: 49s - loss: 0.2464 - categorical_accuracy: 0.9239
33664/60000 [===============>..............] - ETA: 49s - loss: 0.2462 - categorical_accuracy: 0.9240
33696/60000 [===============>..............] - ETA: 49s - loss: 0.2460 - categorical_accuracy: 0.9240
33728/60000 [===============>..............] - ETA: 49s - loss: 0.2458 - categorical_accuracy: 0.9241
33760/60000 [===============>..............] - ETA: 49s - loss: 0.2455 - categorical_accuracy: 0.9241
33792/60000 [===============>..............] - ETA: 49s - loss: 0.2453 - categorical_accuracy: 0.9242
33824/60000 [===============>..............] - ETA: 49s - loss: 0.2452 - categorical_accuracy: 0.9242
33856/60000 [===============>..............] - ETA: 49s - loss: 0.2450 - categorical_accuracy: 0.9243
33888/60000 [===============>..............] - ETA: 49s - loss: 0.2449 - categorical_accuracy: 0.9243
33920/60000 [===============>..............] - ETA: 49s - loss: 0.2448 - categorical_accuracy: 0.9244
33952/60000 [===============>..............] - ETA: 49s - loss: 0.2448 - categorical_accuracy: 0.9244
33984/60000 [===============>..............] - ETA: 48s - loss: 0.2445 - categorical_accuracy: 0.9245
34016/60000 [================>.............] - ETA: 48s - loss: 0.2444 - categorical_accuracy: 0.9245
34048/60000 [================>.............] - ETA: 48s - loss: 0.2443 - categorical_accuracy: 0.9245
34080/60000 [================>.............] - ETA: 48s - loss: 0.2441 - categorical_accuracy: 0.9246
34112/60000 [================>.............] - ETA: 48s - loss: 0.2440 - categorical_accuracy: 0.9246
34144/60000 [================>.............] - ETA: 48s - loss: 0.2439 - categorical_accuracy: 0.9246
34176/60000 [================>.............] - ETA: 48s - loss: 0.2437 - categorical_accuracy: 0.9247
34208/60000 [================>.............] - ETA: 48s - loss: 0.2435 - categorical_accuracy: 0.9248
34240/60000 [================>.............] - ETA: 48s - loss: 0.2433 - categorical_accuracy: 0.9249
34272/60000 [================>.............] - ETA: 48s - loss: 0.2433 - categorical_accuracy: 0.9249
34304/60000 [================>.............] - ETA: 48s - loss: 0.2431 - categorical_accuracy: 0.9250
34336/60000 [================>.............] - ETA: 48s - loss: 0.2429 - categorical_accuracy: 0.9250
34368/60000 [================>.............] - ETA: 48s - loss: 0.2427 - categorical_accuracy: 0.9251
34400/60000 [================>.............] - ETA: 48s - loss: 0.2426 - categorical_accuracy: 0.9251
34432/60000 [================>.............] - ETA: 48s - loss: 0.2425 - categorical_accuracy: 0.9251
34464/60000 [================>.............] - ETA: 48s - loss: 0.2423 - categorical_accuracy: 0.9252
34496/60000 [================>.............] - ETA: 48s - loss: 0.2421 - categorical_accuracy: 0.9252
34528/60000 [================>.............] - ETA: 47s - loss: 0.2419 - categorical_accuracy: 0.9253
34560/60000 [================>.............] - ETA: 47s - loss: 0.2419 - categorical_accuracy: 0.9253
34592/60000 [================>.............] - ETA: 47s - loss: 0.2417 - categorical_accuracy: 0.9254
34624/60000 [================>.............] - ETA: 47s - loss: 0.2416 - categorical_accuracy: 0.9254
34656/60000 [================>.............] - ETA: 47s - loss: 0.2416 - categorical_accuracy: 0.9254
34688/60000 [================>.............] - ETA: 47s - loss: 0.2414 - categorical_accuracy: 0.9254
34720/60000 [================>.............] - ETA: 47s - loss: 0.2414 - categorical_accuracy: 0.9254
34752/60000 [================>.............] - ETA: 47s - loss: 0.2412 - categorical_accuracy: 0.9255
34784/60000 [================>.............] - ETA: 47s - loss: 0.2410 - categorical_accuracy: 0.9255
34816/60000 [================>.............] - ETA: 47s - loss: 0.2408 - categorical_accuracy: 0.9256
34848/60000 [================>.............] - ETA: 47s - loss: 0.2407 - categorical_accuracy: 0.9256
34880/60000 [================>.............] - ETA: 47s - loss: 0.2407 - categorical_accuracy: 0.9257
34912/60000 [================>.............] - ETA: 47s - loss: 0.2407 - categorical_accuracy: 0.9257
34944/60000 [================>.............] - ETA: 47s - loss: 0.2406 - categorical_accuracy: 0.9257
34976/60000 [================>.............] - ETA: 47s - loss: 0.2405 - categorical_accuracy: 0.9258
35008/60000 [================>.............] - ETA: 47s - loss: 0.2403 - categorical_accuracy: 0.9258
35040/60000 [================>.............] - ETA: 46s - loss: 0.2401 - categorical_accuracy: 0.9259
35072/60000 [================>.............] - ETA: 46s - loss: 0.2399 - categorical_accuracy: 0.9260
35104/60000 [================>.............] - ETA: 46s - loss: 0.2400 - categorical_accuracy: 0.9260
35136/60000 [================>.............] - ETA: 46s - loss: 0.2400 - categorical_accuracy: 0.9260
35168/60000 [================>.............] - ETA: 46s - loss: 0.2400 - categorical_accuracy: 0.9260
35200/60000 [================>.............] - ETA: 46s - loss: 0.2400 - categorical_accuracy: 0.9260
35232/60000 [================>.............] - ETA: 46s - loss: 0.2399 - categorical_accuracy: 0.9261
35264/60000 [================>.............] - ETA: 46s - loss: 0.2397 - categorical_accuracy: 0.9261
35296/60000 [================>.............] - ETA: 46s - loss: 0.2396 - categorical_accuracy: 0.9261
35328/60000 [================>.............] - ETA: 46s - loss: 0.2396 - categorical_accuracy: 0.9261
35360/60000 [================>.............] - ETA: 46s - loss: 0.2394 - categorical_accuracy: 0.9262
35392/60000 [================>.............] - ETA: 46s - loss: 0.2392 - categorical_accuracy: 0.9262
35424/60000 [================>.............] - ETA: 46s - loss: 0.2391 - categorical_accuracy: 0.9262
35456/60000 [================>.............] - ETA: 46s - loss: 0.2389 - categorical_accuracy: 0.9263
35488/60000 [================>.............] - ETA: 46s - loss: 0.2387 - categorical_accuracy: 0.9264
35520/60000 [================>.............] - ETA: 46s - loss: 0.2385 - categorical_accuracy: 0.9264
35552/60000 [================>.............] - ETA: 45s - loss: 0.2385 - categorical_accuracy: 0.9265
35584/60000 [================>.............] - ETA: 45s - loss: 0.2383 - categorical_accuracy: 0.9265
35616/60000 [================>.............] - ETA: 45s - loss: 0.2384 - categorical_accuracy: 0.9265
35648/60000 [================>.............] - ETA: 45s - loss: 0.2382 - categorical_accuracy: 0.9265
35680/60000 [================>.............] - ETA: 45s - loss: 0.2380 - categorical_accuracy: 0.9266
35712/60000 [================>.............] - ETA: 45s - loss: 0.2380 - categorical_accuracy: 0.9266
35744/60000 [================>.............] - ETA: 45s - loss: 0.2379 - categorical_accuracy: 0.9266
35776/60000 [================>.............] - ETA: 45s - loss: 0.2377 - categorical_accuracy: 0.9267
35808/60000 [================>.............] - ETA: 45s - loss: 0.2375 - categorical_accuracy: 0.9268
35840/60000 [================>.............] - ETA: 45s - loss: 0.2373 - categorical_accuracy: 0.9268
35872/60000 [================>.............] - ETA: 45s - loss: 0.2373 - categorical_accuracy: 0.9269
35904/60000 [================>.............] - ETA: 45s - loss: 0.2371 - categorical_accuracy: 0.9269
35936/60000 [================>.............] - ETA: 45s - loss: 0.2370 - categorical_accuracy: 0.9270
35968/60000 [================>.............] - ETA: 45s - loss: 0.2368 - categorical_accuracy: 0.9270
36000/60000 [=================>............] - ETA: 45s - loss: 0.2366 - categorical_accuracy: 0.9271
36032/60000 [=================>............] - ETA: 45s - loss: 0.2366 - categorical_accuracy: 0.9271
36064/60000 [=================>............] - ETA: 45s - loss: 0.2366 - categorical_accuracy: 0.9270
36096/60000 [=================>............] - ETA: 44s - loss: 0.2365 - categorical_accuracy: 0.9271
36128/60000 [=================>............] - ETA: 44s - loss: 0.2363 - categorical_accuracy: 0.9271
36160/60000 [=================>............] - ETA: 44s - loss: 0.2361 - categorical_accuracy: 0.9272
36192/60000 [=================>............] - ETA: 44s - loss: 0.2360 - categorical_accuracy: 0.9272
36224/60000 [=================>............] - ETA: 44s - loss: 0.2358 - categorical_accuracy: 0.9273
36256/60000 [=================>............] - ETA: 44s - loss: 0.2358 - categorical_accuracy: 0.9273
36288/60000 [=================>............] - ETA: 44s - loss: 0.2357 - categorical_accuracy: 0.9273
36320/60000 [=================>............] - ETA: 44s - loss: 0.2357 - categorical_accuracy: 0.9272
36352/60000 [=================>............] - ETA: 44s - loss: 0.2357 - categorical_accuracy: 0.9272
36384/60000 [=================>............] - ETA: 44s - loss: 0.2357 - categorical_accuracy: 0.9272
36416/60000 [=================>............] - ETA: 44s - loss: 0.2356 - categorical_accuracy: 0.9273
36448/60000 [=================>............] - ETA: 44s - loss: 0.2354 - categorical_accuracy: 0.9273
36480/60000 [=================>............] - ETA: 44s - loss: 0.2352 - categorical_accuracy: 0.9274
36512/60000 [=================>............] - ETA: 44s - loss: 0.2351 - categorical_accuracy: 0.9274
36544/60000 [=================>............] - ETA: 44s - loss: 0.2350 - categorical_accuracy: 0.9275
36576/60000 [=================>............] - ETA: 44s - loss: 0.2349 - categorical_accuracy: 0.9275
36608/60000 [=================>............] - ETA: 44s - loss: 0.2348 - categorical_accuracy: 0.9275
36640/60000 [=================>............] - ETA: 43s - loss: 0.2347 - categorical_accuracy: 0.9276
36672/60000 [=================>............] - ETA: 43s - loss: 0.2345 - categorical_accuracy: 0.9276
36704/60000 [=================>............] - ETA: 43s - loss: 0.2344 - categorical_accuracy: 0.9277
36736/60000 [=================>............] - ETA: 43s - loss: 0.2343 - categorical_accuracy: 0.9277
36768/60000 [=================>............] - ETA: 43s - loss: 0.2341 - categorical_accuracy: 0.9277
36800/60000 [=================>............] - ETA: 43s - loss: 0.2340 - categorical_accuracy: 0.9278
36832/60000 [=================>............] - ETA: 43s - loss: 0.2339 - categorical_accuracy: 0.9278
36864/60000 [=================>............] - ETA: 43s - loss: 0.2338 - categorical_accuracy: 0.9278
36896/60000 [=================>............] - ETA: 43s - loss: 0.2336 - categorical_accuracy: 0.9279
36928/60000 [=================>............] - ETA: 43s - loss: 0.2334 - categorical_accuracy: 0.9279
36960/60000 [=================>............] - ETA: 43s - loss: 0.2333 - categorical_accuracy: 0.9280
36992/60000 [=================>............] - ETA: 43s - loss: 0.2331 - categorical_accuracy: 0.9281
37024/60000 [=================>............] - ETA: 43s - loss: 0.2329 - categorical_accuracy: 0.9281
37056/60000 [=================>............] - ETA: 43s - loss: 0.2328 - categorical_accuracy: 0.9281
37088/60000 [=================>............] - ETA: 43s - loss: 0.2327 - categorical_accuracy: 0.9281
37120/60000 [=================>............] - ETA: 43s - loss: 0.2325 - categorical_accuracy: 0.9282
37152/60000 [=================>............] - ETA: 42s - loss: 0.2324 - categorical_accuracy: 0.9282
37184/60000 [=================>............] - ETA: 42s - loss: 0.2323 - categorical_accuracy: 0.9282
37216/60000 [=================>............] - ETA: 42s - loss: 0.2322 - categorical_accuracy: 0.9282
37248/60000 [=================>............] - ETA: 42s - loss: 0.2320 - categorical_accuracy: 0.9283
37280/60000 [=================>............] - ETA: 42s - loss: 0.2320 - categorical_accuracy: 0.9282
37312/60000 [=================>............] - ETA: 42s - loss: 0.2319 - categorical_accuracy: 0.9283
37344/60000 [=================>............] - ETA: 42s - loss: 0.2318 - categorical_accuracy: 0.9283
37376/60000 [=================>............] - ETA: 42s - loss: 0.2318 - categorical_accuracy: 0.9283
37408/60000 [=================>............] - ETA: 42s - loss: 0.2319 - categorical_accuracy: 0.9283
37440/60000 [=================>............] - ETA: 42s - loss: 0.2321 - categorical_accuracy: 0.9283
37472/60000 [=================>............] - ETA: 42s - loss: 0.2319 - categorical_accuracy: 0.9284
37504/60000 [=================>............] - ETA: 42s - loss: 0.2319 - categorical_accuracy: 0.9284
37536/60000 [=================>............] - ETA: 42s - loss: 0.2317 - categorical_accuracy: 0.9285
37568/60000 [=================>............] - ETA: 42s - loss: 0.2316 - categorical_accuracy: 0.9285
37600/60000 [=================>............] - ETA: 42s - loss: 0.2316 - categorical_accuracy: 0.9286
37632/60000 [=================>............] - ETA: 42s - loss: 0.2316 - categorical_accuracy: 0.9286
37664/60000 [=================>............] - ETA: 42s - loss: 0.2315 - categorical_accuracy: 0.9286
37696/60000 [=================>............] - ETA: 41s - loss: 0.2314 - categorical_accuracy: 0.9287
37728/60000 [=================>............] - ETA: 41s - loss: 0.2312 - categorical_accuracy: 0.9287
37760/60000 [=================>............] - ETA: 41s - loss: 0.2312 - categorical_accuracy: 0.9287
37792/60000 [=================>............] - ETA: 41s - loss: 0.2311 - categorical_accuracy: 0.9287
37824/60000 [=================>............] - ETA: 41s - loss: 0.2310 - categorical_accuracy: 0.9287
37856/60000 [=================>............] - ETA: 41s - loss: 0.2309 - categorical_accuracy: 0.9288
37888/60000 [=================>............] - ETA: 41s - loss: 0.2308 - categorical_accuracy: 0.9288
37920/60000 [=================>............] - ETA: 41s - loss: 0.2306 - categorical_accuracy: 0.9289
37952/60000 [=================>............] - ETA: 41s - loss: 0.2305 - categorical_accuracy: 0.9289
37984/60000 [=================>............] - ETA: 41s - loss: 0.2303 - categorical_accuracy: 0.9289
38016/60000 [==================>...........] - ETA: 41s - loss: 0.2302 - categorical_accuracy: 0.9290
38048/60000 [==================>...........] - ETA: 41s - loss: 0.2301 - categorical_accuracy: 0.9290
38080/60000 [==================>...........] - ETA: 41s - loss: 0.2299 - categorical_accuracy: 0.9291
38112/60000 [==================>...........] - ETA: 41s - loss: 0.2297 - categorical_accuracy: 0.9291
38144/60000 [==================>...........] - ETA: 41s - loss: 0.2296 - categorical_accuracy: 0.9292
38176/60000 [==================>...........] - ETA: 41s - loss: 0.2295 - categorical_accuracy: 0.9292
38208/60000 [==================>...........] - ETA: 41s - loss: 0.2294 - categorical_accuracy: 0.9292
38240/60000 [==================>...........] - ETA: 40s - loss: 0.2292 - categorical_accuracy: 0.9293
38272/60000 [==================>...........] - ETA: 40s - loss: 0.2291 - categorical_accuracy: 0.9293
38304/60000 [==================>...........] - ETA: 40s - loss: 0.2289 - categorical_accuracy: 0.9294
38336/60000 [==================>...........] - ETA: 40s - loss: 0.2288 - categorical_accuracy: 0.9294
38368/60000 [==================>...........] - ETA: 40s - loss: 0.2287 - categorical_accuracy: 0.9294
38400/60000 [==================>...........] - ETA: 40s - loss: 0.2286 - categorical_accuracy: 0.9294
38432/60000 [==================>...........] - ETA: 40s - loss: 0.2285 - categorical_accuracy: 0.9295
38464/60000 [==================>...........] - ETA: 40s - loss: 0.2283 - categorical_accuracy: 0.9295
38496/60000 [==================>...........] - ETA: 40s - loss: 0.2283 - categorical_accuracy: 0.9295
38528/60000 [==================>...........] - ETA: 40s - loss: 0.2281 - categorical_accuracy: 0.9296
38560/60000 [==================>...........] - ETA: 40s - loss: 0.2280 - categorical_accuracy: 0.9296
38592/60000 [==================>...........] - ETA: 40s - loss: 0.2280 - categorical_accuracy: 0.9296
38624/60000 [==================>...........] - ETA: 40s - loss: 0.2278 - categorical_accuracy: 0.9297
38656/60000 [==================>...........] - ETA: 40s - loss: 0.2276 - categorical_accuracy: 0.9297
38688/60000 [==================>...........] - ETA: 40s - loss: 0.2275 - categorical_accuracy: 0.9298
38720/60000 [==================>...........] - ETA: 40s - loss: 0.2275 - categorical_accuracy: 0.9298
38752/60000 [==================>...........] - ETA: 39s - loss: 0.2275 - categorical_accuracy: 0.9298
38784/60000 [==================>...........] - ETA: 39s - loss: 0.2273 - categorical_accuracy: 0.9299
38816/60000 [==================>...........] - ETA: 39s - loss: 0.2272 - categorical_accuracy: 0.9299
38848/60000 [==================>...........] - ETA: 39s - loss: 0.2274 - categorical_accuracy: 0.9299
38880/60000 [==================>...........] - ETA: 39s - loss: 0.2272 - categorical_accuracy: 0.9300
38912/60000 [==================>...........] - ETA: 39s - loss: 0.2272 - categorical_accuracy: 0.9300
38944/60000 [==================>...........] - ETA: 39s - loss: 0.2271 - categorical_accuracy: 0.9301
38976/60000 [==================>...........] - ETA: 39s - loss: 0.2269 - categorical_accuracy: 0.9301
39008/60000 [==================>...........] - ETA: 39s - loss: 0.2268 - categorical_accuracy: 0.9302
39040/60000 [==================>...........] - ETA: 39s - loss: 0.2268 - categorical_accuracy: 0.9302
39072/60000 [==================>...........] - ETA: 39s - loss: 0.2270 - categorical_accuracy: 0.9302
39104/60000 [==================>...........] - ETA: 39s - loss: 0.2268 - categorical_accuracy: 0.9302
39136/60000 [==================>...........] - ETA: 39s - loss: 0.2269 - categorical_accuracy: 0.9302
39168/60000 [==================>...........] - ETA: 39s - loss: 0.2268 - categorical_accuracy: 0.9302
39200/60000 [==================>...........] - ETA: 39s - loss: 0.2267 - categorical_accuracy: 0.9302
39232/60000 [==================>...........] - ETA: 39s - loss: 0.2267 - categorical_accuracy: 0.9302
39264/60000 [==================>...........] - ETA: 39s - loss: 0.2266 - categorical_accuracy: 0.9302
39296/60000 [==================>...........] - ETA: 38s - loss: 0.2266 - categorical_accuracy: 0.9302
39328/60000 [==================>...........] - ETA: 38s - loss: 0.2264 - categorical_accuracy: 0.9303
39360/60000 [==================>...........] - ETA: 38s - loss: 0.2264 - categorical_accuracy: 0.9303
39392/60000 [==================>...........] - ETA: 38s - loss: 0.2267 - categorical_accuracy: 0.9303
39424/60000 [==================>...........] - ETA: 38s - loss: 0.2268 - categorical_accuracy: 0.9302
39456/60000 [==================>...........] - ETA: 38s - loss: 0.2267 - categorical_accuracy: 0.9303
39488/60000 [==================>...........] - ETA: 38s - loss: 0.2266 - categorical_accuracy: 0.9303
39520/60000 [==================>...........] - ETA: 38s - loss: 0.2265 - categorical_accuracy: 0.9304
39552/60000 [==================>...........] - ETA: 38s - loss: 0.2265 - categorical_accuracy: 0.9304
39584/60000 [==================>...........] - ETA: 38s - loss: 0.2263 - categorical_accuracy: 0.9305
39616/60000 [==================>...........] - ETA: 38s - loss: 0.2261 - categorical_accuracy: 0.9305
39648/60000 [==================>...........] - ETA: 38s - loss: 0.2261 - categorical_accuracy: 0.9305
39680/60000 [==================>...........] - ETA: 38s - loss: 0.2259 - categorical_accuracy: 0.9306
39712/60000 [==================>...........] - ETA: 38s - loss: 0.2257 - categorical_accuracy: 0.9307
39744/60000 [==================>...........] - ETA: 38s - loss: 0.2255 - categorical_accuracy: 0.9307
39776/60000 [==================>...........] - ETA: 38s - loss: 0.2255 - categorical_accuracy: 0.9307
39808/60000 [==================>...........] - ETA: 37s - loss: 0.2254 - categorical_accuracy: 0.9307
39840/60000 [==================>...........] - ETA: 37s - loss: 0.2253 - categorical_accuracy: 0.9308
39872/60000 [==================>...........] - ETA: 37s - loss: 0.2252 - categorical_accuracy: 0.9308
39904/60000 [==================>...........] - ETA: 37s - loss: 0.2251 - categorical_accuracy: 0.9308
39936/60000 [==================>...........] - ETA: 37s - loss: 0.2250 - categorical_accuracy: 0.9309
39968/60000 [==================>...........] - ETA: 37s - loss: 0.2249 - categorical_accuracy: 0.9309
40000/60000 [===================>..........] - ETA: 37s - loss: 0.2249 - categorical_accuracy: 0.9309
40032/60000 [===================>..........] - ETA: 37s - loss: 0.2249 - categorical_accuracy: 0.9309
40064/60000 [===================>..........] - ETA: 37s - loss: 0.2248 - categorical_accuracy: 0.9310
40096/60000 [===================>..........] - ETA: 37s - loss: 0.2247 - categorical_accuracy: 0.9310
40128/60000 [===================>..........] - ETA: 37s - loss: 0.2245 - categorical_accuracy: 0.9310
40160/60000 [===================>..........] - ETA: 37s - loss: 0.2244 - categorical_accuracy: 0.9311
40192/60000 [===================>..........] - ETA: 37s - loss: 0.2243 - categorical_accuracy: 0.9311
40224/60000 [===================>..........] - ETA: 37s - loss: 0.2244 - categorical_accuracy: 0.9311
40256/60000 [===================>..........] - ETA: 37s - loss: 0.2242 - categorical_accuracy: 0.9311
40288/60000 [===================>..........] - ETA: 37s - loss: 0.2241 - categorical_accuracy: 0.9312
40320/60000 [===================>..........] - ETA: 37s - loss: 0.2240 - categorical_accuracy: 0.9312
40352/60000 [===================>..........] - ETA: 36s - loss: 0.2238 - categorical_accuracy: 0.9313
40384/60000 [===================>..........] - ETA: 36s - loss: 0.2237 - categorical_accuracy: 0.9313
40416/60000 [===================>..........] - ETA: 36s - loss: 0.2237 - categorical_accuracy: 0.9313
40448/60000 [===================>..........] - ETA: 36s - loss: 0.2235 - categorical_accuracy: 0.9314
40480/60000 [===================>..........] - ETA: 36s - loss: 0.2234 - categorical_accuracy: 0.9314
40512/60000 [===================>..........] - ETA: 36s - loss: 0.2232 - categorical_accuracy: 0.9315
40544/60000 [===================>..........] - ETA: 36s - loss: 0.2231 - categorical_accuracy: 0.9315
40576/60000 [===================>..........] - ETA: 36s - loss: 0.2229 - categorical_accuracy: 0.9316
40608/60000 [===================>..........] - ETA: 36s - loss: 0.2229 - categorical_accuracy: 0.9316
40640/60000 [===================>..........] - ETA: 36s - loss: 0.2227 - categorical_accuracy: 0.9316
40672/60000 [===================>..........] - ETA: 36s - loss: 0.2226 - categorical_accuracy: 0.9317
40704/60000 [===================>..........] - ETA: 36s - loss: 0.2225 - categorical_accuracy: 0.9317
40736/60000 [===================>..........] - ETA: 36s - loss: 0.2224 - categorical_accuracy: 0.9317
40768/60000 [===================>..........] - ETA: 36s - loss: 0.2224 - categorical_accuracy: 0.9318
40800/60000 [===================>..........] - ETA: 36s - loss: 0.2222 - categorical_accuracy: 0.9318
40832/60000 [===================>..........] - ETA: 36s - loss: 0.2225 - categorical_accuracy: 0.9318
40864/60000 [===================>..........] - ETA: 35s - loss: 0.2223 - categorical_accuracy: 0.9318
40896/60000 [===================>..........] - ETA: 35s - loss: 0.2222 - categorical_accuracy: 0.9319
40928/60000 [===================>..........] - ETA: 35s - loss: 0.2221 - categorical_accuracy: 0.9319
40960/60000 [===================>..........] - ETA: 35s - loss: 0.2219 - categorical_accuracy: 0.9320
40992/60000 [===================>..........] - ETA: 35s - loss: 0.2219 - categorical_accuracy: 0.9320
41024/60000 [===================>..........] - ETA: 35s - loss: 0.2220 - categorical_accuracy: 0.9320
41056/60000 [===================>..........] - ETA: 35s - loss: 0.2219 - categorical_accuracy: 0.9320
41088/60000 [===================>..........] - ETA: 35s - loss: 0.2217 - categorical_accuracy: 0.9321
41120/60000 [===================>..........] - ETA: 35s - loss: 0.2217 - categorical_accuracy: 0.9321
41152/60000 [===================>..........] - ETA: 35s - loss: 0.2215 - categorical_accuracy: 0.9322
41184/60000 [===================>..........] - ETA: 35s - loss: 0.2215 - categorical_accuracy: 0.9322
41216/60000 [===================>..........] - ETA: 35s - loss: 0.2215 - categorical_accuracy: 0.9322
41248/60000 [===================>..........] - ETA: 35s - loss: 0.2214 - categorical_accuracy: 0.9322
41280/60000 [===================>..........] - ETA: 35s - loss: 0.2213 - categorical_accuracy: 0.9322
41312/60000 [===================>..........] - ETA: 35s - loss: 0.2213 - categorical_accuracy: 0.9322
41344/60000 [===================>..........] - ETA: 35s - loss: 0.2213 - categorical_accuracy: 0.9322
41376/60000 [===================>..........] - ETA: 35s - loss: 0.2212 - categorical_accuracy: 0.9323
41408/60000 [===================>..........] - ETA: 34s - loss: 0.2211 - categorical_accuracy: 0.9323
41440/60000 [===================>..........] - ETA: 34s - loss: 0.2210 - categorical_accuracy: 0.9323
41472/60000 [===================>..........] - ETA: 34s - loss: 0.2209 - categorical_accuracy: 0.9323
41504/60000 [===================>..........] - ETA: 34s - loss: 0.2208 - categorical_accuracy: 0.9324
41536/60000 [===================>..........] - ETA: 34s - loss: 0.2207 - categorical_accuracy: 0.9324
41568/60000 [===================>..........] - ETA: 34s - loss: 0.2205 - categorical_accuracy: 0.9325
41600/60000 [===================>..........] - ETA: 34s - loss: 0.2205 - categorical_accuracy: 0.9325
41632/60000 [===================>..........] - ETA: 34s - loss: 0.2204 - categorical_accuracy: 0.9326
41664/60000 [===================>..........] - ETA: 34s - loss: 0.2202 - categorical_accuracy: 0.9326
41696/60000 [===================>..........] - ETA: 34s - loss: 0.2201 - categorical_accuracy: 0.9327
41728/60000 [===================>..........] - ETA: 34s - loss: 0.2200 - categorical_accuracy: 0.9327
41760/60000 [===================>..........] - ETA: 34s - loss: 0.2200 - categorical_accuracy: 0.9327
41792/60000 [===================>..........] - ETA: 34s - loss: 0.2198 - categorical_accuracy: 0.9328
41824/60000 [===================>..........] - ETA: 34s - loss: 0.2198 - categorical_accuracy: 0.9327
41856/60000 [===================>..........] - ETA: 34s - loss: 0.2198 - categorical_accuracy: 0.9327
41888/60000 [===================>..........] - ETA: 34s - loss: 0.2197 - categorical_accuracy: 0.9328
41920/60000 [===================>..........] - ETA: 34s - loss: 0.2198 - categorical_accuracy: 0.9328
41952/60000 [===================>..........] - ETA: 33s - loss: 0.2197 - categorical_accuracy: 0.9328
41984/60000 [===================>..........] - ETA: 33s - loss: 0.2196 - categorical_accuracy: 0.9328
42016/60000 [====================>.........] - ETA: 33s - loss: 0.2195 - categorical_accuracy: 0.9328
42048/60000 [====================>.........] - ETA: 33s - loss: 0.2194 - categorical_accuracy: 0.9329
42080/60000 [====================>.........] - ETA: 33s - loss: 0.2195 - categorical_accuracy: 0.9329
42112/60000 [====================>.........] - ETA: 33s - loss: 0.2194 - categorical_accuracy: 0.9329
42144/60000 [====================>.........] - ETA: 33s - loss: 0.2192 - categorical_accuracy: 0.9330
42176/60000 [====================>.........] - ETA: 33s - loss: 0.2191 - categorical_accuracy: 0.9330
42208/60000 [====================>.........] - ETA: 33s - loss: 0.2190 - categorical_accuracy: 0.9330
42240/60000 [====================>.........] - ETA: 33s - loss: 0.2190 - categorical_accuracy: 0.9330
42272/60000 [====================>.........] - ETA: 33s - loss: 0.2188 - categorical_accuracy: 0.9331
42304/60000 [====================>.........] - ETA: 33s - loss: 0.2187 - categorical_accuracy: 0.9332
42336/60000 [====================>.........] - ETA: 33s - loss: 0.2185 - categorical_accuracy: 0.9332
42368/60000 [====================>.........] - ETA: 33s - loss: 0.2185 - categorical_accuracy: 0.9332
42400/60000 [====================>.........] - ETA: 33s - loss: 0.2186 - categorical_accuracy: 0.9332
42432/60000 [====================>.........] - ETA: 33s - loss: 0.2185 - categorical_accuracy: 0.9332
42464/60000 [====================>.........] - ETA: 32s - loss: 0.2184 - categorical_accuracy: 0.9332
42496/60000 [====================>.........] - ETA: 32s - loss: 0.2184 - categorical_accuracy: 0.9332
42528/60000 [====================>.........] - ETA: 32s - loss: 0.2184 - categorical_accuracy: 0.9332
42560/60000 [====================>.........] - ETA: 32s - loss: 0.2183 - categorical_accuracy: 0.9333
42592/60000 [====================>.........] - ETA: 32s - loss: 0.2182 - categorical_accuracy: 0.9333
42624/60000 [====================>.........] - ETA: 32s - loss: 0.2181 - categorical_accuracy: 0.9333
42656/60000 [====================>.........] - ETA: 32s - loss: 0.2180 - categorical_accuracy: 0.9334
42688/60000 [====================>.........] - ETA: 32s - loss: 0.2180 - categorical_accuracy: 0.9334
42720/60000 [====================>.........] - ETA: 32s - loss: 0.2179 - categorical_accuracy: 0.9334
42752/60000 [====================>.........] - ETA: 32s - loss: 0.2178 - categorical_accuracy: 0.9335
42784/60000 [====================>.........] - ETA: 32s - loss: 0.2179 - categorical_accuracy: 0.9334
42816/60000 [====================>.........] - ETA: 32s - loss: 0.2177 - categorical_accuracy: 0.9334
42848/60000 [====================>.........] - ETA: 32s - loss: 0.2176 - categorical_accuracy: 0.9335
42880/60000 [====================>.........] - ETA: 32s - loss: 0.2175 - categorical_accuracy: 0.9335
42912/60000 [====================>.........] - ETA: 32s - loss: 0.2174 - categorical_accuracy: 0.9336
42944/60000 [====================>.........] - ETA: 32s - loss: 0.2173 - categorical_accuracy: 0.9336
42976/60000 [====================>.........] - ETA: 32s - loss: 0.2172 - categorical_accuracy: 0.9336
43008/60000 [====================>.........] - ETA: 31s - loss: 0.2172 - categorical_accuracy: 0.9336
43040/60000 [====================>.........] - ETA: 31s - loss: 0.2171 - categorical_accuracy: 0.9336
43072/60000 [====================>.........] - ETA: 31s - loss: 0.2171 - categorical_accuracy: 0.9336
43104/60000 [====================>.........] - ETA: 31s - loss: 0.2170 - categorical_accuracy: 0.9336
43136/60000 [====================>.........] - ETA: 31s - loss: 0.2170 - categorical_accuracy: 0.9336
43168/60000 [====================>.........] - ETA: 31s - loss: 0.2169 - categorical_accuracy: 0.9337
43200/60000 [====================>.........] - ETA: 31s - loss: 0.2168 - categorical_accuracy: 0.9337
43232/60000 [====================>.........] - ETA: 31s - loss: 0.2167 - categorical_accuracy: 0.9337
43264/60000 [====================>.........] - ETA: 31s - loss: 0.2166 - categorical_accuracy: 0.9337
43296/60000 [====================>.........] - ETA: 31s - loss: 0.2164 - categorical_accuracy: 0.9338
43328/60000 [====================>.........] - ETA: 31s - loss: 0.2163 - categorical_accuracy: 0.9338
43360/60000 [====================>.........] - ETA: 31s - loss: 0.2162 - categorical_accuracy: 0.9338
43392/60000 [====================>.........] - ETA: 31s - loss: 0.2161 - categorical_accuracy: 0.9339
43424/60000 [====================>.........] - ETA: 31s - loss: 0.2159 - categorical_accuracy: 0.9339
43456/60000 [====================>.........] - ETA: 31s - loss: 0.2159 - categorical_accuracy: 0.9339
43488/60000 [====================>.........] - ETA: 31s - loss: 0.2158 - categorical_accuracy: 0.9340
43520/60000 [====================>.........] - ETA: 30s - loss: 0.2157 - categorical_accuracy: 0.9340
43552/60000 [====================>.........] - ETA: 30s - loss: 0.2157 - categorical_accuracy: 0.9340
43584/60000 [====================>.........] - ETA: 30s - loss: 0.2156 - categorical_accuracy: 0.9340
43616/60000 [====================>.........] - ETA: 30s - loss: 0.2155 - categorical_accuracy: 0.9340
43648/60000 [====================>.........] - ETA: 30s - loss: 0.2156 - categorical_accuracy: 0.9341
43680/60000 [====================>.........] - ETA: 30s - loss: 0.2154 - categorical_accuracy: 0.9341
43712/60000 [====================>.........] - ETA: 30s - loss: 0.2155 - categorical_accuracy: 0.9341
43744/60000 [====================>.........] - ETA: 30s - loss: 0.2154 - categorical_accuracy: 0.9341
43776/60000 [====================>.........] - ETA: 30s - loss: 0.2154 - categorical_accuracy: 0.9341
43808/60000 [====================>.........] - ETA: 30s - loss: 0.2152 - categorical_accuracy: 0.9342
43840/60000 [====================>.........] - ETA: 30s - loss: 0.2151 - categorical_accuracy: 0.9342
43872/60000 [====================>.........] - ETA: 30s - loss: 0.2150 - categorical_accuracy: 0.9343
43904/60000 [====================>.........] - ETA: 30s - loss: 0.2149 - categorical_accuracy: 0.9343
43936/60000 [====================>.........] - ETA: 30s - loss: 0.2147 - categorical_accuracy: 0.9343
43968/60000 [====================>.........] - ETA: 30s - loss: 0.2146 - categorical_accuracy: 0.9343
44000/60000 [=====================>........] - ETA: 30s - loss: 0.2146 - categorical_accuracy: 0.9343
44032/60000 [=====================>........] - ETA: 30s - loss: 0.2145 - categorical_accuracy: 0.9344
44064/60000 [=====================>........] - ETA: 29s - loss: 0.2145 - categorical_accuracy: 0.9344
44096/60000 [=====================>........] - ETA: 29s - loss: 0.2144 - categorical_accuracy: 0.9344
44128/60000 [=====================>........] - ETA: 29s - loss: 0.2143 - categorical_accuracy: 0.9345
44160/60000 [=====================>........] - ETA: 29s - loss: 0.2141 - categorical_accuracy: 0.9345
44192/60000 [=====================>........] - ETA: 29s - loss: 0.2142 - categorical_accuracy: 0.9345
44224/60000 [=====================>........] - ETA: 29s - loss: 0.2141 - categorical_accuracy: 0.9345
44256/60000 [=====================>........] - ETA: 29s - loss: 0.2141 - categorical_accuracy: 0.9345
44288/60000 [=====================>........] - ETA: 29s - loss: 0.2141 - categorical_accuracy: 0.9345
44320/60000 [=====================>........] - ETA: 29s - loss: 0.2141 - categorical_accuracy: 0.9345
44352/60000 [=====================>........] - ETA: 29s - loss: 0.2140 - categorical_accuracy: 0.9345
44384/60000 [=====================>........] - ETA: 29s - loss: 0.2140 - categorical_accuracy: 0.9345
44416/60000 [=====================>........] - ETA: 29s - loss: 0.2138 - categorical_accuracy: 0.9346
44448/60000 [=====================>........] - ETA: 29s - loss: 0.2138 - categorical_accuracy: 0.9346
44480/60000 [=====================>........] - ETA: 29s - loss: 0.2136 - categorical_accuracy: 0.9346
44512/60000 [=====================>........] - ETA: 29s - loss: 0.2136 - categorical_accuracy: 0.9346
44544/60000 [=====================>........] - ETA: 29s - loss: 0.2136 - categorical_accuracy: 0.9346
44576/60000 [=====================>........] - ETA: 28s - loss: 0.2135 - categorical_accuracy: 0.9347
44608/60000 [=====================>........] - ETA: 28s - loss: 0.2134 - categorical_accuracy: 0.9347
44640/60000 [=====================>........] - ETA: 28s - loss: 0.2133 - categorical_accuracy: 0.9348
44672/60000 [=====================>........] - ETA: 28s - loss: 0.2132 - categorical_accuracy: 0.9348
44704/60000 [=====================>........] - ETA: 28s - loss: 0.2130 - categorical_accuracy: 0.9348
44736/60000 [=====================>........] - ETA: 28s - loss: 0.2129 - categorical_accuracy: 0.9349
44768/60000 [=====================>........] - ETA: 28s - loss: 0.2128 - categorical_accuracy: 0.9349
44800/60000 [=====================>........] - ETA: 28s - loss: 0.2127 - categorical_accuracy: 0.9350
44832/60000 [=====================>........] - ETA: 28s - loss: 0.2128 - categorical_accuracy: 0.9350
44864/60000 [=====================>........] - ETA: 28s - loss: 0.2127 - categorical_accuracy: 0.9350
44896/60000 [=====================>........] - ETA: 28s - loss: 0.2126 - categorical_accuracy: 0.9350
44928/60000 [=====================>........] - ETA: 28s - loss: 0.2124 - categorical_accuracy: 0.9351
44960/60000 [=====================>........] - ETA: 28s - loss: 0.2125 - categorical_accuracy: 0.9351
44992/60000 [=====================>........] - ETA: 28s - loss: 0.2125 - categorical_accuracy: 0.9351
45024/60000 [=====================>........] - ETA: 28s - loss: 0.2124 - categorical_accuracy: 0.9351
45056/60000 [=====================>........] - ETA: 28s - loss: 0.2122 - categorical_accuracy: 0.9351
45088/60000 [=====================>........] - ETA: 28s - loss: 0.2121 - categorical_accuracy: 0.9352
45120/60000 [=====================>........] - ETA: 27s - loss: 0.2120 - categorical_accuracy: 0.9352
45152/60000 [=====================>........] - ETA: 27s - loss: 0.2120 - categorical_accuracy: 0.9353
45184/60000 [=====================>........] - ETA: 27s - loss: 0.2119 - categorical_accuracy: 0.9353
45216/60000 [=====================>........] - ETA: 27s - loss: 0.2118 - categorical_accuracy: 0.9353
45248/60000 [=====================>........] - ETA: 27s - loss: 0.2117 - categorical_accuracy: 0.9354
45280/60000 [=====================>........] - ETA: 27s - loss: 0.2116 - categorical_accuracy: 0.9354
45312/60000 [=====================>........] - ETA: 27s - loss: 0.2115 - categorical_accuracy: 0.9354
45344/60000 [=====================>........] - ETA: 27s - loss: 0.2113 - categorical_accuracy: 0.9355
45376/60000 [=====================>........] - ETA: 27s - loss: 0.2113 - categorical_accuracy: 0.9355
45408/60000 [=====================>........] - ETA: 27s - loss: 0.2113 - categorical_accuracy: 0.9355
45440/60000 [=====================>........] - ETA: 27s - loss: 0.2112 - categorical_accuracy: 0.9356
45472/60000 [=====================>........] - ETA: 27s - loss: 0.2110 - categorical_accuracy: 0.9356
45504/60000 [=====================>........] - ETA: 27s - loss: 0.2109 - categorical_accuracy: 0.9357
45536/60000 [=====================>........] - ETA: 27s - loss: 0.2108 - categorical_accuracy: 0.9357
45568/60000 [=====================>........] - ETA: 27s - loss: 0.2106 - categorical_accuracy: 0.9358
45600/60000 [=====================>........] - ETA: 27s - loss: 0.2107 - categorical_accuracy: 0.9358
45632/60000 [=====================>........] - ETA: 26s - loss: 0.2105 - categorical_accuracy: 0.9358
45664/60000 [=====================>........] - ETA: 26s - loss: 0.2105 - categorical_accuracy: 0.9359
45696/60000 [=====================>........] - ETA: 26s - loss: 0.2103 - categorical_accuracy: 0.9359
45728/60000 [=====================>........] - ETA: 26s - loss: 0.2102 - categorical_accuracy: 0.9359
45760/60000 [=====================>........] - ETA: 26s - loss: 0.2102 - categorical_accuracy: 0.9360
45792/60000 [=====================>........] - ETA: 26s - loss: 0.2101 - categorical_accuracy: 0.9360
45824/60000 [=====================>........] - ETA: 26s - loss: 0.2100 - categorical_accuracy: 0.9360
45856/60000 [=====================>........] - ETA: 26s - loss: 0.2099 - categorical_accuracy: 0.9360
45888/60000 [=====================>........] - ETA: 26s - loss: 0.2100 - categorical_accuracy: 0.9360
45920/60000 [=====================>........] - ETA: 26s - loss: 0.2099 - categorical_accuracy: 0.9360
45952/60000 [=====================>........] - ETA: 26s - loss: 0.2098 - categorical_accuracy: 0.9360
45984/60000 [=====================>........] - ETA: 26s - loss: 0.2098 - categorical_accuracy: 0.9360
46016/60000 [======================>.......] - ETA: 26s - loss: 0.2098 - categorical_accuracy: 0.9360
46048/60000 [======================>.......] - ETA: 26s - loss: 0.2098 - categorical_accuracy: 0.9361
46080/60000 [======================>.......] - ETA: 26s - loss: 0.2096 - categorical_accuracy: 0.9361
46112/60000 [======================>.......] - ETA: 26s - loss: 0.2095 - categorical_accuracy: 0.9361
46144/60000 [======================>.......] - ETA: 26s - loss: 0.2094 - categorical_accuracy: 0.9362
46176/60000 [======================>.......] - ETA: 25s - loss: 0.2093 - categorical_accuracy: 0.9362
46208/60000 [======================>.......] - ETA: 25s - loss: 0.2092 - categorical_accuracy: 0.9362
46240/60000 [======================>.......] - ETA: 25s - loss: 0.2091 - categorical_accuracy: 0.9363
46272/60000 [======================>.......] - ETA: 25s - loss: 0.2090 - categorical_accuracy: 0.9363
46304/60000 [======================>.......] - ETA: 25s - loss: 0.2089 - categorical_accuracy: 0.9363
46336/60000 [======================>.......] - ETA: 25s - loss: 0.2089 - categorical_accuracy: 0.9363
46368/60000 [======================>.......] - ETA: 25s - loss: 0.2088 - categorical_accuracy: 0.9363
46400/60000 [======================>.......] - ETA: 25s - loss: 0.2087 - categorical_accuracy: 0.9364
46432/60000 [======================>.......] - ETA: 25s - loss: 0.2087 - categorical_accuracy: 0.9363
46464/60000 [======================>.......] - ETA: 25s - loss: 0.2086 - categorical_accuracy: 0.9364
46496/60000 [======================>.......] - ETA: 25s - loss: 0.2085 - categorical_accuracy: 0.9364
46528/60000 [======================>.......] - ETA: 25s - loss: 0.2085 - categorical_accuracy: 0.9364
46560/60000 [======================>.......] - ETA: 25s - loss: 0.2086 - categorical_accuracy: 0.9364
46592/60000 [======================>.......] - ETA: 25s - loss: 0.2084 - categorical_accuracy: 0.9364
46624/60000 [======================>.......] - ETA: 25s - loss: 0.2084 - categorical_accuracy: 0.9364
46656/60000 [======================>.......] - ETA: 25s - loss: 0.2083 - categorical_accuracy: 0.9365
46688/60000 [======================>.......] - ETA: 25s - loss: 0.2083 - categorical_accuracy: 0.9365
46720/60000 [======================>.......] - ETA: 24s - loss: 0.2082 - categorical_accuracy: 0.9365
46752/60000 [======================>.......] - ETA: 24s - loss: 0.2081 - categorical_accuracy: 0.9366
46784/60000 [======================>.......] - ETA: 24s - loss: 0.2080 - categorical_accuracy: 0.9366
46816/60000 [======================>.......] - ETA: 24s - loss: 0.2081 - categorical_accuracy: 0.9366
46848/60000 [======================>.......] - ETA: 24s - loss: 0.2081 - categorical_accuracy: 0.9366
46880/60000 [======================>.......] - ETA: 24s - loss: 0.2080 - categorical_accuracy: 0.9366
46912/60000 [======================>.......] - ETA: 24s - loss: 0.2079 - categorical_accuracy: 0.9366
46944/60000 [======================>.......] - ETA: 24s - loss: 0.2078 - categorical_accuracy: 0.9366
46976/60000 [======================>.......] - ETA: 24s - loss: 0.2077 - categorical_accuracy: 0.9367
47008/60000 [======================>.......] - ETA: 24s - loss: 0.2076 - categorical_accuracy: 0.9367
47040/60000 [======================>.......] - ETA: 24s - loss: 0.2075 - categorical_accuracy: 0.9367
47072/60000 [======================>.......] - ETA: 24s - loss: 0.2074 - categorical_accuracy: 0.9367
47104/60000 [======================>.......] - ETA: 24s - loss: 0.2073 - categorical_accuracy: 0.9368
47136/60000 [======================>.......] - ETA: 24s - loss: 0.2073 - categorical_accuracy: 0.9368
47168/60000 [======================>.......] - ETA: 24s - loss: 0.2072 - categorical_accuracy: 0.9368
47200/60000 [======================>.......] - ETA: 24s - loss: 0.2071 - categorical_accuracy: 0.9368
47232/60000 [======================>.......] - ETA: 23s - loss: 0.2071 - categorical_accuracy: 0.9368
47264/60000 [======================>.......] - ETA: 23s - loss: 0.2070 - categorical_accuracy: 0.9369
47296/60000 [======================>.......] - ETA: 23s - loss: 0.2069 - categorical_accuracy: 0.9369
47328/60000 [======================>.......] - ETA: 23s - loss: 0.2069 - categorical_accuracy: 0.9369
47360/60000 [======================>.......] - ETA: 23s - loss: 0.2068 - categorical_accuracy: 0.9369
47392/60000 [======================>.......] - ETA: 23s - loss: 0.2067 - categorical_accuracy: 0.9369
47424/60000 [======================>.......] - ETA: 23s - loss: 0.2066 - categorical_accuracy: 0.9369
47456/60000 [======================>.......] - ETA: 23s - loss: 0.2065 - categorical_accuracy: 0.9370
47488/60000 [======================>.......] - ETA: 23s - loss: 0.2065 - categorical_accuracy: 0.9370
47520/60000 [======================>.......] - ETA: 23s - loss: 0.2064 - categorical_accuracy: 0.9370
47552/60000 [======================>.......] - ETA: 23s - loss: 0.2063 - categorical_accuracy: 0.9370
47584/60000 [======================>.......] - ETA: 23s - loss: 0.2063 - categorical_accuracy: 0.9370
47616/60000 [======================>.......] - ETA: 23s - loss: 0.2062 - categorical_accuracy: 0.9371
47648/60000 [======================>.......] - ETA: 23s - loss: 0.2062 - categorical_accuracy: 0.9371
47680/60000 [======================>.......] - ETA: 23s - loss: 0.2060 - categorical_accuracy: 0.9371
47712/60000 [======================>.......] - ETA: 23s - loss: 0.2059 - categorical_accuracy: 0.9371
47744/60000 [======================>.......] - ETA: 23s - loss: 0.2059 - categorical_accuracy: 0.9371
47776/60000 [======================>.......] - ETA: 22s - loss: 0.2059 - categorical_accuracy: 0.9371
47808/60000 [======================>.......] - ETA: 22s - loss: 0.2058 - categorical_accuracy: 0.9371
47840/60000 [======================>.......] - ETA: 22s - loss: 0.2057 - categorical_accuracy: 0.9372
47872/60000 [======================>.......] - ETA: 22s - loss: 0.2056 - categorical_accuracy: 0.9372
47904/60000 [======================>.......] - ETA: 22s - loss: 0.2055 - categorical_accuracy: 0.9372
47936/60000 [======================>.......] - ETA: 22s - loss: 0.2054 - categorical_accuracy: 0.9372
47968/60000 [======================>.......] - ETA: 22s - loss: 0.2053 - categorical_accuracy: 0.9373
48000/60000 [=======================>......] - ETA: 22s - loss: 0.2052 - categorical_accuracy: 0.9373
48032/60000 [=======================>......] - ETA: 22s - loss: 0.2050 - categorical_accuracy: 0.9374
48064/60000 [=======================>......] - ETA: 22s - loss: 0.2050 - categorical_accuracy: 0.9374
48096/60000 [=======================>......] - ETA: 22s - loss: 0.2050 - categorical_accuracy: 0.9373
48128/60000 [=======================>......] - ETA: 22s - loss: 0.2049 - categorical_accuracy: 0.9374
48160/60000 [=======================>......] - ETA: 22s - loss: 0.2048 - categorical_accuracy: 0.9374
48192/60000 [=======================>......] - ETA: 22s - loss: 0.2047 - categorical_accuracy: 0.9374
48224/60000 [=======================>......] - ETA: 22s - loss: 0.2048 - categorical_accuracy: 0.9374
48256/60000 [=======================>......] - ETA: 22s - loss: 0.2051 - categorical_accuracy: 0.9373
48288/60000 [=======================>......] - ETA: 22s - loss: 0.2051 - categorical_accuracy: 0.9374
48320/60000 [=======================>......] - ETA: 21s - loss: 0.2050 - categorical_accuracy: 0.9374
48352/60000 [=======================>......] - ETA: 21s - loss: 0.2049 - categorical_accuracy: 0.9374
48384/60000 [=======================>......] - ETA: 21s - loss: 0.2048 - categorical_accuracy: 0.9374
48416/60000 [=======================>......] - ETA: 21s - loss: 0.2048 - categorical_accuracy: 0.9374
48448/60000 [=======================>......] - ETA: 21s - loss: 0.2047 - categorical_accuracy: 0.9375
48480/60000 [=======================>......] - ETA: 21s - loss: 0.2047 - categorical_accuracy: 0.9375
48512/60000 [=======================>......] - ETA: 21s - loss: 0.2046 - categorical_accuracy: 0.9375
48544/60000 [=======================>......] - ETA: 21s - loss: 0.2045 - categorical_accuracy: 0.9376
48576/60000 [=======================>......] - ETA: 21s - loss: 0.2045 - categorical_accuracy: 0.9375
48608/60000 [=======================>......] - ETA: 21s - loss: 0.2044 - categorical_accuracy: 0.9376
48640/60000 [=======================>......] - ETA: 21s - loss: 0.2043 - categorical_accuracy: 0.9376
48672/60000 [=======================>......] - ETA: 21s - loss: 0.2043 - categorical_accuracy: 0.9376
48704/60000 [=======================>......] - ETA: 21s - loss: 0.2042 - categorical_accuracy: 0.9376
48736/60000 [=======================>......] - ETA: 21s - loss: 0.2041 - categorical_accuracy: 0.9377
48768/60000 [=======================>......] - ETA: 21s - loss: 0.2041 - categorical_accuracy: 0.9377
48800/60000 [=======================>......] - ETA: 21s - loss: 0.2040 - categorical_accuracy: 0.9377
48832/60000 [=======================>......] - ETA: 20s - loss: 0.2039 - categorical_accuracy: 0.9377
48864/60000 [=======================>......] - ETA: 20s - loss: 0.2038 - categorical_accuracy: 0.9378
48896/60000 [=======================>......] - ETA: 20s - loss: 0.2039 - categorical_accuracy: 0.9377
48928/60000 [=======================>......] - ETA: 20s - loss: 0.2037 - categorical_accuracy: 0.9378
48960/60000 [=======================>......] - ETA: 20s - loss: 0.2037 - categorical_accuracy: 0.9378
48992/60000 [=======================>......] - ETA: 20s - loss: 0.2036 - categorical_accuracy: 0.9378
49024/60000 [=======================>......] - ETA: 20s - loss: 0.2035 - categorical_accuracy: 0.9378
49056/60000 [=======================>......] - ETA: 20s - loss: 0.2037 - categorical_accuracy: 0.9379
49088/60000 [=======================>......] - ETA: 20s - loss: 0.2037 - categorical_accuracy: 0.9379
49120/60000 [=======================>......] - ETA: 20s - loss: 0.2036 - categorical_accuracy: 0.9379
49152/60000 [=======================>......] - ETA: 20s - loss: 0.2035 - categorical_accuracy: 0.9379
49184/60000 [=======================>......] - ETA: 20s - loss: 0.2034 - categorical_accuracy: 0.9380
49216/60000 [=======================>......] - ETA: 20s - loss: 0.2033 - categorical_accuracy: 0.9380
49248/60000 [=======================>......] - ETA: 20s - loss: 0.2032 - categorical_accuracy: 0.9380
49280/60000 [=======================>......] - ETA: 20s - loss: 0.2031 - categorical_accuracy: 0.9381
49312/60000 [=======================>......] - ETA: 20s - loss: 0.2030 - categorical_accuracy: 0.9381
49344/60000 [=======================>......] - ETA: 20s - loss: 0.2029 - categorical_accuracy: 0.9381
49376/60000 [=======================>......] - ETA: 19s - loss: 0.2028 - categorical_accuracy: 0.9381
49408/60000 [=======================>......] - ETA: 19s - loss: 0.2027 - categorical_accuracy: 0.9382
49440/60000 [=======================>......] - ETA: 19s - loss: 0.2026 - categorical_accuracy: 0.9382
49472/60000 [=======================>......] - ETA: 19s - loss: 0.2025 - categorical_accuracy: 0.9382
49504/60000 [=======================>......] - ETA: 19s - loss: 0.2024 - categorical_accuracy: 0.9383
49536/60000 [=======================>......] - ETA: 19s - loss: 0.2023 - categorical_accuracy: 0.9383
49568/60000 [=======================>......] - ETA: 19s - loss: 0.2022 - categorical_accuracy: 0.9383
49600/60000 [=======================>......] - ETA: 19s - loss: 0.2021 - categorical_accuracy: 0.9384
49632/60000 [=======================>......] - ETA: 19s - loss: 0.2022 - categorical_accuracy: 0.9384
49664/60000 [=======================>......] - ETA: 19s - loss: 0.2022 - categorical_accuracy: 0.9384
49696/60000 [=======================>......] - ETA: 19s - loss: 0.2021 - categorical_accuracy: 0.9384
49728/60000 [=======================>......] - ETA: 19s - loss: 0.2021 - categorical_accuracy: 0.9384
49760/60000 [=======================>......] - ETA: 19s - loss: 0.2021 - categorical_accuracy: 0.9384
49792/60000 [=======================>......] - ETA: 19s - loss: 0.2020 - categorical_accuracy: 0.9385
49824/60000 [=======================>......] - ETA: 19s - loss: 0.2022 - categorical_accuracy: 0.9384
49856/60000 [=======================>......] - ETA: 19s - loss: 0.2021 - categorical_accuracy: 0.9385
49888/60000 [=======================>......] - ETA: 19s - loss: 0.2021 - categorical_accuracy: 0.9385
49920/60000 [=======================>......] - ETA: 18s - loss: 0.2020 - categorical_accuracy: 0.9385
49952/60000 [=======================>......] - ETA: 18s - loss: 0.2019 - categorical_accuracy: 0.9385
49984/60000 [=======================>......] - ETA: 18s - loss: 0.2019 - categorical_accuracy: 0.9386
50016/60000 [========================>.....] - ETA: 18s - loss: 0.2018 - categorical_accuracy: 0.9386
50048/60000 [========================>.....] - ETA: 18s - loss: 0.2019 - categorical_accuracy: 0.9386
50080/60000 [========================>.....] - ETA: 18s - loss: 0.2018 - categorical_accuracy: 0.9386
50112/60000 [========================>.....] - ETA: 18s - loss: 0.2018 - categorical_accuracy: 0.9386
50144/60000 [========================>.....] - ETA: 18s - loss: 0.2017 - categorical_accuracy: 0.9386
50176/60000 [========================>.....] - ETA: 18s - loss: 0.2017 - categorical_accuracy: 0.9386
50208/60000 [========================>.....] - ETA: 18s - loss: 0.2016 - categorical_accuracy: 0.9386
50240/60000 [========================>.....] - ETA: 18s - loss: 0.2017 - categorical_accuracy: 0.9386
50272/60000 [========================>.....] - ETA: 18s - loss: 0.2016 - categorical_accuracy: 0.9386
50304/60000 [========================>.....] - ETA: 18s - loss: 0.2015 - categorical_accuracy: 0.9387
50336/60000 [========================>.....] - ETA: 18s - loss: 0.2013 - categorical_accuracy: 0.9387
50368/60000 [========================>.....] - ETA: 18s - loss: 0.2014 - categorical_accuracy: 0.9387
50400/60000 [========================>.....] - ETA: 18s - loss: 0.2013 - categorical_accuracy: 0.9387
50432/60000 [========================>.....] - ETA: 17s - loss: 0.2012 - categorical_accuracy: 0.9387
50464/60000 [========================>.....] - ETA: 17s - loss: 0.2012 - categorical_accuracy: 0.9387
50496/60000 [========================>.....] - ETA: 17s - loss: 0.2012 - categorical_accuracy: 0.9388
50528/60000 [========================>.....] - ETA: 17s - loss: 0.2011 - categorical_accuracy: 0.9388
50560/60000 [========================>.....] - ETA: 17s - loss: 0.2010 - categorical_accuracy: 0.9388
50592/60000 [========================>.....] - ETA: 17s - loss: 0.2009 - categorical_accuracy: 0.9389
50624/60000 [========================>.....] - ETA: 17s - loss: 0.2009 - categorical_accuracy: 0.9389
50656/60000 [========================>.....] - ETA: 17s - loss: 0.2008 - categorical_accuracy: 0.9389
50688/60000 [========================>.....] - ETA: 17s - loss: 0.2008 - categorical_accuracy: 0.9389
50720/60000 [========================>.....] - ETA: 17s - loss: 0.2007 - categorical_accuracy: 0.9389
50752/60000 [========================>.....] - ETA: 17s - loss: 0.2007 - categorical_accuracy: 0.9389
50784/60000 [========================>.....] - ETA: 17s - loss: 0.2006 - categorical_accuracy: 0.9390
50816/60000 [========================>.....] - ETA: 17s - loss: 0.2005 - categorical_accuracy: 0.9390
50848/60000 [========================>.....] - ETA: 17s - loss: 0.2004 - categorical_accuracy: 0.9391
50880/60000 [========================>.....] - ETA: 17s - loss: 0.2004 - categorical_accuracy: 0.9391
50912/60000 [========================>.....] - ETA: 17s - loss: 0.2004 - categorical_accuracy: 0.9390
50944/60000 [========================>.....] - ETA: 17s - loss: 0.2003 - categorical_accuracy: 0.9391
50976/60000 [========================>.....] - ETA: 16s - loss: 0.2003 - categorical_accuracy: 0.9391
51008/60000 [========================>.....] - ETA: 16s - loss: 0.2003 - categorical_accuracy: 0.9391
51040/60000 [========================>.....] - ETA: 16s - loss: 0.2002 - categorical_accuracy: 0.9391
51072/60000 [========================>.....] - ETA: 16s - loss: 0.2001 - categorical_accuracy: 0.9391
51104/60000 [========================>.....] - ETA: 16s - loss: 0.2000 - categorical_accuracy: 0.9392
51136/60000 [========================>.....] - ETA: 16s - loss: 0.2000 - categorical_accuracy: 0.9392
51168/60000 [========================>.....] - ETA: 16s - loss: 0.1999 - categorical_accuracy: 0.9392
51200/60000 [========================>.....] - ETA: 16s - loss: 0.1998 - categorical_accuracy: 0.9392
51232/60000 [========================>.....] - ETA: 16s - loss: 0.1997 - categorical_accuracy: 0.9393
51264/60000 [========================>.....] - ETA: 16s - loss: 0.1996 - categorical_accuracy: 0.9393
51296/60000 [========================>.....] - ETA: 16s - loss: 0.1995 - categorical_accuracy: 0.9393
51328/60000 [========================>.....] - ETA: 16s - loss: 0.1994 - categorical_accuracy: 0.9394
51360/60000 [========================>.....] - ETA: 16s - loss: 0.1994 - categorical_accuracy: 0.9393
51392/60000 [========================>.....] - ETA: 16s - loss: 0.1994 - categorical_accuracy: 0.9394
51424/60000 [========================>.....] - ETA: 16s - loss: 0.1993 - categorical_accuracy: 0.9394
51456/60000 [========================>.....] - ETA: 16s - loss: 0.1993 - categorical_accuracy: 0.9394
51488/60000 [========================>.....] - ETA: 15s - loss: 0.1991 - categorical_accuracy: 0.9394
51520/60000 [========================>.....] - ETA: 15s - loss: 0.1990 - categorical_accuracy: 0.9395
51552/60000 [========================>.....] - ETA: 15s - loss: 0.1990 - categorical_accuracy: 0.9395
51584/60000 [========================>.....] - ETA: 15s - loss: 0.1989 - categorical_accuracy: 0.9395
51616/60000 [========================>.....] - ETA: 15s - loss: 0.1990 - categorical_accuracy: 0.9395
51648/60000 [========================>.....] - ETA: 15s - loss: 0.1991 - categorical_accuracy: 0.9395
51680/60000 [========================>.....] - ETA: 15s - loss: 0.1990 - categorical_accuracy: 0.9396
51712/60000 [========================>.....] - ETA: 15s - loss: 0.1990 - categorical_accuracy: 0.9396
51744/60000 [========================>.....] - ETA: 15s - loss: 0.1989 - categorical_accuracy: 0.9396
51776/60000 [========================>.....] - ETA: 15s - loss: 0.1989 - categorical_accuracy: 0.9396
51808/60000 [========================>.....] - ETA: 15s - loss: 0.1988 - categorical_accuracy: 0.9396
51840/60000 [========================>.....] - ETA: 15s - loss: 0.1987 - categorical_accuracy: 0.9397
51872/60000 [========================>.....] - ETA: 15s - loss: 0.1986 - categorical_accuracy: 0.9397
51904/60000 [========================>.....] - ETA: 15s - loss: 0.1985 - categorical_accuracy: 0.9397
51936/60000 [========================>.....] - ETA: 15s - loss: 0.1985 - categorical_accuracy: 0.9397
51968/60000 [========================>.....] - ETA: 15s - loss: 0.1983 - categorical_accuracy: 0.9397
52000/60000 [=========================>....] - ETA: 15s - loss: 0.1984 - categorical_accuracy: 0.9397
52032/60000 [=========================>....] - ETA: 14s - loss: 0.1983 - categorical_accuracy: 0.9397
52064/60000 [=========================>....] - ETA: 14s - loss: 0.1981 - categorical_accuracy: 0.9398
52096/60000 [=========================>....] - ETA: 14s - loss: 0.1980 - categorical_accuracy: 0.9398
52128/60000 [=========================>....] - ETA: 14s - loss: 0.1980 - categorical_accuracy: 0.9398
52160/60000 [=========================>....] - ETA: 14s - loss: 0.1979 - categorical_accuracy: 0.9399
52192/60000 [=========================>....] - ETA: 14s - loss: 0.1978 - categorical_accuracy: 0.9399
52224/60000 [=========================>....] - ETA: 14s - loss: 0.1977 - categorical_accuracy: 0.9400
52256/60000 [=========================>....] - ETA: 14s - loss: 0.1976 - categorical_accuracy: 0.9400
52288/60000 [=========================>....] - ETA: 14s - loss: 0.1975 - categorical_accuracy: 0.9400
52320/60000 [=========================>....] - ETA: 14s - loss: 0.1976 - categorical_accuracy: 0.9399
52352/60000 [=========================>....] - ETA: 14s - loss: 0.1976 - categorical_accuracy: 0.9399
52384/60000 [=========================>....] - ETA: 14s - loss: 0.1975 - categorical_accuracy: 0.9400
52416/60000 [=========================>....] - ETA: 14s - loss: 0.1975 - categorical_accuracy: 0.9400
52448/60000 [=========================>....] - ETA: 14s - loss: 0.1974 - categorical_accuracy: 0.9400
52480/60000 [=========================>....] - ETA: 14s - loss: 0.1973 - categorical_accuracy: 0.9400
52512/60000 [=========================>....] - ETA: 14s - loss: 0.1972 - categorical_accuracy: 0.9401
52544/60000 [=========================>....] - ETA: 14s - loss: 0.1971 - categorical_accuracy: 0.9401
52576/60000 [=========================>....] - ETA: 13s - loss: 0.1971 - categorical_accuracy: 0.9401
52608/60000 [=========================>....] - ETA: 13s - loss: 0.1970 - categorical_accuracy: 0.9401
52640/60000 [=========================>....] - ETA: 13s - loss: 0.1971 - categorical_accuracy: 0.9401
52672/60000 [=========================>....] - ETA: 13s - loss: 0.1970 - categorical_accuracy: 0.9402
52704/60000 [=========================>....] - ETA: 13s - loss: 0.1969 - categorical_accuracy: 0.9402
52736/60000 [=========================>....] - ETA: 13s - loss: 0.1968 - categorical_accuracy: 0.9402
52768/60000 [=========================>....] - ETA: 13s - loss: 0.1967 - categorical_accuracy: 0.9403
52800/60000 [=========================>....] - ETA: 13s - loss: 0.1968 - categorical_accuracy: 0.9402
52832/60000 [=========================>....] - ETA: 13s - loss: 0.1967 - categorical_accuracy: 0.9403
52864/60000 [=========================>....] - ETA: 13s - loss: 0.1966 - categorical_accuracy: 0.9403
52896/60000 [=========================>....] - ETA: 13s - loss: 0.1966 - categorical_accuracy: 0.9403
52928/60000 [=========================>....] - ETA: 13s - loss: 0.1965 - categorical_accuracy: 0.9403
52960/60000 [=========================>....] - ETA: 13s - loss: 0.1965 - categorical_accuracy: 0.9403
52992/60000 [=========================>....] - ETA: 13s - loss: 0.1964 - categorical_accuracy: 0.9404
53024/60000 [=========================>....] - ETA: 13s - loss: 0.1964 - categorical_accuracy: 0.9404
53056/60000 [=========================>....] - ETA: 13s - loss: 0.1962 - categorical_accuracy: 0.9404
53088/60000 [=========================>....] - ETA: 12s - loss: 0.1961 - categorical_accuracy: 0.9405
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1961 - categorical_accuracy: 0.9405
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1960 - categorical_accuracy: 0.9405
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1959 - categorical_accuracy: 0.9405
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1960 - categorical_accuracy: 0.9405
53248/60000 [=========================>....] - ETA: 12s - loss: 0.1959 - categorical_accuracy: 0.9405
53280/60000 [=========================>....] - ETA: 12s - loss: 0.1961 - categorical_accuracy: 0.9405
53312/60000 [=========================>....] - ETA: 12s - loss: 0.1963 - categorical_accuracy: 0.9405
53344/60000 [=========================>....] - ETA: 12s - loss: 0.1962 - categorical_accuracy: 0.9406
53376/60000 [=========================>....] - ETA: 12s - loss: 0.1961 - categorical_accuracy: 0.9406
53408/60000 [=========================>....] - ETA: 12s - loss: 0.1960 - categorical_accuracy: 0.9406
53440/60000 [=========================>....] - ETA: 12s - loss: 0.1959 - categorical_accuracy: 0.9407
53472/60000 [=========================>....] - ETA: 12s - loss: 0.1958 - categorical_accuracy: 0.9407
53504/60000 [=========================>....] - ETA: 12s - loss: 0.1957 - categorical_accuracy: 0.9407
53536/60000 [=========================>....] - ETA: 12s - loss: 0.1956 - categorical_accuracy: 0.9408
53568/60000 [=========================>....] - ETA: 12s - loss: 0.1955 - categorical_accuracy: 0.9408
53600/60000 [=========================>....] - ETA: 12s - loss: 0.1954 - categorical_accuracy: 0.9408
53632/60000 [=========================>....] - ETA: 11s - loss: 0.1953 - categorical_accuracy: 0.9408
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1953 - categorical_accuracy: 0.9409
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1952 - categorical_accuracy: 0.9409
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1953 - categorical_accuracy: 0.9409
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1952 - categorical_accuracy: 0.9409
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1952 - categorical_accuracy: 0.9409
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1951 - categorical_accuracy: 0.9409
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1951 - categorical_accuracy: 0.9410
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1950 - categorical_accuracy: 0.9410
53920/60000 [=========================>....] - ETA: 11s - loss: 0.1949 - categorical_accuracy: 0.9410
53952/60000 [=========================>....] - ETA: 11s - loss: 0.1949 - categorical_accuracy: 0.9410
53984/60000 [=========================>....] - ETA: 11s - loss: 0.1948 - categorical_accuracy: 0.9410
54016/60000 [==========================>...] - ETA: 11s - loss: 0.1947 - categorical_accuracy: 0.9411
54048/60000 [==========================>...] - ETA: 11s - loss: 0.1946 - categorical_accuracy: 0.9411
54080/60000 [==========================>...] - ETA: 11s - loss: 0.1946 - categorical_accuracy: 0.9411
54112/60000 [==========================>...] - ETA: 11s - loss: 0.1945 - categorical_accuracy: 0.9411
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1945 - categorical_accuracy: 0.9412
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1944 - categorical_accuracy: 0.9412
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1943 - categorical_accuracy: 0.9412
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1943 - categorical_accuracy: 0.9412
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1943 - categorical_accuracy: 0.9412
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1943 - categorical_accuracy: 0.9412
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1943 - categorical_accuracy: 0.9413
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1941 - categorical_accuracy: 0.9413
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1940 - categorical_accuracy: 0.9413
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1940 - categorical_accuracy: 0.9413
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1940 - categorical_accuracy: 0.9413
54496/60000 [==========================>...] - ETA: 10s - loss: 0.1939 - categorical_accuracy: 0.9413
54528/60000 [==========================>...] - ETA: 10s - loss: 0.1939 - categorical_accuracy: 0.9413
54560/60000 [==========================>...] - ETA: 10s - loss: 0.1939 - categorical_accuracy: 0.9413
54592/60000 [==========================>...] - ETA: 10s - loss: 0.1938 - categorical_accuracy: 0.9413
54624/60000 [==========================>...] - ETA: 10s - loss: 0.1938 - categorical_accuracy: 0.9413
54656/60000 [==========================>...] - ETA: 10s - loss: 0.1937 - categorical_accuracy: 0.9414
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1936 - categorical_accuracy: 0.9414 
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1935 - categorical_accuracy: 0.9414
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1934 - categorical_accuracy: 0.9415
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1933 - categorical_accuracy: 0.9415
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1932 - categorical_accuracy: 0.9415
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1931 - categorical_accuracy: 0.9416
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1930 - categorical_accuracy: 0.9416
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1930 - categorical_accuracy: 0.9416
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1930 - categorical_accuracy: 0.9416
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1929 - categorical_accuracy: 0.9416
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1930 - categorical_accuracy: 0.9416
55040/60000 [==========================>...] - ETA: 9s - loss: 0.1929 - categorical_accuracy: 0.9416
55072/60000 [==========================>...] - ETA: 9s - loss: 0.1928 - categorical_accuracy: 0.9417
55104/60000 [==========================>...] - ETA: 9s - loss: 0.1927 - categorical_accuracy: 0.9417
55136/60000 [==========================>...] - ETA: 9s - loss: 0.1926 - categorical_accuracy: 0.9417
55168/60000 [==========================>...] - ETA: 9s - loss: 0.1926 - categorical_accuracy: 0.9417
55200/60000 [==========================>...] - ETA: 9s - loss: 0.1925 - categorical_accuracy: 0.9418
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1924 - categorical_accuracy: 0.9418
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1923 - categorical_accuracy: 0.9418
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1923 - categorical_accuracy: 0.9418
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1922 - categorical_accuracy: 0.9418
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1925 - categorical_accuracy: 0.9418
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1925 - categorical_accuracy: 0.9418
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1925 - categorical_accuracy: 0.9418
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1924 - categorical_accuracy: 0.9418
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1924 - categorical_accuracy: 0.9418
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1923 - categorical_accuracy: 0.9419
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1923 - categorical_accuracy: 0.9419
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1922 - categorical_accuracy: 0.9419
55616/60000 [==========================>...] - ETA: 8s - loss: 0.1923 - categorical_accuracy: 0.9419
55648/60000 [==========================>...] - ETA: 8s - loss: 0.1922 - categorical_accuracy: 0.9419
55680/60000 [==========================>...] - ETA: 8s - loss: 0.1922 - categorical_accuracy: 0.9419
55712/60000 [==========================>...] - ETA: 8s - loss: 0.1921 - categorical_accuracy: 0.9420
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1920 - categorical_accuracy: 0.9420
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1920 - categorical_accuracy: 0.9420
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1919 - categorical_accuracy: 0.9420
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1918 - categorical_accuracy: 0.9420
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1917 - categorical_accuracy: 0.9420
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1916 - categorical_accuracy: 0.9421
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1915 - categorical_accuracy: 0.9421
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1914 - categorical_accuracy: 0.9421
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1913 - categorical_accuracy: 0.9422
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1914 - categorical_accuracy: 0.9422
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1913 - categorical_accuracy: 0.9422
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1912 - categorical_accuracy: 0.9422
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1911 - categorical_accuracy: 0.9423
56160/60000 [===========================>..] - ETA: 7s - loss: 0.1910 - categorical_accuracy: 0.9423
56192/60000 [===========================>..] - ETA: 7s - loss: 0.1910 - categorical_accuracy: 0.9423
56224/60000 [===========================>..] - ETA: 7s - loss: 0.1909 - categorical_accuracy: 0.9423
56256/60000 [===========================>..] - ETA: 7s - loss: 0.1908 - categorical_accuracy: 0.9424
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1907 - categorical_accuracy: 0.9424
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1907 - categorical_accuracy: 0.9424
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1906 - categorical_accuracy: 0.9424
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1905 - categorical_accuracy: 0.9425
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1906 - categorical_accuracy: 0.9424
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1905 - categorical_accuracy: 0.9425
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1904 - categorical_accuracy: 0.9425
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1903 - categorical_accuracy: 0.9425
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1902 - categorical_accuracy: 0.9425
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1902 - categorical_accuracy: 0.9426
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1901 - categorical_accuracy: 0.9426
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1900 - categorical_accuracy: 0.9426
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1900 - categorical_accuracy: 0.9426
56704/60000 [===========================>..] - ETA: 6s - loss: 0.1899 - categorical_accuracy: 0.9426
56736/60000 [===========================>..] - ETA: 6s - loss: 0.1898 - categorical_accuracy: 0.9427
56768/60000 [===========================>..] - ETA: 6s - loss: 0.1900 - categorical_accuracy: 0.9427
56800/60000 [===========================>..] - ETA: 6s - loss: 0.1900 - categorical_accuracy: 0.9427
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1899 - categorical_accuracy: 0.9427
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1899 - categorical_accuracy: 0.9427
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1898 - categorical_accuracy: 0.9428
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1897 - categorical_accuracy: 0.9428
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1896 - categorical_accuracy: 0.9428
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1897 - categorical_accuracy: 0.9428
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1897 - categorical_accuracy: 0.9428
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1896 - categorical_accuracy: 0.9429
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1896 - categorical_accuracy: 0.9428
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1895 - categorical_accuracy: 0.9429
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1897 - categorical_accuracy: 0.9429
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1896 - categorical_accuracy: 0.9429
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1895 - categorical_accuracy: 0.9429
57248/60000 [===========================>..] - ETA: 5s - loss: 0.1895 - categorical_accuracy: 0.9429
57280/60000 [===========================>..] - ETA: 5s - loss: 0.1894 - categorical_accuracy: 0.9429
57312/60000 [===========================>..] - ETA: 5s - loss: 0.1893 - categorical_accuracy: 0.9430
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1892 - categorical_accuracy: 0.9430
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1892 - categorical_accuracy: 0.9430
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1891 - categorical_accuracy: 0.9431
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1890 - categorical_accuracy: 0.9431
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1889 - categorical_accuracy: 0.9431
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1889 - categorical_accuracy: 0.9431
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1888 - categorical_accuracy: 0.9431
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1887 - categorical_accuracy: 0.9432
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1887 - categorical_accuracy: 0.9432
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1886 - categorical_accuracy: 0.9432
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1886 - categorical_accuracy: 0.9432
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1886 - categorical_accuracy: 0.9432
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1885 - categorical_accuracy: 0.9433
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1884 - categorical_accuracy: 0.9433
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1883 - categorical_accuracy: 0.9433
57824/60000 [===========================>..] - ETA: 4s - loss: 0.1883 - categorical_accuracy: 0.9433
57856/60000 [===========================>..] - ETA: 4s - loss: 0.1883 - categorical_accuracy: 0.9433
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1883 - categorical_accuracy: 0.9433
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1882 - categorical_accuracy: 0.9434
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1881 - categorical_accuracy: 0.9434
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1880 - categorical_accuracy: 0.9434
58016/60000 [============================>.] - ETA: 3s - loss: 0.1879 - categorical_accuracy: 0.9434
58048/60000 [============================>.] - ETA: 3s - loss: 0.1879 - categorical_accuracy: 0.9434
58080/60000 [============================>.] - ETA: 3s - loss: 0.1881 - categorical_accuracy: 0.9434
58112/60000 [============================>.] - ETA: 3s - loss: 0.1881 - categorical_accuracy: 0.9434
58144/60000 [============================>.] - ETA: 3s - loss: 0.1880 - categorical_accuracy: 0.9434
58176/60000 [============================>.] - ETA: 3s - loss: 0.1880 - categorical_accuracy: 0.9434
58208/60000 [============================>.] - ETA: 3s - loss: 0.1879 - categorical_accuracy: 0.9435
58240/60000 [============================>.] - ETA: 3s - loss: 0.1878 - categorical_accuracy: 0.9435
58272/60000 [============================>.] - ETA: 3s - loss: 0.1877 - categorical_accuracy: 0.9435
58304/60000 [============================>.] - ETA: 3s - loss: 0.1876 - categorical_accuracy: 0.9435
58336/60000 [============================>.] - ETA: 3s - loss: 0.1876 - categorical_accuracy: 0.9436
58368/60000 [============================>.] - ETA: 3s - loss: 0.1875 - categorical_accuracy: 0.9436
58400/60000 [============================>.] - ETA: 3s - loss: 0.1874 - categorical_accuracy: 0.9436
58432/60000 [============================>.] - ETA: 2s - loss: 0.1873 - categorical_accuracy: 0.9436
58464/60000 [============================>.] - ETA: 2s - loss: 0.1872 - categorical_accuracy: 0.9437
58496/60000 [============================>.] - ETA: 2s - loss: 0.1872 - categorical_accuracy: 0.9437
58528/60000 [============================>.] - ETA: 2s - loss: 0.1871 - categorical_accuracy: 0.9437
58560/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9437
58592/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9437
58624/60000 [============================>.] - ETA: 2s - loss: 0.1869 - categorical_accuracy: 0.9437
58656/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9438
58688/60000 [============================>.] - ETA: 2s - loss: 0.1867 - categorical_accuracy: 0.9438
58720/60000 [============================>.] - ETA: 2s - loss: 0.1866 - categorical_accuracy: 0.9438
58752/60000 [============================>.] - ETA: 2s - loss: 0.1866 - categorical_accuracy: 0.9438
58784/60000 [============================>.] - ETA: 2s - loss: 0.1865 - categorical_accuracy: 0.9439
58816/60000 [============================>.] - ETA: 2s - loss: 0.1865 - categorical_accuracy: 0.9439
58848/60000 [============================>.] - ETA: 2s - loss: 0.1864 - categorical_accuracy: 0.9439
58880/60000 [============================>.] - ETA: 2s - loss: 0.1863 - categorical_accuracy: 0.9439
58912/60000 [============================>.] - ETA: 2s - loss: 0.1862 - categorical_accuracy: 0.9440
58944/60000 [============================>.] - ETA: 1s - loss: 0.1863 - categorical_accuracy: 0.9439
58976/60000 [============================>.] - ETA: 1s - loss: 0.1862 - categorical_accuracy: 0.9439
59008/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9440
59040/60000 [============================>.] - ETA: 1s - loss: 0.1862 - categorical_accuracy: 0.9440
59072/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9440
59104/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9440
59136/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9440
59168/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9440
59200/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9441
59232/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9441
59264/60000 [============================>.] - ETA: 1s - loss: 0.1859 - categorical_accuracy: 0.9440
59296/60000 [============================>.] - ETA: 1s - loss: 0.1859 - categorical_accuracy: 0.9441
59328/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9441
59360/60000 [============================>.] - ETA: 1s - loss: 0.1859 - categorical_accuracy: 0.9441
59392/60000 [============================>.] - ETA: 1s - loss: 0.1858 - categorical_accuracy: 0.9441
59424/60000 [============================>.] - ETA: 1s - loss: 0.1857 - categorical_accuracy: 0.9441
59456/60000 [============================>.] - ETA: 1s - loss: 0.1857 - categorical_accuracy: 0.9441
59488/60000 [============================>.] - ETA: 0s - loss: 0.1856 - categorical_accuracy: 0.9442
59520/60000 [============================>.] - ETA: 0s - loss: 0.1857 - categorical_accuracy: 0.9442
59552/60000 [============================>.] - ETA: 0s - loss: 0.1857 - categorical_accuracy: 0.9442
59584/60000 [============================>.] - ETA: 0s - loss: 0.1856 - categorical_accuracy: 0.9442
59616/60000 [============================>.] - ETA: 0s - loss: 0.1855 - categorical_accuracy: 0.9442
59648/60000 [============================>.] - ETA: 0s - loss: 0.1854 - categorical_accuracy: 0.9442
59680/60000 [============================>.] - ETA: 0s - loss: 0.1854 - categorical_accuracy: 0.9443
59712/60000 [============================>.] - ETA: 0s - loss: 0.1854 - categorical_accuracy: 0.9442
59744/60000 [============================>.] - ETA: 0s - loss: 0.1853 - categorical_accuracy: 0.9443
59776/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9443
59808/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9443
59840/60000 [============================>.] - ETA: 0s - loss: 0.1851 - categorical_accuracy: 0.9443
59872/60000 [============================>.] - ETA: 0s - loss: 0.1850 - categorical_accuracy: 0.9443
59904/60000 [============================>.] - ETA: 0s - loss: 0.1850 - categorical_accuracy: 0.9443
59936/60000 [============================>.] - ETA: 0s - loss: 0.1850 - categorical_accuracy: 0.9443
59968/60000 [============================>.] - ETA: 0s - loss: 0.1849 - categorical_accuracy: 0.9444
60000/60000 [==============================] - 117s 2ms/step - loss: 0.1848 - categorical_accuracy: 0.9444 - val_loss: 0.0485 - val_categorical_accuracy: 0.9844

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 18s
  192/10000 [..............................] - ETA: 6s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 4s
  640/10000 [>.............................] - ETA: 4s
  768/10000 [=>............................] - ETA: 4s
  928/10000 [=>............................] - ETA: 3s
 1088/10000 [==>...........................] - ETA: 3s
 1248/10000 [==>...........................] - ETA: 3s
 1376/10000 [===>..........................] - ETA: 3s
 1536/10000 [===>..........................] - ETA: 3s
 1696/10000 [====>.........................] - ETA: 3s
 1824/10000 [====>.........................] - ETA: 3s
 1952/10000 [====>.........................] - ETA: 3s
 2112/10000 [=====>........................] - ETA: 3s
 2272/10000 [=====>........................] - ETA: 3s
 2432/10000 [======>.......................] - ETA: 3s
 2592/10000 [======>.......................] - ETA: 2s
 2752/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3072/10000 [========>.....................] - ETA: 2s
 3232/10000 [========>.....................] - ETA: 2s
 3392/10000 [=========>....................] - ETA: 2s
 3552/10000 [=========>....................] - ETA: 2s
 3712/10000 [==========>...................] - ETA: 2s
 3840/10000 [==========>...................] - ETA: 2s
 4000/10000 [===========>..................] - ETA: 2s
 4160/10000 [===========>..................] - ETA: 2s
 4320/10000 [===========>..................] - ETA: 2s
 4480/10000 [============>.................] - ETA: 2s
 4640/10000 [============>.................] - ETA: 2s
 4800/10000 [=============>................] - ETA: 1s
 4960/10000 [=============>................] - ETA: 1s
 5120/10000 [==============>...............] - ETA: 1s
 5280/10000 [==============>...............] - ETA: 1s
 5440/10000 [===============>..............] - ETA: 1s
 5600/10000 [===============>..............] - ETA: 1s
 5760/10000 [================>.............] - ETA: 1s
 5920/10000 [================>.............] - ETA: 1s
 6080/10000 [=================>............] - ETA: 1s
 6208/10000 [=================>............] - ETA: 1s
 6368/10000 [==================>...........] - ETA: 1s
 6496/10000 [==================>...........] - ETA: 1s
 6656/10000 [==================>...........] - ETA: 1s
 6816/10000 [===================>..........] - ETA: 1s
 6976/10000 [===================>..........] - ETA: 1s
 7136/10000 [====================>.........] - ETA: 1s
 7296/10000 [====================>.........] - ETA: 1s
 7456/10000 [=====================>........] - ETA: 0s
 7616/10000 [=====================>........] - ETA: 0s
 7776/10000 [======================>.......] - ETA: 0s
 7936/10000 [======================>.......] - ETA: 0s
 8096/10000 [=======================>......] - ETA: 0s
 8256/10000 [=======================>......] - ETA: 0s
 8416/10000 [========================>.....] - ETA: 0s
 8576/10000 [========================>.....] - ETA: 0s
 8736/10000 [=========================>....] - ETA: 0s
 8896/10000 [=========================>....] - ETA: 0s
 9056/10000 [==========================>...] - ETA: 0s
 9216/10000 [==========================>...] - ETA: 0s
 9376/10000 [===========================>..] - ETA: 0s
 9536/10000 [===========================>..] - ETA: 0s
 9696/10000 [============================>.] - ETA: 0s
 9856/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 373us/step
[[5.14954728e-08 3.80254406e-09 6.17756882e-07 ... 9.99996901e-01
  3.23250027e-09 1.26492478e-06]
 [2.34611834e-05 3.40165534e-05 9.99929070e-01 ... 1.67173098e-09
  1.49298899e-06 8.65610605e-09]
 [3.42919384e-06 9.99623179e-01 7.59255636e-05 ... 1.06069398e-04
  7.45244542e-05 6.49190179e-06]
 ...
 [4.63301203e-10 7.82226607e-07 6.67596112e-08 ... 5.42216583e-07
  1.07290680e-05 6.94663977e-05]
 [1.98695716e-06 6.80065355e-08 1.77034831e-08 ... 2.68904241e-08
  2.55450024e-04 1.96380725e-07]
 [1.59422802e-06 9.41114081e-07 1.47650453e-05 ... 6.63308242e-10
  7.38418407e-07 1.25697035e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04845768650218379, 'accuracy_test:': 0.9843999743461609}

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
   1e4627e..f1118e0  master     -> origin/master
Updating 1e4627e..f1118e0
Fast-forward
 .../20200522/list_log_pullrequest_20200522.md      |  2 +-
 error_list/20200522/list_log_testall_20200522.md   | 26 ++++++++++++++++++++++
 2 files changed, 27 insertions(+), 1 deletion(-)
[master 933788c] ml_store
 1 file changed, 2045 insertions(+)
To github.com:arita37/mlmodels_store.git
   f1118e0..933788c  master -> master





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

2020-05-22 05:06:46.030843: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-22 05:06:46.036608: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-22 05:06:46.036870: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55686ac5d120 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 05:06:46.036887: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3865
256/354 [====================>.........] - ETA: 3s - loss: 1.2802
354/354 [==============================] - 16s 44ms/step - loss: 1.3637 - val_loss: 1.7022

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
[master 6c92e43] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   933788c..6c92e43  master -> master





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
[master 993d461] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   6c92e43..993d461  master -> master





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
[master aeeb3f8] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   993d461..aeeb3f8  master -> master





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
[master 32eafff] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   aeeb3f8..32eafff  master -> master





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
[master ed33615] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   32eafff..ed33615  master -> master





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
	Data preprocessing and feature engineering runtime = 0.26s ...
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
 40%|████      | 2/5 [00:22<00:33, 11.04s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.7820477527889869, 'learning_rate': 0.19049875311717132, 'min_data_in_leaf': 27, 'num_leaves': 52} and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\x06\x89\x02Dr\x9eX\r\x00\x00\x00learning_rateq\x02G?\xc8bC]H\x94gX\x10\x00\x00\x00min_data_in_leafq\x03K\x1bX\n\x00\x00\x00num_leavesq\x04K4u.' and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\x06\x89\x02Dr\x9eX\r\x00\x00\x00learning_rateq\x02G?\xc8bC]H\x94gX\x10\x00\x00\x00min_data_in_leafq\x03K\x1bX\n\x00\x00\x00num_leavesq\x04K4u.' and reward: 0.3888
 60%|██████    | 3/5 [00:48<00:31, 15.71s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.8808559084715806, 'learning_rate': 0.009551140172905169, 'min_data_in_leaf': 15, 'num_leaves': 43} and reward: 0.3892
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec/\xf8\xba\xeb\xf6<X\r\x00\x00\x00learning_rateq\x02G?\x83\x8f\x8cUu\x05\x8cX\x10\x00\x00\x00min_data_in_leafq\x03K\x0fX\n\x00\x00\x00num_leavesq\x04K+u.' and reward: 0.3892
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec/\xf8\xba\xeb\xf6<X\r\x00\x00\x00learning_rateq\x02G?\x83\x8f\x8cUu\x05\x8cX\x10\x00\x00\x00min_data_in_leafq\x03K\x0fX\n\x00\x00\x00num_leavesq\x04K+u.' and reward: 0.3892
 80%|████████  | 4/5 [01:13<00:18, 18.38s/it] 80%|████████  | 4/5 [01:13<00:18, 18.33s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.7516036531074322, 'learning_rate': 0.029833720205631808, 'min_data_in_leaf': 10, 'num_leaves': 34} and reward: 0.3908
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\r#\x1a\xb4\xd1NX\r\x00\x00\x00learning_rateq\x02G?\x9e\x8c\xbb\x12g\xa1\xc3X\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K"u.' and reward: 0.3908
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\r#\x1a\xb4\xd1NX\r\x00\x00\x00learning_rateq\x02G?\x9e\x8c\xbb\x12g\xa1\xc3X\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K"u.' and reward: 0.3908
Time for Gradient Boosting hyperparameter optimization: 95.09398770332336
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
 40%|████      | 2/5 [00:56<01:25, 28.48s/it] 40%|████      | 2/5 [00:56<01:25, 28.48s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.20359269001963892, 'embedding_size_factor': 0.5806122888425673, 'layers.choice': 1, 'learning_rate': 0.0023902773454701617, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.0005729927291111072} and reward: 0.3778
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xca\x0fSD\xabd\xd6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe2\x94`9\x07\x81JX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?c\x94\xc6`\xde\x87\x1eX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?B\xc6\x9c\x84$\xf6Ju.' and reward: 0.3778
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xca\x0fSD\xabd\xd6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe2\x94`9\x07\x81JX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?c\x94\xc6`\xde\x87\x1eX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?B\xc6\x9c\x84$\xf6Ju.' and reward: 0.3778
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 162.46007990837097
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -141.38s of remaining time.
Ensemble size: 48
Ensemble weights: 
[0.10416667 0.08333333 0.20833333 0.45833333 0.04166667 0.10416667]
	0.3996	 = Validation accuracy score
	1.64s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 263.08s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f57ffa31710>

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
   ed33615..a258e9a  master     -> origin/master
Updating ed33615..a258e9a
Fast-forward
 deps.txt                                           |  7 +-
 .../20200522/list_log_pullrequest_20200522.md      |  2 +-
 error_list/20200522/list_log_testall_20200522.md   | 53 +++++++++++++++
 ...-10_51b64e342c7b2661e79b8abaa33db92672ae95c7.py | 75 ++++++++++++++++++++++
 4 files changed, 134 insertions(+), 3 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-22-05-10_51b64e342c7b2661e79b8abaa33db92672ae95c7.py

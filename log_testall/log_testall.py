
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

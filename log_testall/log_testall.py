
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

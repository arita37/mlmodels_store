
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

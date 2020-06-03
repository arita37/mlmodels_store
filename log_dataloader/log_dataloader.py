
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/ce3071beb7006ef52315b8a20bcbe252eb8bee27', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'ce3071beb7006ef52315b8a20bcbe252eb8bee27', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/ce3071beb7006ef52315b8a20bcbe252eb8bee27

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/ce3071beb7006ef52315b8a20bcbe252eb8bee27

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/ce3071beb7006ef52315b8a20bcbe252eb8bee27

 ************************************************************************************************************************

  ############Check model ################################ 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/ag_news_csv",
    "train":true,
    "alphabet_size":69,
    "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size":1014,
    "num_of_classes":4
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "colX":[
          "colX"
        ],
        "coly":[
          "coly"
        ],
        "encoding":"'ISO-8859-1'",
        "read_csv_parm":{
          "usecols":[
            0,
            1
          ],
          "names":[
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name":"tokenizer",
      "uri":"mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args":{
        "data_source":"",
        "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size":1014,
        "num_of_classes":4
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'> 
cls_name : Data

  
 Object Creation 

  
 Object Compute 

  
 Object get_data 

  
 #####  get_Data DataLoader  

  ((array([[65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       ...,
       [ 5,  3,  9, ...,  0,  0,  0],
       [20,  5, 11, ...,  0,  0,  0],
       [20,  3,  5, ...,  0,  0,  0]]), array([[0, 0, 1, 0],
       [0, 0, 1, 0],
       [0, 0, 1, 0],
       ...,
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0]])), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/ag_news_csv",
    "train":true,
    "alphabet_size":69,
    "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size":1014,
    "num_of_classes":4
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "colX":[
          "colX"
        ],
        "coly":[
          "coly"
        ],
        "encoding":"'ISO-8859-1'",
        "read_csv_parm":{
          "usecols":[
            0,
            1
          ],
          "names":[
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name":"tokenizer",
      "uri":"mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args":{
        "data_source":"",
        "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size":1014,
        "num_of_classes":4
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'> 
cls_name : Data

  
 Object Creation 

  
 Object Compute 

  
 Object get_data 

  
 #####  get_Data DataLoader  

  ((array([[65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       ...,
       [ 5,  3,  9, ...,  0,  0,  0],
       [20,  5, 11, ...,  0,  0,  0],
       [20,  3,  5, ...,  0,  0,  0]]), array([[0, 0, 1, 0],
       [0, 0, 1, 0],
       [0, 0, 1, 0],
       ...,
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0]])), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/imdb",
    "pass_data_pars":false,
    "train":true,
    "maxlen":40,
    "max_features":5
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:NumpyDataset",
      "args":{
        "numpy_loader_args":{
          "allow_pickle":true
        },
        "encoding":"'ISO-8859-1'"
      }
    },
    {
      "name":"imdb_process",
      "uri":"mlmodels.preprocess.text_keras:IMDBDataset",
      "args":{
        "num_words":5
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:NumpyDataset {'numpy_loader_args': {'allow_pickle': True}, 'encoding': "'ISO-8859-1'"} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.NumpyDataset'> 
cls_name : NumpyDataset
Dataset File path :  mlmodels/dataset/text/imdb.npz

  URL:  mlmodels.preprocess.text_keras:IMDBDataset {'num_words': 5} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.text_keras.IMDBDataset'> 
cls_name : IMDBDataset

  
 Object Creation 

  
 Object Compute 

  
 Object get_data 

  
 #####  get_Data DataLoader  

  ((array([list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       ...,
       list([1, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       list([1, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])],
      dtype=object), array([1, 1, 1, ..., 0, 0, 0]), array([list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2]),
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       ...,
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2]),
       list([1, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])],
      dtype=object), array([1, 0, 0, ..., 0, 1, 0])), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/text\/",
    "dataset":"ner_dataset.csv",
    "pass_data_pars":false,
    "train":true
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "read_csv_parm":{
          "encoding":"ISO-8859-1"
        },
        "colX":[

        ],
        "coly":[

        ]
      }
    },
    {
      "uri":"mlmodels.preprocess.text_keras:Preprocess_namentity",
      "args":{
        "max_len":75
      },
      "internal_states":[
        "word_count"
      ]
    },
    {
      "name":"split_xy",
      "uri":"mlmodels.dataloader:split_xy_from_dict",
      "args":{
        "col_Xinput":[
          "X"
        ],
        "col_yinput":[
          "y"
        ]
      }
    },
    {
      "name":"split_train_test",
      "uri":"sklearn.model_selection:train_test_split",
      "args":{
        "test_size":0.5
      }
    },
    {
      "name":"saver",
      "uri":"mlmodels.dataloader:pickle_dump",
      "args":{
        "path":"mlmodels\/ztest\/ml_keras\/namentity_crm_bilstm\/data.pkl"
      }
    }
  ],
  "output":{
    "shape":[
      [
        75
      ],
      [
        75,
        18
      ]
    ],
    "max_len":75
  }
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'read_csv_parm': {'encoding': 'ISO-8859-1'}, 'colX': [], 'coly': []} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset

  URL:  mlmodels.preprocess.text_keras:Preprocess_namentity {'max_len': 75} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.text_keras.Preprocess_namentity'> 
cls_name : Preprocess_namentity

  
 Object Creation 

  
 Object Compute 

  
 Object get_data 

  URL:  mlmodels.dataloader:split_xy_from_dict {'col_Xinput': ['X'], 'col_yinput': ['y']} 

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f98547492f0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f98547492f0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f98c679a0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f98c679a0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f985df2c9d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f985df2c9d8> 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/torchhub_cnn_dataloader.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/MNIST",
    "dataset":"MNIST",
    "data_type":"tch_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tch_dataset_start",
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"torchvision.datasets:MNIST",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars":false,
          "arg":{
            "fixed_size":256,
            "path":"dataset\/vision\/MNIST\/"
          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f987407e8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f987407e8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f987407e8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 445, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 297, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 92, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|██▉       | 2965504/9912422 [00:00<00:00, 29496275.18it/s]9920512it [00:00, 33661682.36it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 269798.15it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7827110.95it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 54226.82it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f98577179b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9857717cc0>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/model_list_CIFAR.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/cifar10\/",
    "dataset":"CIFAR10",
    "data_type":"tf_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tf_dataset_start",
      "uri":"mlmodels.preprocess.generic:tf_dataset_download",
      "arg":{
        "train_samples":2000,
        "test_samples":500,
        "shuffle":true,
        "download":true
      }
    },
    {
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"mlmodels.preprocess.generic:NumpyDataset",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_generic",
          "pass_data_pars":false,
          "arg":{
            "fixed_size":256
          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:tf_dataset_download {} 

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f987407e510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f987407e510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f987407e510> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:05,  2.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:05,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:05,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:04,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:04,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:03,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:03,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:03,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:02,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:44,  3.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:44,  3.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:43,  3.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:43,  3.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:43,  3.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:43,  3.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:42,  3.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:42,  3.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:42,  3.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:41,  3.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:41,  3.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:29,  4.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:29,  4.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:29,  4.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:28,  4.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:28,  4.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:28,  4.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:28,  4.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:28,  4.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:27,  4.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:27,  4.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:19,  6.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:19,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:19,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:19,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:19,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:19,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:19,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:18,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:18,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:18,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:13,  9.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:13,  9.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:13,  9.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:13,  9.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:13,  9.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:12,  9.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:12,  9.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:12,  9.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:12,  9.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:12,  9.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:12,  9.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 12.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 12.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:08, 12.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:08, 12.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:08, 12.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:08, 12.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:08, 12.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 12.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 17.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 17.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 17.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 17.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 17.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 17.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 17.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 17.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 17.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 17.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 17.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 22.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 22.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 22.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 22.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 22.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 22.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 22.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 22.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 22.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 22.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 29.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 36.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 36.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 36.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 36.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 36.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 36.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 36.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 36.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 36.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 36.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 36.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 45.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 45.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 45.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 45.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 45.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 45.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 45.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 60.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 60.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 60.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 60.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 60.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 60.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 60.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 60.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 60.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 60.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 66.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 71.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 71.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 71.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 71.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 71.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 71.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 71.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 71.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 71.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 75.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 75.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 75.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 75.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 75.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 75.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 75.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 82.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.19s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.08s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 82.42 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.08s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.09s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 39.65 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.09s/ url]
0 examples [00:00, ? examples/s]2020-06-03 18:08:04.407422: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-03 18:08:04.421241: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-03 18:08:04.421971: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f2d356c5d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-03 18:08:04.422006: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
30 examples [00:00, 299.54 examples/s]135 examples [00:00, 381.19 examples/s]251 examples [00:00, 477.08 examples/s]363 examples [00:00, 575.78 examples/s]467 examples [00:00, 664.01 examples/s]581 examples [00:00, 757.89 examples/s]686 examples [00:00, 825.07 examples/s]799 examples [00:00, 896.01 examples/s]911 examples [00:00, 952.45 examples/s]1018 examples [00:01, 983.03 examples/s]1132 examples [00:01, 1023.39 examples/s]1240 examples [00:01, 1031.45 examples/s]1354 examples [00:01, 1060.64 examples/s]1468 examples [00:01, 1081.47 examples/s]1584 examples [00:01, 1103.16 examples/s]1703 examples [00:01, 1126.73 examples/s]1817 examples [00:01, 1118.17 examples/s]1930 examples [00:01, 1110.00 examples/s]2042 examples [00:01, 1109.29 examples/s]2160 examples [00:02, 1128.55 examples/s]2274 examples [00:02, 1126.38 examples/s]2387 examples [00:02, 1113.99 examples/s]2506 examples [00:02, 1134.43 examples/s]2624 examples [00:02, 1145.99 examples/s]2739 examples [00:02, 1140.02 examples/s]2857 examples [00:02, 1151.02 examples/s]2973 examples [00:02, 1128.42 examples/s]3089 examples [00:02, 1135.12 examples/s]3203 examples [00:02, 1131.96 examples/s]3320 examples [00:03, 1142.12 examples/s]3438 examples [00:03, 1151.74 examples/s]3554 examples [00:03, 1124.86 examples/s]3667 examples [00:03, 1125.00 examples/s]3780 examples [00:03, 1123.21 examples/s]3893 examples [00:03, 1074.95 examples/s]4008 examples [00:03, 1095.23 examples/s]4118 examples [00:03, 1082.03 examples/s]4235 examples [00:03, 1104.80 examples/s]4351 examples [00:03, 1119.80 examples/s]4464 examples [00:04, 1110.38 examples/s]4576 examples [00:04, 1100.83 examples/s]4687 examples [00:04, 1098.66 examples/s]4797 examples [00:04, 1039.87 examples/s]4902 examples [00:04, 1020.72 examples/s]5005 examples [00:04, 944.56 examples/s] 5102 examples [00:04, 930.95 examples/s]5214 examples [00:04, 979.91 examples/s]5332 examples [00:04, 1031.04 examples/s]5448 examples [00:05, 1064.62 examples/s]5568 examples [00:05, 1100.23 examples/s]5680 examples [00:05, 1080.77 examples/s]5790 examples [00:05, 1078.42 examples/s]5900 examples [00:05, 1084.06 examples/s]6009 examples [00:05, 1070.62 examples/s]6119 examples [00:05, 1077.71 examples/s]6228 examples [00:05, 1059.97 examples/s]6340 examples [00:05, 1074.97 examples/s]6453 examples [00:05, 1090.83 examples/s]6566 examples [00:06, 1102.06 examples/s]6679 examples [00:06, 1107.80 examples/s]6790 examples [00:06, 1091.16 examples/s]6900 examples [00:06, 1072.48 examples/s]7012 examples [00:06, 1084.59 examples/s]7133 examples [00:06, 1117.75 examples/s]7251 examples [00:06, 1133.35 examples/s]7365 examples [00:06, 1082.75 examples/s]7480 examples [00:06, 1099.77 examples/s]7591 examples [00:07, 1096.95 examples/s]7703 examples [00:07, 1101.70 examples/s]7814 examples [00:07, 1090.33 examples/s]7924 examples [00:07, 1067.14 examples/s]8038 examples [00:07, 1087.60 examples/s]8150 examples [00:07, 1097.04 examples/s]8265 examples [00:07, 1110.85 examples/s]8379 examples [00:07, 1118.39 examples/s]8491 examples [00:07, 1090.33 examples/s]8601 examples [00:07, 1089.19 examples/s]8715 examples [00:08, 1101.63 examples/s]8828 examples [00:08, 1107.58 examples/s]8942 examples [00:08, 1115.69 examples/s]9054 examples [00:08, 1089.94 examples/s]9167 examples [00:08, 1098.80 examples/s]9279 examples [00:08, 1104.84 examples/s]9391 examples [00:08, 1107.38 examples/s]9502 examples [00:08, 1074.78 examples/s]9610 examples [00:08, 1064.92 examples/s]9728 examples [00:08, 1094.69 examples/s]9838 examples [00:09, 1066.54 examples/s]9956 examples [00:09, 1095.88 examples/s]10067 examples [00:09, 1054.57 examples/s]10174 examples [00:09, 1049.21 examples/s]10289 examples [00:09, 1076.05 examples/s]10402 examples [00:09, 1091.54 examples/s]10516 examples [00:09, 1104.75 examples/s]10629 examples [00:09, 1111.37 examples/s]10741 examples [00:09, 1053.95 examples/s]10856 examples [00:09, 1078.51 examples/s]10970 examples [00:10, 1093.67 examples/s]11084 examples [00:10, 1106.81 examples/s]11198 examples [00:10, 1114.33 examples/s]11310 examples [00:10, 1072.55 examples/s]11425 examples [00:10, 1091.55 examples/s]11535 examples [00:10, 1091.37 examples/s]11650 examples [00:10, 1107.91 examples/s]11769 examples [00:10, 1128.86 examples/s]11883 examples [00:10, 1115.85 examples/s]12000 examples [00:11, 1131.38 examples/s]12118 examples [00:11, 1144.76 examples/s]12236 examples [00:11, 1154.41 examples/s]12352 examples [00:11, 1155.84 examples/s]12468 examples [00:11, 1131.60 examples/s]12582 examples [00:11, 1131.53 examples/s]12699 examples [00:11, 1141.51 examples/s]12814 examples [00:11, 1125.91 examples/s]12927 examples [00:11, 1120.20 examples/s]13040 examples [00:11, 1086.37 examples/s]13149 examples [00:12, 1086.15 examples/s]13258 examples [00:12, 1079.12 examples/s]13367 examples [00:12, 1042.20 examples/s]13472 examples [00:12, 1031.13 examples/s]13576 examples [00:12, 1029.04 examples/s]13686 examples [00:12, 1047.87 examples/s]13794 examples [00:12, 1054.46 examples/s]13906 examples [00:12, 1071.96 examples/s]14014 examples [00:12, 1047.62 examples/s]14120 examples [00:12, 1026.74 examples/s]14230 examples [00:13, 1046.07 examples/s]14335 examples [00:13, 1044.03 examples/s]14447 examples [00:13, 1065.14 examples/s]14556 examples [00:13, 1071.71 examples/s]14668 examples [00:13, 1083.80 examples/s]14777 examples [00:13, 1060.19 examples/s]14888 examples [00:13, 1073.17 examples/s]14996 examples [00:13, 1050.36 examples/s]15102 examples [00:13, 1037.95 examples/s]15207 examples [00:14, 1039.68 examples/s]15325 examples [00:14, 1075.85 examples/s]15443 examples [00:14, 1103.93 examples/s]15562 examples [00:14, 1127.04 examples/s]15677 examples [00:14, 1130.53 examples/s]15791 examples [00:14, 1115.27 examples/s]15903 examples [00:14, 1105.78 examples/s]16017 examples [00:14, 1115.69 examples/s]16131 examples [00:14, 1120.19 examples/s]16244 examples [00:14, 1119.39 examples/s]16359 examples [00:15, 1127.29 examples/s]16476 examples [00:15, 1138.10 examples/s]16590 examples [00:15, 1128.82 examples/s]16709 examples [00:15, 1145.31 examples/s]16824 examples [00:15, 1146.69 examples/s]16939 examples [00:15, 1136.49 examples/s]17053 examples [00:15, 1128.52 examples/s]17166 examples [00:15, 1113.67 examples/s]17284 examples [00:15, 1131.23 examples/s]17398 examples [00:15, 1115.17 examples/s]17510 examples [00:16, 1116.12 examples/s]17622 examples [00:16, 1111.94 examples/s]17734 examples [00:16, 1106.35 examples/s]17848 examples [00:16, 1114.44 examples/s]17960 examples [00:16, 1084.00 examples/s]18069 examples [00:16, 1039.43 examples/s]18183 examples [00:16, 1067.12 examples/s]18294 examples [00:16, 1078.95 examples/s]18403 examples [00:16, 1073.47 examples/s]18511 examples [00:16, 1069.19 examples/s]18634 examples [00:17, 1111.60 examples/s]18753 examples [00:17, 1133.26 examples/s]18867 examples [00:17, 1117.51 examples/s]18981 examples [00:17, 1124.02 examples/s]19094 examples [00:17, 1114.27 examples/s]19212 examples [00:17, 1130.77 examples/s]19329 examples [00:17, 1139.88 examples/s]19447 examples [00:17, 1151.51 examples/s]19564 examples [00:17, 1156.42 examples/s]19680 examples [00:18, 1128.58 examples/s]19799 examples [00:18, 1146.07 examples/s]19914 examples [00:18, 1145.59 examples/s]20029 examples [00:18, 1068.17 examples/s]20138 examples [00:18, 1074.00 examples/s]20251 examples [00:18, 1089.90 examples/s]20369 examples [00:18, 1114.57 examples/s]20482 examples [00:18, 1118.87 examples/s]20601 examples [00:18, 1137.73 examples/s]20716 examples [00:18, 1098.41 examples/s]20827 examples [00:19, 1086.02 examples/s]20937 examples [00:19, 1089.29 examples/s]21053 examples [00:19, 1106.54 examples/s]21169 examples [00:19, 1121.82 examples/s]21282 examples [00:19, 1099.98 examples/s]21398 examples [00:19, 1117.15 examples/s]21516 examples [00:19, 1134.15 examples/s]21630 examples [00:19, 1129.19 examples/s]21744 examples [00:19, 1087.11 examples/s]21854 examples [00:19, 1083.51 examples/s]21972 examples [00:20, 1109.51 examples/s]22088 examples [00:20, 1123.83 examples/s]22204 examples [00:20, 1133.08 examples/s]22322 examples [00:20, 1144.19 examples/s]22437 examples [00:20, 1119.10 examples/s]22552 examples [00:20, 1127.42 examples/s]22669 examples [00:20, 1138.65 examples/s]22784 examples [00:20, 1140.20 examples/s]22901 examples [00:20, 1147.38 examples/s]23016 examples [00:21, 1093.96 examples/s]23131 examples [00:21, 1108.64 examples/s]23251 examples [00:21, 1132.52 examples/s]23370 examples [00:21, 1147.64 examples/s]23486 examples [00:21, 1123.11 examples/s]23599 examples [00:21, 1060.80 examples/s]23707 examples [00:21, 1053.35 examples/s]23818 examples [00:21, 1068.69 examples/s]23928 examples [00:21, 1076.77 examples/s]24037 examples [00:21, 1069.23 examples/s]24145 examples [00:22, 1060.24 examples/s]24252 examples [00:22, 1048.24 examples/s]24362 examples [00:22, 1060.94 examples/s]24476 examples [00:22, 1083.11 examples/s]24585 examples [00:22, 1058.87 examples/s]24692 examples [00:22, 1006.76 examples/s]24794 examples [00:22, 1007.90 examples/s]24907 examples [00:22, 1040.06 examples/s]25019 examples [00:22, 1062.47 examples/s]25126 examples [00:22, 1040.77 examples/s]25235 examples [00:23, 1052.48 examples/s]25341 examples [00:23, 1049.01 examples/s]25456 examples [00:23, 1076.56 examples/s]25571 examples [00:23, 1095.40 examples/s]25681 examples [00:23, 1078.26 examples/s]25799 examples [00:23, 1106.82 examples/s]25914 examples [00:23, 1118.61 examples/s]26033 examples [00:23, 1137.97 examples/s]26148 examples [00:23, 1137.45 examples/s]26262 examples [00:24, 1110.99 examples/s]26382 examples [00:24, 1133.86 examples/s]26498 examples [00:24, 1138.86 examples/s]26616 examples [00:24, 1149.90 examples/s]26734 examples [00:24, 1156.10 examples/s]26850 examples [00:24, 1141.76 examples/s]26965 examples [00:24, 1132.72 examples/s]27079 examples [00:24, 1119.08 examples/s]27194 examples [00:24, 1125.52 examples/s]27307 examples [00:24, 1126.47 examples/s]27420 examples [00:25, 1106.36 examples/s]27536 examples [00:25, 1119.42 examples/s]27649 examples [00:25, 1076.26 examples/s]27763 examples [00:25, 1092.97 examples/s]27882 examples [00:25, 1119.41 examples/s]27995 examples [00:25, 1109.71 examples/s]28112 examples [00:25, 1126.63 examples/s]28230 examples [00:25, 1139.43 examples/s]28345 examples [00:25, 1126.33 examples/s]28459 examples [00:25, 1128.16 examples/s]28572 examples [00:26, 1114.52 examples/s]28684 examples [00:26, 1082.03 examples/s]28800 examples [00:26, 1102.05 examples/s]28914 examples [00:26, 1113.05 examples/s]29028 examples [00:26, 1118.24 examples/s]29140 examples [00:26, 1087.90 examples/s]29256 examples [00:26, 1106.79 examples/s]29367 examples [00:26, 1096.74 examples/s]29484 examples [00:26, 1117.07 examples/s]29602 examples [00:26, 1133.36 examples/s]29717 examples [00:27, 1137.94 examples/s]29835 examples [00:27, 1148.39 examples/s]29950 examples [00:27, 1146.39 examples/s]30065 examples [00:27, 1087.12 examples/s]30180 examples [00:27, 1101.64 examples/s]30291 examples [00:27, 1090.18 examples/s]30401 examples [00:27, 1074.26 examples/s]30518 examples [00:27, 1099.38 examples/s]30634 examples [00:27, 1116.26 examples/s]30746 examples [00:28, 1097.29 examples/s]30861 examples [00:28, 1111.86 examples/s]30973 examples [00:28, 1112.66 examples/s]31085 examples [00:28, 1109.29 examples/s]31203 examples [00:28, 1129.26 examples/s]31317 examples [00:28, 1122.45 examples/s]31430 examples [00:28, 1110.02 examples/s]31548 examples [00:28, 1129.54 examples/s]31665 examples [00:28, 1141.01 examples/s]31780 examples [00:28, 1140.17 examples/s]31895 examples [00:29, 1114.44 examples/s]32008 examples [00:29, 1116.49 examples/s]32120 examples [00:29, 1094.95 examples/s]32230 examples [00:29, 1077.25 examples/s]32338 examples [00:29, 1025.87 examples/s]32442 examples [00:29, 1018.20 examples/s]32550 examples [00:29, 1033.96 examples/s]32667 examples [00:29, 1069.26 examples/s]32782 examples [00:29, 1090.62 examples/s]32897 examples [00:29, 1105.06 examples/s]33008 examples [00:30, 1091.86 examples/s]33122 examples [00:30, 1102.51 examples/s]33236 examples [00:30, 1112.81 examples/s]33353 examples [00:30, 1128.43 examples/s]33470 examples [00:30, 1140.05 examples/s]33585 examples [00:30, 1123.46 examples/s]33701 examples [00:30, 1133.28 examples/s]33817 examples [00:30, 1140.18 examples/s]33932 examples [00:30, 1137.94 examples/s]34046 examples [00:30, 1132.90 examples/s]34160 examples [00:31, 1080.43 examples/s]34269 examples [00:31, 1081.43 examples/s]34378 examples [00:31, 1067.24 examples/s]34486 examples [00:31, 1068.92 examples/s]34594 examples [00:31, 1039.20 examples/s]34703 examples [00:31, 1053.48 examples/s]34821 examples [00:31, 1088.03 examples/s]34938 examples [00:31, 1110.20 examples/s]35055 examples [00:31, 1125.20 examples/s]35169 examples [00:32, 1128.38 examples/s]35283 examples [00:32, 1117.91 examples/s]35401 examples [00:32, 1134.68 examples/s]35517 examples [00:32, 1141.29 examples/s]35632 examples [00:32, 1124.99 examples/s]35748 examples [00:32, 1132.96 examples/s]35862 examples [00:32, 1118.68 examples/s]35974 examples [00:32, 1109.30 examples/s]36088 examples [00:32, 1116.05 examples/s]36200 examples [00:32, 1114.90 examples/s]36319 examples [00:33, 1134.27 examples/s]36433 examples [00:33, 1106.84 examples/s]36548 examples [00:33, 1117.93 examples/s]36663 examples [00:33, 1125.45 examples/s]36779 examples [00:33, 1135.23 examples/s]36893 examples [00:33, 1135.08 examples/s]37007 examples [00:33, 1113.16 examples/s]37119 examples [00:33, 1113.97 examples/s]37236 examples [00:33, 1127.68 examples/s]37349 examples [00:33, 1105.09 examples/s]37466 examples [00:34, 1121.24 examples/s]37579 examples [00:34, 1107.14 examples/s]37690 examples [00:34, 1060.70 examples/s]37801 examples [00:34, 1072.80 examples/s]37911 examples [00:34, 1080.16 examples/s]38020 examples [00:34, 1073.78 examples/s]38128 examples [00:34, 1042.14 examples/s]38239 examples [00:34, 1059.15 examples/s]38350 examples [00:34, 1071.62 examples/s]38463 examples [00:35, 1085.91 examples/s]38574 examples [00:35, 1092.88 examples/s]38690 examples [00:35, 1110.22 examples/s]38809 examples [00:35, 1131.92 examples/s]38927 examples [00:35, 1143.16 examples/s]39042 examples [00:35, 1106.77 examples/s]39154 examples [00:35, 1070.07 examples/s]39265 examples [00:35, 1079.36 examples/s]39374 examples [00:35, 1078.54 examples/s]39485 examples [00:35, 1087.04 examples/s]39599 examples [00:36, 1099.92 examples/s]39710 examples [00:36, 1099.37 examples/s]39821 examples [00:36, 1102.35 examples/s]39937 examples [00:36, 1118.03 examples/s]40049 examples [00:36, 1060.19 examples/s]40165 examples [00:36, 1086.36 examples/s]40277 examples [00:36, 1093.81 examples/s]40391 examples [00:36, 1105.71 examples/s]40503 examples [00:36, 1107.39 examples/s]40617 examples [00:36, 1115.24 examples/s]40736 examples [00:37, 1134.47 examples/s]40850 examples [00:37, 1125.77 examples/s]40964 examples [00:37, 1128.37 examples/s]41083 examples [00:37, 1143.70 examples/s]41204 examples [00:37, 1160.29 examples/s]41321 examples [00:37, 1154.05 examples/s]41437 examples [00:37, 1124.66 examples/s]41550 examples [00:37, 1120.24 examples/s]41663 examples [00:37, 1102.15 examples/s]41774 examples [00:38, 1054.35 examples/s]41895 examples [00:38, 1094.21 examples/s]42006 examples [00:38, 1077.84 examples/s]42122 examples [00:38, 1100.98 examples/s]42236 examples [00:38, 1111.35 examples/s]42348 examples [00:38, 1110.48 examples/s]42460 examples [00:38, 1108.87 examples/s]42572 examples [00:38, 1046.74 examples/s]42687 examples [00:38, 1073.77 examples/s]42800 examples [00:38, 1089.28 examples/s]42910 examples [00:39, 1091.59 examples/s]43026 examples [00:39, 1108.94 examples/s]43138 examples [00:39, 1109.92 examples/s]43259 examples [00:39, 1136.25 examples/s]43379 examples [00:39, 1152.82 examples/s]43495 examples [00:39, 1114.26 examples/s]43612 examples [00:39, 1127.74 examples/s]43726 examples [00:39, 1103.20 examples/s]43837 examples [00:39, 1104.86 examples/s]43953 examples [00:39, 1120.47 examples/s]44066 examples [00:40, 1118.10 examples/s]44178 examples [00:40, 1118.29 examples/s]44290 examples [00:40, 1078.91 examples/s]44407 examples [00:40, 1102.13 examples/s]44522 examples [00:40, 1115.01 examples/s]44637 examples [00:40, 1123.43 examples/s]44751 examples [00:40, 1128.22 examples/s]44864 examples [00:40, 1067.15 examples/s]44978 examples [00:40, 1085.73 examples/s]45090 examples [00:41, 1094.58 examples/s]45200 examples [00:41, 1084.44 examples/s]45313 examples [00:41, 1096.85 examples/s]45423 examples [00:41, 1085.78 examples/s]45539 examples [00:41, 1106.71 examples/s]45655 examples [00:41, 1119.85 examples/s]45768 examples [00:41, 1109.16 examples/s]45883 examples [00:41, 1118.48 examples/s]45995 examples [00:41, 1048.88 examples/s]46101 examples [00:41, 1043.85 examples/s]46210 examples [00:42, 1056.71 examples/s]46325 examples [00:42, 1081.26 examples/s]46436 examples [00:42, 1087.69 examples/s]46546 examples [00:42, 1078.79 examples/s]46661 examples [00:42, 1097.06 examples/s]46775 examples [00:42, 1107.46 examples/s]46894 examples [00:42, 1130.48 examples/s]47008 examples [00:42, 1102.03 examples/s]47122 examples [00:42, 1112.15 examples/s]47234 examples [00:42, 1113.88 examples/s]47346 examples [00:43, 1080.74 examples/s]47458 examples [00:43, 1089.81 examples/s]47573 examples [00:43, 1106.21 examples/s]47684 examples [00:43, 1090.81 examples/s]47795 examples [00:43, 1094.43 examples/s]47910 examples [00:43, 1109.40 examples/s]48025 examples [00:43, 1120.06 examples/s]48138 examples [00:43, 1070.26 examples/s]48246 examples [00:43, 1071.32 examples/s]48355 examples [00:43, 1076.18 examples/s]48463 examples [00:44, 1067.61 examples/s]48583 examples [00:44, 1102.11 examples/s]48701 examples [00:44, 1120.66 examples/s]48815 examples [00:44, 1124.15 examples/s]48928 examples [00:44, 1109.68 examples/s]49047 examples [00:44, 1131.12 examples/s]49161 examples [00:44, 1132.79 examples/s]49277 examples [00:44, 1140.53 examples/s]49392 examples [00:44, 1083.52 examples/s]49502 examples [00:45, 1064.70 examples/s]49617 examples [00:45, 1087.87 examples/s]49731 examples [00:45, 1101.82 examples/s]49843 examples [00:45, 1106.55 examples/s]49954 examples [00:45, 1103.56 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 20%|█▉        | 9916/50000 [00:00<00:00, 99157.97 examples/s] 50%|████▉     | 24929/50000 [00:00<00:00, 110403.02 examples/s] 80%|███████▉  | 39999/50000 [00:00<00:00, 120029.98 examples/s]                                                                0 examples [00:00, ? examples/s]91 examples [00:00, 901.64 examples/s]206 examples [00:00, 963.18 examples/s]322 examples [00:00, 1014.74 examples/s]443 examples [00:00, 1064.64 examples/s]559 examples [00:00, 1090.12 examples/s]677 examples [00:00, 1115.26 examples/s]780 examples [00:00, 904.63 examples/s] 895 examples [00:00, 964.85 examples/s]1010 examples [00:00, 1013.49 examples/s]1115 examples [00:01, 1022.75 examples/s]1228 examples [00:01, 1051.84 examples/s]1338 examples [00:01, 1065.37 examples/s]1449 examples [00:01, 1075.97 examples/s]1562 examples [00:01, 1088.84 examples/s]1672 examples [00:01, 1074.99 examples/s]1785 examples [00:01, 1090.66 examples/s]1900 examples [00:01, 1107.16 examples/s]2013 examples [00:01, 1113.91 examples/s]2128 examples [00:01, 1122.86 examples/s]2241 examples [00:02, 1086.59 examples/s]2351 examples [00:02, 1044.66 examples/s]2458 examples [00:02, 1049.88 examples/s]2566 examples [00:02, 1057.18 examples/s]2683 examples [00:02, 1087.95 examples/s]2794 examples [00:02, 1091.88 examples/s]2913 examples [00:02, 1118.78 examples/s]3030 examples [00:02, 1132.92 examples/s]3144 examples [00:02, 1126.20 examples/s]3262 examples [00:03, 1140.92 examples/s]3377 examples [00:03, 1116.09 examples/s]3491 examples [00:03, 1121.45 examples/s]3604 examples [00:03, 1102.81 examples/s]3717 examples [00:03, 1110.77 examples/s]3832 examples [00:03, 1121.17 examples/s]3945 examples [00:03, 1104.29 examples/s]4057 examples [00:03, 1108.85 examples/s]4168 examples [00:03, 1085.52 examples/s]4277 examples [00:03, 1056.41 examples/s]4385 examples [00:04, 1059.81 examples/s]4500 examples [00:04, 1083.19 examples/s]4609 examples [00:04, 1083.39 examples/s]4718 examples [00:04, 1054.06 examples/s]4830 examples [00:04, 1070.91 examples/s]4940 examples [00:04, 1079.27 examples/s]5049 examples [00:04, 1080.89 examples/s]5164 examples [00:04, 1098.37 examples/s]5275 examples [00:04, 1093.70 examples/s]5387 examples [00:04, 1099.07 examples/s]5498 examples [00:05, 1096.18 examples/s]5612 examples [00:05, 1106.73 examples/s]5726 examples [00:05, 1115.61 examples/s]5838 examples [00:05, 1084.70 examples/s]5952 examples [00:05, 1097.64 examples/s]6062 examples [00:05, 1083.06 examples/s]6177 examples [00:05, 1092.84 examples/s]6288 examples [00:05, 1096.80 examples/s]6402 examples [00:05, 1108.65 examples/s]6518 examples [00:05, 1121.32 examples/s]6631 examples [00:06, 1110.56 examples/s]6744 examples [00:06, 1114.41 examples/s]6858 examples [00:06, 1119.66 examples/s]6974 examples [00:06, 1129.18 examples/s]7089 examples [00:06, 1135.16 examples/s]7203 examples [00:06, 1111.79 examples/s]7322 examples [00:06, 1131.44 examples/s]7440 examples [00:06, 1144.04 examples/s]7555 examples [00:06, 1126.09 examples/s]7671 examples [00:07, 1135.45 examples/s]7785 examples [00:07, 1115.96 examples/s]7902 examples [00:07, 1129.35 examples/s]8020 examples [00:07, 1142.01 examples/s]8135 examples [00:07, 1117.65 examples/s]8247 examples [00:07, 1104.69 examples/s]8358 examples [00:07, 1077.02 examples/s]8469 examples [00:07, 1083.99 examples/s]8580 examples [00:07, 1089.47 examples/s]8690 examples [00:07, 1085.63 examples/s]8801 examples [00:08, 1090.72 examples/s]8911 examples [00:08, 1075.38 examples/s]9019 examples [00:08, 1070.07 examples/s]9127 examples [00:08, 1064.82 examples/s]9234 examples [00:08, 1053.78 examples/s]9340 examples [00:08, 1041.00 examples/s]9445 examples [00:08, 1025.84 examples/s]9548 examples [00:08, 1014.52 examples/s]9653 examples [00:08, 1022.93 examples/s]9758 examples [00:08, 1029.38 examples/s]9862 examples [00:09, 1023.65 examples/s]9965 examples [00:09, 1007.83 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteO01LJF/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteO01LJF/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f987407e8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f987407e8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f987407e8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f98dff1b5c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f98e051cdd8>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/resnet34_benchmark_mnist.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/MNIST\/",
    "dataset":"MNIST",
    "data_type":"tch_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tch_dataset_start",
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"torchvision.datasets:MNIST",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars":false,
          "arg":{

          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f987407e8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f987407e8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f987407e8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f985644bdd8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f985644bf28>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/recommender\/",
    "dataset":"IMDB_sample.txt",
    "data_type":"csv_dataset",
    "batch_size":64,
    "train":true
  },
  "preprocessors":[
    {
      "uri":"mlmodels.model_tch.textcnn:split_train_valid",
      "args":{
        "frac":0.99
      }
    },
    {
      "uri":"mlmodels.model_tch.textcnn:create_tabular_dataset",
      "args":{
        "lang":"en",
        "pretrained_emb":"glove.6B.300d"
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f9848aff6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f9848aff6a8> 

  function with postional parmater data_info <function split_train_valid at 0x7f9848aff6a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f9848aff7b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f9848aff7b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f9848aff7b8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=95f1e3c9089f60512a532ebd9404fc23519de540af7ae37b9d9bf02a0583df54
  Stored in directory: /tmp/pip-ephem-wheel-cache-wctglvvm/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<25:06:25, 9.54kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:48:25, 13.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:31:06, 19.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:46:12, 27.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<6:07:22, 38.9kB/s].vector_cache/glove.6B.zip:   1%|          | 7.95M/862M [00:01<4:15:58, 55.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.9M/862M [00:01<2:58:27, 79.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.1M/862M [00:01<2:04:24, 113kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.7M/862M [00:01<1:26:50, 162kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.1M/862M [00:01<1:00:33, 231kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.2M/862M [00:01<42:17, 329kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.8M/862M [00:02<29:31, 468kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.7M/862M [00:02<20:41, 665kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.1M/862M [00:02<14:29, 944kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.2M/862M [00:02<10:11, 1.34MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:02<07:11, 1.88MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:03<05:53, 2.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<06:01, 2.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<06:34, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:05<05:10, 2.59MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:05<03:45, 3.55MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<36:35, 365kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<27:16, 490kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:07<19:25, 687kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:07<13:43, 969kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<40:10, 331kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<29:27, 451kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<21:11, 626kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:09<14:59, 883kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<18:42, 707kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<14:34, 906kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:11<10:31, 1.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<10:09, 1.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:13<09:55, 1.32MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:13<07:38, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:13<05:28, 2.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<35:21, 370kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:15<26:05, 501kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:15<18:31, 705kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<15:56, 817kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:17<13:48, 943kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:17<10:18, 1.26MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:17<07:19, 1.77MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<12:36:25, 17.1kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<8:50:34, 24.4kB/s] .vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:19<6:10:56, 34.8kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<4:21:58, 49.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<3:05:57, 69.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:21<2:10:41, 98.4kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<1:33:11, 137kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<1:06:31, 192kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:23<46:45, 273kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<35:37, 358kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<27:31, 463kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:24<19:53, 640kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<15:55, 796kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<12:27, 1.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<08:59, 1.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:11, 1.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:59, 1.40MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:49, 1.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<04:56, 2.54MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<09:38, 1.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:01, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<05:55, 2.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:03, 1.77MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:12, 2.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:38, 2.68MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:11, 2.00MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:50, 1.81MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:25, 2.28MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:47, 2.13MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:18, 2.32MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<03:59, 3.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:40, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:28, 1.89MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:08, 2.38MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<03:43, 3.28MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<11:47:10, 17.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<8:15:51, 24.6kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<5:46:41, 35.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<4:04:45, 49.5kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<2:53:45, 69.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<2:02:07, 99.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<1:25:15, 141kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<12:37:18, 15.9kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<8:51:04, 22.7kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<6:11:13, 32.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<4:21:54, 45.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<3:05:44, 64.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<2:10:29, 91.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<1:32:52, 128kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<1:06:11, 180kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<46:32, 255kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<35:15, 336kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<26:59, 439kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<19:23, 610kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<13:41, 862kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<15:57, 738kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<12:23, 950kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<08:55, 1.32MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<08:58, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:40, 1.35MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:34, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<04:55, 2.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:08, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:17, 2.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<03:56, 2.95MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<02:54, 3.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<48:40, 238kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<36:24, 318kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<25:56, 445kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<18:16, 631kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<18:32, 621kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<14:11, 811kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<10:12, 1.12MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<09:47, 1.17MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<08:00, 1.43MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:53, 1.94MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:48, 1.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:55, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:25, 2.56MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:46, 1.96MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<06:14, 1.81MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:56, 2.29MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:15, 2.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:50, 2.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<03:40, 3.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:09, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:44, 2.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<03:35, 3.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:08, 2.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:43, 2.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<03:34, 3.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:07, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:46, 1.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<04:36, 2.39MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<04:58, 2.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:37, 2.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<03:28, 3.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:58, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:41, 1.91MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:27, 2.44MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<03:16, 3.32MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:24, 1.46MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:07, 1.76MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<04:33, 2.37MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:43, 1.88MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:05, 2.11MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:49, 2.80MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:12, 2.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:48, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:32, 2.35MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:19, 3.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:30, 1.63MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:29, 1.63MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<04:56, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:35, 2.94MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<08:53, 1.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<08:08, 1.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<06:09, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:04, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:04, 1.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:39, 2.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<03:21, 3.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<21:37, 481kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<17:01, 611kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<12:21, 840kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<10:21, 998kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<09:07, 1.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:50, 1.51MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<06:30, 1.58MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<06:25, 1.60MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:52, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<03:32, 2.88MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:27, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:59, 1.46MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:21, 1.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:26, 1.86MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<05:38, 1.79MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:20, 2.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:08, 3.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<09:57, 1.01MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<08:46, 1.15MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<06:35, 1.53MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<06:16, 1.59MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<06:11, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<04:46, 2.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<05:00, 1.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:13, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:06, 2.42MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:31, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:56, 1.99MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:53, 2.53MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:22, 2.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:49, 2.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:48, 2.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:18, 2.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:45, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<03:45, 2.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:15, 2.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:39, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<03:41, 2.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:11, 2.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:39, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<03:37, 2.64MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<02:39, 3.59MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<07:47, 1.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<07:10, 1.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:22, 1.76MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:50, 2.46MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<1:54:37, 82.4kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<1:21:50, 115kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<57:38, 164kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<41:42, 225kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<30:58, 303kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<22:03, 424kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<16:56, 550kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<13:31, 688kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<09:51, 942kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<06:58, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<18:04, 511kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<14:18, 645kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<10:21, 890kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<07:19, 1.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<14:21, 639kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<13:17, 690kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<10:08, 903kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<07:15, 1.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<07:25, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:51, 1.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:08, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:44, 2.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:02, 1.50MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:47, 1.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:23, 2.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<03:10, 2.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<08:42, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<07:42, 1.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:47, 1.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:32, 1.61MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<05:29, 1.62MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:14, 2.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:26, 1.99MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:41, 1.88MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<03:40, 2.40MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:02, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:21, 2.01MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:26, 2.54MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:51, 2.25MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:57, 1.46MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:56, 1.75MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<03:37, 2.39MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:46, 1.81MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:53, 1.76MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:48, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:05, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<04:21, 1.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:25, 2.49MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:49, 2.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:12, 2.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:19, 2.55MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:44, 2.25MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:07, 2.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:15, 2.57MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:41, 2.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:28, 1.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:36, 1.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<03:22, 2.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:29, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:38, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:34, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<02:35, 3.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<08:21, 981kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<07:01, 1.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:11, 1.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<05:08, 1.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<05:03, 1.61MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<03:50, 2.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:36<02:46, 2.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<08:31, 947kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<08:46, 920kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<06:52, 1.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:56, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:32, 1.44MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:16, 1.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<04:02, 1.97MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:09, 1.91MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:20, 1.83MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:23, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:41, 2.13MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:59, 1.97MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:08, 2.49MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<03:30, 2.22MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:51, 2.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<02:59, 2.60MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:10, 3.56MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<08:04, 957kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<06:59, 1.10MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:14, 1.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:56, 1.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:50, 1.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:43, 2.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:40, 2.85MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<1:15:49, 100kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<54:07, 140kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<38:03, 199kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<27:53, 270kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<20:36, 365kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<14:39, 511kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<11:35, 643kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<09:27, 787kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<06:56, 1.07MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<06:04, 1.22MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<05:35, 1.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<04:13, 1.74MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:10, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:14, 1.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:14, 2.25MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<02:21, 3.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<07:04, 1.02MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<06:15, 1.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:38, 1.55MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<03:19, 2.16MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<09:11, 781kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<07:43, 928kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:43, 1.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:11, 1.37MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<04:55, 1.44MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:45, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:48, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:56, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:04, 2.29MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:18, 2.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:34, 1.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<02:45, 2.52MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:00, 3.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<07:41, 897kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<06:37, 1.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:56, 1.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:35, 1.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:35, 1.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<04:31, 1.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:18, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:04, 1.66MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<04:01, 1.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:35, 1.88MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:06, 2.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:40, 2.52MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:32, 2.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:15, 2.99MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:08, 3.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<01:56, 3.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:17, 2.93MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:09, 3.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<2:51:17, 39.1kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<2:01:16, 55.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<1:25:21, 78.3kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<1:00:12, 111kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<42:33, 157kB/s]  .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<30:16, 220kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<21:40, 307kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<15:39, 424kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<11:28, 578kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<12:43, 521kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<11:35, 572kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<08:45, 755kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<06:42, 985kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:07, 1.29MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:04, 1.62MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:14, 2.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:02, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:38, 2.49MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:20, 2.81MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<08:19, 788kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<08:29, 773kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<06:40, 981kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<05:05, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<04:07, 1.59MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:17, 1.98MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:40, 2.43MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:34, 2.53MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:11, 2.97MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:10, 2.99MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<01:56, 3.33MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<50:13, 129kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<37:47, 172kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<27:09, 239kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<19:29, 332kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<14:06, 458kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<10:20, 625kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<07:42, 837kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<05:51, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<04:33, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<07:06, 902kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<07:23, 867kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<05:53, 1.09MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:36, 1.39MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 479M/862M [03:26<03:40, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:01, 2.11MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:34, 2.48MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:14, 2.85MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:00, 3.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<06:47, 936kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<07:08, 890kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<05:40, 1.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:27, 1.42MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:35, 1.76MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:57, 2.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:28, 2.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:11, 2.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<01:57, 3.20MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<01:46, 3.55MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<08:39, 726kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<08:25, 746kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<06:27, 972kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:57, 1.26MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<03:52, 1.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<03:07, 2.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:35, 2.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:12, 2.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<01:53, 3.28MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<19:28, 319kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<15:38, 397kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<11:32, 538kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<08:25, 735kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<06:12, 997kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:37, 1.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:46, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:56, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:28, 2.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<24:43, 249kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<19:10, 320kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<13:56, 440kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<10:02, 610kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<07:17, 838kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<05:31, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:06, 1.48MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:14, 1.88MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<07:28, 814kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<08:11, 742kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<06:31, 930kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:49, 1.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<03:38, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:48, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:13, 2.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<09:58, 603kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<09:39, 622kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<07:18, 820kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<05:25, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<03:59, 1.49MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:00, 1.97MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<05:19, 1.12MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<05:56, 999kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<04:44, 1.25MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:30, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:37, 2.25MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<02:05, 2.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:15, 1.38MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<05:01, 1.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:55, 1.49MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:56, 1.99MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<02:18, 2.53MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<01:44, 3.34MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<07:03, 823kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<08:27, 685kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<06:50, 848kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:59, 1.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:40, 1.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<02:42, 2.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<06:28, 884kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<06:05, 941kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:34, 1.25MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:21, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<02:27, 2.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<11:53, 476kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<11:04, 511kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<08:24, 673kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<06:03, 930kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<04:20, 1.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<06:14, 897kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<06:37, 843kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<05:13, 1.07MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:47, 1.47MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<02:44, 2.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<05:12, 1.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<05:42, 968kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:31, 1.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<03:18, 1.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:31, 1.55MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<04:21, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:28, 1.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:33, 2.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:01, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:51, 1.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:04, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:15, 2.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:56, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:35, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:53, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:06, 2.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:59, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:36, 1.45MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:52, 1.82MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:06, 2.47MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<03:07, 1.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:40, 1.41MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:53, 1.78MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:07, 2.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:04, 1.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:23, 1.51MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<02:38, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:03, 2.46MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:28, 2.03MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:58, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:20, 1.51MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:29, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:50, 2.72MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:03, 1.63MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<04:20, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:33, 1.39MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:36, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<01:54, 2.57MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<30:04, 163kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<23:02, 213kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<16:36, 295kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<11:41, 416kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<09:14, 523kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<08:25, 574kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<06:21, 758kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<04:35, 1.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<03:16, 1.46MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<04:26, 1.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<04:55, 967kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:50, 1.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:46, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<02:02, 2.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:54, 1.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<04:26, 1.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:31, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:33, 1.82MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:54, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:39, 1.26MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:57, 1.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:11, 2.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:36, 2.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<03:54, 1.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<04:19, 1.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<03:21, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:27, 1.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:47, 1.61MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<03:24, 1.32MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:45, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:01, 2.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:29, 1.77MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<03:10, 1.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:35, 1.70MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<01:54, 2.29MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:24, 1.81MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<03:06, 1.40MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:31, 1.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:50, 2.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:25, 1.77MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<03:05, 1.38MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:27, 1.73MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:48, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:20, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<03:06, 1.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:28, 1.69MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:49, 2.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:21, 3.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<04:17, 964kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<04:19, 956kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<03:20, 1.24MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:24, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:49, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<03:13, 1.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:34, 1.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:52, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:25, 1.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<03:05, 1.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:30, 1.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:50, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:15, 1.75MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:47, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:12, 1.77MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:37, 2.40MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:12, 3.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<04:16, 903kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<04:15, 906kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<03:13, 1.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:19, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:41, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<05:24, 703kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<04:57, 766kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<03:42, 1.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:39, 1.41MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:51, 1.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:09, 1.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:29, 1.49MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:49, 2.03MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:14, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:41, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:07, 1.72MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:34, 2.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:45<01:09, 3.10MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<05:02, 712kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<04:41, 764kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:33, 1.01MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:32, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:43, 1.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<03:03, 1.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:22, 1.48MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:43, 2.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:15, 2.73MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<04:16, 808kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<04:01, 855kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:02, 1.13MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:11, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:25, 1.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:47, 1.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:10, 1.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:34, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:09, 2.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:40, 899kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:35, 923kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:43, 1.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:59, 1.65MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:26, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<03:56, 820kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<03:49, 846kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<02:53, 1.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<02:04, 1.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:31, 2.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<03:38, 870kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<03:35, 883kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:42, 1.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:57, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:25, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<03:07, 994kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<03:07, 990kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<02:25, 1.27MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<01:44, 1.76MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:15, 2.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<14:45, 205kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<11:15, 269kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<08:02, 376kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<05:37, 532kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<03:58, 747kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<06:06, 485kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<05:10, 572kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<03:47, 777kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<02:40, 1.09MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:56, 1.49MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<05:10, 559kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<04:30, 642kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<03:21, 858kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<02:23, 1.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:26, 1.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:34, 1.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:59, 1.41MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:26, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:48, 1.53MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:09, 1.28MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:42, 1.61MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:14, 2.19MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:34, 1.71MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:58, 1.36MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:33, 1.71MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:07, 2.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<00:50, 3.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<03:43, 704kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<03:18, 790kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:29, 1.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:46, 1.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:05, 1.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:09, 1.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:38, 1.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:20, 1.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:58, 2.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:53, 1.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:39, 929kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:10, 1.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:35, 1.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<01:09, 2.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:06, 1.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:38, 910kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:08, 1.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:32, 1.54MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<01:07, 2.11MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:56, 1.21MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:23, 977kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:54, 1.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:23, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<01:00, 2.25MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:19, 976kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:44, 830kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:10, 1.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:34, 1.42MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<01:08, 1.94MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:15, 973kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:34, 855kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:03, 1.07MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:29, 1.46MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<01:04, 1.99MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<02:27, 870kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<02:39, 800kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:05, 1.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:30, 1.39MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<01:05, 1.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<02:46, 745kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<02:52, 720kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:13, 923kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:35, 1.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<01:08, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<02:01, 985kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:18, 864kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:49, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:19, 1.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:56, 2.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:40, 720kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:43, 706kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:06, 908kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:31, 1.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<01:05, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:34, 722kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:37, 706kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:02, 907kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:27, 1.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<01:02, 1.74MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<02:11, 815kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<02:16, 787kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:46, 1.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:16, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:54, 1.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<02:30, 687kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<02:27, 700kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:53, 902kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:21, 1.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:57, 1.73MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:36, 1.02MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:47, 916kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:25, 1.15MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:02, 1.57MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:44, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<02:23, 662kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<02:22, 667kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:49, 865kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:18, 1.19MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:55, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<02:27, 616kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:46<02:23, 630kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:49, 822kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:18, 1.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:55, 1.57MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<02:24, 598kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<02:20, 617kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:46, 805kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:15, 1.12MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:53, 1.54MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<02:03, 667kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<02:03, 668kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:34, 867kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:07, 1.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:47, 1.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:54, 682kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:53, 690kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:27, 892kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:02, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:43, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<02:10, 568kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<02:01, 608kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<01:32, 795kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:05, 1.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:46, 1.53MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<02:05, 559kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<01:56, 602kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:27, 799kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:01, 1.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:44, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:32, 2.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<02:30, 438kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<02:13, 494kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:39, 656kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:09, 917kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:49, 1.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:56, 1.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:07, 912kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:53, 1.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:38, 1.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:27, 2.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:46, 1.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:55, 1.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:43, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:31, 1.76MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:22, 2.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<02:01, 440kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:46, 501kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:19, 666kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:55, 930kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:38, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<03:49, 215kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<03:00, 273kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<02:10, 374kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<01:29, 529kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<01:01, 744kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:11, 627kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:07, 668kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:51, 871kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:35, 1.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:24, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<01:32, 445kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<01:19, 512kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:59, 682kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:40, 950kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:35, 1.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:38, 948kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:30, 1.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:21, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:21, 1.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:26, 1.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:20, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:14, 2.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:16, 1.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:20, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:16, 1.67MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:08, 3.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:23, 1.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:24, 992kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:18, 1.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:12, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:13, 1.49MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:15, 1.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:11, 1.64MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:08, 2.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:05, 3.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:26, 599kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:23, 686kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:16, 923kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:10, 1.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:06, 1.77MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<01:10, 170kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:51, 227kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:35, 317kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:20, 449kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:14, 535kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:12, 632kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:08, 858kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:04, 1.20MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 1.09MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 1.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.40MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.93MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 437/400000 [00:00<01:31, 4368.73it/s]  0%|          | 1395/400000 [00:00<01:16, 5219.85it/s]  1%|          | 2364/400000 [00:00<01:05, 6057.55it/s]  1%|          | 3297/400000 [00:00<00:58, 6769.44it/s]  1%|          | 4254/400000 [00:00<00:53, 7420.50it/s]  1%|▏         | 5052/400000 [00:00<00:52, 7579.74it/s]  1%|▏         | 5985/400000 [00:00<00:49, 8031.11it/s]  2%|▏         | 6921/400000 [00:00<00:46, 8386.43it/s]  2%|▏         | 7848/400000 [00:00<00:45, 8630.99it/s]  2%|▏         | 8791/400000 [00:01<00:44, 8855.95it/s]  2%|▏         | 9687/400000 [00:01<00:43, 8885.63it/s]  3%|▎         | 10656/400000 [00:01<00:42, 9110.03it/s]  3%|▎         | 11597/400000 [00:01<00:42, 9196.46it/s]  3%|▎         | 12584/400000 [00:01<00:41, 9386.12it/s]  3%|▎         | 13561/400000 [00:01<00:40, 9496.32it/s]  4%|▎         | 14514/400000 [00:01<00:41, 9292.80it/s]  4%|▍         | 15460/400000 [00:01<00:41, 9340.78it/s]  4%|▍         | 16397/400000 [00:01<00:41, 9312.90it/s]  4%|▍         | 17346/400000 [00:01<00:40, 9362.82it/s]  5%|▍         | 18297/400000 [00:02<00:40, 9405.10it/s]  5%|▍         | 19239/400000 [00:02<00:41, 9240.18it/s]  5%|▌         | 20165/400000 [00:02<00:41, 9230.38it/s]  5%|▌         | 21101/400000 [00:02<00:40, 9266.37it/s]  6%|▌         | 22067/400000 [00:02<00:40, 9380.84it/s]  6%|▌         | 23042/400000 [00:02<00:39, 9487.03it/s]  6%|▌         | 23992/400000 [00:02<00:40, 9325.78it/s]  6%|▌         | 24926/400000 [00:02<00:40, 9269.89it/s]  6%|▋         | 25896/400000 [00:02<00:39, 9394.09it/s]  7%|▋         | 26862/400000 [00:02<00:39, 9472.30it/s]  7%|▋         | 27839/400000 [00:03<00:38, 9557.26it/s]  7%|▋         | 28796/400000 [00:03<00:39, 9311.12it/s]  7%|▋         | 29773/400000 [00:03<00:39, 9443.72it/s]  8%|▊         | 30744/400000 [00:03<00:38, 9522.04it/s]  8%|▊         | 31717/400000 [00:03<00:38, 9581.26it/s]  8%|▊         | 32693/400000 [00:03<00:38, 9616.46it/s]  8%|▊         | 33656/400000 [00:03<00:39, 9341.74it/s]  9%|▊         | 34617/400000 [00:03<00:38, 9419.66it/s]  9%|▉         | 35561/400000 [00:03<00:39, 9135.95it/s]  9%|▉         | 36530/400000 [00:03<00:39, 9293.68it/s]  9%|▉         | 37463/400000 [00:04<00:38, 9298.37it/s] 10%|▉         | 38395/400000 [00:04<00:39, 9105.97it/s] 10%|▉         | 39360/400000 [00:04<00:38, 9261.24it/s] 10%|█         | 40338/400000 [00:04<00:38, 9409.42it/s] 10%|█         | 41282/400000 [00:04<00:38, 9290.99it/s] 11%|█         | 42213/400000 [00:04<00:40, 8739.98it/s] 11%|█         | 43095/400000 [00:04<00:42, 8439.08it/s] 11%|█         | 43947/400000 [00:04<00:42, 8396.37it/s] 11%|█         | 44906/400000 [00:04<00:40, 8721.17it/s] 11%|█▏        | 45874/400000 [00:05<00:39, 8986.91it/s] 12%|█▏        | 46827/400000 [00:05<00:38, 9142.24it/s] 12%|█▏        | 47747/400000 [00:05<00:38, 9064.54it/s] 12%|█▏        | 48697/400000 [00:05<00:38, 9190.46it/s] 12%|█▏        | 49620/400000 [00:05<00:39, 8924.28it/s] 13%|█▎        | 50543/400000 [00:05<00:38, 9013.56it/s] 13%|█▎        | 51505/400000 [00:05<00:37, 9185.83it/s] 13%|█▎        | 52427/400000 [00:05<00:38, 9110.10it/s] 13%|█▎        | 53351/400000 [00:05<00:37, 9147.98it/s] 14%|█▎        | 54268/400000 [00:05<00:37, 9138.78it/s] 14%|█▍        | 55190/400000 [00:06<00:37, 9162.44it/s] 14%|█▍        | 56120/400000 [00:06<00:37, 9200.87it/s] 14%|█▍        | 57041/400000 [00:06<00:37, 9106.37it/s] 14%|█▍        | 57999/400000 [00:06<00:37, 9240.85it/s] 15%|█▍        | 58964/400000 [00:06<00:36, 9358.95it/s] 15%|█▍        | 59922/400000 [00:06<00:36, 9421.79it/s] 15%|█▌        | 60888/400000 [00:06<00:35, 9490.18it/s] 15%|█▌        | 61838/400000 [00:06<00:37, 9019.43it/s] 16%|█▌        | 62767/400000 [00:06<00:37, 9097.08it/s] 16%|█▌        | 63681/400000 [00:06<00:38, 8768.71it/s] 16%|█▌        | 64597/400000 [00:07<00:37, 8881.85it/s] 16%|█▋        | 65554/400000 [00:07<00:36, 9076.73it/s] 17%|█▋        | 66466/400000 [00:07<00:37, 9013.07it/s] 17%|█▋        | 67443/400000 [00:07<00:36, 9226.87it/s] 17%|█▋        | 68372/400000 [00:07<00:35, 9245.23it/s] 17%|█▋        | 69342/400000 [00:07<00:35, 9375.37it/s] 18%|█▊        | 70300/400000 [00:07<00:34, 9434.26it/s] 18%|█▊        | 71245/400000 [00:07<00:35, 9269.68it/s] 18%|█▊        | 72178/400000 [00:07<00:35, 9286.20it/s] 18%|█▊        | 73147/400000 [00:07<00:34, 9401.35it/s] 19%|█▊        | 74129/400000 [00:08<00:34, 9522.17it/s] 19%|█▉        | 75083/400000 [00:08<00:34, 9497.37it/s] 19%|█▉        | 76034/400000 [00:08<00:34, 9329.70it/s] 19%|█▉        | 76992/400000 [00:08<00:34, 9400.83it/s] 19%|█▉        | 77963/400000 [00:08<00:33, 9490.05it/s] 20%|█▉        | 78913/400000 [00:08<00:34, 9344.22it/s] 20%|█▉        | 79849/400000 [00:08<00:34, 9288.74it/s] 20%|██        | 80779/400000 [00:08<00:34, 9220.75it/s] 20%|██        | 81720/400000 [00:08<00:34, 9275.82it/s] 21%|██        | 82694/400000 [00:08<00:33, 9409.18it/s] 21%|██        | 83636/400000 [00:09<00:34, 9286.39it/s] 21%|██        | 84566/400000 [00:09<00:34, 9224.95it/s] 21%|██▏       | 85491/400000 [00:09<00:34, 9230.84it/s] 22%|██▏       | 86455/400000 [00:09<00:33, 9347.12it/s] 22%|██▏       | 87391/400000 [00:09<00:33, 9245.55it/s] 22%|██▏       | 88352/400000 [00:09<00:33, 9351.33it/s] 22%|██▏       | 89288/400000 [00:09<00:33, 9291.20it/s] 23%|██▎       | 90218/400000 [00:09<00:33, 9197.80it/s] 23%|██▎       | 91189/400000 [00:09<00:33, 9345.48it/s] 23%|██▎       | 92147/400000 [00:10<00:32, 9414.52it/s] 23%|██▎       | 93101/400000 [00:10<00:32, 9449.64it/s] 24%|██▎       | 94047/400000 [00:10<00:33, 9201.30it/s] 24%|██▎       | 94992/400000 [00:10<00:32, 9273.24it/s] 24%|██▍       | 95956/400000 [00:10<00:32, 9372.99it/s] 24%|██▍       | 96938/400000 [00:10<00:31, 9500.65it/s] 24%|██▍       | 97916/400000 [00:10<00:31, 9582.35it/s] 25%|██▍       | 98876/400000 [00:10<00:32, 9391.57it/s] 25%|██▍       | 99817/400000 [00:10<00:32, 9295.79it/s] 25%|██▌       | 100748/400000 [00:10<00:32, 9293.99it/s] 25%|██▌       | 101727/400000 [00:11<00:31, 9437.34it/s] 26%|██▌       | 102672/400000 [00:11<00:31, 9315.57it/s] 26%|██▌       | 103605/400000 [00:11<00:32, 9114.04it/s] 26%|██▌       | 104526/400000 [00:11<00:32, 9141.97it/s] 26%|██▋       | 105506/400000 [00:11<00:31, 9328.92it/s] 27%|██▋       | 106488/400000 [00:11<00:30, 9469.47it/s] 27%|██▋       | 107437/400000 [00:11<00:30, 9461.44it/s] 27%|██▋       | 108385/400000 [00:11<00:33, 8688.13it/s] 27%|██▋       | 109304/400000 [00:11<00:32, 8830.61it/s] 28%|██▊       | 110231/400000 [00:11<00:32, 8955.75it/s] 28%|██▊       | 111154/400000 [00:12<00:31, 9035.79it/s] 28%|██▊       | 112092/400000 [00:12<00:31, 9135.33it/s] 28%|██▊       | 113010/400000 [00:12<00:31, 8984.03it/s] 28%|██▊       | 113918/400000 [00:12<00:31, 9011.26it/s] 29%|██▊       | 114822/400000 [00:12<00:31, 8937.28it/s] 29%|██▉       | 115718/400000 [00:12<00:31, 8929.70it/s] 29%|██▉       | 116636/400000 [00:12<00:31, 9001.26it/s] 29%|██▉       | 117538/400000 [00:12<00:32, 8806.99it/s] 30%|██▉       | 118458/400000 [00:12<00:31, 8920.31it/s] 30%|██▉       | 119354/400000 [00:12<00:31, 8929.75it/s] 30%|███       | 120249/400000 [00:13<00:31, 8923.61it/s] 30%|███       | 121175/400000 [00:13<00:30, 9019.64it/s] 31%|███       | 122078/400000 [00:13<00:31, 8843.63it/s] 31%|███       | 122975/400000 [00:13<00:31, 8880.83it/s] 31%|███       | 123870/400000 [00:13<00:31, 8898.58it/s] 31%|███       | 124810/400000 [00:13<00:30, 9040.71it/s] 31%|███▏      | 125750/400000 [00:13<00:29, 9144.63it/s] 32%|███▏      | 126666/400000 [00:13<00:30, 9076.44it/s] 32%|███▏      | 127629/400000 [00:13<00:29, 9234.53it/s] 32%|███▏      | 128558/400000 [00:13<00:29, 9249.40it/s] 32%|███▏      | 129542/400000 [00:14<00:28, 9417.29it/s] 33%|███▎      | 130527/400000 [00:14<00:28, 9540.63it/s] 33%|███▎      | 131483/400000 [00:14<00:28, 9371.24it/s] 33%|███▎      | 132470/400000 [00:14<00:28, 9513.15it/s] 33%|███▎      | 133423/400000 [00:14<00:28, 9508.61it/s] 34%|███▎      | 134411/400000 [00:14<00:27, 9615.21it/s] 34%|███▍      | 135391/400000 [00:14<00:27, 9669.84it/s] 34%|███▍      | 136359/400000 [00:14<00:28, 9398.02it/s] 34%|███▍      | 137302/400000 [00:14<00:29, 9019.37it/s] 35%|███▍      | 138212/400000 [00:15<00:28, 9041.76it/s] 35%|███▍      | 139140/400000 [00:15<00:28, 9111.28it/s] 35%|███▌      | 140054/400000 [00:15<00:28, 9110.06it/s] 35%|███▌      | 140967/400000 [00:15<00:29, 8896.21it/s] 35%|███▌      | 141876/400000 [00:15<00:28, 8950.95it/s] 36%|███▌      | 142827/400000 [00:15<00:28, 9110.20it/s] 36%|███▌      | 143793/400000 [00:15<00:27, 9267.74it/s] 36%|███▌      | 144747/400000 [00:15<00:27, 9345.97it/s] 36%|███▋      | 145684/400000 [00:15<00:27, 9155.16it/s] 37%|███▋      | 146630/400000 [00:15<00:27, 9241.88it/s] 37%|███▋      | 147585/400000 [00:16<00:27, 9329.88it/s] 37%|███▋      | 148561/400000 [00:16<00:26, 9452.46it/s] 37%|███▋      | 149525/400000 [00:16<00:26, 9507.03it/s] 38%|███▊      | 150477/400000 [00:16<00:27, 9069.20it/s] 38%|███▊      | 151389/400000 [00:16<00:28, 8815.07it/s] 38%|███▊      | 152323/400000 [00:16<00:27, 8963.38it/s] 38%|███▊      | 153269/400000 [00:16<00:27, 9104.33it/s] 39%|███▊      | 154183/400000 [00:16<00:27, 8941.21it/s] 39%|███▉      | 155081/400000 [00:16<00:27, 8832.17it/s] 39%|███▉      | 155977/400000 [00:16<00:27, 8868.06it/s] 39%|███▉      | 156941/400000 [00:17<00:26, 9084.70it/s] 39%|███▉      | 157861/400000 [00:17<00:26, 9118.15it/s] 40%|███▉      | 158775/400000 [00:17<00:26, 9042.41it/s] 40%|███▉      | 159681/400000 [00:17<00:26, 8994.75it/s] 40%|████      | 160649/400000 [00:17<00:26, 9188.18it/s] 40%|████      | 161627/400000 [00:17<00:25, 9357.09it/s] 41%|████      | 162565/400000 [00:17<00:26, 9126.24it/s] 41%|████      | 163481/400000 [00:17<00:26, 8963.36it/s] 41%|████      | 164380/400000 [00:17<00:26, 8767.38it/s] 41%|████▏     | 165293/400000 [00:18<00:26, 8873.03it/s] 42%|████▏     | 166263/400000 [00:18<00:25, 9105.07it/s] 42%|████▏     | 167194/400000 [00:18<00:25, 9165.59it/s] 42%|████▏     | 168173/400000 [00:18<00:24, 9343.26it/s] 42%|████▏     | 169110/400000 [00:18<00:24, 9258.22it/s] 43%|████▎     | 170085/400000 [00:18<00:24, 9398.99it/s] 43%|████▎     | 171065/400000 [00:18<00:24, 9515.74it/s] 43%|████▎     | 172019/400000 [00:18<00:24, 9494.29it/s] 43%|████▎     | 172981/400000 [00:18<00:23, 9530.67it/s] 43%|████▎     | 173935/400000 [00:18<00:24, 9372.32it/s] 44%|████▎     | 174911/400000 [00:19<00:23, 9483.63it/s] 44%|████▍     | 175861/400000 [00:19<00:23, 9382.67it/s] 44%|████▍     | 176801/400000 [00:19<00:23, 9350.71it/s] 44%|████▍     | 177780/400000 [00:19<00:23, 9475.60it/s] 45%|████▍     | 178729/400000 [00:19<00:23, 9325.90it/s] 45%|████▍     | 179694/400000 [00:19<00:23, 9418.12it/s] 45%|████▌     | 180637/400000 [00:19<00:23, 9209.07it/s] 45%|████▌     | 181618/400000 [00:19<00:23, 9379.36it/s] 46%|████▌     | 182558/400000 [00:19<00:23, 9195.92it/s] 46%|████▌     | 183480/400000 [00:19<00:24, 9021.32it/s] 46%|████▌     | 184441/400000 [00:20<00:23, 9188.45it/s] 46%|████▋     | 185437/400000 [00:20<00:22, 9405.27it/s] 47%|████▋     | 186385/400000 [00:20<00:22, 9424.91it/s] 47%|████▋     | 187379/400000 [00:20<00:22, 9571.77it/s] 47%|████▋     | 188339/400000 [00:20<00:22, 9449.56it/s] 47%|████▋     | 189305/400000 [00:20<00:22, 9509.88it/s] 48%|████▊     | 190313/400000 [00:20<00:21, 9671.92it/s] 48%|████▊     | 191314/400000 [00:20<00:21, 9770.60it/s] 48%|████▊     | 192293/400000 [00:20<00:21, 9678.59it/s] 48%|████▊     | 193263/400000 [00:20<00:21, 9544.01it/s] 49%|████▊     | 194257/400000 [00:21<00:21, 9658.14it/s] 49%|████▉     | 195224/400000 [00:21<00:21, 9602.77it/s] 49%|████▉     | 196211/400000 [00:21<00:21, 9679.99it/s] 49%|████▉     | 197184/400000 [00:21<00:20, 9694.65it/s] 50%|████▉     | 198155/400000 [00:21<00:21, 9520.03it/s] 50%|████▉     | 199109/400000 [00:21<00:21, 9426.55it/s] 50%|█████     | 200068/400000 [00:21<00:21, 9473.11it/s] 50%|█████     | 201017/400000 [00:21<00:21, 9386.75it/s] 50%|█████     | 201957/400000 [00:21<00:21, 9366.97it/s] 51%|█████     | 202895/400000 [00:21<00:21, 9231.83it/s] 51%|█████     | 203879/400000 [00:22<00:20, 9406.14it/s] 51%|█████     | 204869/400000 [00:22<00:20, 9548.64it/s] 51%|█████▏    | 205849/400000 [00:22<00:20, 9621.85it/s] 52%|█████▏    | 206834/400000 [00:22<00:19, 9686.65it/s] 52%|█████▏    | 207804/400000 [00:22<00:20, 9464.58it/s] 52%|█████▏    | 208791/400000 [00:22<00:19, 9579.42it/s] 52%|█████▏    | 209751/400000 [00:22<00:19, 9544.91it/s] 53%|█████▎    | 210707/400000 [00:22<00:20, 9303.86it/s] 53%|█████▎    | 211679/400000 [00:22<00:19, 9422.39it/s] 53%|█████▎    | 212624/400000 [00:23<00:20, 9297.18it/s] 53%|█████▎    | 213556/400000 [00:23<00:20, 9224.10it/s] 54%|█████▎    | 214517/400000 [00:23<00:19, 9335.83it/s] 54%|█████▍    | 215452/400000 [00:23<00:19, 9320.08it/s] 54%|█████▍    | 216402/400000 [00:23<00:19, 9372.47it/s] 54%|█████▍    | 217340/400000 [00:23<00:20, 9115.19it/s] 55%|█████▍    | 218272/400000 [00:23<00:19, 9174.89it/s] 55%|█████▍    | 219226/400000 [00:23<00:19, 9278.94it/s] 55%|█████▌    | 220156/400000 [00:23<00:19, 9135.13it/s] 55%|█████▌    | 221110/400000 [00:23<00:19, 9250.33it/s] 56%|█████▌    | 222037/400000 [00:24<00:19, 9124.64it/s] 56%|█████▌    | 222951/400000 [00:24<00:19, 9106.61it/s] 56%|█████▌    | 223913/400000 [00:24<00:19, 9254.29it/s] 56%|█████▌    | 224840/400000 [00:24<00:19, 9101.52it/s] 56%|█████▋    | 225802/400000 [00:24<00:18, 9248.64it/s] 57%|█████▋    | 226729/400000 [00:24<00:19, 8971.61it/s] 57%|█████▋    | 227679/400000 [00:24<00:18, 9121.65it/s] 57%|█████▋    | 228638/400000 [00:24<00:18, 9257.07it/s] 57%|█████▋    | 229569/400000 [00:24<00:18, 9270.34it/s] 58%|█████▊    | 230498/400000 [00:24<00:18, 9156.31it/s] 58%|█████▊    | 231416/400000 [00:25<00:18, 9032.59it/s] 58%|█████▊    | 232387/400000 [00:25<00:18, 9225.47it/s] 58%|█████▊    | 233369/400000 [00:25<00:17, 9394.71it/s] 59%|█████▊    | 234356/400000 [00:25<00:17, 9530.43it/s] 59%|█████▉    | 235331/400000 [00:25<00:17, 9589.79it/s] 59%|█████▉    | 236292/400000 [00:25<00:17, 9254.03it/s] 59%|█████▉    | 237261/400000 [00:25<00:17, 9379.98it/s] 60%|█████▉    | 238202/400000 [00:25<00:17, 9371.46it/s] 60%|█████▉    | 239142/400000 [00:25<00:17, 9346.66it/s] 60%|██████    | 240086/400000 [00:25<00:17, 9374.09it/s] 60%|██████    | 241025/400000 [00:26<00:17, 9252.00it/s] 61%|██████    | 242005/400000 [00:26<00:16, 9406.52it/s] 61%|██████    | 242982/400000 [00:26<00:16, 9511.60it/s] 61%|██████    | 243945/400000 [00:26<00:16, 9546.72it/s] 61%|██████    | 244916/400000 [00:26<00:16, 9593.83it/s] 61%|██████▏   | 245877/400000 [00:26<00:16, 9466.45it/s] 62%|██████▏   | 246825/400000 [00:26<00:16, 9361.60it/s] 62%|██████▏   | 247768/400000 [00:26<00:16, 9382.00it/s] 62%|██████▏   | 248707/400000 [00:26<00:16, 9279.75it/s] 62%|██████▏   | 249671/400000 [00:26<00:16, 9384.21it/s] 63%|██████▎   | 250611/400000 [00:27<00:16, 9254.44it/s] 63%|██████▎   | 251600/400000 [00:27<00:15, 9435.92it/s] 63%|██████▎   | 252563/400000 [00:27<00:15, 9491.20it/s] 63%|██████▎   | 253524/400000 [00:27<00:15, 9526.50it/s] 64%|██████▎   | 254478/400000 [00:27<00:15, 9380.10it/s] 64%|██████▍   | 255418/400000 [00:27<00:15, 9197.00it/s] 64%|██████▍   | 256395/400000 [00:27<00:15, 9359.96it/s] 64%|██████▍   | 257333/400000 [00:27<00:15, 9328.67it/s] 65%|██████▍   | 258314/400000 [00:27<00:14, 9465.85it/s] 65%|██████▍   | 259280/400000 [00:28<00:14, 9521.90it/s] 65%|██████▌   | 260234/400000 [00:28<00:14, 9389.82it/s] 65%|██████▌   | 261194/400000 [00:28<00:14, 9450.61it/s] 66%|██████▌   | 262167/400000 [00:28<00:14, 9532.50it/s] 66%|██████▌   | 263141/400000 [00:28<00:14, 9593.21it/s] 66%|██████▌   | 264101/400000 [00:28<00:14, 9586.04it/s] 66%|██████▋   | 265061/400000 [00:28<00:14, 9469.18it/s] 67%|██████▋   | 266009/400000 [00:28<00:14, 9440.72it/s] 67%|██████▋   | 266954/400000 [00:28<00:14, 9395.88it/s] 67%|██████▋   | 267922/400000 [00:28<00:13, 9479.06it/s] 67%|██████▋   | 268871/400000 [00:29<00:14, 8998.14it/s] 67%|██████▋   | 269777/400000 [00:29<00:14, 8912.66it/s] 68%|██████▊   | 270758/400000 [00:29<00:14, 9161.93it/s] 68%|██████▊   | 271711/400000 [00:29<00:13, 9268.37it/s] 68%|██████▊   | 272687/400000 [00:29<00:13, 9407.52it/s] 68%|██████▊   | 273631/400000 [00:29<00:13, 9128.34it/s] 69%|██████▊   | 274568/400000 [00:29<00:13, 9197.94it/s] 69%|██████▉   | 275539/400000 [00:29<00:13, 9345.76it/s] 69%|██████▉   | 276517/400000 [00:29<00:13, 9471.68it/s] 69%|██████▉   | 277488/400000 [00:29<00:12, 9539.67it/s] 70%|██████▉   | 278444/400000 [00:30<00:13, 9239.91it/s] 70%|██████▉   | 279380/400000 [00:30<00:13, 9275.11it/s] 70%|███████   | 280370/400000 [00:30<00:12, 9453.25it/s] 70%|███████   | 281351/400000 [00:30<00:12, 9554.58it/s] 71%|███████   | 282336/400000 [00:30<00:12, 9640.24it/s] 71%|███████   | 283302/400000 [00:30<00:12, 9230.27it/s] 71%|███████   | 284232/400000 [00:30<00:12, 9248.90it/s] 71%|███████▏  | 285209/400000 [00:30<00:12, 9398.40it/s] 72%|███████▏  | 286180/400000 [00:30<00:11, 9487.33it/s] 72%|███████▏  | 287138/400000 [00:30<00:11, 9513.56it/s] 72%|███████▏  | 288091/400000 [00:31<00:11, 9409.30it/s] 72%|███████▏  | 289034/400000 [00:31<00:11, 9304.77it/s] 72%|███████▏  | 289990/400000 [00:31<00:11, 9379.17it/s] 73%|███████▎  | 290939/400000 [00:31<00:11, 9409.32it/s] 73%|███████▎  | 291895/400000 [00:31<00:11, 9453.05it/s] 73%|███████▎  | 292841/400000 [00:31<00:11, 9208.29it/s] 73%|███████▎  | 293811/400000 [00:31<00:11, 9348.80it/s] 74%|███████▎  | 294748/400000 [00:31<00:11, 8842.24it/s] 74%|███████▍  | 295650/400000 [00:31<00:11, 8893.52it/s] 74%|███████▍  | 296614/400000 [00:32<00:11, 9102.67it/s] 74%|███████▍  | 297529/400000 [00:32<00:11, 9060.50it/s] 75%|███████▍  | 298468/400000 [00:32<00:11, 9155.90it/s] 75%|███████▍  | 299433/400000 [00:32<00:10, 9298.37it/s] 75%|███████▌  | 300398/400000 [00:32<00:10, 9400.20it/s] 75%|███████▌  | 301341/400000 [00:32<00:10, 9406.25it/s] 76%|███████▌  | 302283/400000 [00:32<00:10, 9204.14it/s] 76%|███████▌  | 303206/400000 [00:32<00:10, 9198.99it/s] 76%|███████▌  | 304128/400000 [00:32<00:10, 9204.60it/s] 76%|███████▋  | 305101/400000 [00:32<00:10, 9355.62it/s] 77%|███████▋  | 306079/400000 [00:33<00:09, 9478.52it/s] 77%|███████▋  | 307029/400000 [00:33<00:09, 9334.49it/s] 77%|███████▋  | 308014/400000 [00:33<00:09, 9480.81it/s] 77%|███████▋  | 308985/400000 [00:33<00:09, 9546.50it/s] 77%|███████▋  | 309959/400000 [00:33<00:09, 9601.63it/s] 78%|███████▊  | 310946/400000 [00:33<00:09, 9678.16it/s] 78%|███████▊  | 311915/400000 [00:33<00:09, 9470.97it/s] 78%|███████▊  | 312864/400000 [00:33<00:09, 8956.03it/s] 78%|███████▊  | 313781/400000 [00:33<00:09, 9017.30it/s] 79%|███████▊  | 314736/400000 [00:33<00:09, 9169.86it/s] 79%|███████▉  | 315715/400000 [00:34<00:09, 9345.69it/s] 79%|███████▉  | 316654/400000 [00:34<00:09, 8983.06it/s] 79%|███████▉  | 317636/400000 [00:34<00:08, 9216.70it/s] 80%|███████▉  | 318578/400000 [00:34<00:08, 9274.56it/s] 80%|███████▉  | 319521/400000 [00:34<00:08, 9317.34it/s] 80%|████████  | 320472/400000 [00:34<00:08, 9372.35it/s] 80%|████████  | 321412/400000 [00:34<00:08, 9031.61it/s] 81%|████████  | 322384/400000 [00:34<00:08, 9225.88it/s] 81%|████████  | 323344/400000 [00:34<00:08, 9333.81it/s] 81%|████████  | 324320/400000 [00:34<00:08, 9457.53it/s] 81%|████████▏ | 325278/400000 [00:35<00:07, 9491.67it/s] 82%|████████▏ | 326229/400000 [00:35<00:07, 9334.88it/s] 82%|████████▏ | 327180/400000 [00:35<00:07, 9386.10it/s] 82%|████████▏ | 328138/400000 [00:35<00:07, 9441.86it/s] 82%|████████▏ | 329105/400000 [00:35<00:07, 9506.60it/s] 83%|████████▎ | 330057/400000 [00:35<00:07, 9503.47it/s] 83%|████████▎ | 331008/400000 [00:35<00:07, 9362.69it/s] 83%|████████▎ | 331985/400000 [00:35<00:07, 9480.82it/s] 83%|████████▎ | 332935/400000 [00:35<00:07, 9483.81it/s] 83%|████████▎ | 333897/400000 [00:35<00:06, 9522.92it/s] 84%|████████▎ | 334854/400000 [00:36<00:06, 9536.82it/s] 84%|████████▍ | 335809/400000 [00:36<00:06, 9382.23it/s] 84%|████████▍ | 336791/400000 [00:36<00:06, 9507.13it/s] 84%|████████▍ | 337751/400000 [00:36<00:06, 9532.86it/s] 85%|████████▍ | 338713/400000 [00:36<00:06, 9557.46it/s] 85%|████████▍ | 339683/400000 [00:36<00:06, 9597.50it/s] 85%|████████▌ | 340644/400000 [00:36<00:06, 9338.64it/s] 85%|████████▌ | 341593/400000 [00:36<00:06, 9381.67it/s] 86%|████████▌ | 342533/400000 [00:36<00:06, 9035.44it/s] 86%|████████▌ | 343441/400000 [00:37<00:06, 8892.46it/s] 86%|████████▌ | 344355/400000 [00:37<00:06, 8963.08it/s] 86%|████████▋ | 345256/400000 [00:37<00:06, 8975.82it/s] 87%|████████▋ | 346235/400000 [00:37<00:05, 9205.11it/s] 87%|████████▋ | 347159/400000 [00:37<00:05, 9185.42it/s] 87%|████████▋ | 348142/400000 [00:37<00:05, 9367.19it/s] 87%|████████▋ | 349109/400000 [00:37<00:05, 9453.39it/s] 88%|████████▊ | 350057/400000 [00:37<00:05, 9336.86it/s] 88%|████████▊ | 351001/400000 [00:37<00:05, 9365.23it/s] 88%|████████▊ | 351968/400000 [00:37<00:05, 9454.49it/s] 88%|████████▊ | 352946/400000 [00:38<00:04, 9548.77it/s] 88%|████████▊ | 353905/400000 [00:38<00:04, 9559.35it/s] 89%|████████▊ | 354862/400000 [00:38<00:04, 9338.57it/s] 89%|████████▉ | 355845/400000 [00:38<00:04, 9477.89it/s] 89%|████████▉ | 356795/400000 [00:38<00:04, 9302.39it/s] 89%|████████▉ | 357766/400000 [00:38<00:04, 9420.68it/s] 90%|████████▉ | 358742/400000 [00:38<00:04, 9517.54it/s] 90%|████████▉ | 359696/400000 [00:38<00:04, 9173.40it/s] 90%|█████████ | 360650/400000 [00:38<00:04, 9274.59it/s] 90%|█████████ | 361581/400000 [00:38<00:04, 9180.21it/s] 91%|█████████ | 362555/400000 [00:39<00:04, 9338.67it/s] 91%|█████████ | 363492/400000 [00:39<00:03, 9291.92it/s] 91%|█████████ | 364423/400000 [00:39<00:03, 9168.74it/s] 91%|█████████▏| 365391/400000 [00:39<00:03, 9316.02it/s] 92%|█████████▏| 366371/400000 [00:39<00:03, 9454.82it/s] 92%|█████████▏| 367340/400000 [00:39<00:03, 9521.60it/s] 92%|█████████▏| 368294/400000 [00:39<00:03, 9499.42it/s] 92%|█████████▏| 369245/400000 [00:39<00:03, 9355.48it/s] 93%|█████████▎| 370225/400000 [00:39<00:03, 9483.40it/s] 93%|█████████▎| 371175/400000 [00:39<00:03, 9370.89it/s] 93%|█████████▎| 372114/400000 [00:40<00:03, 9226.97it/s] 93%|█████████▎| 373073/400000 [00:40<00:02, 9330.38it/s] 94%|█████████▎| 374008/400000 [00:40<00:02, 9276.04it/s] 94%|█████████▎| 374985/400000 [00:40<00:02, 9416.45it/s] 94%|█████████▍| 375963/400000 [00:40<00:02, 9521.83it/s] 94%|█████████▍| 376917/400000 [00:40<00:02, 9485.16it/s] 94%|█████████▍| 377867/400000 [00:40<00:02, 9468.50it/s] 95%|█████████▍| 378815/400000 [00:40<00:02, 9420.60it/s] 95%|█████████▍| 379781/400000 [00:40<00:02, 9490.00it/s] 95%|█████████▌| 380739/400000 [00:40<00:02, 9513.72it/s] 95%|█████████▌| 381719/400000 [00:41<00:01, 9597.16it/s] 96%|█████████▌| 382680/400000 [00:41<00:01, 9467.22it/s] 96%|█████████▌| 383628/400000 [00:41<00:01, 9466.04it/s] 96%|█████████▌| 384587/400000 [00:41<00:01, 9501.80it/s] 96%|█████████▋| 385561/400000 [00:41<00:01, 9571.88it/s] 97%|█████████▋| 386519/400000 [00:41<00:01, 9376.70it/s] 97%|█████████▋| 387487/400000 [00:41<00:01, 9462.80it/s] 97%|█████████▋| 388435/400000 [00:41<00:01, 9218.57it/s] 97%|█████████▋| 389386/400000 [00:41<00:01, 9303.86it/s] 98%|█████████▊| 390348/400000 [00:42<00:01, 9396.07it/s] 98%|█████████▊| 391289/400000 [00:42<00:00, 9320.60it/s] 98%|█████████▊| 392223/400000 [00:42<00:00, 9292.24it/s] 98%|█████████▊| 393154/400000 [00:42<00:00, 9251.72it/s] 99%|█████████▊| 394140/400000 [00:42<00:00, 9424.79it/s] 99%|█████████▉| 395136/400000 [00:42<00:00, 9576.57it/s] 99%|█████████▉| 396096/400000 [00:42<00:00, 9449.36it/s] 99%|█████████▉| 397043/400000 [00:42<00:00, 9406.80it/s] 99%|█████████▉| 397985/400000 [00:42<00:00, 9346.40it/s]100%|█████████▉| 398962/400000 [00:42<00:00, 9467.84it/s]100%|█████████▉| 399923/400000 [00:43<00:00, 9508.55it/s]100%|█████████▉| 399999/400000 [00:43<00:00, 9293.30it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f984b88c128>, <torchtext.data.dataset.TabularDataset object at 0x7f984b88c278>, <torchtext.vocab.Vocab object at 0x7f984b88c198>), {}) 


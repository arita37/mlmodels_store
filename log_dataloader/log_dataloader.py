
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fddd35032f0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fddd35032f0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fde455540d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fde455540d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fdddcce99d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fdddcce99d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fddf2e388c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fddf2e388c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fddf2e388c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3047424/9912422 [00:00<00:00, 30335116.49it/s]9920512it [00:00, 34361979.98it/s]                             
0it [00:00, ?it/s]32768it [00:00, 528393.91it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:13, 125292.36it/s]1654784it [00:00, 8663391.14it/s]                          
0it [00:00, ?it/s]8192it [00:00, 153772.00it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fde59081748>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fddd6512e10>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fddf2e38510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fddf2e38510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fddf2e38510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:24,  1.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:24,  1.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:24,  1.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:23,  1.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:23,  1.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:22,  1.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:22,  1.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:57,  2.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:57,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:57,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:57,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:56,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:56,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:56,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:55,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:55,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:39,  3.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:39,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:38,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:38,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:38,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:37,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:37,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:37,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:26,  5.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:26,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:26,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:26,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:25,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:25,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:25,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:25,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:25,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:24,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:24,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:17,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:17,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:17,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:17,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:16,  7.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:16,  7.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:16,  7.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:16,  7.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:16,  7.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:11, 10.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:11, 10.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11, 10.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11, 10.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:11, 10.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:11, 10.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 13.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 13.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 13.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 13.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 13.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 13.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 13.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 13.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 13.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 18.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 18.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 18.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 18.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 18.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 18.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 18.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 18.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 18.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 18.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 18.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 24.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 31.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 31.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 31.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 31.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 31.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 31.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 31.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 31.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 31.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 31.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 31.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 39.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 39.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 39.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 39.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 39.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 39.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 39.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 39.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 39.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 39.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 39.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 47.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 55.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 63.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 63.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 63.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 63.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 63.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 63.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 63.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 63.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 63.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 63.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 63.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 70.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 70.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 70.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 70.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 76.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 81.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 81.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 81.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 81.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 81.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 81.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 81.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 81.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 81.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 81.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.32s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.32s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.32s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.12 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.33s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.32s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 81.12 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.33s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.33s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.41 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.33s/ url]
0 examples [00:00, ? examples/s]2020-05-31 18:09:37.633173: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-31 18:09:37.644983: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-31 18:09:37.645155: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55577aa43790 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-31 18:09:37.645172: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
31 examples [00:00, 306.52 examples/s]137 examples [00:00, 389.40 examples/s]246 examples [00:00, 482.25 examples/s]348 examples [00:00, 572.81 examples/s]453 examples [00:00, 662.69 examples/s]556 examples [00:00, 740.44 examples/s]660 examples [00:00, 810.03 examples/s]766 examples [00:00, 869.72 examples/s]873 examples [00:00, 921.18 examples/s]975 examples [00:01, 948.49 examples/s]1081 examples [00:01, 976.99 examples/s]1183 examples [00:01, 972.99 examples/s]1284 examples [00:01, 978.45 examples/s]1392 examples [00:01, 1003.93 examples/s]1498 examples [00:01, 1019.53 examples/s]1602 examples [00:01, 1011.84 examples/s]1718 examples [00:01, 1052.00 examples/s]1825 examples [00:01, 1037.13 examples/s]1939 examples [00:01, 1063.71 examples/s]2048 examples [00:02, 1069.88 examples/s]2156 examples [00:02, 1066.18 examples/s]2263 examples [00:02, 1025.60 examples/s]2367 examples [00:02, 1009.48 examples/s]2471 examples [00:02, 1017.18 examples/s]2575 examples [00:02, 1022.55 examples/s]2678 examples [00:02, 1010.02 examples/s]2785 examples [00:02, 1024.17 examples/s]2891 examples [00:02, 1034.23 examples/s]3000 examples [00:02, 1049.67 examples/s]3115 examples [00:03, 1075.32 examples/s]3232 examples [00:03, 1100.85 examples/s]3343 examples [00:03, 1059.07 examples/s]3450 examples [00:03, 1043.16 examples/s]3568 examples [00:03, 1079.39 examples/s]3687 examples [00:03, 1108.30 examples/s]3804 examples [00:03, 1125.80 examples/s]3923 examples [00:03, 1143.93 examples/s]4041 examples [00:03, 1152.61 examples/s]4158 examples [00:03, 1157.56 examples/s]4274 examples [00:04, 1157.50 examples/s]4394 examples [00:04, 1168.97 examples/s]4516 examples [00:04, 1183.01 examples/s]4635 examples [00:04, 1165.27 examples/s]4754 examples [00:04, 1171.15 examples/s]4872 examples [00:04, 1164.49 examples/s]4993 examples [00:04, 1175.58 examples/s]5114 examples [00:04, 1185.13 examples/s]5233 examples [00:04, 1158.60 examples/s]5350 examples [00:05, 1057.35 examples/s]5458 examples [00:05, 1051.06 examples/s]5567 examples [00:05, 1060.16 examples/s]5674 examples [00:05, 1057.00 examples/s]5781 examples [00:05, 1054.45 examples/s]5887 examples [00:05, 1053.21 examples/s]5993 examples [00:05, 1043.10 examples/s]6098 examples [00:05, 1037.89 examples/s]6202 examples [00:05, 1021.55 examples/s]6305 examples [00:05, 990.26 examples/s] 6412 examples [00:06, 1011.18 examples/s]6520 examples [00:06, 1029.73 examples/s]6626 examples [00:06, 1037.65 examples/s]6731 examples [00:06, 1033.40 examples/s]6835 examples [00:06, 1008.23 examples/s]6937 examples [00:06, 999.02 examples/s] 7038 examples [00:06, 1001.03 examples/s]7146 examples [00:06, 1023.22 examples/s]7257 examples [00:06, 1046.58 examples/s]7369 examples [00:06, 1066.84 examples/s]7476 examples [00:07, 1064.96 examples/s]7583 examples [00:07, 1041.88 examples/s]7698 examples [00:07, 1070.87 examples/s]7808 examples [00:07, 1078.45 examples/s]7919 examples [00:07, 1085.40 examples/s]8030 examples [00:07, 1092.56 examples/s]8144 examples [00:07, 1103.60 examples/s]8257 examples [00:07, 1109.74 examples/s]8369 examples [00:07, 1099.08 examples/s]8480 examples [00:08, 1082.29 examples/s]8589 examples [00:08, 1049.40 examples/s]8695 examples [00:08, 1042.02 examples/s]8809 examples [00:08, 1068.14 examples/s]8917 examples [00:08, 1067.81 examples/s]9032 examples [00:08, 1089.30 examples/s]9146 examples [00:08, 1102.57 examples/s]9260 examples [00:08, 1111.43 examples/s]9372 examples [00:08, 1112.36 examples/s]9484 examples [00:08, 1108.31 examples/s]9598 examples [00:09, 1117.11 examples/s]9710 examples [00:09, 1117.08 examples/s]9826 examples [00:09, 1128.58 examples/s]9943 examples [00:09, 1140.38 examples/s]10058 examples [00:09, 1074.24 examples/s]10174 examples [00:09, 1096.55 examples/s]10285 examples [00:09, 1097.42 examples/s]10396 examples [00:09, 1050.14 examples/s]10504 examples [00:09, 1058.45 examples/s]10616 examples [00:09, 1075.86 examples/s]10731 examples [00:10, 1096.28 examples/s]10846 examples [00:10, 1110.65 examples/s]10960 examples [00:10, 1117.09 examples/s]11073 examples [00:10, 1119.44 examples/s]11190 examples [00:10, 1134.00 examples/s]11310 examples [00:10, 1152.94 examples/s]11428 examples [00:10, 1159.42 examples/s]11545 examples [00:10, 1146.03 examples/s]11663 examples [00:10, 1154.30 examples/s]11779 examples [00:10, 1149.13 examples/s]11894 examples [00:11, 1144.16 examples/s]12011 examples [00:11, 1150.96 examples/s]12127 examples [00:11, 1147.18 examples/s]12242 examples [00:11, 1094.33 examples/s]12352 examples [00:11, 1086.89 examples/s]12468 examples [00:11, 1106.97 examples/s]12584 examples [00:11, 1121.65 examples/s]12701 examples [00:11, 1134.70 examples/s]12817 examples [00:11, 1140.96 examples/s]12932 examples [00:12, 1136.61 examples/s]13046 examples [00:12, 1136.26 examples/s]13162 examples [00:12, 1141.29 examples/s]13277 examples [00:12, 1093.44 examples/s]13387 examples [00:12, 1070.81 examples/s]13502 examples [00:12, 1092.35 examples/s]13619 examples [00:12, 1112.07 examples/s]13736 examples [00:12, 1128.48 examples/s]13850 examples [00:12, 1129.08 examples/s]13964 examples [00:12, 1130.99 examples/s]14078 examples [00:13, 1122.53 examples/s]14191 examples [00:13, 1105.12 examples/s]14302 examples [00:13, 1102.98 examples/s]14419 examples [00:13, 1120.82 examples/s]14535 examples [00:13, 1132.29 examples/s]14649 examples [00:13, 1130.32 examples/s]14768 examples [00:13, 1145.88 examples/s]14884 examples [00:13, 1149.48 examples/s]15001 examples [00:13, 1153.86 examples/s]15117 examples [00:13, 1153.54 examples/s]15233 examples [00:14, 1144.83 examples/s]15348 examples [00:14, 1136.04 examples/s]15464 examples [00:14, 1142.97 examples/s]15579 examples [00:14, 1141.81 examples/s]15694 examples [00:14, 1142.29 examples/s]15809 examples [00:14, 1084.56 examples/s]15919 examples [00:14, 1078.52 examples/s]16028 examples [00:14, 1079.28 examples/s]16137 examples [00:14, 1078.91 examples/s]16249 examples [00:14, 1089.32 examples/s]16360 examples [00:15, 1085.77 examples/s]16469 examples [00:15, 1073.06 examples/s]16578 examples [00:15, 1076.82 examples/s]16696 examples [00:15, 1104.94 examples/s]16812 examples [00:15, 1119.20 examples/s]16925 examples [00:15, 1121.10 examples/s]17040 examples [00:15, 1129.29 examples/s]17158 examples [00:15, 1141.46 examples/s]17276 examples [00:15, 1150.59 examples/s]17394 examples [00:15, 1157.60 examples/s]17510 examples [00:16, 1149.30 examples/s]17626 examples [00:16, 1151.21 examples/s]17744 examples [00:16, 1157.90 examples/s]17860 examples [00:16, 1139.57 examples/s]17975 examples [00:16, 1138.70 examples/s]18091 examples [00:16, 1142.96 examples/s]18206 examples [00:16, 1142.27 examples/s]18321 examples [00:16, 1134.29 examples/s]18435 examples [00:16, 1123.53 examples/s]18554 examples [00:17, 1139.98 examples/s]18669 examples [00:17, 1115.87 examples/s]18789 examples [00:17, 1137.73 examples/s]18904 examples [00:17, 1108.27 examples/s]19018 examples [00:17, 1115.15 examples/s]19131 examples [00:17, 1119.34 examples/s]19246 examples [00:17, 1127.40 examples/s]19359 examples [00:17, 1099.03 examples/s]19477 examples [00:17, 1119.63 examples/s]19596 examples [00:17, 1138.55 examples/s]19711 examples [00:18, 1120.63 examples/s]19824 examples [00:18, 1122.21 examples/s]19938 examples [00:18, 1127.06 examples/s]20051 examples [00:18, 1075.53 examples/s]20163 examples [00:18, 1088.00 examples/s]20278 examples [00:18, 1104.05 examples/s]20394 examples [00:18, 1118.61 examples/s]20511 examples [00:18, 1133.40 examples/s]20628 examples [00:18, 1144.01 examples/s]20743 examples [00:18, 1142.93 examples/s]20858 examples [00:19, 1144.99 examples/s]20973 examples [00:19, 1136.42 examples/s]21088 examples [00:19, 1138.50 examples/s]21203 examples [00:19, 1139.04 examples/s]21317 examples [00:19, 1138.73 examples/s]21431 examples [00:19, 1139.08 examples/s]21545 examples [00:19, 1129.26 examples/s]21662 examples [00:19, 1140.39 examples/s]21780 examples [00:19, 1149.66 examples/s]21903 examples [00:19, 1172.51 examples/s]22021 examples [00:20, 1167.73 examples/s]22138 examples [00:20, 1138.05 examples/s]22253 examples [00:20, 1091.60 examples/s]22368 examples [00:20, 1107.11 examples/s]22482 examples [00:20, 1116.75 examples/s]22595 examples [00:20, 1093.42 examples/s]22707 examples [00:20, 1100.19 examples/s]22820 examples [00:20, 1107.50 examples/s]22935 examples [00:20, 1119.21 examples/s]23048 examples [00:21, 1096.06 examples/s]23158 examples [00:21, 1095.71 examples/s]23268 examples [00:21, 1084.01 examples/s]23380 examples [00:21, 1093.54 examples/s]23490 examples [00:21, 1089.91 examples/s]23604 examples [00:21, 1103.78 examples/s]23718 examples [00:21, 1112.50 examples/s]23830 examples [00:21, 1113.04 examples/s]23942 examples [00:21, 1104.51 examples/s]24057 examples [00:21, 1115.81 examples/s]24172 examples [00:22, 1125.52 examples/s]24285 examples [00:22, 1125.34 examples/s]24398 examples [00:22, 1119.84 examples/s]24511 examples [00:22, 1023.21 examples/s]24615 examples [00:22, 1016.83 examples/s]24718 examples [00:22, 1015.69 examples/s]24832 examples [00:22, 1049.66 examples/s]24950 examples [00:22, 1085.37 examples/s]25075 examples [00:22, 1128.97 examples/s]25199 examples [00:22, 1160.00 examples/s]25316 examples [00:23, 1158.92 examples/s]25437 examples [00:23, 1171.58 examples/s]25556 examples [00:23, 1175.15 examples/s]25674 examples [00:23, 1169.44 examples/s]25792 examples [00:23, 1125.30 examples/s]25906 examples [00:23, 1123.52 examples/s]26019 examples [00:23, 1115.37 examples/s]26131 examples [00:23, 1084.01 examples/s]26244 examples [00:23, 1097.41 examples/s]26361 examples [00:24, 1116.07 examples/s]26475 examples [00:24, 1120.64 examples/s]26589 examples [00:24, 1124.08 examples/s]26706 examples [00:24, 1137.27 examples/s]26820 examples [00:24, 1128.00 examples/s]26937 examples [00:24, 1137.99 examples/s]27051 examples [00:24, 1102.36 examples/s]27165 examples [00:24, 1113.20 examples/s]27282 examples [00:24, 1128.39 examples/s]27403 examples [00:24, 1151.40 examples/s]27523 examples [00:25, 1164.14 examples/s]27640 examples [00:25, 1143.48 examples/s]27755 examples [00:25, 1142.26 examples/s]27877 examples [00:25, 1164.51 examples/s]28001 examples [00:25, 1184.52 examples/s]28120 examples [00:25, 1181.77 examples/s]28244 examples [00:25, 1196.86 examples/s]28364 examples [00:25, 1194.29 examples/s]28487 examples [00:25, 1204.44 examples/s]28609 examples [00:25, 1206.49 examples/s]28730 examples [00:26, 1196.18 examples/s]28853 examples [00:26, 1205.94 examples/s]28974 examples [00:26, 1201.10 examples/s]29095 examples [00:26, 1200.25 examples/s]29216 examples [00:26, 1178.25 examples/s]29337 examples [00:26, 1185.28 examples/s]29461 examples [00:26, 1199.17 examples/s]29582 examples [00:26, 1153.12 examples/s]29698 examples [00:26, 1089.16 examples/s]29821 examples [00:26, 1126.05 examples/s]29936 examples [00:27, 1132.27 examples/s]30050 examples [00:27, 1103.68 examples/s]30170 examples [00:27, 1130.18 examples/s]30293 examples [00:27, 1157.61 examples/s]30410 examples [00:27, 1160.70 examples/s]30527 examples [00:27, 1138.77 examples/s]30647 examples [00:27, 1155.77 examples/s]30763 examples [00:27, 1135.75 examples/s]30879 examples [00:27, 1140.92 examples/s]30994 examples [00:28, 1128.85 examples/s]31111 examples [00:28, 1139.42 examples/s]31235 examples [00:28, 1166.60 examples/s]31357 examples [00:28, 1180.15 examples/s]31476 examples [00:28, 1175.59 examples/s]31594 examples [00:28, 1165.55 examples/s]31715 examples [00:28, 1177.61 examples/s]31836 examples [00:28, 1184.53 examples/s]31955 examples [00:28, 1174.04 examples/s]32075 examples [00:28, 1180.62 examples/s]32194 examples [00:29, 1141.18 examples/s]32309 examples [00:29, 1104.29 examples/s]32420 examples [00:29, 1103.90 examples/s]32531 examples [00:29, 1104.75 examples/s]32648 examples [00:29, 1123.22 examples/s]32771 examples [00:29, 1151.13 examples/s]32888 examples [00:29, 1155.92 examples/s]33004 examples [00:29, 1129.49 examples/s]33120 examples [00:29, 1137.75 examples/s]33240 examples [00:29, 1155.60 examples/s]33362 examples [00:30, 1174.12 examples/s]33480 examples [00:30, 1118.97 examples/s]33594 examples [00:30, 1122.42 examples/s]33707 examples [00:30, 1073.02 examples/s]33822 examples [00:30, 1092.69 examples/s]33932 examples [00:30, 1083.52 examples/s]34049 examples [00:30, 1105.67 examples/s]34163 examples [00:30, 1114.80 examples/s]34275 examples [00:30, 1097.19 examples/s]34393 examples [00:30, 1120.54 examples/s]34506 examples [00:31, 1106.86 examples/s]34620 examples [00:31, 1115.23 examples/s]34732 examples [00:31, 1109.55 examples/s]34844 examples [00:31, 1112.31 examples/s]34965 examples [00:31, 1132.91 examples/s]35085 examples [00:31, 1151.82 examples/s]35208 examples [00:31, 1170.79 examples/s]35326 examples [00:31, 1164.16 examples/s]35443 examples [00:31, 1159.87 examples/s]35560 examples [00:32, 1157.97 examples/s]35677 examples [00:32, 1159.80 examples/s]35799 examples [00:32, 1175.95 examples/s]35917 examples [00:32, 1144.11 examples/s]36032 examples [00:32, 1128.85 examples/s]36151 examples [00:32, 1144.28 examples/s]36266 examples [00:32, 1141.57 examples/s]36386 examples [00:32, 1156.62 examples/s]36504 examples [00:32, 1160.89 examples/s]36623 examples [00:32, 1167.12 examples/s]36744 examples [00:33, 1178.47 examples/s]36862 examples [00:33, 1157.05 examples/s]36979 examples [00:33, 1158.24 examples/s]37095 examples [00:33, 1151.96 examples/s]37211 examples [00:33, 1141.69 examples/s]37327 examples [00:33, 1145.72 examples/s]37442 examples [00:33, 1130.11 examples/s]37556 examples [00:33, 1132.08 examples/s]37670 examples [00:33, 1130.15 examples/s]37787 examples [00:33, 1140.76 examples/s]37902 examples [00:34, 1134.73 examples/s]38017 examples [00:34, 1137.41 examples/s]38134 examples [00:34, 1146.56 examples/s]38249 examples [00:34, 1132.13 examples/s]38363 examples [00:34, 1087.68 examples/s]38476 examples [00:34, 1097.81 examples/s]38587 examples [00:34, 1085.59 examples/s]38698 examples [00:34, 1090.34 examples/s]38813 examples [00:34, 1107.55 examples/s]38929 examples [00:34, 1121.87 examples/s]39046 examples [00:35, 1133.16 examples/s]39160 examples [00:35, 1122.07 examples/s]39276 examples [00:35, 1130.60 examples/s]39390 examples [00:35, 1117.21 examples/s]39507 examples [00:35, 1130.86 examples/s]39625 examples [00:35, 1143.19 examples/s]39743 examples [00:35, 1152.65 examples/s]39861 examples [00:35, 1160.46 examples/s]39978 examples [00:35, 1153.06 examples/s]40094 examples [00:36, 1101.19 examples/s]40208 examples [00:36, 1112.34 examples/s]40328 examples [00:36, 1136.59 examples/s]40450 examples [00:36, 1159.94 examples/s]40567 examples [00:36, 1146.52 examples/s]40682 examples [00:36, 1139.56 examples/s]40803 examples [00:36, 1157.29 examples/s]40926 examples [00:36, 1177.29 examples/s]41044 examples [00:36, 1176.66 examples/s]41162 examples [00:36, 1162.93 examples/s]41280 examples [00:37, 1166.20 examples/s]41404 examples [00:37, 1186.97 examples/s]41524 examples [00:37, 1189.43 examples/s]41646 examples [00:37, 1197.99 examples/s]41766 examples [00:37, 1198.32 examples/s]41888 examples [00:37, 1204.70 examples/s]42009 examples [00:37, 1198.35 examples/s]42131 examples [00:37, 1202.74 examples/s]42252 examples [00:37, 1167.91 examples/s]42372 examples [00:37, 1174.12 examples/s]42494 examples [00:38, 1184.86 examples/s]42617 examples [00:38, 1196.26 examples/s]42738 examples [00:38, 1197.82 examples/s]42858 examples [00:38, 1196.22 examples/s]42978 examples [00:38, 1177.70 examples/s]43096 examples [00:38, 1172.16 examples/s]43214 examples [00:38, 1167.27 examples/s]43331 examples [00:38, 1157.50 examples/s]43447 examples [00:38, 1151.17 examples/s]43563 examples [00:38, 1145.84 examples/s]43679 examples [00:39, 1150.05 examples/s]43795 examples [00:39, 1129.50 examples/s]43910 examples [00:39, 1135.14 examples/s]44028 examples [00:39, 1147.18 examples/s]44143 examples [00:39, 1136.52 examples/s]44260 examples [00:39, 1145.16 examples/s]44377 examples [00:39, 1151.12 examples/s]44494 examples [00:39, 1156.70 examples/s]44611 examples [00:39, 1157.79 examples/s]44727 examples [00:39, 1146.61 examples/s]44844 examples [00:40, 1152.56 examples/s]44961 examples [00:40, 1155.15 examples/s]45077 examples [00:40, 1149.32 examples/s]45192 examples [00:40, 1097.09 examples/s]45303 examples [00:40, 1099.44 examples/s]45415 examples [00:40, 1103.27 examples/s]45533 examples [00:40, 1123.98 examples/s]45652 examples [00:40, 1141.24 examples/s]45774 examples [00:40, 1162.66 examples/s]45891 examples [00:41, 1148.71 examples/s]46007 examples [00:41, 1121.16 examples/s]46120 examples [00:41, 1116.40 examples/s]46238 examples [00:41, 1131.50 examples/s]46352 examples [00:41, 1128.57 examples/s]46466 examples [00:41, 1123.19 examples/s]46583 examples [00:41, 1135.49 examples/s]46700 examples [00:41, 1143.26 examples/s]46816 examples [00:41, 1147.30 examples/s]46931 examples [00:41, 1143.57 examples/s]47046 examples [00:42, 1140.53 examples/s]47162 examples [00:42, 1144.36 examples/s]47279 examples [00:42, 1149.58 examples/s]47394 examples [00:42, 1122.02 examples/s]47507 examples [00:42, 1082.94 examples/s]47616 examples [00:42, 1035.03 examples/s]47732 examples [00:42, 1067.18 examples/s]47842 examples [00:42, 1076.53 examples/s]47957 examples [00:42, 1094.66 examples/s]48067 examples [00:42, 1090.90 examples/s]48180 examples [00:43, 1101.82 examples/s]48297 examples [00:43, 1121.29 examples/s]48410 examples [00:43, 1120.78 examples/s]48525 examples [00:43, 1128.81 examples/s]48639 examples [00:43, 1121.05 examples/s]48752 examples [00:43, 1110.57 examples/s]48864 examples [00:43, 1112.31 examples/s]48976 examples [00:43, 1095.40 examples/s]49086 examples [00:43, 1088.05 examples/s]49196 examples [00:43, 1088.89 examples/s]49307 examples [00:44, 1094.49 examples/s]49419 examples [00:44, 1099.72 examples/s]49530 examples [00:44, 1073.06 examples/s]49638 examples [00:44, 1073.08 examples/s]49746 examples [00:44, 1035.09 examples/s]49857 examples [00:44, 1054.76 examples/s]49977 examples [00:44, 1092.17 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 8805/50000 [00:00<00:00, 88045.47 examples/s] 46%|████▌     | 23073/50000 [00:00<00:00, 99472.12 examples/s] 74%|███████▎  | 36848/50000 [00:00<00:00, 108517.75 examples/s]                                                                0 examples [00:00, ? examples/s]96 examples [00:00, 954.55 examples/s]213 examples [00:00, 1008.33 examples/s]326 examples [00:00, 1040.44 examples/s]432 examples [00:00, 1044.74 examples/s]547 examples [00:00, 1072.81 examples/s]666 examples [00:00, 1104.18 examples/s]766 examples [00:00, 878.55 examples/s] 879 examples [00:00, 939.47 examples/s]993 examples [00:00, 991.44 examples/s]1104 examples [00:01, 1022.83 examples/s]1222 examples [00:01, 1063.16 examples/s]1335 examples [00:01, 1080.30 examples/s]1445 examples [00:01, 1076.24 examples/s]1560 examples [00:01, 1097.32 examples/s]1678 examples [00:01, 1119.20 examples/s]1796 examples [00:01, 1135.72 examples/s]1910 examples [00:01, 1136.35 examples/s]2026 examples [00:01, 1141.63 examples/s]2141 examples [00:01, 1130.71 examples/s]2257 examples [00:02, 1138.06 examples/s]2373 examples [00:02, 1144.19 examples/s]2488 examples [00:02, 1140.51 examples/s]2603 examples [00:02, 1109.04 examples/s]2715 examples [00:02, 1102.36 examples/s]2832 examples [00:02, 1121.63 examples/s]2950 examples [00:02, 1136.25 examples/s]3067 examples [00:02, 1143.89 examples/s]3185 examples [00:02, 1153.20 examples/s]3304 examples [00:03, 1163.31 examples/s]3422 examples [00:03, 1165.49 examples/s]3541 examples [00:03, 1171.61 examples/s]3660 examples [00:03, 1176.92 examples/s]3778 examples [00:03, 1165.76 examples/s]3895 examples [00:03, 1159.31 examples/s]4012 examples [00:03, 1160.56 examples/s]4129 examples [00:03, 1156.15 examples/s]4247 examples [00:03, 1160.85 examples/s]4364 examples [00:03, 1156.00 examples/s]4480 examples [00:04, 1146.87 examples/s]4598 examples [00:04, 1154.55 examples/s]4714 examples [00:04, 1147.12 examples/s]4829 examples [00:04, 1133.23 examples/s]4943 examples [00:04, 1121.64 examples/s]5056 examples [00:04, 1123.28 examples/s]5172 examples [00:04, 1131.79 examples/s]5286 examples [00:04, 1130.40 examples/s]5400 examples [00:04, 1132.16 examples/s]5514 examples [00:04, 1133.31 examples/s]5628 examples [00:05, 1096.33 examples/s]5744 examples [00:05, 1113.09 examples/s]5860 examples [00:05, 1126.57 examples/s]5973 examples [00:05, 1122.85 examples/s]6086 examples [00:05, 1113.99 examples/s]6198 examples [00:05, 1098.15 examples/s]6312 examples [00:05, 1110.24 examples/s]6432 examples [00:05, 1133.27 examples/s]6554 examples [00:05, 1157.86 examples/s]6671 examples [00:05, 1160.80 examples/s]6793 examples [00:06, 1176.42 examples/s]6911 examples [00:06, 1134.73 examples/s]7032 examples [00:06, 1154.27 examples/s]7148 examples [00:06, 1133.39 examples/s]7262 examples [00:06, 1127.89 examples/s]7376 examples [00:06, 1128.21 examples/s]7491 examples [00:06, 1132.71 examples/s]7607 examples [00:06, 1139.69 examples/s]7729 examples [00:06, 1162.35 examples/s]7846 examples [00:06, 1159.92 examples/s]7963 examples [00:07, 1154.93 examples/s]8082 examples [00:07, 1163.22 examples/s]8201 examples [00:07, 1168.69 examples/s]8318 examples [00:07, 1140.79 examples/s]8435 examples [00:07, 1148.48 examples/s]8552 examples [00:07, 1154.46 examples/s]8674 examples [00:07, 1171.52 examples/s]8794 examples [00:07, 1177.41 examples/s]8917 examples [00:07, 1191.81 examples/s]9037 examples [00:07, 1185.20 examples/s]9156 examples [00:08, 1181.45 examples/s]9278 examples [00:08, 1190.42 examples/s]9398 examples [00:08, 1188.97 examples/s]9517 examples [00:08, 1148.09 examples/s]9633 examples [00:08, 1139.76 examples/s]9754 examples [00:08, 1158.31 examples/s]9871 examples [00:08, 1154.11 examples/s]9991 examples [00:08, 1166.32 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete7UYF1E/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete7UYF1E/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fddf2e388c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fddf2e388c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fddf2e388c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fde5ecd54e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fddd44905f8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fddf2e388c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fddf2e388c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fddf2e388c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fddd40f2400>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fddd40f22b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fddd0d5b620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fddd0d5b620> 

  function with postional parmater data_info <function split_train_valid at 0x7fddd0d5b620> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fddd0d5b730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fddd0d5b730> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fddd0d5b730> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=acbbcecd5056038aff572042af2050e0aff28d5abb8eb9a07c5f8e233b0ae71a
  Stored in directory: /tmp/pip-ephem-wheel-cache-owm5b2cm/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:00:39, 10.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:20:19, 14.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:29:25, 20.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:03:03, 29.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.63M/862M [00:01<5:37:15, 42.4kB/s].vector_cache/glove.6B.zip:   1%|          | 8.27M/862M [00:01<3:54:53, 60.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:01<2:43:45, 86.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:01<1:53:57, 123kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:01<1:19:38, 176kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.5M/862M [00:01<55:31, 251kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 28.5M/862M [00:01<38:52, 357kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.3M/862M [00:02<27:08, 509kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.5M/862M [00:02<18:59, 723kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.6M/862M [00:02<13:20, 1.03MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.8M/862M [00:02<09:22, 1.45MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.8M/862M [00:02<06:38, 2.04MB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.6M/862M [00:02<04:44, 2.85MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.2M/862M [00:04<07:41, 1.75MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.5M/862M [00:04<07:12, 1.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<05:26, 2.47MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.1M/862M [00:04<03:57, 3.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.4M/862M [00:06<26:14, 510kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<20:00, 669kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<14:24, 927kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.5M/862M [00:08<12:43, 1.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<11:44, 1.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<08:49, 1.51MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.0M/862M [00:08<06:19, 2.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.6M/862M [00:10<13:57, 949kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<11:06, 1.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:10<08:06, 1.63MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.7M/862M [00:12<08:43, 1.51MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:12<07:27, 1.77MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:12<05:32, 2.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.9M/862M [00:14<06:56, 1.89MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<06:12, 2.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:14<04:40, 2.80MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.0M/862M [00:16<06:19, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:16<07:04, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<05:31, 2.35MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:16<04:00, 3.23MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.1M/862M [00:18<12:30, 1.04MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<10:04, 1.29MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<07:19, 1.76MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.2M/862M [00:20<08:09, 1.58MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:20<08:20, 1.55MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<06:28, 1.99MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.3M/862M [00:22<06:35, 1.94MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:22<05:57, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<04:29, 2.85MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.4M/862M [00:23<06:07, 2.09MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:24<06:53, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<05:22, 2.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:24<03:55, 3.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<09:23, 1.35MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:51, 1.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:48, 2.18MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<07:01, 1.80MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:12, 2.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:39, 2.71MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<06:13, 2.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:36, 2.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:14, 2.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<05:54, 2.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<06:41, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:18, 2.35MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<03:49, 3.25MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<6:05:10, 34.0kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<4:16:46, 48.3kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<2:59:42, 68.9kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<2:08:17, 96.3kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<1:32:17, 134kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:05:04, 190kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<45:39, 270kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<36:04, 340kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<26:31, 463kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<18:50, 650kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:39<15:59, 764kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<12:28, 979kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<09:20, 1.31MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<06:40, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<12:06, 1.00MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<11:05, 1.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<08:18, 1.46MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<06:07, 1.98MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<07:03, 1.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:11, 1.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<04:38, 2.59MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<06:02, 1.99MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<06:39, 1.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:11, 2.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<03:49, 3.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<06:58, 1.71MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<06:06, 1.95MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<04:34, 2.60MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<05:58, 1.99MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<05:24, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<04:04, 2.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<05:36, 2.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<06:20, 1.86MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:03, 2.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<03:40, 3.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<1:38:20, 119kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<1:10:01, 167kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<49:10, 238kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<37:02, 315kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<27:05, 430kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<19:13, 605kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<16:07, 719kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<13:44, 844kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<10:12, 1.13MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:57<07:15, 1.59MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<1:19:00, 146kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<56:28, 204kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<39:44, 289kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<30:23, 377kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<22:16, 514kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<15:47, 724kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<11:10, 1.02MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<46:04, 247kB/s] .vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<34:39, 329kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<24:44, 460kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<17:22, 652kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<22:38, 500kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<17:01, 665kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<12:10, 927kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<11:06, 1.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<08:56, 1.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:31, 1.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<07:11, 1.56MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<07:18, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:41, 1.96MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<04:05, 2.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<1:07:16, 165kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<48:13, 230kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<33:58, 326kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<26:16, 421kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<20:37, 535kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<14:56, 739kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<10:32, 1.04MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<16:30, 665kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<12:29, 878kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<09:00, 1.21MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<08:51, 1.23MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<08:25, 1.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<06:23, 1.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<04:34, 2.37MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<15:02, 721kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<11:39, 930kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<08:25, 1.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<08:23, 1.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<06:58, 1.55MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:08, 2.09MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<06:05, 1.76MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<05:22, 1.99MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:01, 2.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:18, 2.00MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:58, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:40, 2.27MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:26, 3.08MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<06:15, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:29, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:06, 2.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<05:18, 1.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<05:56, 1.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:42, 2.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:24, 3.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<1:16:55, 136kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<54:54, 190kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<38:35, 269kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<29:21, 353kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<21:38, 479kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<15:22, 672kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<13:06, 785kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<11:16, 913kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<08:20, 1.23MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:34<05:56, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<11:22, 899kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<08:50, 1.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<06:26, 1.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<06:53, 1.47MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<06:53, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<05:20, 1.90MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:38<03:51, 2.62MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<1:25:09, 118kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<1:00:36, 166kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<42:34, 236kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<32:02, 313kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<24:34, 407kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<17:42, 565kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<12:27, 799kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<1:19:40, 125kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<56:36, 176kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<39:50, 249kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<30:02, 329kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<22:02, 448kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<15:38, 630kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<13:12, 743kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<10:15, 956kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<07:24, 1.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<07:27, 1.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<06:13, 1.56MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:33, 2.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:27, 1.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:46, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:32, 2.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<03:16, 2.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<36:59, 260kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<26:42, 359kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<18:55, 506kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<13:19, 716kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<1:01:01, 156kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<44:38, 214kB/s]  .vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<31:38, 301kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<22:12, 427kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<20:06, 471kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<15:03, 629kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<10:45, 878kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<09:41, 971kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<08:41, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:33, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<04:41, 1.99MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<1:19:33, 117kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<56:37, 165kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<39:45, 234kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<29:53, 310kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<22:50, 405kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<16:25, 563kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<11:33, 797kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<13:49, 666kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<10:38, 864kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<07:39, 1.20MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<07:28, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:08<07:10, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:26, 1.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<03:54, 2.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<07:58, 1.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<06:31, 1.39MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:47, 1.89MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:25, 1.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:43, 1.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:31, 2.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:32, 1.97MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:06, 2.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:05, 2.87MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:14, 2.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:52, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<02:53, 3.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:05, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:39, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<03:38, 2.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<02:40, 3.26MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:07, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:28, 1.95MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:20, 2.60MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:21, 1.98MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:48, 1.80MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:44, 2.31MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:43, 3.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:10, 1.39MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:12, 1.65MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<03:51, 2.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:40, 1.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:04, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:59, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<02:52, 2.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<30:40, 275kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<22:20, 378kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<15:48, 532kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<12:57, 646kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<10:51, 771kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<07:56, 1.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<05:42, 1.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<06:28, 1.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:23, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:58, 2.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:42, 1.75MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:07, 1.99MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:05, 2.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:04, 2.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:30, 1.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:34, 2.28MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<02:35, 3.12MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<59:41, 136kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<42:34, 190kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<29:53, 270kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<22:43, 354kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<17:32, 458kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<12:40, 632kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<08:54, 894kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<36:55, 216kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<26:39, 299kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<18:48, 422kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<14:56, 528kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<11:15, 701kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<08:03, 977kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<07:26, 1.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<06:01, 1.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<04:23, 1.77MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:53, 1.59MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:03, 1.53MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:52, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<02:48, 2.75MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<06:03, 1.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<05:02, 1.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:42, 2.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:22, 1.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:36, 1.65MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:36, 2.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<02:36, 2.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<55:47, 135kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<39:47, 190kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<27:56, 269kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<21:13, 353kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<16:22, 457kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<11:49, 631kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<08:19, 891kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<1:06:16, 112kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<47:07, 157kB/s]  .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<33:03, 223kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<24:44, 297kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<18:48, 391kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<13:28, 544kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<09:28, 770kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<11:37, 626kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<08:51, 820kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<06:22, 1.14MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<06:07, 1.18MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<05:45, 1.25MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:20, 1.66MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:07, 2.29MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:16, 1.36MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:24, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:14, 2.20MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:54, 1.81MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:14, 1.67MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:17, 2.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<02:21, 2.97MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<16:23, 427kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<12:11, 574kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<08:41, 803kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<07:39, 905kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<06:49, 1.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<05:08, 1.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<03:39, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<51:34, 133kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<36:46, 187kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<25:49, 265kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<19:34, 347kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<15:08, 449kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<10:55, 621kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<07:40, 877kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<53:37, 126kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<38:12, 176kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<26:47, 250kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<20:12, 330kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<15:29, 430kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:18<11:10, 595kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<07:50, 842kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<11:35, 569kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<08:47, 749kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<06:18, 1.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<05:54, 1.11MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<05:27, 1.20MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<04:08, 1.57MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:22<02:57, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<48:13, 134kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<34:23, 188kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<24:08, 266kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<18:18, 349kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<14:06, 453kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<10:08, 628kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<07:09, 886kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<07:43, 818kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<06:02, 1.05MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<04:22, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<04:30, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:48, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:49, 2.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:23, 1.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:41, 1.67MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:53, 2.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<02:05, 2.92MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<45:09, 135kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<32:12, 190kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<22:36, 269kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<17:09, 352kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<12:37, 479kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<08:55, 673kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<07:35, 786kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<06:35, 905kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<04:54, 1.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<03:29, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<44:38, 132kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<31:43, 186kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<22:16, 264kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<15:34, 375kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<25:35, 228kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<18:29, 316kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<13:01, 446kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<10:25, 554kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<08:28, 681kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<06:11, 928kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<04:22, 1.30MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<28:17, 202kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<20:22, 279kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<14:20, 395kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<11:17, 499kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<09:02, 622kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<06:36, 849kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<04:39, 1.20MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<07:21, 756kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:43, 971kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<04:07, 1.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:09, 1.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:28, 1.58MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:33, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<03:02, 1.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:18, 1.64MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:35, 2.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<01:51, 2.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<21:50, 245kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<15:49, 338kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<11:09, 477kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<08:59, 588kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<06:50, 772kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<04:54, 1.07MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:37, 1.13MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:17, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:15, 1.60MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<02:19, 2.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<14:33, 354kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<10:42, 481kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<07:35, 674kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<06:27, 787kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<05:33, 915kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<04:05, 1.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:55, 1.72MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<04:21, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:34, 1.40MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<02:35, 1.92MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:57, 1.67MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:35, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:54, 2.56MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:28, 1.97MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:14, 2.17MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<01:40, 2.91MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<01:13, 3.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<09:48, 491kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<07:16, 660kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<05:11, 920kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:41, 1.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:17, 1.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:12, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<02:17, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:59, 1.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<03:16, 1.42MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<02:24, 1.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:44, 1.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:54, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:13, 2.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:36, 2.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:06, 1.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:38, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:57, 2.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:24, 1.85MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:08, 2.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:36, 2.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:08, 2.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<01:57, 2.25MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:28, 2.97MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:02, 2.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:52, 2.31MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:24, 3.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<01:58, 2.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:49, 2.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:22, 3.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:56, 2.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:16, 1.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:46, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:17, 3.22MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:57, 1.04MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:12, 1.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:20, 1.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:33, 1.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:40, 1.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:04, 1.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<01:28, 2.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<13:56, 286kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<10:10, 392kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<07:10, 551kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<05:53, 666kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<04:57, 789kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:39, 1.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<02:34, 1.49MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<29:16, 132kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<20:50, 184kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<14:35, 262kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<10:59, 344kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<08:28, 446kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<06:06, 616kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<04:16, 870kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<16:13, 229kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<11:43, 316kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<08:14, 447kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<06:32, 557kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<04:53, 743kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<03:34, 1.01MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<02:30, 1.43MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<08:42, 411kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<06:27, 553kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<04:33, 777kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<03:12, 1.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<16:10, 217kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<12:01, 291kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<08:34, 407kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<05:57, 578kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<28:44, 120kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<20:25, 168kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<14:16, 239kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<10:40, 316kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<08:09, 413kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<05:50, 575kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<04:06, 811kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<04:04, 809kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:11, 1.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<02:17, 1.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:20, 1.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:58, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:26, 2.22MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:44, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:32, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:08, 2.74MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:31, 2.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<01:42, 1.80MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:21, 2.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<00:58, 3.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<19:57, 152kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<14:15, 212kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<09:58, 301kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<07:34, 390kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<05:54, 500kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<04:16, 689kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:04<02:58, 972kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<22:41, 127kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<16:08, 179kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<11:16, 254kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<08:26, 334kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<06:11, 455kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<04:21, 639kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:39, 752kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:50, 966kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<02:01, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:02, 1.32MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:41, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:14, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:28, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:17, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<00:57, 2.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:15, 2.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:25, 1.79MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:06, 2.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<00:47, 3.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:39, 1.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:24, 1.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:02, 2.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:16, 1.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:38, 1.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:16, 1.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<00:55, 2.55MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:14, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:08, 2.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:51, 2.69MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:04, 2.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:13, 1.84MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:58, 2.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<00:41, 3.22MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<03:48, 579kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:53, 761kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<02:03, 1.06MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:54, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:33, 1.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:07, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:15, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:05, 1.89MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:48, 2.52MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:01, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:07, 1.78MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:52, 2.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:37, 3.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<02:32, 760kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:58, 978kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:24, 1.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:23, 1.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:09, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:50, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:36<00:36, 2.96MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<09:18, 193kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<06:53, 260kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<04:51, 365kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<03:22, 517kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<02:54, 594kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<02:12, 779kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:33, 1.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:27, 1.13MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<01:11, 1.39MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:51, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:57, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:48, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:35, 2.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:25, 3.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<06:25, 236kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<04:38, 326kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<03:13, 460kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<02:32, 570kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<02:04, 699kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<01:29, 957kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:03, 1.33MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:10, 1.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:57, 1.42MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:41, 1.95MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:46, 1.68MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:48, 1.61MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:37, 2.06MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:26, 2.83MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<10:27, 119kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<07:24, 167kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<05:06, 237kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<03:44, 314kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<02:43, 429kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<01:53, 603kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:32, 716kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:18, 839kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:57, 1.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:40, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:47, 1.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:38, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:27, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [05:59<00:19, 2.98MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<02:07, 457kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:34, 612kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<01:05, 857kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:56, 952kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:44, 1.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:31, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:33, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:33, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:25, 1.93MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:17, 2.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<06:26, 119kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<04:33, 167kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<03:05, 237kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<02:13, 313kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<01:36, 427kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<01:06, 601kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:52, 715kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:40, 923kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:28, 1.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:26, 1.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:21, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:15, 2.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:16, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:17, 1.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:09, 2.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:16, 1.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:13, 1.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:09, 2.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:11, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:11, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:08, 2.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:05, 3.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:14, 1.21MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:11, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 1.99MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:07, 1.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:04, 2.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 1.98MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:02, 2.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:01, 2.29MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:00, 3.01MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 2.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.86MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<17:23:32,  6.39it/s]  0%|          | 813/400000 [00:00<12:09:14,  9.12it/s]  0%|          | 1675/400000 [00:00<8:29:35, 13.03it/s]  1%|          | 2558/400000 [00:00<5:56:09, 18.60it/s]  1%|          | 3429/400000 [00:00<4:08:59, 26.55it/s]  1%|          | 4330/400000 [00:00<2:54:06, 37.87it/s]  1%|▏         | 5231/400000 [00:00<2:01:49, 54.01it/s]  2%|▏         | 6115/400000 [00:00<1:25:18, 76.95it/s]  2%|▏         | 7014/400000 [00:00<59:47, 109.53it/s]   2%|▏         | 7881/400000 [00:01<41:59, 155.63it/s]  2%|▏         | 8728/400000 [00:01<29:34, 220.53it/s]  2%|▏         | 9567/400000 [00:01<20:53, 311.42it/s]  3%|▎         | 10430/400000 [00:01<14:49, 438.11it/s]  3%|▎         | 11309/400000 [00:01<10:34, 612.78it/s]  3%|▎         | 12166/400000 [00:01<07:36, 849.35it/s]  3%|▎         | 13046/400000 [00:01<05:32, 1165.16it/s]  3%|▎         | 13961/400000 [00:01<04:04, 1578.36it/s]  4%|▎         | 14868/400000 [00:01<03:03, 2098.27it/s]  4%|▍         | 15765/400000 [00:01<02:21, 2724.20it/s]  4%|▍         | 16655/400000 [00:02<01:51, 3427.46it/s]  4%|▍         | 17536/400000 [00:02<01:32, 4117.96it/s]  5%|▍         | 18432/400000 [00:02<01:17, 4913.58it/s]  5%|▍         | 19333/400000 [00:02<01:06, 5688.84it/s]  5%|▌         | 20230/400000 [00:02<00:59, 6390.04it/s]  5%|▌         | 21112/400000 [00:02<00:54, 6955.36it/s]  6%|▌         | 22011/400000 [00:02<00:50, 7461.61it/s]  6%|▌         | 22905/400000 [00:02<00:48, 7850.37it/s]  6%|▌         | 23793/400000 [00:02<00:46, 8061.40it/s]  6%|▌         | 24673/400000 [00:02<00:45, 8234.66it/s]  6%|▋         | 25549/400000 [00:03<00:44, 8348.81it/s]  7%|▋         | 26421/400000 [00:03<00:44, 8317.72it/s]  7%|▋         | 27279/400000 [00:03<00:44, 8377.56it/s]  7%|▋         | 28136/400000 [00:03<00:44, 8406.04it/s]  7%|▋         | 28990/400000 [00:03<00:44, 8401.84it/s]  7%|▋         | 29878/400000 [00:03<00:43, 8537.31it/s]  8%|▊         | 30745/400000 [00:03<00:43, 8574.88it/s]  8%|▊         | 31621/400000 [00:03<00:42, 8627.05it/s]  8%|▊         | 32508/400000 [00:03<00:42, 8696.49it/s]  8%|▊         | 33408/400000 [00:03<00:41, 8782.68it/s]  9%|▊         | 34289/400000 [00:04<00:42, 8561.31it/s]  9%|▉         | 35148/400000 [00:04<00:42, 8497.90it/s]  9%|▉         | 36011/400000 [00:04<00:42, 8536.30it/s]  9%|▉         | 36876/400000 [00:04<00:42, 8568.85it/s]  9%|▉         | 37768/400000 [00:04<00:41, 8668.97it/s] 10%|▉         | 38677/400000 [00:04<00:41, 8790.33it/s] 10%|▉         | 39558/400000 [00:04<00:41, 8724.96it/s] 10%|█         | 40432/400000 [00:04<00:41, 8592.91it/s] 10%|█         | 41298/400000 [00:04<00:41, 8611.17it/s] 11%|█         | 42169/400000 [00:05<00:41, 8638.37it/s] 11%|█         | 43080/400000 [00:05<00:40, 8772.38it/s] 11%|█         | 43959/400000 [00:05<00:40, 8756.93it/s] 11%|█         | 44864/400000 [00:05<00:40, 8842.83it/s] 11%|█▏        | 45749/400000 [00:05<00:40, 8781.62it/s] 12%|█▏        | 46628/400000 [00:05<00:40, 8693.93it/s] 12%|█▏        | 47526/400000 [00:05<00:40, 8777.41it/s] 12%|█▏        | 48420/400000 [00:05<00:39, 8822.98it/s] 12%|█▏        | 49331/400000 [00:05<00:39, 8906.43it/s] 13%|█▎        | 50245/400000 [00:05<00:38, 8972.78it/s] 13%|█▎        | 51143/400000 [00:06<00:38, 8956.22it/s] 13%|█▎        | 52039/400000 [00:06<00:38, 8938.82it/s] 13%|█▎        | 52934/400000 [00:06<00:39, 8855.71it/s] 13%|█▎        | 53820/400000 [00:06<00:39, 8834.67it/s] 14%|█▎        | 54704/400000 [00:06<00:39, 8699.72it/s] 14%|█▍        | 55575/400000 [00:06<00:39, 8696.00it/s] 14%|█▍        | 56446/400000 [00:06<00:39, 8676.91it/s] 14%|█▍        | 57334/400000 [00:06<00:39, 8734.17it/s] 15%|█▍        | 58228/400000 [00:06<00:38, 8792.52it/s] 15%|█▍        | 59127/400000 [00:06<00:38, 8848.26it/s] 15%|█▌        | 60013/400000 [00:07<00:38, 8798.31it/s] 15%|█▌        | 60894/400000 [00:07<00:38, 8776.80it/s] 15%|█▌        | 61794/400000 [00:07<00:38, 8840.48it/s] 16%|█▌        | 62679/400000 [00:07<00:38, 8754.26it/s] 16%|█▌        | 63555/400000 [00:07<00:38, 8688.42it/s] 16%|█▌        | 64454/400000 [00:07<00:38, 8773.75it/s] 16%|█▋        | 65332/400000 [00:07<00:38, 8771.36it/s] 17%|█▋        | 66236/400000 [00:07<00:37, 8850.08it/s] 17%|█▋        | 67122/400000 [00:07<00:37, 8827.23it/s] 17%|█▋        | 68006/400000 [00:07<00:37, 8807.24it/s] 17%|█▋        | 68887/400000 [00:08<00:37, 8799.62it/s] 17%|█▋        | 69768/400000 [00:08<00:37, 8722.61it/s] 18%|█▊        | 70641/400000 [00:08<00:37, 8697.98it/s] 18%|█▊        | 71511/400000 [00:08<00:37, 8691.61it/s] 18%|█▊        | 72390/400000 [00:08<00:37, 8719.42it/s] 18%|█▊        | 73300/400000 [00:08<00:37, 8829.34it/s] 19%|█▊        | 74186/400000 [00:08<00:36, 8835.99it/s] 19%|█▉        | 75070/400000 [00:08<00:36, 8803.79it/s] 19%|█▉        | 75951/400000 [00:08<00:37, 8729.75it/s] 19%|█▉        | 76829/400000 [00:08<00:36, 8744.73it/s] 19%|█▉        | 77717/400000 [00:09<00:36, 8782.43it/s] 20%|█▉        | 78606/400000 [00:09<00:36, 8805.49it/s] 20%|█▉        | 79497/400000 [00:09<00:36, 8836.13it/s] 20%|██        | 80398/400000 [00:09<00:35, 8885.33it/s] 20%|██        | 81287/400000 [00:09<00:36, 8808.81it/s] 21%|██        | 82169/400000 [00:09<00:36, 8787.15it/s] 21%|██        | 83048/400000 [00:09<00:36, 8779.12it/s] 21%|██        | 83927/400000 [00:09<00:36, 8731.43it/s] 21%|██        | 84801/400000 [00:09<00:36, 8630.58it/s] 21%|██▏       | 85681/400000 [00:09<00:36, 8678.07it/s] 22%|██▏       | 86550/400000 [00:10<00:36, 8602.82it/s] 22%|██▏       | 87411/400000 [00:10<00:37, 8380.97it/s] 22%|██▏       | 88282/400000 [00:10<00:36, 8475.02it/s] 22%|██▏       | 89162/400000 [00:10<00:36, 8569.77it/s] 23%|██▎       | 90038/400000 [00:10<00:35, 8625.65it/s] 23%|██▎       | 90941/400000 [00:10<00:35, 8741.00it/s] 23%|██▎       | 91817/400000 [00:10<00:35, 8703.07it/s] 23%|██▎       | 92690/400000 [00:10<00:35, 8710.28it/s] 23%|██▎       | 93570/400000 [00:10<00:35, 8736.46it/s] 24%|██▎       | 94458/400000 [00:10<00:34, 8777.95it/s] 24%|██▍       | 95355/400000 [00:11<00:34, 8833.29it/s] 24%|██▍       | 96239/400000 [00:11<00:34, 8793.00it/s] 24%|██▍       | 97119/400000 [00:11<00:34, 8757.43it/s] 24%|██▍       | 97995/400000 [00:11<00:34, 8756.16it/s] 25%|██▍       | 98871/400000 [00:11<00:34, 8695.60it/s] 25%|██▍       | 99774/400000 [00:11<00:34, 8792.62it/s] 25%|██▌       | 100654/400000 [00:11<00:34, 8713.98it/s] 25%|██▌       | 101537/400000 [00:11<00:34, 8747.95it/s] 26%|██▌       | 102450/400000 [00:11<00:33, 8859.02it/s] 26%|██▌       | 103353/400000 [00:11<00:33, 8907.22it/s] 26%|██▌       | 104245/400000 [00:12<00:33, 8756.54it/s] 26%|██▋       | 105122/400000 [00:12<00:33, 8703.56it/s] 26%|██▋       | 105994/400000 [00:12<00:33, 8693.76it/s] 27%|██▋       | 106891/400000 [00:12<00:33, 8773.56it/s] 27%|██▋       | 107771/400000 [00:12<00:33, 8779.22it/s] 27%|██▋       | 108662/400000 [00:12<00:33, 8815.38it/s] 27%|██▋       | 109544/400000 [00:12<00:33, 8715.96it/s] 28%|██▊       | 110448/400000 [00:12<00:32, 8808.30it/s] 28%|██▊       | 111341/400000 [00:12<00:32, 8843.31it/s] 28%|██▊       | 112261/400000 [00:13<00:32, 8946.34it/s] 28%|██▊       | 113170/400000 [00:13<00:31, 8988.89it/s] 29%|██▊       | 114070/400000 [00:13<00:31, 8965.80it/s] 29%|██▊       | 114967/400000 [00:13<00:31, 8928.92it/s] 29%|██▉       | 115861/400000 [00:13<00:32, 8794.81it/s] 29%|██▉       | 116742/400000 [00:13<00:32, 8704.10it/s] 29%|██▉       | 117614/400000 [00:13<00:32, 8707.91it/s] 30%|██▉       | 118528/400000 [00:13<00:31, 8832.13it/s] 30%|██▉       | 119412/400000 [00:13<00:31, 8804.77it/s] 30%|███       | 120294/400000 [00:13<00:31, 8792.41it/s] 30%|███       | 121203/400000 [00:14<00:31, 8878.77it/s] 31%|███       | 122112/400000 [00:14<00:31, 8939.39it/s] 31%|███       | 123020/400000 [00:14<00:30, 8979.36it/s] 31%|███       | 123919/400000 [00:14<00:30, 8907.31it/s] 31%|███       | 124811/400000 [00:14<00:30, 8908.51it/s] 31%|███▏      | 125728/400000 [00:14<00:30, 8985.05it/s] 32%|███▏      | 126648/400000 [00:14<00:30, 9047.52it/s] 32%|███▏      | 127573/400000 [00:14<00:29, 9107.12it/s] 32%|███▏      | 128485/400000 [00:14<00:30, 8914.22it/s] 32%|███▏      | 129378/400000 [00:14<00:30, 8844.10it/s] 33%|███▎      | 130267/400000 [00:15<00:30, 8854.92it/s] 33%|███▎      | 131154/400000 [00:15<00:30, 8704.75it/s] 33%|███▎      | 132054/400000 [00:15<00:30, 8790.07it/s] 33%|███▎      | 132934/400000 [00:15<00:30, 8775.67it/s] 33%|███▎      | 133849/400000 [00:15<00:29, 8882.26it/s] 34%|███▎      | 134738/400000 [00:15<00:30, 8649.44it/s] 34%|███▍      | 135614/400000 [00:15<00:30, 8680.57it/s] 34%|███▍      | 136504/400000 [00:15<00:30, 8744.83it/s] 34%|███▍      | 137380/400000 [00:15<00:30, 8666.19it/s] 35%|███▍      | 138254/400000 [00:15<00:30, 8685.41it/s] 35%|███▍      | 139124/400000 [00:16<00:30, 8508.81it/s] 35%|███▍      | 139977/400000 [00:16<00:30, 8469.48it/s] 35%|███▌      | 140898/400000 [00:16<00:29, 8677.18it/s] 35%|███▌      | 141768/400000 [00:16<00:29, 8677.25it/s] 36%|███▌      | 142679/400000 [00:16<00:29, 8800.29it/s] 36%|███▌      | 143587/400000 [00:16<00:28, 8880.02it/s] 36%|███▌      | 144477/400000 [00:16<00:28, 8837.11it/s] 36%|███▋      | 145362/400000 [00:16<00:29, 8731.51it/s] 37%|███▋      | 146237/400000 [00:16<00:29, 8672.97it/s] 37%|███▋      | 147140/400000 [00:16<00:28, 8775.28it/s] 37%|███▋      | 148043/400000 [00:17<00:28, 8847.19it/s] 37%|███▋      | 148953/400000 [00:17<00:28, 8919.17it/s] 37%|███▋      | 149862/400000 [00:17<00:27, 8969.20it/s] 38%|███▊      | 150760/400000 [00:17<00:27, 8944.39it/s] 38%|███▊      | 151669/400000 [00:17<00:27, 8985.99it/s] 38%|███▊      | 152577/400000 [00:17<00:27, 9012.69it/s] 38%|███▊      | 153492/400000 [00:17<00:27, 9052.30it/s] 39%|███▊      | 154398/400000 [00:17<00:27, 9011.66it/s] 39%|███▉      | 155300/400000 [00:17<00:28, 8734.74it/s] 39%|███▉      | 156199/400000 [00:17<00:27, 8809.16it/s] 39%|███▉      | 157082/400000 [00:18<00:27, 8808.80it/s] 39%|███▉      | 157964/400000 [00:18<00:27, 8658.53it/s] 40%|███▉      | 158832/400000 [00:18<00:27, 8636.73it/s] 40%|███▉      | 159697/400000 [00:18<00:27, 8613.98it/s] 40%|████      | 160574/400000 [00:18<00:27, 8657.69it/s] 40%|████      | 161441/400000 [00:18<00:27, 8588.97it/s] 41%|████      | 162312/400000 [00:18<00:27, 8624.50it/s] 41%|████      | 163175/400000 [00:18<00:27, 8614.00it/s] 41%|████      | 164061/400000 [00:18<00:27, 8685.17it/s] 41%|████      | 164976/400000 [00:18<00:26, 8817.88it/s] 41%|████▏     | 165887/400000 [00:19<00:26, 8902.67it/s] 42%|████▏     | 166789/400000 [00:19<00:26, 8935.78it/s] 42%|████▏     | 167684/400000 [00:19<00:26, 8785.94it/s] 42%|████▏     | 168566/400000 [00:19<00:26, 8794.22it/s] 42%|████▏     | 169493/400000 [00:19<00:25, 8930.74it/s] 43%|████▎     | 170444/400000 [00:19<00:25, 9096.53it/s] 43%|████▎     | 171356/400000 [00:19<00:25, 9091.16it/s] 43%|████▎     | 172267/400000 [00:19<00:25, 9079.28it/s] 43%|████▎     | 173176/400000 [00:19<00:26, 8688.11it/s] 44%|████▎     | 174049/400000 [00:20<00:27, 8320.55it/s] 44%|████▎     | 174922/400000 [00:20<00:26, 8439.01it/s] 44%|████▍     | 175771/400000 [00:20<00:26, 8385.89it/s] 44%|████▍     | 176663/400000 [00:20<00:26, 8537.82it/s] 44%|████▍     | 177561/400000 [00:20<00:25, 8665.21it/s] 45%|████▍     | 178485/400000 [00:20<00:25, 8828.92it/s] 45%|████▍     | 179412/400000 [00:20<00:24, 8956.27it/s] 45%|████▌     | 180352/400000 [00:20<00:24, 9082.48it/s] 45%|████▌     | 181296/400000 [00:20<00:23, 9185.72it/s] 46%|████▌     | 182217/400000 [00:20<00:23, 9138.92it/s] 46%|████▌     | 183133/400000 [00:21<00:23, 9142.54it/s] 46%|████▌     | 184049/400000 [00:21<00:23, 9021.70it/s] 46%|████▌     | 184984/400000 [00:21<00:23, 9117.63it/s] 46%|████▋     | 185909/400000 [00:21<00:23, 9155.03it/s] 47%|████▋     | 186828/400000 [00:21<00:23, 9164.77it/s] 47%|████▋     | 187776/400000 [00:21<00:22, 9254.96it/s] 47%|████▋     | 188703/400000 [00:21<00:23, 9175.21it/s] 47%|████▋     | 189624/400000 [00:21<00:22, 9184.56it/s] 48%|████▊     | 190543/400000 [00:21<00:22, 9124.26it/s] 48%|████▊     | 191456/400000 [00:21<00:22, 9102.68it/s] 48%|████▊     | 192376/400000 [00:22<00:22, 9130.14it/s] 48%|████▊     | 193290/400000 [00:22<00:22, 9057.16it/s] 49%|████▊     | 194196/400000 [00:22<00:22, 9031.78it/s] 49%|████▉     | 195157/400000 [00:22<00:22, 9195.61it/s] 49%|████▉     | 196078/400000 [00:22<00:22, 9124.39it/s] 49%|████▉     | 197024/400000 [00:22<00:22, 9220.95it/s] 49%|████▉     | 197968/400000 [00:22<00:21, 9284.18it/s] 50%|████▉     | 198898/400000 [00:22<00:21, 9252.10it/s] 50%|████▉     | 199831/400000 [00:22<00:21, 9275.12it/s] 50%|█████     | 200759/400000 [00:22<00:21, 9142.36it/s] 50%|█████     | 201683/400000 [00:23<00:21, 9170.74it/s] 51%|█████     | 202611/400000 [00:23<00:21, 9200.80it/s] 51%|█████     | 203543/400000 [00:23<00:21, 9236.21it/s] 51%|█████     | 204473/400000 [00:23<00:21, 9254.18it/s] 51%|█████▏    | 205399/400000 [00:23<00:21, 9068.05it/s] 52%|█████▏    | 206307/400000 [00:23<00:21, 8824.06it/s] 52%|█████▏    | 207203/400000 [00:23<00:21, 8863.85it/s] 52%|█████▏    | 208105/400000 [00:23<00:21, 8908.85it/s] 52%|█████▏    | 208998/400000 [00:23<00:21, 8880.87it/s] 52%|█████▏    | 209887/400000 [00:23<00:21, 8854.41it/s] 53%|█████▎    | 210774/400000 [00:24<00:21, 8853.89it/s] 53%|█████▎    | 211660/400000 [00:24<00:21, 8840.71it/s] 53%|█████▎    | 212553/400000 [00:24<00:21, 8866.64it/s] 53%|█████▎    | 213440/400000 [00:24<00:21, 8863.83it/s] 54%|█████▎    | 214328/400000 [00:24<00:20, 8866.58it/s] 54%|█████▍    | 215293/400000 [00:24<00:20, 9086.75it/s] 54%|█████▍    | 216235/400000 [00:24<00:20, 9182.63it/s] 54%|█████▍    | 217200/400000 [00:24<00:19, 9317.37it/s] 55%|█████▍    | 218134/400000 [00:24<00:20, 9035.02it/s] 55%|█████▍    | 219041/400000 [00:24<00:20, 9009.89it/s] 55%|█████▍    | 219944/400000 [00:25<00:20, 8951.38it/s] 55%|█████▌    | 220850/400000 [00:25<00:19, 8981.24it/s] 55%|█████▌    | 221750/400000 [00:25<00:20, 8803.08it/s] 56%|█████▌    | 222670/400000 [00:25<00:19, 8916.75it/s] 56%|█████▌    | 223564/400000 [00:25<00:20, 8711.99it/s] 56%|█████▌    | 224479/400000 [00:25<00:19, 8837.68it/s] 56%|█████▋    | 225393/400000 [00:25<00:19, 8925.68it/s] 57%|█████▋    | 226300/400000 [00:25<00:19, 8967.24it/s] 57%|█████▋    | 227244/400000 [00:25<00:18, 9101.97it/s] 57%|█████▋    | 228168/400000 [00:26<00:18, 9140.73it/s] 57%|█████▋    | 229083/400000 [00:26<00:18, 9067.03it/s] 58%|█████▊    | 230008/400000 [00:26<00:18, 9118.96it/s] 58%|█████▊    | 230948/400000 [00:26<00:18, 9201.15it/s] 58%|█████▊    | 231878/400000 [00:26<00:18, 9228.35it/s] 58%|█████▊    | 232815/400000 [00:26<00:18, 9269.02it/s] 58%|█████▊    | 233743/400000 [00:26<00:18, 9181.55it/s] 59%|█████▊    | 234675/400000 [00:26<00:17, 9222.18it/s] 59%|█████▉    | 235598/400000 [00:26<00:17, 9167.38it/s] 59%|█████▉    | 236516/400000 [00:26<00:17, 9141.31it/s] 59%|█████▉    | 237431/400000 [00:27<00:18, 9017.71it/s] 60%|█████▉    | 238353/400000 [00:27<00:17, 9075.91it/s] 60%|█████▉    | 239262/400000 [00:27<00:18, 8895.44it/s] 60%|██████    | 240193/400000 [00:27<00:17, 9015.84it/s] 60%|██████    | 241110/400000 [00:27<00:17, 9059.13it/s] 61%|██████    | 242031/400000 [00:27<00:17, 9102.51it/s] 61%|██████    | 242986/400000 [00:27<00:17, 9229.74it/s] 61%|██████    | 243952/400000 [00:27<00:16, 9352.99it/s] 61%|██████    | 244889/400000 [00:27<00:16, 9221.28it/s] 61%|██████▏   | 245820/400000 [00:27<00:16, 9245.04it/s] 62%|██████▏   | 246746/400000 [00:28<00:16, 9154.57it/s] 62%|██████▏   | 247663/400000 [00:28<00:16, 9153.79it/s] 62%|██████▏   | 248581/400000 [00:28<00:16, 9161.50it/s] 62%|██████▏   | 249517/400000 [00:28<00:16, 9218.17it/s] 63%|██████▎   | 250440/400000 [00:28<00:16, 9143.02it/s] 63%|██████▎   | 251355/400000 [00:28<00:16, 8991.59it/s] 63%|██████▎   | 252259/400000 [00:28<00:16, 9004.54it/s] 63%|██████▎   | 253163/400000 [00:28<00:16, 9012.64it/s] 64%|██████▎   | 254105/400000 [00:28<00:15, 9130.72it/s] 64%|██████▍   | 255041/400000 [00:28<00:15, 9198.10it/s] 64%|██████▍   | 255962/400000 [00:29<00:15, 9136.62it/s] 64%|██████▍   | 256884/400000 [00:29<00:15, 9158.56it/s] 64%|██████▍   | 257801/400000 [00:29<00:15, 9121.81it/s] 65%|██████▍   | 258762/400000 [00:29<00:15, 9261.51it/s] 65%|██████▍   | 259700/400000 [00:29<00:15, 9295.87it/s] 65%|██████▌   | 260631/400000 [00:29<00:15, 9240.30it/s] 65%|██████▌   | 261562/400000 [00:29<00:14, 9260.14it/s] 66%|██████▌   | 262489/400000 [00:29<00:14, 9240.65it/s] 66%|██████▌   | 263414/400000 [00:29<00:14, 9148.46it/s] 66%|██████▌   | 264344/400000 [00:29<00:14, 9190.99it/s] 66%|██████▋   | 265268/400000 [00:30<00:14, 9204.14it/s] 67%|██████▋   | 266189/400000 [00:30<00:14, 9028.21it/s] 67%|██████▋   | 267118/400000 [00:30<00:14, 9101.94it/s] 67%|██████▋   | 268033/400000 [00:30<00:14, 9114.85it/s] 67%|██████▋   | 268946/400000 [00:30<00:14, 9102.41it/s] 67%|██████▋   | 269857/400000 [00:30<00:14, 9088.26it/s] 68%|██████▊   | 270820/400000 [00:30<00:13, 9242.71it/s] 68%|██████▊   | 271769/400000 [00:30<00:13, 9314.76it/s] 68%|██████▊   | 272702/400000 [00:30<00:13, 9259.22it/s] 68%|██████▊   | 273629/400000 [00:30<00:13, 9232.97it/s] 69%|██████▊   | 274553/400000 [00:31<00:13, 9016.27it/s] 69%|██████▉   | 275461/400000 [00:31<00:13, 9034.50it/s] 69%|██████▉   | 276392/400000 [00:31<00:13, 9113.48it/s] 69%|██████▉   | 277305/400000 [00:31<00:14, 8704.85it/s] 70%|██████▉   | 278253/400000 [00:31<00:13, 8923.36it/s] 70%|██████▉   | 279159/400000 [00:31<00:13, 8962.80it/s] 70%|███████   | 280118/400000 [00:31<00:13, 9142.01it/s] 70%|███████   | 281050/400000 [00:31<00:12, 9193.94it/s] 70%|███████   | 281993/400000 [00:31<00:12, 9263.37it/s] 71%|███████   | 282922/400000 [00:31<00:12, 9225.83it/s] 71%|███████   | 283846/400000 [00:32<00:12, 9088.69it/s] 71%|███████   | 284757/400000 [00:32<00:12, 9033.71it/s] 71%|███████▏  | 285670/400000 [00:32<00:12, 9060.01it/s] 72%|███████▏  | 286578/400000 [00:32<00:12, 9064.80it/s] 72%|███████▏  | 287485/400000 [00:32<00:12, 9061.42it/s] 72%|███████▏  | 288392/400000 [00:32<00:12, 8863.71it/s] 72%|███████▏  | 289298/400000 [00:32<00:12, 8919.90it/s] 73%|███████▎  | 290242/400000 [00:32<00:12, 9067.48it/s] 73%|███████▎  | 291185/400000 [00:32<00:11, 9170.61it/s] 73%|███████▎  | 292110/400000 [00:33<00:11, 9192.84it/s] 73%|███████▎  | 293036/400000 [00:33<00:11, 9212.55it/s] 73%|███████▎  | 293958/400000 [00:33<00:11, 9176.02it/s] 74%|███████▎  | 294877/400000 [00:33<00:11, 9136.65it/s] 74%|███████▍  | 295792/400000 [00:33<00:11, 9125.32it/s] 74%|███████▍  | 296705/400000 [00:33<00:11, 9125.87it/s] 74%|███████▍  | 297618/400000 [00:33<00:11, 9081.86it/s] 75%|███████▍  | 298547/400000 [00:33<00:11, 9142.35it/s] 75%|███████▍  | 299462/400000 [00:33<00:10, 9140.69it/s] 75%|███████▌  | 300377/400000 [00:33<00:10, 9123.70it/s] 75%|███████▌  | 301309/400000 [00:34<00:10, 9181.47it/s] 76%|███████▌  | 302257/400000 [00:34<00:10, 9268.27it/s] 76%|███████▌  | 303206/400000 [00:34<00:10, 9332.37it/s] 76%|███████▌  | 304140/400000 [00:34<00:10, 9265.71it/s] 76%|███████▋  | 305067/400000 [00:34<00:10, 8904.80it/s] 76%|███████▋  | 305961/400000 [00:34<00:10, 8909.81it/s] 77%|███████▋  | 306855/400000 [00:34<00:10, 8862.59it/s] 77%|███████▋  | 307743/400000 [00:34<00:10, 8703.69it/s] 77%|███████▋  | 308635/400000 [00:34<00:10, 8765.23it/s] 77%|███████▋  | 309546/400000 [00:34<00:10, 8864.82it/s] 78%|███████▊  | 310474/400000 [00:35<00:09, 8984.23it/s] 78%|███████▊  | 311379/400000 [00:35<00:09, 9002.25it/s] 78%|███████▊  | 312297/400000 [00:35<00:09, 9053.23it/s] 78%|███████▊  | 313204/400000 [00:35<00:09, 9036.53it/s] 79%|███████▊  | 314109/400000 [00:35<00:09, 9008.94it/s] 79%|███████▉  | 315011/400000 [00:35<00:09, 8949.09it/s] 79%|███████▉  | 315907/400000 [00:35<00:09, 8810.07it/s] 79%|███████▉  | 316816/400000 [00:35<00:09, 8890.20it/s] 79%|███████▉  | 317738/400000 [00:35<00:09, 8986.33it/s] 80%|███████▉  | 318656/400000 [00:35<00:08, 9041.91it/s] 80%|███████▉  | 319577/400000 [00:36<00:08, 9090.12it/s] 80%|████████  | 320487/400000 [00:36<00:08, 8876.65it/s] 80%|████████  | 321393/400000 [00:36<00:08, 8929.38it/s] 81%|████████  | 322303/400000 [00:36<00:08, 8977.92it/s] 81%|████████  | 323202/400000 [00:36<00:08, 8976.82it/s] 81%|████████  | 324101/400000 [00:36<00:08, 8948.10it/s] 81%|████████  | 324997/400000 [00:36<00:08, 8888.14it/s] 81%|████████▏ | 325887/400000 [00:36<00:08, 8882.19it/s] 82%|████████▏ | 326811/400000 [00:36<00:08, 8985.05it/s] 82%|████████▏ | 327723/400000 [00:36<00:08, 9023.92it/s] 82%|████████▏ | 328629/400000 [00:37<00:07, 9032.65it/s] 82%|████████▏ | 329533/400000 [00:37<00:07, 8956.56it/s] 83%|████████▎ | 330440/400000 [00:37<00:07, 8987.57it/s] 83%|████████▎ | 331358/400000 [00:37<00:07, 9042.29it/s] 83%|████████▎ | 332278/400000 [00:37<00:07, 9088.14it/s] 83%|████████▎ | 333192/400000 [00:37<00:07, 9100.89it/s] 84%|████████▎ | 334103/400000 [00:37<00:07, 8968.69it/s] 84%|████████▍ | 335001/400000 [00:37<00:07, 8962.17it/s] 84%|████████▍ | 335918/400000 [00:37<00:07, 9023.38it/s] 84%|████████▍ | 336838/400000 [00:37<00:06, 9073.96it/s] 84%|████████▍ | 337748/400000 [00:38<00:06, 9081.32it/s] 85%|████████▍ | 338657/400000 [00:38<00:06, 8907.58it/s] 85%|████████▍ | 339572/400000 [00:38<00:06, 8976.67it/s] 85%|████████▌ | 340478/400000 [00:38<00:06, 8999.50it/s] 85%|████████▌ | 341379/400000 [00:38<00:06, 8826.46it/s] 86%|████████▌ | 342269/400000 [00:38<00:06, 8847.34it/s] 86%|████████▌ | 343169/400000 [00:38<00:06, 8891.48it/s] 86%|████████▌ | 344074/400000 [00:38<00:06, 8935.54it/s] 86%|████████▌ | 344995/400000 [00:38<00:06, 9013.58it/s] 86%|████████▋ | 345897/400000 [00:38<00:06, 8772.35it/s] 87%|████████▋ | 346805/400000 [00:39<00:06, 8860.26it/s] 87%|████████▋ | 347711/400000 [00:39<00:05, 8917.87it/s] 87%|████████▋ | 348649/400000 [00:39<00:05, 9050.18it/s] 87%|████████▋ | 349629/400000 [00:39<00:05, 9262.10it/s] 88%|████████▊ | 350571/400000 [00:39<00:05, 9306.74it/s] 88%|████████▊ | 351504/400000 [00:39<00:05, 9247.87it/s] 88%|████████▊ | 352430/400000 [00:39<00:05, 9186.64it/s] 88%|████████▊ | 353350/400000 [00:39<00:05, 9177.57it/s] 89%|████████▊ | 354274/400000 [00:39<00:04, 9194.60it/s] 89%|████████▉ | 355194/400000 [00:40<00:04, 9122.99it/s] 89%|████████▉ | 356107/400000 [00:40<00:04, 9086.21it/s] 89%|████████▉ | 357016/400000 [00:40<00:04, 9017.71it/s] 89%|████████▉ | 357933/400000 [00:40<00:04, 9062.59it/s] 90%|████████▉ | 358840/400000 [00:40<00:04, 9034.92it/s] 90%|████████▉ | 359744/400000 [00:40<00:04, 9005.58it/s] 90%|█████████ | 360655/400000 [00:40<00:04, 9035.61it/s] 90%|█████████ | 361559/400000 [00:40<00:04, 9035.46it/s] 91%|█████████ | 362472/400000 [00:40<00:04, 9062.03it/s] 91%|█████████ | 363382/400000 [00:40<00:04, 9071.35it/s] 91%|█████████ | 364304/400000 [00:41<00:03, 9113.63it/s] 91%|█████████▏| 365216/400000 [00:41<00:03, 9006.61it/s] 92%|█████████▏| 366118/400000 [00:41<00:03, 8992.81it/s] 92%|█████████▏| 367018/400000 [00:41<00:03, 8984.61it/s] 92%|█████████▏| 367917/400000 [00:41<00:03, 8886.92it/s] 92%|█████████▏| 368838/400000 [00:41<00:03, 8980.41it/s] 92%|█████████▏| 369737/400000 [00:41<00:03, 8953.01it/s] 93%|█████████▎| 370633/400000 [00:41<00:03, 8802.43it/s] 93%|█████████▎| 371539/400000 [00:41<00:03, 8876.45it/s] 93%|█████████▎| 372434/400000 [00:41<00:03, 8896.54it/s] 93%|█████████▎| 373353/400000 [00:42<00:02, 8980.91it/s] 94%|█████████▎| 374252/400000 [00:42<00:02, 8786.19it/s] 94%|█████████▍| 375132/400000 [00:42<00:02, 8730.96it/s] 94%|█████████▍| 376007/400000 [00:42<00:02, 8675.68it/s] 94%|█████████▍| 376894/400000 [00:42<00:02, 8731.48it/s] 94%|█████████▍| 377800/400000 [00:42<00:02, 8827.09it/s] 95%|█████████▍| 378720/400000 [00:42<00:02, 8934.35it/s] 95%|█████████▍| 379615/400000 [00:42<00:02, 8919.77it/s] 95%|█████████▌| 380508/400000 [00:42<00:02, 8907.85it/s] 95%|█████████▌| 381421/400000 [00:42<00:02, 8972.73it/s] 96%|█████████▌| 382333/400000 [00:43<00:01, 9015.69it/s] 96%|█████████▌| 383235/400000 [00:43<00:01, 9010.21it/s] 96%|█████████▌| 384151/400000 [00:43<00:01, 9052.61it/s] 96%|█████████▋| 385057/400000 [00:43<00:01, 8998.72it/s] 96%|█████████▋| 385958/400000 [00:43<00:01, 8958.57it/s] 97%|█████████▋| 386878/400000 [00:43<00:01, 9027.08it/s] 97%|█████████▋| 387781/400000 [00:43<00:01, 8994.31it/s] 97%|█████████▋| 388681/400000 [00:43<00:01, 8952.65it/s] 97%|█████████▋| 389577/400000 [00:43<00:01, 8909.00it/s] 98%|█████████▊| 390479/400000 [00:43<00:01, 8939.76it/s] 98%|█████████▊| 391377/400000 [00:44<00:00, 8949.56it/s] 98%|█████████▊| 392273/400000 [00:44<00:00, 8926.79it/s] 98%|█████████▊| 393182/400000 [00:44<00:00, 8973.98it/s] 99%|█████████▊| 394082/400000 [00:44<00:00, 8981.76it/s] 99%|█████████▊| 394996/400000 [00:44<00:00, 9026.49it/s] 99%|█████████▉| 395899/400000 [00:44<00:00, 9002.81it/s] 99%|█████████▉| 396826/400000 [00:44<00:00, 9078.81it/s] 99%|█████████▉| 397735/400000 [00:44<00:00, 9069.09it/s]100%|█████████▉| 398643/400000 [00:44<00:00, 8966.34it/s]100%|█████████▉| 399559/400000 [00:44<00:00, 9022.42it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8887.77it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fddd0d880f0>, <torchtext.data.dataset.TabularDataset object at 0x7fddd0d88240>, <torchtext.vocab.Vocab object at 0x7fddd0d88160>), {}) 


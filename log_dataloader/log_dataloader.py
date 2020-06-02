
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fc77960e2f0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fc77960e2f0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fc7eb65e0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fc7eb65e0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fc782df49d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fc782df49d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc798f428c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc798f428c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc798f428c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:04, 154602.28it/s] 63%|██████▎   | 6250496/9912422 [00:00<00:16, 220604.35it/s]9920512it [00:00, 38872186.08it/s]                           
0it [00:00, ?it/s]32768it [00:00, 381569.25it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 10953350.39it/s]         
0it [00:00, ?it/s]8192it [00:00, 152692.30it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc779613940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc77b95c048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fc798f42510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fc798f42510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fc798f42510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:09,  2.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:09,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:08,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:08,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:07,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:07,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:07,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:06,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:46,  3.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:46,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:46,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:46,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:46,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:45,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:45,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:45,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:44,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:44,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:31,  4.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:31,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:31,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:30,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:30,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:30,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:30,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:29,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:21,  6.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:21,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:21,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:20,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:20,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:20,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:20,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:20,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:20,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  8.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:14,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:14,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:14,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:00<00:09, 12.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:09, 12.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:09, 12.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:09, 12.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:09, 12.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:09, 12.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 12.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 12.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 12.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 16.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 16.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 16.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 16.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 16.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 16.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 16.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 16.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 16.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 16.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 21.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 21.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 21.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 21.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 21.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 21.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 21.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 27.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 27.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 34.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 34.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 34.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 34.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 34.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 34.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 34.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 34.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 34.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 34.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 34.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 42.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 42.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 42.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 42.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 42.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 42.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 42.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 42.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 42.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 42.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 49.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 49.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 49.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 49.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 49.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 49.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 49.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 49.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 49.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 56.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 62.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 62.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 62.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 62.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 62.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 62.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 62.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 62.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 62.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 62.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 67.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 67.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 67.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 67.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 67.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 67.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 67.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 67.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 67.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 67.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 67.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 73.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 73.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 73.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 73.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 73.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 73.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 76.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 76.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 76.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 76.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 76.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 76.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 79.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.36s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.40s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 79.75 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.40s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.40s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.78 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ url]
0 examples [00:00, ? examples/s]2020-06-02 00:09:12.921058: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-02 00:09:12.932541: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-02 00:09:12.932756: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f30c545da0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-02 00:09:12.932773: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
33 examples [00:00, 323.80 examples/s]140 examples [00:00, 409.33 examples/s]257 examples [00:00, 508.33 examples/s]347 examples [00:00, 583.98 examples/s]461 examples [00:00, 684.04 examples/s]579 examples [00:00, 782.04 examples/s]707 examples [00:00, 884.21 examples/s]811 examples [00:00, 919.17 examples/s]914 examples [00:00, 945.71 examples/s]1042 examples [00:01, 1025.15 examples/s]1174 examples [00:01, 1096.47 examples/s]1300 examples [00:01, 1138.50 examples/s]1432 examples [00:01, 1185.53 examples/s]1555 examples [00:01, 1159.07 examples/s]1676 examples [00:01, 1171.96 examples/s]1801 examples [00:01, 1194.13 examples/s]1926 examples [00:01, 1209.94 examples/s]2049 examples [00:01, 1209.62 examples/s]2171 examples [00:01, 1157.23 examples/s]2291 examples [00:02, 1167.68 examples/s]2415 examples [00:02, 1187.24 examples/s]2535 examples [00:02, 1155.16 examples/s]2659 examples [00:02, 1178.75 examples/s]2778 examples [00:02, 1128.59 examples/s]2897 examples [00:02, 1144.74 examples/s]3028 examples [00:02, 1189.25 examples/s]3153 examples [00:02, 1204.88 examples/s]3278 examples [00:02, 1217.84 examples/s]3401 examples [00:02, 1195.01 examples/s]3521 examples [00:03, 1127.15 examples/s]3643 examples [00:03, 1149.73 examples/s]3770 examples [00:03, 1183.07 examples/s]3890 examples [00:03, 1167.94 examples/s]4011 examples [00:03, 1178.64 examples/s]4134 examples [00:03, 1192.79 examples/s]4259 examples [00:03, 1208.30 examples/s]4381 examples [00:03, 1198.19 examples/s]4502 examples [00:03, 1195.19 examples/s]4622 examples [00:04, 1192.06 examples/s]4742 examples [00:04, 1163.78 examples/s]4865 examples [00:04, 1181.16 examples/s]4996 examples [00:04, 1214.54 examples/s]5118 examples [00:04, 1216.05 examples/s]5240 examples [00:04, 1212.47 examples/s]5364 examples [00:04, 1217.77 examples/s]5486 examples [00:04, 1186.81 examples/s]5607 examples [00:04, 1191.16 examples/s]5727 examples [00:04, 1150.63 examples/s]5843 examples [00:05, 1138.98 examples/s]5962 examples [00:05, 1151.62 examples/s]6078 examples [00:05, 1140.70 examples/s]6193 examples [00:05, 1142.39 examples/s]6310 examples [00:05, 1149.87 examples/s]6432 examples [00:05, 1164.17 examples/s]6557 examples [00:05, 1186.29 examples/s]6679 examples [00:05, 1194.23 examples/s]6799 examples [00:05, 1185.54 examples/s]6918 examples [00:05, 1156.26 examples/s]7037 examples [00:06, 1165.20 examples/s]7154 examples [00:06, 1104.96 examples/s]7266 examples [00:06, 1085.53 examples/s]7381 examples [00:06, 1102.49 examples/s]7492 examples [00:06, 1077.59 examples/s]7601 examples [00:06, 1077.28 examples/s]7710 examples [00:06, 1063.98 examples/s]7828 examples [00:06, 1095.53 examples/s]7938 examples [00:06, 1087.79 examples/s]8053 examples [00:07, 1105.34 examples/s]8173 examples [00:07, 1130.93 examples/s]8296 examples [00:07, 1158.42 examples/s]8413 examples [00:07, 1087.25 examples/s]8523 examples [00:07, 1045.07 examples/s]8629 examples [00:07, 1034.84 examples/s]8736 examples [00:07, 1043.38 examples/s]8855 examples [00:07, 1082.23 examples/s]8966 examples [00:07, 1088.87 examples/s]9076 examples [00:07, 1082.78 examples/s]9190 examples [00:08, 1098.63 examples/s]9304 examples [00:08, 1110.61 examples/s]9416 examples [00:08, 1101.31 examples/s]9527 examples [00:08, 1058.23 examples/s]9639 examples [00:08, 1074.42 examples/s]9762 examples [00:08, 1115.71 examples/s]9882 examples [00:08, 1137.38 examples/s]10001 examples [00:08, 1122.30 examples/s]10127 examples [00:08, 1158.30 examples/s]10244 examples [00:08, 1155.96 examples/s]10361 examples [00:09, 1152.44 examples/s]10477 examples [00:09, 1099.16 examples/s]10599 examples [00:09, 1131.21 examples/s]10721 examples [00:09, 1154.65 examples/s]10840 examples [00:09, 1164.93 examples/s]10964 examples [00:09, 1185.44 examples/s]11085 examples [00:09, 1192.37 examples/s]11214 examples [00:09, 1219.00 examples/s]11337 examples [00:09, 1194.44 examples/s]11457 examples [00:10, 1146.76 examples/s]11573 examples [00:10, 1113.47 examples/s]11686 examples [00:10, 1115.33 examples/s]11800 examples [00:10, 1120.41 examples/s]11913 examples [00:10, 1119.58 examples/s]12026 examples [00:10, 1100.66 examples/s]12137 examples [00:10, 1093.92 examples/s]12247 examples [00:10, 1091.10 examples/s]12357 examples [00:10, 1088.13 examples/s]12469 examples [00:10, 1096.37 examples/s]12579 examples [00:11, 1091.21 examples/s]12694 examples [00:11, 1105.55 examples/s]12807 examples [00:11, 1110.87 examples/s]12924 examples [00:11, 1126.06 examples/s]13037 examples [00:11, 1127.14 examples/s]13150 examples [00:11, 1118.13 examples/s]13262 examples [00:11, 1118.52 examples/s]13377 examples [00:11, 1127.39 examples/s]13490 examples [00:11, 1113.45 examples/s]13602 examples [00:11, 1114.44 examples/s]13714 examples [00:12, 1108.76 examples/s]13825 examples [00:12, 1103.01 examples/s]13936 examples [00:12, 1067.55 examples/s]14044 examples [00:12, 1033.50 examples/s]14148 examples [00:12, 1012.32 examples/s]14257 examples [00:12, 1032.69 examples/s]14361 examples [00:12, 1031.08 examples/s]14467 examples [00:12, 1039.49 examples/s]14577 examples [00:12, 1054.30 examples/s]14685 examples [00:13, 1059.86 examples/s]14802 examples [00:13, 1090.16 examples/s]14920 examples [00:13, 1113.49 examples/s]15036 examples [00:13, 1126.88 examples/s]15149 examples [00:13, 1126.57 examples/s]15263 examples [00:13, 1128.03 examples/s]15376 examples [00:13, 1125.43 examples/s]15489 examples [00:13, 1126.14 examples/s]15602 examples [00:13, 1118.54 examples/s]15719 examples [00:13, 1132.68 examples/s]15843 examples [00:14, 1162.27 examples/s]15960 examples [00:14, 1125.77 examples/s]16074 examples [00:14, 1093.76 examples/s]16191 examples [00:14, 1115.18 examples/s]16317 examples [00:14, 1154.26 examples/s]16438 examples [00:14, 1169.13 examples/s]16556 examples [00:14, 1170.69 examples/s]16682 examples [00:14, 1194.71 examples/s]16811 examples [00:14, 1220.41 examples/s]16941 examples [00:14, 1240.89 examples/s]17067 examples [00:15, 1243.84 examples/s]17192 examples [00:15, 1219.07 examples/s]17316 examples [00:15, 1224.25 examples/s]17439 examples [00:15, 1187.29 examples/s]17560 examples [00:15, 1191.67 examples/s]17680 examples [00:15, 1180.51 examples/s]17799 examples [00:15, 1135.99 examples/s]17914 examples [00:15, 1095.12 examples/s]18025 examples [00:15, 1088.91 examples/s]18140 examples [00:16, 1105.53 examples/s]18251 examples [00:16, 1104.32 examples/s]18366 examples [00:16, 1115.96 examples/s]18478 examples [00:16, 1109.86 examples/s]18590 examples [00:16, 1107.29 examples/s]18701 examples [00:16, 1107.81 examples/s]18812 examples [00:16, 1107.32 examples/s]18925 examples [00:16, 1113.97 examples/s]19037 examples [00:16, 1105.96 examples/s]19148 examples [00:16, 1078.46 examples/s]19262 examples [00:17, 1093.94 examples/s]19372 examples [00:17, 1069.76 examples/s]19484 examples [00:17, 1083.74 examples/s]19603 examples [00:17, 1112.48 examples/s]19715 examples [00:17, 1099.64 examples/s]19826 examples [00:17, 1084.85 examples/s]19935 examples [00:17, 1068.06 examples/s]20043 examples [00:17, 1025.65 examples/s]20159 examples [00:17, 1060.27 examples/s]20280 examples [00:17, 1098.74 examples/s]20405 examples [00:18, 1139.43 examples/s]20523 examples [00:18, 1149.52 examples/s]20646 examples [00:18, 1171.00 examples/s]20764 examples [00:18, 1136.67 examples/s]20879 examples [00:18, 1076.46 examples/s]20999 examples [00:18, 1105.90 examples/s]21114 examples [00:18, 1116.24 examples/s]21232 examples [00:18, 1131.48 examples/s]21346 examples [00:18, 1124.71 examples/s]21459 examples [00:19, 1110.35 examples/s]21571 examples [00:19, 1090.03 examples/s]21682 examples [00:19, 1093.88 examples/s]21795 examples [00:19, 1103.75 examples/s]21909 examples [00:19, 1113.31 examples/s]22032 examples [00:19, 1143.29 examples/s]22156 examples [00:19, 1169.98 examples/s]22274 examples [00:19, 1144.52 examples/s]22389 examples [00:19, 1143.67 examples/s]22504 examples [00:19, 1144.29 examples/s]22619 examples [00:20, 1142.59 examples/s]22739 examples [00:20, 1156.78 examples/s]22855 examples [00:20, 1151.18 examples/s]22974 examples [00:20, 1162.09 examples/s]23091 examples [00:20, 1110.37 examples/s]23211 examples [00:20, 1134.95 examples/s]23326 examples [00:20, 1127.79 examples/s]23444 examples [00:20, 1141.82 examples/s]23559 examples [00:20, 1134.52 examples/s]23673 examples [00:20, 1132.42 examples/s]23788 examples [00:21, 1135.08 examples/s]23909 examples [00:21, 1154.55 examples/s]24025 examples [00:21, 1147.65 examples/s]24140 examples [00:21, 1140.21 examples/s]24255 examples [00:21, 1088.15 examples/s]24381 examples [00:21, 1133.58 examples/s]24509 examples [00:21, 1173.63 examples/s]24640 examples [00:21, 1211.32 examples/s]24763 examples [00:21, 1207.43 examples/s]24885 examples [00:21, 1202.38 examples/s]25006 examples [00:22, 1183.18 examples/s]25125 examples [00:22, 1165.13 examples/s]25242 examples [00:22, 1157.43 examples/s]25360 examples [00:22, 1163.78 examples/s]25477 examples [00:22, 1164.07 examples/s]25594 examples [00:22, 1155.11 examples/s]25710 examples [00:22, 1144.95 examples/s]25826 examples [00:22, 1146.88 examples/s]25941 examples [00:22, 1138.45 examples/s]26055 examples [00:23, 1109.72 examples/s]26171 examples [00:23, 1124.29 examples/s]26284 examples [00:23, 1118.44 examples/s]26412 examples [00:23, 1161.70 examples/s]26536 examples [00:23, 1182.92 examples/s]26656 examples [00:23, 1186.97 examples/s]26776 examples [00:23, 1173.07 examples/s]26894 examples [00:23, 1158.07 examples/s]27014 examples [00:23, 1168.32 examples/s]27137 examples [00:23, 1184.18 examples/s]27259 examples [00:24, 1192.31 examples/s]27379 examples [00:24, 1178.86 examples/s]27498 examples [00:24, 1157.14 examples/s]27614 examples [00:24, 1151.49 examples/s]27737 examples [00:24, 1172.42 examples/s]27855 examples [00:24, 1164.67 examples/s]27984 examples [00:24, 1197.58 examples/s]28105 examples [00:24, 1183.96 examples/s]28233 examples [00:24, 1207.99 examples/s]28360 examples [00:24, 1224.60 examples/s]28483 examples [00:25, 1203.08 examples/s]28604 examples [00:25, 1175.39 examples/s]28722 examples [00:25, 1152.64 examples/s]28838 examples [00:25, 1143.56 examples/s]28965 examples [00:25, 1178.52 examples/s]29084 examples [00:25, 1169.17 examples/s]29202 examples [00:25, 1143.37 examples/s]29318 examples [00:25, 1145.71 examples/s]29433 examples [00:25, 1128.55 examples/s]29549 examples [00:25, 1136.56 examples/s]29676 examples [00:26, 1171.07 examples/s]29797 examples [00:26, 1180.42 examples/s]29916 examples [00:26, 1144.08 examples/s]30031 examples [00:26, 1104.19 examples/s]30148 examples [00:26, 1121.42 examples/s]30266 examples [00:26, 1138.32 examples/s]30391 examples [00:26, 1167.01 examples/s]30509 examples [00:26, 1160.57 examples/s]30638 examples [00:26, 1195.64 examples/s]30759 examples [00:27, 1189.95 examples/s]30879 examples [00:27, 1180.47 examples/s]30998 examples [00:27, 1152.23 examples/s]31116 examples [00:27, 1159.81 examples/s]31234 examples [00:27, 1164.96 examples/s]31351 examples [00:27, 1136.09 examples/s]31465 examples [00:27, 1108.64 examples/s]31598 examples [00:27, 1165.20 examples/s]31719 examples [00:27, 1177.57 examples/s]31838 examples [00:27, 1142.91 examples/s]31954 examples [00:28, 1101.48 examples/s]32065 examples [00:28, 1099.94 examples/s]32176 examples [00:28, 1102.76 examples/s]32287 examples [00:28, 1100.36 examples/s]32398 examples [00:28, 1094.95 examples/s]32520 examples [00:28, 1127.14 examples/s]32634 examples [00:28, 1108.66 examples/s]32746 examples [00:28, 1102.12 examples/s]32861 examples [00:28, 1113.32 examples/s]32982 examples [00:29, 1139.00 examples/s]33097 examples [00:29, 1134.48 examples/s]33211 examples [00:29, 1076.43 examples/s]33320 examples [00:29, 1069.05 examples/s]33432 examples [00:29, 1081.76 examples/s]33562 examples [00:29, 1137.53 examples/s]33677 examples [00:29, 1128.42 examples/s]33794 examples [00:29, 1139.51 examples/s]33909 examples [00:29, 1126.88 examples/s]34023 examples [00:29, 1118.20 examples/s]34146 examples [00:30, 1135.92 examples/s]34269 examples [00:30, 1160.79 examples/s]34392 examples [00:30, 1180.07 examples/s]34513 examples [00:30, 1188.01 examples/s]34633 examples [00:30, 1137.14 examples/s]34748 examples [00:30, 1138.15 examples/s]34863 examples [00:30, 1098.37 examples/s]34977 examples [00:30, 1110.37 examples/s]35089 examples [00:30, 1089.96 examples/s]35199 examples [00:31, 1028.23 examples/s]35317 examples [00:31, 1067.80 examples/s]35426 examples [00:31, 1072.96 examples/s]35538 examples [00:31, 1085.06 examples/s]35648 examples [00:31, 1079.81 examples/s]35758 examples [00:31, 1085.23 examples/s]35873 examples [00:31, 1102.91 examples/s]35993 examples [00:31, 1127.37 examples/s]36115 examples [00:31, 1152.31 examples/s]36231 examples [00:31, 1149.97 examples/s]36347 examples [00:32, 1116.06 examples/s]36466 examples [00:32, 1134.80 examples/s]36580 examples [00:32, 1135.31 examples/s]36694 examples [00:32, 1120.44 examples/s]36807 examples [00:32, 1086.91 examples/s]36917 examples [00:32, 1056.76 examples/s]37028 examples [00:32, 1069.58 examples/s]37136 examples [00:32, 1030.85 examples/s]37242 examples [00:32, 1037.06 examples/s]37347 examples [00:32, 1031.40 examples/s]37458 examples [00:33, 1051.65 examples/s]37572 examples [00:33, 1074.98 examples/s]37697 examples [00:33, 1120.86 examples/s]37810 examples [00:33, 1111.78 examples/s]37922 examples [00:33, 1089.20 examples/s]38035 examples [00:33, 1101.10 examples/s]38146 examples [00:33, 1068.54 examples/s]38254 examples [00:33, 1054.99 examples/s]38360 examples [00:33, 1022.48 examples/s]38463 examples [00:34, 1010.87 examples/s]38571 examples [00:34, 1029.22 examples/s]38679 examples [00:34, 1041.91 examples/s]38784 examples [00:34, 1002.68 examples/s]38891 examples [00:34, 1021.65 examples/s]39001 examples [00:34, 1042.83 examples/s]39113 examples [00:34, 1062.13 examples/s]39220 examples [00:34, 1031.79 examples/s]39332 examples [00:34, 1054.63 examples/s]39445 examples [00:34, 1074.56 examples/s]39553 examples [00:35, 1054.80 examples/s]39660 examples [00:35, 1058.81 examples/s]39768 examples [00:35, 1063.21 examples/s]39875 examples [00:35, 1037.21 examples/s]39991 examples [00:35, 1069.36 examples/s]40099 examples [00:35, 1024.08 examples/s]40217 examples [00:35, 1064.56 examples/s]40342 examples [00:35, 1113.71 examples/s]40455 examples [00:35, 1111.82 examples/s]40570 examples [00:35, 1094.13 examples/s]40687 examples [00:36, 1115.13 examples/s]40806 examples [00:36, 1135.69 examples/s]40928 examples [00:36, 1152.06 examples/s]41044 examples [00:36, 1123.59 examples/s]41157 examples [00:36, 1081.24 examples/s]41266 examples [00:36, 1076.90 examples/s]41375 examples [00:36, 1078.51 examples/s]41484 examples [00:36, 1053.78 examples/s]41594 examples [00:36, 1066.64 examples/s]41711 examples [00:37, 1093.24 examples/s]41835 examples [00:37, 1131.32 examples/s]41949 examples [00:37, 1130.62 examples/s]42065 examples [00:37, 1137.71 examples/s]42189 examples [00:37, 1164.09 examples/s]42306 examples [00:37, 1128.91 examples/s]42420 examples [00:37, 1090.24 examples/s]42543 examples [00:37, 1127.35 examples/s]42674 examples [00:37, 1176.34 examples/s]42798 examples [00:37, 1192.79 examples/s]42919 examples [00:38, 1187.92 examples/s]43039 examples [00:38, 1140.10 examples/s]43155 examples [00:38, 1143.65 examples/s]43277 examples [00:38, 1164.56 examples/s]43396 examples [00:38, 1169.28 examples/s]43514 examples [00:38, 1155.93 examples/s]43634 examples [00:38, 1166.20 examples/s]43751 examples [00:38, 1156.24 examples/s]43869 examples [00:38, 1160.61 examples/s]43986 examples [00:38, 1148.61 examples/s]44101 examples [00:39, 1110.90 examples/s]44213 examples [00:39, 1095.82 examples/s]44323 examples [00:39, 1066.70 examples/s]44431 examples [00:39, 1050.41 examples/s]44537 examples [00:39, 1043.42 examples/s]44648 examples [00:39, 1061.71 examples/s]44763 examples [00:39, 1085.66 examples/s]44878 examples [00:39, 1097.51 examples/s]45001 examples [00:39, 1132.48 examples/s]45117 examples [00:40, 1139.84 examples/s]45232 examples [00:40, 1136.46 examples/s]45350 examples [00:40, 1146.87 examples/s]45465 examples [00:40, 1135.66 examples/s]45589 examples [00:40, 1162.60 examples/s]45720 examples [00:40, 1201.92 examples/s]45841 examples [00:40, 1200.84 examples/s]45966 examples [00:40, 1215.04 examples/s]46088 examples [00:40, 1206.72 examples/s]46209 examples [00:40, 1151.57 examples/s]46325 examples [00:41, 1129.38 examples/s]46442 examples [00:41, 1140.67 examples/s]46559 examples [00:41, 1148.38 examples/s]46690 examples [00:41, 1190.44 examples/s]46821 examples [00:41, 1223.14 examples/s]46944 examples [00:41, 1219.49 examples/s]47067 examples [00:41, 1144.27 examples/s]47183 examples [00:41, 1140.21 examples/s]47298 examples [00:41, 1139.84 examples/s]47415 examples [00:41, 1147.01 examples/s]47531 examples [00:42, 1150.73 examples/s]47647 examples [00:42, 1143.40 examples/s]47764 examples [00:42, 1150.45 examples/s]47886 examples [00:42, 1169.26 examples/s]48004 examples [00:42, 1150.09 examples/s]48120 examples [00:42, 1121.75 examples/s]48233 examples [00:42, 1105.87 examples/s]48344 examples [00:42, 1090.23 examples/s]48454 examples [00:42, 1057.60 examples/s]48561 examples [00:43, 1058.90 examples/s]48685 examples [00:43, 1105.82 examples/s]48810 examples [00:43, 1143.87 examples/s]48932 examples [00:43, 1164.00 examples/s]49055 examples [00:43, 1181.10 examples/s]49187 examples [00:43, 1218.85 examples/s]49312 examples [00:43, 1226.63 examples/s]49436 examples [00:43, 1225.07 examples/s]49559 examples [00:43, 1205.93 examples/s]49680 examples [00:43, 1199.54 examples/s]49801 examples [00:44, 1178.28 examples/s]49920 examples [00:44, 1161.45 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 8865/50000 [00:00<00:00, 88648.39 examples/s] 50%|█████     | 25129/50000 [00:00<00:00, 102658.89 examples/s] 83%|████████▎ | 41312/50000 [00:00<00:00, 115306.68 examples/s]                                                                0 examples [00:00, ? examples/s]97 examples [00:00, 965.42 examples/s]225 examples [00:00, 1041.72 examples/s]350 examples [00:00, 1094.90 examples/s]473 examples [00:00, 1131.07 examples/s]594 examples [00:00, 1151.36 examples/s]697 examples [00:00, 909.83 examples/s] 820 examples [00:00, 986.76 examples/s]933 examples [00:00, 1024.05 examples/s]1068 examples [00:00, 1103.36 examples/s]1195 examples [00:01, 1139.76 examples/s]1321 examples [00:01, 1171.32 examples/s]1453 examples [00:01, 1210.04 examples/s]1576 examples [00:01, 1166.36 examples/s]1703 examples [00:01, 1193.50 examples/s]1827 examples [00:01, 1206.77 examples/s]1953 examples [00:01, 1221.91 examples/s]2084 examples [00:01, 1244.26 examples/s]2209 examples [00:01, 1220.54 examples/s]2340 examples [00:01, 1244.73 examples/s]2465 examples [00:02, 1218.05 examples/s]2591 examples [00:02, 1228.36 examples/s]2716 examples [00:02, 1232.90 examples/s]2850 examples [00:02, 1262.70 examples/s]2981 examples [00:02, 1275.85 examples/s]3109 examples [00:02, 1210.63 examples/s]3231 examples [00:02, 1175.56 examples/s]3350 examples [00:02, 1153.33 examples/s]3467 examples [00:02, 1119.23 examples/s]3580 examples [00:03, 1089.39 examples/s]3690 examples [00:03, 1087.62 examples/s]3800 examples [00:03, 1061.35 examples/s]3908 examples [00:03, 1066.69 examples/s]4016 examples [00:03, 1069.72 examples/s]4128 examples [00:03, 1083.13 examples/s]4239 examples [00:03, 1090.50 examples/s]4349 examples [00:03, 1090.45 examples/s]4459 examples [00:03, 1059.04 examples/s]4573 examples [00:03, 1082.07 examples/s]4688 examples [00:04, 1100.68 examples/s]4799 examples [00:04, 1092.85 examples/s]4909 examples [00:04, 1086.27 examples/s]5018 examples [00:04, 1078.20 examples/s]5132 examples [00:04, 1095.55 examples/s]5242 examples [00:04, 1093.24 examples/s]5352 examples [00:04, 1057.45 examples/s]5463 examples [00:04, 1070.19 examples/s]5580 examples [00:04, 1096.77 examples/s]5708 examples [00:05, 1145.39 examples/s]5834 examples [00:05, 1174.55 examples/s]5961 examples [00:05, 1199.41 examples/s]6090 examples [00:05, 1222.89 examples/s]6217 examples [00:05, 1234.39 examples/s]6344 examples [00:05, 1243.51 examples/s]6472 examples [00:05, 1252.15 examples/s]6598 examples [00:05, 1214.77 examples/s]6723 examples [00:05, 1224.94 examples/s]6848 examples [00:05, 1232.16 examples/s]6984 examples [00:06, 1267.77 examples/s]7117 examples [00:06, 1282.95 examples/s]7246 examples [00:06, 1251.54 examples/s]7372 examples [00:06, 1197.87 examples/s]7493 examples [00:06, 1166.15 examples/s]7612 examples [00:06, 1172.39 examples/s]7730 examples [00:06, 1171.12 examples/s]7848 examples [00:06, 1146.24 examples/s]7963 examples [00:06, 1146.49 examples/s]8078 examples [00:06, 1127.05 examples/s]8197 examples [00:07, 1144.42 examples/s]8318 examples [00:07, 1160.86 examples/s]8440 examples [00:07, 1174.72 examples/s]8563 examples [00:07, 1188.50 examples/s]8683 examples [00:07, 1142.35 examples/s]8798 examples [00:07, 1060.97 examples/s]8906 examples [00:07, 1058.62 examples/s]9014 examples [00:07, 1063.04 examples/s]9123 examples [00:07, 1070.71 examples/s]9231 examples [00:08, 1046.59 examples/s]9346 examples [00:08, 1074.06 examples/s]9455 examples [00:08, 1076.76 examples/s]9567 examples [00:08, 1088.16 examples/s]9677 examples [00:08, 1088.51 examples/s]9791 examples [00:08, 1102.30 examples/s]9902 examples [00:08, 1082.12 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteD3KHL0/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteD3KHL0/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc798f428c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc798f428c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc798f428c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc804ddf470>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc7824f9588>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc798f428c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc798f428c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc798f428c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc77b037320>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc77b037080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fc771083620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fc771083620> 

  function with postional parmater data_info <function split_train_valid at 0x7fc771083620> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fc771083730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fc771083730> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fc771083730> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=187f4c75e080bd75db3bc6f332f45690b3d3b063c1d4f7f7a87e6f82f064c283
  Stored in directory: /tmp/pip-ephem-wheel-cache-m1ody_g1/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:59:22, 10.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:19:33, 14.7kB/s].vector_cache/glove.6B.zip:   0%|          | 180k/862M [00:01<11:29:02, 20.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 483k/862M [00:01<8:03:34, 29.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.53M/862M [00:01<5:38:31, 42.4kB/s].vector_cache/glove.6B.zip:   0%|          | 3.92M/862M [00:01<3:56:29, 60.5kB/s].vector_cache/glove.6B.zip:   1%|          | 9.13M/862M [00:01<2:44:37, 86.4kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<1:54:56, 123kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:01<1:20:02, 176kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.5M/862M [00:01<55:45, 251kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.2M/862M [00:01<38:51, 357kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:02<27:06, 509kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.8M/862M [00:02<18:54, 724kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.8M/862M [00:02<13:20, 1.02MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.4M/862M [00:02<09:22, 1.45MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<07:16, 1.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<06:59, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<06:49, 1.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:05<05:11, 2.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:05<03:48, 3.52MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<16:51, 793kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<13:12, 1.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<09:34, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<09:46, 1.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<08:11, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:09<06:01, 2.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<07:18, 1.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<06:29, 2.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:11<04:49, 2.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<06:30, 2.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<07:17, 1.80MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:13<05:47, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<06:09, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<05:40, 2.30MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:14<04:16, 3.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<06:01, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<05:34, 2.33MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:16<04:14, 3.07MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<06:00, 2.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<06:54, 1.87MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:18<05:30, 2.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.9M/862M [00:19<04:04, 3.17MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:19<03:12, 4.02MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<22:51, 564kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<22:18, 577kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<17:18, 744kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:20<12:28, 1.03MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:21<08:58, 1.43MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<11:43, 1.09MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<14:37, 876kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<11:50, 1.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:22<08:39, 1.48MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:23<06:19, 2.02MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<12:55, 986kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<15:21, 830kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<12:15, 1.04MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<08:53, 1.43MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<06:30, 1.95MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<06:06, 2.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<7:53:47, 26.8kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<5:32:56, 38.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<3:52:57, 54.3kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<2:43:16, 77.4kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:58:05, 107kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:28:26, 143kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:03:19, 199kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<44:58, 280kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<31:41, 397kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<22:26, 559kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<26:28, 474kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<24:13, 517kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<18:22, 682kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<13:12, 947kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<09:28, 1.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<19:03, 654kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<18:55, 659kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<14:26, 863kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<10:28, 1.19MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<07:34, 1.64MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<17:19, 716kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<17:10, 722kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<13:20, 929kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<09:38, 1.28MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<07:02, 1.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<10:03, 1.23MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<12:04, 1.02MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<09:49, 1.25MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<07:13, 1.70MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<05:17, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<20:14, 606kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<19:17, 636kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<14:50, 826kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<10:43, 1.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<07:41, 1.59MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<17:30, 697kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<17:51, 683kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<13:48, 883kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<09:55, 1.22MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<07:17, 1.67MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:13, 1.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<11:57, 1.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:39, 1.25MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<07:03, 1.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<05:08, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<12:21, 975kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<14:01, 860kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<11:06, 1.09MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<08:06, 1.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<05:54, 2.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<18:29, 648kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<18:16, 656kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<13:53, 863kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<10:03, 1.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<07:16, 1.64MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<19:14, 620kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<18:19, 650kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<14:05, 846kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<10:29, 1.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<07:32, 1.57MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<05:35, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<18:20, 646kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<17:42, 669kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<13:38, 868kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<09:50, 1.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<07:06, 1.66MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<18:57, 622kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<18:03, 652kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<13:53, 847kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<10:00, 1.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<07:10, 1.63MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<15:32, 754kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<15:38, 749kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<12:12, 959kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<08:52, 1.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<06:23, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<19:40, 591kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<19:01, 612kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<14:32, 800kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<10:27, 1.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<07:29, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<16:31, 700kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<16:48, 689kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<12:59, 890kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<09:25, 1.23MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<06:48, 1.69MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<20:40, 556kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<19:10, 600kB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:00<14:24, 797kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<10:23, 1.10MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<07:35, 1.51MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<09:35, 1.19MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<11:26, 1.00MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<09:09, 1.25MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:41, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<04:53, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<43:50, 259kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<35:21, 321kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<25:51, 439kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<18:20, 618kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<13:00, 868kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<59:24, 190kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<46:11, 245kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<33:25, 338kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<23:36, 477kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<16:41, 673kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<20:28, 548kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<18:36, 603kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<14:05, 796kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<10:09, 1.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<07:17, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<9:53:21, 18.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<6:59:30, 26.6kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<4:54:31, 37.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<3:26:01, 54.0kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<2:26:11, 75.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<1:46:29, 104kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<1:15:24, 147kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<52:57, 209kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<39:25, 279kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<31:27, 350kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<22:47, 483kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<16:12, 678kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<11:33, 949kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<13:17, 824kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<12:55, 847kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<09:48, 1.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<07:03, 1.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<05:11, 2.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<10:34, 1.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<10:59, 990kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<08:26, 1.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<06:07, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<04:31, 2.40MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<09:58, 1.08MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<10:18, 1.05MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<08:01, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<05:48, 1.86MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<07:02, 1.53MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:01, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:16, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<04:33, 2.35MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:25, 3.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<13:44, 777kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<13:22, 798kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<10:16, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<07:25, 1.43MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<07:54, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<08:25, 1.26MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:31, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<04:47, 2.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:52, 1.53MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:31, 1.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:49, 1.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:15, 2.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:01, 1.74MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:39, 1.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:17, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<03:51, 2.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<07:10, 1.45MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<07:26, 1.40MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:43, 1.82MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:08, 2.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:48, 1.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<07:25, 1.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:52, 1.76MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:34<04:15, 2.41MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<07:05, 1.45MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<07:08, 1.44MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<05:27, 1.88MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:56, 2.59MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<08:33, 1.19MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<08:04, 1.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<06:09, 1.65MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<04:26, 2.28MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<12:43, 796kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<10:52, 930kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<08:06, 1.25MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<05:46, 1.74MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<15:13, 660kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<12:37, 796kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<09:14, 1.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<06:36, 1.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<08:47, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<08:46, 1.14MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<06:44, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<04:51, 2.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<09:02, 1.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<08:00, 1.24MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<06:01, 1.64MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:38, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<7:23:03, 22.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<5:11:07, 31.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<3:37:31, 45.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<2:31:56, 64.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<1:54:26, 85.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<1:21:40, 120kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<57:24, 170kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<41:43, 233kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<31:17, 310kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<22:22, 433kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<15:42, 614kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<36:02, 268kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<27:16, 353kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<19:34, 492kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<13:44, 697kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<44:15, 216kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<32:23, 295kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<22:55, 416kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<16:10, 589kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<15:42, 605kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<12:22, 768kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<08:58, 1.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<08:01, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<07:25, 1.27MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:34, 1.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<04:02, 2.32MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:42, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:29, 1.44MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:58, 1.88MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:00, 1.85MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:18, 1.75MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:08, 2.24MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:25, 2.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:52, 1.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<03:50, 2.40MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:11, 2.18MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:41, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<03:38, 2.51MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<02:40, 3.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:56, 1.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:53, 1.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:33, 1.99MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:40, 1.93MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:59, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<03:55, 2.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:13, 2.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<04:19, 2.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<03:56, 2.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<02:54, 3.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:16, 2.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<04:42, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<03:42, 2.39MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:02, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<04:31, 1.95MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<03:32, 2.49MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<02:34, 3.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<08:46, 998kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<07:49, 1.12MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:48, 1.50MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<04:10, 2.09MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<08:18, 1.04MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<07:26, 1.17MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:33, 1.56MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<03:58, 2.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<09:51, 874kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<08:35, 1.00MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:24, 1.34MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:52, 1.45MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:14, 1.63MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<03:56, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:24, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<06:02, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<04:59, 1.70MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:38, 2.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:45, 1.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:48, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:40, 2.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<02:39, 3.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<10:41, 780kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<08:56, 932kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<06:36, 1.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<06:00, 1.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<05:40, 1.46MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:16, 1.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:06, 2.64MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:37, 1.46MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:24, 1.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<04:08, 1.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<04:15, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<04:27, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:28, 2.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:47, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:04, 1.98MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<03:11, 2.51MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<03:35, 2.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<05:41, 1.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<05:23, 1.48MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<04:06, 1.94MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<03:01, 2.62MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:40, 2.97MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<5:28:54, 24.1kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<3:50:44, 34.3kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<2:41:03, 49.0kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<1:54:24, 68.6kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<1:23:06, 94.5kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<58:45, 134kB/s]   .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<41:09, 190kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<30:40, 254kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<22:51, 340kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<16:18, 476kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<11:33, 670kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<10:06, 763kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<08:27, 912kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<06:13, 1.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<04:24, 1.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<21:06, 362kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<16:07, 474kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<11:36, 657kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<09:21, 810kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<07:56, 953kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:52, 1.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:21, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:06, 1.47MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:53, 1.92MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:58, 1.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:09, 1.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:14, 2.29MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:20, 3.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<10:14, 719kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<08:28, 869kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<06:12, 1.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<04:24, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<13:08, 556kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<10:29, 696kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<07:37, 956kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<06:32, 1.11MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<05:38, 1.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:19, 1.67MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:13, 1.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:12, 1.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:12, 2.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:19, 3.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<08:38, 822kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<08:44, 812kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:39, 1.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:49, 1.46MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<04:46, 1.47MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:34, 1.53MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<03:27, 2.02MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<02:30, 2.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<06:23, 1.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:43, 1.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:18, 1.61MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:10, 1.65MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:07, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:08, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:15, 3.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<07:37, 894kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<06:34, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:50, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<03:27, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<08:46, 770kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<07:19, 922kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:21, 1.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:49, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<06:45, 988kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<05:54, 1.13MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:23, 1.52MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<03:07, 2.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<37:45, 175kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<27:34, 240kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<19:33, 337kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<14:41, 445kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<11:25, 573kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<08:15, 790kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<06:51, 945kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<05:58, 1.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:25, 1.46MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<03:09, 2.04MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<07:43, 830kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<06:31, 982kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:50, 1.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:26, 1.43MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:13, 1.50MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:13, 1.96MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:18, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:24, 1.84MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:36, 2.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<01:54, 3.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<05:06, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:39, 1.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<03:31, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:29, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:33, 1.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:42, 2.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<01:58, 3.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<05:12, 1.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<04:42, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:31, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<02:31, 2.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<08:22, 716kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<07:47, 769kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<06:16, 956kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<04:39, 1.29MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<03:21, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<04:09, 1.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:56, 1.50MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<03:01, 1.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:21, 2.49MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<4:25:55, 22.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<3:06:26, 31.4kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<2:09:57, 44.8kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<1:32:03, 62.9kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<1:06:36, 86.9kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<47:05, 123kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<32:54, 175kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<24:25, 234kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<18:04, 316kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<12:50, 444kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<09:01, 628kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<08:55, 634kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<07:13, 782kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<05:15, 1.07MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<03:43, 1.50MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<05:55, 942kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<05:07, 1.09MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:49, 1.46MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:35, 1.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:31, 1.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:41, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:48, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:55, 1.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:16, 2.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:29, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:43, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:23, 2.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<01:46, 3.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<01:27, 3.65MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<01:13, 4.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<05:36, 947kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<06:05, 871kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:48, 1.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:32, 1.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:39, 1.98MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<02:02, 2.57MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<07:59, 655kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<07:34, 692kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<05:46, 906kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:13, 1.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<03:06, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<02:20, 2.21MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<13:37, 380kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<11:29, 450kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<08:30, 607kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<06:05, 845kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<04:27, 1.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<03:15, 1.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<05:25, 942kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<05:34, 914kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<04:18, 1.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<03:11, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:23, 2.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<01:50, 2.74MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<2:02:01, 41.3kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<1:27:08, 57.7kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<1:01:20, 81.9kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<42:56, 117kB/s]   .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<30:05, 166kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<22:47, 218kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<19:25, 256kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<14:29, 342kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<10:21, 478kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<07:21, 669kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<05:17, 928kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<07:10, 682kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<06:44, 726kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<05:03, 965kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:44, 1.30MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<02:43, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:42, 1.30MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<04:18, 1.12MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:21, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:28, 1.93MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:56, 2.46MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:28, 3.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<04:20, 1.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<06:22, 747kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<05:19, 893kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:54, 1.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:52, 1.64MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<02:09, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<04:59, 940kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<05:07, 913kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:59, 1.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:55, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:11, 2.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:42, 2.71MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<04:39, 991kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<06:32, 707kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<05:24, 854kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:57, 1.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:54, 1.57MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<02:10, 2.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<04:45, 956kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<04:54, 925kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:50, 1.18MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:48, 1.62MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:03, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<01:41, 2.66MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<04:45, 943kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<06:29, 691kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<05:17, 847kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:51, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:50, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<02:07, 2.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<05:03, 873kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<05:06, 864kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<03:56, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:52, 1.52MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:08, 2.04MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:15, 1.33MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<04:55, 881kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<04:05, 1.06MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:03, 1.41MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 604M/862M [04:25<02:16, 1.89MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:42, 2.51MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<01:19, 3.22MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<16:29, 259kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<12:55, 331kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<09:22, 455kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<06:40, 636kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<04:46, 884kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<05:07, 820kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<06:10, 682kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<04:58, 844kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:38, 1.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:39, 1.57MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:58, 2.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<10:20, 400kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<08:30, 486kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<06:15, 660kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<04:30, 912kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<03:14, 1.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<04:14, 957kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<05:27, 745kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<04:19, 940kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:10, 1.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:18, 1.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:44, 2.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<04:20, 920kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<04:07, 968kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:09, 1.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:17, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:41, 2.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<07:39, 513kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<07:31, 522kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<05:45, 681kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<04:10, 937kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:58, 1.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:40, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<04:28, 863kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:36, 1.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:37, 1.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:39<01:53, 2.01MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:08, 1.21MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:53, 976kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:08, 1.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:17, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:39, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<04:25, 842kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<04:45, 781kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:43, 999kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:41, 1.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:56, 1.89MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<05:41, 642kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<05:28, 668kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<04:07, 883kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:58, 1.22MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<02:08, 1.67MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<12:02, 298kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<09:47, 366kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<07:07, 502kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<05:03, 703kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<04:16, 822kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<04:08, 849kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<03:11, 1.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<02:17, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:25, 1.43MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:49, 1.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:14, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<01:38, 2.08MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:59, 1.69MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:22, 1.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:52, 1.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<01:22, 2.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:01, 3.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<04:16, 774kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:56, 839kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:57, 1.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<02:06, 1.54MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:44, 1.86MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<2:17:49, 23.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<1:36:46, 33.4kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<1:07:25, 47.7kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<46:43, 68.1kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<35:38, 89.0kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<25:47, 123kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<18:13, 173kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<12:40, 247kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<09:52, 314kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<07:45, 400kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<05:35, 553kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<03:55, 778kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<03:48, 796kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<03:32, 856kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:41, 1.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:55, 1.55MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:16, 1.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:26, 1.21MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:52, 1.57MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:20, 2.16MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:52, 1.55MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:08, 1.35MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:41, 1.70MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:13, 2.33MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:46, 1.59MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:56, 1.46MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:30, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:05, 2.53MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:47, 1.54MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:59, 1.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:33, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:07, 2.41MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:36, 1.67MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:48, 1.48MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:24, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:02, 2.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:23, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:38, 1.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:16, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:55, 2.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:33, 1.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:31, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:10, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:13, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:58, 1.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:40, 1.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:13, 1.99MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:19, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:28, 1.64MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:08, 2.10MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:48, 2.90MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:53, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:40, 1.40MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:13, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<00:52, 2.61MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<03:29, 650kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:53, 786kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<02:07, 1.06MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:48, 1.22MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:42, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:17, 1.69MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:13, 1.73MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:15, 1.69MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:58, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:00, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:06, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:52, 2.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:55, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:02, 1.92MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:48, 2.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:34, 3.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:09, 895kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:53, 1.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:23, 1.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:58, 1.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:48, 1.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:37, 1.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:13, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<01:07, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:08, 1.57MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:52, 2.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:52, 1.96MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:57, 1.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:45, 2.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:46, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:52, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:40, 2.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:28, 3.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:20, 1.18MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:14, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:56, 1.66MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:53, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:54, 1.67MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:42, 2.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:42, 2.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:46, 1.85MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:36, 2.34MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:38, 2.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:42, 1.92MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<00:33, 2.46MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:23, 3.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:01, 1.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:58, 1.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:44, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:41, 1.78MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:43, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:33, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:34, 2.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:37, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:29, 2.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:30, 2.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:34, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:26, 2.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:28, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:31, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:24, 2.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:26, 2.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:29, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:23, 2.45MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:24, 2.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:27, 1.98MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:21, 2.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:22, 2.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:25, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:19, 2.51MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:13, 3.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:31, 1.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:31, 1.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:23, 1.92MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:15, 2.65MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:47, 882kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:40, 1.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:30, 1.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:25, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:25, 1.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:18, 1.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:13, 2.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:19, 1.72MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:19, 1.66MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:15, 2.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:14, 2.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:15, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:11, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:12, 1.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:09, 2.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:09, 2.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:10, 1.94MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:08, 2.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:07, 2.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.98MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:06, 2.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:05, 2.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:04, 2.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:02, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:02, 1.96MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.51MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 3.44MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 829kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 961kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<193:42:16,  1.74s/it]  0%|          | 878/400000 [00:01<135:17:58,  1.22s/it]  0%|          | 1740/400000 [00:01<94:30:32,  1.17it/s]  1%|          | 2603/400000 [00:02<66:01:00,  1.67it/s]  1%|          | 3470/400000 [00:02<46:06:53,  2.39it/s]  1%|          | 4341/400000 [00:02<32:12:47,  3.41it/s]  1%|▏         | 5223/400000 [00:02<22:30:09,  4.87it/s]  2%|▏         | 6084/400000 [00:02<15:43:16,  6.96it/s]  2%|▏         | 6983/400000 [00:02<10:59:00,  9.94it/s]  2%|▏         | 7929/400000 [00:02<7:40:24, 14.19it/s]   2%|▏         | 8835/400000 [00:02<5:21:45, 20.26it/s]  2%|▏         | 9713/400000 [00:02<3:44:56, 28.92it/s]  3%|▎         | 10598/400000 [00:02<2:37:19, 41.25it/s]  3%|▎         | 11478/400000 [00:03<1:50:05, 58.81it/s]  3%|▎         | 12357/400000 [00:03<1:17:07, 83.78it/s]  3%|▎         | 13250/400000 [00:03<54:04, 119.20it/s]   4%|▎         | 14203/400000 [00:03<37:57, 169.38it/s]  4%|▍         | 15106/400000 [00:03<26:43, 239.98it/s]  4%|▍         | 16000/400000 [00:03<18:53, 338.89it/s]  4%|▍         | 16908/400000 [00:03<13:23, 476.50it/s]  4%|▍         | 17833/400000 [00:03<09:33, 665.99it/s]  5%|▍         | 18755/400000 [00:03<06:53, 922.83it/s]  5%|▍         | 19665/400000 [00:03<05:01, 1261.96it/s]  5%|▌         | 20568/400000 [00:04<03:43, 1697.99it/s]  5%|▌         | 21537/400000 [00:04<02:47, 2256.23it/s]  6%|▌         | 22493/400000 [00:04<02:08, 2927.06it/s]  6%|▌         | 23428/400000 [00:04<01:42, 3686.76it/s]  6%|▌         | 24359/400000 [00:04<01:24, 4466.96it/s]  6%|▋         | 25275/400000 [00:04<01:11, 5248.66it/s]  7%|▋         | 26187/400000 [00:04<01:02, 6013.74it/s]  7%|▋         | 27107/400000 [00:04<00:55, 6710.19it/s]  7%|▋         | 28030/400000 [00:04<00:50, 7308.30it/s]  7%|▋         | 28951/400000 [00:04<00:47, 7789.76it/s]  7%|▋         | 29900/400000 [00:05<00:44, 8231.77it/s]  8%|▊         | 30872/400000 [00:05<00:42, 8626.35it/s]  8%|▊         | 31850/400000 [00:05<00:41, 8941.71it/s]  8%|▊         | 32841/400000 [00:05<00:39, 9209.87it/s]  8%|▊         | 33804/400000 [00:05<00:39, 9190.86it/s]  9%|▊         | 34792/400000 [00:05<00:38, 9387.28it/s]  9%|▉         | 35770/400000 [00:05<00:38, 9501.08it/s]  9%|▉         | 36792/400000 [00:05<00:37, 9703.95it/s]  9%|▉         | 37775/400000 [00:05<00:37, 9726.69it/s] 10%|▉         | 38756/400000 [00:05<00:38, 9347.34it/s] 10%|▉         | 39732/400000 [00:06<00:38, 9466.09it/s] 10%|█         | 40686/400000 [00:06<00:38, 9379.97it/s] 10%|█         | 41629/400000 [00:06<00:38, 9264.91it/s] 11%|█         | 42560/400000 [00:06<00:38, 9268.73it/s] 11%|█         | 43490/400000 [00:06<00:39, 9067.39it/s] 11%|█         | 44400/400000 [00:06<00:39, 9005.73it/s] 11%|█▏        | 45307/400000 [00:06<00:39, 9022.19it/s] 12%|█▏        | 46211/400000 [00:06<00:40, 8811.20it/s] 12%|█▏        | 47181/400000 [00:06<00:38, 9059.77it/s] 12%|█▏        | 48098/400000 [00:07<00:38, 9091.08it/s] 12%|█▏        | 49074/400000 [00:07<00:37, 9279.28it/s] 13%|█▎        | 50019/400000 [00:07<00:37, 9327.71it/s] 13%|█▎        | 50954/400000 [00:07<00:38, 9161.31it/s] 13%|█▎        | 51873/400000 [00:07<00:39, 8816.17it/s] 13%|█▎        | 52759/400000 [00:07<00:39, 8809.12it/s] 13%|█▎        | 53667/400000 [00:07<00:38, 8886.97it/s] 14%|█▎        | 54582/400000 [00:07<00:38, 8963.09it/s] 14%|█▍        | 55506/400000 [00:07<00:38, 9042.57it/s] 14%|█▍        | 56412/400000 [00:07<00:38, 8990.38it/s] 14%|█▍        | 57427/400000 [00:08<00:36, 9309.44it/s] 15%|█▍        | 58386/400000 [00:08<00:36, 9389.96it/s] 15%|█▍        | 59450/400000 [00:08<00:34, 9730.41it/s] 15%|█▌        | 60429/400000 [00:08<00:35, 9574.08it/s] 15%|█▌        | 61391/400000 [00:08<00:36, 9332.42it/s] 16%|█▌        | 62366/400000 [00:08<00:35, 9452.51it/s] 16%|█▌        | 63315/400000 [00:08<00:35, 9390.68it/s] 16%|█▌        | 64296/400000 [00:08<00:35, 9510.19it/s] 16%|█▋        | 65250/400000 [00:08<00:35, 9494.69it/s] 17%|█▋        | 66201/400000 [00:08<00:36, 9194.64it/s] 17%|█▋        | 67124/400000 [00:09<00:37, 8915.82it/s] 17%|█▋        | 68024/400000 [00:09<00:37, 8938.28it/s] 17%|█▋        | 68921/400000 [00:09<00:37, 8918.35it/s] 17%|█▋        | 69815/400000 [00:09<00:37, 8828.09it/s] 18%|█▊        | 70700/400000 [00:09<00:37, 8743.81it/s] 18%|█▊        | 71641/400000 [00:09<00:36, 8932.39it/s] 18%|█▊        | 72537/400000 [00:09<00:36, 8933.95it/s] 18%|█▊        | 73460/400000 [00:09<00:36, 9019.76it/s] 19%|█▊        | 74364/400000 [00:09<00:36, 9016.72it/s] 19%|█▉        | 75267/400000 [00:09<00:36, 8858.37it/s] 19%|█▉        | 76155/400000 [00:10<00:36, 8801.34it/s] 19%|█▉        | 77037/400000 [00:10<00:36, 8766.65it/s] 19%|█▉        | 77923/400000 [00:10<00:36, 8791.12it/s] 20%|█▉        | 78808/400000 [00:10<00:36, 8807.93it/s] 20%|█▉        | 79690/400000 [00:10<00:36, 8766.06it/s] 20%|██        | 80567/400000 [00:10<00:36, 8766.92it/s] 20%|██        | 81444/400000 [00:10<00:36, 8616.78it/s] 21%|██        | 82332/400000 [00:10<00:36, 8691.57it/s] 21%|██        | 83204/400000 [00:10<00:36, 8700.01it/s] 21%|██        | 84102/400000 [00:11<00:35, 8780.29it/s] 21%|██        | 84981/400000 [00:11<00:35, 8776.54it/s] 21%|██▏       | 85860/400000 [00:11<00:35, 8740.84it/s] 22%|██▏       | 86741/400000 [00:11<00:35, 8760.94it/s] 22%|██▏       | 87642/400000 [00:11<00:35, 8833.17it/s] 22%|██▏       | 88526/400000 [00:11<00:35, 8799.22it/s] 22%|██▏       | 89420/400000 [00:11<00:35, 8839.86it/s] 23%|██▎       | 90305/400000 [00:11<00:35, 8798.20it/s] 23%|██▎       | 91190/400000 [00:11<00:35, 8812.17it/s] 23%|██▎       | 92072/400000 [00:11<00:34, 8803.20it/s] 23%|██▎       | 92953/400000 [00:12<00:35, 8590.61it/s] 23%|██▎       | 93816/400000 [00:12<00:35, 8602.25it/s] 24%|██▎       | 94678/400000 [00:12<00:36, 8450.84it/s] 24%|██▍       | 95525/400000 [00:12<00:36, 8433.50it/s] 24%|██▍       | 96370/400000 [00:12<00:36, 8396.22it/s] 24%|██▍       | 97250/400000 [00:12<00:35, 8510.93it/s] 25%|██▍       | 98161/400000 [00:12<00:34, 8681.52it/s] 25%|██▍       | 99146/400000 [00:12<00:33, 9001.04it/s] 25%|██▌       | 100203/400000 [00:12<00:31, 9419.15it/s] 25%|██▌       | 101153/400000 [00:12<00:31, 9428.50it/s] 26%|██▌       | 102102/400000 [00:13<00:31, 9386.95it/s] 26%|██▌       | 103045/400000 [00:13<00:32, 9147.28it/s] 26%|██▌       | 103977/400000 [00:13<00:32, 9197.02it/s] 26%|██▌       | 104970/400000 [00:13<00:31, 9404.54it/s] 26%|██▋       | 105914/400000 [00:13<00:31, 9373.40it/s] 27%|██▋       | 106854/400000 [00:13<00:31, 9277.57it/s] 27%|██▋       | 107807/400000 [00:13<00:31, 9349.18it/s] 27%|██▋       | 108744/400000 [00:13<00:31, 9284.89it/s] 27%|██▋       | 109693/400000 [00:13<00:31, 9343.05it/s] 28%|██▊       | 110681/400000 [00:13<00:30, 9495.39it/s] 28%|██▊       | 111632/400000 [00:14<00:31, 9176.82it/s] 28%|██▊       | 112629/400000 [00:14<00:30, 9400.32it/s] 28%|██▊       | 113639/400000 [00:14<00:29, 9598.08it/s] 29%|██▊       | 114624/400000 [00:14<00:29, 9671.67it/s] 29%|██▉       | 115652/400000 [00:14<00:28, 9845.41it/s] 29%|██▉       | 116640/400000 [00:14<00:28, 9832.94it/s] 29%|██▉       | 117626/400000 [00:14<00:28, 9774.61it/s] 30%|██▉       | 118605/400000 [00:14<00:29, 9488.26it/s] 30%|██▉       | 119557/400000 [00:14<00:29, 9357.57it/s] 30%|███       | 120496/400000 [00:14<00:29, 9347.85it/s] 30%|███       | 121433/400000 [00:15<00:30, 9273.65it/s] 31%|███       | 122362/400000 [00:15<00:30, 9213.74it/s] 31%|███       | 123286/400000 [00:15<00:30, 9220.79it/s] 31%|███       | 124209/400000 [00:15<00:30, 9133.34it/s] 31%|███▏      | 125124/400000 [00:15<00:30, 9109.37it/s] 32%|███▏      | 126036/400000 [00:15<00:31, 8815.03it/s] 32%|███▏      | 126920/400000 [00:15<00:31, 8762.04it/s] 32%|███▏      | 127798/400000 [00:15<00:31, 8638.67it/s] 32%|███▏      | 128664/400000 [00:15<00:31, 8545.32it/s] 32%|███▏      | 129566/400000 [00:16<00:31, 8680.63it/s] 33%|███▎      | 130493/400000 [00:16<00:30, 8849.03it/s] 33%|███▎      | 131508/400000 [00:16<00:29, 9201.74it/s] 33%|███▎      | 132497/400000 [00:16<00:28, 9395.83it/s] 33%|███▎      | 133442/400000 [00:16<00:28, 9332.23it/s] 34%|███▎      | 134379/400000 [00:16<00:28, 9230.28it/s] 34%|███▍      | 135309/400000 [00:16<00:28, 9250.35it/s] 34%|███▍      | 136326/400000 [00:16<00:27, 9506.84it/s] 34%|███▍      | 137344/400000 [00:16<00:27, 9697.17it/s] 35%|███▍      | 138364/400000 [00:16<00:26, 9842.73it/s] 35%|███▍      | 139377/400000 [00:17<00:26, 9925.11it/s] 35%|███▌      | 140372/400000 [00:17<00:26, 9855.70it/s] 35%|███▌      | 141360/400000 [00:17<00:26, 9677.97it/s] 36%|███▌      | 142330/400000 [00:17<00:27, 9442.78it/s] 36%|███▌      | 143281/400000 [00:17<00:27, 9461.39it/s] 36%|███▌      | 144329/400000 [00:17<00:26, 9743.32it/s] 36%|███▋      | 145307/400000 [00:17<00:26, 9637.97it/s] 37%|███▋      | 146347/400000 [00:17<00:25, 9853.92it/s] 37%|███▋      | 147361/400000 [00:17<00:25, 9937.48it/s] 37%|███▋      | 148417/400000 [00:17<00:24, 10115.17it/s] 37%|███▋      | 149472/400000 [00:18<00:24, 10240.91it/s] 38%|███▊      | 150499/400000 [00:18<00:24, 9985.04it/s]  38%|███▊      | 151501/400000 [00:18<00:25, 9818.05it/s] 38%|███▊      | 152486/400000 [00:18<00:25, 9816.58it/s] 38%|███▊      | 153488/400000 [00:18<00:24, 9874.59it/s] 39%|███▊      | 154500/400000 [00:18<00:24, 9945.93it/s] 39%|███▉      | 155496/400000 [00:18<00:24, 9783.84it/s] 39%|███▉      | 156523/400000 [00:18<00:24, 9922.39it/s] 39%|███▉      | 157517/400000 [00:18<00:24, 9811.89it/s] 40%|███▉      | 158500/400000 [00:18<00:25, 9511.17it/s] 40%|███▉      | 159455/400000 [00:19<00:26, 9075.15it/s] 40%|████      | 160418/400000 [00:19<00:25, 9233.02it/s] 40%|████      | 161386/400000 [00:19<00:25, 9362.40it/s] 41%|████      | 162327/400000 [00:19<00:25, 9299.04it/s] 41%|████      | 163283/400000 [00:19<00:25, 9370.49it/s] 41%|████      | 164223/400000 [00:19<00:25, 9353.58it/s] 41%|████▏     | 165160/400000 [00:19<00:25, 9343.70it/s] 42%|████▏     | 166127/400000 [00:19<00:24, 9436.75it/s] 42%|████▏     | 167114/400000 [00:19<00:24, 9560.85it/s] 42%|████▏     | 168072/400000 [00:19<00:24, 9552.06it/s] 42%|████▏     | 169074/400000 [00:20<00:23, 9685.32it/s] 43%|████▎     | 170044/400000 [00:20<00:23, 9653.41it/s] 43%|████▎     | 171057/400000 [00:20<00:23, 9790.33it/s] 43%|████▎     | 172037/400000 [00:20<00:23, 9551.62it/s] 43%|████▎     | 172995/400000 [00:20<00:24, 9254.20it/s] 44%|████▎     | 174017/400000 [00:20<00:23, 9521.95it/s] 44%|████▎     | 174974/400000 [00:20<00:23, 9380.66it/s] 44%|████▍     | 175968/400000 [00:20<00:23, 9541.18it/s] 44%|████▍     | 176959/400000 [00:20<00:23, 9648.75it/s] 44%|████▍     | 177991/400000 [00:21<00:22, 9840.24it/s] 45%|████▍     | 179061/400000 [00:21<00:21, 10080.27it/s] 45%|████▌     | 180073/400000 [00:21<00:22, 9914.46it/s]  45%|████▌     | 181148/400000 [00:21<00:21, 10150.43it/s] 46%|████▌     | 182215/400000 [00:21<00:21, 10298.90it/s] 46%|████▌     | 183295/400000 [00:21<00:20, 10443.78it/s] 46%|████▌     | 184342/400000 [00:21<00:20, 10408.77it/s] 46%|████▋     | 185385/400000 [00:21<00:20, 10252.01it/s] 47%|████▋     | 186421/400000 [00:21<00:20, 10282.29it/s] 47%|████▋     | 187451/400000 [00:21<00:20, 10267.35it/s] 47%|████▋     | 188505/400000 [00:22<00:20, 10347.48it/s] 47%|████▋     | 189541/400000 [00:22<00:20, 10240.22it/s] 48%|████▊     | 190566/400000 [00:22<00:21, 9801.88it/s]  48%|████▊     | 191551/400000 [00:22<00:21, 9765.15it/s] 48%|████▊     | 192531/400000 [00:22<00:21, 9602.82it/s] 48%|████▊     | 193550/400000 [00:22<00:21, 9771.64it/s] 49%|████▊     | 194669/400000 [00:22<00:20, 10156.77it/s] 49%|████▉     | 195691/400000 [00:22<00:20, 10029.46it/s] 49%|████▉     | 196758/400000 [00:22<00:19, 10210.28it/s] 49%|████▉     | 197783/400000 [00:22<00:20, 9704.00it/s]  50%|████▉     | 198766/400000 [00:23<00:20, 9740.42it/s] 50%|████▉     | 199824/400000 [00:23<00:20, 9977.66it/s] 50%|█████     | 200862/400000 [00:23<00:19, 10092.80it/s] 50%|█████     | 201918/400000 [00:23<00:19, 10226.26it/s] 51%|█████     | 202990/400000 [00:23<00:18, 10368.96it/s] 51%|█████     | 204053/400000 [00:23<00:18, 10444.43it/s] 51%|█████▏    | 205136/400000 [00:23<00:18, 10557.17it/s] 52%|█████▏    | 206194/400000 [00:23<00:18, 10417.69it/s] 52%|█████▏    | 207254/400000 [00:23<00:18, 10469.89it/s] 52%|█████▏    | 208303/400000 [00:23<00:18, 10299.19it/s] 52%|█████▏    | 209335/400000 [00:24<00:18, 10108.00it/s] 53%|█████▎    | 210348/400000 [00:24<00:19, 9828.51it/s]  53%|█████▎    | 211334/400000 [00:24<00:19, 9829.77it/s] 53%|█████▎    | 212379/400000 [00:24<00:18, 10007.00it/s] 53%|█████▎    | 213455/400000 [00:24<00:18, 10220.48it/s] 54%|█████▎    | 214480/400000 [00:24<00:18, 10110.14it/s] 54%|█████▍    | 215494/400000 [00:24<00:18, 9975.92it/s]  54%|█████▍    | 216494/400000 [00:24<00:18, 9940.80it/s] 54%|█████▍    | 217490/400000 [00:24<00:19, 9464.81it/s] 55%|█████▍    | 218492/400000 [00:25<00:18, 9622.23it/s] 55%|█████▍    | 219498/400000 [00:25<00:18, 9747.35it/s] 55%|█████▌    | 220477/400000 [00:25<00:18, 9733.31it/s] 55%|█████▌    | 221504/400000 [00:25<00:18, 9887.61it/s] 56%|█████▌    | 222581/400000 [00:25<00:17, 10135.57it/s] 56%|█████▌    | 223635/400000 [00:25<00:17, 10252.28it/s] 56%|█████▌    | 224663/400000 [00:25<00:17, 10221.98it/s] 56%|█████▋    | 225688/400000 [00:25<00:17, 9820.15it/s]  57%|█████▋    | 226675/400000 [00:25<00:18, 9547.95it/s] 57%|█████▋    | 227635/400000 [00:25<00:18, 9271.94it/s] 57%|█████▋    | 228568/400000 [00:26<00:18, 9182.66it/s] 57%|█████▋    | 229492/400000 [00:26<00:18, 9198.85it/s] 58%|█████▊    | 230415/400000 [00:26<00:18, 9015.73it/s] 58%|█████▊    | 231320/400000 [00:26<00:19, 8657.34it/s] 58%|█████▊    | 232210/400000 [00:26<00:19, 8726.54it/s] 58%|█████▊    | 233136/400000 [00:26<00:18, 8879.61it/s] 59%|█████▊    | 234053/400000 [00:26<00:18, 8962.51it/s] 59%|█████▊    | 234956/400000 [00:26<00:18, 8980.18it/s] 59%|█████▉    | 235856/400000 [00:26<00:18, 8905.87it/s] 59%|█████▉    | 236800/400000 [00:27<00:18, 9057.11it/s] 59%|█████▉    | 237708/400000 [00:27<00:17, 9050.58it/s] 60%|█████▉    | 238615/400000 [00:27<00:17, 9019.13it/s] 60%|█████▉    | 239518/400000 [00:27<00:17, 8929.61it/s] 60%|██████    | 240412/400000 [00:27<00:18, 8831.70it/s] 60%|██████    | 241329/400000 [00:27<00:17, 8929.57it/s] 61%|██████    | 242244/400000 [00:27<00:17, 8992.46it/s] 61%|██████    | 243193/400000 [00:27<00:17, 9133.63it/s] 61%|██████    | 244108/400000 [00:27<00:17, 9033.36it/s] 61%|██████▏   | 245013/400000 [00:27<00:17, 9003.56it/s] 61%|██████▏   | 245915/400000 [00:28<00:17, 8978.79it/s] 62%|██████▏   | 246814/400000 [00:28<00:17, 8942.74it/s] 62%|██████▏   | 247737/400000 [00:28<00:16, 9026.21it/s] 62%|██████▏   | 248651/400000 [00:28<00:16, 9059.93it/s] 62%|██████▏   | 249619/400000 [00:28<00:16, 9236.78it/s] 63%|██████▎   | 250575/400000 [00:28<00:16, 9330.75it/s] 63%|██████▎   | 251510/400000 [00:28<00:16, 9229.01it/s] 63%|██████▎   | 252500/400000 [00:28<00:15, 9417.59it/s] 63%|██████▎   | 253444/400000 [00:28<00:15, 9332.10it/s] 64%|██████▎   | 254379/400000 [00:28<00:15, 9255.48it/s] 64%|██████▍   | 255306/400000 [00:29<00:15, 9126.79it/s] 64%|██████▍   | 256220/400000 [00:29<00:15, 8997.04it/s] 64%|██████▍   | 257121/400000 [00:29<00:16, 8849.18it/s] 65%|██████▍   | 258041/400000 [00:29<00:15, 8951.13it/s] 65%|██████▍   | 258949/400000 [00:29<00:15, 8987.99it/s] 65%|██████▍   | 259856/400000 [00:29<00:15, 9010.11it/s] 65%|██████▌   | 260758/400000 [00:29<00:15, 8932.82it/s] 65%|██████▌   | 261652/400000 [00:29<00:15, 8786.85it/s] 66%|██████▌   | 262532/400000 [00:29<00:15, 8753.28it/s] 66%|██████▌   | 263409/400000 [00:29<00:15, 8682.00it/s] 66%|██████▌   | 264281/400000 [00:30<00:15, 8690.91it/s] 66%|██████▋   | 265151/400000 [00:30<00:15, 8680.36it/s] 67%|██████▋   | 266020/400000 [00:30<00:15, 8607.74it/s] 67%|██████▋   | 266882/400000 [00:30<00:15, 8587.12it/s] 67%|██████▋   | 267741/400000 [00:30<00:15, 8569.79it/s] 67%|██████▋   | 268615/400000 [00:30<00:15, 8619.50it/s] 67%|██████▋   | 269478/400000 [00:30<00:15, 8603.01it/s] 68%|██████▊   | 270339/400000 [00:30<00:15, 8463.85it/s] 68%|██████▊   | 271186/400000 [00:30<00:15, 8408.37it/s] 68%|██████▊   | 272036/400000 [00:30<00:15, 8435.27it/s] 68%|██████▊   | 272880/400000 [00:31<00:15, 8285.45it/s] 68%|██████▊   | 273743/400000 [00:31<00:15, 8385.67it/s] 69%|██████▊   | 274583/400000 [00:31<00:15, 8349.51it/s] 69%|██████▉   | 275424/400000 [00:31<00:14, 8365.95it/s] 69%|██████▉   | 276262/400000 [00:31<00:14, 8340.30it/s] 69%|██████▉   | 277097/400000 [00:31<00:14, 8263.16it/s] 69%|██████▉   | 277962/400000 [00:31<00:14, 8374.60it/s] 70%|██████▉   | 278811/400000 [00:31<00:14, 8408.09it/s] 70%|██████▉   | 279661/400000 [00:31<00:14, 8435.17it/s] 70%|███████   | 280538/400000 [00:31<00:14, 8531.28it/s] 70%|███████   | 281401/400000 [00:32<00:13, 8558.36it/s] 71%|███████   | 282269/400000 [00:32<00:13, 8591.34it/s] 71%|███████   | 283129/400000 [00:32<00:13, 8532.19it/s] 71%|███████   | 283983/400000 [00:32<00:13, 8481.17it/s] 71%|███████   | 284832/400000 [00:32<00:13, 8367.01it/s] 71%|███████▏  | 285678/400000 [00:32<00:13, 8393.46it/s] 72%|███████▏  | 286548/400000 [00:32<00:13, 8481.81it/s] 72%|███████▏  | 287401/400000 [00:32<00:13, 8495.11it/s] 72%|███████▏  | 288257/400000 [00:32<00:13, 8511.78it/s] 72%|███████▏  | 289109/400000 [00:33<00:13, 8419.55it/s] 72%|███████▏  | 289952/400000 [00:33<00:13, 8412.05it/s] 73%|███████▎  | 290808/400000 [00:33<00:12, 8453.24it/s] 73%|███████▎  | 291670/400000 [00:33<00:12, 8501.81it/s] 73%|███████▎  | 292521/400000 [00:33<00:12, 8477.00it/s] 73%|███████▎  | 293428/400000 [00:33<00:12, 8644.30it/s] 74%|███████▎  | 294356/400000 [00:33<00:11, 8823.02it/s] 74%|███████▍  | 295252/400000 [00:33<00:11, 8863.53it/s] 74%|███████▍  | 296162/400000 [00:33<00:11, 8930.57it/s] 74%|███████▍  | 297056/400000 [00:33<00:11, 8667.60it/s] 74%|███████▍  | 297955/400000 [00:34<00:11, 8761.60it/s] 75%|███████▍  | 298881/400000 [00:34<00:11, 8904.79it/s] 75%|███████▍  | 299774/400000 [00:34<00:11, 8766.58it/s] 75%|███████▌  | 300680/400000 [00:34<00:11, 8850.86it/s] 75%|███████▌  | 301567/400000 [00:34<00:11, 8664.61it/s] 76%|███████▌  | 302436/400000 [00:34<00:11, 8669.22it/s] 76%|███████▌  | 303307/400000 [00:34<00:11, 8680.24it/s] 76%|███████▌  | 304176/400000 [00:34<00:11, 8623.36it/s] 76%|███████▋  | 305109/400000 [00:34<00:10, 8821.55it/s] 76%|███████▋  | 305993/400000 [00:34<00:11, 8443.74it/s] 77%|███████▋  | 306896/400000 [00:35<00:10, 8609.01it/s] 77%|███████▋  | 307835/400000 [00:35<00:10, 8827.95it/s] 77%|███████▋  | 308722/400000 [00:35<00:10, 8721.36it/s] 77%|███████▋  | 309598/400000 [00:35<00:10, 8662.29it/s] 78%|███████▊  | 310536/400000 [00:35<00:10, 8863.22it/s] 78%|███████▊  | 311527/400000 [00:35<00:09, 9150.95it/s] 78%|███████▊  | 312596/400000 [00:35<00:09, 9562.24it/s] 78%|███████▊  | 313683/400000 [00:35<00:08, 9919.06it/s] 79%|███████▊  | 314684/400000 [00:35<00:08, 9841.89it/s] 79%|███████▉  | 315675/400000 [00:35<00:08, 9846.35it/s] 79%|███████▉  | 316775/400000 [00:36<00:08, 10165.83it/s] 79%|███████▉  | 317888/400000 [00:36<00:07, 10434.72it/s] 80%|███████▉  | 318965/400000 [00:36<00:07, 10532.45it/s] 80%|████████  | 320068/400000 [00:36<00:07, 10673.74it/s] 80%|████████  | 321139/400000 [00:36<00:07, 10255.07it/s] 81%|████████  | 322198/400000 [00:36<00:07, 10350.68it/s] 81%|████████  | 323238/400000 [00:36<00:07, 10351.72it/s] 81%|████████  | 324322/400000 [00:36<00:07, 10492.18it/s] 81%|████████▏ | 325374/400000 [00:36<00:07, 10322.00it/s] 82%|████████▏ | 326409/400000 [00:36<00:07, 10229.49it/s] 82%|████████▏ | 327436/400000 [00:37<00:07, 10240.84it/s] 82%|████████▏ | 328462/400000 [00:37<00:07, 10209.89it/s] 82%|████████▏ | 329579/400000 [00:37<00:06, 10479.18it/s] 83%|████████▎ | 330676/400000 [00:37<00:06, 10618.83it/s] 83%|████████▎ | 331741/400000 [00:37<00:06, 10348.61it/s] 83%|████████▎ | 332784/400000 [00:37<00:06, 10371.27it/s] 83%|████████▎ | 333824/400000 [00:37<00:06, 10333.34it/s] 84%|████████▎ | 334921/400000 [00:37<00:06, 10513.16it/s] 84%|████████▍ | 336026/400000 [00:37<00:05, 10668.12it/s] 84%|████████▍ | 337095/400000 [00:38<00:06, 10189.58it/s] 85%|████████▍ | 338205/400000 [00:38<00:05, 10444.56it/s] 85%|████████▍ | 339282/400000 [00:38<00:05, 10539.60it/s] 85%|████████▌ | 340378/400000 [00:38<00:05, 10662.06it/s] 85%|████████▌ | 341448/400000 [00:38<00:05, 10189.68it/s] 86%|████████▌ | 342474/400000 [00:38<00:05, 9716.46it/s]  86%|████████▌ | 343455/400000 [00:38<00:05, 9579.60it/s] 86%|████████▌ | 344420/400000 [00:38<00:05, 9394.04it/s] 86%|████████▋ | 345410/400000 [00:38<00:05, 9537.75it/s] 87%|████████▋ | 346369/400000 [00:38<00:05, 9276.86it/s] 87%|████████▋ | 347321/400000 [00:39<00:05, 9347.31it/s] 87%|████████▋ | 348260/400000 [00:39<00:05, 9203.75it/s] 87%|████████▋ | 349184/400000 [00:39<00:05, 8902.22it/s] 88%|████████▊ | 350134/400000 [00:39<00:05, 9070.97it/s] 88%|████████▊ | 351045/400000 [00:39<00:05, 8972.48it/s] 88%|████████▊ | 351979/400000 [00:39<00:05, 9079.36it/s] 88%|████████▊ | 352890/400000 [00:39<00:05, 8945.75it/s] 88%|████████▊ | 353787/400000 [00:39<00:05, 8938.13it/s] 89%|████████▊ | 354683/400000 [00:39<00:05, 8923.85it/s] 89%|████████▉ | 355596/400000 [00:39<00:04, 8982.22it/s] 89%|████████▉ | 356591/400000 [00:40<00:04, 9250.07it/s] 89%|████████▉ | 357617/400000 [00:40<00:04, 9530.67it/s] 90%|████████▉ | 358616/400000 [00:40<00:04, 9663.24it/s] 90%|████████▉ | 359670/400000 [00:40<00:04, 9908.63it/s] 90%|█████████ | 360665/400000 [00:40<00:04, 9830.37it/s] 90%|█████████ | 361729/400000 [00:40<00:03, 10058.89it/s] 91%|█████████ | 362803/400000 [00:40<00:03, 10253.27it/s] 91%|█████████ | 363832/400000 [00:40<00:03, 9825.06it/s]  91%|█████████ | 364821/400000 [00:40<00:03, 9649.04it/s] 91%|█████████▏| 365791/400000 [00:41<00:03, 9604.67it/s] 92%|█████████▏| 366760/400000 [00:41<00:03, 9625.72it/s] 92%|█████████▏| 367771/400000 [00:41<00:03, 9765.61it/s] 92%|█████████▏| 368844/400000 [00:41<00:03, 10034.41it/s] 92%|█████████▏| 369874/400000 [00:41<00:02, 10111.13it/s] 93%|█████████▎| 370888/400000 [00:41<00:02, 10033.54it/s] 93%|█████████▎| 371894/400000 [00:41<00:02, 10031.78it/s] 93%|█████████▎| 372996/400000 [00:41<00:02, 10297.65it/s] 94%|█████████▎| 374029/400000 [00:41<00:02, 10207.68it/s] 94%|█████████▍| 375094/400000 [00:41<00:02, 10334.47it/s] 94%|█████████▍| 376130/400000 [00:42<00:02, 10152.34it/s] 94%|█████████▍| 377186/400000 [00:42<00:02, 10269.46it/s] 95%|█████████▍| 378215/400000 [00:42<00:02, 10149.23it/s] 95%|█████████▍| 379232/400000 [00:42<00:02, 9791.98it/s]  95%|█████████▌| 380255/400000 [00:42<00:01, 9919.34it/s] 95%|█████████▌| 381250/400000 [00:42<00:01, 9697.61it/s] 96%|█████████▌| 382336/400000 [00:42<00:01, 10017.24it/s] 96%|█████████▌| 383414/400000 [00:42<00:01, 10234.05it/s] 96%|█████████▌| 384520/400000 [00:42<00:01, 10466.35it/s] 96%|█████████▋| 385572/400000 [00:42<00:01, 9980.46it/s]  97%|█████████▋| 386578/400000 [00:43<00:01, 9725.53it/s] 97%|█████████▋| 387558/400000 [00:43<00:01, 9727.72it/s] 97%|█████████▋| 388644/400000 [00:43<00:01, 10039.33it/s] 97%|█████████▋| 389654/400000 [00:43<00:01, 9975.99it/s]  98%|█████████▊| 390663/400000 [00:43<00:00, 10007.66it/s] 98%|█████████▊| 391667/400000 [00:43<00:00, 9736.65it/s]  98%|█████████▊| 392645/400000 [00:43<00:00, 9728.95it/s] 98%|█████████▊| 393721/400000 [00:43<00:00, 10014.97it/s] 99%|█████████▊| 394768/400000 [00:43<00:00, 10146.31it/s] 99%|█████████▉| 395813/400000 [00:43<00:00, 10232.93it/s] 99%|█████████▉| 396871/400000 [00:44<00:00, 10332.39it/s] 99%|█████████▉| 397907/400000 [00:44<00:00, 10201.24it/s]100%|█████████▉| 398929/400000 [00:44<00:00, 10189.48it/s]100%|█████████▉| 399950/400000 [00:44<00:00, 10040.41it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 9005.96it/s] Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fc7710af0f0>, <torchtext.data.dataset.TabularDataset object at 0x7fc7710af240>, <torchtext.vocab.Vocab object at 0x7fc7710af160>), {}) 


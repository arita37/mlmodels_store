
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7feb2c542378> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7feb2c542378> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7feb9e5920d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7feb9e5920d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7feb35d289d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7feb35d289d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feb4be768c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feb4be768c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feb4be768c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▋      | 3596288/9912422 [00:00<00:00, 35957941.93it/s]9920512it [00:00, 34331700.34it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 248927.24it/s]           
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 149257.46it/s]1654784it [00:00, 10690152.68it/s]                         
0it [00:00, ?it/s]8192it [00:00, 210506.66it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feb2c6cf4e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feb2c6cfcc0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7feb4be76510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7feb4be76510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7feb4be76510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:16,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:15,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:15,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:14,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:52,  2.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:52,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:36,  4.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:36,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:36,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:36,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:35,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:35,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:25,  5.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:25,  5.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:25,  5.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:24,  5.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:24,  5.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:17,  7.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:17,  7.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  7.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:17,  7.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:17,  7.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:17,  7.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:17,  7.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:01<00:12, 10.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:12, 10.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:12, 10.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:12, 10.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:12, 10.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:12, 10.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:12, 10.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:09, 13.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:09, 13.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:09, 13.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:09, 13.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:06, 17.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:06, 17.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 17.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 17.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 17.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 17.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 17.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 22.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 22.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 22.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 22.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 22.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 22.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 22.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 27.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 27.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 27.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 27.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 27.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 27.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 27.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 32.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 32.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 32.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 32.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 32.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 32.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 32.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 36.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 36.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 36.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 36.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 36.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 36.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 36.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 41.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 41.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 41.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 41.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 41.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 41.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 41.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 45.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 45.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 45.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 45.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 45.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 45.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 45.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 47.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 50.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 50.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 50.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 50.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 50.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 50.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 50.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 52.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 52.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 52.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 52.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 52.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 52.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 52.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 53.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 53.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 53.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 53.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 53.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 53.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 53.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 54.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 54.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 54.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 54.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 54.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 54.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 54.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 55.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 55.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 55.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 55.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 55.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 55.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 55.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 55.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 56.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 56.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 56.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 56.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 56.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 56.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 56.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 56.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 56.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 56.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 56.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 56.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 56.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 56.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 56.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 56.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 56.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 56.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 56.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 56.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 56.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 56.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 57.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 57.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 57.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 57.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 57.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 57.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 57.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 56.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 56.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 56.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 56.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 56.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 56.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 56.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 57.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 57.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.38s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.38s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 57.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.38s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 57.02 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.84s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.38s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 57.02 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.84s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.84s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 27.75 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.84s/ url]
0 examples [00:00, ? examples/s]2020-05-30 18:10:11.707535: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-30 18:10:11.721554: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397215000 Hz
2020-05-30 18:10:11.721839: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558165b3aa60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-30 18:10:11.721864: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
44 examples [00:00, 438.33 examples/s]125 examples [00:00, 507.43 examples/s]220 examples [00:00, 589.17 examples/s]312 examples [00:00, 660.09 examples/s]407 examples [00:00, 725.92 examples/s]494 examples [00:00, 762.32 examples/s]587 examples [00:00, 804.32 examples/s]680 examples [00:00, 836.04 examples/s]775 examples [00:00, 866.44 examples/s]865 examples [00:01, 874.37 examples/s]954 examples [00:01, 876.52 examples/s]1050 examples [00:01, 898.10 examples/s]1148 examples [00:01, 919.58 examples/s]1246 examples [00:01, 936.38 examples/s]1343 examples [00:01, 941.58 examples/s]1441 examples [00:01, 951.05 examples/s]1541 examples [00:01, 961.53 examples/s]1638 examples [00:01, 960.15 examples/s]1735 examples [00:01, 935.50 examples/s]1829 examples [00:02, 933.48 examples/s]1927 examples [00:02, 944.76 examples/s]2022 examples [00:02, 944.73 examples/s]2117 examples [00:02, 941.12 examples/s]2212 examples [00:02, 925.99 examples/s]2305 examples [00:02, 892.87 examples/s]2401 examples [00:02, 909.61 examples/s]2498 examples [00:02, 926.73 examples/s]2596 examples [00:02, 941.82 examples/s]2691 examples [00:02, 937.10 examples/s]2785 examples [00:03, 909.77 examples/s]2881 examples [00:03, 922.35 examples/s]2978 examples [00:03, 935.37 examples/s]3076 examples [00:03, 945.60 examples/s]3171 examples [00:03, 937.62 examples/s]3268 examples [00:03, 946.79 examples/s]3366 examples [00:03, 954.93 examples/s]3466 examples [00:03, 967.26 examples/s]3565 examples [00:03, 969.76 examples/s]3663 examples [00:03, 968.11 examples/s]3761 examples [00:04, 969.68 examples/s]3859 examples [00:04, 945.18 examples/s]3954 examples [00:04, 932.39 examples/s]4048 examples [00:04, 926.20 examples/s]4142 examples [00:04, 928.89 examples/s]4238 examples [00:04, 935.91 examples/s]4332 examples [00:04, 905.94 examples/s]4423 examples [00:04, 868.76 examples/s]4511 examples [00:04, 868.68 examples/s]4599 examples [00:05, 822.94 examples/s]4685 examples [00:05, 833.09 examples/s]4777 examples [00:05, 856.41 examples/s]4873 examples [00:05, 884.07 examples/s]4969 examples [00:05, 905.37 examples/s]5064 examples [00:05, 917.90 examples/s]5157 examples [00:05, 912.59 examples/s]5252 examples [00:05, 921.01 examples/s]5347 examples [00:05, 929.11 examples/s]5442 examples [00:05, 934.58 examples/s]5536 examples [00:06, 931.94 examples/s]5630 examples [00:06, 883.72 examples/s]5724 examples [00:06, 897.95 examples/s]5820 examples [00:06, 913.59 examples/s]5912 examples [00:06, 910.70 examples/s]6004 examples [00:06, 901.22 examples/s]6095 examples [00:06, 884.28 examples/s]6189 examples [00:06, 900.23 examples/s]6280 examples [00:06, 901.76 examples/s]6371 examples [00:06, 896.44 examples/s]6463 examples [00:07, 902.71 examples/s]6554 examples [00:07, 884.03 examples/s]6646 examples [00:07, 892.87 examples/s]6743 examples [00:07, 912.30 examples/s]6835 examples [00:07, 891.09 examples/s]6925 examples [00:07, 864.10 examples/s]7012 examples [00:07, 863.39 examples/s]7099 examples [00:07, 861.54 examples/s]7186 examples [00:07, 863.88 examples/s]7276 examples [00:08, 872.00 examples/s]7367 examples [00:08, 877.89 examples/s]7455 examples [00:08, 837.87 examples/s]7543 examples [00:08, 848.75 examples/s]7640 examples [00:08, 881.79 examples/s]7739 examples [00:08, 910.64 examples/s]7838 examples [00:08, 932.76 examples/s]7934 examples [00:08, 940.72 examples/s]8029 examples [00:08, 932.77 examples/s]8123 examples [00:08, 919.70 examples/s]8216 examples [00:09, 910.85 examples/s]8308 examples [00:09, 847.91 examples/s]8395 examples [00:09, 853.77 examples/s]8492 examples [00:09, 883.09 examples/s]8588 examples [00:09, 904.52 examples/s]8687 examples [00:09, 926.08 examples/s]8781 examples [00:09, 926.74 examples/s]8875 examples [00:09, 929.99 examples/s]8969 examples [00:09, 898.16 examples/s]9060 examples [00:10, 871.95 examples/s]9153 examples [00:10, 885.85 examples/s]9242 examples [00:10, 871.23 examples/s]9330 examples [00:10, 873.20 examples/s]9422 examples [00:10, 885.20 examples/s]9514 examples [00:10, 894.59 examples/s]9609 examples [00:10, 908.59 examples/s]9702 examples [00:10, 912.82 examples/s]9800 examples [00:10, 930.69 examples/s]9894 examples [00:10, 928.73 examples/s]9991 examples [00:11, 938.37 examples/s]10085 examples [00:11, 876.93 examples/s]10174 examples [00:11, 874.07 examples/s]10263 examples [00:11, 868.85 examples/s]10351 examples [00:11, 864.08 examples/s]10438 examples [00:11, 837.31 examples/s]10529 examples [00:11, 855.65 examples/s]10624 examples [00:11, 880.09 examples/s]10724 examples [00:11, 910.54 examples/s]10816 examples [00:11, 910.79 examples/s]10916 examples [00:12, 934.65 examples/s]11015 examples [00:12, 950.53 examples/s]11111 examples [00:12, 905.12 examples/s]11210 examples [00:12, 928.20 examples/s]11307 examples [00:12, 937.99 examples/s]11404 examples [00:12, 945.75 examples/s]11499 examples [00:12, 932.13 examples/s]11596 examples [00:12, 942.37 examples/s]11694 examples [00:12, 951.30 examples/s]11793 examples [00:12, 961.03 examples/s]11890 examples [00:13, 955.82 examples/s]11986 examples [00:13, 941.67 examples/s]12081 examples [00:13, 927.90 examples/s]12174 examples [00:13, 927.34 examples/s]12267 examples [00:13, 901.51 examples/s]12358 examples [00:13, 887.24 examples/s]12449 examples [00:13, 893.03 examples/s]12539 examples [00:13, 892.30 examples/s]12637 examples [00:13, 916.30 examples/s]12737 examples [00:14, 938.05 examples/s]12834 examples [00:14, 945.88 examples/s]12933 examples [00:14, 957.01 examples/s]13029 examples [00:14, 953.79 examples/s]13125 examples [00:14, 950.84 examples/s]13222 examples [00:14, 954.98 examples/s]13318 examples [00:14, 931.70 examples/s]13412 examples [00:14, 896.64 examples/s]13504 examples [00:14, 901.54 examples/s]13603 examples [00:14, 924.14 examples/s]13701 examples [00:15, 939.73 examples/s]13796 examples [00:15, 901.38 examples/s]13887 examples [00:15, 880.65 examples/s]13976 examples [00:15, 879.05 examples/s]14067 examples [00:15, 885.73 examples/s]14158 examples [00:15, 891.94 examples/s]14255 examples [00:15, 912.74 examples/s]14350 examples [00:15, 921.79 examples/s]14444 examples [00:15, 925.25 examples/s]14537 examples [00:15, 921.58 examples/s]14633 examples [00:16, 931.10 examples/s]14727 examples [00:16, 931.72 examples/s]14823 examples [00:16, 938.62 examples/s]14917 examples [00:16, 933.20 examples/s]15015 examples [00:16, 946.71 examples/s]15114 examples [00:16, 958.15 examples/s]15212 examples [00:16, 963.57 examples/s]15309 examples [00:16, 951.35 examples/s]15405 examples [00:16, 952.15 examples/s]15501 examples [00:16, 949.12 examples/s]15598 examples [00:17, 955.13 examples/s]15694 examples [00:17, 955.63 examples/s]15790 examples [00:17, 954.72 examples/s]15886 examples [00:17, 937.94 examples/s]15980 examples [00:17, 937.11 examples/s]16075 examples [00:17, 937.68 examples/s]16170 examples [00:17, 939.27 examples/s]16268 examples [00:17, 948.58 examples/s]16363 examples [00:17, 945.32 examples/s]16458 examples [00:18, 944.39 examples/s]16558 examples [00:18, 958.83 examples/s]16654 examples [00:18, 956.89 examples/s]16750 examples [00:18, 955.49 examples/s]16846 examples [00:18, 928.21 examples/s]16945 examples [00:18, 944.39 examples/s]17043 examples [00:18, 952.31 examples/s]17140 examples [00:18, 955.44 examples/s]17236 examples [00:18, 949.50 examples/s]17332 examples [00:18, 936.85 examples/s]17426 examples [00:19, 933.52 examples/s]17520 examples [00:19, 925.38 examples/s]17618 examples [00:19, 939.33 examples/s]17717 examples [00:19, 952.87 examples/s]17813 examples [00:19, 952.83 examples/s]17911 examples [00:19, 960.49 examples/s]18008 examples [00:19, 948.25 examples/s]18103 examples [00:19, 922.92 examples/s]18198 examples [00:19, 930.24 examples/s]18292 examples [00:19, 931.80 examples/s]18386 examples [00:20, 909.64 examples/s]18479 examples [00:20, 914.15 examples/s]18571 examples [00:20, 905.84 examples/s]18662 examples [00:20, 898.15 examples/s]18757 examples [00:20, 912.47 examples/s]18855 examples [00:20, 930.57 examples/s]18953 examples [00:20, 944.49 examples/s]19048 examples [00:20, 935.69 examples/s]19144 examples [00:20, 940.96 examples/s]19244 examples [00:20, 956.04 examples/s]19343 examples [00:21, 965.36 examples/s]19440 examples [00:21, 939.84 examples/s]19535 examples [00:21, 921.09 examples/s]19628 examples [00:21, 887.36 examples/s]19718 examples [00:21, 869.52 examples/s]19807 examples [00:21, 874.14 examples/s]19895 examples [00:21, 871.43 examples/s]19983 examples [00:21, 860.28 examples/s]20070 examples [00:21, 826.94 examples/s]20160 examples [00:22, 845.46 examples/s]20257 examples [00:22, 879.18 examples/s]20349 examples [00:22, 890.57 examples/s]20447 examples [00:22, 913.80 examples/s]20539 examples [00:22, 906.44 examples/s]20631 examples [00:22, 908.37 examples/s]20723 examples [00:22, 910.22 examples/s]20815 examples [00:22, 896.80 examples/s]20905 examples [00:22, 886.24 examples/s]21003 examples [00:22, 911.63 examples/s]21100 examples [00:23, 925.79 examples/s]21198 examples [00:23, 941.22 examples/s]21293 examples [00:23, 935.98 examples/s]21389 examples [00:23, 941.06 examples/s]21484 examples [00:23, 922.84 examples/s]21577 examples [00:23, 923.97 examples/s]21675 examples [00:23, 939.58 examples/s]21773 examples [00:23, 949.95 examples/s]21870 examples [00:23, 953.76 examples/s]21966 examples [00:23, 951.40 examples/s]22062 examples [00:24, 930.90 examples/s]22156 examples [00:24, 913.21 examples/s]22249 examples [00:24, 915.75 examples/s]22341 examples [00:24, 916.78 examples/s]22433 examples [00:24, 910.66 examples/s]22525 examples [00:24, 899.16 examples/s]22623 examples [00:24, 920.31 examples/s]22721 examples [00:24, 935.89 examples/s]22815 examples [00:24, 928.39 examples/s]22908 examples [00:25, 904.28 examples/s]22999 examples [00:25, 861.36 examples/s]23086 examples [00:25, 843.88 examples/s]23174 examples [00:25, 851.95 examples/s]23260 examples [00:25, 851.98 examples/s]23356 examples [00:25, 879.82 examples/s]23451 examples [00:25, 899.74 examples/s]23548 examples [00:25, 918.35 examples/s]23642 examples [00:25, 923.62 examples/s]23735 examples [00:25, 908.87 examples/s]23827 examples [00:26, 895.71 examples/s]23917 examples [00:26, 890.14 examples/s]24007 examples [00:26, 868.59 examples/s]24097 examples [00:26, 876.96 examples/s]24185 examples [00:26, 870.28 examples/s]24276 examples [00:26, 879.70 examples/s]24366 examples [00:26, 885.31 examples/s]24455 examples [00:26, 876.80 examples/s]24553 examples [00:26, 903.83 examples/s]24647 examples [00:26, 913.39 examples/s]24744 examples [00:27, 928.92 examples/s]24842 examples [00:27, 942.05 examples/s]24941 examples [00:27, 953.58 examples/s]25041 examples [00:27, 965.18 examples/s]25139 examples [00:27, 968.06 examples/s]25236 examples [00:27, 953.48 examples/s]25332 examples [00:27, 876.61 examples/s]25424 examples [00:27, 888.40 examples/s]25516 examples [00:27, 895.99 examples/s]25608 examples [00:28, 901.63 examples/s]25705 examples [00:28, 919.99 examples/s]25801 examples [00:28, 931.62 examples/s]25901 examples [00:28, 951.01 examples/s]26001 examples [00:28, 962.64 examples/s]26098 examples [00:28, 935.45 examples/s]26192 examples [00:28, 909.76 examples/s]26284 examples [00:28, 896.26 examples/s]26374 examples [00:28, 897.32 examples/s]26468 examples [00:28, 908.50 examples/s]26562 examples [00:29, 916.87 examples/s]26656 examples [00:29, 922.23 examples/s]26749 examples [00:29, 896.43 examples/s]26841 examples [00:29, 903.33 examples/s]26940 examples [00:29, 926.07 examples/s]27036 examples [00:29, 930.88 examples/s]27130 examples [00:29, 917.68 examples/s]27222 examples [00:29, 909.37 examples/s]27314 examples [00:29, 903.86 examples/s]27408 examples [00:29, 911.86 examples/s]27500 examples [00:30, 905.76 examples/s]27591 examples [00:30, 882.24 examples/s]27680 examples [00:30, 879.59 examples/s]27779 examples [00:30, 908.19 examples/s]27875 examples [00:30, 922.72 examples/s]27974 examples [00:30, 939.67 examples/s]28069 examples [00:30, 913.93 examples/s]28164 examples [00:30, 923.26 examples/s]28257 examples [00:30, 898.77 examples/s]28349 examples [00:31, 902.83 examples/s]28440 examples [00:31, 901.91 examples/s]28536 examples [00:31, 916.82 examples/s]28633 examples [00:31, 932.02 examples/s]28733 examples [00:31, 950.08 examples/s]28830 examples [00:31, 955.92 examples/s]28929 examples [00:31, 963.83 examples/s]29026 examples [00:31, 940.03 examples/s]29126 examples [00:31, 954.05 examples/s]29227 examples [00:31, 967.90 examples/s]29324 examples [00:32, 957.02 examples/s]29421 examples [00:32, 959.16 examples/s]29521 examples [00:32, 969.59 examples/s]29619 examples [00:32, 947.72 examples/s]29714 examples [00:32, 931.82 examples/s]29808 examples [00:32, 914.17 examples/s]29900 examples [00:32, 897.53 examples/s]29991 examples [00:32, 899.56 examples/s]30082 examples [00:32, 880.47 examples/s]30175 examples [00:32, 892.14 examples/s]30265 examples [00:33, 888.28 examples/s]30354 examples [00:33, 887.23 examples/s]30443 examples [00:33, 869.63 examples/s]30539 examples [00:33, 894.47 examples/s]30640 examples [00:33, 923.69 examples/s]30733 examples [00:33, 921.39 examples/s]30830 examples [00:33, 932.72 examples/s]30924 examples [00:33, 906.45 examples/s]31021 examples [00:33, 924.14 examples/s]31118 examples [00:33, 936.05 examples/s]31212 examples [00:34, 892.99 examples/s]31302 examples [00:34, 889.50 examples/s]31392 examples [00:34, 890.94 examples/s]31482 examples [00:34, 892.19 examples/s]31577 examples [00:34, 906.82 examples/s]31675 examples [00:34, 926.10 examples/s]31768 examples [00:34, 922.74 examples/s]31861 examples [00:34, 917.26 examples/s]31953 examples [00:34, 917.06 examples/s]32045 examples [00:35, 911.32 examples/s]32138 examples [00:35, 914.88 examples/s]32230 examples [00:35, 909.58 examples/s]32327 examples [00:35, 926.81 examples/s]32422 examples [00:35, 930.74 examples/s]32516 examples [00:35, 906.82 examples/s]32607 examples [00:35, 875.04 examples/s]32697 examples [00:35, 880.44 examples/s]32792 examples [00:35, 899.50 examples/s]32888 examples [00:35, 914.10 examples/s]32985 examples [00:36, 929.41 examples/s]33081 examples [00:36, 935.66 examples/s]33177 examples [00:36, 941.23 examples/s]33274 examples [00:36, 949.26 examples/s]33371 examples [00:36, 955.37 examples/s]33467 examples [00:36, 956.28 examples/s]33563 examples [00:36, 951.68 examples/s]33659 examples [00:36, 931.04 examples/s]33753 examples [00:36, 876.21 examples/s]33842 examples [00:36, 878.55 examples/s]33931 examples [00:37, 871.88 examples/s]34026 examples [00:37, 893.31 examples/s]34116 examples [00:37, 887.04 examples/s]34209 examples [00:37, 896.85 examples/s]34307 examples [00:37, 919.22 examples/s]34407 examples [00:37, 940.77 examples/s]34502 examples [00:37, 933.75 examples/s]34596 examples [00:37, 930.32 examples/s]34692 examples [00:37, 937.33 examples/s]34788 examples [00:38, 943.60 examples/s]34886 examples [00:38, 952.72 examples/s]34982 examples [00:38, 948.66 examples/s]35077 examples [00:38, 932.23 examples/s]35171 examples [00:38, 921.88 examples/s]35270 examples [00:38, 938.96 examples/s]35370 examples [00:38, 955.43 examples/s]35466 examples [00:38, 948.34 examples/s]35561 examples [00:38, 936.25 examples/s]35655 examples [00:38, 900.99 examples/s]35746 examples [00:39, 887.97 examples/s]35838 examples [00:39, 895.50 examples/s]35928 examples [00:39, 892.13 examples/s]36018 examples [00:39, 893.43 examples/s]36117 examples [00:39, 918.77 examples/s]36210 examples [00:39, 919.20 examples/s]36304 examples [00:39, 923.96 examples/s]36397 examples [00:39, 918.32 examples/s]36489 examples [00:39, 913.01 examples/s]36581 examples [00:39, 912.40 examples/s]36680 examples [00:40, 931.71 examples/s]36776 examples [00:40, 937.96 examples/s]36870 examples [00:40, 926.15 examples/s]36966 examples [00:40, 935.32 examples/s]37060 examples [00:40, 935.03 examples/s]37154 examples [00:40, 898.80 examples/s]37249 examples [00:40, 912.22 examples/s]37342 examples [00:40, 917.03 examples/s]37439 examples [00:40, 931.90 examples/s]37533 examples [00:40, 929.80 examples/s]37627 examples [00:41, 929.21 examples/s]37721 examples [00:41, 929.67 examples/s]37815 examples [00:41, 929.47 examples/s]37909 examples [00:41, 930.04 examples/s]38003 examples [00:41, 928.10 examples/s]38101 examples [00:41, 942.44 examples/s]38196 examples [00:41, 937.25 examples/s]38291 examples [00:41, 939.79 examples/s]38390 examples [00:41, 953.57 examples/s]38489 examples [00:41, 963.56 examples/s]38586 examples [00:42, 921.64 examples/s]38679 examples [00:42, 898.44 examples/s]38770 examples [00:42, 882.11 examples/s]38859 examples [00:42, 882.75 examples/s]38951 examples [00:42, 891.77 examples/s]39045 examples [00:42, 902.67 examples/s]39136 examples [00:42, 877.72 examples/s]39229 examples [00:42, 890.77 examples/s]39319 examples [00:42, 882.04 examples/s]39409 examples [00:43, 886.94 examples/s]39504 examples [00:43, 904.74 examples/s]39603 examples [00:43, 926.29 examples/s]39700 examples [00:43, 937.62 examples/s]39795 examples [00:43, 940.02 examples/s]39890 examples [00:43, 919.01 examples/s]39983 examples [00:43, 882.83 examples/s]40072 examples [00:43, 832.74 examples/s]40164 examples [00:43, 855.35 examples/s]40262 examples [00:43, 888.51 examples/s]40361 examples [00:44, 915.41 examples/s]40460 examples [00:44, 936.30 examples/s]40557 examples [00:44, 944.89 examples/s]40652 examples [00:44, 940.15 examples/s]40751 examples [00:44, 953.47 examples/s]40850 examples [00:44, 962.06 examples/s]40949 examples [00:44, 967.66 examples/s]41046 examples [00:44, 963.65 examples/s]41146 examples [00:44, 972.73 examples/s]41245 examples [00:44, 976.97 examples/s]41346 examples [00:45, 985.15 examples/s]41445 examples [00:45, 972.21 examples/s]41543 examples [00:45, 941.79 examples/s]41638 examples [00:45, 917.13 examples/s]41737 examples [00:45, 936.49 examples/s]41835 examples [00:45, 946.87 examples/s]41930 examples [00:45, 939.50 examples/s]42025 examples [00:45, 929.85 examples/s]42119 examples [00:45, 926.58 examples/s]42212 examples [00:46, 909.24 examples/s]42310 examples [00:46, 929.27 examples/s]42404 examples [00:46, 913.61 examples/s]42496 examples [00:46, 899.59 examples/s]42588 examples [00:46, 904.84 examples/s]42679 examples [00:46, 904.62 examples/s]42772 examples [00:46, 910.98 examples/s]42872 examples [00:46, 934.76 examples/s]42970 examples [00:46, 945.00 examples/s]43065 examples [00:46, 933.68 examples/s]43159 examples [00:47, 918.24 examples/s]43257 examples [00:47, 935.52 examples/s]43351 examples [00:47, 930.27 examples/s]43445 examples [00:47, 915.48 examples/s]43538 examples [00:47, 919.42 examples/s]43636 examples [00:47, 935.12 examples/s]43730 examples [00:47, 912.89 examples/s]43822 examples [00:47, 902.15 examples/s]43913 examples [00:47, 876.39 examples/s]44004 examples [00:47, 884.71 examples/s]44093 examples [00:48, 867.85 examples/s]44187 examples [00:48, 886.52 examples/s]44281 examples [00:48, 898.38 examples/s]44372 examples [00:48, 898.44 examples/s]44463 examples [00:48, 897.02 examples/s]44553 examples [00:48, 883.38 examples/s]44643 examples [00:48, 886.48 examples/s]44732 examples [00:48, 883.84 examples/s]44828 examples [00:48, 903.76 examples/s]44921 examples [00:49, 908.75 examples/s]45012 examples [00:49, 908.99 examples/s]45103 examples [00:49, 907.04 examples/s]45194 examples [00:49, 907.61 examples/s]45285 examples [00:49, 906.87 examples/s]45382 examples [00:49, 924.48 examples/s]45480 examples [00:49, 939.90 examples/s]45578 examples [00:49, 950.34 examples/s]45675 examples [00:49, 955.28 examples/s]45771 examples [00:49, 954.95 examples/s]45868 examples [00:50, 957.35 examples/s]45964 examples [00:50, 952.77 examples/s]46060 examples [00:50, 904.78 examples/s]46155 examples [00:50, 915.10 examples/s]46251 examples [00:50, 927.78 examples/s]46350 examples [00:50, 944.56 examples/s]46445 examples [00:50, 943.73 examples/s]46540 examples [00:50, 934.71 examples/s]46634 examples [00:50, 926.51 examples/s]46732 examples [00:50, 939.46 examples/s]46831 examples [00:51, 953.11 examples/s]46929 examples [00:51, 960.43 examples/s]47026 examples [00:51, 962.39 examples/s]47123 examples [00:51, 960.57 examples/s]47220 examples [00:51, 928.96 examples/s]47314 examples [00:51, 925.41 examples/s]47411 examples [00:51, 936.65 examples/s]47505 examples [00:51, 924.20 examples/s]47598 examples [00:51, 911.97 examples/s]47690 examples [00:51, 908.03 examples/s]47781 examples [00:52, 889.45 examples/s]47871 examples [00:52, 886.17 examples/s]47969 examples [00:52, 912.12 examples/s]48066 examples [00:52, 927.94 examples/s]48160 examples [00:52, 913.59 examples/s]48252 examples [00:52, 905.85 examples/s]48343 examples [00:52, 891.23 examples/s]48434 examples [00:52, 896.20 examples/s]48526 examples [00:52, 902.51 examples/s]48620 examples [00:53, 911.46 examples/s]48715 examples [00:53, 920.20 examples/s]48811 examples [00:53, 929.31 examples/s]48908 examples [00:53, 939.66 examples/s]49003 examples [00:53, 917.31 examples/s]49100 examples [00:53, 929.94 examples/s]49195 examples [00:53, 934.91 examples/s]49295 examples [00:53, 952.60 examples/s]49395 examples [00:53, 965.57 examples/s]49492 examples [00:53, 964.95 examples/s]49591 examples [00:54, 970.39 examples/s]49689 examples [00:54, 969.00 examples/s]49786 examples [00:54, 968.07 examples/s]49883 examples [00:54, 965.36 examples/s]49980 examples [00:54, 949.35 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6173/50000 [00:00<00:00, 61725.20 examples/s] 34%|███▍      | 17213/50000 [00:00<00:00, 71133.80 examples/s] 56%|█████▋    | 28184/50000 [00:00<00:00, 79521.27 examples/s] 79%|███████▉  | 39435/50000 [00:00<00:00, 87190.58 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 697.22 examples/s]151 examples [00:00, 727.54 examples/s]239 examples [00:00, 766.66 examples/s]335 examples [00:00, 814.23 examples/s]431 examples [00:00, 852.50 examples/s]531 examples [00:00, 890.22 examples/s]633 examples [00:00, 923.84 examples/s]721 examples [00:00, 718.60 examples/s]811 examples [00:00, 763.68 examples/s]902 examples [00:01, 800.25 examples/s]994 examples [00:01, 830.38 examples/s]1091 examples [00:01, 867.46 examples/s]1190 examples [00:01, 900.59 examples/s]1288 examples [00:01, 919.80 examples/s]1382 examples [00:01, 881.04 examples/s]1473 examples [00:01, 887.61 examples/s]1570 examples [00:01, 908.65 examples/s]1669 examples [00:01, 931.37 examples/s]1765 examples [00:02, 938.13 examples/s]1860 examples [00:02, 929.75 examples/s]1957 examples [00:02, 940.37 examples/s]2052 examples [00:02, 942.02 examples/s]2147 examples [00:02, 937.07 examples/s]2241 examples [00:02, 914.08 examples/s]2333 examples [00:02, 914.85 examples/s]2434 examples [00:02, 940.04 examples/s]2533 examples [00:02, 953.83 examples/s]2632 examples [00:02, 963.37 examples/s]2730 examples [00:03, 967.24 examples/s]2827 examples [00:03, 946.36 examples/s]2922 examples [00:03, 873.41 examples/s]3016 examples [00:03, 890.20 examples/s]3115 examples [00:03, 916.78 examples/s]3211 examples [00:03, 928.85 examples/s]3309 examples [00:03, 941.30 examples/s]3406 examples [00:03, 947.55 examples/s]3502 examples [00:03, 941.36 examples/s]3597 examples [00:03, 937.30 examples/s]3691 examples [00:04, 931.78 examples/s]3791 examples [00:04, 949.11 examples/s]3890 examples [00:04, 959.46 examples/s]3990 examples [00:04, 968.23 examples/s]4090 examples [00:04, 976.36 examples/s]4188 examples [00:04, 971.52 examples/s]4287 examples [00:04, 976.70 examples/s]4385 examples [00:04, 968.98 examples/s]4482 examples [00:04, 925.70 examples/s]4576 examples [00:05, 925.03 examples/s]4673 examples [00:05, 937.88 examples/s]4773 examples [00:05, 953.71 examples/s]4869 examples [00:05, 948.33 examples/s]4965 examples [00:05, 935.23 examples/s]5059 examples [00:05, 923.05 examples/s]5152 examples [00:05, 912.35 examples/s]5247 examples [00:05, 922.73 examples/s]5345 examples [00:05, 937.73 examples/s]5443 examples [00:05, 949.87 examples/s]5541 examples [00:06, 958.15 examples/s]5637 examples [00:06, 951.41 examples/s]5733 examples [00:06, 942.60 examples/s]5828 examples [00:06, 906.55 examples/s]5920 examples [00:06, 903.72 examples/s]6012 examples [00:06, 908.51 examples/s]6107 examples [00:06, 918.26 examples/s]6205 examples [00:06, 935.25 examples/s]6299 examples [00:06, 927.31 examples/s]6396 examples [00:06, 939.37 examples/s]6494 examples [00:07, 950.83 examples/s]6590 examples [00:07, 933.38 examples/s]6684 examples [00:07, 920.90 examples/s]6777 examples [00:07, 921.94 examples/s]6870 examples [00:07, 918.26 examples/s]6962 examples [00:07, 885.05 examples/s]7051 examples [00:07, 876.01 examples/s]7144 examples [00:07, 889.66 examples/s]7241 examples [00:07, 912.02 examples/s]7344 examples [00:07, 942.20 examples/s]7439 examples [00:08, 938.31 examples/s]7534 examples [00:08, 917.12 examples/s]7627 examples [00:08, 909.28 examples/s]7719 examples [00:08, 905.49 examples/s]7810 examples [00:08, 887.38 examples/s]7904 examples [00:08, 900.98 examples/s]8001 examples [00:08, 918.19 examples/s]8100 examples [00:08, 936.95 examples/s]8196 examples [00:08, 942.28 examples/s]8291 examples [00:09, 924.57 examples/s]8388 examples [00:09, 936.74 examples/s]8482 examples [00:09, 928.56 examples/s]8575 examples [00:09, 912.07 examples/s]8667 examples [00:09, 900.68 examples/s]8762 examples [00:09, 913.58 examples/s]8856 examples [00:09, 918.78 examples/s]8948 examples [00:09, 895.38 examples/s]9042 examples [00:09, 907.45 examples/s]9140 examples [00:09, 926.39 examples/s]9233 examples [00:10, 922.13 examples/s]9326 examples [00:10, 920.63 examples/s]9419 examples [00:10, 907.82 examples/s]9510 examples [00:10, 907.98 examples/s]9607 examples [00:10, 923.80 examples/s]9702 examples [00:10, 930.82 examples/s]9799 examples [00:10, 941.54 examples/s]9894 examples [00:10, 929.96 examples/s]9988 examples [00:10, 928.76 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]100%|█████████▉| 9984/10000 [00:00<00:00, 99831.53 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteU1J9WB/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteU1J9WB/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feb4be768c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feb4be768c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feb4be768c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feb24722390>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feb24a22780>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feb4be768c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feb4be768c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feb4be768c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feb2d3e7438>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feb2d3e72e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7feaeb8356a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7feaeb8356a8> 

  function with postional parmater data_info <function split_train_valid at 0x7feaeb8356a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7feaeb8357b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7feaeb8357b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7feaeb8357b8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=12449a96620169642b5d14a9e0f7bf8efee19c02aae1dfca385c9e3b21414df1
  Stored in directory: /tmp/pip-ephem-wheel-cache-ygizkbuy/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<128:00:06, 1.87kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<89:49:47, 2.67kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:04<62:55:28, 3.81kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:04<44:01:35, 5.43kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:04<30:43:25, 7.76kB/s].vector_cache/glove.6B.zip:   1%|          | 9.37M/862M [00:05<21:21:53, 11.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:05<14:51:21, 15.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.0M/862M [00:05<10:19:45, 22.6kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.7M/862M [00:05<7:10:58, 32.3kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 32.4M/862M [00:05<4:59:42, 46.1kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.9M/862M [00:05<3:28:28, 65.9kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.9M/862M [00:05<2:25:33, 94.0kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.7M/862M [00:05<1:41:14, 134kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 49.5M/862M [00:05<1:10:46, 191kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:06<50:10, 269kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:08<36:51, 364kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:08<28:03, 478kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:08<20:10, 665kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:10<16:24, 815kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:10<12:39, 1.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:10<09:07, 1.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:10<06:36, 2.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:12<24:47, 536kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:12<18:39, 712kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.8M/862M [00:12<13:20, 994kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:14<12:28, 1.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:14<11:39, 1.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:14<08:52, 1.49MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:14<06:21, 2.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:16<23:39, 556kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:16<18:00, 730kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:16<12:56, 1.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:18<11:55, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:18<11:13, 1.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:18<08:27, 1.54MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:18<06:05, 2.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:20<09:44, 1.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:20<08:13, 1.58MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:20<06:03, 2.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:22<07:08, 1.81MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:22<07:44, 1.67MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:22<06:00, 2.15MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:22<04:22, 2.95MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:24<08:29, 1.52MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:24<07:19, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:24<05:28, 2.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:26<06:39, 1.92MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:26<07:22, 1.74MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:26<05:50, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:26<04:12, 3.03MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:28<20:09, 632kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:28<15:28, 822kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:28<11:10, 1.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<10:35, 1.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<10:13, 1.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<07:46, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<05:35, 2.25MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<21:31, 585kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<16:25, 766kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:32<11:45, 1.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:34<10:59, 1.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:34<10:28, 1.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<08:00, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<05:45, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:36<23:37, 527kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:36<17:52, 696kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:36<12:49, 969kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<11:41, 1.06MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<10:50, 1.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:38<08:08, 1.52MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<05:53, 2.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<08:20, 1.48MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<07:09, 1.72MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:40<05:20, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<06:26, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:42<07:13, 1.70MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:42<05:42, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<04:07, 2.95MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<21:17, 572kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<16:14, 749kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:44<11:40, 1.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:46<10:49, 1.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:46<10:16, 1.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:46<07:50, 1.54MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:46<05:37, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:48<22:54, 525kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:48<17:19, 694kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:48<12:25, 966kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:50<11:19, 1.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:50<10:27, 1.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:50<07:57, 1.50MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<05:42, 2.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:52<09:50, 1.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:52<08:10, 1.46MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:52<05:59, 1.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:54<06:48, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:54<07:15, 1.63MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:54<05:37, 2.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<04:06, 2.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:56<07:33, 1.56MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:56<06:34, 1.79MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:56<04:54, 2.39MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:58<06:00, 1.95MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:58<05:28, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:58<04:06, 2.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:00<05:26, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:00<06:22, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:00<05:01, 2.31MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:00<03:39, 3.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:02<18:44, 616kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:02<14:20, 805kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:02<10:19, 1.12MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<09:46, 1.17MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<09:24, 1.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:04<07:07, 1.61MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:04<05:06, 2.24MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:06<10:17, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:06<08:27, 1.35MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:06<06:10, 1.84MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:08<06:49, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:08<06:01, 1.88MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:08<04:28, 2.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:10<05:35, 2.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:10<06:25, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:10<05:00, 2.25MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<03:38, 3.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:12<08:04, 1.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:12<06:51, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<05:05, 2.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:12<04:14, 2.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:14<7:51:04, 23.6kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:14<5:30:00, 33.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<3:50:08, 48.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<2:53:56, 63.6kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:16<2:04:04, 89.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:16<1:27:20, 127kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<1:00:59, 180kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<1:16:05, 145kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:18<54:23, 202kB/s]  .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<38:17, 287kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<29:09, 375kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:20<22:50, 478kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:20<16:27, 663kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<11:40, 933kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<11:47, 921kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:22<09:25, 1.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:22<06:49, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<07:09, 1.51MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:24<06:10, 1.75MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:24<04:33, 2.36MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:25<05:34, 1.93MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:26<06:15, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:26<04:52, 2.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<03:32, 3.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<07:16, 1.46MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:28<06:14, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<04:36, 2.31MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:29<05:33, 1.90MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:30<06:07, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:30<04:51, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:30<03:31, 2.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<19:01, 553kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:32<14:26, 727kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<10:20, 1.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<09:31, 1.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:34<08:46, 1.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:34<06:40, 1.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<04:47, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<1:17:33, 134kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<55:22, 187kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:36<38:56, 266kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:37<29:25, 350kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:37<22:46, 452kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:38<16:23, 628kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<11:35, 885kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:39<12:11, 840kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:39<09:36, 1.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<06:56, 1.47MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<07:09, 1.42MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<07:15, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:42<05:37, 1.80MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:42<04:02, 2.50MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<15:31, 651kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:43<11:56, 845kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:44<08:35, 1.17MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:45<08:11, 1.22MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:45<06:49, 1.47MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<05:00, 1.99MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:47<05:41, 1.75MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:47<05:04, 1.96MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<03:49, 2.60MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:49<04:51, 2.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:49<05:30, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:49<04:21, 2.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:50<03:09, 3.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:51<1:12:24, 136kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:51<51:41, 190kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<36:18, 270kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:53<27:32, 354kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:53<21:20, 457kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:53<15:22, 633kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:54<10:50, 895kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:55<13:07, 737kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:55<10:13, 946kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<07:24, 1.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<07:17, 1.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<07:13, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:57<05:30, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:57<03:57, 2.41MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:59<07:54, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:59<06:20, 1.50MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:59<04:37, 2.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:59<03:23, 2.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:01<26:55, 352kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:01<20:50, 454kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:01<15:04, 628kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:02<10:37, 886kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:03<37:19, 252kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:03<27:06, 347kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<19:08, 490kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:05<15:27, 604kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:05<12:49, 728kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:05<09:25, 990kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:05<06:39, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:07<15:07, 613kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:07<11:33, 802kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<08:18, 1.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<07:51, 1.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<07:22, 1.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:09<05:37, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:09<04:01, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:11<50:41, 180kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:11<36:26, 250kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<25:39, 355kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:13<19:53, 456kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:13<15:50, 572kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:13<11:33, 782kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<08:09, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:15<17:00, 529kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:15<12:52, 698kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:15<09:12, 973kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:17<08:24, 1.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:17<07:45, 1.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:17<05:50, 1.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<04:12, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:19<06:09, 1.44MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:19<05:15, 1.68MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<03:52, 2.28MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:21<04:39, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:21<05:12, 1.69MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:21<04:03, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<02:54, 2.99MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:23<12:30, 697kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:23<09:41, 898kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:23<07:00, 1.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:25<06:47, 1.27MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:25<06:39, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:25<05:07, 1.68MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:25<03:40, 2.33MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:27<16:04, 533kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:27<12:08, 706kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<08:41, 982kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:29<08:00, 1.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:29<07:24, 1.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:29<05:37, 1.51MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:29<04:00, 2.10MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:31<23:07, 365kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:31<17:05, 493kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:31<12:07, 693kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:33<10:18, 812kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:33<08:59, 930kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:33<06:42, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<04:46, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:35<14:40, 565kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:35<11:08, 744kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<07:59, 1.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:37<07:28, 1.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:37<06:57, 1.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:37<05:18, 1.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:37<03:47, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:39<12:53, 633kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:39<09:52, 825kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:39<07:06, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:41<06:49, 1.19MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:41<06:29, 1.25MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:41<04:57, 1.63MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<03:50, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:43<5:25:36, 24.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:43<3:47:28, 35.2kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:45<2:39:56, 49.8kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:45<1:53:35, 70.0kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:45<1:19:45, 99.6kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:45<55:41, 142kB/s]   .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:46<42:13, 187kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:47<30:21, 260kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:47<21:22, 368kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:48<16:41, 469kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:49<13:24, 583kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:49<09:47, 797kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:49<06:54, 1.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:50<17:02, 455kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:51<12:43, 609kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:51<09:04, 850kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:52<08:05, 949kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:53<06:30, 1.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:53<04:44, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:54<04:59, 1.52MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:55<04:19, 1.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:55<03:11, 2.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 410M/862M [02:56<03:54, 1.93MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:57<04:15, 1.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:57<03:18, 2.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<02:24, 3.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:58<05:15, 1.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:58<04:26, 1.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:59<03:18, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:00<03:58, 1.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:00<04:16, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:01<03:22, 2.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<02:26, 3.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<54:10, 135kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<38:40, 190kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:03<27:10, 269kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:04<20:32, 354kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:04<15:54, 457kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:05<11:30, 630kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:05<08:05, 890kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:06<16:30, 436kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:06<12:18, 584kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<08:44, 819kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:08<07:43, 923kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<06:54, 1.03MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<05:10, 1.37MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<03:40, 1.92MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<11:03, 639kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<08:29, 830kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:10<06:06, 1.15MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<05:48, 1.20MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<05:32, 1.26MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:12<04:10, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<03:01, 2.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<04:44, 1.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<04:02, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<02:58, 2.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<03:38, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<03:15, 2.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:16<02:25, 2.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:18<03:16, 2.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:18<03:44, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:18<02:58, 2.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:18<02:08, 3.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:20<15:52, 423kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:20<11:49, 567kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:20<08:26, 793kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:22<07:20, 905kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:22<05:52, 1.13MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:22<04:14, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:24<04:24, 1.49MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:24<03:46, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:24<02:48, 2.34MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:26<03:27, 1.88MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:26<03:07, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:26<02:19, 2.78MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:28<03:02, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:28<03:35, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<02:49, 2.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<02:03, 3.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:30<10:06, 630kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:30<07:46, 819kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<05:35, 1.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:32<05:17, 1.19MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:32<04:21, 1.45MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:32<03:10, 1.97MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:34<03:40, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:34<03:53, 1.60MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:34<02:59, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<02:12, 2.81MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:36<03:26, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:36<03:04, 2.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:36<02:18, 2.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:38<02:57, 2.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:38<03:22, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<02:41, 2.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<01:57, 3.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:40<10:33, 570kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:40<08:01, 750kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<05:45, 1.04MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:42<05:22, 1.11MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:42<05:01, 1.19MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<03:49, 1.55MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<02:44, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:44<11:10, 527kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:44<08:25, 698kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:44<06:00, 975kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:46<05:31, 1.05MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:46<05:05, 1.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:46<03:51, 1.50MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:46<02:44, 2.09MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:48<18:31, 310kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:48<13:34, 423kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<09:36, 595kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:50<07:56, 716kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:50<06:48, 834kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:50<05:03, 1.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<03:34, 1.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:52<09:32, 588kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:52<07:16, 770kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<05:13, 1.07MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:54<04:51, 1.14MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:54<04:34, 1.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:54<03:28, 1.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<02:28, 2.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:56<28:04, 195kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:56<20:12, 270kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<14:13, 382kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<10:11, 530kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:58<4:05:23, 22.0kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:58<2:51:41, 31.4kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<1:59:12, 44.9kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:59<1:28:00, 60.7kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:00<1:02:45, 85.0kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<44:04, 121kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:00<30:40, 172kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:01<24:56, 211kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<17:53, 294kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<12:36, 416kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:03<09:57, 523kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:04<08:06, 640kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:04<05:54, 878kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<04:10, 1.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:05<05:14, 980kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:06<04:13, 1.22MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:06<03:04, 1.66MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:07<03:15, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:08<03:23, 1.49MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:08<02:36, 1.94MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:08<01:52, 2.67MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:09<03:41, 1.35MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:10<03:02, 1.64MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<02:15, 2.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:11<02:39, 1.85MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<02:23, 2.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<01:47, 2.72MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:13<02:19, 2.09MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:14<02:09, 2.24MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<01:38, 2.95MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:15<02:11, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:15<02:32, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<01:59, 2.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<01:26, 3.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:17<07:19, 643kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<05:38, 835kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<04:03, 1.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:19<03:50, 1.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:19<03:40, 1.26MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:20<02:48, 1.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<01:59, 2.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<34:06, 134kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<24:20, 188kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<17:04, 266kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:23<12:51, 351kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:23<09:56, 453kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:24<07:11, 625kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<05:02, 883kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<10:19, 430kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<07:41, 576kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<05:28, 804kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:27<04:46, 916kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:27<03:47, 1.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:27<02:43, 1.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:29<02:54, 1.48MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:29<02:56, 1.46MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:29<02:16, 1.88MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:30<01:37, 2.60MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:31<13:21, 317kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<09:47, 431kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<06:54, 607kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:33<05:42, 729kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:33<04:54, 846kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:33<03:36, 1.15MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<02:34, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:35<03:24, 1.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:35<02:49, 1.45MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:35<02:04, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:37<02:19, 1.73MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:37<02:28, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<01:56, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:38<01:23, 2.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:39<29:13, 135kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:39<20:51, 189kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<14:35, 269kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<10:59, 353kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<08:28, 458kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:41<06:05, 636kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<04:15, 898kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:43<05:21, 713kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:43<04:08, 919kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:43<02:58, 1.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<02:54, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<02:51, 1.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<02:10, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<01:33, 2.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<02:56, 1.25MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<02:27, 1.50MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<01:48, 2.02MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<02:03, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<02:09, 1.67MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<01:41, 2.12MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<01:12, 2.91MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<29:44, 119kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<21:06, 167kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<14:45, 238kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:53<11:00, 315kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:53<08:27, 410kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<06:03, 569kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:53<04:14, 805kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:55<05:05, 669kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<03:54, 869kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<02:47, 1.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:57<02:42, 1.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<02:15, 1.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<01:38, 2.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:59<01:51, 1.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<02:01, 1.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<01:35, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<01:08, 2.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<05:49, 548kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<04:24, 722kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<03:08, 1.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:03<02:51, 1.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:03<02:39, 1.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<02:01, 1.54MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:03<01:26, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:05<05:40, 537kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:05<04:17, 709kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<03:03, 986kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<02:47, 1.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<02:16, 1.31MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<01:39, 1.78MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:09<01:47, 1.63MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<01:52, 1.56MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<01:26, 2.02MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<01:01, 2.78MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<02:32, 1.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<02:04, 1.37MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<01:30, 1.86MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<01:41, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<01:47, 1.55MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<01:23, 1.97MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<00:59, 2.72MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<04:57, 546kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<03:42, 728kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<02:41, 996kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:15<01:53, 1.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<04:13, 624kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<03:14, 813kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<02:18, 1.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<02:09, 1.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<02:03, 1.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:33, 1.63MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<01:06, 2.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<12:47, 195kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<09:11, 271kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<06:26, 383kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:21<04:35, 531kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<1:47:09, 22.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<1:14:47, 32.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:23<51:22, 46.3kB/s]  .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:24<37:52, 62.5kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:25<27:00, 87.6kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<18:56, 124kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<13:04, 177kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<10:10, 226kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<07:20, 312kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<05:08, 441kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:28<04:01, 553kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<03:17, 677kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<02:23, 927kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<01:40, 1.30MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:30<02:00, 1.08MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<01:37, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<01:10, 1.80MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:32<01:16, 1.64MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:33<01:19, 1.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:01, 2.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:33<00:43, 2.77MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:34<11:05, 182kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:35<07:57, 253kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:35<05:33, 358kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:36<04:14, 459kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:37<03:21, 579kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<02:26, 793kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:37<01:41, 1.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:38<11:11, 168kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:38<08:00, 234kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<05:35, 332kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:40<04:13, 429kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:40<03:19, 545kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<02:24, 748kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:41<01:39, 1.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:42<15:25, 113kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<10:56, 159kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<07:35, 226kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:44<05:33, 301kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:44<04:15, 393kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:45<03:02, 545kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:45<02:05, 771kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:46<03:52, 414kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:46<02:52, 556kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:47<02:01, 779kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:48<01:43, 893kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:48<01:31, 1.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:49<01:07, 1.34MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:49<00:47, 1.87MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:50<01:07, 1.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<00:56, 1.55MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:51<00:41, 2.08MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<00:46, 1.79MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:52<00:51, 1.63MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:52<00:39, 2.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:53<00:27, 2.87MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:54<01:55, 691kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:54<01:29, 891kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<01:03, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:56<00:59, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:56<00:58, 1.29MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:43, 1.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:57<00:30, 2.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:58<00:50, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:58<00:43, 1.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:31, 2.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<00:36, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<00:39, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:31, 2.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:01<00:21, 2.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<01:34, 669kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<01:12, 867kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:50, 1.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:47, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:46, 1.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:34, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:04<00:24, 2.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:37, 1.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<00:31, 1.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:23, 2.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:26, 1.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:28, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:22, 2.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:15, 3.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:25, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:22, 2.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:16, 2.75MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:20, 2.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:22, 1.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:17, 2.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:12<00:11, 3.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<04:40, 136kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<03:18, 190kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<02:14, 270kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<01:35, 355kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<01:14, 456kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:52, 632kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:34, 893kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:18<00:37, 785kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:28, 1.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<00:20, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:18, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:18, 1.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:13, 1.82MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:20<00:08, 2.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<02:40, 134kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<01:52, 188kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<01:13, 266kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:49, 351kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:35, 476kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:23, 669kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:26<00:16, 788kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:26<00:12, 1.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:08, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:04, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:28<00:24, 373kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:28<00:18, 479kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:12, 663kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:05, 936kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:30<00:07, 685kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:30<00:05, 885kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:02, 1.22MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:32<00:00, 1.26MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 1.29MB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<12:05:46,  9.19it/s]  0%|          | 639/400000 [00:00<8:27:32, 13.11it/s]  0%|          | 1308/400000 [00:00<5:54:59, 18.72it/s]  1%|          | 2036/400000 [00:00<4:08:18, 26.71it/s]  1%|          | 2741/400000 [00:00<2:53:47, 38.10it/s]  1%|          | 3481/400000 [00:00<2:01:41, 54.30it/s]  1%|          | 4223/400000 [00:00<1:25:17, 77.34it/s]  1%|          | 4974/400000 [00:00<59:51, 109.99it/s]   1%|▏         | 5684/400000 [00:00<42:06, 156.10it/s]  2%|▏         | 6367/400000 [00:01<29:43, 220.70it/s]  2%|▏         | 7043/400000 [00:01<21:03, 310.93it/s]  2%|▏         | 7767/400000 [00:01<14:59, 436.16it/s]  2%|▏         | 8510/400000 [00:01<10:44, 607.79it/s]  2%|▏         | 9257/400000 [00:01<07:45, 838.98it/s]  2%|▏         | 9976/400000 [00:01<05:41, 1141.45it/s]  3%|▎         | 10694/400000 [00:01<04:16, 1516.74it/s]  3%|▎         | 11392/400000 [00:01<03:17, 1972.29it/s]  3%|▎         | 12117/400000 [00:01<02:33, 2523.12it/s]  3%|▎         | 12856/400000 [00:01<02:03, 3144.23it/s]  3%|▎         | 13580/400000 [00:02<01:42, 3786.48it/s]  4%|▎         | 14304/400000 [00:02<01:27, 4418.36it/s]  4%|▍         | 15021/400000 [00:02<01:18, 4905.36it/s]  4%|▍         | 15721/400000 [00:02<01:12, 5332.03it/s]  4%|▍         | 16412/400000 [00:02<01:07, 5707.78it/s]  4%|▍         | 17101/400000 [00:02<01:04, 5921.29it/s]  4%|▍         | 17846/400000 [00:02<01:00, 6309.12it/s]  5%|▍         | 18543/400000 [00:02<00:58, 6467.23it/s]  5%|▍         | 19237/400000 [00:02<01:00, 6268.70it/s]  5%|▍         | 19899/400000 [00:02<01:00, 6286.15it/s]  5%|▌         | 20552/400000 [00:03<01:00, 6230.95it/s]  5%|▌         | 21243/400000 [00:03<00:58, 6420.22it/s]  5%|▌         | 21989/400000 [00:03<00:56, 6698.49it/s]  6%|▌         | 22725/400000 [00:03<00:54, 6881.80it/s]  6%|▌         | 23448/400000 [00:03<00:53, 6981.39it/s]  6%|▌         | 24179/400000 [00:03<00:53, 7075.52it/s]  6%|▌         | 24892/400000 [00:03<00:53, 7076.10it/s]  6%|▋         | 25621/400000 [00:03<00:52, 7138.38it/s]  7%|▋         | 26338/400000 [00:03<00:52, 7141.37it/s]  7%|▋         | 27074/400000 [00:03<00:51, 7203.58it/s]  7%|▋         | 27796/400000 [00:04<00:51, 7201.37it/s]  7%|▋         | 28539/400000 [00:04<00:51, 7267.82it/s]  7%|▋         | 29267/400000 [00:04<00:51, 7259.99it/s]  8%|▊         | 30006/400000 [00:04<00:50, 7298.26it/s]  8%|▊         | 30748/400000 [00:04<00:50, 7332.00it/s]  8%|▊         | 31482/400000 [00:04<00:50, 7316.25it/s]  8%|▊         | 32214/400000 [00:04<00:50, 7270.31it/s]  8%|▊         | 32958/400000 [00:04<00:50, 7318.86it/s]  8%|▊         | 33691/400000 [00:04<00:50, 7274.30it/s]  9%|▊         | 34419/400000 [00:05<00:51, 7061.76it/s]  9%|▉         | 35127/400000 [00:05<00:54, 6714.92it/s]  9%|▉         | 35858/400000 [00:05<00:52, 6881.27it/s]  9%|▉         | 36600/400000 [00:05<00:51, 7033.05it/s]  9%|▉         | 37349/400000 [00:05<00:50, 7162.17it/s] 10%|▉         | 38069/400000 [00:05<00:50, 7146.83it/s] 10%|▉         | 38811/400000 [00:05<00:49, 7224.58it/s] 10%|▉         | 39536/400000 [00:05<00:49, 7217.99it/s] 10%|█         | 40290/400000 [00:05<00:49, 7309.11it/s] 10%|█         | 41023/400000 [00:05<00:50, 7105.93it/s] 10%|█         | 41736/400000 [00:06<00:51, 6953.68it/s] 11%|█         | 42434/400000 [00:06<00:53, 6645.03it/s] 11%|█         | 43168/400000 [00:06<00:52, 6836.99it/s] 11%|█         | 43904/400000 [00:06<00:50, 6984.30it/s] 11%|█         | 44640/400000 [00:06<00:50, 7091.54it/s] 11%|█▏        | 45377/400000 [00:06<00:49, 7170.30it/s] 12%|█▏        | 46097/400000 [00:06<00:49, 7141.80it/s] 12%|█▏        | 46813/400000 [00:06<00:50, 7016.55it/s] 12%|█▏        | 47548/400000 [00:06<00:49, 7112.33it/s] 12%|█▏        | 48286/400000 [00:06<00:48, 7190.09it/s] 12%|█▏        | 49025/400000 [00:07<00:48, 7245.18it/s] 12%|█▏        | 49760/400000 [00:07<00:48, 7276.12it/s] 13%|█▎        | 50489/400000 [00:07<00:48, 7229.83it/s] 13%|█▎        | 51213/400000 [00:07<00:48, 7225.07it/s] 13%|█▎        | 51951/400000 [00:07<00:47, 7269.90it/s] 13%|█▎        | 52684/400000 [00:07<00:47, 7287.02it/s] 13%|█▎        | 53432/400000 [00:07<00:47, 7342.81it/s] 14%|█▎        | 54167/400000 [00:07<00:47, 7300.64it/s] 14%|█▎        | 54898/400000 [00:07<00:47, 7229.80it/s] 14%|█▍        | 55622/400000 [00:07<00:47, 7209.17it/s] 14%|█▍        | 56344/400000 [00:08<00:50, 6825.90it/s] 14%|█▍        | 57076/400000 [00:08<00:49, 6966.65it/s] 14%|█▍        | 57777/400000 [00:08<00:49, 6865.55it/s] 15%|█▍        | 58467/400000 [00:08<00:49, 6867.94it/s] 15%|█▍        | 59189/400000 [00:08<00:48, 6964.37it/s] 15%|█▍        | 59913/400000 [00:08<00:48, 7044.20it/s] 15%|█▌        | 60660/400000 [00:08<00:47, 7166.10it/s] 15%|█▌        | 61379/400000 [00:08<00:49, 6822.03it/s] 16%|█▌        | 62066/400000 [00:08<00:50, 6649.71it/s] 16%|█▌        | 62736/400000 [00:09<00:51, 6609.29it/s] 16%|█▌        | 63468/400000 [00:09<00:49, 6806.34it/s] 16%|█▌        | 64191/400000 [00:09<00:48, 6926.60it/s] 16%|█▌        | 64936/400000 [00:09<00:47, 7075.29it/s] 16%|█▋        | 65681/400000 [00:09<00:46, 7181.07it/s] 17%|█▋        | 66433/400000 [00:09<00:45, 7277.04it/s] 17%|█▋        | 67178/400000 [00:09<00:45, 7326.35it/s] 17%|█▋        | 67913/400000 [00:09<00:45, 7275.49it/s] 17%|█▋        | 68642/400000 [00:09<00:47, 7041.15it/s] 17%|█▋        | 69349/400000 [00:09<00:48, 6757.80it/s] 18%|█▊        | 70101/400000 [00:10<00:47, 6968.79it/s] 18%|█▊        | 70850/400000 [00:10<00:46, 7116.63it/s] 18%|█▊        | 71589/400000 [00:10<00:45, 7196.36it/s] 18%|█▊        | 72312/400000 [00:10<00:47, 6958.41it/s] 18%|█▊        | 73012/400000 [00:10<00:47, 6852.16it/s] 18%|█▊        | 73701/400000 [00:10<00:48, 6792.35it/s] 19%|█▊        | 74383/400000 [00:10<00:48, 6707.17it/s] 19%|█▉        | 75056/400000 [00:10<00:48, 6649.65it/s] 19%|█▉        | 75724/400000 [00:10<00:48, 6658.19it/s] 19%|█▉        | 76440/400000 [00:10<00:47, 6799.61it/s] 19%|█▉        | 77181/400000 [00:11<00:46, 6971.52it/s] 19%|█▉        | 77936/400000 [00:11<00:45, 7134.22it/s] 20%|█▉        | 78674/400000 [00:11<00:44, 7205.62it/s] 20%|█▉        | 79424/400000 [00:11<00:43, 7291.13it/s] 20%|██        | 80155/400000 [00:11<00:45, 7063.44it/s] 20%|██        | 80864/400000 [00:11<00:46, 6821.79it/s] 20%|██        | 81550/400000 [00:11<00:46, 6778.63it/s] 21%|██        | 82278/400000 [00:11<00:45, 6919.65it/s] 21%|██        | 83013/400000 [00:11<00:45, 7042.20it/s] 21%|██        | 83761/400000 [00:12<00:44, 7165.00it/s] 21%|██        | 84480/400000 [00:12<00:45, 6991.02it/s] 21%|██▏       | 85182/400000 [00:12<00:46, 6807.77it/s] 21%|██▏       | 85866/400000 [00:12<00:47, 6627.30it/s] 22%|██▏       | 86535/400000 [00:12<00:47, 6644.96it/s] 22%|██▏       | 87208/400000 [00:12<00:47, 6643.88it/s] 22%|██▏       | 87927/400000 [00:12<00:45, 6796.69it/s] 22%|██▏       | 88656/400000 [00:12<00:44, 6936.81it/s] 22%|██▏       | 89370/400000 [00:12<00:44, 6995.06it/s] 23%|██▎       | 90072/400000 [00:12<00:44, 6999.57it/s] 23%|██▎       | 90826/400000 [00:13<00:43, 7152.49it/s] 23%|██▎       | 91554/400000 [00:13<00:42, 7187.59it/s] 23%|██▎       | 92274/400000 [00:13<00:42, 7168.74it/s] 23%|██▎       | 92993/400000 [00:13<00:42, 7172.23it/s] 23%|██▎       | 93732/400000 [00:13<00:42, 7234.91it/s] 24%|██▎       | 94457/400000 [00:13<00:42, 7226.07it/s] 24%|██▍       | 95202/400000 [00:13<00:41, 7290.16it/s] 24%|██▍       | 95949/400000 [00:13<00:41, 7342.08it/s] 24%|██▍       | 96684/400000 [00:13<00:42, 7128.44it/s] 24%|██▍       | 97399/400000 [00:13<00:43, 6976.85it/s] 25%|██▍       | 98099/400000 [00:14<00:43, 6976.47it/s] 25%|██▍       | 98806/400000 [00:14<00:43, 7003.17it/s] 25%|██▍       | 99508/400000 [00:14<00:43, 6901.12it/s] 25%|██▌       | 100200/400000 [00:14<00:43, 6894.32it/s] 25%|██▌       | 100938/400000 [00:14<00:42, 7030.76it/s] 25%|██▌       | 101686/400000 [00:14<00:41, 7159.09it/s] 26%|██▌       | 102431/400000 [00:14<00:41, 7241.27it/s] 26%|██▌       | 103157/400000 [00:14<00:41, 7198.84it/s] 26%|██▌       | 103898/400000 [00:14<00:40, 7260.55it/s] 26%|██▌       | 104628/400000 [00:14<00:40, 7264.57it/s] 26%|██▋       | 105355/400000 [00:15<00:42, 6954.91it/s] 27%|██▋       | 106054/400000 [00:15<00:42, 6875.82it/s] 27%|██▋       | 106745/400000 [00:15<00:43, 6803.13it/s] 27%|██▋       | 107429/400000 [00:15<00:42, 6811.84it/s] 27%|██▋       | 108180/400000 [00:15<00:41, 7005.29it/s] 27%|██▋       | 108927/400000 [00:15<00:40, 7137.76it/s] 27%|██▋       | 109677/400000 [00:15<00:40, 7240.60it/s] 28%|██▊       | 110426/400000 [00:15<00:39, 7312.34it/s] 28%|██▊       | 111159/400000 [00:15<00:39, 7245.11it/s] 28%|██▊       | 111903/400000 [00:15<00:39, 7299.76it/s] 28%|██▊       | 112650/400000 [00:16<00:39, 7348.97it/s] 28%|██▊       | 113401/400000 [00:16<00:38, 7394.29it/s] 29%|██▊       | 114149/400000 [00:16<00:38, 7410.20it/s] 29%|██▊       | 114891/400000 [00:16<00:38, 7374.44it/s] 29%|██▉       | 115629/400000 [00:16<00:39, 7152.84it/s] 29%|██▉       | 116370/400000 [00:16<00:39, 7227.26it/s] 29%|██▉       | 117117/400000 [00:16<00:38, 7298.01it/s] 29%|██▉       | 117858/400000 [00:16<00:38, 7330.97it/s] 30%|██▉       | 118592/400000 [00:16<00:39, 7212.44it/s] 30%|██▉       | 119315/400000 [00:17<00:39, 7023.59it/s] 30%|███       | 120020/400000 [00:17<00:40, 6849.75it/s] 30%|███       | 120708/400000 [00:17<00:41, 6810.92it/s] 30%|███       | 121391/400000 [00:17<00:41, 6756.63it/s] 31%|███       | 122068/400000 [00:17<00:41, 6699.98it/s] 31%|███       | 122747/400000 [00:17<00:41, 6726.51it/s] 31%|███       | 123443/400000 [00:17<00:40, 6793.13it/s] 31%|███       | 124152/400000 [00:17<00:40, 6876.90it/s] 31%|███       | 124909/400000 [00:17<00:38, 7070.26it/s] 31%|███▏      | 125618/400000 [00:17<00:39, 6953.60it/s] 32%|███▏      | 126357/400000 [00:18<00:38, 7078.24it/s] 32%|███▏      | 127113/400000 [00:18<00:37, 7214.18it/s] 32%|███▏      | 127848/400000 [00:18<00:37, 7253.30it/s] 32%|███▏      | 128575/400000 [00:18<00:37, 7226.08it/s] 32%|███▏      | 129299/400000 [00:18<00:39, 6936.95it/s] 32%|███▏      | 129996/400000 [00:18<00:39, 6765.39it/s] 33%|███▎      | 130676/400000 [00:18<00:40, 6725.18it/s] 33%|███▎      | 131351/400000 [00:18<00:39, 6725.87it/s] 33%|███▎      | 132044/400000 [00:18<00:39, 6784.41it/s] 33%|███▎      | 132754/400000 [00:18<00:38, 6875.79it/s] 33%|███▎      | 133509/400000 [00:19<00:37, 7049.50it/s] 34%|███▎      | 134216/400000 [00:19<00:37, 7051.66it/s] 34%|███▎      | 134963/400000 [00:19<00:36, 7170.84it/s] 34%|███▍      | 135710/400000 [00:19<00:36, 7257.22it/s] 34%|███▍      | 136437/400000 [00:19<00:36, 7160.35it/s] 34%|███▍      | 137155/400000 [00:19<00:36, 7155.65it/s] 34%|███▍      | 137904/400000 [00:19<00:36, 7252.35it/s] 35%|███▍      | 138651/400000 [00:19<00:35, 7315.04it/s] 35%|███▍      | 139398/400000 [00:19<00:35, 7358.05it/s] 35%|███▌      | 140135/400000 [00:19<00:35, 7291.72it/s] 35%|███▌      | 140865/400000 [00:20<00:36, 7158.59it/s] 35%|███▌      | 141593/400000 [00:20<00:35, 7192.46it/s] 36%|███▌      | 142342/400000 [00:20<00:35, 7277.50it/s] 36%|███▌      | 143090/400000 [00:20<00:35, 7335.12it/s] 36%|███▌      | 143825/400000 [00:20<00:35, 7308.69it/s] 36%|███▌      | 144574/400000 [00:20<00:34, 7360.14it/s] 36%|███▋      | 145324/400000 [00:20<00:34, 7401.03it/s] 37%|███▋      | 146065/400000 [00:20<00:34, 7335.71it/s] 37%|███▋      | 146818/400000 [00:20<00:34, 7391.79it/s] 37%|███▋      | 147558/400000 [00:20<00:34, 7374.70it/s] 37%|███▋      | 148296/400000 [00:21<00:34, 7289.74it/s] 37%|███▋      | 149026/400000 [00:21<00:34, 7270.64it/s] 37%|███▋      | 149770/400000 [00:21<00:34, 7320.31it/s] 38%|███▊      | 150504/400000 [00:21<00:34, 7325.10it/s] 38%|███▊      | 151237/400000 [00:21<00:34, 7230.67it/s] 38%|███▊      | 151961/400000 [00:21<00:34, 7092.46it/s] 38%|███▊      | 152672/400000 [00:21<00:35, 6977.68it/s] 38%|███▊      | 153371/400000 [00:21<00:35, 6910.82it/s] 39%|███▊      | 154063/400000 [00:21<00:36, 6803.91it/s] 39%|███▊      | 154782/400000 [00:22<00:35, 6914.66it/s] 39%|███▉      | 155541/400000 [00:22<00:34, 7104.13it/s] 39%|███▉      | 156304/400000 [00:22<00:33, 7253.44it/s] 39%|███▉      | 157032/400000 [00:22<00:33, 7148.64it/s] 39%|███▉      | 157749/400000 [00:22<00:34, 6945.30it/s] 40%|███▉      | 158447/400000 [00:22<00:35, 6781.04it/s] 40%|███▉      | 159128/400000 [00:22<00:35, 6724.01it/s] 40%|███▉      | 159803/400000 [00:22<00:35, 6717.86it/s] 40%|████      | 160477/400000 [00:22<00:36, 6540.09it/s] 40%|████      | 161201/400000 [00:22<00:35, 6733.29it/s] 40%|████      | 161945/400000 [00:23<00:34, 6929.85it/s] 41%|████      | 162696/400000 [00:23<00:33, 7092.40it/s] 41%|████      | 163423/400000 [00:23<00:33, 7142.12it/s] 41%|████      | 164186/400000 [00:23<00:32, 7281.51it/s] 41%|████      | 164939/400000 [00:23<00:31, 7351.86it/s] 41%|████▏     | 165676/400000 [00:23<00:31, 7337.31it/s] 42%|████▏     | 166436/400000 [00:23<00:31, 7411.80it/s] 42%|████▏     | 167179/400000 [00:23<00:33, 6990.18it/s] 42%|████▏     | 167932/400000 [00:23<00:32, 7141.56it/s] 42%|████▏     | 168663/400000 [00:23<00:32, 7190.60it/s] 42%|████▏     | 169412/400000 [00:24<00:31, 7277.27it/s] 43%|████▎     | 170165/400000 [00:24<00:31, 7349.42it/s] 43%|████▎     | 170919/400000 [00:24<00:30, 7404.26it/s] 43%|████▎     | 171661/400000 [00:24<00:31, 7319.97it/s] 43%|████▎     | 172395/400000 [00:24<00:31, 7168.49it/s] 43%|████▎     | 173114/400000 [00:24<00:32, 7003.41it/s] 43%|████▎     | 173817/400000 [00:24<00:32, 6921.10it/s] 44%|████▎     | 174511/400000 [00:24<00:32, 6861.97it/s] 44%|████▍     | 175199/400000 [00:24<00:32, 6839.43it/s] 44%|████▍     | 175884/400000 [00:25<00:33, 6755.50it/s] 44%|████▍     | 176640/400000 [00:25<00:32, 6976.74it/s] 44%|████▍     | 177387/400000 [00:25<00:31, 7115.43it/s] 45%|████▍     | 178144/400000 [00:25<00:30, 7245.79it/s] 45%|████▍     | 178898/400000 [00:25<00:30, 7328.96it/s] 45%|████▍     | 179635/400000 [00:25<00:30, 7340.49it/s] 45%|████▌     | 180371/400000 [00:25<00:29, 7324.80it/s] 45%|████▌     | 181111/400000 [00:25<00:29, 7345.65it/s] 45%|████▌     | 181869/400000 [00:25<00:29, 7413.63it/s] 46%|████▌     | 182621/400000 [00:25<00:29, 7440.68it/s] 46%|████▌     | 183366/400000 [00:26<00:30, 7095.99it/s] 46%|████▌     | 184092/400000 [00:26<00:30, 7142.09it/s] 46%|████▌     | 184843/400000 [00:26<00:29, 7239.01it/s] 46%|████▋     | 185597/400000 [00:26<00:29, 7323.60it/s] 47%|████▋     | 186351/400000 [00:26<00:28, 7385.55it/s] 47%|████▋     | 187098/400000 [00:26<00:28, 7410.42it/s] 47%|████▋     | 187841/400000 [00:26<00:28, 7334.19it/s] 47%|████▋     | 188576/400000 [00:26<00:28, 7320.17it/s] 47%|████▋     | 189328/400000 [00:26<00:28, 7378.07it/s] 48%|████▊     | 190080/400000 [00:26<00:28, 7418.67it/s] 48%|████▊     | 190823/400000 [00:27<00:28, 7411.19it/s] 48%|████▊     | 191565/400000 [00:27<00:28, 7200.36it/s] 48%|████▊     | 192287/400000 [00:27<00:29, 7078.13it/s] 48%|████▊     | 192997/400000 [00:27<00:29, 7062.01it/s] 48%|████▊     | 193734/400000 [00:27<00:28, 7151.65it/s] 49%|████▊     | 194451/400000 [00:27<00:28, 7101.02it/s] 49%|████▉     | 195162/400000 [00:27<00:30, 6782.30it/s] 49%|████▉     | 195850/400000 [00:27<00:29, 6811.13it/s] 49%|████▉     | 196595/400000 [00:27<00:29, 6989.44it/s] 49%|████▉     | 197297/400000 [00:27<00:29, 6834.87it/s] 49%|████▉     | 197984/400000 [00:28<00:30, 6726.08it/s] 50%|████▉     | 198659/400000 [00:28<00:31, 6473.25it/s] 50%|████▉     | 199415/400000 [00:28<00:29, 6762.67it/s] 50%|█████     | 200174/400000 [00:28<00:28, 6990.73it/s] 50%|█████     | 200922/400000 [00:28<00:27, 7130.18it/s] 50%|█████     | 201640/400000 [00:28<00:28, 6888.15it/s] 51%|█████     | 202380/400000 [00:28<00:28, 7031.47it/s] 51%|█████     | 203088/400000 [00:28<00:28, 6998.55it/s] 51%|█████     | 203791/400000 [00:28<00:28, 6833.61it/s] 51%|█████     | 204478/400000 [00:29<00:28, 6805.00it/s] 51%|█████▏    | 205201/400000 [00:29<00:28, 6924.91it/s] 51%|█████▏    | 205928/400000 [00:29<00:27, 7024.47it/s] 52%|█████▏    | 206646/400000 [00:29<00:27, 7069.35it/s] 52%|█████▏    | 207355/400000 [00:29<00:31, 6184.62it/s] 52%|█████▏    | 208071/400000 [00:29<00:29, 6446.70it/s] 52%|█████▏    | 208734/400000 [00:29<00:29, 6482.58it/s] 52%|█████▏    | 209399/400000 [00:29<00:29, 6529.36it/s] 53%|█████▎    | 210079/400000 [00:29<00:28, 6605.88it/s] 53%|█████▎    | 210813/400000 [00:29<00:27, 6808.38it/s] 53%|█████▎    | 211500/400000 [00:30<00:27, 6820.33it/s] 53%|█████▎    | 212240/400000 [00:30<00:26, 6984.17it/s] 53%|█████▎    | 212989/400000 [00:30<00:26, 7128.17it/s] 53%|█████▎    | 213744/400000 [00:30<00:25, 7247.16it/s] 54%|█████▎    | 214472/400000 [00:30<00:25, 7189.93it/s] 54%|█████▍    | 215194/400000 [00:30<00:25, 7168.50it/s] 54%|█████▍    | 215931/400000 [00:30<00:25, 7225.27it/s] 54%|█████▍    | 216677/400000 [00:30<00:25, 7292.24it/s] 54%|█████▍    | 217408/400000 [00:30<00:25, 7283.67it/s] 55%|█████▍    | 218138/400000 [00:30<00:25, 7170.49it/s] 55%|█████▍    | 218856/400000 [00:31<00:25, 7159.66it/s] 55%|█████▍    | 219573/400000 [00:31<00:25, 7084.71it/s] 55%|█████▌    | 220283/400000 [00:31<00:25, 6987.66it/s] 55%|█████▌    | 220983/400000 [00:31<00:25, 6909.54it/s] 55%|█████▌    | 221720/400000 [00:31<00:25, 7038.42it/s] 56%|█████▌    | 222458/400000 [00:31<00:24, 7135.52it/s] 56%|█████▌    | 223173/400000 [00:31<00:25, 7044.65it/s] 56%|█████▌    | 223919/400000 [00:31<00:24, 7161.86it/s] 56%|█████▌    | 224672/400000 [00:31<00:24, 7267.28it/s] 56%|█████▋    | 225419/400000 [00:32<00:23, 7324.38it/s] 57%|█████▋    | 226153/400000 [00:32<00:24, 7133.31it/s] 57%|█████▋    | 226869/400000 [00:32<00:24, 6925.45it/s] 57%|█████▋    | 227565/400000 [00:32<00:25, 6760.84it/s] 57%|█████▋    | 228244/400000 [00:32<00:25, 6739.51it/s] 57%|█████▋    | 228920/400000 [00:32<00:25, 6663.75it/s] 57%|█████▋    | 229636/400000 [00:32<00:25, 6804.46it/s] 58%|█████▊    | 230387/400000 [00:32<00:24, 7000.68it/s] 58%|█████▊    | 231132/400000 [00:32<00:23, 7127.11it/s] 58%|█████▊    | 231848/400000 [00:32<00:23, 7052.74it/s] 58%|█████▊    | 232597/400000 [00:33<00:23, 7177.49it/s] 58%|█████▊    | 233335/400000 [00:33<00:23, 7235.38it/s] 59%|█████▊    | 234087/400000 [00:33<00:22, 7316.08it/s] 59%|█████▊    | 234842/400000 [00:33<00:22, 7382.42it/s] 59%|█████▉    | 235582/400000 [00:33<00:23, 7112.42it/s] 59%|█████▉    | 236297/400000 [00:33<00:23, 6929.82it/s] 59%|█████▉    | 236993/400000 [00:33<00:24, 6727.27it/s] 59%|█████▉    | 237738/400000 [00:33<00:23, 6928.17it/s] 60%|█████▉    | 238489/400000 [00:33<00:22, 7091.70it/s] 60%|█████▉    | 239202/400000 [00:33<00:22, 7059.49it/s] 60%|█████▉    | 239911/400000 [00:34<00:22, 7014.71it/s] 60%|██████    | 240623/400000 [00:34<00:22, 7045.66it/s] 60%|██████    | 241364/400000 [00:34<00:22, 7150.72it/s] 61%|██████    | 242115/400000 [00:34<00:21, 7254.40it/s] 61%|██████    | 242871/400000 [00:34<00:21, 7341.63it/s] 61%|██████    | 243607/400000 [00:34<00:21, 7268.48it/s] 61%|██████    | 244335/400000 [00:34<00:21, 7196.90it/s] 61%|██████▏   | 245073/400000 [00:34<00:21, 7249.38it/s] 61%|██████▏   | 245823/400000 [00:34<00:21, 7322.17it/s] 62%|██████▏   | 246577/400000 [00:34<00:20, 7385.42it/s] 62%|██████▏   | 247317/400000 [00:35<00:21, 7242.35it/s] 62%|██████▏   | 248057/400000 [00:35<00:20, 7286.18it/s] 62%|██████▏   | 248787/400000 [00:35<00:20, 7280.32it/s] 62%|██████▏   | 249535/400000 [00:35<00:20, 7337.92it/s] 63%|██████▎   | 250287/400000 [00:35<00:20, 7391.35it/s] 63%|██████▎   | 251044/400000 [00:35<00:20, 7443.67it/s] 63%|██████▎   | 251789/400000 [00:35<00:19, 7417.95it/s] 63%|██████▎   | 252532/400000 [00:35<00:20, 7183.86it/s] 63%|██████▎   | 253256/400000 [00:35<00:20, 7199.69it/s] 63%|██████▎   | 253978/400000 [00:36<00:20, 7200.92it/s] 64%|██████▎   | 254730/400000 [00:36<00:19, 7292.75it/s] 64%|██████▍   | 255461/400000 [00:36<00:19, 7246.77it/s] 64%|██████▍   | 256187/400000 [00:36<00:20, 7103.29it/s] 64%|██████▍   | 256899/400000 [00:36<00:20, 7106.33it/s] 64%|██████▍   | 257657/400000 [00:36<00:19, 7241.87it/s] 65%|██████▍   | 258415/400000 [00:36<00:19, 7338.12it/s] 65%|██████▍   | 259150/400000 [00:36<00:19, 7048.77it/s] 65%|██████▍   | 259898/400000 [00:36<00:19, 7170.63it/s] 65%|██████▌   | 260645/400000 [00:36<00:19, 7257.64it/s] 65%|██████▌   | 261373/400000 [00:37<00:19, 7209.35it/s] 66%|██████▌   | 262127/400000 [00:37<00:18, 7303.65it/s] 66%|██████▌   | 262859/400000 [00:37<00:18, 7228.64it/s] 66%|██████▌   | 263584/400000 [00:37<00:19, 7057.03it/s] 66%|██████▌   | 264292/400000 [00:37<00:20, 6607.85it/s] 66%|██████▋   | 265044/400000 [00:37<00:19, 6853.49it/s] 66%|██████▋   | 265757/400000 [00:37<00:19, 6934.15it/s] 67%|██████▋   | 266456/400000 [00:37<00:19, 6944.31it/s] 67%|██████▋   | 267210/400000 [00:37<00:18, 7110.61it/s] 67%|██████▋   | 267961/400000 [00:37<00:18, 7223.30it/s] 67%|██████▋   | 268692/400000 [00:38<00:18, 7246.80it/s] 67%|██████▋   | 269444/400000 [00:38<00:17, 7323.92it/s] 68%|██████▊   | 270179/400000 [00:38<00:17, 7277.92it/s] 68%|██████▊   | 270936/400000 [00:38<00:17, 7361.23it/s] 68%|██████▊   | 271686/400000 [00:38<00:17, 7399.58it/s] 68%|██████▊   | 272437/400000 [00:38<00:17, 7430.07it/s] 68%|██████▊   | 273187/400000 [00:38<00:17, 7449.62it/s] 68%|██████▊   | 273933/400000 [00:38<00:17, 7399.51it/s] 69%|██████▊   | 274674/400000 [00:38<00:16, 7400.21it/s] 69%|██████▉   | 275425/400000 [00:38<00:16, 7431.38it/s] 69%|██████▉   | 276171/400000 [00:39<00:16, 7437.19it/s] 69%|██████▉   | 276919/400000 [00:39<00:16, 7448.43it/s] 69%|██████▉   | 277664/400000 [00:39<00:16, 7207.19it/s] 70%|██████▉   | 278387/400000 [00:39<00:17, 6959.65it/s] 70%|██████▉   | 279087/400000 [00:39<00:17, 6900.78it/s] 70%|██████▉   | 279843/400000 [00:39<00:16, 7084.03it/s] 70%|███████   | 280555/400000 [00:39<00:16, 7088.60it/s] 70%|███████   | 281266/400000 [00:39<00:16, 7080.09it/s] 71%|███████   | 282023/400000 [00:39<00:16, 7217.59it/s] 71%|███████   | 282747/400000 [00:40<00:16, 7193.38it/s] 71%|███████   | 283477/400000 [00:40<00:16, 7224.60it/s] 71%|███████   | 284206/400000 [00:40<00:15, 7242.89it/s] 71%|███████   | 284931/400000 [00:40<00:15, 7242.64it/s] 71%|███████▏  | 285690/400000 [00:40<00:15, 7339.16it/s] 72%|███████▏  | 286425/400000 [00:40<00:15, 7210.66it/s] 72%|███████▏  | 287147/400000 [00:40<00:16, 6911.40it/s] 72%|███████▏  | 287842/400000 [00:40<00:16, 6826.88it/s] 72%|███████▏  | 288528/400000 [00:40<00:16, 6735.43it/s] 72%|███████▏  | 289204/400000 [00:40<00:16, 6718.66it/s] 72%|███████▏  | 289878/400000 [00:41<00:16, 6705.67it/s] 73%|███████▎  | 290550/400000 [00:41<00:16, 6563.08it/s] 73%|███████▎  | 291208/400000 [00:41<00:16, 6519.40it/s] 73%|███████▎  | 291861/400000 [00:41<00:16, 6472.56it/s] 73%|███████▎  | 292520/400000 [00:41<00:16, 6507.24it/s] 73%|███████▎  | 293269/400000 [00:41<00:15, 6771.83it/s] 74%|███████▎  | 294017/400000 [00:41<00:15, 6968.27it/s] 74%|███████▎  | 294718/400000 [00:41<00:15, 6894.37it/s] 74%|███████▍  | 295411/400000 [00:41<00:15, 6754.15it/s] 74%|███████▍  | 296149/400000 [00:41<00:14, 6928.59it/s] 74%|███████▍  | 296907/400000 [00:42<00:14, 7109.55it/s] 74%|███████▍  | 297654/400000 [00:42<00:14, 7205.57it/s] 75%|███████▍  | 298378/400000 [00:42<00:14, 7051.48it/s] 75%|███████▍  | 299086/400000 [00:42<00:14, 6892.38it/s] 75%|███████▍  | 299778/400000 [00:42<00:14, 6871.77it/s] 75%|███████▌  | 300532/400000 [00:42<00:14, 7057.06it/s] 75%|███████▌  | 301286/400000 [00:42<00:13, 7193.29it/s] 76%|███████▌  | 302024/400000 [00:42<00:13, 7246.71it/s] 76%|███████▌  | 302772/400000 [00:42<00:13, 7313.01it/s] 76%|███████▌  | 303518/400000 [00:42<00:13, 7356.10it/s] 76%|███████▌  | 304255/400000 [00:43<00:13, 7306.08it/s] 76%|███████▌  | 304987/400000 [00:43<00:13, 7098.18it/s] 76%|███████▋  | 305699/400000 [00:43<00:13, 6979.58it/s] 77%|███████▋  | 306450/400000 [00:43<00:13, 7130.30it/s] 77%|███████▋  | 307166/400000 [00:43<00:13, 6926.88it/s] 77%|███████▋  | 307862/400000 [00:43<00:13, 6858.78it/s] 77%|███████▋  | 308550/400000 [00:43<00:13, 6741.83it/s] 77%|███████▋  | 309227/400000 [00:43<00:13, 6571.69it/s] 77%|███████▋  | 309972/400000 [00:43<00:13, 6810.86it/s] 78%|███████▊  | 310700/400000 [00:44<00:12, 6943.32it/s] 78%|███████▊  | 311437/400000 [00:44<00:12, 7065.87it/s] 78%|███████▊  | 312191/400000 [00:44<00:12, 7200.62it/s] 78%|███████▊  | 312914/400000 [00:44<00:12, 7173.58it/s] 78%|███████▊  | 313644/400000 [00:44<00:11, 7209.31it/s] 79%|███████▊  | 314391/400000 [00:44<00:11, 7285.00it/s] 79%|███████▉  | 315130/400000 [00:44<00:11, 7315.40it/s] 79%|███████▉  | 315883/400000 [00:44<00:11, 7376.65it/s] 79%|███████▉  | 316622/400000 [00:44<00:11, 7257.94it/s] 79%|███████▉  | 317349/400000 [00:44<00:11, 7002.40it/s] 80%|███████▉  | 318052/400000 [00:45<00:11, 6924.81it/s] 80%|███████▉  | 318747/400000 [00:45<00:11, 6867.60it/s] 80%|███████▉  | 319436/400000 [00:45<00:11, 6816.61it/s] 80%|████████  | 320119/400000 [00:45<00:11, 6773.49it/s] 80%|████████  | 320798/400000 [00:45<00:11, 6736.16it/s] 80%|████████  | 321473/400000 [00:45<00:11, 6650.37it/s] 81%|████████  | 322146/400000 [00:45<00:11, 6673.38it/s] 81%|████████  | 322831/400000 [00:45<00:11, 6723.01it/s] 81%|████████  | 323556/400000 [00:45<00:11, 6872.86it/s] 81%|████████  | 324313/400000 [00:45<00:10, 7066.85it/s] 81%|████████▏ | 325059/400000 [00:46<00:10, 7178.08it/s] 81%|████████▏ | 325789/400000 [00:46<00:10, 7211.79it/s] 82%|████████▏ | 326544/400000 [00:46<00:10, 7307.64it/s] 82%|████████▏ | 327289/400000 [00:46<00:09, 7348.54it/s] 82%|████████▏ | 328053/400000 [00:46<00:09, 7432.24it/s] 82%|████████▏ | 328821/400000 [00:46<00:09, 7503.34it/s] 82%|████████▏ | 329573/400000 [00:46<00:09, 7400.06it/s] 83%|████████▎ | 330333/400000 [00:46<00:09, 7456.79it/s] 83%|████████▎ | 331080/400000 [00:46<00:09, 7264.87it/s] 83%|████████▎ | 331809/400000 [00:47<00:09, 7105.46it/s] 83%|████████▎ | 332522/400000 [00:47<00:09, 6898.34it/s] 83%|████████▎ | 333215/400000 [00:47<00:10, 6668.77it/s] 83%|████████▎ | 333886/400000 [00:47<00:09, 6662.61it/s] 84%|████████▎ | 334555/400000 [00:47<00:09, 6615.71it/s] 84%|████████▍ | 335233/400000 [00:47<00:09, 6662.58it/s] 84%|████████▍ | 335910/400000 [00:47<00:09, 6686.74it/s] 84%|████████▍ | 336628/400000 [00:47<00:09, 6825.60it/s] 84%|████████▍ | 337378/400000 [00:47<00:08, 7013.38it/s] 85%|████████▍ | 338082/400000 [00:47<00:08, 6973.32it/s] 85%|████████▍ | 338828/400000 [00:48<00:08, 7112.13it/s] 85%|████████▍ | 339573/400000 [00:48<00:08, 7208.37it/s] 85%|████████▌ | 340327/400000 [00:48<00:08, 7301.95it/s] 85%|████████▌ | 341059/400000 [00:48<00:08, 7203.41it/s] 85%|████████▌ | 341789/400000 [00:48<00:08, 7229.58it/s] 86%|████████▌ | 342515/400000 [00:48<00:07, 7238.19it/s] 86%|████████▌ | 343243/400000 [00:48<00:07, 7248.96it/s] 86%|████████▌ | 343980/400000 [00:48<00:07, 7284.39it/s] 86%|████████▌ | 344717/400000 [00:48<00:07, 7308.31it/s] 86%|████████▋ | 345449/400000 [00:48<00:07, 7043.47it/s] 87%|████████▋ | 346156/400000 [00:49<00:07, 6961.48it/s] 87%|████████▋ | 346854/400000 [00:49<00:07, 6874.70it/s] 87%|████████▋ | 347599/400000 [00:49<00:07, 7037.20it/s] 87%|████████▋ | 348343/400000 [00:49<00:07, 7153.06it/s] 87%|████████▋ | 349061/400000 [00:49<00:07, 7094.37it/s] 87%|████████▋ | 349806/400000 [00:49<00:06, 7196.06it/s] 88%|████████▊ | 350533/400000 [00:49<00:06, 7150.80it/s] 88%|████████▊ | 351250/400000 [00:49<00:06, 7149.17it/s] 88%|████████▊ | 351997/400000 [00:49<00:06, 7240.77it/s] 88%|████████▊ | 352725/400000 [00:49<00:06, 7251.54it/s] 88%|████████▊ | 353477/400000 [00:50<00:06, 7327.44it/s] 89%|████████▊ | 354211/400000 [00:50<00:06, 7263.98it/s] 89%|████████▊ | 354938/400000 [00:50<00:06, 7247.04it/s] 89%|████████▉ | 355664/400000 [00:50<00:06, 7163.11it/s] 89%|████████▉ | 356382/400000 [00:50<00:06, 7166.38it/s] 89%|████████▉ | 357126/400000 [00:50<00:05, 7246.06it/s] 89%|████████▉ | 357879/400000 [00:50<00:05, 7326.55it/s] 90%|████████▉ | 358613/400000 [00:50<00:05, 7298.95it/s] 90%|████████▉ | 359347/400000 [00:50<00:05, 7310.58it/s] 90%|█████████ | 360079/400000 [00:50<00:05, 7258.83it/s] 90%|█████████ | 360828/400000 [00:51<00:05, 7326.54it/s] 90%|█████████ | 361562/400000 [00:51<00:05, 7236.57it/s] 91%|█████████ | 362318/400000 [00:51<00:05, 7330.26it/s] 91%|█████████ | 363070/400000 [00:51<00:05, 7385.92it/s] 91%|█████████ | 363810/400000 [00:51<00:05, 7222.76it/s] 91%|█████████ | 364554/400000 [00:51<00:04, 7286.05it/s] 91%|█████████▏| 365303/400000 [00:51<00:04, 7344.51it/s] 92%|█████████▏| 366054/400000 [00:51<00:04, 7392.33it/s] 92%|█████████▏| 366797/400000 [00:51<00:04, 7402.18it/s] 92%|█████████▏| 367538/400000 [00:51<00:04, 7169.29it/s] 92%|█████████▏| 368257/400000 [00:52<00:04, 6940.87it/s] 92%|█████████▏| 368954/400000 [00:52<00:04, 6918.35it/s] 92%|█████████▏| 369693/400000 [00:52<00:04, 7052.06it/s] 93%|█████████▎| 370418/400000 [00:52<00:04, 7107.68it/s] 93%|█████████▎| 371152/400000 [00:52<00:04, 7173.39it/s] 93%|█████████▎| 371871/400000 [00:52<00:04, 7016.72it/s] 93%|█████████▎| 372575/400000 [00:52<00:03, 6860.47it/s] 93%|█████████▎| 373263/400000 [00:52<00:03, 6786.36it/s] 93%|█████████▎| 373944/400000 [00:52<00:03, 6762.02it/s] 94%|█████████▎| 374664/400000 [00:53<00:03, 6887.13it/s] 94%|█████████▍| 375414/400000 [00:53<00:03, 7057.64it/s] 94%|█████████▍| 376162/400000 [00:53<00:03, 7177.88it/s] 94%|█████████▍| 376883/400000 [00:53<00:03, 7185.60it/s] 94%|█████████▍| 377638/400000 [00:53<00:03, 7289.93it/s] 95%|█████████▍| 378371/400000 [00:53<00:02, 7298.91it/s] 95%|█████████▍| 379102/400000 [00:53<00:03, 6956.69it/s] 95%|█████████▍| 379811/400000 [00:53<00:02, 6993.60it/s] 95%|█████████▌| 380531/400000 [00:53<00:02, 7054.24it/s] 95%|█████████▌| 381278/400000 [00:53<00:02, 7171.51it/s] 95%|█████████▌| 381998/400000 [00:54<00:02, 7001.79it/s] 96%|█████████▌| 382701/400000 [00:54<00:02, 6908.16it/s] 96%|█████████▌| 383394/400000 [00:54<00:02, 6823.36it/s] 96%|█████████▌| 384078/400000 [00:54<00:02, 6803.17it/s] 96%|█████████▌| 384763/400000 [00:54<00:02, 6814.79it/s] 96%|█████████▋| 385505/400000 [00:54<00:02, 6983.76it/s] 97%|█████████▋| 386257/400000 [00:54<00:01, 7135.83it/s] 97%|█████████▋| 386973/400000 [00:54<00:01, 7013.56it/s] 97%|█████████▋| 387677/400000 [00:54<00:01, 6888.86it/s] 97%|█████████▋| 388396/400000 [00:54<00:01, 6975.78it/s] 97%|█████████▋| 389116/400000 [00:55<00:01, 7040.29it/s] 97%|█████████▋| 389851/400000 [00:55<00:01, 7128.76it/s] 98%|█████████▊| 390601/400000 [00:55<00:01, 7234.00it/s] 98%|█████████▊| 391326/400000 [00:55<00:01, 7092.83it/s] 98%|█████████▊| 392037/400000 [00:55<00:01, 7082.68it/s] 98%|█████████▊| 392747/400000 [00:55<00:01, 7016.49it/s] 98%|█████████▊| 393450/400000 [00:55<00:00, 6818.59it/s] 99%|█████████▊| 394134/400000 [00:55<00:00, 6741.25it/s] 99%|█████████▊| 394836/400000 [00:55<00:00, 6821.77it/s] 99%|█████████▉| 395573/400000 [00:56<00:00, 6976.46it/s] 99%|█████████▉| 396308/400000 [00:56<00:00, 7082.76it/s] 99%|█████████▉| 397024/400000 [00:56<00:00, 7104.06it/s] 99%|█████████▉| 397736/400000 [00:56<00:00, 7076.24it/s]100%|█████████▉| 398488/400000 [00:56<00:00, 7203.54it/s]100%|█████████▉| 399230/400000 [00:56<00:00, 7265.69it/s]100%|█████████▉| 399958/400000 [00:56<00:00, 7247.36it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7064.95it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7feaeb862128>, <torchtext.data.dataset.TabularDataset object at 0x7feaeb862278>, <torchtext.vocab.Vocab object at 0x7feaeb862198>), {}) 


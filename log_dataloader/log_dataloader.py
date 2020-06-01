
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7febf52822f0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7febf52822f0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fec672d60d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fec672d60d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7febfea689d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7febfea689d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fec14bba8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fec14bba8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fec14bba8c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 43%|████▎     | 4292608/9912422 [00:00<00:00, 42661261.66it/s]9920512it [00:00, 34029057.98it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 282457.95it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7271361.40it/s]          
0it [00:00, ?it/s]8192it [00:00, 157575.17it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7febf7555978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7febf7555fd0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fec14bba510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fec14bba510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fec14bba510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:53,  2.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:53,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:53,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:53,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:52,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:52,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:52,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:37,  4.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:37,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:25,  5.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:25,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:25,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:25,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:25,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:25,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:17,  7.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:17,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:17,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:17,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:17,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:17,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:17,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:12, 10.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:12, 10.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:12, 10.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:12, 10.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:12, 10.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:12, 10.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:12, 10.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:11, 10.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:08, 14.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:08, 14.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:08, 14.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:08, 14.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:08, 14.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:08, 14.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:08, 14.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:08, 14.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:00<00:06, 18.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 23.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 29.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 29.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 29.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 29.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 29.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 29.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 29.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 29.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 29.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 36.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 36.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 36.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 36.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 36.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 36.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 36.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 36.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 41.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 41.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 41.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 41.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 41.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 41.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 41.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 41.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 47.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 47.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 47.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 47.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 47.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 47.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 47.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 47.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 51.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 55.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 55.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 55.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 55.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 55.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 55.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 55.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 55.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 58.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 58.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 58.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 58.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 58.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 58.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 58.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 58.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 58.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 62.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 64.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 68.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 67.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 67.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 69.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 69.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 69.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 69.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.73s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.73s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.73s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.98 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.11s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.73s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 69.98 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.11s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.11s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.69 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.11s/ url]
0 examples [00:00, ? examples/s]2020-06-01 00:09:54.378076: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-01 00:09:54.391731: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-01 00:09:54.391924: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5630dba9ba90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-01 00:09:54.391945: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
10 examples [00:00, 99.35 examples/s]97 examples [00:00, 135.25 examples/s]184 examples [00:00, 181.05 examples/s]282 examples [00:00, 239.57 examples/s]377 examples [00:00, 308.81 examples/s]480 examples [00:00, 390.62 examples/s]581 examples [00:00, 478.29 examples/s]674 examples [00:00, 559.17 examples/s]769 examples [00:00, 637.62 examples/s]863 examples [00:01, 705.22 examples/s]966 examples [00:01, 777.34 examples/s]1068 examples [00:01, 835.37 examples/s]1171 examples [00:01, 884.70 examples/s]1270 examples [00:01, 909.95 examples/s]1369 examples [00:01, 901.81 examples/s]1465 examples [00:01, 909.48 examples/s]1564 examples [00:01, 930.65 examples/s]1665 examples [00:01, 951.41 examples/s]1763 examples [00:01, 934.19 examples/s]1859 examples [00:02, 941.09 examples/s]1960 examples [00:02, 959.31 examples/s]2060 examples [00:02, 970.44 examples/s]2161 examples [00:02, 979.23 examples/s]2262 examples [00:02, 986.94 examples/s]2362 examples [00:02, 973.75 examples/s]2461 examples [00:02, 977.50 examples/s]2560 examples [00:02, 979.98 examples/s]2659 examples [00:02, 982.95 examples/s]2758 examples [00:02, 984.37 examples/s]2857 examples [00:03, 975.52 examples/s]2959 examples [00:03, 987.34 examples/s]3059 examples [00:03, 990.28 examples/s]3161 examples [00:03, 996.68 examples/s]3261 examples [00:03, 952.61 examples/s]3357 examples [00:03, 931.75 examples/s]3455 examples [00:03, 943.98 examples/s]3554 examples [00:03, 955.18 examples/s]3655 examples [00:03, 969.85 examples/s]3753 examples [00:03, 972.86 examples/s]3851 examples [00:04, 951.31 examples/s]3948 examples [00:04, 955.05 examples/s]4049 examples [00:04, 969.80 examples/s]4149 examples [00:04, 978.37 examples/s]4248 examples [00:04, 979.83 examples/s]4348 examples [00:04, 983.41 examples/s]4451 examples [00:04, 994.64 examples/s]4553 examples [00:04, 999.79 examples/s]4654 examples [00:04, 966.06 examples/s]4751 examples [00:05, 930.15 examples/s]4846 examples [00:05, 933.85 examples/s]4947 examples [00:05, 953.61 examples/s]5044 examples [00:05, 957.85 examples/s]5141 examples [00:05, 951.65 examples/s]5237 examples [00:05, 950.57 examples/s]5333 examples [00:05, 934.66 examples/s]5427 examples [00:05, 930.88 examples/s]5521 examples [00:05, 929.72 examples/s]5615 examples [00:05, 932.20 examples/s]5709 examples [00:06, 916.58 examples/s]5801 examples [00:06, 913.78 examples/s]5893 examples [00:06, 902.72 examples/s]5984 examples [00:06, 864.51 examples/s]6084 examples [00:06, 898.85 examples/s]6186 examples [00:06, 932.03 examples/s]6280 examples [00:06, 920.90 examples/s]6373 examples [00:06, 915.88 examples/s]6475 examples [00:06, 942.35 examples/s]6576 examples [00:06, 961.67 examples/s]6676 examples [00:07, 970.46 examples/s]6779 examples [00:07, 985.58 examples/s]6881 examples [00:07, 995.50 examples/s]6981 examples [00:07, 992.32 examples/s]7082 examples [00:07, 995.84 examples/s]7184 examples [00:07, 1002.52 examples/s]7285 examples [00:07, 996.78 examples/s] 7385 examples [00:07, 991.55 examples/s]7487 examples [00:07, 998.70 examples/s]7587 examples [00:08, 937.44 examples/s]7682 examples [00:08, 920.84 examples/s]7775 examples [00:08, 912.08 examples/s]7867 examples [00:08, 889.11 examples/s]7964 examples [00:08, 909.96 examples/s]8056 examples [00:08, 912.64 examples/s]8148 examples [00:08, 908.55 examples/s]8242 examples [00:08, 915.28 examples/s]8336 examples [00:08, 921.85 examples/s]8430 examples [00:08, 926.31 examples/s]8531 examples [00:09, 949.62 examples/s]8632 examples [00:09, 965.09 examples/s]8733 examples [00:09, 977.38 examples/s]8835 examples [00:09, 989.23 examples/s]8937 examples [00:09, 995.62 examples/s]9039 examples [00:09, 999.81 examples/s]9140 examples [00:09, 996.33 examples/s]9240 examples [00:09, 980.48 examples/s]9341 examples [00:09, 987.11 examples/s]9440 examples [00:09, 979.33 examples/s]9539 examples [00:10, 982.11 examples/s]9638 examples [00:10, 981.33 examples/s]9737 examples [00:10, 962.14 examples/s]9834 examples [00:10, 952.93 examples/s]9930 examples [00:10, 935.20 examples/s]10024 examples [00:10, 892.09 examples/s]10125 examples [00:10, 922.80 examples/s]10222 examples [00:10, 935.21 examples/s]10318 examples [00:10, 942.35 examples/s]10417 examples [00:10, 955.25 examples/s]10513 examples [00:11, 930.08 examples/s]10611 examples [00:11, 941.92 examples/s]10709 examples [00:11, 952.64 examples/s]10809 examples [00:11, 965.40 examples/s]10907 examples [00:11, 969.52 examples/s]11005 examples [00:11, 915.14 examples/s]11105 examples [00:11, 936.94 examples/s]11208 examples [00:11, 961.15 examples/s]11309 examples [00:11, 975.27 examples/s]11410 examples [00:12, 984.55 examples/s]11509 examples [00:12, 977.44 examples/s]11608 examples [00:12, 973.66 examples/s]11710 examples [00:12, 986.97 examples/s]11812 examples [00:12, 995.63 examples/s]11914 examples [00:12, 1002.68 examples/s]12017 examples [00:12, 1009.09 examples/s]12118 examples [00:12, 1004.03 examples/s]12221 examples [00:12, 1009.18 examples/s]12322 examples [00:12, 999.12 examples/s] 12423 examples [00:13, 1001.82 examples/s]12524 examples [00:13, 990.58 examples/s] 12624 examples [00:13, 947.09 examples/s]12720 examples [00:13, 937.36 examples/s]12815 examples [00:13, 909.22 examples/s]12914 examples [00:13, 928.53 examples/s]13008 examples [00:13, 921.22 examples/s]13101 examples [00:13, 909.43 examples/s]13194 examples [00:13, 913.73 examples/s]13288 examples [00:13, 920.74 examples/s]13382 examples [00:14, 926.26 examples/s]13475 examples [00:14, 906.63 examples/s]13575 examples [00:14, 932.59 examples/s]13678 examples [00:14, 957.64 examples/s]13780 examples [00:14, 973.74 examples/s]13878 examples [00:14, 965.81 examples/s]13979 examples [00:14, 978.48 examples/s]14078 examples [00:14, 977.51 examples/s]14176 examples [00:14, 944.44 examples/s]14274 examples [00:15, 951.16 examples/s]14371 examples [00:15, 956.18 examples/s]14467 examples [00:15, 946.65 examples/s]14562 examples [00:15, 918.92 examples/s]14655 examples [00:15, 894.07 examples/s]14745 examples [00:15, 891.18 examples/s]14835 examples [00:15, 891.81 examples/s]14925 examples [00:15, 880.62 examples/s]15018 examples [00:15, 894.85 examples/s]15118 examples [00:15, 921.69 examples/s]15215 examples [00:16, 935.36 examples/s]15314 examples [00:16, 949.17 examples/s]15410 examples [00:16, 940.87 examples/s]15505 examples [00:16, 941.09 examples/s]15600 examples [00:16, 942.38 examples/s]15699 examples [00:16, 954.63 examples/s]15797 examples [00:16, 961.75 examples/s]15894 examples [00:16, 947.58 examples/s]15989 examples [00:16, 944.30 examples/s]16084 examples [00:16, 910.01 examples/s]16176 examples [00:17, 893.67 examples/s]16266 examples [00:17, 883.35 examples/s]16360 examples [00:17, 898.24 examples/s]16459 examples [00:17, 922.66 examples/s]16554 examples [00:17, 929.76 examples/s]16648 examples [00:17, 931.36 examples/s]16742 examples [00:17, 931.06 examples/s]16836 examples [00:17, 923.72 examples/s]16929 examples [00:17, 921.23 examples/s]17023 examples [00:17, 925.56 examples/s]17121 examples [00:18, 938.54 examples/s]17215 examples [00:18, 935.46 examples/s]17309 examples [00:18, 925.40 examples/s]17402 examples [00:18, 924.68 examples/s]17499 examples [00:18, 936.00 examples/s]17598 examples [00:18, 949.43 examples/s]17698 examples [00:18, 962.48 examples/s]17798 examples [00:18, 972.72 examples/s]17899 examples [00:18, 981.87 examples/s]18002 examples [00:18, 995.18 examples/s]18104 examples [00:19, 1000.98 examples/s]18205 examples [00:19, 991.83 examples/s] 18305 examples [00:19, 992.24 examples/s]18405 examples [00:19, 975.03 examples/s]18503 examples [00:19, 951.62 examples/s]18599 examples [00:19, 938.70 examples/s]18694 examples [00:19, 928.61 examples/s]18788 examples [00:19, 921.01 examples/s]18881 examples [00:19, 923.12 examples/s]18974 examples [00:20, 920.70 examples/s]19067 examples [00:20, 919.67 examples/s]19160 examples [00:20, 911.29 examples/s]19252 examples [00:20, 910.61 examples/s]19354 examples [00:20, 937.92 examples/s]19453 examples [00:20, 951.34 examples/s]19553 examples [00:20, 964.41 examples/s]19653 examples [00:20, 973.00 examples/s]19753 examples [00:20, 980.86 examples/s]19852 examples [00:20, 978.59 examples/s]19951 examples [00:21, 979.50 examples/s]20050 examples [00:21, 932.59 examples/s]20144 examples [00:21, 927.91 examples/s]20238 examples [00:21, 916.27 examples/s]20331 examples [00:21, 920.09 examples/s]20424 examples [00:21, 908.78 examples/s]20518 examples [00:21, 915.78 examples/s]20615 examples [00:21, 929.26 examples/s]20709 examples [00:21, 924.32 examples/s]20802 examples [00:21, 918.70 examples/s]20899 examples [00:22, 931.81 examples/s]21000 examples [00:22, 953.45 examples/s]21100 examples [00:22, 964.68 examples/s]21197 examples [00:22, 965.05 examples/s]21296 examples [00:22, 969.96 examples/s]21394 examples [00:22, 949.25 examples/s]21492 examples [00:22, 957.67 examples/s]21589 examples [00:22, 959.47 examples/s]21687 examples [00:22, 965.19 examples/s]21788 examples [00:22, 975.54 examples/s]21889 examples [00:23, 982.91 examples/s]21991 examples [00:23, 991.10 examples/s]22091 examples [00:23, 983.02 examples/s]22190 examples [00:23, 932.16 examples/s]22292 examples [00:23, 954.84 examples/s]22395 examples [00:23, 974.37 examples/s]22494 examples [00:23, 977.16 examples/s]22593 examples [00:23, 973.00 examples/s]22691 examples [00:23, 959.50 examples/s]22788 examples [00:24, 921.99 examples/s]22881 examples [00:24, 910.12 examples/s]22973 examples [00:24, 893.67 examples/s]23063 examples [00:24, 856.85 examples/s]23154 examples [00:24, 871.69 examples/s]23242 examples [00:24, 863.59 examples/s]23335 examples [00:24, 880.85 examples/s]23428 examples [00:24, 892.57 examples/s]23521 examples [00:24, 902.42 examples/s]23623 examples [00:24, 932.54 examples/s]23725 examples [00:25, 956.54 examples/s]23827 examples [00:25, 973.34 examples/s]23926 examples [00:25, 977.25 examples/s]24024 examples [00:25, 956.58 examples/s]24124 examples [00:25, 966.58 examples/s]24223 examples [00:25, 972.89 examples/s]24324 examples [00:25, 981.89 examples/s]24423 examples [00:25, 962.94 examples/s]24520 examples [00:25, 928.88 examples/s]24614 examples [00:26, 925.71 examples/s]24707 examples [00:26, 906.56 examples/s]24798 examples [00:26, 864.18 examples/s]24888 examples [00:26, 873.72 examples/s]24976 examples [00:26, 863.09 examples/s]25067 examples [00:26, 874.63 examples/s]25164 examples [00:26, 900.80 examples/s]25263 examples [00:26, 925.11 examples/s]25356 examples [00:26, 922.91 examples/s]25453 examples [00:26, 935.18 examples/s]25553 examples [00:27, 952.87 examples/s]25653 examples [00:27, 966.30 examples/s]25754 examples [00:27, 976.53 examples/s]25853 examples [00:27, 979.89 examples/s]25952 examples [00:27, 978.82 examples/s]26053 examples [00:27, 984.92 examples/s]26154 examples [00:27, 990.79 examples/s]26255 examples [00:27, 996.19 examples/s]26355 examples [00:27, 932.52 examples/s]26450 examples [00:27, 907.77 examples/s]26542 examples [00:28, 883.37 examples/s]26633 examples [00:28, 890.92 examples/s]26733 examples [00:28, 919.17 examples/s]26833 examples [00:28, 941.73 examples/s]26928 examples [00:28, 938.97 examples/s]27023 examples [00:28, 935.68 examples/s]27118 examples [00:28, 937.77 examples/s]27213 examples [00:28, 941.38 examples/s]27308 examples [00:28, 939.48 examples/s]27403 examples [00:29, 928.75 examples/s]27497 examples [00:29, 931.55 examples/s]27592 examples [00:29, 936.64 examples/s]27686 examples [00:29, 931.12 examples/s]27780 examples [00:29, 896.13 examples/s]27870 examples [00:29, 867.09 examples/s]27961 examples [00:29, 878.68 examples/s]28050 examples [00:29, 873.20 examples/s]28141 examples [00:29, 882.22 examples/s]28230 examples [00:29, 884.44 examples/s]28319 examples [00:30, 876.35 examples/s]28417 examples [00:30, 904.04 examples/s]28511 examples [00:30, 913.06 examples/s]28608 examples [00:30, 927.65 examples/s]28701 examples [00:30, 918.47 examples/s]28794 examples [00:30, 916.34 examples/s]28888 examples [00:30, 921.85 examples/s]28987 examples [00:30, 940.62 examples/s]29089 examples [00:30, 963.08 examples/s]29186 examples [00:30, 963.23 examples/s]29283 examples [00:31, 946.27 examples/s]29378 examples [00:31, 909.71 examples/s]29470 examples [00:31, 910.10 examples/s]29562 examples [00:31, 910.45 examples/s]29654 examples [00:31, 886.53 examples/s]29743 examples [00:31, 885.32 examples/s]29834 examples [00:31, 892.24 examples/s]29924 examples [00:31, 884.12 examples/s]30013 examples [00:31, 837.66 examples/s]30112 examples [00:32, 876.84 examples/s]30210 examples [00:32, 901.19 examples/s]30312 examples [00:32, 931.53 examples/s]30414 examples [00:32, 956.29 examples/s]30515 examples [00:32, 969.20 examples/s]30613 examples [00:32, 969.66 examples/s]30713 examples [00:32, 978.51 examples/s]30812 examples [00:32, 970.81 examples/s]30910 examples [00:32, 968.25 examples/s]31011 examples [00:32, 979.53 examples/s]31110 examples [00:33, 949.25 examples/s]31209 examples [00:33, 960.70 examples/s]31312 examples [00:33, 978.17 examples/s]31415 examples [00:33, 990.95 examples/s]31515 examples [00:33, 992.35 examples/s]31615 examples [00:33, 958.73 examples/s]31712 examples [00:33, 950.97 examples/s]31808 examples [00:33, 936.02 examples/s]31905 examples [00:33, 944.95 examples/s]32000 examples [00:33, 945.56 examples/s]32098 examples [00:34, 952.21 examples/s]32194 examples [00:34, 946.66 examples/s]32289 examples [00:34, 916.62 examples/s]32383 examples [00:34, 922.92 examples/s]32483 examples [00:34, 941.87 examples/s]32583 examples [00:34, 957.77 examples/s]32685 examples [00:34, 975.54 examples/s]32783 examples [00:34, 976.28 examples/s]32882 examples [00:34, 980.11 examples/s]32984 examples [00:34, 990.80 examples/s]33084 examples [00:35, 992.68 examples/s]33184 examples [00:35, 994.48 examples/s]33286 examples [00:35, 999.95 examples/s]33387 examples [00:35, 993.17 examples/s]33489 examples [00:35, 998.94 examples/s]33589 examples [00:35, 994.32 examples/s]33692 examples [00:35, 1002.86 examples/s]33795 examples [00:35, 1008.61 examples/s]33896 examples [00:35, 987.98 examples/s] 33995 examples [00:36, 949.27 examples/s]34091 examples [00:36, 909.80 examples/s]34184 examples [00:36, 914.12 examples/s]34283 examples [00:36, 933.79 examples/s]34381 examples [00:36, 944.82 examples/s]34481 examples [00:36, 958.58 examples/s]34578 examples [00:36, 956.13 examples/s]34674 examples [00:36, 940.58 examples/s]34773 examples [00:36, 953.84 examples/s]34871 examples [00:36, 958.98 examples/s]34968 examples [00:37, 926.12 examples/s]35064 examples [00:37, 934.00 examples/s]35161 examples [00:37, 944.19 examples/s]35258 examples [00:37, 951.11 examples/s]35358 examples [00:37, 963.81 examples/s]35455 examples [00:37, 957.67 examples/s]35551 examples [00:37, 944.27 examples/s]35646 examples [00:37, 940.68 examples/s]35741 examples [00:37, 929.49 examples/s]35836 examples [00:37, 933.76 examples/s]35937 examples [00:38, 954.43 examples/s]36034 examples [00:38, 957.26 examples/s]36130 examples [00:38, 926.31 examples/s]36223 examples [00:38, 907.88 examples/s]36315 examples [00:38, 873.28 examples/s]36413 examples [00:38, 901.26 examples/s]36515 examples [00:38, 931.59 examples/s]36617 examples [00:38, 954.11 examples/s]36720 examples [00:38, 972.96 examples/s]36821 examples [00:39, 981.48 examples/s]36920 examples [00:39, 947.03 examples/s]37016 examples [00:39, 910.65 examples/s]37108 examples [00:39, 910.39 examples/s]37200 examples [00:39, 908.09 examples/s]37295 examples [00:39, 918.05 examples/s]37395 examples [00:39, 938.70 examples/s]37491 examples [00:39, 943.18 examples/s]37588 examples [00:39, 948.54 examples/s]37687 examples [00:39, 959.65 examples/s]37787 examples [00:40, 970.16 examples/s]37886 examples [00:40, 974.54 examples/s]37984 examples [00:40, 971.98 examples/s]38082 examples [00:40, 965.84 examples/s]38179 examples [00:40, 945.82 examples/s]38274 examples [00:40, 930.68 examples/s]38369 examples [00:40, 935.96 examples/s]38463 examples [00:40, 936.56 examples/s]38558 examples [00:40, 939.90 examples/s]38653 examples [00:40, 937.81 examples/s]38747 examples [00:41, 937.85 examples/s]38841 examples [00:41, 930.77 examples/s]38937 examples [00:41, 939.26 examples/s]39037 examples [00:41, 953.32 examples/s]39136 examples [00:41, 962.11 examples/s]39233 examples [00:41, 954.23 examples/s]39330 examples [00:41, 956.81 examples/s]39429 examples [00:41, 965.08 examples/s]39528 examples [00:41, 972.09 examples/s]39627 examples [00:41, 975.17 examples/s]39725 examples [00:42, 940.31 examples/s]39820 examples [00:42, 937.00 examples/s]39914 examples [00:42, 924.32 examples/s]40007 examples [00:42, 896.63 examples/s]40104 examples [00:42, 916.88 examples/s]40204 examples [00:42, 938.17 examples/s]40303 examples [00:42, 950.68 examples/s]40399 examples [00:42, 948.12 examples/s]40495 examples [00:42, 945.20 examples/s]40590 examples [00:43, 941.83 examples/s]40685 examples [00:43, 935.98 examples/s]40779 examples [00:43, 923.12 examples/s]40879 examples [00:43, 944.22 examples/s]40980 examples [00:43, 962.23 examples/s]41082 examples [00:43, 978.70 examples/s]41181 examples [00:43, 980.36 examples/s]41280 examples [00:43, 975.60 examples/s]41381 examples [00:43, 983.19 examples/s]41480 examples [00:43, 965.92 examples/s]41578 examples [00:44, 970.03 examples/s]41676 examples [00:44, 968.84 examples/s]41777 examples [00:44, 980.15 examples/s]41876 examples [00:44, 982.18 examples/s]41977 examples [00:44, 988.34 examples/s]42076 examples [00:44, 984.18 examples/s]42179 examples [00:44, 994.88 examples/s]42280 examples [00:44, 997.86 examples/s]42383 examples [00:44, 1004.47 examples/s]42485 examples [00:44, 1008.07 examples/s]42587 examples [00:45, 1009.28 examples/s]42688 examples [00:45, 1006.51 examples/s]42789 examples [00:45, 976.80 examples/s] 42887 examples [00:45, 951.05 examples/s]42984 examples [00:45, 954.47 examples/s]43086 examples [00:45, 971.98 examples/s]43184 examples [00:45, 935.87 examples/s]43279 examples [00:45, 908.41 examples/s]43379 examples [00:45, 933.30 examples/s]43479 examples [00:45, 952.24 examples/s]43579 examples [00:46, 965.78 examples/s]43677 examples [00:46, 967.88 examples/s]43775 examples [00:46, 958.23 examples/s]43872 examples [00:46, 958.12 examples/s]43972 examples [00:46, 969.97 examples/s]44070 examples [00:46, 968.17 examples/s]44170 examples [00:46, 976.22 examples/s]44268 examples [00:46, 971.90 examples/s]44370 examples [00:46, 985.14 examples/s]44472 examples [00:46, 994.08 examples/s]44572 examples [00:47, 992.67 examples/s]44672 examples [00:47, 970.68 examples/s]44770 examples [00:47, 942.17 examples/s]44865 examples [00:47, 941.08 examples/s]44963 examples [00:47, 950.08 examples/s]45065 examples [00:47, 969.98 examples/s]45166 examples [00:47, 980.37 examples/s]45267 examples [00:47, 988.42 examples/s]45370 examples [00:47, 999.63 examples/s]45473 examples [00:48, 1006.99 examples/s]45575 examples [00:48, 1010.49 examples/s]45678 examples [00:48, 1015.28 examples/s]45780 examples [00:48, 994.84 examples/s] 45880 examples [00:48, 980.03 examples/s]45979 examples [00:48, 950.98 examples/s]46075 examples [00:48, 935.15 examples/s]46169 examples [00:48, 931.73 examples/s]46267 examples [00:48, 944.21 examples/s]46362 examples [00:48, 930.92 examples/s]46456 examples [00:49, 910.97 examples/s]46549 examples [00:49, 916.57 examples/s]46641 examples [00:49, 916.41 examples/s]46740 examples [00:49, 937.20 examples/s]46841 examples [00:49, 956.27 examples/s]46942 examples [00:49, 969.91 examples/s]47040 examples [00:49, 952.71 examples/s]47138 examples [00:49, 959.76 examples/s]47238 examples [00:49, 970.20 examples/s]47338 examples [00:49, 977.83 examples/s]47438 examples [00:50, 981.73 examples/s]47539 examples [00:50, 989.13 examples/s]47638 examples [00:50, 972.35 examples/s]47736 examples [00:50, 964.22 examples/s]47835 examples [00:50, 969.50 examples/s]47933 examples [00:50, 959.64 examples/s]48033 examples [00:50, 969.13 examples/s]48130 examples [00:50, 967.05 examples/s]48227 examples [00:50, 963.46 examples/s]48324 examples [00:51, 931.83 examples/s]48418 examples [00:51, 911.79 examples/s]48517 examples [00:51, 933.55 examples/s]48615 examples [00:51, 946.38 examples/s]48710 examples [00:51, 929.91 examples/s]48807 examples [00:51, 939.05 examples/s]48902 examples [00:51, 940.19 examples/s]48997 examples [00:51, 931.46 examples/s]49091 examples [00:51, 930.14 examples/s]49185 examples [00:51, 931.70 examples/s]49279 examples [00:52, 934.03 examples/s]49373 examples [00:52, 928.77 examples/s]49466 examples [00:52, 904.47 examples/s]49562 examples [00:52, 918.32 examples/s]49659 examples [00:52, 932.58 examples/s]49757 examples [00:52, 944.03 examples/s]49854 examples [00:52, 949.36 examples/s]49952 examples [00:52, 955.66 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6529/50000 [00:00<00:00, 65287.42 examples/s] 36%|███▌      | 18120/50000 [00:00<00:00, 75129.97 examples/s] 60%|█████▉    | 29826/50000 [00:00<00:00, 84174.88 examples/s] 83%|████████▎ | 41559/50000 [00:00<00:00, 91970.22 examples/s]                                                               0 examples [00:00, ? examples/s]75 examples [00:00, 743.95 examples/s]168 examples [00:00, 790.57 examples/s]261 examples [00:00, 826.89 examples/s]355 examples [00:00, 856.66 examples/s]439 examples [00:00, 849.30 examples/s]528 examples [00:00, 859.07 examples/s]621 examples [00:00, 879.18 examples/s]714 examples [00:00, 893.11 examples/s]808 examples [00:00, 905.05 examples/s]903 examples [00:01, 915.95 examples/s]1002 examples [00:01, 934.95 examples/s]1099 examples [00:01, 944.44 examples/s]1201 examples [00:01, 964.96 examples/s]1298 examples [00:01, 963.55 examples/s]1395 examples [00:01, 776.87 examples/s]1496 examples [00:01, 834.65 examples/s]1598 examples [00:01, 881.97 examples/s]1699 examples [00:01, 916.39 examples/s]1794 examples [00:01, 916.46 examples/s]1888 examples [00:02, 906.01 examples/s]1981 examples [00:02, 891.22 examples/s]2072 examples [00:02, 891.53 examples/s]2163 examples [00:02, 892.88 examples/s]2255 examples [00:02, 898.99 examples/s]2351 examples [00:02, 914.78 examples/s]2448 examples [00:02, 925.62 examples/s]2547 examples [00:02, 942.77 examples/s]2647 examples [00:02, 957.62 examples/s]2747 examples [00:03, 967.66 examples/s]2844 examples [00:03, 955.40 examples/s]2940 examples [00:03, 947.78 examples/s]3035 examples [00:03, 938.38 examples/s]3129 examples [00:03, 936.12 examples/s]3231 examples [00:03, 958.37 examples/s]3330 examples [00:03, 965.24 examples/s]3427 examples [00:03, 956.77 examples/s]3523 examples [00:03, 956.44 examples/s]3623 examples [00:03, 968.17 examples/s]3723 examples [00:04, 976.87 examples/s]3821 examples [00:04, 963.42 examples/s]3918 examples [00:04, 945.64 examples/s]4018 examples [00:04, 959.47 examples/s]4115 examples [00:04, 939.58 examples/s]4212 examples [00:04, 946.99 examples/s]4314 examples [00:04, 965.35 examples/s]4414 examples [00:04, 973.56 examples/s]4512 examples [00:04, 975.17 examples/s]4612 examples [00:04, 981.37 examples/s]4712 examples [00:05, 985.78 examples/s]4813 examples [00:05, 991.42 examples/s]4913 examples [00:05, 988.90 examples/s]5012 examples [00:05, 987.91 examples/s]5111 examples [00:05, 987.34 examples/s]5210 examples [00:05, 973.59 examples/s]5308 examples [00:05, 955.84 examples/s]5404 examples [00:05, 947.63 examples/s]5499 examples [00:05, 933.07 examples/s]5600 examples [00:05, 952.43 examples/s]5701 examples [00:06, 967.25 examples/s]5798 examples [00:06, 952.19 examples/s]5894 examples [00:06, 948.26 examples/s]5993 examples [00:06, 958.07 examples/s]6095 examples [00:06, 975.09 examples/s]6193 examples [00:06, 943.92 examples/s]6293 examples [00:06, 959.01 examples/s]6391 examples [00:06, 964.63 examples/s]6489 examples [00:06, 968.96 examples/s]6587 examples [00:07, 943.98 examples/s]6684 examples [00:07, 944.82 examples/s]6779 examples [00:07, 939.61 examples/s]6880 examples [00:07, 957.91 examples/s]6980 examples [00:07, 970.05 examples/s]7078 examples [00:07, 942.20 examples/s]7179 examples [00:07, 960.83 examples/s]7280 examples [00:07, 971.96 examples/s]7381 examples [00:07, 981.22 examples/s]7484 examples [00:07, 992.86 examples/s]7584 examples [00:08, 990.01 examples/s]7684 examples [00:08, 986.13 examples/s]7783 examples [00:08, 987.00 examples/s]7882 examples [00:08, 985.84 examples/s]7982 examples [00:08, 988.31 examples/s]8084 examples [00:08, 995.85 examples/s]8184 examples [00:08, 974.45 examples/s]8282 examples [00:08, 952.07 examples/s]8378 examples [00:08, 942.75 examples/s]8473 examples [00:08, 934.49 examples/s]8567 examples [00:09, 933.47 examples/s]8661 examples [00:09, 930.11 examples/s]8755 examples [00:09, 923.48 examples/s]8848 examples [00:09, 899.93 examples/s]8939 examples [00:09, 899.26 examples/s]9030 examples [00:09, 902.26 examples/s]9126 examples [00:09, 918.18 examples/s]9220 examples [00:09, 923.57 examples/s]9315 examples [00:09, 930.54 examples/s]9409 examples [00:09, 927.94 examples/s]9502 examples [00:10, 925.54 examples/s]9599 examples [00:10, 936.67 examples/s]9693 examples [00:10, 904.77 examples/s]9785 examples [00:10, 908.99 examples/s]9881 examples [00:10, 922.26 examples/s]9974 examples [00:10, 917.19 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteINH02B/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteINH02B/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fec14bba8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fec14bba8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fec14bba8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fec80a574e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fec8105dcf8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fec14bba8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fec14bba8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fec14bba8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7febf5ea6b70>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7febf81a90f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7febec41b620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7febec41b620> 

  function with postional parmater data_info <function split_train_valid at 0x7febec41b620> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7febec41b730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7febec41b730> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7febec41b730> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=3e41f328a8a2711d6c346ecad68f0c8a64545bda466182a67fce54d9bfe34b07
  Stored in directory: /tmp/pip-ephem-wheel-cache-tjrpagw6/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<32:20:18, 7.41kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<22:52:14, 10.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<16:03:45, 14.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<11:14:56, 21.3kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<7:51:09, 30.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.84M/862M [00:01<5:27:31, 43.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.4M/862M [00:01<3:47:51, 61.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.1M/862M [00:01<2:38:54, 88.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.6M/862M [00:02<1:50:44, 126kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 27.7M/862M [00:02<1:17:15, 180kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.5M/862M [00:02<53:48, 257kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 39.2M/862M [00:02<37:30, 366kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.9M/862M [00:02<26:10, 521kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.7M/862M [00:02<18:16, 740kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:03<14:04, 959kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<11:43, 1.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<12:18, 1.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<09:39, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.0M/862M [00:05<07:00, 1.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:07<09:14, 1.45MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<08:09, 1.64MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:07<06:03, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:09<06:53, 1.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<07:47, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:09<06:11, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:09<04:28, 2.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:11<17:19, 763kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<13:29, 980kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:11<09:46, 1.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:13<09:55, 1.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:13<09:45, 1.35MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:13<07:30, 1.75MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:13<05:22, 2.43MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:15<51:02, 256kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<37:04, 353kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:15<26:14, 498kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:17<21:22, 609kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:17<16:17, 798kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:17<11:40, 1.11MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<11:14, 1.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:19<11:34, 1.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:19<09:03, 1.43MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.1M/862M [00:19<06:32, 1.97MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<09:21, 1.38MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:21<10:14, 1.26MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:21<07:57, 1.62MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.9M/862M [00:21<05:46, 2.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<07:46, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:23<09:07, 1.40MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:23<07:17, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.5M/862M [00:23<05:17, 2.41MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<08:38, 1.47MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:25<09:19, 1.37MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:25<07:13, 1.76MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<05:17, 2.40MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:07, 1.78MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<08:26, 1.50MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<06:35, 1.92MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<04:50, 2.61MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:53, 1.83MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<08:03, 1.56MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<06:19, 1.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<04:40, 2.69MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:30, 1.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<07:38, 1.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<05:59, 2.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<04:28, 2.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:07, 2.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<07:22, 1.69MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<05:55, 2.10MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<04:19, 2.87MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:47, 1.41MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<09:12, 1.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<07:03, 1.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<05:07, 2.41MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:43, 1.60MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:16, 1.49MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<06:24, 1.92MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<04:38, 2.64MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:03, 1.52MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:39, 1.41MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<06:50, 1.79MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<04:57, 2.46MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:57, 1.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<09:06, 1.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<06:57, 1.75MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:00, 2.42MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:40, 1.25MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:27, 1.28MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<07:12, 1.68MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:11, 2.32MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<09:45, 1.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<09:55, 1.21MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:45<07:35, 1.59MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<05:29, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:38, 1.57MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<08:25, 1.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<06:39, 1.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<04:48, 2.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:45, 1.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:46, 1.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<06:47, 1.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<04:55, 2.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<10:12, 1.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<10:02, 1.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<07:46, 1.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:37, 2.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:14, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:28, 1.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<07:17, 1.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:15, 2.23MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:00, 1.46MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:36, 1.36MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<06:46, 1.72MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<04:54, 2.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:29, 1.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<08:30, 1.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:30, 1.79MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<04:41, 2.47MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<09:26, 1.23MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<09:34, 1.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<07:26, 1.55MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:22, 2.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<08:49, 1.30MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<08:36, 1.33MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<06:32, 1.76MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<04:44, 2.42MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:45, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:57, 1.44MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:10, 1.85MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:29, 2.53MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<10:07, 1.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<09:59, 1.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<07:37, 1.49MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:28, 2.06MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:19, 1.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<08:42, 1.29MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:49, 1.65MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:55, 2.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<08:19, 1.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<08:53, 1.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:57, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:02, 2.21MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<08:12, 1.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<08:06, 1.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<06:12, 1.79MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:30, 2.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<09:52, 1.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<09:45, 1.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<07:31, 1.47MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:24, 2.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:52, 1.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:32, 1.29MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<06:30, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<04:42, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<10:00, 1.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<09:28, 1.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<07:08, 1.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<05:07, 2.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<10:05, 1.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<09:22, 1.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<07:03, 1.54MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:18<05:03, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<09:23, 1.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<09:20, 1.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<07:14, 1.49MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:20<05:12, 2.06MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<08:23, 1.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<08:36, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:35, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<04:44, 2.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<07:49, 1.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<07:50, 1.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<06:03, 1.76MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<04:23, 2.42MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<10:52, 974kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<09:52, 1.07MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<07:28, 1.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<05:22, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<11:24, 922kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<10:18, 1.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<07:42, 1.36MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<05:32, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<10:16, 1.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<09:52, 1.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:28, 1.40MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<05:20, 1.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<08:13, 1.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<08:24, 1.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<06:26, 1.61MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<04:40, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<08:15, 1.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<08:18, 1.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<06:28, 1.59MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<04:41, 2.19MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<08:01, 1.28MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<07:47, 1.31MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:56, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<04:16, 2.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<09:13, 1.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<08:36, 1.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<06:28, 1.57MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<04:39, 2.17MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<08:00, 1.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<08:03, 1.25MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<06:16, 1.61MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<04:32, 2.21MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<08:02, 1.25MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<07:56, 1.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<06:07, 1.64MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<04:25, 2.25MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<08:57, 1.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<08:17, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<06:17, 1.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<04:31, 2.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<12:41, 780kB/s] .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<11:18, 875kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<08:24, 1.18MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<06:00, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:45, 1.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:49, 1.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:58, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<04:17, 2.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<07:28, 1.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<07:36, 1.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:54, 1.65MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<04:16, 2.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<08:23, 1.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<08:07, 1.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<06:09, 1.57MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<04:26, 2.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<06:33, 1.47MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<06:55, 1.39MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<05:24, 1.77MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:55, 2.43MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<08:08, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<07:54, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:06, 1.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<04:23, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<08:25, 1.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<08:06, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:08, 1.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<04:24, 2.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<07:24, 1.27MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<07:21, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:37, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<04:03, 2.31MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:34, 1.42MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:52, 1.36MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:17, 1.76MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:50, 2.42MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<07:38, 1.21MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<07:36, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:53, 1.57MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<04:15, 2.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<07:55, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<07:34, 1.21MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:49, 1.58MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<04:12, 2.18MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<08:38, 1.06MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<08:16, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<06:15, 1.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<04:30, 2.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<06:04, 1.49MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<06:21, 1.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:59, 1.81MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<03:36, 2.50MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<07:20, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<06:35, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:55, 1.82MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<03:31, 2.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<18:33, 481kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<14:53, 599kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<10:49, 822kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<07:38, 1.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<11:36, 763kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<09:59, 885kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<07:26, 1.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<05:17, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<29:05, 302kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<22:12, 395kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<15:55, 551kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<11:12, 779kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<12:26, 701kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<09:57, 875kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<07:12, 1.20MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<06:39, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<06:17, 1.37MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:46, 1.81MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<04:44, 1.81MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:56, 1.73MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<03:51, 2.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:05, 2.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:31, 1.88MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:34, 2.38MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:52, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<04:21, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:26, 2.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:47, 2.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<04:13, 1.98MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:18, 2.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<02:23, 3.48MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<16:33, 502kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<14:42, 565kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<11:06, 747kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<07:55, 1.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<07:28, 1.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:46, 1.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:07, 1.60MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:55, 1.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:00, 1.63MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:54, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<02:47, 2.90MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<38:11, 212kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<28:18, 286kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<20:09, 401kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<15:21, 523kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<12:15, 655kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<08:56, 895kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<07:32, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<06:46, 1.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<05:03, 1.57MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:38, 2.17MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:41, 1.39MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:31, 1.43MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:10, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<03:00, 2.60MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<07:05, 1.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<06:29, 1.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:54, 1.59MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:41, 1.65MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:44, 1.64MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:40, 2.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:49, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:07, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:14, 2.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:31, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:57, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:07, 2.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:25, 2.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:48, 1.98MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:01, 2.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<03:20, 2.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:44, 2.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<02:55, 2.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<02:07, 3.50MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<08:33, 866kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<07:23, 1.00MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:30, 1.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:02, 1.45MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:57, 1.48MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:48, 1.92MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:51, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:04, 1.79MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<03:11, 2.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:24, 2.11MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:44, 1.92MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<02:54, 2.48MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:07, 3.37MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<05:08, 1.39MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<06:29, 1.10MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:14, 1.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:48, 1.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<04:18, 1.64MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<04:21, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:22, 2.09MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<03:30, 2.00MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<03:48, 1.83MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<02:59, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<03:13, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:33, 1.94MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<02:46, 2.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<02:00, 3.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<07:28, 919kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<06:30, 1.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:52, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:30, 1.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:29, 1.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:27, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:30, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:46, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:57, 2.27MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<03:08, 2.11MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<04:54, 1.36MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:04, 1.63MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:58, 2.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<03:38, 1.81MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<03:46, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:57, 2.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<03:07, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:27, 1.88MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:43, 2.38MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<02:59, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:30, 1.43MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:47, 1.70MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:47, 2.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:27, 1.84MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:39, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:49, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:02, 3.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<05:24, 1.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<04:57, 1.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:43, 1.69MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:40, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<05:38, 1.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<05:07, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:50, 1.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:44, 2.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<07:59, 772kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<06:45, 913kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<05:00, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:29, 1.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<04:20, 1.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:19, 1.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:18, 1.82MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:26, 1.75MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:41, 2.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:51, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:07, 1.91MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:25, 2.45MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<01:45, 3.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<06:05, 969kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<05:24, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<04:03, 1.45MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:46, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<04:50, 1.20MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<03:51, 1.51MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:48, 2.06MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:12, 1.79MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:22, 1.71MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:37, 2.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:45, 2.06MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:00, 1.89MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:22, 2.39MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:34, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:51, 1.96MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:13, 2.52MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<01:38, 3.40MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:18, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:23, 1.64MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:38, 2.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:44, 2.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:56, 1.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:16, 2.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<01:39, 3.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:42, 1.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:19, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:16, 1.65MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:09, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:55, 1.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:13, 2.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:33, 2.06MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:38, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:01, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:13, 2.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:54, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:58, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:18, 2.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:28, 2.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:39, 1.93MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:05, 2.45MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:18, 2.20MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:29, 2.03MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<01:58, 2.56MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:12, 2.26MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:26, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<01:55, 2.58MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:10, 2.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:23, 2.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<01:50, 2.66MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:22, 3.57MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:51, 1.70MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:50, 1.70MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:12, 2.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:20, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:30, 1.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:57, 2.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:09, 2.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:22, 1.98MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:52, 2.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:05, 2.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:16, 2.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:45, 2.64MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:16, 3.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<04:05, 1.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:41, 1.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:47, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:41, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:40, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:01, 2.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:28, 3.03MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<03:41, 1.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<03:23, 1.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:33, 1.73MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<02:30, 1.74MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<03:24, 1.29MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:46, 1.58MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:00, 2.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:31, 1.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:31, 1.71MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:54, 2.24MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:23, 3.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<03:22, 1.26MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<03:07, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:20, 1.81MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:40, 2.50MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<04:20, 961kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:47, 1.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:49, 1.47MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:39, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:34, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:57, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:23, 2.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<24:52, 162kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<18:05, 223kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<12:47, 314kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<09:31, 417kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<07:21, 539kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<05:16, 749kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<03:42, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<05:16, 738kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<04:23, 886kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<03:14, 1.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:53, 1.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:41, 1.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:02, 1.86MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:03, 1.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:06, 1.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:36, 2.32MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:10, 3.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<03:21, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:59, 1.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:15, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:10, 1.67MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:09, 1.68MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:37, 2.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:10, 3.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:56, 1.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:41, 1.32MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:00, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:26, 2.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:57, 881kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:22, 1.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:29, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:45, 1.95MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<10:00, 341kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<07:37, 448kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<05:27, 622kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<04:20, 772kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:38, 920kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:39, 1.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:53, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:52, 1.14MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:36, 1.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:57, 1.66MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:53, 1.69MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:53, 1.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:26, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:31, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:15, 1.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:51, 1.69MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:20, 2.31MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:43, 1.78MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:44, 1.76MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:19, 2.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<00:56, 3.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<06:48, 441kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<05:18, 566kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<03:48, 784kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<02:39, 1.11MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<05:12, 564kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<04:08, 706kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:59, 973kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<02:06, 1.37MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<03:31, 814kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:57, 967kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:10, 1.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:07<01:31, 1.83MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<05:11, 539kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<04:08, 675kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:59, 931kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<02:05, 1.31MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<03:12, 852kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:43, 997kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:01, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<01:50, 1.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:45, 1.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:18, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<00:58, 2.70MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:31, 1.71MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:31, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:09, 2.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:50, 3.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:02, 1.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:51, 1.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:24, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:23, 1.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:24, 1.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:05, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:09, 2.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:13, 1.95MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<00:56, 2.52MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:40, 3.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<02:15, 1.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:59, 1.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:29, 1.54MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:24, 1.60MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:23, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:02, 2.13MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:44, 2.92MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:52, 1.16MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:42, 1.27MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:17, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:14, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:13, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:56, 2.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:59, 2.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:03, 1.92MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:49, 2.43MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:54, 2.19MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<00:59, 1.99MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:46, 2.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:51, 2.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:55, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:43, 2.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:48, 2.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<00:53, 2.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:42, 2.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:46, 2.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:51, 2.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:39, 2.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:28, 3.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:18, 737kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:54, 890kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:23, 1.21MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:57, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<03:05, 528kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<02:27, 663kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:46, 908kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:28, 1.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:17, 1.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:57, 1.61MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:40, 2.24MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:22, 1.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:13, 1.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:54, 1.61MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:51, 1.66MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:51, 1.65MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:38, 2.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:28, 2.92MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:45, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:45, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:35, 2.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:36, 2.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:39, 1.97MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:30, 2.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:32, 2.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:36, 2.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:27, 2.60MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:19, 3.55MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:06, 1.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:58, 1.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:43, 1.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:40, 1.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:39, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:29, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:20, 2.97MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:48, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:44, 1.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:32, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:22, 2.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:07, 840kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:56, 992kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:41, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:36, 1.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:34, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:26, 1.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:25, 1.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:26, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:19, 2.39MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:13, 3.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:33, 1.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:31, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:22, 1.87MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:15, 2.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:38, 1.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:33, 1.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:24, 1.58MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:16, 2.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:34, 1.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:30, 1.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:21, 1.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:14, 2.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:26, 1.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:23, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:17, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:12, 2.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:11, 2.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:12, 1.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:09, 2.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:09, 2.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:07, 2.60MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:10, 1.46MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:08, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:05, 2.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 1.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:02, 3.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:05, 1.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:05, 1.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.79MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 2.48MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:03, 892kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.04MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.41MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<11:12:26,  9.91it/s]  0%|          | 677/400000 [00:00<7:50:12, 14.15it/s]  0%|          | 1411/400000 [00:00<5:28:48, 20.20it/s]  1%|          | 2113/400000 [00:00<3:50:02, 28.83it/s]  1%|          | 2814/400000 [00:00<2:41:01, 41.11it/s]  1%|          | 3559/400000 [00:00<1:52:46, 58.59it/s]  1%|          | 4245/400000 [00:00<1:19:05, 83.39it/s]  1%|          | 4919/400000 [00:00<55:33, 118.50it/s]   1%|▏         | 5663/400000 [00:00<39:05, 168.14it/s]  2%|▏         | 6398/400000 [00:01<27:34, 237.87it/s]  2%|▏         | 7088/400000 [00:01<19:34, 334.67it/s]  2%|▏         | 7839/400000 [00:01<13:55, 469.14it/s]  2%|▏         | 8587/400000 [00:01<09:59, 652.65it/s]  2%|▏         | 9326/400000 [00:01<07:14, 898.32it/s]  3%|▎         | 10072/400000 [00:01<05:19, 1220.27it/s]  3%|▎         | 10822/400000 [00:01<03:58, 1629.52it/s]  3%|▎         | 11558/400000 [00:01<03:03, 2120.05it/s]  3%|▎         | 12287/400000 [00:01<02:25, 2671.45it/s]  3%|▎         | 13029/400000 [00:01<01:57, 3305.93it/s]  3%|▎         | 13768/400000 [00:02<01:37, 3962.85it/s]  4%|▎         | 14496/400000 [00:02<01:24, 4579.66it/s]  4%|▍         | 15246/400000 [00:02<01:14, 5184.57it/s]  4%|▍         | 15979/400000 [00:02<01:08, 5627.44it/s]  4%|▍         | 16720/400000 [00:02<01:03, 6063.21it/s]  4%|▍         | 17463/400000 [00:02<00:59, 6385.84it/s]  5%|▍         | 18192/400000 [00:02<00:58, 6541.90it/s]  5%|▍         | 18936/400000 [00:02<00:56, 6785.55it/s]  5%|▍         | 19662/400000 [00:02<00:55, 6900.62it/s]  5%|▌         | 20386/400000 [00:02<00:55, 6866.69it/s]  5%|▌         | 21125/400000 [00:03<00:54, 7010.28it/s]  5%|▌         | 21850/400000 [00:03<00:53, 7079.70it/s]  6%|▌         | 22588/400000 [00:03<00:52, 7165.39it/s]  6%|▌         | 23334/400000 [00:03<00:51, 7251.30it/s]  6%|▌         | 24083/400000 [00:03<00:51, 7319.72it/s]  6%|▌         | 24820/400000 [00:03<00:51, 7252.25it/s]  6%|▋         | 25567/400000 [00:03<00:51, 7315.93it/s]  7%|▋         | 26314/400000 [00:03<00:50, 7359.80it/s]  7%|▋         | 27061/400000 [00:03<00:50, 7391.71it/s]  7%|▋         | 27802/400000 [00:03<00:51, 7263.89it/s]  7%|▋         | 28550/400000 [00:04<00:50, 7324.16it/s]  7%|▋         | 29289/400000 [00:04<00:50, 7342.13it/s]  8%|▊         | 30027/400000 [00:04<00:50, 7350.65it/s]  8%|▊         | 30763/400000 [00:04<00:50, 7287.92it/s]  8%|▊         | 31493/400000 [00:04<00:51, 7118.97it/s]  8%|▊         | 32207/400000 [00:04<00:52, 6978.43it/s]  8%|▊         | 32907/400000 [00:04<00:53, 6882.73it/s]  8%|▊         | 33599/400000 [00:04<00:53, 6891.61it/s]  9%|▊         | 34290/400000 [00:04<00:53, 6884.55it/s]  9%|▊         | 34980/400000 [00:04<00:54, 6736.74it/s]  9%|▉         | 35655/400000 [00:05<00:54, 6696.34it/s]  9%|▉         | 36351/400000 [00:05<00:53, 6771.96it/s]  9%|▉         | 37103/400000 [00:05<00:52, 6978.03it/s]  9%|▉         | 37833/400000 [00:05<00:51, 7071.47it/s] 10%|▉         | 38542/400000 [00:05<00:51, 7062.65it/s] 10%|▉         | 39292/400000 [00:05<00:50, 7187.42it/s] 10%|█         | 40033/400000 [00:05<00:49, 7251.20it/s] 10%|█         | 40779/400000 [00:05<00:49, 7310.43it/s] 10%|█         | 41511/400000 [00:05<00:50, 7098.59it/s] 11%|█         | 42223/400000 [00:05<00:50, 7065.40it/s] 11%|█         | 42931/400000 [00:06<00:51, 6928.32it/s] 11%|█         | 43626/400000 [00:06<00:52, 6841.31it/s] 11%|█         | 44312/400000 [00:06<00:52, 6783.79it/s] 11%|█▏        | 45022/400000 [00:06<00:51, 6874.28it/s] 11%|█▏        | 45722/400000 [00:06<00:51, 6911.05it/s] 12%|█▏        | 46442/400000 [00:06<00:50, 6994.37it/s] 12%|█▏        | 47156/400000 [00:06<00:50, 7037.30it/s] 12%|█▏        | 47906/400000 [00:06<00:49, 7169.47it/s] 12%|█▏        | 48656/400000 [00:06<00:48, 7262.96it/s] 12%|█▏        | 49397/400000 [00:07<00:47, 7304.44it/s] 13%|█▎        | 50147/400000 [00:07<00:47, 7361.04it/s] 13%|█▎        | 50884/400000 [00:07<00:48, 7252.69it/s] 13%|█▎        | 51633/400000 [00:07<00:47, 7320.72it/s] 13%|█▎        | 52366/400000 [00:07<00:48, 7242.35it/s] 13%|█▎        | 53091/400000 [00:07<00:48, 7084.63it/s] 13%|█▎        | 53801/400000 [00:07<00:49, 6967.86it/s] 14%|█▎        | 54500/400000 [00:07<00:50, 6878.40it/s] 14%|█▍        | 55190/400000 [00:07<00:50, 6765.80it/s] 14%|█▍        | 55952/400000 [00:07<00:49, 7000.14it/s] 14%|█▍        | 56692/400000 [00:08<00:48, 7113.68it/s] 14%|█▍        | 57455/400000 [00:08<00:47, 7259.97it/s] 15%|█▍        | 58189/400000 [00:08<00:46, 7282.90it/s] 15%|█▍        | 58949/400000 [00:08<00:46, 7374.87it/s] 15%|█▍        | 59691/400000 [00:08<00:46, 7385.64it/s] 15%|█▌        | 60431/400000 [00:08<00:46, 7296.26it/s] 15%|█▌        | 61162/400000 [00:08<00:47, 7100.71it/s] 15%|█▌        | 61874/400000 [00:08<00:48, 6990.94it/s] 16%|█▌        | 62575/400000 [00:08<00:48, 6958.04it/s] 16%|█▌        | 63273/400000 [00:08<00:49, 6868.46it/s] 16%|█▌        | 63961/400000 [00:09<00:51, 6511.19it/s] 16%|█▌        | 64617/400000 [00:09<00:51, 6491.61it/s] 16%|█▋        | 65272/400000 [00:09<00:51, 6507.99it/s] 16%|█▋        | 65944/400000 [00:09<00:50, 6567.99it/s] 17%|█▋        | 66603/400000 [00:09<00:50, 6573.21it/s] 17%|█▋        | 67262/400000 [00:09<00:51, 6413.47it/s] 17%|█▋        | 67916/400000 [00:09<00:51, 6447.93it/s] 17%|█▋        | 68622/400000 [00:09<00:50, 6617.77it/s] 17%|█▋        | 69357/400000 [00:09<00:48, 6819.62it/s] 18%|█▊        | 70084/400000 [00:09<00:47, 6947.86it/s] 18%|█▊        | 70816/400000 [00:10<00:46, 7055.39it/s] 18%|█▊        | 71548/400000 [00:10<00:46, 7130.74it/s] 18%|█▊        | 72280/400000 [00:10<00:45, 7185.52it/s] 18%|█▊        | 73020/400000 [00:10<00:45, 7248.11it/s] 18%|█▊        | 73746/400000 [00:10<00:45, 7222.23it/s] 19%|█▊        | 74508/400000 [00:10<00:44, 7336.66it/s] 19%|█▉        | 75243/400000 [00:10<00:45, 7131.63it/s] 19%|█▉        | 75964/400000 [00:10<00:45, 7153.56it/s] 19%|█▉        | 76696/400000 [00:10<00:44, 7200.05it/s] 19%|█▉        | 77429/400000 [00:11<00:44, 7237.37it/s] 20%|█▉        | 78154/400000 [00:11<00:44, 7226.96it/s] 20%|█▉        | 78904/400000 [00:11<00:43, 7304.35it/s] 20%|█▉        | 79657/400000 [00:11<00:43, 7369.19it/s] 20%|██        | 80403/400000 [00:11<00:43, 7395.73it/s] 20%|██        | 81143/400000 [00:11<00:43, 7369.60it/s] 20%|██        | 81881/400000 [00:11<00:44, 7214.04it/s] 21%|██        | 82604/400000 [00:11<00:45, 6921.33it/s] 21%|██        | 83350/400000 [00:11<00:44, 7072.67it/s] 21%|██        | 84075/400000 [00:11<00:44, 7123.73it/s] 21%|██        | 84790/400000 [00:12<00:45, 6945.11it/s] 21%|██▏       | 85488/400000 [00:12<00:46, 6793.18it/s] 22%|██▏       | 86170/400000 [00:12<00:46, 6792.57it/s] 22%|██▏       | 86852/400000 [00:12<00:46, 6698.88it/s] 22%|██▏       | 87547/400000 [00:12<00:46, 6770.70it/s] 22%|██▏       | 88271/400000 [00:12<00:45, 6902.73it/s] 22%|██▏       | 88963/400000 [00:12<00:46, 6732.56it/s] 22%|██▏       | 89639/400000 [00:12<00:46, 6713.22it/s] 23%|██▎       | 90393/400000 [00:12<00:44, 6939.43it/s] 23%|██▎       | 91143/400000 [00:12<00:43, 7096.32it/s] 23%|██▎       | 91867/400000 [00:13<00:43, 7137.72it/s] 23%|██▎       | 92607/400000 [00:13<00:42, 7212.02it/s] 23%|██▎       | 93348/400000 [00:13<00:42, 7269.69it/s] 24%|██▎       | 94077/400000 [00:13<00:42, 7222.87it/s] 24%|██▎       | 94801/400000 [00:13<00:42, 7172.54it/s] 24%|██▍       | 95540/400000 [00:13<00:42, 7235.22it/s] 24%|██▍       | 96285/400000 [00:13<00:41, 7295.90it/s] 24%|██▍       | 97016/400000 [00:13<00:41, 7266.53it/s] 24%|██▍       | 97758/400000 [00:13<00:41, 7309.09it/s] 25%|██▍       | 98490/400000 [00:13<00:44, 6778.10it/s] 25%|██▍       | 99213/400000 [00:14<00:43, 6905.34it/s] 25%|██▍       | 99941/400000 [00:14<00:42, 7013.19it/s] 25%|██▌       | 100669/400000 [00:14<00:42, 7090.39it/s] 25%|██▌       | 101429/400000 [00:14<00:41, 7234.68it/s] 26%|██▌       | 102185/400000 [00:14<00:40, 7327.51it/s] 26%|██▌       | 102921/400000 [00:14<00:40, 7299.39it/s] 26%|██▌       | 103686/400000 [00:14<00:40, 7400.69it/s] 26%|██▌       | 104448/400000 [00:14<00:39, 7464.98it/s] 26%|██▋       | 105196/400000 [00:14<00:39, 7450.10it/s] 26%|██▋       | 105942/400000 [00:15<00:39, 7363.54it/s] 27%|██▋       | 106680/400000 [00:15<00:39, 7343.92it/s] 27%|██▋       | 107415/400000 [00:15<00:39, 7316.81it/s] 27%|██▋       | 108171/400000 [00:15<00:39, 7385.93it/s] 27%|██▋       | 108924/400000 [00:15<00:39, 7428.37it/s] 27%|██▋       | 109674/400000 [00:15<00:38, 7449.61it/s] 28%|██▊       | 110420/400000 [00:15<00:39, 7336.47it/s] 28%|██▊       | 111155/400000 [00:15<00:39, 7329.73it/s] 28%|██▊       | 111903/400000 [00:15<00:39, 7373.38it/s] 28%|██▊       | 112641/400000 [00:15<00:39, 7256.38it/s] 28%|██▊       | 113368/400000 [00:16<00:40, 7091.52it/s] 29%|██▊       | 114079/400000 [00:16<00:40, 6981.13it/s] 29%|██▊       | 114779/400000 [00:16<00:41, 6929.07it/s] 29%|██▉       | 115473/400000 [00:16<00:41, 6916.42it/s] 29%|██▉       | 116241/400000 [00:16<00:39, 7127.04it/s] 29%|██▉       | 117000/400000 [00:16<00:39, 7254.77it/s] 29%|██▉       | 117764/400000 [00:16<00:38, 7365.46it/s] 30%|██▉       | 118532/400000 [00:16<00:37, 7455.99it/s] 30%|██▉       | 119280/400000 [00:16<00:38, 7358.77it/s] 30%|███       | 120018/400000 [00:16<00:38, 7304.75it/s] 30%|███       | 120767/400000 [00:17<00:37, 7356.86it/s] 30%|███       | 121504/400000 [00:17<00:38, 7306.60it/s] 31%|███       | 122236/400000 [00:17<00:39, 7085.92it/s] 31%|███       | 122977/400000 [00:17<00:38, 7178.36it/s] 31%|███       | 123729/400000 [00:17<00:37, 7275.60it/s] 31%|███       | 124459/400000 [00:17<00:37, 7272.41it/s] 31%|███▏      | 125198/400000 [00:17<00:37, 7305.55it/s] 31%|███▏      | 125951/400000 [00:17<00:37, 7369.74it/s] 32%|███▏      | 126705/400000 [00:17<00:36, 7419.00it/s] 32%|███▏      | 127458/400000 [00:17<00:36, 7451.46it/s] 32%|███▏      | 128204/400000 [00:18<00:36, 7441.84it/s] 32%|███▏      | 128949/400000 [00:18<00:36, 7343.85it/s] 32%|███▏      | 129694/400000 [00:18<00:36, 7375.37it/s] 33%|███▎      | 130446/400000 [00:18<00:36, 7415.59it/s] 33%|███▎      | 131195/400000 [00:18<00:36, 7435.36it/s] 33%|███▎      | 131943/400000 [00:18<00:35, 7446.11it/s] 33%|███▎      | 132688/400000 [00:18<00:36, 7393.02it/s] 33%|███▎      | 133431/400000 [00:18<00:36, 7401.39it/s] 34%|███▎      | 134172/400000 [00:18<00:36, 7309.03it/s] 34%|███▎      | 134934/400000 [00:18<00:35, 7398.35it/s] 34%|███▍      | 135677/400000 [00:19<00:35, 7406.27it/s] 34%|███▍      | 136439/400000 [00:19<00:35, 7467.92it/s] 34%|███▍      | 137187/400000 [00:19<00:35, 7445.47it/s] 34%|███▍      | 137932/400000 [00:19<00:36, 7230.40it/s] 35%|███▍      | 138673/400000 [00:19<00:35, 7281.26it/s] 35%|███▍      | 139403/400000 [00:19<00:35, 7249.88it/s] 35%|███▌      | 140129/400000 [00:19<00:35, 7222.00it/s] 35%|███▌      | 140894/400000 [00:19<00:35, 7343.77it/s] 35%|███▌      | 141630/400000 [00:19<00:35, 7346.09it/s] 36%|███▌      | 142396/400000 [00:19<00:34, 7437.13it/s] 36%|███▌      | 143141/400000 [00:20<00:35, 7265.26it/s] 36%|███▌      | 143897/400000 [00:20<00:34, 7350.45it/s] 36%|███▌      | 144662/400000 [00:20<00:34, 7435.23it/s] 36%|███▋      | 145407/400000 [00:20<00:34, 7425.97it/s] 37%|███▋      | 146151/400000 [00:20<00:35, 7096.14it/s] 37%|███▋      | 146865/400000 [00:20<00:36, 6994.03it/s] 37%|███▋      | 147568/400000 [00:20<00:36, 6863.00it/s] 37%|███▋      | 148314/400000 [00:20<00:35, 7030.65it/s] 37%|███▋      | 149079/400000 [00:20<00:34, 7205.42it/s] 37%|███▋      | 149823/400000 [00:21<00:34, 7272.98it/s] 38%|███▊      | 150553/400000 [00:21<00:35, 6936.43it/s] 38%|███▊      | 151252/400000 [00:21<00:36, 6833.02it/s] 38%|███▊      | 151940/400000 [00:21<00:36, 6772.52it/s] 38%|███▊      | 152684/400000 [00:21<00:35, 6959.25it/s] 38%|███▊      | 153437/400000 [00:21<00:34, 7118.69it/s] 39%|███▊      | 154153/400000 [00:21<00:34, 7094.36it/s] 39%|███▊      | 154865/400000 [00:21<00:34, 7090.48it/s] 39%|███▉      | 155613/400000 [00:21<00:33, 7201.65it/s] 39%|███▉      | 156352/400000 [00:21<00:33, 7256.56it/s] 39%|███▉      | 157089/400000 [00:22<00:33, 7289.92it/s] 39%|███▉      | 157819/400000 [00:22<00:33, 7264.11it/s] 40%|███▉      | 158552/400000 [00:22<00:33, 7282.89it/s] 40%|███▉      | 159306/400000 [00:22<00:32, 7355.66it/s] 40%|████      | 160053/400000 [00:22<00:32, 7389.23it/s] 40%|████      | 160800/400000 [00:22<00:32, 7412.76it/s] 40%|████      | 161542/400000 [00:22<00:32, 7355.73it/s] 41%|████      | 162278/400000 [00:22<00:33, 7144.41it/s] 41%|████      | 162994/400000 [00:22<00:34, 6956.52it/s] 41%|████      | 163700/400000 [00:22<00:33, 6984.88it/s] 41%|████      | 164440/400000 [00:23<00:33, 7103.63it/s] 41%|████▏     | 165174/400000 [00:23<00:32, 7170.21it/s] 41%|████▏     | 165925/400000 [00:23<00:32, 7268.83it/s] 42%|████▏     | 166687/400000 [00:23<00:31, 7370.19it/s] 42%|████▏     | 167426/400000 [00:23<00:31, 7373.62it/s] 42%|████▏     | 168193/400000 [00:23<00:31, 7459.23it/s] 42%|████▏     | 168940/400000 [00:23<00:31, 7296.53it/s] 42%|████▏     | 169672/400000 [00:23<00:32, 7052.06it/s] 43%|████▎     | 170380/400000 [00:23<00:33, 6940.24it/s] 43%|████▎     | 171077/400000 [00:24<00:33, 6912.95it/s] 43%|████▎     | 171807/400000 [00:24<00:32, 7023.50it/s] 43%|████▎     | 172553/400000 [00:24<00:31, 7148.08it/s] 43%|████▎     | 173314/400000 [00:24<00:31, 7278.05it/s] 44%|████▎     | 174056/400000 [00:24<00:30, 7319.78it/s] 44%|████▎     | 174820/400000 [00:24<00:30, 7411.41it/s] 44%|████▍     | 175563/400000 [00:24<00:30, 7399.37it/s] 44%|████▍     | 176311/400000 [00:24<00:30, 7420.80it/s] 44%|████▍     | 177054/400000 [00:24<00:30, 7398.72it/s] 44%|████▍     | 177802/400000 [00:24<00:29, 7421.78it/s] 45%|████▍     | 178545/400000 [00:25<00:29, 7422.70it/s] 45%|████▍     | 179294/400000 [00:25<00:29, 7441.50it/s] 45%|████▌     | 180039/400000 [00:25<00:29, 7436.16it/s] 45%|████▌     | 180783/400000 [00:25<00:30, 7102.27it/s] 45%|████▌     | 181530/400000 [00:25<00:30, 7206.77it/s] 46%|████▌     | 182294/400000 [00:25<00:29, 7329.13it/s] 46%|████▌     | 183053/400000 [00:25<00:29, 7403.68it/s] 46%|████▌     | 183803/400000 [00:25<00:29, 7429.60it/s] 46%|████▌     | 184548/400000 [00:25<00:29, 7376.26it/s] 46%|████▋     | 185303/400000 [00:25<00:28, 7427.45it/s] 47%|████▋     | 186054/400000 [00:26<00:28, 7449.87it/s] 47%|████▋     | 186806/400000 [00:26<00:28, 7470.66it/s] 47%|████▋     | 187560/400000 [00:26<00:28, 7489.01it/s] 47%|████▋     | 188310/400000 [00:26<00:28, 7423.52it/s] 47%|████▋     | 189053/400000 [00:26<00:28, 7417.49it/s] 47%|████▋     | 189804/400000 [00:26<00:28, 7443.47it/s] 48%|████▊     | 190549/400000 [00:26<00:28, 7261.69it/s] 48%|████▊     | 191277/400000 [00:26<00:29, 7093.12it/s] 48%|████▊     | 191988/400000 [00:26<00:29, 7004.54it/s] 48%|████▊     | 192690/400000 [00:26<00:30, 6807.92it/s] 48%|████▊     | 193394/400000 [00:27<00:30, 6875.13it/s] 49%|████▊     | 194084/400000 [00:27<00:30, 6705.05it/s] 49%|████▊     | 194821/400000 [00:27<00:29, 6891.52it/s] 49%|████▉     | 195576/400000 [00:27<00:28, 7074.15it/s] 49%|████▉     | 196327/400000 [00:27<00:28, 7198.05it/s] 49%|████▉     | 197056/400000 [00:27<00:28, 7223.99it/s] 49%|████▉     | 197809/400000 [00:27<00:27, 7311.62it/s] 50%|████▉     | 198559/400000 [00:27<00:27, 7367.04it/s] 50%|████▉     | 199319/400000 [00:27<00:26, 7432.88it/s] 50%|█████     | 200064/400000 [00:27<00:27, 7364.65it/s] 50%|█████     | 200817/400000 [00:28<00:26, 7412.04it/s] 50%|█████     | 201559/400000 [00:28<00:26, 7406.76it/s] 51%|█████     | 202319/400000 [00:28<00:26, 7463.64it/s] 51%|█████     | 203079/400000 [00:28<00:26, 7502.66it/s] 51%|█████     | 203843/400000 [00:28<00:26, 7542.23it/s] 51%|█████     | 204598/400000 [00:28<00:26, 7338.40it/s] 51%|█████▏    | 205341/400000 [00:28<00:26, 7364.09it/s] 52%|█████▏    | 206079/400000 [00:28<00:26, 7352.79it/s] 52%|█████▏    | 206815/400000 [00:28<00:26, 7260.08it/s] 52%|█████▏    | 207542/400000 [00:28<00:26, 7143.78it/s] 52%|█████▏    | 208294/400000 [00:29<00:26, 7250.48it/s] 52%|█████▏    | 209049/400000 [00:29<00:26, 7337.13it/s] 52%|█████▏    | 209784/400000 [00:29<00:26, 7285.25it/s] 53%|█████▎    | 210535/400000 [00:29<00:25, 7350.64it/s] 53%|█████▎    | 211284/400000 [00:29<00:25, 7390.74it/s] 53%|█████▎    | 212030/400000 [00:29<00:25, 7409.71it/s] 53%|█████▎    | 212779/400000 [00:29<00:25, 7430.90it/s] 53%|█████▎    | 213524/400000 [00:29<00:25, 7434.05it/s] 54%|█████▎    | 214268/400000 [00:29<00:25, 7403.12it/s] 54%|█████▍    | 215009/400000 [00:30<00:25, 7243.28it/s] 54%|█████▍    | 215735/400000 [00:30<00:26, 7069.95it/s] 54%|█████▍    | 216444/400000 [00:30<00:26, 6971.33it/s] 54%|█████▍    | 217180/400000 [00:30<00:25, 7082.84it/s] 54%|█████▍    | 217947/400000 [00:30<00:25, 7248.38it/s] 55%|█████▍    | 218680/400000 [00:30<00:24, 7270.11it/s] 55%|█████▍    | 219441/400000 [00:30<00:24, 7368.49it/s] 55%|█████▌    | 220180/400000 [00:30<00:24, 7304.61it/s] 55%|█████▌    | 220920/400000 [00:30<00:24, 7330.52it/s] 55%|█████▌    | 221665/400000 [00:30<00:24, 7365.31it/s] 56%|█████▌    | 222413/400000 [00:31<00:24, 7397.14it/s] 56%|█████▌    | 223154/400000 [00:31<00:24, 7320.96it/s] 56%|█████▌    | 223895/400000 [00:31<00:23, 7347.09it/s] 56%|█████▌    | 224637/400000 [00:31<00:23, 7366.91it/s] 56%|█████▋    | 225387/400000 [00:31<00:23, 7404.21it/s] 57%|█████▋    | 226137/400000 [00:31<00:23, 7430.68it/s] 57%|█████▋    | 226881/400000 [00:31<00:23, 7413.09it/s] 57%|█████▋    | 227632/400000 [00:31<00:23, 7438.96it/s] 57%|█████▋    | 228377/400000 [00:31<00:23, 7322.41it/s] 57%|█████▋    | 229110/400000 [00:31<00:23, 7127.27it/s] 57%|█████▋    | 229857/400000 [00:32<00:23, 7224.77it/s] 58%|█████▊    | 230610/400000 [00:32<00:23, 7313.68it/s] 58%|█████▊    | 231343/400000 [00:32<00:23, 7313.80it/s] 58%|█████▊    | 232076/400000 [00:32<00:23, 7157.40it/s] 58%|█████▊    | 232794/400000 [00:32<00:23, 7155.77it/s] 58%|█████▊    | 233558/400000 [00:32<00:22, 7292.53it/s] 59%|█████▊    | 234315/400000 [00:32<00:22, 7371.42it/s] 59%|█████▉    | 235072/400000 [00:32<00:22, 7429.81it/s] 59%|█████▉    | 235816/400000 [00:32<00:22, 7339.04it/s] 59%|█████▉    | 236551/400000 [00:32<00:22, 7162.47it/s] 59%|█████▉    | 237269/400000 [00:33<00:23, 7050.24it/s] 59%|█████▉    | 237976/400000 [00:33<00:23, 6887.12it/s] 60%|█████▉    | 238667/400000 [00:33<00:23, 6831.44it/s] 60%|█████▉    | 239352/400000 [00:33<00:23, 6746.08it/s] 60%|██████    | 240028/400000 [00:33<00:23, 6670.38it/s] 60%|██████    | 240747/400000 [00:33<00:23, 6815.80it/s] 60%|██████    | 241495/400000 [00:33<00:22, 7000.76it/s] 61%|██████    | 242240/400000 [00:33<00:22, 7127.89it/s] 61%|██████    | 242988/400000 [00:33<00:21, 7228.30it/s] 61%|██████    | 243741/400000 [00:33<00:21, 7314.94it/s] 61%|██████    | 244475/400000 [00:34<00:21, 7275.99it/s] 61%|██████▏   | 245232/400000 [00:34<00:21, 7359.45it/s] 61%|██████▏   | 245969/400000 [00:34<00:20, 7353.13it/s] 62%|██████▏   | 246709/400000 [00:34<00:20, 7365.09it/s] 62%|██████▏   | 247447/400000 [00:34<00:20, 7351.98it/s] 62%|██████▏   | 248183/400000 [00:34<00:20, 7353.55it/s] 62%|██████▏   | 248931/400000 [00:34<00:20, 7388.99it/s] 62%|██████▏   | 249673/400000 [00:34<00:20, 7396.06it/s] 63%|██████▎   | 250415/400000 [00:34<00:20, 7402.11it/s] 63%|██████▎   | 251161/400000 [00:34<00:20, 7418.08it/s] 63%|██████▎   | 251903/400000 [00:35<00:20, 7191.14it/s] 63%|██████▎   | 252633/400000 [00:35<00:20, 7222.47it/s] 63%|██████▎   | 253357/400000 [00:35<00:20, 7216.85it/s] 64%|██████▎   | 254119/400000 [00:35<00:19, 7330.68it/s] 64%|██████▎   | 254881/400000 [00:35<00:19, 7412.54it/s] 64%|██████▍   | 255648/400000 [00:35<00:19, 7487.74it/s] 64%|██████▍   | 256413/400000 [00:35<00:19, 7535.48it/s] 64%|██████▍   | 257168/400000 [00:35<00:19, 7465.86it/s] 64%|██████▍   | 257926/400000 [00:35<00:18, 7497.15it/s] 65%|██████▍   | 258677/400000 [00:36<00:18, 7488.91it/s] 65%|██████▍   | 259427/400000 [00:36<00:18, 7426.85it/s] 65%|██████▌   | 260185/400000 [00:36<00:18, 7471.53it/s] 65%|██████▌   | 260934/400000 [00:36<00:18, 7474.18it/s] 65%|██████▌   | 261682/400000 [00:36<00:18, 7446.11it/s] 66%|██████▌   | 262427/400000 [00:36<00:18, 7423.36it/s] 66%|██████▌   | 263186/400000 [00:36<00:18, 7472.15it/s] 66%|██████▌   | 263934/400000 [00:36<00:18, 7409.78it/s] 66%|██████▌   | 264676/400000 [00:36<00:18, 7190.98it/s] 66%|██████▋   | 265417/400000 [00:36<00:18, 7254.46it/s] 67%|██████▋   | 266169/400000 [00:37<00:18, 7331.85it/s] 67%|██████▋   | 266928/400000 [00:37<00:17, 7406.57it/s] 67%|██████▋   | 267689/400000 [00:37<00:17, 7465.91it/s] 67%|██████▋   | 268452/400000 [00:37<00:17, 7512.06it/s] 67%|██████▋   | 269210/400000 [00:37<00:17, 7530.18it/s] 67%|██████▋   | 269964/400000 [00:37<00:17, 7269.91it/s] 68%|██████▊   | 270724/400000 [00:37<00:17, 7364.68it/s] 68%|██████▊   | 271480/400000 [00:37<00:17, 7420.95it/s] 68%|██████▊   | 272231/400000 [00:37<00:17, 7445.99it/s] 68%|██████▊   | 272992/400000 [00:37<00:16, 7493.74it/s] 68%|██████▊   | 273743/400000 [00:38<00:16, 7433.08it/s] 69%|██████▊   | 274487/400000 [00:38<00:16, 7429.94it/s] 69%|██████▉   | 275239/400000 [00:38<00:16, 7455.83it/s] 69%|██████▉   | 275994/400000 [00:38<00:16, 7481.93it/s] 69%|██████▉   | 276755/400000 [00:38<00:16, 7517.50it/s] 69%|██████▉   | 277507/400000 [00:38<00:16, 7256.80it/s] 70%|██████▉   | 278235/400000 [00:38<00:17, 7059.42it/s] 70%|██████▉   | 278944/400000 [00:38<00:17, 6941.83it/s] 70%|██████▉   | 279700/400000 [00:38<00:16, 7114.01it/s] 70%|███████   | 280461/400000 [00:38<00:16, 7255.24it/s] 70%|███████   | 281222/400000 [00:39<00:16, 7355.54it/s] 70%|███████   | 281960/400000 [00:39<00:16, 7214.55it/s] 71%|███████   | 282699/400000 [00:39<00:16, 7265.02it/s] 71%|███████   | 283456/400000 [00:39<00:15, 7353.46it/s] 71%|███████   | 284199/400000 [00:39<00:15, 7374.44it/s] 71%|███████   | 284938/400000 [00:39<00:15, 7243.70it/s] 71%|███████▏  | 285664/400000 [00:39<00:15, 7181.43it/s] 72%|███████▏  | 286384/400000 [00:39<00:16, 7018.11it/s] 72%|███████▏  | 287088/400000 [00:39<00:16, 6924.88it/s] 72%|███████▏  | 287812/400000 [00:40<00:15, 7016.25it/s] 72%|███████▏  | 288547/400000 [00:40<00:15, 7111.90it/s] 72%|███████▏  | 289299/400000 [00:40<00:15, 7227.44it/s] 73%|███████▎  | 290034/400000 [00:40<00:15, 7261.64it/s] 73%|███████▎  | 290785/400000 [00:40<00:14, 7334.03it/s] 73%|███████▎  | 291520/400000 [00:40<00:15, 7229.77it/s] 73%|███████▎  | 292277/400000 [00:40<00:14, 7326.70it/s] 73%|███████▎  | 293034/400000 [00:40<00:14, 7395.92it/s] 73%|███████▎  | 293790/400000 [00:40<00:14, 7443.97it/s] 74%|███████▎  | 294542/400000 [00:40<00:14, 7463.91it/s] 74%|███████▍  | 295289/400000 [00:41<00:14, 7378.53it/s] 74%|███████▍  | 296045/400000 [00:41<00:13, 7431.73it/s] 74%|███████▍  | 296804/400000 [00:41<00:13, 7475.08it/s] 74%|███████▍  | 297565/400000 [00:41<00:13, 7514.42it/s] 75%|███████▍  | 298317/400000 [00:41<00:13, 7393.07it/s] 75%|███████▍  | 299057/400000 [00:41<00:14, 7102.91it/s] 75%|███████▍  | 299771/400000 [00:41<00:15, 6670.74it/s] 75%|███████▌  | 300446/400000 [00:41<00:14, 6657.77it/s] 75%|███████▌  | 301175/400000 [00:41<00:14, 6834.24it/s] 75%|███████▌  | 301937/400000 [00:41<00:13, 7049.90it/s] 76%|███████▌  | 302648/400000 [00:42<00:13, 7059.97it/s] 76%|███████▌  | 303358/400000 [00:42<00:13, 6965.32it/s] 76%|███████▌  | 304058/400000 [00:42<00:14, 6816.45it/s] 76%|███████▌  | 304743/400000 [00:42<00:14, 6791.62it/s] 76%|███████▋  | 305425/400000 [00:42<00:13, 6780.36it/s] 77%|███████▋  | 306153/400000 [00:42<00:13, 6922.78it/s] 77%|███████▋  | 306852/400000 [00:42<00:13, 6940.47it/s] 77%|███████▋  | 307588/400000 [00:42<00:13, 7059.07it/s] 77%|███████▋  | 308310/400000 [00:42<00:12, 7105.14it/s] 77%|███████▋  | 309022/400000 [00:42<00:13, 6976.81it/s] 77%|███████▋  | 309721/400000 [00:43<00:13, 6930.62it/s] 78%|███████▊  | 310483/400000 [00:43<00:12, 7121.40it/s] 78%|███████▊  | 311238/400000 [00:43<00:12, 7244.42it/s] 78%|███████▊  | 312001/400000 [00:43<00:11, 7353.19it/s] 78%|███████▊  | 312739/400000 [00:43<00:11, 7332.72it/s] 78%|███████▊  | 313474/400000 [00:43<00:11, 7314.46it/s] 79%|███████▊  | 314235/400000 [00:43<00:11, 7398.52it/s] 79%|███████▊  | 314997/400000 [00:43<00:11, 7463.09it/s] 79%|███████▉  | 315761/400000 [00:43<00:11, 7513.92it/s] 79%|███████▉  | 316519/400000 [00:43<00:11, 7531.58it/s] 79%|███████▉  | 317273/400000 [00:44<00:11, 7468.95it/s] 80%|███████▉  | 318032/400000 [00:44<00:10, 7504.32it/s] 80%|███████▉  | 318793/400000 [00:44<00:10, 7534.20it/s] 80%|███████▉  | 319554/400000 [00:44<00:10, 7555.56it/s] 80%|████████  | 320320/400000 [00:44<00:10, 7584.84it/s] 80%|████████  | 321079/400000 [00:44<00:10, 7380.63it/s] 80%|████████  | 321819/400000 [00:44<00:10, 7169.26it/s] 81%|████████  | 322539/400000 [00:44<00:11, 7009.87it/s] 81%|████████  | 323243/400000 [00:44<00:11, 6842.08it/s] 81%|████████  | 323930/400000 [00:45<00:11, 6848.79it/s] 81%|████████  | 324628/400000 [00:45<00:10, 6886.03it/s] 81%|████████▏ | 325318/400000 [00:45<00:10, 6815.54it/s] 82%|████████▏ | 326001/400000 [00:45<00:10, 6790.13it/s] 82%|████████▏ | 326688/400000 [00:45<00:10, 6813.01it/s] 82%|████████▏ | 327447/400000 [00:45<00:10, 7028.36it/s] 82%|████████▏ | 328204/400000 [00:45<00:09, 7182.19it/s] 82%|████████▏ | 328925/400000 [00:45<00:10, 7074.27it/s] 82%|████████▏ | 329635/400000 [00:45<00:10, 6823.01it/s] 83%|████████▎ | 330321/400000 [00:45<00:10, 6702.36it/s] 83%|████████▎ | 330995/400000 [00:46<00:10, 6641.14it/s] 83%|████████▎ | 331667/400000 [00:46<00:10, 6662.81it/s] 83%|████████▎ | 332419/400000 [00:46<00:09, 6897.77it/s] 83%|████████▎ | 333172/400000 [00:46<00:09, 7074.85it/s] 83%|████████▎ | 333890/400000 [00:46<00:09, 7105.01it/s] 84%|████████▎ | 334603/400000 [00:46<00:09, 7000.10it/s] 84%|████████▍ | 335321/400000 [00:46<00:09, 7050.76it/s] 84%|████████▍ | 336085/400000 [00:46<00:08, 7216.99it/s] 84%|████████▍ | 336846/400000 [00:46<00:08, 7328.52it/s] 84%|████████▍ | 337610/400000 [00:46<00:08, 7417.23it/s] 85%|████████▍ | 338356/400000 [00:47<00:08, 7427.58it/s] 85%|████████▍ | 339102/400000 [00:47<00:08, 7435.08it/s] 85%|████████▍ | 339863/400000 [00:47<00:08, 7486.63it/s] 85%|████████▌ | 340623/400000 [00:47<00:07, 7519.95it/s] 85%|████████▌ | 341383/400000 [00:47<00:07, 7542.27it/s] 86%|████████▌ | 342138/400000 [00:47<00:07, 7484.66it/s] 86%|████████▌ | 342887/400000 [00:47<00:07, 7358.53it/s] 86%|████████▌ | 343624/400000 [00:47<00:07, 7142.11it/s] 86%|████████▌ | 344341/400000 [00:47<00:08, 6923.67it/s] 86%|████████▋ | 345054/400000 [00:47<00:07, 6983.76it/s] 86%|████████▋ | 345774/400000 [00:48<00:07, 7046.20it/s] 87%|████████▋ | 346481/400000 [00:48<00:07, 6927.49it/s] 87%|████████▋ | 347216/400000 [00:48<00:07, 7046.40it/s] 87%|████████▋ | 347978/400000 [00:48<00:07, 7207.14it/s] 87%|████████▋ | 348721/400000 [00:48<00:07, 7270.91it/s] 87%|████████▋ | 349450/400000 [00:48<00:06, 7231.52it/s] 88%|████████▊ | 350181/400000 [00:48<00:06, 7236.00it/s] 88%|████████▊ | 350906/400000 [00:48<00:06, 7145.55it/s] 88%|████████▊ | 351644/400000 [00:48<00:06, 7213.28it/s] 88%|████████▊ | 352399/400000 [00:49<00:06, 7308.40it/s] 88%|████████▊ | 353142/400000 [00:49<00:06, 7342.48it/s] 88%|████████▊ | 353886/400000 [00:49<00:06, 7368.76it/s] 89%|████████▊ | 354635/400000 [00:49<00:06, 7402.40it/s] 89%|████████▉ | 355376/400000 [00:49<00:06, 7352.53it/s] 89%|████████▉ | 356114/400000 [00:49<00:05, 7360.31it/s] 89%|████████▉ | 356852/400000 [00:49<00:05, 7365.52it/s] 89%|████████▉ | 357595/400000 [00:49<00:05, 7382.48it/s] 90%|████████▉ | 358339/400000 [00:49<00:05, 7398.17it/s] 90%|████████▉ | 359079/400000 [00:49<00:05, 7300.72it/s] 90%|████████▉ | 359815/400000 [00:50<00:05, 7317.35it/s] 90%|█████████ | 360556/400000 [00:50<00:05, 7342.51it/s] 90%|█████████ | 361292/400000 [00:50<00:05, 7345.07it/s] 91%|█████████ | 362049/400000 [00:50<00:05, 7410.93it/s] 91%|█████████ | 362804/400000 [00:50<00:04, 7451.41it/s] 91%|█████████ | 363550/400000 [00:50<00:05, 7179.50it/s] 91%|█████████ | 364271/400000 [00:50<00:05, 7001.16it/s] 91%|█████████ | 364974/400000 [00:50<00:05, 6928.26it/s] 91%|█████████▏| 365721/400000 [00:50<00:04, 7080.92it/s] 92%|█████████▏| 366474/400000 [00:50<00:04, 7208.48it/s] 92%|█████████▏| 367216/400000 [00:51<00:04, 7268.44it/s] 92%|█████████▏| 367945/400000 [00:51<00:04, 7113.51it/s] 92%|█████████▏| 368659/400000 [00:51<00:04, 7094.76it/s] 92%|█████████▏| 369407/400000 [00:51<00:04, 7204.40it/s] 93%|█████████▎| 370129/400000 [00:51<00:04, 7054.82it/s] 93%|█████████▎| 370877/400000 [00:51<00:04, 7176.76it/s] 93%|█████████▎| 371638/400000 [00:51<00:03, 7298.85it/s] 93%|█████████▎| 372370/400000 [00:51<00:03, 7068.32it/s] 93%|█████████▎| 373125/400000 [00:51<00:03, 7204.32it/s] 93%|█████████▎| 373868/400000 [00:51<00:03, 7270.24it/s] 94%|█████████▎| 374622/400000 [00:52<00:03, 7348.81it/s] 94%|█████████▍| 375359/400000 [00:52<00:03, 7295.68it/s] 94%|█████████▍| 376101/400000 [00:52<00:03, 7331.98it/s] 94%|█████████▍| 376853/400000 [00:52<00:03, 7384.75it/s] 94%|█████████▍| 377593/400000 [00:52<00:03, 7362.60it/s] 95%|█████████▍| 378330/400000 [00:52<00:02, 7339.05it/s] 95%|█████████▍| 379072/400000 [00:52<00:02, 7360.86it/s] 95%|█████████▍| 379820/400000 [00:52<00:02, 7394.87it/s] 95%|█████████▌| 380560/400000 [00:52<00:02, 7343.58it/s] 95%|█████████▌| 381295/400000 [00:52<00:02, 7299.28it/s] 96%|█████████▌| 382026/400000 [00:53<00:02, 7189.65it/s] 96%|█████████▌| 382778/400000 [00:53<00:02, 7283.70it/s] 96%|█████████▌| 383539/400000 [00:53<00:02, 7376.74it/s] 96%|█████████▌| 384294/400000 [00:53<00:02, 7427.09it/s] 96%|█████████▋| 385038/400000 [00:53<00:02, 7423.18it/s] 96%|█████████▋| 385781/400000 [00:53<00:01, 7325.97it/s] 97%|█████████▋| 386531/400000 [00:53<00:01, 7374.62it/s] 97%|█████████▋| 387273/400000 [00:53<00:01, 7386.70it/s] 97%|█████████▋| 388018/400000 [00:53<00:01, 7404.01it/s] 97%|█████████▋| 388762/400000 [00:53<00:01, 7414.05it/s] 97%|█████████▋| 389504/400000 [00:54<00:01, 7354.74it/s] 98%|█████████▊| 390246/400000 [00:54<00:01, 7373.59it/s] 98%|█████████▊| 390990/400000 [00:54<00:01, 7386.60it/s] 98%|█████████▊| 391744/400000 [00:54<00:01, 7431.50it/s] 98%|█████████▊| 392488/400000 [00:54<00:01, 7239.16it/s] 98%|█████████▊| 393214/400000 [00:54<00:00, 7052.09it/s] 98%|█████████▊| 393945/400000 [00:54<00:00, 7124.82it/s] 99%|█████████▊| 394683/400000 [00:54<00:00, 7198.47it/s] 99%|█████████▉| 395423/400000 [00:54<00:00, 7257.37it/s] 99%|█████████▉| 396158/400000 [00:55<00:00, 7282.29it/s] 99%|█████████▉| 396901/400000 [00:55<00:00, 7324.35it/s] 99%|█████████▉| 397634/400000 [00:55<00:00, 7234.24it/s]100%|█████████▉| 398359/400000 [00:55<00:00, 7212.51it/s]100%|█████████▉| 399104/400000 [00:55<00:00, 7276.84it/s]100%|█████████▉| 399855/400000 [00:55<00:00, 7343.05it/s]100%|█████████▉| 399999/400000 [00:55<00:00, 7202.88it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7febec4850f0>, <torchtext.data.dataset.TabularDataset object at 0x7febec485240>, <torchtext.vocab.Vocab object at 0x7febec485160>), {}) 


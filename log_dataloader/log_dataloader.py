
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/ef95df9f077ea4a54e765587d8a0bd21d47e9e81', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'ef95df9f077ea4a54e765587d8a0bd21d47e9e81', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/ef95df9f077ea4a54e765587d8a0bd21d47e9e81

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/ef95df9f077ea4a54e765587d8a0bd21d47e9e81

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/ef95df9f077ea4a54e765587d8a0bd21d47e9e81

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7efc9cf862f0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7efc9cf862f0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7efd0efd40d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7efd0efd40d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7efca67689d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7efca67689d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7efcbc8b88c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7efcbc8b88c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7efcbc8b88c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:12, 135783.48it/s] 87%|████████▋ | 8642560/9912422 [00:00<00:06, 193844.04it/s]9920512it [00:00, 42728708.03it/s]                           
0it [00:00, ?it/s]32768it [00:00, 463684.79it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 11978666.82it/s]         
0it [00:00, ?it/s]8192it [00:00, 145817.02it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efc9d192d30>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efc9d192e48>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7efcbc8b8510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7efcbc8b8510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7efcbc8b8510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:48,  3.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:48,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:48,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:48,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:47,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:47,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:47,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:33,  4.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:33,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:33,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:33,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:33,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:32,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:32,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:32,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:32,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:31,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:31,  4.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:22,  6.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:22,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:22,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:22,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:22,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:21,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:21,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:21,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:21,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:21,  6.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:15,  8.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:15,  8.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:15,  8.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:15,  8.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:14,  8.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:14,  8.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:14,  8.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:14,  8.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:14,  8.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  8.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:14,  8.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:00<00:10, 12.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:10, 12.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:10, 12.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:10, 12.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:10, 12.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:09, 12.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:09, 12.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:09, 12.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:09, 12.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:09, 12.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:00<00:07, 16.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:07, 16.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:07, 16.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:07, 16.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:06, 16.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:06, 16.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:06, 16.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:06, 16.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:06, 16.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:06, 16.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:06, 16.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:00<00:04, 21.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:04, 21.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:04, 21.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:04, 21.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:04, 21.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:00<00:04, 21.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 21.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 21.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 21.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 21.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 27.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 27.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 27.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 27.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 27.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 27.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 27.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 27.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 27.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 27.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 35.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 35.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 35.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 35.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 35.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 35.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 35.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 35.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 35.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 35.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 42.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 42.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 42.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 42.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 42.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 42.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 42.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 42.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 42.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 42.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 50.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 50.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 50.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 50.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 50.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 50.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 50.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 50.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 50.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 50.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 57.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 57.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 57.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 57.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 57.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 57.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 57.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 57.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 57.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 63.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 63.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 68.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 68.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 68.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 68.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 68.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 68.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 68.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 68.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 68.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 68.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 73.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 73.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 73.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 73.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 73.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 73.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 73.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 73.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 73.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 73.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 76.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 76.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 76.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 76.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 76.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 76.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 76.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 76.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 76.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 76.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 78.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 78.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 78.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 78.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 78.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 78.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 78.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.26s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.26s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.26s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.34s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.26s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 75.99 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.34s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.35s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.28 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.35s/ url]
0 examples [00:00, ? examples/s]2020-05-29 15:20:27.428387: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-29 15:20:27.432437: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-05-29 15:20:27.432573: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5616b3ed6fc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-29 15:20:27.432586: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
75 examples [00:00, 747.08 examples/s]170 examples [00:00, 796.65 examples/s]271 examples [00:00, 850.43 examples/s]381 examples [00:00, 911.37 examples/s]483 examples [00:00, 940.90 examples/s]589 examples [00:00, 971.54 examples/s]684 examples [00:00, 950.01 examples/s]788 examples [00:00, 973.06 examples/s]897 examples [00:00, 1004.85 examples/s]1009 examples [00:01, 1034.44 examples/s]1112 examples [00:01, 1011.90 examples/s]1219 examples [00:01, 1028.15 examples/s]1326 examples [00:01, 1039.83 examples/s]1430 examples [00:01, 1031.20 examples/s]1543 examples [00:01, 1057.28 examples/s]1651 examples [00:01, 1063.66 examples/s]1759 examples [00:01, 1067.25 examples/s]1866 examples [00:01, 1058.30 examples/s]1975 examples [00:01, 1067.19 examples/s]2089 examples [00:02, 1086.03 examples/s]2202 examples [00:02, 1097.18 examples/s]2312 examples [00:02, 1086.06 examples/s]2421 examples [00:02, 1072.95 examples/s]2529 examples [00:02, 1026.97 examples/s]2641 examples [00:02, 1051.66 examples/s]2753 examples [00:02, 1067.12 examples/s]2864 examples [00:02, 1079.61 examples/s]2973 examples [00:02, 1052.90 examples/s]3081 examples [00:02, 1060.31 examples/s]3192 examples [00:03, 1073.96 examples/s]3302 examples [00:03, 1081.28 examples/s]3411 examples [00:03, 1067.95 examples/s]3519 examples [00:03, 1068.98 examples/s]3633 examples [00:03, 1087.40 examples/s]3745 examples [00:03, 1094.71 examples/s]3855 examples [00:03, 1015.10 examples/s]3960 examples [00:03, 1023.56 examples/s]4069 examples [00:03, 1040.98 examples/s]4175 examples [00:03, 1045.61 examples/s]4282 examples [00:04, 1051.69 examples/s]4396 examples [00:04, 1076.08 examples/s]4510 examples [00:04, 1092.44 examples/s]4620 examples [00:04, 1077.57 examples/s]4730 examples [00:04, 1083.55 examples/s]4839 examples [00:04, 1082.58 examples/s]4948 examples [00:04, 1082.39 examples/s]5057 examples [00:04, 1082.40 examples/s]5166 examples [00:04, 1079.77 examples/s]5281 examples [00:05, 1097.56 examples/s]5394 examples [00:05, 1105.11 examples/s]5506 examples [00:05, 1108.95 examples/s]5617 examples [00:05, 1096.49 examples/s]5727 examples [00:05, 1093.80 examples/s]5837 examples [00:05, 1086.38 examples/s]5946 examples [00:05, 1051.89 examples/s]6052 examples [00:05, 1044.17 examples/s]6157 examples [00:05, 1026.95 examples/s]6262 examples [00:05, 1031.11 examples/s]6372 examples [00:06, 1049.30 examples/s]6482 examples [00:06, 1062.29 examples/s]6594 examples [00:06, 1076.42 examples/s]6704 examples [00:06, 1081.00 examples/s]6816 examples [00:06, 1090.83 examples/s]6926 examples [00:06, 1073.89 examples/s]7036 examples [00:06, 1079.72 examples/s]7146 examples [00:06, 1085.65 examples/s]7255 examples [00:06, 1083.62 examples/s]7366 examples [00:06, 1091.21 examples/s]7477 examples [00:07, 1096.32 examples/s]7587 examples [00:07, 1096.16 examples/s]7697 examples [00:07, 1093.88 examples/s]7807 examples [00:07, 1028.24 examples/s]7911 examples [00:07, 1025.62 examples/s]8023 examples [00:07, 1050.93 examples/s]8130 examples [00:07, 1054.54 examples/s]8236 examples [00:07, 1036.43 examples/s]8345 examples [00:07, 1050.17 examples/s]8453 examples [00:07, 1057.88 examples/s]8568 examples [00:08, 1082.68 examples/s]8683 examples [00:08, 1099.76 examples/s]8796 examples [00:08, 1107.75 examples/s]8908 examples [00:08, 1109.70 examples/s]9021 examples [00:08, 1115.16 examples/s]9133 examples [00:08, 1111.88 examples/s]9245 examples [00:08, 1096.04 examples/s]9360 examples [00:08, 1109.70 examples/s]9472 examples [00:08, 1107.78 examples/s]9584 examples [00:08, 1109.79 examples/s]9696 examples [00:09, 1104.62 examples/s]9807 examples [00:09, 1085.23 examples/s]9921 examples [00:09, 1100.53 examples/s]10032 examples [00:09, 1059.12 examples/s]10147 examples [00:09, 1083.80 examples/s]10259 examples [00:09, 1091.62 examples/s]10373 examples [00:09, 1103.55 examples/s]10487 examples [00:09, 1112.38 examples/s]10600 examples [00:09, 1116.73 examples/s]10716 examples [00:10, 1128.08 examples/s]10832 examples [00:10, 1136.29 examples/s]10946 examples [00:10, 1091.80 examples/s]11062 examples [00:10, 1110.76 examples/s]11178 examples [00:10, 1124.83 examples/s]11292 examples [00:10, 1127.93 examples/s]11407 examples [00:10, 1133.67 examples/s]11521 examples [00:10, 1095.57 examples/s]11631 examples [00:10, 1078.01 examples/s]11740 examples [00:10, 1078.72 examples/s]11856 examples [00:11, 1099.16 examples/s]11971 examples [00:11, 1112.86 examples/s]12086 examples [00:11, 1122.99 examples/s]12201 examples [00:11, 1128.93 examples/s]12315 examples [00:11, 1113.76 examples/s]12427 examples [00:11, 1112.55 examples/s]12539 examples [00:11, 1080.01 examples/s]12649 examples [00:11, 1085.51 examples/s]12761 examples [00:11, 1093.71 examples/s]12871 examples [00:11, 1088.96 examples/s]12984 examples [00:12, 1100.25 examples/s]13095 examples [00:12, 1103.03 examples/s]13207 examples [00:12, 1107.19 examples/s]13319 examples [00:12, 1110.26 examples/s]13431 examples [00:12, 1092.58 examples/s]13541 examples [00:12, 1092.53 examples/s]13656 examples [00:12, 1107.89 examples/s]13769 examples [00:12, 1112.46 examples/s]13881 examples [00:12, 1106.89 examples/s]13992 examples [00:12, 1098.35 examples/s]14104 examples [00:13, 1102.84 examples/s]14218 examples [00:13, 1111.03 examples/s]14333 examples [00:13, 1120.67 examples/s]14447 examples [00:13, 1124.79 examples/s]14560 examples [00:13, 1117.49 examples/s]14675 examples [00:13, 1124.26 examples/s]14789 examples [00:13, 1128.17 examples/s]14902 examples [00:13, 1125.35 examples/s]15017 examples [00:13, 1132.55 examples/s]15131 examples [00:14, 1118.73 examples/s]15243 examples [00:14, 1109.80 examples/s]15355 examples [00:14, 1096.28 examples/s]15468 examples [00:14, 1104.65 examples/s]15579 examples [00:14, 1097.04 examples/s]15689 examples [00:14, 1093.48 examples/s]15806 examples [00:14, 1113.74 examples/s]15918 examples [00:14, 1094.47 examples/s]16028 examples [00:14, 1090.13 examples/s]16139 examples [00:14, 1094.12 examples/s]16253 examples [00:15, 1106.92 examples/s]16372 examples [00:15, 1128.39 examples/s]16486 examples [00:15, 1129.32 examples/s]16600 examples [00:15, 1128.12 examples/s]16717 examples [00:15, 1137.87 examples/s]16832 examples [00:15, 1138.63 examples/s]16946 examples [00:15, 1127.51 examples/s]17059 examples [00:15, 1117.27 examples/s]17171 examples [00:15, 1098.11 examples/s]17281 examples [00:15, 1084.48 examples/s]17390 examples [00:16, 1083.52 examples/s]17502 examples [00:16, 1092.71 examples/s]17613 examples [00:16, 1096.23 examples/s]17726 examples [00:16, 1103.72 examples/s]17837 examples [00:16, 1103.20 examples/s]17948 examples [00:16, 1097.63 examples/s]18058 examples [00:16, 1096.74 examples/s]18168 examples [00:16, 1083.84 examples/s]18278 examples [00:16, 1088.28 examples/s]18387 examples [00:16, 1081.96 examples/s]18496 examples [00:17, 1073.57 examples/s]18607 examples [00:17, 1081.58 examples/s]18719 examples [00:17, 1092.31 examples/s]18831 examples [00:17, 1098.23 examples/s]18941 examples [00:17, 1066.72 examples/s]19048 examples [00:17, 1050.54 examples/s]19159 examples [00:17, 1066.96 examples/s]19273 examples [00:17, 1085.73 examples/s]19382 examples [00:17, 1069.09 examples/s]19491 examples [00:17, 1073.10 examples/s]19600 examples [00:18, 1077.68 examples/s]19709 examples [00:18, 1078.98 examples/s]19820 examples [00:18, 1086.66 examples/s]19931 examples [00:18, 1091.51 examples/s]20041 examples [00:18, 1036.18 examples/s]20153 examples [00:18, 1057.63 examples/s]20263 examples [00:18, 1069.27 examples/s]20375 examples [00:18, 1082.06 examples/s]20486 examples [00:18, 1089.43 examples/s]20597 examples [00:19, 1092.73 examples/s]20707 examples [00:19, 1092.76 examples/s]20817 examples [00:19, 1074.51 examples/s]20926 examples [00:19, 1077.89 examples/s]21034 examples [00:19, 1055.82 examples/s]21145 examples [00:19, 1070.59 examples/s]21256 examples [00:19, 1080.39 examples/s]21365 examples [00:19, 1071.09 examples/s]21475 examples [00:19, 1078.23 examples/s]21588 examples [00:19, 1092.53 examples/s]21702 examples [00:20, 1105.99 examples/s]21815 examples [00:20, 1110.32 examples/s]21931 examples [00:20, 1121.69 examples/s]22048 examples [00:20, 1134.21 examples/s]22167 examples [00:20, 1148.15 examples/s]22284 examples [00:20, 1152.35 examples/s]22403 examples [00:20, 1163.24 examples/s]22520 examples [00:20, 1157.05 examples/s]22636 examples [00:20, 1139.08 examples/s]22751 examples [00:20, 1088.65 examples/s]22861 examples [00:21, 1086.86 examples/s]22971 examples [00:21, 1087.46 examples/s]23081 examples [00:21, 1048.47 examples/s]23192 examples [00:21, 1063.68 examples/s]23300 examples [00:21, 1066.73 examples/s]23411 examples [00:21, 1077.19 examples/s]23519 examples [00:21, 1064.41 examples/s]23626 examples [00:21, 1047.84 examples/s]23737 examples [00:21, 1063.67 examples/s]23847 examples [00:21, 1071.23 examples/s]23955 examples [00:22, 1060.29 examples/s]24062 examples [00:22, 1053.62 examples/s]24168 examples [00:22, 1052.12 examples/s]24281 examples [00:22, 1071.70 examples/s]24389 examples [00:22, 1047.07 examples/s]24495 examples [00:22, 1049.57 examples/s]24601 examples [00:22, 1034.91 examples/s]24709 examples [00:22, 1047.42 examples/s]24818 examples [00:22, 1059.48 examples/s]24928 examples [00:23, 1069.16 examples/s]25039 examples [00:23, 1080.13 examples/s]25148 examples [00:23, 1070.60 examples/s]25260 examples [00:23, 1083.14 examples/s]25372 examples [00:23, 1092.85 examples/s]25484 examples [00:23, 1100.03 examples/s]25601 examples [00:23, 1118.87 examples/s]25715 examples [00:23, 1124.33 examples/s]25831 examples [00:23, 1132.95 examples/s]25948 examples [00:23, 1142.27 examples/s]26063 examples [00:24, 1099.29 examples/s]26177 examples [00:24, 1110.94 examples/s]26289 examples [00:24, 1103.81 examples/s]26400 examples [00:24, 1096.28 examples/s]26515 examples [00:24, 1111.17 examples/s]26631 examples [00:24, 1123.28 examples/s]26747 examples [00:24, 1131.38 examples/s]26861 examples [00:24, 1081.32 examples/s]26970 examples [00:24, 1077.72 examples/s]27079 examples [00:24, 1080.10 examples/s]27188 examples [00:25, 1081.62 examples/s]27301 examples [00:25, 1094.49 examples/s]27411 examples [00:25, 1087.99 examples/s]27522 examples [00:25, 1094.29 examples/s]27633 examples [00:25, 1097.89 examples/s]27747 examples [00:25, 1107.55 examples/s]27863 examples [00:25, 1122.43 examples/s]27976 examples [00:25, 1123.97 examples/s]28090 examples [00:25, 1127.66 examples/s]28203 examples [00:25, 1108.50 examples/s]28314 examples [00:26, 1099.05 examples/s]28431 examples [00:26, 1118.82 examples/s]28545 examples [00:26, 1124.12 examples/s]28662 examples [00:26, 1137.22 examples/s]28781 examples [00:26, 1150.85 examples/s]28897 examples [00:26, 1151.95 examples/s]29013 examples [00:26, 1144.03 examples/s]29128 examples [00:26, 1128.11 examples/s]29243 examples [00:26, 1133.05 examples/s]29357 examples [00:26, 1125.61 examples/s]29470 examples [00:27, 1090.09 examples/s]29583 examples [00:27, 1101.48 examples/s]29694 examples [00:27, 1100.29 examples/s]29809 examples [00:27, 1113.74 examples/s]29921 examples [00:27, 1108.21 examples/s]30032 examples [00:27, 1029.37 examples/s]30142 examples [00:27, 1047.76 examples/s]30253 examples [00:27, 1064.38 examples/s]30365 examples [00:27, 1079.26 examples/s]30476 examples [00:28, 1086.51 examples/s]30589 examples [00:28, 1098.25 examples/s]30700 examples [00:28, 1092.76 examples/s]30810 examples [00:28, 1083.24 examples/s]30927 examples [00:28, 1106.89 examples/s]31038 examples [00:28, 1079.58 examples/s]31151 examples [00:28, 1091.97 examples/s]31263 examples [00:28, 1097.99 examples/s]31373 examples [00:28, 1087.75 examples/s]31485 examples [00:28, 1095.99 examples/s]31598 examples [00:29, 1104.60 examples/s]31712 examples [00:29, 1113.19 examples/s]31824 examples [00:29, 1106.03 examples/s]31935 examples [00:29, 1083.35 examples/s]32050 examples [00:29, 1102.18 examples/s]32161 examples [00:29, 1098.11 examples/s]32271 examples [00:29, 1081.20 examples/s]32381 examples [00:29, 1085.51 examples/s]32490 examples [00:29, 1067.70 examples/s]32603 examples [00:29, 1083.18 examples/s]32714 examples [00:30, 1089.19 examples/s]32824 examples [00:30, 1075.31 examples/s]32937 examples [00:30, 1086.26 examples/s]33048 examples [00:30, 1092.49 examples/s]33158 examples [00:30, 1085.21 examples/s]33270 examples [00:30, 1093.07 examples/s]33382 examples [00:30, 1099.90 examples/s]33496 examples [00:30, 1109.07 examples/s]33607 examples [00:30, 1108.24 examples/s]33718 examples [00:30, 1094.08 examples/s]33829 examples [00:31, 1097.20 examples/s]33943 examples [00:31, 1107.50 examples/s]34058 examples [00:31, 1119.58 examples/s]34171 examples [00:31, 1112.97 examples/s]34283 examples [00:31, 1114.17 examples/s]34395 examples [00:31, 1105.25 examples/s]34506 examples [00:31, 1093.83 examples/s]34616 examples [00:31, 1078.54 examples/s]34724 examples [00:31, 1064.13 examples/s]34834 examples [00:32, 1072.58 examples/s]34947 examples [00:32, 1088.08 examples/s]35060 examples [00:32, 1099.93 examples/s]35171 examples [00:32, 1102.25 examples/s]35282 examples [00:32, 1094.59 examples/s]35392 examples [00:32, 1093.13 examples/s]35502 examples [00:32, 1089.23 examples/s]35611 examples [00:32, 1082.77 examples/s]35722 examples [00:32, 1088.60 examples/s]35833 examples [00:32, 1092.24 examples/s]35943 examples [00:33, 1085.62 examples/s]36056 examples [00:33, 1097.73 examples/s]36166 examples [00:33, 1068.34 examples/s]36274 examples [00:33, 1054.59 examples/s]36381 examples [00:33, 1057.25 examples/s]36487 examples [00:33, 1024.71 examples/s]36596 examples [00:33, 1042.61 examples/s]36708 examples [00:33, 1064.59 examples/s]36817 examples [00:33, 1071.78 examples/s]36926 examples [00:33, 1077.15 examples/s]37038 examples [00:34, 1089.36 examples/s]37150 examples [00:34, 1096.21 examples/s]37263 examples [00:34, 1105.85 examples/s]37374 examples [00:34, 1105.42 examples/s]37485 examples [00:34, 1106.25 examples/s]37597 examples [00:34, 1109.44 examples/s]37708 examples [00:34, 1084.66 examples/s]37821 examples [00:34, 1097.27 examples/s]37932 examples [00:34, 1098.41 examples/s]38042 examples [00:34, 1076.54 examples/s]38150 examples [00:35, 1053.50 examples/s]38258 examples [00:35, 1060.76 examples/s]38368 examples [00:35, 1069.32 examples/s]38480 examples [00:35, 1082.56 examples/s]38590 examples [00:35, 1085.45 examples/s]38702 examples [00:35, 1095.55 examples/s]38812 examples [00:35, 1090.68 examples/s]38922 examples [00:35, 1091.96 examples/s]39034 examples [00:35, 1099.82 examples/s]39145 examples [00:36, 1061.79 examples/s]39258 examples [00:36, 1080.20 examples/s]39373 examples [00:36, 1099.94 examples/s]39486 examples [00:36, 1107.39 examples/s]39599 examples [00:36, 1113.21 examples/s]39711 examples [00:36, 1112.00 examples/s]39823 examples [00:36, 1090.47 examples/s]39933 examples [00:36, 1070.03 examples/s]40041 examples [00:36, 1017.78 examples/s]40151 examples [00:36, 1039.00 examples/s]40256 examples [00:37, 1036.73 examples/s]40368 examples [00:37, 1060.14 examples/s]40478 examples [00:37, 1069.60 examples/s]40589 examples [00:37, 1080.99 examples/s]40700 examples [00:37, 1086.58 examples/s]40810 examples [00:37, 1088.93 examples/s]40920 examples [00:37, 1089.37 examples/s]41030 examples [00:37, 1060.30 examples/s]41137 examples [00:37, 1038.13 examples/s]41244 examples [00:37, 1047.23 examples/s]41352 examples [00:38, 1054.62 examples/s]41458 examples [00:38, 1054.22 examples/s]41567 examples [00:38, 1064.13 examples/s]41674 examples [00:38, 1046.14 examples/s]41783 examples [00:38, 1056.85 examples/s]41889 examples [00:38, 1039.18 examples/s]41997 examples [00:38, 1049.08 examples/s]42103 examples [00:38, 1047.76 examples/s]42216 examples [00:38, 1069.37 examples/s]42326 examples [00:38, 1076.54 examples/s]42436 examples [00:39, 1080.75 examples/s]42545 examples [00:39, 1078.49 examples/s]42656 examples [00:39, 1087.54 examples/s]42766 examples [00:39, 1089.90 examples/s]42876 examples [00:39, 1085.66 examples/s]42985 examples [00:39, 1083.14 examples/s]43094 examples [00:39, 1070.63 examples/s]43202 examples [00:39, 1060.02 examples/s]43309 examples [00:39, 986.62 examples/s] 43415 examples [00:40, 1004.83 examples/s]43526 examples [00:40, 1032.48 examples/s]43639 examples [00:40, 1058.78 examples/s]43746 examples [00:40, 1056.33 examples/s]43856 examples [00:40, 1067.21 examples/s]43964 examples [00:40, 1070.33 examples/s]44075 examples [00:40, 1081.72 examples/s]44188 examples [00:40, 1093.49 examples/s]44299 examples [00:40, 1097.91 examples/s]44410 examples [00:40, 1099.10 examples/s]44521 examples [00:41, 1089.37 examples/s]44635 examples [00:41, 1102.62 examples/s]44752 examples [00:41, 1120.85 examples/s]44865 examples [00:41, 1100.97 examples/s]44978 examples [00:41, 1107.84 examples/s]45089 examples [00:41, 1092.47 examples/s]45199 examples [00:41, 1068.31 examples/s]45308 examples [00:41, 1072.53 examples/s]45417 examples [00:41, 1076.98 examples/s]45529 examples [00:41, 1089.46 examples/s]45639 examples [00:42, 1092.59 examples/s]45749 examples [00:42, 1092.91 examples/s]45859 examples [00:42, 1088.27 examples/s]45971 examples [00:42, 1095.54 examples/s]46081 examples [00:42, 1063.34 examples/s]46190 examples [00:42, 1068.79 examples/s]46304 examples [00:42, 1087.78 examples/s]46419 examples [00:42, 1103.83 examples/s]46530 examples [00:42, 1097.47 examples/s]46640 examples [00:42, 1072.24 examples/s]46748 examples [00:43, 1041.39 examples/s]46861 examples [00:43, 1065.60 examples/s]46968 examples [00:43, 1059.55 examples/s]47075 examples [00:43, 1057.52 examples/s]47185 examples [00:43, 1069.39 examples/s]47293 examples [00:43, 1068.55 examples/s]47403 examples [00:43, 1077.27 examples/s]47514 examples [00:43, 1085.33 examples/s]47625 examples [00:43, 1092.40 examples/s]47736 examples [00:43, 1095.55 examples/s]47846 examples [00:44, 1072.88 examples/s]47962 examples [00:44, 1097.34 examples/s]48079 examples [00:44, 1115.74 examples/s]48194 examples [00:44, 1125.55 examples/s]48309 examples [00:44, 1129.68 examples/s]48423 examples [00:44, 1118.87 examples/s]48536 examples [00:44, 1108.60 examples/s]48647 examples [00:44, 1102.91 examples/s]48758 examples [00:44, 1081.99 examples/s]48867 examples [00:45, 1067.89 examples/s]48974 examples [00:45, 1058.77 examples/s]49084 examples [00:45, 1069.03 examples/s]49194 examples [00:45, 1076.87 examples/s]49303 examples [00:45, 1080.35 examples/s]49412 examples [00:45, 1081.53 examples/s]49521 examples [00:45, 1077.38 examples/s]49629 examples [00:45, 1075.93 examples/s]49739 examples [00:45, 1081.72 examples/s]49850 examples [00:45, 1089.74 examples/s]49960 examples [00:46, 1050.95 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▌        | 7701/50000 [00:00<00:00, 77009.16 examples/s] 42%|████▏     | 21213/50000 [00:00<00:00, 88416.41 examples/s] 69%|██████▉   | 34620/50000 [00:00<00:00, 98474.70 examples/s] 96%|█████████▌| 48101/50000 [00:00<00:00, 107136.42 examples/s]                                                                0 examples [00:00, ? examples/s]85 examples [00:00, 844.50 examples/s]193 examples [00:00, 902.49 examples/s]305 examples [00:00, 958.24 examples/s]431 examples [00:00, 1031.63 examples/s]554 examples [00:00, 1081.65 examples/s]675 examples [00:00, 1115.93 examples/s]801 examples [00:00, 1153.92 examples/s]927 examples [00:00, 1182.67 examples/s]1054 examples [00:00, 1207.16 examples/s]1180 examples [00:01, 1219.98 examples/s]1302 examples [00:01, 1217.52 examples/s]1423 examples [00:01, 968.76 examples/s] 1544 examples [00:01, 1028.44 examples/s]1660 examples [00:01, 1061.92 examples/s]1778 examples [00:01, 1092.67 examples/s]1898 examples [00:01, 1121.96 examples/s]2013 examples [00:01, 1121.16 examples/s]2130 examples [00:01, 1133.68 examples/s]2245 examples [00:01, 1125.07 examples/s]2359 examples [00:02, 1128.36 examples/s]2478 examples [00:02, 1144.18 examples/s]2598 examples [00:02, 1157.77 examples/s]2715 examples [00:02, 1160.06 examples/s]2833 examples [00:02, 1162.93 examples/s]2950 examples [00:02, 1122.53 examples/s]3072 examples [00:02, 1147.86 examples/s]3188 examples [00:02, 1133.81 examples/s]3312 examples [00:02, 1163.68 examples/s]3435 examples [00:03, 1181.72 examples/s]3554 examples [00:03, 1172.46 examples/s]3672 examples [00:03, 1164.32 examples/s]3792 examples [00:03, 1174.14 examples/s]3913 examples [00:03, 1184.51 examples/s]4037 examples [00:03, 1200.27 examples/s]4158 examples [00:03, 1183.90 examples/s]4277 examples [00:03, 1183.97 examples/s]4396 examples [00:03, 1182.80 examples/s]4519 examples [00:03, 1195.54 examples/s]4643 examples [00:04, 1206.02 examples/s]4764 examples [00:04, 1185.30 examples/s]4883 examples [00:04, 1185.19 examples/s]5002 examples [00:04, 1179.55 examples/s]5121 examples [00:04, 1154.48 examples/s]5238 examples [00:04, 1157.45 examples/s]5354 examples [00:04, 1137.06 examples/s]5473 examples [00:04, 1152.26 examples/s]5596 examples [00:04, 1173.69 examples/s]5720 examples [00:04, 1192.73 examples/s]5840 examples [00:05, 1182.12 examples/s]5959 examples [00:05, 1171.22 examples/s]6077 examples [00:05, 1161.38 examples/s]6194 examples [00:05, 1156.67 examples/s]6310 examples [00:05, 1147.84 examples/s]6435 examples [00:05, 1174.10 examples/s]6553 examples [00:05, 1142.11 examples/s]6672 examples [00:05, 1154.80 examples/s]6794 examples [00:05, 1170.72 examples/s]6913 examples [00:05, 1175.47 examples/s]7033 examples [00:06, 1180.61 examples/s]7152 examples [00:06, 1168.18 examples/s]7275 examples [00:06, 1184.68 examples/s]7398 examples [00:06, 1195.56 examples/s]7519 examples [00:06, 1198.18 examples/s]7639 examples [00:06, 1191.68 examples/s]7759 examples [00:06, 1183.13 examples/s]7879 examples [00:06, 1186.34 examples/s]7998 examples [00:06, 1187.12 examples/s]8117 examples [00:07, 1166.81 examples/s]8234 examples [00:07, 1156.17 examples/s]8352 examples [00:07, 1161.47 examples/s]8470 examples [00:07, 1165.55 examples/s]8591 examples [00:07, 1177.69 examples/s]8709 examples [00:07, 1162.34 examples/s]8826 examples [00:07, 1163.33 examples/s]8943 examples [00:07, 1163.22 examples/s]9067 examples [00:07, 1184.62 examples/s]9190 examples [00:07, 1197.35 examples/s]9314 examples [00:08, 1208.37 examples/s]9435 examples [00:08, 1191.56 examples/s]9555 examples [00:08, 1174.97 examples/s]9678 examples [00:08, 1189.68 examples/s]9803 examples [00:08, 1206.71 examples/s]9924 examples [00:08, 1190.33 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s] 95%|█████████▍| 9462/10000 [00:00<00:00, 94613.55 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteM84QMZ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteM84QMZ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7efcbc8b88c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7efcbc8b88c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7efcbc8b88c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efd28d56dd8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efd22b01ac8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7efcbc8b88c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7efcbc8b88c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7efcbc8b88c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efc9def74e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efc9def7390>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7efc94059620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7efc94059620> 

  function with postional parmater data_info <function split_train_valid at 0x7efc94059620> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7efc94059730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7efc94059730> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7efc94059730> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=7287010cac664a5124d7cac3eb9d2b9c495fceed8b5b573fcc3964cce284f974
  Stored in directory: /tmp/pip-ephem-wheel-cache-49skkv2f/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:23:08, 11.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:12:16, 15.8kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:01<10:42:02, 22.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 877k/862M [00:01<7:29:47, 31.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.57M/862M [00:01<5:14:29, 45.6kB/s].vector_cache/glove.6B.zip:   1%|          | 6.37M/862M [00:01<3:39:18, 65.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.2M/862M [00:01<2:32:44, 92.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.4M/862M [00:01<1:46:29, 133kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.0M/862M [00:01<1:14:14, 189kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.6M/862M [00:01<51:47, 270kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.8M/862M [00:01<36:06, 384kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.2M/862M [00:01<25:18, 546kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.6M/862M [00:02<17:42, 776kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.7M/862M [00:02<12:26, 1.10MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.2M/862M [00:02<08:45, 1.55MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.1M/862M [00:02<06:12, 2.18MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<04:52, 2.77MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<05:19, 2.53MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<07:22, 1.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<06:04, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.8M/862M [00:04<04:26, 3.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:06<08:40, 1.54MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<07:45, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:06<05:50, 2.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<06:44, 1.97MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<07:28, 1.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<05:51, 2.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:08<04:21, 3.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:08<03:25, 3.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<13:43, 964kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<11:11, 1.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:10<08:12, 1.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<08:29, 1.55MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<08:40, 1.52MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<06:44, 1.95MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:12<04:52, 2.69MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<1:36:46, 135kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<1:09:04, 189kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:14<48:36, 269kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<36:57, 352kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<28:32, 456kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<20:37, 631kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<16:28, 786kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<12:52, 1.01MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:18<09:15, 1.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<09:30, 1.36MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<09:17, 1.39MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<07:03, 1.83MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.0M/862M [00:20<05:05, 2.52MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<10:09, 1.26MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<08:27, 1.52MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<06:14, 2.05MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<07:20, 1.74MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<07:51, 1.62MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<06:02, 2.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<04:24, 2.88MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<08:26, 1.50MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:12, 1.76MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:19, 2.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:40, 1.89MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<07:15, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:43, 2.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:01, 2.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:30, 2.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:10, 3.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:49, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:46, 1.84MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:23, 2.31MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<03:54, 3.18MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<13:39, 909kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:50, 1.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<07:53, 1.57MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:24, 1.47MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:08, 1.73MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:18, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<06:36, 1.86MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:52, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:25, 2.76MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:58, 2.04MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:25, 2.25MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:05, 2.97MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:44, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<06:29, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:09, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<03:44, 3.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<1:29:47, 134kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<1:04:03, 188kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<45:03, 267kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<34:15, 350kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<26:05, 460kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<18:52, 635kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<13:20, 895kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<14:58, 797kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<11:41, 1.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:28, 1.40MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<08:41, 1.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<07:18, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<05:22, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:31, 1.81MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:59, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<05:23, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<03:55, 3.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:49, 1.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<07:22, 1.59MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<05:26, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:32, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:45, 2.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<04:16, 2.72MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:43, 2.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:21, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<04:57, 2.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:57<03:35, 3.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<10:26, 1.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<08:30, 1.35MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<06:14, 1.84MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<07:02, 1.63MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<07:16, 1.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:34, 2.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<04:05, 2.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:42, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:52, 1.94MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<04:21, 2.60MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<05:41, 1.99MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:07, 2.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<03:52, 2.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:22, 2.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:53, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<03:42, 3.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:13, 2.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:57, 1.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:40, 2.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<03:23, 3.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<12:08, 915kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<09:37, 1.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<07:00, 1.58MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:29, 1.47MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:21, 1.73MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<04:43, 2.33MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:52, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:15, 1.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:52, 2.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<03:32, 3.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<08:16, 1.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:55, 1.57MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:04, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<06:05, 1.78MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:21, 2.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:01, 2.69MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:21, 2.01MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:56, 1.81MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:38, 2.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:21<03:21, 3.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<22:02, 485kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<16:30, 647kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<11:46, 906kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<10:42, 992kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<08:34, 1.24MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<06:12, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:51, 1.54MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:52, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:20, 2.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<05:31, 1.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:00, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:39, 2.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:24, 3.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<06:43, 1.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<05:46, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:18, 2.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:25, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:55, 1.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:39, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:21, 3.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<9:56:37, 17.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<6:58:27, 24.6kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<4:52:26, 35.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<3:26:23, 49.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<2:25:25, 70.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<1:41:47, 100kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<1:13:25, 138kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<52:22, 194kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<36:49, 275kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<28:04, 359kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<20:40, 487kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<14:41, 684kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<12:36, 793kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<10:52, 921kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<08:06, 1.23MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<07:15, 1.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:05, 1.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<04:30, 2.20MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:27, 1.81MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:49, 2.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<03:37, 2.72MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:50, 2.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:23, 2.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:18, 2.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:37, 2.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:13, 2.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<03:11, 3.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:31, 2.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:08, 2.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:08, 3.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:27, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:04, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:02, 2.37MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<02:55, 3.26MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<1:11:06, 134kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<50:42, 188kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<35:38, 266kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<27:04, 349kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<20:59, 451kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<15:10, 622kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:58<10:41, 879kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<1:14:21, 126kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<52:59, 177kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<37:14, 251kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<28:08, 331kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<20:39, 451kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<14:39, 634kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<12:23, 747kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<09:36, 962kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<06:54, 1.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<07:01, 1.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:50, 1.57MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<04:17, 2.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:09, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:32, 2.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:24, 2.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:30, 2.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:04, 2.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:04, 2.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<04:16, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:54, 2.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<02:55, 3.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<04:09, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:50, 2.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<02:52, 3.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<04:05, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<03:45, 2.35MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<02:50, 3.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:04, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:35, 1.91MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:35, 2.44MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<02:37, 3.32MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<06:08, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<05:12, 1.67MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:51, 2.25MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:43, 1.83MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:03, 1.71MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:55, 2.19MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<02:49, 3.04MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<21:24, 400kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<15:51, 540kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<11:15, 759kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<09:50, 863kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<08:37, 985kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:27, 1.31MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<05:51, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<04:58, 1.70MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:41, 2.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:31, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:53, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:46, 2.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<02:46, 3.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:56, 1.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:19, 1.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<03:11, 2.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:09, 1.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:45, 2.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<02:49, 2.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:54, 2.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:33, 2.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<02:39, 3.06MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:46, 2.14MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:17, 1.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:21, 2.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<02:27, 3.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:15, 1.53MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:29, 1.78MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<03:18, 2.41MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:10, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:33, 1.75MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:31, 2.26MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:41<02:34, 3.07MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:10, 1.52MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:26, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<03:17, 2.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:07, 1.89MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:33, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:33, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<02:33, 3.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<13:29, 574kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<10:06, 765kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<07:12, 1.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<05:08, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<22:56, 335kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<17:37, 435kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<12:42, 603kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<10:04, 755kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<07:49, 972kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<05:39, 1.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:42, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:45, 1.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:28, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:12, 1.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:29, 1.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:27, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<02:32, 2.93MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:17, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:46, 1.96MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:47, 2.63MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:40, 2.00MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:19, 2.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:30, 2.91MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:28, 2.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<03:10, 2.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:24, 3.01MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:22, 2.13MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:06, 2.32MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:21, 3.05MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:19, 2.15MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:04, 2.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:19, 3.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:17, 2.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:01, 2.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:17, 3.07MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:14, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<02:59, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:15, 3.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:12, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<02:58, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:14, 3.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:11, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:38, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:53, 2.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<02:05, 3.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<6:33:55, 17.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<4:36:11, 24.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<3:12:46, 35.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<2:15:47, 49.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<1:36:23, 69.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<1:07:38, 99.1kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<47:19, 141kB/s]   .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<34:46, 191kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<25:00, 266kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<17:36, 376kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<13:48, 477kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<10:19, 636kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<07:22, 889kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<06:40, 976kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<05:19, 1.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:51, 1.68MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:13, 1.53MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:15, 1.51MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:18, 1.94MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:22, 2.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<07:28, 853kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<05:52, 1.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<04:14, 1.49MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:26, 1.42MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:43, 1.69MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<02:45, 2.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:23, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:00, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:15, 2.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:01, 2.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:22, 1.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:40, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:50, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:36, 2.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<01:57, 3.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:46, 2.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:10, 1.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:29, 2.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<01:50, 3.25MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<03:23, 1.76MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:59, 1.99MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:13, 2.66MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:55, 2.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:38, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<01:59, 2.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:45, 2.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:31, 2.30MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<01:54, 3.03MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:41, 2.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:03, 1.88MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:23, 2.40MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<01:43, 3.30MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<05:24, 1.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:22, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<03:11, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:32, 1.59MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:02, 1.84MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:15, 2.47MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:53, 1.92MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:09, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:29, 2.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<01:47, 3.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<40:57, 134kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<29:12, 188kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<20:29, 266kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<15:31, 349kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<11:57, 452kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<08:37, 626kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<06:02, 884kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<5:11:29, 17.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<3:38:19, 24.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<2:32:13, 34.9kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<1:47:04, 49.3kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<1:15:23, 69.9kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<52:38, 99.6kB/s]  .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<37:50, 138kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<26:54, 193kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<18:50, 275kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<13:10, 390kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<28:38, 179kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<20:27, 251kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<14:22, 355kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<11:11, 453kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<08:51, 572kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<06:24, 789kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<04:30, 1.11MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<06:33, 764kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<05:06, 979kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:39, 1.36MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:42, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:36, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:43, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:57, 2.50MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<04:35, 1.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<03:42, 1.31MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:42, 1.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:00, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:35, 1.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:55, 2.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:27, 1.93MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:40, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:06, 2.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<01:31, 3.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<19:00, 245kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<13:46, 338kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<09:41, 478kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<07:48, 588kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<06:23, 717kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<04:41, 974kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<03:59, 1.13MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<03:15, 1.39MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<02:21, 1.90MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:41, 1.66MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:47, 1.59MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:10, 2.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:13, 1.98MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<02:00, 2.18MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:30, 2.88MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:03, 2.10MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:52, 2.30MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:24, 3.03MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:59, 2.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:13, 1.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:44, 2.43MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:16, 3.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:24, 1.74MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:07, 1.96MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:35, 2.62MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:03, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:52, 2.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:24, 2.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:55, 2.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:10, 1.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:42, 2.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<01:13, 3.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<04:34, 868kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:36, 1.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<02:36, 1.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:43, 1.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:18, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:41, 2.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:04, 1.85MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:13, 1.72MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:44, 2.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<01:15, 3.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<3:39:31, 17.2kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<2:33:47, 24.5kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<1:47:00, 34.9kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<1:15:02, 49.3kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<52:48, 69.9kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<36:48, 99.7kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<26:22, 138kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<18:47, 193kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<13:09, 274kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<09:57, 358kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<07:41, 463kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<05:32, 640kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<04:23, 795kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:26, 1.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:27, 1.41MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:30, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:06, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:32, 2.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:52, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:38, 2.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:13, 2.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:37, 2.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:52, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:28, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:31, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:24, 2.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:03, 3.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:28, 2.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:21, 2.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:00, 3.07MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:26, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:36, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:15, 2.45MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<00:54, 3.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:18, 1.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:55, 1.56MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:25, 2.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:40, 1.76MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:46, 1.66MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:21, 2.15MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<00:59, 2.94MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:57, 1.46MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:39, 1.72MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:13, 2.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:30, 1.86MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:37, 1.72MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:15, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<00:54, 3.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:43, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:29, 1.84MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<01:05, 2.48MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:22, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:30, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:10, 2.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<00:50, 3.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:10, 1.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:47, 1.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:18, 1.97MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:29, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:18, 1.94MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<00:57, 2.59MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:15, 1.97MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:22, 1.79MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:04, 2.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:07, 2.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:02, 2.31MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<00:46, 3.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:04, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:13, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:57, 2.40MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<00:40, 3.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<10:43, 211kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<07:43, 292kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:24<05:24, 412kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<04:14, 518kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:24, 642kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<02:28, 881kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<01:44, 1.23MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:02, 1.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:38, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<01:10, 1.77MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:17, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:06, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:49, 2.46MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:04, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:09, 1.71MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:54, 2.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:38, 3.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<1:51:18, 17.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<1:17:51, 24.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<53:48, 35.0kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<37:22, 49.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<26:15, 70.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<18:10, 100kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<12:53, 138kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<09:21, 190kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<06:35, 268kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<04:31, 381kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<04:24, 388kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<03:15, 524kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<02:17, 734kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:57, 843kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:31, 1.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:05, 1.47MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:07, 1.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:56, 1.66MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:41, 2.24MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:49, 1.82MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:43, 2.06MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:31, 2.77MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:42, 2.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:47, 1.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:37, 2.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:25, 3.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<1:19:28, 17.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<55:31, 24.5kB/s]  .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<38:09, 35.0kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<26:17, 49.5kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<18:26, 70.1kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<12:41, 99.9kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<08:55, 138kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<06:28, 190kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<04:32, 267kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<03:03, 380kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<10:27, 111kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<07:23, 156kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<05:05, 222kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<03:42, 295kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<02:48, 389kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:59, 540kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<01:29, 685kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:08, 890kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:48, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:46, 1.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:42, 1.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:32, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:30, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:26, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:19, 2.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:24, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:21, 2.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:15, 3.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:21, 2.14MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:19, 2.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:14, 3.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:19, 2.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:21, 1.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:16, 2.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:16, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:15, 2.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:11, 3.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:15, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:13, 2.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:09, 3.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:14, 1.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:11, 2.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:15<00:07, 3.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<23:40, 17.3kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<16:21, 24.6kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<10:43, 35.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<06:51, 49.6kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<04:49, 69.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<03:15, 99.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<02:01, 142kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<01:29, 183kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<01:02, 254kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:39, 360kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:26, 459kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:20, 578kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:14, 791kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:07, 1.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<01:02, 129kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:42, 181kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:23, 256kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:11, 337kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:08, 438kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:04, 606kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<12:12:43,  9.10it/s]  0%|          | 887/400000 [00:00<8:31:59, 12.99it/s]  0%|          | 1847/400000 [00:00<5:57:44, 18.55it/s]  1%|          | 2707/400000 [00:00<4:10:06, 26.47it/s]  1%|          | 3588/400000 [00:00<2:54:54, 37.77it/s]  1%|          | 4468/400000 [00:00<2:02:23, 53.86it/s]  1%|▏         | 5344/400000 [00:00<1:25:42, 76.74it/s]  2%|▏         | 6217/400000 [00:00<1:00:05, 109.22it/s]  2%|▏         | 7075/400000 [00:00<42:12, 155.18it/s]    2%|▏         | 7950/400000 [00:01<29:41, 220.02it/s]  2%|▏         | 8816/400000 [00:01<20:58, 310.92it/s]  2%|▏         | 9679/400000 [00:01<14:52, 437.41it/s]  3%|▎         | 10541/400000 [00:01<10:36, 611.56it/s]  3%|▎         | 11424/400000 [00:01<07:37, 848.45it/s]  3%|▎         | 12425/400000 [00:01<05:31, 1169.57it/s]  3%|▎         | 13329/400000 [00:01<04:04, 1579.47it/s]  4%|▎         | 14223/400000 [00:01<03:03, 2097.55it/s]  4%|▍         | 15117/400000 [00:01<02:21, 2716.81it/s]  4%|▍         | 16005/400000 [00:01<01:52, 3411.61it/s]  4%|▍         | 16913/400000 [00:02<01:31, 4197.73it/s]  4%|▍         | 17798/400000 [00:02<01:17, 4947.76it/s]  5%|▍         | 18676/400000 [00:02<01:06, 5692.19it/s]  5%|▍         | 19551/400000 [00:02<01:00, 6295.86it/s]  5%|▌         | 20415/400000 [00:02<00:55, 6804.89it/s]  5%|▌         | 21271/400000 [00:02<00:52, 7205.44it/s]  6%|▌         | 22155/400000 [00:02<00:49, 7627.21it/s]  6%|▌         | 23052/400000 [00:02<00:47, 7985.10it/s]  6%|▌         | 23945/400000 [00:02<00:45, 8244.83it/s]  6%|▌         | 24867/400000 [00:02<00:44, 8512.67it/s]  6%|▋         | 25758/400000 [00:03<00:43, 8615.39it/s]  7%|▋         | 26648/400000 [00:03<00:43, 8599.02it/s]  7%|▋         | 27542/400000 [00:03<00:42, 8696.14it/s]  7%|▋         | 28432/400000 [00:03<00:42, 8754.65it/s]  7%|▋         | 29318/400000 [00:03<00:42, 8719.08it/s]  8%|▊         | 30237/400000 [00:03<00:41, 8854.67it/s]  8%|▊         | 31128/400000 [00:03<00:41, 8794.61it/s]  8%|▊         | 32028/400000 [00:03<00:41, 8852.77it/s]  8%|▊         | 32917/400000 [00:03<00:41, 8755.74it/s]  8%|▊         | 33797/400000 [00:03<00:41, 8766.95it/s]  9%|▊         | 34676/400000 [00:04<00:41, 8729.77it/s]  9%|▉         | 35551/400000 [00:04<00:41, 8735.47it/s]  9%|▉         | 36426/400000 [00:04<00:41, 8739.08it/s]  9%|▉         | 37334/400000 [00:04<00:41, 8838.00it/s] 10%|▉         | 38224/400000 [00:04<00:40, 8837.08it/s] 10%|▉         | 39109/400000 [00:04<00:40, 8832.79it/s] 10%|▉         | 39993/400000 [00:04<00:40, 8821.70it/s] 10%|█         | 40920/400000 [00:04<00:40, 8949.34it/s] 10%|█         | 41816/400000 [00:04<00:40, 8892.77it/s] 11%|█         | 42706/400000 [00:04<00:40, 8848.89it/s] 11%|█         | 43614/400000 [00:05<00:39, 8916.00it/s] 11%|█         | 44512/400000 [00:05<00:39, 8934.81it/s] 11%|█▏        | 45412/400000 [00:05<00:39, 8953.06it/s] 12%|█▏        | 46308/400000 [00:05<00:39, 8901.38it/s] 12%|█▏        | 47221/400000 [00:05<00:39, 8966.65it/s] 12%|█▏        | 48118/400000 [00:05<00:39, 8817.00it/s] 12%|█▏        | 49001/400000 [00:05<00:40, 8769.89it/s] 12%|█▏        | 49904/400000 [00:05<00:39, 8845.28it/s] 13%|█▎        | 50805/400000 [00:05<00:39, 8891.88it/s] 13%|█▎        | 51719/400000 [00:05<00:38, 8963.33it/s] 13%|█▎        | 52627/400000 [00:06<00:38, 8995.59it/s] 13%|█▎        | 53527/400000 [00:06<00:38, 8927.81it/s] 14%|█▎        | 54445/400000 [00:06<00:38, 8999.48it/s] 14%|█▍        | 55346/400000 [00:06<00:38, 8980.49it/s] 14%|█▍        | 56250/400000 [00:06<00:38, 8996.25it/s] 14%|█▍        | 57172/400000 [00:06<00:37, 9060.98it/s] 15%|█▍        | 58079/400000 [00:06<00:37, 9034.12it/s] 15%|█▍        | 58984/400000 [00:06<00:37, 9037.04it/s] 15%|█▍        | 59888/400000 [00:06<00:37, 9031.95it/s] 15%|█▌        | 60795/400000 [00:06<00:37, 9042.25it/s] 15%|█▌        | 61700/400000 [00:07<00:38, 8869.74it/s] 16%|█▌        | 62588/400000 [00:07<00:38, 8848.97it/s] 16%|█▌        | 63474/400000 [00:07<00:38, 8823.09it/s] 16%|█▌        | 64357/400000 [00:07<00:38, 8662.99it/s] 16%|█▋        | 65225/400000 [00:07<00:38, 8637.77it/s] 17%|█▋        | 66109/400000 [00:07<00:38, 8694.46it/s] 17%|█▋        | 66980/400000 [00:07<00:38, 8596.40it/s] 17%|█▋        | 67841/400000 [00:07<00:38, 8530.92it/s] 17%|█▋        | 68724/400000 [00:07<00:38, 8618.49it/s] 17%|█▋        | 69610/400000 [00:07<00:38, 8687.99it/s] 18%|█▊        | 70505/400000 [00:08<00:37, 8763.85it/s] 18%|█▊        | 71394/400000 [00:08<00:37, 8799.09it/s] 18%|█▊        | 72289/400000 [00:08<00:37, 8841.29it/s] 18%|█▊        | 73188/400000 [00:08<00:36, 8884.77it/s] 19%|█▊        | 74088/400000 [00:08<00:36, 8918.90it/s] 19%|█▉        | 75033/400000 [00:08<00:35, 9071.85it/s] 19%|█▉        | 75941/400000 [00:08<00:36, 8894.86it/s] 19%|█▉        | 76859/400000 [00:08<00:35, 8976.48it/s] 19%|█▉        | 77800/400000 [00:08<00:35, 9101.00it/s] 20%|█▉        | 78712/400000 [00:09<00:35, 9106.11it/s] 20%|█▉        | 79652/400000 [00:09<00:34, 9191.15it/s] 20%|██        | 80572/400000 [00:09<00:35, 8910.67it/s] 20%|██        | 81466/400000 [00:09<00:36, 8837.83it/s] 21%|██        | 82372/400000 [00:09<00:35, 8902.40it/s] 21%|██        | 83315/400000 [00:09<00:34, 9052.09it/s] 21%|██        | 84250/400000 [00:09<00:34, 9137.50it/s] 21%|██▏       | 85167/400000 [00:09<00:34, 9146.09it/s] 22%|██▏       | 86083/400000 [00:09<00:34, 9100.86it/s] 22%|██▏       | 87009/400000 [00:09<00:34, 9145.94it/s] 22%|██▏       | 87925/400000 [00:10<00:34, 9149.27it/s] 22%|██▏       | 88841/400000 [00:10<00:34, 9149.02it/s] 22%|██▏       | 89757/400000 [00:10<00:34, 9001.11it/s] 23%|██▎       | 90684/400000 [00:10<00:34, 9077.64it/s] 23%|██▎       | 91610/400000 [00:10<00:33, 9130.02it/s] 23%|██▎       | 92524/400000 [00:10<00:33, 9121.33it/s] 23%|██▎       | 93445/400000 [00:10<00:33, 9145.52it/s] 24%|██▎       | 94360/400000 [00:10<00:33, 9093.19it/s] 24%|██▍       | 95292/400000 [00:10<00:33, 9158.11it/s] 24%|██▍       | 96209/400000 [00:10<00:33, 9116.72it/s] 24%|██▍       | 97121/400000 [00:11<00:33, 9098.58it/s] 25%|██▍       | 98032/400000 [00:11<00:33, 9067.16it/s] 25%|██▍       | 98939/400000 [00:11<00:33, 9038.49it/s] 25%|██▍       | 99843/400000 [00:11<00:33, 9026.31it/s] 25%|██▌       | 100781/400000 [00:11<00:32, 9128.63it/s] 25%|██▌       | 101711/400000 [00:11<00:32, 9168.54it/s] 26%|██▌       | 102629/400000 [00:11<00:32, 9154.82it/s] 26%|██▌       | 103545/400000 [00:11<00:32, 9142.68it/s] 26%|██▌       | 104484/400000 [00:11<00:32, 9214.48it/s] 26%|██▋       | 105406/400000 [00:11<00:32, 9183.96it/s] 27%|██▋       | 106325/400000 [00:12<00:32, 9148.10it/s] 27%|██▋       | 107240/400000 [00:12<00:32, 9124.12it/s] 27%|██▋       | 108153/400000 [00:12<00:32, 9054.74it/s] 27%|██▋       | 109091/400000 [00:12<00:31, 9148.81it/s] 28%|██▊       | 110037/400000 [00:12<00:31, 9238.34it/s] 28%|██▊       | 110962/400000 [00:12<00:31, 9033.34it/s] 28%|██▊       | 111867/400000 [00:12<00:32, 8972.47it/s] 28%|██▊       | 112766/400000 [00:12<00:32, 8934.46it/s] 28%|██▊       | 113705/400000 [00:12<00:31, 9063.78it/s] 29%|██▊       | 114660/400000 [00:12<00:31, 9202.67it/s] 29%|██▉       | 115582/400000 [00:13<00:30, 9186.29it/s] 29%|██▉       | 116537/400000 [00:13<00:30, 9288.26it/s] 29%|██▉       | 117467/400000 [00:13<00:30, 9168.03it/s] 30%|██▉       | 118408/400000 [00:13<00:30, 9239.16it/s] 30%|██▉       | 119336/400000 [00:13<00:30, 9250.21it/s] 30%|███       | 120262/400000 [00:13<00:30, 9167.26it/s] 30%|███       | 121206/400000 [00:13<00:30, 9244.95it/s] 31%|███       | 122132/400000 [00:13<00:30, 9230.50it/s] 31%|███       | 123075/400000 [00:13<00:29, 9288.74it/s] 31%|███       | 124007/400000 [00:13<00:29, 9295.72it/s] 31%|███       | 124964/400000 [00:14<00:29, 9374.20it/s] 31%|███▏      | 125907/400000 [00:14<00:29, 9388.32it/s] 32%|███▏      | 126847/400000 [00:14<00:29, 9148.38it/s] 32%|███▏      | 127764/400000 [00:14<00:30, 9072.23it/s] 32%|███▏      | 128679/400000 [00:14<00:29, 9093.01it/s] 32%|███▏      | 129590/400000 [00:14<00:29, 9046.00it/s] 33%|███▎      | 130521/400000 [00:14<00:29, 9123.51it/s] 33%|███▎      | 131442/400000 [00:14<00:29, 9149.03it/s] 33%|███▎      | 132372/400000 [00:14<00:29, 9192.50it/s] 33%|███▎      | 133305/400000 [00:14<00:28, 9231.79it/s] 34%|███▎      | 134263/400000 [00:15<00:28, 9331.34it/s] 34%|███▍      | 135199/400000 [00:15<00:28, 9338.74it/s] 34%|███▍      | 136134/400000 [00:15<00:28, 9288.17it/s] 34%|███▍      | 137075/400000 [00:15<00:28, 9321.69it/s] 35%|███▍      | 138015/400000 [00:15<00:28, 9344.59it/s] 35%|███▍      | 138950/400000 [00:15<00:28, 9291.98it/s] 35%|███▍      | 139884/400000 [00:15<00:27, 9305.46it/s] 35%|███▌      | 140828/400000 [00:15<00:27, 9344.78it/s] 35%|███▌      | 141763/400000 [00:15<00:27, 9274.50it/s] 36%|███▌      | 142691/400000 [00:15<00:28, 9183.25it/s] 36%|███▌      | 143610/400000 [00:16<00:28, 9094.56it/s] 36%|███▌      | 144526/400000 [00:16<00:28, 9112.34it/s] 36%|███▋      | 145438/400000 [00:16<00:28, 9002.87it/s] 37%|███▋      | 146339/400000 [00:16<00:28, 8990.28it/s] 37%|███▋      | 147275/400000 [00:16<00:27, 9097.38it/s] 37%|███▋      | 148219/400000 [00:16<00:27, 9195.97it/s] 37%|███▋      | 149166/400000 [00:16<00:27, 9276.08it/s] 38%|███▊      | 150095/400000 [00:16<00:26, 9256.89it/s] 38%|███▊      | 151039/400000 [00:16<00:26, 9309.15it/s] 38%|███▊      | 151985/400000 [00:16<00:26, 9353.38it/s] 38%|███▊      | 152921/400000 [00:17<00:26, 9316.22it/s] 38%|███▊      | 153866/400000 [00:17<00:26, 9355.93it/s] 39%|███▊      | 154802/400000 [00:17<00:26, 9216.36it/s] 39%|███▉      | 155738/400000 [00:17<00:26, 9258.58it/s] 39%|███▉      | 156665/400000 [00:17<00:26, 9247.68it/s] 39%|███▉      | 157591/400000 [00:17<00:26, 9041.92it/s] 40%|███▉      | 158497/400000 [00:17<00:27, 8889.62it/s] 40%|███▉      | 159388/400000 [00:17<00:27, 8812.70it/s] 40%|████      | 160289/400000 [00:17<00:27, 8870.88it/s] 40%|████      | 161198/400000 [00:18<00:26, 8934.82it/s] 41%|████      | 162131/400000 [00:18<00:26, 9049.01it/s] 41%|████      | 163046/400000 [00:18<00:26, 9078.02it/s] 41%|████      | 163955/400000 [00:18<00:26, 9014.08it/s] 41%|████      | 164857/400000 [00:18<00:26, 8998.32it/s] 41%|████▏     | 165763/400000 [00:18<00:25, 9015.59it/s] 42%|████▏     | 166682/400000 [00:18<00:25, 9064.96it/s] 42%|████▏     | 167593/400000 [00:18<00:25, 9076.55it/s] 42%|████▏     | 168517/400000 [00:18<00:25, 9122.75it/s] 42%|████▏     | 169452/400000 [00:18<00:25, 9187.25it/s] 43%|████▎     | 170404/400000 [00:19<00:24, 9283.02it/s] 43%|████▎     | 171335/400000 [00:19<00:24, 9288.50it/s] 43%|████▎     | 172265/400000 [00:19<00:24, 9259.56it/s] 43%|████▎     | 173192/400000 [00:19<00:25, 8957.00it/s] 44%|████▎     | 174090/400000 [00:19<00:25, 8908.88it/s] 44%|████▎     | 174983/400000 [00:19<00:25, 8908.74it/s] 44%|████▍     | 175876/400000 [00:19<00:25, 8912.08it/s] 44%|████▍     | 176780/400000 [00:19<00:24, 8949.87it/s] 44%|████▍     | 177688/400000 [00:19<00:24, 8985.68it/s] 45%|████▍     | 178628/400000 [00:19<00:24, 9105.60it/s] 45%|████▍     | 179577/400000 [00:20<00:23, 9214.57it/s] 45%|████▌     | 180510/400000 [00:20<00:23, 9248.29it/s] 45%|████▌     | 181436/400000 [00:20<00:23, 9237.33it/s] 46%|████▌     | 182361/400000 [00:20<00:23, 9230.83it/s] 46%|████▌     | 183317/400000 [00:20<00:23, 9326.57it/s] 46%|████▌     | 184257/400000 [00:20<00:23, 9346.26it/s] 46%|████▋     | 185271/400000 [00:20<00:22, 9569.39it/s] 47%|████▋     | 186274/400000 [00:20<00:22, 9701.49it/s] 47%|████▋     | 187312/400000 [00:20<00:21, 9892.87it/s] 47%|████▋     | 188343/400000 [00:20<00:21, 10013.09it/s] 47%|████▋     | 189347/400000 [00:21<00:21, 10011.58it/s] 48%|████▊     | 190350/400000 [00:21<00:20, 10001.91it/s] 48%|████▊     | 191376/400000 [00:21<00:20, 10076.87it/s] 48%|████▊     | 192385/400000 [00:21<00:21, 9789.44it/s]  48%|████▊     | 193367/400000 [00:21<00:21, 9729.25it/s] 49%|████▊     | 194342/400000 [00:21<00:21, 9597.74it/s] 49%|████▉     | 195304/400000 [00:21<00:21, 9479.30it/s] 49%|████▉     | 196292/400000 [00:21<00:21, 9595.93it/s] 49%|████▉     | 197253/400000 [00:21<00:21, 9350.00it/s] 50%|████▉     | 198220/400000 [00:21<00:21, 9443.24it/s] 50%|████▉     | 199167/400000 [00:22<00:21, 9422.55it/s] 50%|█████     | 200111/400000 [00:22<00:21, 9327.32it/s] 50%|█████     | 201045/400000 [00:22<00:21, 9318.12it/s] 50%|█████     | 201978/400000 [00:22<00:21, 9104.52it/s] 51%|█████     | 202911/400000 [00:22<00:21, 9169.66it/s] 51%|█████     | 203830/400000 [00:22<00:21, 9150.60it/s] 51%|█████     | 204785/400000 [00:22<00:21, 9266.60it/s] 51%|█████▏    | 205744/400000 [00:22<00:20, 9361.20it/s] 52%|█████▏    | 206721/400000 [00:22<00:20, 9478.89it/s] 52%|█████▏    | 207757/400000 [00:22<00:19, 9726.78it/s] 52%|█████▏    | 208732/400000 [00:23<00:19, 9723.68it/s] 52%|█████▏    | 209748/400000 [00:23<00:19, 9850.53it/s] 53%|█████▎    | 210766/400000 [00:23<00:19, 9945.96it/s] 53%|█████▎    | 211774/400000 [00:23<00:18, 9984.78it/s] 53%|█████▎    | 212818/400000 [00:23<00:18, 10115.42it/s] 53%|█████▎    | 213846/400000 [00:23<00:18, 10161.92it/s] 54%|█████▎    | 214868/400000 [00:23<00:18, 10178.74it/s] 54%|█████▍    | 215887/400000 [00:23<00:18, 10124.84it/s] 54%|█████▍    | 216900/400000 [00:23<00:18, 10020.88it/s] 54%|█████▍    | 217903/400000 [00:23<00:18, 9924.51it/s]  55%|█████▍    | 218920/400000 [00:24<00:18, 9996.38it/s] 55%|█████▍    | 219936/400000 [00:24<00:17, 10043.98it/s] 55%|█████▌    | 220941/400000 [00:24<00:18, 9608.50it/s]  55%|█████▌    | 221907/400000 [00:24<00:19, 9336.15it/s] 56%|█████▌    | 222846/400000 [00:24<00:19, 9297.52it/s] 56%|█████▌    | 223807/400000 [00:24<00:18, 9386.36it/s] 56%|█████▌    | 224753/400000 [00:24<00:18, 9405.48it/s] 56%|█████▋    | 225741/400000 [00:24<00:18, 9540.83it/s] 57%|█████▋    | 226697/400000 [00:24<00:18, 9374.66it/s] 57%|█████▋    | 227693/400000 [00:25<00:18, 9540.70it/s] 57%|█████▋    | 228716/400000 [00:25<00:17, 9736.29it/s] 57%|█████▋    | 229709/400000 [00:25<00:17, 9793.52it/s] 58%|█████▊    | 230708/400000 [00:25<00:17, 9849.68it/s] 58%|█████▊    | 231695/400000 [00:25<00:17, 9732.66it/s] 58%|█████▊    | 232670/400000 [00:25<00:17, 9627.34it/s] 58%|█████▊    | 233634/400000 [00:25<00:17, 9496.10it/s] 59%|█████▊    | 234623/400000 [00:25<00:17, 9607.46it/s] 59%|█████▉    | 235601/400000 [00:25<00:17, 9656.41it/s] 59%|█████▉    | 236595/400000 [00:25<00:16, 9738.04it/s] 59%|█████▉    | 237570/400000 [00:26<00:16, 9689.39it/s] 60%|█████▉    | 238540/400000 [00:26<00:16, 9669.07it/s] 60%|█████▉    | 239508/400000 [00:26<00:16, 9637.79it/s] 60%|██████    | 240478/400000 [00:26<00:16, 9655.80it/s] 60%|██████    | 241468/400000 [00:26<00:16, 9727.57it/s] 61%|██████    | 242509/400000 [00:26<00:15, 9920.96it/s] 61%|██████    | 243505/400000 [00:26<00:15, 9930.51it/s] 61%|██████    | 244538/400000 [00:26<00:15, 10045.03it/s] 61%|██████▏   | 245544/400000 [00:26<00:15, 9910.57it/s]  62%|██████▏   | 246537/400000 [00:26<00:15, 9746.05it/s] 62%|██████▏   | 247567/400000 [00:27<00:15, 9905.49it/s] 62%|██████▏   | 248574/400000 [00:27<00:15, 9954.01it/s] 62%|██████▏   | 249613/400000 [00:27<00:14, 10079.66it/s] 63%|██████▎   | 250627/400000 [00:27<00:14, 10096.73it/s] 63%|██████▎   | 251638/400000 [00:27<00:14, 10014.56it/s] 63%|██████▎   | 252641/400000 [00:27<00:14, 9947.44it/s]  63%|██████▎   | 253637/400000 [00:27<00:14, 9783.78it/s] 64%|██████▎   | 254624/400000 [00:27<00:14, 9808.27it/s] 64%|██████▍   | 255606/400000 [00:27<00:14, 9647.75it/s] 64%|██████▍   | 256572/400000 [00:27<00:15, 9544.59it/s] 64%|██████▍   | 257605/400000 [00:28<00:14, 9766.42it/s] 65%|██████▍   | 258648/400000 [00:28<00:14, 9953.81it/s] 65%|██████▍   | 259677/400000 [00:28<00:13, 10051.06it/s] 65%|██████▌   | 260693/400000 [00:28<00:13, 10081.34it/s] 65%|██████▌   | 261703/400000 [00:28<00:13, 9973.23it/s]  66%|██████▌   | 262702/400000 [00:28<00:13, 9895.57it/s] 66%|██████▌   | 263693/400000 [00:28<00:13, 9791.19it/s] 66%|██████▌   | 264674/400000 [00:28<00:13, 9696.44it/s] 66%|██████▋   | 265669/400000 [00:28<00:13, 9768.94it/s] 67%|██████▋   | 266670/400000 [00:28<00:13, 9839.05it/s] 67%|██████▋   | 267656/400000 [00:29<00:13, 9844.65it/s] 67%|██████▋   | 268664/400000 [00:29<00:13, 9913.47it/s] 67%|██████▋   | 269656/400000 [00:29<00:13, 9804.38it/s] 68%|██████▊   | 270638/400000 [00:29<00:13, 9563.42it/s] 68%|██████▊   | 271597/400000 [00:29<00:13, 9383.13it/s] 68%|██████▊   | 272577/400000 [00:29<00:13, 9503.43it/s] 68%|██████▊   | 273552/400000 [00:29<00:13, 9574.20it/s] 69%|██████▊   | 274594/400000 [00:29<00:12, 9810.90it/s] 69%|██████▉   | 275587/400000 [00:29<00:12, 9843.52it/s] 69%|██████▉   | 276574/400000 [00:30<00:12, 9598.88it/s] 69%|██████▉   | 277537/400000 [00:30<00:12, 9512.32it/s] 70%|██████▉   | 278491/400000 [00:30<00:12, 9397.33it/s] 70%|██████▉   | 279433/400000 [00:30<00:12, 9310.24it/s] 70%|███████   | 280366/400000 [00:30<00:13, 9201.77it/s] 70%|███████   | 281288/400000 [00:30<00:12, 9177.88it/s] 71%|███████   | 282286/400000 [00:30<00:12, 9402.42it/s] 71%|███████   | 283348/400000 [00:30<00:11, 9737.18it/s] 71%|███████   | 284382/400000 [00:30<00:11, 9909.04it/s] 71%|███████▏  | 285377/400000 [00:30<00:11, 9832.49it/s] 72%|███████▏  | 286364/400000 [00:31<00:11, 9630.65it/s] 72%|███████▏  | 287330/400000 [00:31<00:11, 9457.84it/s] 72%|███████▏  | 288279/400000 [00:31<00:11, 9345.00it/s] 72%|███████▏  | 289230/400000 [00:31<00:11, 9391.39it/s] 73%|███████▎  | 290171/400000 [00:31<00:11, 9363.97it/s] 73%|███████▎  | 291141/400000 [00:31<00:11, 9459.76it/s] 73%|███████▎  | 292132/400000 [00:31<00:11, 9589.99it/s] 73%|███████▎  | 293181/400000 [00:31<00:10, 9840.84it/s] 74%|███████▎  | 294230/400000 [00:31<00:10, 10025.14it/s] 74%|███████▍  | 295236/400000 [00:31<00:10, 9950.01it/s]  74%|███████▍  | 296238/400000 [00:32<00:10, 9970.23it/s] 74%|███████▍  | 297237/400000 [00:32<00:10, 9895.09it/s] 75%|███████▍  | 298265/400000 [00:32<00:10, 10006.77it/s] 75%|███████▍  | 299267/400000 [00:32<00:10, 9990.46it/s]  75%|███████▌  | 300267/400000 [00:32<00:09, 9990.77it/s] 75%|███████▌  | 301299/400000 [00:32<00:09, 10086.16it/s] 76%|███████▌  | 302359/400000 [00:32<00:09, 10233.29it/s] 76%|███████▌  | 303384/400000 [00:32<00:09, 10229.30it/s] 76%|███████▌  | 304408/400000 [00:32<00:09, 10105.72it/s] 76%|███████▋  | 305420/400000 [00:32<00:09, 9836.89it/s]  77%|███████▋  | 306406/400000 [00:33<00:09, 9803.92it/s] 77%|███████▋  | 307453/400000 [00:33<00:09, 9992.16it/s] 77%|███████▋  | 308457/400000 [00:33<00:09, 10006.42it/s] 77%|███████▋  | 309466/400000 [00:33<00:09, 10030.52it/s] 78%|███████▊  | 310471/400000 [00:33<00:09, 9894.40it/s]  78%|███████▊  | 311462/400000 [00:33<00:09, 9698.07it/s] 78%|███████▊  | 312434/400000 [00:33<00:09, 9692.48it/s] 78%|███████▊  | 313448/400000 [00:33<00:08, 9820.13it/s] 79%|███████▊  | 314458/400000 [00:33<00:08, 9900.77it/s] 79%|███████▉  | 315450/400000 [00:33<00:08, 9892.69it/s] 79%|███████▉  | 316440/400000 [00:34<00:08, 9887.89it/s] 79%|███████▉  | 317450/400000 [00:34<00:08, 9950.41it/s] 80%|███████▉  | 318460/400000 [00:34<00:08, 9994.82it/s] 80%|███████▉  | 319460/400000 [00:34<00:08, 9892.51it/s] 80%|████████  | 320450/400000 [00:34<00:08, 9602.70it/s] 80%|████████  | 321413/400000 [00:34<00:08, 9390.10it/s] 81%|████████  | 322355/400000 [00:34<00:08, 9266.23it/s] 81%|████████  | 323300/400000 [00:34<00:08, 9317.95it/s] 81%|████████  | 324236/400000 [00:34<00:08, 9329.73it/s] 81%|████████▏ | 325171/400000 [00:35<00:08, 9268.37it/s] 82%|████████▏ | 326197/400000 [00:35<00:07, 9544.92it/s] 82%|████████▏ | 327189/400000 [00:35<00:07, 9653.36it/s] 82%|████████▏ | 328157/400000 [00:35<00:07, 9635.80it/s] 82%|████████▏ | 329129/400000 [00:35<00:07, 9658.80it/s] 83%|████████▎ | 330096/400000 [00:35<00:07, 9632.33it/s] 83%|████████▎ | 331061/400000 [00:35<00:07, 9600.90it/s] 83%|████████▎ | 332070/400000 [00:35<00:06, 9740.83it/s] 83%|████████▎ | 333069/400000 [00:35<00:06, 9813.97it/s] 84%|████████▎ | 334052/400000 [00:35<00:06, 9799.95it/s] 84%|████████▍ | 335033/400000 [00:36<00:06, 9607.40it/s] 84%|████████▍ | 335995/400000 [00:36<00:06, 9452.60it/s] 84%|████████▍ | 336942/400000 [00:36<00:06, 9404.60it/s] 84%|████████▍ | 337884/400000 [00:36<00:06, 9288.04it/s] 85%|████████▍ | 338814/400000 [00:36<00:06, 9249.11it/s] 85%|████████▍ | 339754/400000 [00:36<00:06, 9291.53it/s] 85%|████████▌ | 340684/400000 [00:36<00:06, 9266.75it/s] 85%|████████▌ | 341669/400000 [00:36<00:06, 9433.72it/s] 86%|████████▌ | 342623/400000 [00:36<00:06, 9463.20it/s] 86%|████████▌ | 343628/400000 [00:36<00:05, 9630.60it/s] 86%|████████▌ | 344593/400000 [00:37<00:05, 9583.15it/s] 86%|████████▋ | 345603/400000 [00:37<00:05, 9729.73it/s] 87%|████████▋ | 346578/400000 [00:37<00:05, 9674.31it/s] 87%|████████▋ | 347547/400000 [00:37<00:05, 9339.48it/s] 87%|████████▋ | 348485/400000 [00:37<00:05, 9189.75it/s] 87%|████████▋ | 349407/400000 [00:37<00:05, 9052.54it/s] 88%|████████▊ | 350315/400000 [00:37<00:05, 8894.33it/s] 88%|████████▊ | 351207/400000 [00:37<00:05, 8668.34it/s] 88%|████████▊ | 352084/400000 [00:37<00:05, 8696.89it/s] 88%|████████▊ | 352981/400000 [00:37<00:05, 8777.03it/s] 88%|████████▊ | 353861/400000 [00:38<00:05, 8747.84it/s] 89%|████████▊ | 354762/400000 [00:38<00:05, 8821.96it/s] 89%|████████▉ | 355687/400000 [00:38<00:04, 8943.58it/s] 89%|████████▉ | 356714/400000 [00:38<00:04, 9303.50it/s] 89%|████████▉ | 357718/400000 [00:38<00:04, 9512.72it/s] 90%|████████▉ | 358674/400000 [00:38<00:04, 9517.52it/s] 90%|████████▉ | 359649/400000 [00:38<00:04, 9583.67it/s] 90%|█████████ | 360617/400000 [00:38<00:04, 9611.52it/s] 90%|█████████ | 361622/400000 [00:38<00:03, 9737.12it/s] 91%|█████████ | 362598/400000 [00:38<00:03, 9642.39it/s] 91%|█████████ | 363564/400000 [00:39<00:03, 9387.93it/s] 91%|█████████ | 364506/400000 [00:39<00:03, 8892.84it/s] 91%|█████████▏| 365441/400000 [00:39<00:03, 9024.96it/s] 92%|█████████▏| 366360/400000 [00:39<00:03, 9073.09it/s] 92%|█████████▏| 367312/400000 [00:39<00:03, 9200.98it/s] 92%|█████████▏| 368236/400000 [00:39<00:03, 9013.84it/s] 92%|█████████▏| 369198/400000 [00:39<00:03, 9186.51it/s] 93%|█████████▎| 370139/400000 [00:39<00:03, 9250.55it/s] 93%|█████████▎| 371067/400000 [00:39<00:03, 9192.50it/s] 93%|█████████▎| 371988/400000 [00:40<00:03, 9066.37it/s] 93%|█████████▎| 372926/400000 [00:40<00:02, 9157.29it/s] 93%|█████████▎| 373880/400000 [00:40<00:02, 9267.91it/s] 94%|█████████▎| 374833/400000 [00:40<00:02, 9343.75it/s] 94%|█████████▍| 375809/400000 [00:40<00:02, 9462.28it/s] 94%|█████████▍| 376763/400000 [00:40<00:02, 9483.49it/s] 94%|█████████▍| 377713/400000 [00:40<00:02, 9348.54it/s] 95%|█████████▍| 378710/400000 [00:40<00:02, 9525.42it/s] 95%|█████████▍| 379674/400000 [00:40<00:02, 9556.85it/s] 95%|█████████▌| 380631/400000 [00:40<00:02, 9544.57it/s] 95%|█████████▌| 381587/400000 [00:41<00:01, 9530.54it/s] 96%|█████████▌| 382541/400000 [00:41<00:01, 9371.07it/s] 96%|█████████▌| 383480/400000 [00:41<00:01, 9306.11it/s] 96%|█████████▌| 384412/400000 [00:41<00:01, 9224.16it/s] 96%|█████████▋| 385395/400000 [00:41<00:01, 9394.72it/s] 97%|█████████▋| 386374/400000 [00:41<00:01, 9507.48it/s] 97%|█████████▋| 387337/400000 [00:41<00:01, 9542.06it/s] 97%|█████████▋| 388355/400000 [00:41<00:01, 9722.56it/s] 97%|█████████▋| 389329/400000 [00:41<00:01, 9576.37it/s] 98%|█████████▊| 390289/400000 [00:41<00:01, 9566.59it/s] 98%|█████████▊| 391247/400000 [00:42<00:00, 9360.81it/s] 98%|█████████▊| 392185/400000 [00:42<00:00, 9129.61it/s] 98%|█████████▊| 393105/400000 [00:42<00:00, 9148.24it/s] 99%|█████████▊| 394072/400000 [00:42<00:00, 9296.02it/s] 99%|█████████▉| 395036/400000 [00:42<00:00, 9395.92it/s] 99%|█████████▉| 395988/400000 [00:42<00:00, 9431.62it/s] 99%|█████████▉| 396933/400000 [00:42<00:00, 9284.01it/s] 99%|█████████▉| 397918/400000 [00:42<00:00, 9442.82it/s]100%|█████████▉| 398864/400000 [00:42<00:00, 9440.19it/s]100%|█████████▉| 399868/400000 [00:42<00:00, 9612.37it/s]100%|█████████▉| 399999/400000 [00:42<00:00, 9302.47it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7efc940c4160>, <torchtext.data.dataset.TabularDataset object at 0x7efc940c42b0>, <torchtext.vocab.Vocab object at 0x7efc940c41d0>), {}) 


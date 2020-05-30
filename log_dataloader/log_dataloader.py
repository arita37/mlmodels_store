
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'cd0e1dbcf68c34dccf0d76405c260752e880d933', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/cd0e1dbcf68c34dccf0d76405c260752e880d933

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f545780b268> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f545780b268> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f54c98590d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f54c98590d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5460fef950> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5460fef950> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f547713d8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f547713d8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f547713d8c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▌      | 3538944/9912422 [00:00<00:00, 35386689.59it/s]9920512it [00:00, 34216261.76it/s]                             
0it [00:00, ?it/s]32768it [00:00, 498945.23it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 155095.31it/s]1654784it [00:00, 11102189.75it/s]                         
0it [00:00, ?it/s]8192it [00:00, 135121.88it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f54579d9550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f54579d9da0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f547713d510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f547713d510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f547713d510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:49,  3.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:49,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:49,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:49,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:48,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:48,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:34,  4.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:34,  4.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:34,  4.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:34,  4.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:34,  4.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:33,  4.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:33,  4.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:24,  6.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:24,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:23,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:23,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:23,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:23,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:23,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:23,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:22,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:16,  8.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:16,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:16,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:16,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:16,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:16,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:15,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:15,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:15,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:11, 11.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:11, 11.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:11, 11.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:11, 11.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:11, 11.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:11, 11.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:11, 11.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:10, 11.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:08, 15.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:08, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:08, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:08, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:07, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:07, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:07, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:07, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:07, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:00<00:05, 20.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:05, 20.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:05, 20.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:05, 20.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:05, 20.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 20.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 20.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 20.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 20.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 25.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 25.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 25.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 25.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 25.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 25.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 25.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 25.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 31.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 38.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 38.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 38.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 38.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 38.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 38.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 38.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 38.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 43.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:01, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 48.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 53.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 58.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 58.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 58.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 58.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 58.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 58.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 58.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 58.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:00, 60.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 62.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 65.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 63.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 63.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 63.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 63.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 63.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 63.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 63.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 63.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 64.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 65.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 66.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 66.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 66.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 66.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 66.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 66.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 66.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 66.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 65.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.70s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.70s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.70s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.01s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.70s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 65.25 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.01s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.01s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 32.31 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.02s/ url]
0 examples [00:00, ? examples/s]2020-05-30 00:09:39.782461: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-30 00:09:39.798353: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-30 00:09:39.798530: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f3936bbcc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-30 00:09:39.798549: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
13 examples [00:00, 127.78 examples/s]109 examples [00:00, 172.68 examples/s]202 examples [00:00, 228.35 examples/s]288 examples [00:00, 292.80 examples/s]375 examples [00:00, 365.06 examples/s]460 examples [00:00, 440.32 examples/s]558 examples [00:00, 527.37 examples/s]647 examples [00:00, 600.45 examples/s]734 examples [00:00, 661.04 examples/s]825 examples [00:01, 719.76 examples/s]912 examples [00:01, 748.54 examples/s]998 examples [00:01, 772.63 examples/s]1083 examples [00:01, 787.00 examples/s]1169 examples [00:01, 805.53 examples/s]1254 examples [00:01, 805.23 examples/s]1344 examples [00:01, 829.31 examples/s]1429 examples [00:01, 792.03 examples/s]1521 examples [00:01, 824.92 examples/s]1618 examples [00:01, 862.52 examples/s]1709 examples [00:02, 870.74 examples/s]1798 examples [00:02, 868.64 examples/s]1889 examples [00:02, 879.87 examples/s]1978 examples [00:02, 868.97 examples/s]2069 examples [00:02, 878.94 examples/s]2164 examples [00:02, 896.91 examples/s]2255 examples [00:02, 888.78 examples/s]2350 examples [00:02, 903.83 examples/s]2441 examples [00:02, 903.76 examples/s]2540 examples [00:02, 926.86 examples/s]2638 examples [00:03, 941.89 examples/s]2733 examples [00:03, 925.59 examples/s]2826 examples [00:03, 914.85 examples/s]2918 examples [00:03, 887.02 examples/s]3011 examples [00:03, 898.12 examples/s]3102 examples [00:03, 887.08 examples/s]3192 examples [00:03, 889.77 examples/s]3282 examples [00:03, 884.64 examples/s]3371 examples [00:03, 864.89 examples/s]3458 examples [00:04, 839.98 examples/s]3554 examples [00:04, 871.18 examples/s]3647 examples [00:04, 886.73 examples/s]3742 examples [00:04, 901.96 examples/s]3835 examples [00:04, 907.54 examples/s]3927 examples [00:04, 882.18 examples/s]4021 examples [00:04, 896.88 examples/s]4116 examples [00:04, 911.94 examples/s]4213 examples [00:04, 928.17 examples/s]4307 examples [00:04, 926.20 examples/s]4400 examples [00:05, 926.40 examples/s]4493 examples [00:05, 919.27 examples/s]4586 examples [00:05, 915.80 examples/s]4678 examples [00:05, 905.05 examples/s]4774 examples [00:05, 919.97 examples/s]4869 examples [00:05, 927.96 examples/s]4964 examples [00:05, 932.02 examples/s]5059 examples [00:05, 935.07 examples/s]5153 examples [00:05, 923.43 examples/s]5251 examples [00:05, 937.66 examples/s]5349 examples [00:06, 949.55 examples/s]5449 examples [00:06, 961.55 examples/s]5546 examples [00:06, 953.87 examples/s]5642 examples [00:06, 941.21 examples/s]5737 examples [00:06, 914.79 examples/s]5829 examples [00:06, 902.10 examples/s]5920 examples [00:06, 873.24 examples/s]6008 examples [00:06, 844.50 examples/s]6095 examples [00:06, 850.28 examples/s]6189 examples [00:07, 874.13 examples/s]6285 examples [00:07, 896.33 examples/s]6376 examples [00:07, 890.82 examples/s]6466 examples [00:07, 876.01 examples/s]6554 examples [00:07, 861.02 examples/s]6642 examples [00:07, 866.03 examples/s]6729 examples [00:07, 843.28 examples/s]6814 examples [00:07, 836.50 examples/s]6899 examples [00:07, 839.48 examples/s]6992 examples [00:07, 862.50 examples/s]7087 examples [00:08, 885.15 examples/s]7182 examples [00:08, 902.33 examples/s]7273 examples [00:08, 889.45 examples/s]7371 examples [00:08, 914.33 examples/s]7470 examples [00:08, 932.93 examples/s]7564 examples [00:08, 922.85 examples/s]7657 examples [00:08, 919.39 examples/s]7750 examples [00:08, 883.92 examples/s]7839 examples [00:08, 858.52 examples/s]7928 examples [00:08, 866.30 examples/s]8025 examples [00:09, 894.01 examples/s]8122 examples [00:09, 913.41 examples/s]8215 examples [00:09, 916.03 examples/s]8309 examples [00:09, 922.08 examples/s]8402 examples [00:09, 909.67 examples/s]8494 examples [00:09, 899.41 examples/s]8591 examples [00:09, 918.79 examples/s]8684 examples [00:09, 907.14 examples/s]8775 examples [00:09, 899.07 examples/s]8866 examples [00:10, 882.55 examples/s]8955 examples [00:10, 883.97 examples/s]9053 examples [00:10, 908.23 examples/s]9145 examples [00:10, 910.76 examples/s]9237 examples [00:10, 895.30 examples/s]9332 examples [00:10, 908.77 examples/s]9424 examples [00:10, 888.58 examples/s]9514 examples [00:10, 885.25 examples/s]9603 examples [00:10, 886.46 examples/s]9695 examples [00:10, 893.66 examples/s]9785 examples [00:11, 892.11 examples/s]9875 examples [00:11, 889.40 examples/s]9972 examples [00:11, 910.14 examples/s]10064 examples [00:11, 854.33 examples/s]10151 examples [00:11, 855.15 examples/s]10238 examples [00:11, 840.60 examples/s]10323 examples [00:11, 839.12 examples/s]10408 examples [00:11, 838.13 examples/s]10499 examples [00:11, 856.86 examples/s]10594 examples [00:11, 882.06 examples/s]10692 examples [00:12, 906.71 examples/s]10785 examples [00:12, 913.31 examples/s]10877 examples [00:12, 897.79 examples/s]10968 examples [00:12, 893.35 examples/s]11058 examples [00:12, 872.95 examples/s]11146 examples [00:12, 868.41 examples/s]11234 examples [00:12, 868.72 examples/s]11321 examples [00:12, 851.55 examples/s]11409 examples [00:12, 858.15 examples/s]11495 examples [00:12, 851.22 examples/s]11581 examples [00:13, 852.22 examples/s]11673 examples [00:13, 871.17 examples/s]11764 examples [00:13, 879.32 examples/s]11857 examples [00:13, 893.44 examples/s]11948 examples [00:13, 896.26 examples/s]12038 examples [00:13, 883.15 examples/s]12127 examples [00:13, 851.44 examples/s]12213 examples [00:13, 851.05 examples/s]12299 examples [00:13, 852.59 examples/s]12390 examples [00:14, 867.77 examples/s]12483 examples [00:14, 883.38 examples/s]12574 examples [00:14, 889.58 examples/s]12667 examples [00:14, 898.50 examples/s]12757 examples [00:14, 884.76 examples/s]12846 examples [00:14, 868.23 examples/s]12935 examples [00:14, 874.02 examples/s]13023 examples [00:14, 872.68 examples/s]13111 examples [00:14, 858.27 examples/s]13197 examples [00:14, 848.86 examples/s]13283 examples [00:15, 849.62 examples/s]13369 examples [00:15, 849.48 examples/s]13454 examples [00:15, 848.10 examples/s]13539 examples [00:15, 842.35 examples/s]13634 examples [00:15, 870.96 examples/s]13725 examples [00:15, 882.31 examples/s]13818 examples [00:15, 893.34 examples/s]13908 examples [00:15, 886.34 examples/s]13997 examples [00:15, 849.46 examples/s]14083 examples [00:15, 848.48 examples/s]14169 examples [00:16, 848.12 examples/s]14255 examples [00:16, 846.34 examples/s]14340 examples [00:16, 836.73 examples/s]14424 examples [00:16, 817.82 examples/s]14506 examples [00:16, 810.20 examples/s]14595 examples [00:16, 832.38 examples/s]14685 examples [00:16, 848.63 examples/s]14771 examples [00:16, 851.30 examples/s]14860 examples [00:16, 862.25 examples/s]14952 examples [00:16, 876.98 examples/s]15047 examples [00:17, 896.60 examples/s]15141 examples [00:17, 908.77 examples/s]15236 examples [00:17, 919.80 examples/s]15330 examples [00:17, 925.50 examples/s]15427 examples [00:17, 937.12 examples/s]15524 examples [00:17, 944.37 examples/s]15622 examples [00:17, 953.20 examples/s]15718 examples [00:17, 919.78 examples/s]15811 examples [00:17, 861.05 examples/s]15899 examples [00:18, 847.09 examples/s]15985 examples [00:18, 844.84 examples/s]16071 examples [00:18, 844.44 examples/s]16156 examples [00:18, 795.48 examples/s]16244 examples [00:18, 816.77 examples/s]16334 examples [00:18, 839.88 examples/s]16429 examples [00:18, 869.03 examples/s]16521 examples [00:18, 882.18 examples/s]16610 examples [00:18, 855.16 examples/s]16699 examples [00:18, 864.71 examples/s]16786 examples [00:19, 821.71 examples/s]16870 examples [00:19, 824.73 examples/s]16956 examples [00:19, 833.29 examples/s]17042 examples [00:19, 838.90 examples/s]17130 examples [00:19, 849.74 examples/s]17216 examples [00:19, 841.91 examples/s]17302 examples [00:19, 846.24 examples/s]17387 examples [00:19, 839.70 examples/s]17477 examples [00:19, 855.50 examples/s]17569 examples [00:20, 872.02 examples/s]17659 examples [00:20, 880.13 examples/s]17748 examples [00:20, 878.34 examples/s]17836 examples [00:20, 874.09 examples/s]17924 examples [00:20, 850.30 examples/s]18013 examples [00:20, 860.02 examples/s]18110 examples [00:20, 888.66 examples/s]18205 examples [00:20, 904.36 examples/s]18299 examples [00:20, 911.90 examples/s]18391 examples [00:20, 902.68 examples/s]18482 examples [00:21, 902.17 examples/s]18574 examples [00:21, 905.85 examples/s]18665 examples [00:21, 900.75 examples/s]18756 examples [00:21, 888.45 examples/s]18845 examples [00:21, 861.02 examples/s]18932 examples [00:21, 842.24 examples/s]19022 examples [00:21, 857.38 examples/s]19121 examples [00:21, 891.16 examples/s]19216 examples [00:21, 905.79 examples/s]19311 examples [00:21, 916.34 examples/s]19405 examples [00:22, 922.18 examples/s]19498 examples [00:22, 892.99 examples/s]19588 examples [00:22, 858.91 examples/s]19680 examples [00:22, 874.77 examples/s]19775 examples [00:22, 893.66 examples/s]19868 examples [00:22, 901.49 examples/s]19961 examples [00:22, 909.75 examples/s]20053 examples [00:22, 822.45 examples/s]20138 examples [00:22, 825.14 examples/s]20228 examples [00:23, 845.28 examples/s]20315 examples [00:23, 850.21 examples/s]20401 examples [00:23, 850.85 examples/s]20496 examples [00:23, 875.96 examples/s]20588 examples [00:23, 887.44 examples/s]20679 examples [00:23, 892.67 examples/s]20769 examples [00:23, 879.79 examples/s]20858 examples [00:23, 845.04 examples/s]20947 examples [00:23, 857.24 examples/s]21034 examples [00:23, 855.95 examples/s]21122 examples [00:24, 862.57 examples/s]21209 examples [00:24, 863.39 examples/s]21296 examples [00:24, 851.26 examples/s]21384 examples [00:24, 859.44 examples/s]21471 examples [00:24, 836.19 examples/s]21562 examples [00:24, 855.10 examples/s]21653 examples [00:24, 868.48 examples/s]21741 examples [00:24, 864.60 examples/s]21832 examples [00:24, 876.46 examples/s]21927 examples [00:24, 895.53 examples/s]22017 examples [00:25, 879.81 examples/s]22106 examples [00:25, 877.54 examples/s]22194 examples [00:25, 875.61 examples/s]22282 examples [00:25, 872.74 examples/s]22370 examples [00:25, 869.17 examples/s]22457 examples [00:25, 860.90 examples/s]22544 examples [00:25, 855.30 examples/s]22632 examples [00:25, 862.11 examples/s]22721 examples [00:25, 868.80 examples/s]22808 examples [00:26, 840.93 examples/s]22893 examples [00:26, 794.54 examples/s]22974 examples [00:26, 797.45 examples/s]23055 examples [00:26, 799.21 examples/s]23151 examples [00:26, 839.46 examples/s]23243 examples [00:26, 860.64 examples/s]23343 examples [00:26, 895.75 examples/s]23435 examples [00:26, 900.62 examples/s]23526 examples [00:26, 902.73 examples/s]23621 examples [00:26, 916.19 examples/s]23713 examples [00:27, 891.25 examples/s]23806 examples [00:27, 901.08 examples/s]23897 examples [00:27, 896.75 examples/s]23987 examples [00:27, 897.60 examples/s]24077 examples [00:27, 895.22 examples/s]24167 examples [00:27, 892.99 examples/s]24257 examples [00:27, 891.87 examples/s]24347 examples [00:27, 889.55 examples/s]24436 examples [00:27, 871.92 examples/s]24524 examples [00:27, 873.26 examples/s]24613 examples [00:28, 878.15 examples/s]24707 examples [00:28, 894.20 examples/s]24797 examples [00:28, 884.54 examples/s]24886 examples [00:28, 870.96 examples/s]24974 examples [00:28, 838.58 examples/s]25059 examples [00:28, 812.91 examples/s]25141 examples [00:28, 811.47 examples/s]25225 examples [00:28, 818.63 examples/s]25314 examples [00:28, 836.20 examples/s]25398 examples [00:29, 817.88 examples/s]25482 examples [00:29, 822.02 examples/s]25565 examples [00:29, 804.96 examples/s]25646 examples [00:29, 783.16 examples/s]25726 examples [00:29, 785.93 examples/s]25805 examples [00:29, 785.11 examples/s]25884 examples [00:29, 783.09 examples/s]25967 examples [00:29, 794.80 examples/s]26050 examples [00:29, 802.69 examples/s]26140 examples [00:29, 829.00 examples/s]26226 examples [00:30, 837.30 examples/s]26311 examples [00:30, 840.53 examples/s]26401 examples [00:30, 857.10 examples/s]26487 examples [00:30, 847.70 examples/s]26581 examples [00:30, 872.13 examples/s]26669 examples [00:30, 872.10 examples/s]26757 examples [00:30, 839.56 examples/s]26842 examples [00:30, 809.28 examples/s]26924 examples [00:30, 812.36 examples/s]27012 examples [00:30, 831.29 examples/s]27100 examples [00:31, 844.31 examples/s]27185 examples [00:31, 840.10 examples/s]27270 examples [00:31, 837.82 examples/s]27354 examples [00:31, 821.66 examples/s]27437 examples [00:31, 820.91 examples/s]27520 examples [00:31, 822.69 examples/s]27603 examples [00:31, 804.32 examples/s]27684 examples [00:31, 773.07 examples/s]27764 examples [00:31, 780.18 examples/s]27850 examples [00:32, 802.23 examples/s]27942 examples [00:32, 832.97 examples/s]28030 examples [00:32, 845.24 examples/s]28126 examples [00:32, 875.04 examples/s]28215 examples [00:32, 836.21 examples/s]28300 examples [00:32, 831.31 examples/s]28384 examples [00:32, 822.58 examples/s]28467 examples [00:32, 819.74 examples/s]28559 examples [00:32, 845.79 examples/s]28644 examples [00:32, 842.94 examples/s]28729 examples [00:33, 839.35 examples/s]28814 examples [00:33, 774.39 examples/s]28893 examples [00:33, 757.99 examples/s]28979 examples [00:33, 783.95 examples/s]29072 examples [00:33, 819.45 examples/s]29161 examples [00:33, 836.79 examples/s]29251 examples [00:33, 849.75 examples/s]29342 examples [00:33, 865.71 examples/s]29430 examples [00:33, 868.78 examples/s]29521 examples [00:33, 878.04 examples/s]29613 examples [00:34, 887.70 examples/s]29702 examples [00:34, 854.63 examples/s]29788 examples [00:34, 853.52 examples/s]29874 examples [00:34, 854.11 examples/s]29960 examples [00:34, 837.21 examples/s]30044 examples [00:34, 777.40 examples/s]30129 examples [00:34, 795.83 examples/s]30215 examples [00:34, 800.42 examples/s]30296 examples [00:34, 769.20 examples/s]30384 examples [00:35, 799.01 examples/s]30477 examples [00:35, 832.00 examples/s]30562 examples [00:35, 834.83 examples/s]30647 examples [00:35, 827.10 examples/s]30731 examples [00:35, 810.39 examples/s]30813 examples [00:35, 810.73 examples/s]30908 examples [00:35, 847.66 examples/s]30999 examples [00:35, 863.54 examples/s]31086 examples [00:35, 839.99 examples/s]31171 examples [00:35, 835.38 examples/s]31255 examples [00:36, 836.57 examples/s]31346 examples [00:36, 855.69 examples/s]31432 examples [00:36, 856.92 examples/s]31518 examples [00:36, 856.73 examples/s]31604 examples [00:36, 852.93 examples/s]31690 examples [00:36, 835.12 examples/s]31782 examples [00:36, 858.76 examples/s]31869 examples [00:36, 844.24 examples/s]31954 examples [00:36, 833.27 examples/s]32040 examples [00:37, 838.83 examples/s]32128 examples [00:37, 849.04 examples/s]32216 examples [00:37, 854.77 examples/s]32302 examples [00:37, 828.25 examples/s]32389 examples [00:37, 838.82 examples/s]32476 examples [00:37, 845.04 examples/s]32563 examples [00:37, 851.10 examples/s]32651 examples [00:37, 857.05 examples/s]32741 examples [00:37, 865.89 examples/s]32828 examples [00:37, 864.44 examples/s]32921 examples [00:38, 882.30 examples/s]33012 examples [00:38, 888.43 examples/s]33104 examples [00:38, 895.17 examples/s]33194 examples [00:38, 884.31 examples/s]33283 examples [00:38, 881.88 examples/s]33372 examples [00:38, 883.14 examples/s]33465 examples [00:38, 894.30 examples/s]33559 examples [00:38, 907.35 examples/s]33652 examples [00:38, 911.74 examples/s]33748 examples [00:38, 924.00 examples/s]33841 examples [00:39, 882.35 examples/s]33930 examples [00:39, 873.29 examples/s]34018 examples [00:39, 868.71 examples/s]34106 examples [00:39, 864.94 examples/s]34198 examples [00:39, 879.83 examples/s]34289 examples [00:39, 886.08 examples/s]34379 examples [00:39, 887.67 examples/s]34468 examples [00:39, 861.38 examples/s]34555 examples [00:39, 834.60 examples/s]34641 examples [00:40, 839.75 examples/s]34727 examples [00:40, 844.65 examples/s]34815 examples [00:40, 854.28 examples/s]34903 examples [00:40, 860.53 examples/s]34990 examples [00:40, 855.29 examples/s]35076 examples [00:40, 829.56 examples/s]35160 examples [00:40, 816.74 examples/s]35242 examples [00:40, 808.91 examples/s]35329 examples [00:40, 824.86 examples/s]35412 examples [00:40, 819.31 examples/s]35501 examples [00:41, 838.40 examples/s]35586 examples [00:41, 837.76 examples/s]35670 examples [00:41, 831.64 examples/s]35754 examples [00:41, 810.46 examples/s]35841 examples [00:41, 827.20 examples/s]35924 examples [00:41, 810.47 examples/s]36012 examples [00:41, 828.53 examples/s]36111 examples [00:41, 869.79 examples/s]36204 examples [00:41, 886.38 examples/s]36295 examples [00:41, 892.12 examples/s]36388 examples [00:42, 900.41 examples/s]36479 examples [00:42, 873.98 examples/s]36567 examples [00:42, 869.19 examples/s]36655 examples [00:42, 845.81 examples/s]36743 examples [00:42, 853.70 examples/s]36834 examples [00:42, 866.96 examples/s]36921 examples [00:42, 857.23 examples/s]37016 examples [00:42, 881.55 examples/s]37108 examples [00:42, 889.98 examples/s]37201 examples [00:42, 899.07 examples/s]37292 examples [00:43, 882.26 examples/s]37386 examples [00:43, 898.02 examples/s]37477 examples [00:43, 885.13 examples/s]37568 examples [00:43, 890.34 examples/s]37664 examples [00:43, 908.84 examples/s]37756 examples [00:43, 910.81 examples/s]37850 examples [00:43, 919.16 examples/s]37944 examples [00:43, 923.37 examples/s]38037 examples [00:43, 908.85 examples/s]38128 examples [00:44, 899.46 examples/s]38221 examples [00:44, 907.31 examples/s]38313 examples [00:44, 908.73 examples/s]38404 examples [00:44, 902.10 examples/s]38495 examples [00:44, 866.57 examples/s]38587 examples [00:44, 877.86 examples/s]38676 examples [00:44, 867.02 examples/s]38772 examples [00:44, 891.20 examples/s]38862 examples [00:44, 889.70 examples/s]38952 examples [00:44, 863.25 examples/s]39039 examples [00:45, 823.26 examples/s]39123 examples [00:45, 827.66 examples/s]39207 examples [00:45, 830.31 examples/s]39293 examples [00:45, 836.59 examples/s]39380 examples [00:45, 845.28 examples/s]39465 examples [00:45, 832.11 examples/s]39549 examples [00:45, 830.80 examples/s]39635 examples [00:45, 837.78 examples/s]39721 examples [00:45, 841.64 examples/s]39815 examples [00:45, 867.64 examples/s]39909 examples [00:46, 887.08 examples/s]40001 examples [00:46, 874.47 examples/s]40095 examples [00:46, 890.24 examples/s]40185 examples [00:46, 868.83 examples/s]40273 examples [00:46, 867.17 examples/s]40361 examples [00:46, 868.52 examples/s]40452 examples [00:46, 878.06 examples/s]40540 examples [00:46, 874.53 examples/s]40628 examples [00:46, 862.91 examples/s]40720 examples [00:47, 877.93 examples/s]40814 examples [00:47, 893.85 examples/s]40904 examples [00:47, 888.94 examples/s]40994 examples [00:47, 882.34 examples/s]41085 examples [00:47, 890.46 examples/s]41184 examples [00:47, 915.99 examples/s]41276 examples [00:47, 880.65 examples/s]41367 examples [00:47, 887.29 examples/s]41457 examples [00:47, 858.12 examples/s]41544 examples [00:47, 835.53 examples/s]41628 examples [00:48, 811.69 examples/s]41717 examples [00:48, 832.11 examples/s]41808 examples [00:48, 852.73 examples/s]41894 examples [00:48, 848.62 examples/s]41980 examples [00:48, 842.20 examples/s]42065 examples [00:48, 817.82 examples/s]42148 examples [00:48, 814.93 examples/s]42230 examples [00:48, 783.31 examples/s]42309 examples [00:48, 770.67 examples/s]42387 examples [00:49, 768.00 examples/s]42475 examples [00:49, 797.61 examples/s]42565 examples [00:49, 824.90 examples/s]42660 examples [00:49, 856.53 examples/s]42747 examples [00:49, 848.83 examples/s]42833 examples [00:49, 851.17 examples/s]42922 examples [00:49, 859.94 examples/s]43010 examples [00:49, 861.90 examples/s]43109 examples [00:49, 895.47 examples/s]43200 examples [00:49, 885.81 examples/s]43292 examples [00:50, 894.88 examples/s]43383 examples [00:50, 896.71 examples/s]43473 examples [00:50, 885.47 examples/s]43562 examples [00:50, 877.62 examples/s]43650 examples [00:50, 863.95 examples/s]43737 examples [00:50, 837.78 examples/s]43822 examples [00:50, 776.62 examples/s]43909 examples [00:50, 798.94 examples/s]43999 examples [00:50, 825.85 examples/s]44085 examples [00:50, 834.43 examples/s]44174 examples [00:51, 849.33 examples/s]44261 examples [00:51, 854.07 examples/s]44347 examples [00:51, 841.73 examples/s]44436 examples [00:51, 850.47 examples/s]44522 examples [00:51, 842.67 examples/s]44608 examples [00:51, 836.64 examples/s]44701 examples [00:51, 860.68 examples/s]44796 examples [00:51, 883.03 examples/s]44889 examples [00:51, 894.89 examples/s]44981 examples [00:52, 901.94 examples/s]45073 examples [00:52, 905.51 examples/s]45164 examples [00:52, 901.19 examples/s]45255 examples [00:52, 884.90 examples/s]45349 examples [00:52, 898.22 examples/s]45439 examples [00:52, 892.58 examples/s]45529 examples [00:52, 867.50 examples/s]45616 examples [00:52, 859.49 examples/s]45703 examples [00:52, 846.83 examples/s]45792 examples [00:52, 858.47 examples/s]45879 examples [00:53, 858.38 examples/s]45969 examples [00:53, 867.66 examples/s]46056 examples [00:53, 858.40 examples/s]46142 examples [00:53, 846.60 examples/s]46232 examples [00:53, 860.34 examples/s]46328 examples [00:53, 886.72 examples/s]46422 examples [00:53, 901.17 examples/s]46513 examples [00:53, 854.21 examples/s]46600 examples [00:53, 856.72 examples/s]46687 examples [00:53, 860.10 examples/s]46774 examples [00:54, 853.19 examples/s]46861 examples [00:54, 857.18 examples/s]46947 examples [00:54, 849.71 examples/s]47035 examples [00:54, 856.60 examples/s]47123 examples [00:54, 863.05 examples/s]47210 examples [00:54, 831.15 examples/s]47296 examples [00:54, 837.84 examples/s]47381 examples [00:54, 833.66 examples/s]47465 examples [00:54, 805.15 examples/s]47546 examples [00:55, 802.02 examples/s]47639 examples [00:55, 836.18 examples/s]47741 examples [00:55, 882.74 examples/s]47831 examples [00:55, 875.00 examples/s]47920 examples [00:55, 849.09 examples/s]48009 examples [00:55, 858.38 examples/s]48100 examples [00:55, 871.69 examples/s]48188 examples [00:55, 844.30 examples/s]48278 examples [00:55, 859.95 examples/s]48369 examples [00:55, 872.03 examples/s]48457 examples [00:56, 874.40 examples/s]48552 examples [00:56, 893.70 examples/s]48652 examples [00:56, 922.65 examples/s]48746 examples [00:56, 926.08 examples/s]48847 examples [00:56, 948.72 examples/s]48945 examples [00:56, 956.64 examples/s]49041 examples [00:56, 928.89 examples/s]49135 examples [00:56, 888.21 examples/s]49227 examples [00:56, 895.63 examples/s]49318 examples [00:56, 854.36 examples/s]49405 examples [00:57, 856.02 examples/s]49492 examples [00:57, 845.88 examples/s]49578 examples [00:57, 848.97 examples/s]49668 examples [00:57, 861.91 examples/s]49760 examples [00:57, 876.56 examples/s]49848 examples [00:57, 868.44 examples/s]49943 examples [00:57, 891.09 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█▏        | 5660/50000 [00:00<00:00, 56598.84 examples/s] 32%|███▏      | 16034/50000 [00:00<00:00, 65531.95 examples/s] 57%|█████▋    | 28372/50000 [00:00<00:00, 76257.15 examples/s] 81%|████████  | 40494/50000 [00:00<00:00, 85803.60 examples/s]                                                               0 examples [00:00, ? examples/s]81 examples [00:00, 806.64 examples/s]184 examples [00:00, 862.17 examples/s]272 examples [00:00, 865.52 examples/s]357 examples [00:00, 858.85 examples/s]447 examples [00:00, 870.49 examples/s]534 examples [00:00, 869.27 examples/s]625 examples [00:00, 878.69 examples/s]707 examples [00:00, 676.61 examples/s]797 examples [00:00, 730.15 examples/s]883 examples [00:01, 763.28 examples/s]970 examples [00:01, 790.28 examples/s]1054 examples [00:01, 803.75 examples/s]1136 examples [00:01, 800.82 examples/s]1222 examples [00:01, 817.23 examples/s]1305 examples [00:01, 812.66 examples/s]1400 examples [00:01, 849.19 examples/s]1498 examples [00:01, 882.76 examples/s]1588 examples [00:01, 879.88 examples/s]1677 examples [00:02, 880.36 examples/s]1766 examples [00:02, 881.30 examples/s]1855 examples [00:02, 875.09 examples/s]1943 examples [00:02, 854.57 examples/s]2029 examples [00:02, 844.98 examples/s]2114 examples [00:02, 845.65 examples/s]2204 examples [00:02, 859.64 examples/s]2291 examples [00:02, 862.03 examples/s]2381 examples [00:02, 871.25 examples/s]2469 examples [00:02, 864.51 examples/s]2557 examples [00:03, 867.10 examples/s]2645 examples [00:03, 869.79 examples/s]2733 examples [00:03, 858.88 examples/s]2819 examples [00:03, 858.48 examples/s]2905 examples [00:03, 821.88 examples/s]2989 examples [00:03, 826.34 examples/s]3078 examples [00:03, 842.62 examples/s]3169 examples [00:03, 859.88 examples/s]3257 examples [00:03, 865.18 examples/s]3350 examples [00:03, 882.11 examples/s]3442 examples [00:04, 891.66 examples/s]3536 examples [00:04, 904.19 examples/s]3627 examples [00:04, 892.05 examples/s]3717 examples [00:04, 893.80 examples/s]3807 examples [00:04, 877.94 examples/s]3898 examples [00:04, 884.57 examples/s]3987 examples [00:04, 869.81 examples/s]4076 examples [00:04, 873.15 examples/s]4171 examples [00:04, 892.21 examples/s]4261 examples [00:04, 878.59 examples/s]4350 examples [00:05, 869.02 examples/s]4438 examples [00:05, 847.59 examples/s]4526 examples [00:05, 854.95 examples/s]4619 examples [00:05, 875.81 examples/s]4711 examples [00:05, 887.01 examples/s]4806 examples [00:05, 903.90 examples/s]4898 examples [00:05, 907.60 examples/s]4989 examples [00:05, 895.18 examples/s]5079 examples [00:05, 888.18 examples/s]5172 examples [00:06, 898.26 examples/s]5266 examples [00:06, 908.08 examples/s]5357 examples [00:06, 904.74 examples/s]5451 examples [00:06, 912.44 examples/s]5543 examples [00:06, 892.46 examples/s]5634 examples [00:06, 897.58 examples/s]5726 examples [00:06, 903.87 examples/s]5817 examples [00:06, 895.37 examples/s]5907 examples [00:06, 889.19 examples/s]5999 examples [00:06, 896.52 examples/s]6089 examples [00:07, 865.61 examples/s]6181 examples [00:07, 878.64 examples/s]6273 examples [00:07, 890.43 examples/s]6363 examples [00:07, 890.44 examples/s]6453 examples [00:07, 877.07 examples/s]6541 examples [00:07, 862.87 examples/s]6628 examples [00:07, 829.75 examples/s]6714 examples [00:07, 837.66 examples/s]6800 examples [00:07, 842.10 examples/s]6886 examples [00:07, 845.07 examples/s]6971 examples [00:08, 845.25 examples/s]7058 examples [00:08, 852.02 examples/s]7144 examples [00:08, 832.19 examples/s]7228 examples [00:08, 833.14 examples/s]7314 examples [00:08, 839.34 examples/s]7400 examples [00:08, 845.10 examples/s]7495 examples [00:08, 873.47 examples/s]7587 examples [00:08, 885.66 examples/s]7678 examples [00:08, 891.85 examples/s]7768 examples [00:08, 863.72 examples/s]7855 examples [00:09, 852.11 examples/s]7941 examples [00:09, 853.22 examples/s]8027 examples [00:09, 848.37 examples/s]8115 examples [00:09, 856.35 examples/s]8204 examples [00:09, 864.20 examples/s]8294 examples [00:09, 872.40 examples/s]8393 examples [00:09, 903.82 examples/s]8485 examples [00:09, 906.73 examples/s]8576 examples [00:09, 894.93 examples/s]8666 examples [00:10, 884.26 examples/s]8756 examples [00:10, 887.13 examples/s]8848 examples [00:10, 896.61 examples/s]8940 examples [00:10, 899.94 examples/s]9031 examples [00:10, 892.38 examples/s]9121 examples [00:10, 879.78 examples/s]9210 examples [00:10, 875.30 examples/s]9298 examples [00:10, 854.39 examples/s]9384 examples [00:10, 847.77 examples/s]9469 examples [00:10, 844.87 examples/s]9554 examples [00:11, 834.68 examples/s]9638 examples [00:11, 818.88 examples/s]9733 examples [00:11, 853.32 examples/s]9819 examples [00:11, 833.45 examples/s]9906 examples [00:11, 842.89 examples/s]9993 examples [00:11, 846.18 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteEJPXCX/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteEJPXCX/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f547713d8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f547713d8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f547713d8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f54dd33f278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f54542444a8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f547713d8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f547713d8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f547713d8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f54586ea470>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f54586ea320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f544c1926a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f544c1926a8> 

  function with postional parmater data_info <function split_train_valid at 0x7f544c1926a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f544c1927b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f544c1927b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f544c1927b8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=42700bc46d1b7475f61cadf09f770b1c345cd1e0aa589a9773d68e4e68eab9d5
  Stored in directory: /tmp/pip-ephem-wheel-cache-h3b711fc/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:08<254:45:21, 940B/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:08<178:33:15, 1.34kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:08<125:01:07, 1.92kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:09<87:27:30, 2.74kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:09<61:01:45, 3.91kB/s].vector_cache/glove.6B.zip:   1%|          | 9.27M/862M [00:09<42:26:31, 5.58kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.7M/862M [00:09<29:31:17, 7.97kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.8M/862M [00:09<20:35:29, 11.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.4M/862M [00:09<14:19:11, 16.3kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:09<9:59:27, 23.2kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 31.1M/862M [00:09<6:57:19, 33.2kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.9M/862M [00:09<4:50:55, 47.4kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.9M/862M [00:10<3:22:28, 67.7kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.3M/862M [00:10<2:21:16, 96.6kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.5M/862M [00:10<1:38:20, 138kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:10<1:09:00, 196kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:12<49:58, 269kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:12<38:37, 348kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:12<27:57, 480kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:13<19:43, 679kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:13<16:19, 819kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:14<7:56:18, 28.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:14<5:33:16, 40.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:16<3:55:05, 56.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:16<2:47:37, 79.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:16<1:57:58, 113kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:16<1:22:27, 161kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:18<1:11:51, 184kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:18<51:24, 257kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:18<36:15, 364kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:20<28:26, 463kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:20<22:35, 583kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:20<16:22, 803kB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.5M/862M [00:20<11:34, 1.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:22<19:09, 684kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:22<14:44, 888kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:22<10:38, 1.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:24<10:28, 1.24MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:24<10:06, 1.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:24<07:38, 1.70MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:24<05:30, 2.36MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:26<10:24, 1.25MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:26<08:38, 1.50MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:26<06:21, 2.03MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:28<07:28, 1.73MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:28<07:58, 1.62MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:28<06:07, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:28<04:28, 2.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:30<08:37, 1.49MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:30<07:10, 1.79MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:30<05:16, 2.43MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.4M/862M [00:30<03:52, 3.30MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:32<27:34, 463kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:32<20:35, 619kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:32<14:42, 865kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:33<13:16, 955kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:34<11:52, 1.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:34<08:51, 1.43MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:34<06:19, 1.99MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:35<13:19, 948kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:36<10:37, 1.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:36<07:44, 1.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:37<08:21, 1.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:38<07:09, 1.75MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:38<05:19, 2.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:39<06:38, 1.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:40<05:55, 2.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:40<04:24, 2.83MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:41<06:02, 2.06MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:41<05:29, 2.26MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:42<04:08, 2.99MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:43<05:49, 2.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:43<06:36, 1.87MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:44<05:09, 2.39MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:44<03:45, 3.27MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:45<09:49, 1.25MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:45<08:07, 1.51MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:46<05:59, 2.05MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:47<07:02, 1.73MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:47<07:25, 1.64MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:47<05:43, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:48<04:09, 2.92MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:49<08:40, 1.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:49<07:18, 1.66MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:49<05:24, 2.24MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:51<06:37, 1.82MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:51<07:05, 1.70MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:51<05:29, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:52<03:59, 3.01MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:53<09:15, 1.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:53<07:42, 1.56MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:53<05:41, 2.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:54<04:43, 2.53MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:55<8:37:33, 23.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:55<6:02:43, 32.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:55<4:13:12, 47.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:57<3:03:21, 64.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:57<2:10:59, 90.6kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:57<1:32:08, 129kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:57<1:04:37, 183kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:59<48:16, 245kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:59<34:59, 337kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:59<24:44, 476kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:01<20:03, 585kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:01<16:26, 714kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:01<11:59, 977kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:01<08:35, 1.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [01:03<10:07, 1.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [01:03<09:34, 1.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:03<07:12, 1.62MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:03<05:14, 2.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:05<07:34, 1.53MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:05<06:30, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:05<04:48, 2.40MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:07<06:02, 1.91MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:07<05:24, 2.13MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:07<04:04, 2.82MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:09<05:33, 2.06MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:09<06:13, 1.84MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:09<04:56, 2.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:11<05:21, 2.13MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:11<04:45, 2.39MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:11<03:38, 3.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:11<02:41, 4.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:13<23:52, 474kB/s] .vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:13<19:02, 595kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:13<13:48, 819kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:13<09:47, 1.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:15<12:12, 922kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:15<09:43, 1.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:15<07:02, 1.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:17<07:32, 1.48MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:17<07:37, 1.47MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:17<05:50, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:17<04:14, 2.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:19<08:13, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:19<06:53, 1.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:19<05:03, 2.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:21<06:04, 1.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:21<06:32, 1.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:21<05:03, 2.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:21<03:40, 2.99MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:23<08:34, 1.28MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:23<07:06, 1.54MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:23<05:15, 2.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:25<06:14, 1.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:25<05:28, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:25<04:05, 2.65MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:27<05:25, 2.00MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:27<04:54, 2.21MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:27<03:39, 2.95MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:29<05:07, 2.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:29<04:31, 2.38MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:29<03:24, 3.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:29<02:31, 4.25MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:30<44:53, 238kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:31<32:31, 329kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:31<22:58, 464kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:32<18:33, 573kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:33<15:10, 701kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:33<11:08, 953kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:33<07:53, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:34<10:16:04, 17.2kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:35<7:12:04, 24.4kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:35<5:01:58, 34.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:36<3:33:07, 49.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:36<2:30:10, 69.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:37<1:45:07, 99.6kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:38<1:15:47, 138kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:38<55:10, 189kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:39<39:00, 267kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:39<27:20, 380kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:40<25:39, 404kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:40<19:01, 544kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:41<13:30, 764kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:42<11:50, 869kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:42<10:29, 981kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:42<07:46, 1.32MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:43<05:34, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:44<08:15, 1.24MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:44<06:40, 1.53MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:44<04:52, 2.09MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:45<03:32, 2.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:46<28:04, 362kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:46<21:42, 468kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:46<15:36, 650kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:47<11:01, 916kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:48<13:06, 770kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:48<10:12, 987kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:48<07:21, 1.37MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:49<05:43, 1.75MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:50<7:52:04, 21.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:50<5:30:32, 30.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:50<3:50:18, 43.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:52<3:04:03, 54.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:52<2:10:49, 76.0kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:52<1:31:57, 108kB/s] .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:54<1:05:38, 151kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:54<46:58, 210kB/s]  .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:54<33:03, 298kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:56<25:18, 388kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:56<19:42, 498kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:56<14:16, 686kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:58<11:31, 846kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:58<09:03, 1.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:58<06:34, 1.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [02:00<06:51, 1.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [02:00<05:47, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [02:00<04:14, 2.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [02:02<05:14, 1.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [02:02<04:38, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:02<03:29, 2.75MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:04<04:41, 2.03MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:04<05:13, 1.82MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:04<04:08, 2.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:06<04:25, 2.14MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:06<03:53, 2.43MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:06<02:55, 3.22MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:06<02:10, 4.33MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:08<39:25, 238kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:08<29:30, 319kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:08<21:02, 446kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:08<14:46, 633kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:10<16:54, 552kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:10<12:48, 728kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:10<09:10, 1.01MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:12<08:33, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:12<06:56, 1.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:12<05:04, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:14<05:43, 1.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:14<05:52, 1.57MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:14<04:34, 2.01MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:16<04:40, 1.96MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:16<04:11, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:16<03:09, 2.88MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:18<04:26, 2.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:18<04:56, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:18<03:51, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:18<02:47, 3.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:20<07:56, 1.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:20<06:29, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:20<04:45, 1.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:22<05:25, 1.65MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:22<04:42, 1.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:22<03:30, 2.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:24<04:32, 1.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:24<04:05, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:24<03:03, 2.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:25<04:13, 2.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:26<04:44, 1.85MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:26<03:45, 2.33MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:26<02:43, 3.20MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:27<08:58, 972kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:28<07:27, 1.17MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:28<05:26, 1.59MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:28<03:57, 2.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:29<08:40, 998kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:30<07:06, 1.22MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:30<05:11, 1.66MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:31<05:24, 1.59MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:32<04:51, 1.77MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:32<03:40, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:33<04:18, 1.97MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:34<04:00, 2.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:34<03:03, 2.78MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:35<03:52, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:36<04:40, 1.80MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:36<03:46, 2.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:36<02:44, 3.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:37<08:50, 947kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:38<07:10, 1.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:38<05:15, 1.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:39<05:23, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:40<05:48, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:40<04:33, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:40<03:17, 2.50MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:41<08:12, 1.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:41<06:43, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:42<04:56, 1.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:43<05:11, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:44<05:38, 1.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:44<04:25, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:44<03:11, 2.54MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:45<07:44, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:46<06:21, 1.27MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:46<04:38, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:46<03:20, 2.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:47<20:09, 398kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:47<16:04, 499kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:48<11:39, 687kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:48<08:12, 970kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:49<10:22, 767kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:49<08:10, 972kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:50<05:56, 1.33MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:51<05:47, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:51<04:57, 1.59MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:52<03:42, 2.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:53<04:12, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:53<03:51, 2.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:54<02:55, 2.67MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:55<03:39, 2.11MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:55<04:27, 1.74MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:56<03:34, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:56<02:36, 2.95MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:57<07:19, 1.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:57<06:02, 1.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:58<04:26, 1.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:59<04:40, 1.63MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:59<05:07, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:00<03:58, 1.91MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [03:00<02:55, 2.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:01<04:07, 1.83MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:01<03:45, 2.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:02<02:51, 2.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:03<03:32, 2.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:03<04:13, 1.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:04<03:20, 2.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:04<02:28, 3.01MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:05<03:47, 1.95MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:05<03:30, 2.10MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:06<02:40, 2.76MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:07<03:23, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:07<04:04, 1.80MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:07<03:13, 2.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:08<02:21, 3.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:09<04:31, 1.61MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:09<04:01, 1.80MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:09<02:59, 2.41MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:11<03:35, 2.00MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:11<04:16, 1.68MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:11<03:21, 2.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:12<02:28, 2.88MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:13<03:38, 1.96MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:13<03:24, 2.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:13<02:32, 2.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:15<03:14, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:15<03:06, 2.27MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:15<02:23, 2.94MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:17<03:06, 2.25MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:17<03:52, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:17<03:07, 2.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:18<02:15, 3.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:19<06:15, 1.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:19<05:11, 1.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:19<03:47, 1.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:21<04:05, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:21<04:27, 1.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:21<03:27, 1.98MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:21<02:32, 2.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:23<03:35, 1.88MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:23<03:19, 2.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:23<02:28, 2.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:25<03:07, 2.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:25<03:50, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:25<03:01, 2.21MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:25<02:12, 3.01MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:27<03:50, 1.73MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:27<03:28, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:27<02:37, 2.52MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:29<03:11, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:29<03:49, 1.71MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:29<03:00, 2.17MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:29<02:10, 2.99MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:31<04:48, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:31<04:06, 1.58MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:31<03:03, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:33<03:28, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:33<03:55, 1.64MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:33<03:07, 2.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:33<02:16, 2.81MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:35<06:12, 1.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:35<05:02, 1.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:35<03:41, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:37<03:55, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:37<04:17, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:37<03:18, 1.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:37<02:24, 2.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:39<03:45, 1.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:39<03:22, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:39<02:32, 2.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:41<03:02, 2.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:41<03:38, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:41<02:51, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:41<02:04, 2.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:43<03:39, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:43<03:16, 1.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:43<02:26, 2.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:45<02:57, 2.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:45<03:32, 1.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:45<02:50, 2.11MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:45<02:03, 2.90MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:47<05:24, 1.10MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:47<04:28, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:47<03:17, 1.79MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:49<03:30, 1.67MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:49<03:08, 1.87MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:49<02:20, 2.50MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:49<01:41, 3.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:51<22:58, 252kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:51<17:25, 333kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:51<12:30, 463kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:51<08:46, 655kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:53<10:04, 569kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:53<07:42, 743kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:53<05:32, 1.03MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:55<05:02, 1.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:55<04:50, 1.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:55<03:40, 1.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:55<02:39, 2.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:57<03:35, 1.56MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:57<03:00, 1.86MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:57<02:13, 2.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:57<01:37, 3.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:59<07:18, 755kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:59<06:24, 861kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:59<04:46, 1.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:59<03:24, 1.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:01<04:10, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:01<03:34, 1.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:01<02:39, 2.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:03<02:58, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:03<03:23, 1.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:03<02:38, 2.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:03<01:55, 2.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:05<03:17, 1.61MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:05<02:56, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:05<02:12, 2.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:07<02:37, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:07<03:07, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:07<02:27, 2.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:07<01:46, 2.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:09<03:31, 1.47MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:09<03:04, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:09<02:16, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:11<02:39, 1.93MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:11<03:06, 1.64MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:11<02:25, 2.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:11<01:47, 2.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:13<02:39, 1.89MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:13<02:26, 2.06MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:13<01:51, 2.70MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:15<02:19, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:15<02:46, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:15<02:12, 2.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:15<01:35, 3.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:17<03:33, 1.38MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:17<03:03, 1.60MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:17<02:16, 2.14MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:19<02:35, 1.87MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:19<02:57, 1.63MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:19<02:18, 2.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:19<01:40, 2.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:21<02:40, 1.78MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:21<02:26, 1.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:21<01:49, 2.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:23<02:15, 2.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:23<02:43, 1.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:23<02:09, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:23<01:34, 2.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:25<04:16, 1.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:25<03:32, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:25<02:36, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:27<02:45, 1.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:27<02:57, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:27<02:15, 2.01MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:27<01:39, 2.72MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:29<02:29, 1.80MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:29<02:15, 1.98MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:29<01:40, 2.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:31<02:06, 2.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:31<02:24, 1.83MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:31<01:55, 2.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:31<01:23, 3.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:33<07:48, 557kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:33<05:55, 732kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:33<04:14, 1.02MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:35<03:53, 1.10MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:35<03:38, 1.18MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:35<02:46, 1.54MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:35<01:57, 2.14MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:37<07:32, 557kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:37<05:46, 727kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:37<04:08, 1.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:39<03:44, 1.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:39<03:31, 1.17MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:39<02:38, 1.56MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:39<01:55, 2.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:41<02:28, 1.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:41<02:12, 1.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:41<01:39, 2.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:43<01:59, 2.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:43<02:21, 1.69MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:43<01:51, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:43<01:20, 2.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:45<02:20, 1.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:45<02:06, 1.86MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:45<01:34, 2.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:47<01:53, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:47<01:45, 2.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:47<01:19, 2.89MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:49<01:45, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:49<02:09, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:49<01:42, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:49<01:13, 3.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:51<02:34, 1.44MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:51<02:14, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:51<01:39, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:53<01:54, 1.91MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:53<02:15, 1.62MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:53<01:47, 2.03MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:53<01:17, 2.77MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:55<03:22, 1.06MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:55<02:47, 1.28MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:55<02:02, 1.74MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:57<02:08, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:57<01:55, 1.82MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:57<01:25, 2.45MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:58<01:42, 2.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:59<02:02, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:59<01:37, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:59<01:10, 2.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:00<03:06, 1.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:01<02:30, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:01<01:50, 1.81MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:02<01:58, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:03<02:08, 1.54MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:03<01:41, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:03<01:12, 2.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:04<03:03, 1.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:05<02:30, 1.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:05<01:50, 1.74MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:06<01:55, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:07<02:06, 1.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:07<01:38, 1.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:07<01:10, 2.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:08<02:25, 1.28MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:09<02:02, 1.52MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:09<01:29, 2.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:10<01:41, 1.78MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:11<01:33, 1.95MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:11<01:09, 2.57MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:12<01:25, 2.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:13<01:42, 1.72MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:13<01:20, 2.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:13<00:59, 2.93MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:14<01:25, 2.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:15<01:21, 2.13MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:15<01:00, 2.82MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:16<01:17, 2.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:17<01:36, 1.75MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:17<01:15, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:17<00:55, 3.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:18<01:34, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:18<01:26, 1.91MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:19<01:04, 2.52MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:20<01:17, 2.06MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:20<01:33, 1.71MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:21<01:13, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:21<00:53, 2.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:22<01:29, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:22<01:18, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:23<00:58, 2.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:23<00:42, 3.57MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:24<02:09, 1.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:24<02:05, 1.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:25<01:36, 1.57MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:25<01:08, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:26<02:35, 953kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:26<02:05, 1.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:27<01:31, 1.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:28<01:33, 1.55MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:28<01:21, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:29<01:00, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:30<01:11, 1.97MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:30<01:22, 1.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:31<01:05, 2.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:31<00:47, 2.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:32<02:11, 1.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:32<01:46, 1.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:33<01:16, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:34<01:21, 1.60MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:34<01:27, 1.49MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:34<01:07, 1.92MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:35<00:48, 2.63MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:36<01:16, 1.67MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:36<01:08, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:36<00:50, 2.48MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:38<01:00, 2.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:38<01:12, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:38<00:56, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:39<00:40, 2.94MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:40<01:08, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:40<01:02, 1.91MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:40<00:45, 2.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:42<00:55, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:42<01:06, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:42<00:53, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:43<00:37, 2.94MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:44<01:41, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:44<01:23, 1.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:44<01:00, 1.80MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<01:03, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<01:10, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<00:54, 1.94MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:47<00:39, 2.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:48<00:56, 1.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:48<00:51, 1.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:48<00:38, 2.61MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:50<00:46, 2.10MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:50<00:55, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:50<00:44, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:51<00:31, 3.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:52<01:20, 1.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:52<01:06, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:52<00:48, 1.91MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:54<00:52, 1.70MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:54<00:58, 1.54MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:54<00:45, 1.97MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:54<00:32, 2.69MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:56<01:15, 1.13MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:56<01:02, 1.36MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:56<00:45, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:58<00:47, 1.71MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:58<00:51, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:58<00:40, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:59<00:28, 2.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:00<01:49, 705kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:00<01:25, 898kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [06:00<01:01, 1.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:02<00:56, 1.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:02<00:57, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:02<00:43, 1.64MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:02<00:30, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:04<00:49, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:04<00:42, 1.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:04<00:30, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:06<00:34, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:06<00:39, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:06<00:30, 2.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:06<00:21, 2.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:08<00:55, 1.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:08<00:45, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:08<00:32, 1.80MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:10<00:33, 1.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:10<00:37, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:10<00:28, 1.94MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:10<00:20, 2.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:12<00:29, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:12<00:26, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:12<00:19, 2.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:14<00:23, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:14<00:27, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:14<00:21, 2.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:14<00:15, 2.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:16<00:25, 1.74MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:16<00:22, 1.92MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:16<00:16, 2.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:18<00:19, 2.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:18<00:23, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:18<00:18, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:18<00:12, 2.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:20<00:32, 1.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:20<00:26, 1.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:20<00:18, 1.79MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:22<00:18, 1.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:22<00:20, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:22<00:15, 1.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:22<00:10, 2.67MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:24<00:23, 1.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:24<00:19, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:24<00:13, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:26<00:13, 1.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:26<00:15, 1.52MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:26<00:11, 1.95MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:26<00:07, 2.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:28<00:16, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:28<00:13, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:28<00:09, 1.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:30<00:08, 1.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:30<00:09, 1.53MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:30<00:06, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:30<00:04, 2.67MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:32<00:05, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:32<00:05, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:32<00:03, 2.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:34<00:02, 2.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:34<00:03, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:34<00:02, 2.13MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:34<00:01, 2.93MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:36<00:01, 1.44MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:36<00:01, 1.65MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:36<00:00, 2.20MB/s].vector_cache/glove.6B.zip: 862MB [06:36, 2.17MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<13:05:15,  8.49it/s]  0%|          | 735/400000 [00:00<9:08:56, 12.12it/s]  0%|          | 1496/400000 [00:00<6:23:47, 17.31it/s]  1%|          | 2265/400000 [00:00<4:28:23, 24.70it/s]  1%|          | 3011/400000 [00:00<3:07:47, 35.23it/s]  1%|          | 3720/400000 [00:00<2:11:29, 50.23it/s]  1%|          | 4496/400000 [00:00<1:32:07, 71.55it/s]  1%|▏         | 5286/400000 [00:00<1:04:36, 101.82it/s]  1%|▏         | 5987/400000 [00:00<45:29, 144.35it/s]    2%|▏         | 6793/400000 [00:01<32:01, 204.65it/s]  2%|▏         | 7498/400000 [00:01<22:41, 288.28it/s]  2%|▏         | 8177/400000 [00:01<16:10, 403.89it/s]  2%|▏         | 8929/400000 [00:01<11:33, 563.99it/s]  2%|▏         | 9716/400000 [00:01<08:19, 781.70it/s]  3%|▎         | 10520/400000 [00:01<06:03, 1071.86it/s]  3%|▎         | 11302/400000 [00:01<04:28, 1446.24it/s]  3%|▎         | 12112/400000 [00:01<03:22, 1919.13it/s]  3%|▎         | 12884/400000 [00:01<02:36, 2470.81it/s]  3%|▎         | 13650/400000 [00:01<02:08, 3007.93it/s]  4%|▎         | 14369/400000 [00:02<01:48, 3538.82it/s]  4%|▍         | 15053/400000 [00:02<01:33, 4130.52it/s]  4%|▍         | 15735/400000 [00:02<01:22, 4670.44it/s]  4%|▍         | 16446/400000 [00:02<01:13, 5204.87it/s]  4%|▍         | 17193/400000 [00:02<01:06, 5725.58it/s]  4%|▍         | 17916/400000 [00:02<01:02, 6105.79it/s]  5%|▍         | 18627/400000 [00:02<01:00, 6331.04it/s]  5%|▍         | 19378/400000 [00:02<00:57, 6643.28it/s]  5%|▌         | 20110/400000 [00:02<00:55, 6831.94it/s]  5%|▌         | 20853/400000 [00:03<00:54, 6999.40it/s]  5%|▌         | 21646/400000 [00:03<00:52, 7253.68it/s]  6%|▌         | 22394/400000 [00:03<00:51, 7301.49it/s]  6%|▌         | 23201/400000 [00:03<00:50, 7514.24it/s]  6%|▌         | 23982/400000 [00:03<00:49, 7600.16it/s]  6%|▌         | 24752/400000 [00:03<00:49, 7509.48it/s]  6%|▋         | 25510/400000 [00:03<00:50, 7422.17it/s]  7%|▋         | 26258/400000 [00:03<00:50, 7375.25it/s]  7%|▋         | 26999/400000 [00:03<00:51, 7253.05it/s]  7%|▋         | 27728/400000 [00:03<00:53, 7016.71it/s]  7%|▋         | 28434/400000 [00:04<00:54, 6768.53it/s]  7%|▋         | 29145/400000 [00:04<00:54, 6866.12it/s]  7%|▋         | 29930/400000 [00:04<00:51, 7133.19it/s]  8%|▊         | 30649/400000 [00:04<00:52, 7093.12it/s]  8%|▊         | 31362/400000 [00:04<00:52, 7083.07it/s]  8%|▊         | 32098/400000 [00:04<00:51, 7162.01it/s]  8%|▊         | 32817/400000 [00:04<00:51, 7166.82it/s]  8%|▊         | 33543/400000 [00:04<00:50, 7193.61it/s]  9%|▊         | 34264/400000 [00:04<00:50, 7192.18it/s]  9%|▉         | 35014/400000 [00:04<00:50, 7279.53it/s]  9%|▉         | 35787/400000 [00:05<00:49, 7408.95it/s]  9%|▉         | 36555/400000 [00:05<00:48, 7486.91it/s]  9%|▉         | 37325/400000 [00:05<00:48, 7547.32it/s] 10%|▉         | 38081/400000 [00:05<00:47, 7548.93it/s] 10%|▉         | 38837/400000 [00:05<00:48, 7406.33it/s] 10%|▉         | 39579/400000 [00:05<00:49, 7321.81it/s] 10%|█         | 40313/400000 [00:05<00:50, 7181.90it/s] 10%|█         | 41033/400000 [00:05<00:52, 6861.31it/s] 10%|█         | 41764/400000 [00:05<00:51, 6989.55it/s] 11%|█         | 42531/400000 [00:06<00:49, 7179.24it/s] 11%|█         | 43317/400000 [00:06<00:48, 7367.85it/s] 11%|█         | 44087/400000 [00:06<00:47, 7463.93it/s] 11%|█         | 44837/400000 [00:06<00:48, 7388.37it/s] 11%|█▏        | 45579/400000 [00:06<00:48, 7344.07it/s] 12%|█▏        | 46316/400000 [00:06<00:48, 7231.32it/s] 12%|█▏        | 47047/400000 [00:06<00:48, 7254.01it/s] 12%|█▏        | 47809/400000 [00:06<00:47, 7358.06it/s] 12%|█▏        | 48578/400000 [00:06<00:47, 7452.31it/s] 12%|█▏        | 49325/400000 [00:06<00:47, 7433.37it/s] 13%|█▎        | 50070/400000 [00:07<00:48, 7288.64it/s] 13%|█▎        | 50801/400000 [00:07<00:48, 7189.84it/s] 13%|█▎        | 51522/400000 [00:07<00:49, 7042.64it/s] 13%|█▎        | 52228/400000 [00:07<00:49, 7013.78it/s] 13%|█▎        | 52977/400000 [00:07<00:48, 7149.07it/s] 13%|█▎        | 53707/400000 [00:07<00:48, 7193.66it/s] 14%|█▎        | 54431/400000 [00:07<00:47, 7207.27it/s] 14%|█▍        | 55153/400000 [00:07<00:48, 7158.88it/s] 14%|█▍        | 55870/400000 [00:07<00:48, 7073.89it/s] 14%|█▍        | 56579/400000 [00:07<00:48, 7048.30it/s] 14%|█▍        | 57285/400000 [00:08<00:48, 7032.96it/s] 14%|█▍        | 57989/400000 [00:08<00:48, 6997.90it/s] 15%|█▍        | 58690/400000 [00:08<00:50, 6744.84it/s] 15%|█▍        | 59367/400000 [00:08<00:50, 6683.08it/s] 15%|█▌        | 60037/400000 [00:08<00:51, 6634.68it/s] 15%|█▌        | 60720/400000 [00:08<00:50, 6691.68it/s] 15%|█▌        | 61484/400000 [00:08<00:48, 6948.87it/s] 16%|█▌        | 62257/400000 [00:08<00:47, 7163.89it/s] 16%|█▌        | 62978/400000 [00:08<00:47, 7052.44it/s] 16%|█▌        | 63687/400000 [00:08<00:49, 6851.16it/s] 16%|█▌        | 64376/400000 [00:09<00:49, 6801.41it/s] 16%|█▋        | 65102/400000 [00:09<00:48, 6930.79it/s] 16%|█▋        | 65841/400000 [00:09<00:47, 7061.04it/s] 17%|█▋        | 66628/400000 [00:09<00:45, 7283.56it/s] 17%|█▋        | 67459/400000 [00:09<00:43, 7563.06it/s] 17%|█▋        | 68221/400000 [00:09<00:43, 7564.85it/s] 17%|█▋        | 69046/400000 [00:09<00:42, 7755.26it/s] 17%|█▋        | 69836/400000 [00:09<00:42, 7796.90it/s] 18%|█▊        | 70619/400000 [00:09<00:44, 7389.95it/s] 18%|█▊        | 71365/400000 [00:10<00:46, 7043.57it/s] 18%|█▊        | 72078/400000 [00:10<00:48, 6783.64it/s] 18%|█▊        | 72778/400000 [00:10<00:47, 6846.71it/s] 18%|█▊        | 73511/400000 [00:10<00:46, 6976.04it/s] 19%|█▊        | 74281/400000 [00:10<00:45, 7177.46it/s] 19%|█▉        | 75094/400000 [00:10<00:43, 7438.39it/s] 19%|█▉        | 75868/400000 [00:10<00:43, 7526.09it/s] 19%|█▉        | 76630/400000 [00:10<00:42, 7551.79it/s] 19%|█▉        | 77406/400000 [00:10<00:42, 7612.35it/s] 20%|█▉        | 78199/400000 [00:10<00:41, 7703.49it/s] 20%|█▉        | 78988/400000 [00:11<00:41, 7756.06it/s] 20%|█▉        | 79765/400000 [00:11<00:41, 7698.93it/s] 20%|██        | 80536/400000 [00:11<00:43, 7393.57it/s] 20%|██        | 81279/400000 [00:11<00:43, 7367.15it/s] 21%|██        | 82058/400000 [00:11<00:42, 7487.59it/s] 21%|██        | 82817/400000 [00:11<00:42, 7515.40it/s] 21%|██        | 83586/400000 [00:11<00:41, 7564.57it/s] 21%|██        | 84408/400000 [00:11<00:40, 7748.75it/s] 21%|██▏       | 85185/400000 [00:11<00:42, 7343.67it/s] 21%|██▏       | 85962/400000 [00:11<00:42, 7464.04it/s] 22%|██▏       | 86741/400000 [00:12<00:41, 7557.06it/s] 22%|██▏       | 87504/400000 [00:12<00:41, 7578.13it/s] 22%|██▏       | 88295/400000 [00:12<00:40, 7671.89it/s] 22%|██▏       | 89065/400000 [00:12<00:41, 7558.31it/s] 22%|██▏       | 89823/400000 [00:12<00:41, 7495.71it/s] 23%|██▎       | 90595/400000 [00:12<00:40, 7560.41it/s] 23%|██▎       | 91353/400000 [00:12<00:40, 7559.78it/s] 23%|██▎       | 92110/400000 [00:12<00:40, 7550.12it/s] 23%|██▎       | 92892/400000 [00:12<00:40, 7626.12it/s] 23%|██▎       | 93702/400000 [00:12<00:39, 7759.87it/s] 24%|██▎       | 94492/400000 [00:13<00:39, 7801.04it/s] 24%|██▍       | 95273/400000 [00:13<00:40, 7615.23it/s] 24%|██▍       | 96037/400000 [00:13<00:40, 7552.26it/s] 24%|██▍       | 96801/400000 [00:13<00:40, 7577.13it/s] 24%|██▍       | 97594/400000 [00:13<00:39, 7677.06it/s] 25%|██▍       | 98379/400000 [00:13<00:39, 7727.22it/s] 25%|██▍       | 99209/400000 [00:13<00:38, 7890.06it/s] 25%|██▌       | 100024/400000 [00:13<00:37, 7964.50it/s] 25%|██▌       | 100822/400000 [00:13<00:37, 7881.89it/s] 25%|██▌       | 101646/400000 [00:13<00:37, 7984.98it/s] 26%|██▌       | 102446/400000 [00:14<00:37, 7926.82it/s] 26%|██▌       | 103240/400000 [00:14<00:38, 7631.92it/s] 26%|██▌       | 104007/400000 [00:14<00:39, 7553.28it/s] 26%|██▌       | 104765/400000 [00:14<00:40, 7269.59it/s] 26%|██▋       | 105496/400000 [00:14<00:41, 7112.07it/s] 27%|██▋       | 106211/400000 [00:14<00:41, 7064.76it/s] 27%|██▋       | 106996/400000 [00:14<00:40, 7282.34it/s] 27%|██▋       | 107728/400000 [00:14<00:41, 6996.87it/s] 27%|██▋       | 108504/400000 [00:14<00:40, 7208.28it/s] 27%|██▋       | 109339/400000 [00:15<00:38, 7514.91it/s] 28%|██▊       | 110098/400000 [00:15<00:38, 7516.41it/s] 28%|██▊       | 110855/400000 [00:15<00:38, 7431.22it/s] 28%|██▊       | 111602/400000 [00:15<00:39, 7342.38it/s] 28%|██▊       | 112396/400000 [00:15<00:38, 7511.56it/s] 28%|██▊       | 113180/400000 [00:15<00:37, 7605.26it/s] 28%|██▊       | 113994/400000 [00:15<00:36, 7757.39it/s] 29%|██▊       | 114807/400000 [00:15<00:36, 7864.06it/s] 29%|██▉       | 115596/400000 [00:15<00:36, 7774.47it/s] 29%|██▉       | 116390/400000 [00:15<00:36, 7822.05it/s] 29%|██▉       | 117174/400000 [00:16<00:36, 7759.22it/s] 29%|██▉       | 117951/400000 [00:16<00:36, 7749.99it/s] 30%|██▉       | 118727/400000 [00:16<00:36, 7667.05it/s] 30%|██▉       | 119495/400000 [00:16<00:37, 7538.56it/s] 30%|███       | 120253/400000 [00:16<00:37, 7549.89it/s] 30%|███       | 121078/400000 [00:16<00:36, 7747.04it/s] 30%|███       | 121877/400000 [00:16<00:35, 7817.69it/s] 31%|███       | 122686/400000 [00:16<00:35, 7894.86it/s] 31%|███       | 123477/400000 [00:16<00:35, 7810.45it/s] 31%|███       | 124269/400000 [00:16<00:35, 7840.73it/s] 31%|███▏      | 125104/400000 [00:17<00:34, 7986.70it/s] 31%|███▏      | 125904/400000 [00:17<00:34, 7929.25it/s] 32%|███▏      | 126698/400000 [00:17<00:34, 7813.46it/s] 32%|███▏      | 127481/400000 [00:17<00:35, 7726.43it/s] 32%|███▏      | 128255/400000 [00:17<00:36, 7470.83it/s] 32%|███▏      | 129067/400000 [00:17<00:35, 7653.01it/s] 32%|███▏      | 129907/400000 [00:17<00:34, 7861.39it/s] 33%|███▎      | 130702/400000 [00:17<00:34, 7887.48it/s] 33%|███▎      | 131494/400000 [00:17<00:35, 7656.26it/s] 33%|███▎      | 132263/400000 [00:18<00:36, 7345.31it/s] 33%|███▎      | 133022/400000 [00:18<00:36, 7415.00it/s] 33%|███▎      | 133810/400000 [00:18<00:35, 7546.79it/s] 34%|███▎      | 134604/400000 [00:18<00:34, 7659.78it/s] 34%|███▍      | 135383/400000 [00:18<00:34, 7697.60it/s] 34%|███▍      | 136172/400000 [00:18<00:34, 7751.26it/s] 34%|███▍      | 136951/400000 [00:18<00:33, 7760.90it/s] 34%|███▍      | 137760/400000 [00:18<00:33, 7855.07it/s] 35%|███▍      | 138547/400000 [00:18<00:33, 7825.08it/s] 35%|███▍      | 139342/400000 [00:18<00:33, 7860.97it/s] 35%|███▌      | 140159/400000 [00:19<00:32, 7951.16it/s] 35%|███▌      | 140955/400000 [00:19<00:32, 7947.32it/s] 35%|███▌      | 141751/400000 [00:19<00:35, 7366.95it/s] 36%|███▌      | 142497/400000 [00:19<00:36, 7114.05it/s] 36%|███▌      | 143217/400000 [00:19<00:36, 7096.35it/s] 36%|███▌      | 143933/400000 [00:19<00:36, 7099.82it/s] 36%|███▌      | 144647/400000 [00:19<00:35, 7103.42it/s] 36%|███▋      | 145379/400000 [00:19<00:35, 7166.67it/s] 37%|███▋      | 146098/400000 [00:19<00:36, 6918.16it/s] 37%|███▋      | 146794/400000 [00:20<00:37, 6821.36it/s] 37%|███▋      | 147479/400000 [00:20<00:37, 6778.63it/s] 37%|███▋      | 148159/400000 [00:20<00:37, 6711.93it/s] 37%|███▋      | 148907/400000 [00:20<00:36, 6925.24it/s] 37%|███▋      | 149652/400000 [00:20<00:35, 7072.50it/s] 38%|███▊      | 150362/400000 [00:20<00:35, 7027.89it/s] 38%|███▊      | 151107/400000 [00:20<00:34, 7149.09it/s] 38%|███▊      | 151842/400000 [00:20<00:34, 7207.79it/s] 38%|███▊      | 152602/400000 [00:20<00:33, 7319.06it/s] 38%|███▊      | 153336/400000 [00:20<00:34, 7217.59it/s] 39%|███▊      | 154060/400000 [00:21<00:35, 6989.34it/s] 39%|███▊      | 154762/400000 [00:21<00:35, 6892.94it/s] 39%|███▉      | 155454/400000 [00:21<00:35, 6847.18it/s] 39%|███▉      | 156141/400000 [00:21<00:35, 6837.26it/s] 39%|███▉      | 156857/400000 [00:21<00:35, 6929.42it/s] 39%|███▉      | 157596/400000 [00:21<00:34, 7059.01it/s] 40%|███▉      | 158304/400000 [00:21<00:36, 6699.15it/s] 40%|███▉      | 159129/400000 [00:21<00:33, 7098.75it/s] 40%|███▉      | 159897/400000 [00:21<00:33, 7263.02it/s] 40%|████      | 160633/400000 [00:21<00:32, 7291.43it/s] 40%|████      | 161436/400000 [00:22<00:31, 7496.95it/s] 41%|████      | 162233/400000 [00:22<00:31, 7631.28it/s] 41%|████      | 163001/400000 [00:22<00:31, 7610.02it/s] 41%|████      | 163826/400000 [00:22<00:30, 7789.86it/s] 41%|████      | 164640/400000 [00:22<00:29, 7890.91it/s] 41%|████▏     | 165432/400000 [00:22<00:31, 7403.77it/s] 42%|████▏     | 166181/400000 [00:22<00:33, 6901.00it/s] 42%|████▏     | 166884/400000 [00:22<00:34, 6778.03it/s] 42%|████▏     | 167571/400000 [00:22<00:34, 6794.97it/s] 42%|████▏     | 168270/400000 [00:23<00:33, 6851.42it/s] 42%|████▏     | 169024/400000 [00:23<00:32, 7042.10it/s] 42%|████▏     | 169780/400000 [00:23<00:32, 7187.28it/s] 43%|████▎     | 170554/400000 [00:23<00:31, 7342.85it/s] 43%|████▎     | 171327/400000 [00:23<00:30, 7454.03it/s] 43%|████▎     | 172076/400000 [00:23<00:30, 7417.69it/s] 43%|████▎     | 172820/400000 [00:23<00:30, 7413.45it/s] 43%|████▎     | 173599/400000 [00:23<00:30, 7520.77it/s] 44%|████▎     | 174358/400000 [00:23<00:29, 7540.84it/s] 44%|████▍     | 175124/400000 [00:23<00:29, 7575.64it/s] 44%|████▍     | 175883/400000 [00:24<00:29, 7513.44it/s] 44%|████▍     | 176635/400000 [00:24<00:31, 6995.64it/s] 44%|████▍     | 177343/400000 [00:24<00:33, 6675.02it/s] 45%|████▍     | 178019/400000 [00:24<00:34, 6453.10it/s] 45%|████▍     | 178672/400000 [00:24<00:34, 6454.97it/s] 45%|████▍     | 179395/400000 [00:24<00:33, 6668.41it/s] 45%|████▌     | 180159/400000 [00:24<00:31, 6932.88it/s] 45%|████▌     | 180971/400000 [00:24<00:30, 7249.46it/s] 45%|████▌     | 181752/400000 [00:24<00:29, 7406.84it/s] 46%|████▌     | 182500/400000 [00:24<00:30, 7047.06it/s] 46%|████▌     | 183214/400000 [00:25<00:32, 6744.29it/s] 46%|████▌     | 183898/400000 [00:25<00:32, 6667.02it/s] 46%|████▌     | 184571/400000 [00:25<00:32, 6625.71it/s] 46%|████▋     | 185238/400000 [00:25<00:32, 6632.79it/s] 46%|████▋     | 185911/400000 [00:25<00:32, 6659.08it/s] 47%|████▋     | 186622/400000 [00:25<00:31, 6788.22it/s] 47%|████▋     | 187370/400000 [00:25<00:30, 6979.59it/s] 47%|████▋     | 188110/400000 [00:25<00:29, 7098.44it/s] 47%|████▋     | 188846/400000 [00:25<00:29, 7174.21it/s] 47%|████▋     | 189657/400000 [00:26<00:28, 7429.30it/s] 48%|████▊     | 190404/400000 [00:26<00:29, 7156.26it/s] 48%|████▊     | 191125/400000 [00:26<00:30, 6817.58it/s] 48%|████▊     | 191814/400000 [00:26<00:31, 6604.59it/s] 48%|████▊     | 192481/400000 [00:26<00:32, 6424.78it/s] 48%|████▊     | 193129/400000 [00:26<00:33, 6227.50it/s] 48%|████▊     | 193819/400000 [00:26<00:32, 6414.41it/s] 49%|████▊     | 194541/400000 [00:26<00:30, 6636.40it/s] 49%|████▉     | 195262/400000 [00:26<00:30, 6798.25it/s] 49%|████▉     | 196004/400000 [00:26<00:29, 6971.25it/s] 49%|████▉     | 196726/400000 [00:27<00:28, 7043.51it/s] 49%|████▉     | 197445/400000 [00:27<00:28, 7084.54it/s] 50%|████▉     | 198179/400000 [00:27<00:28, 7159.06it/s] 50%|████▉     | 198897/400000 [00:27<00:28, 7142.92it/s] 50%|████▉     | 199665/400000 [00:27<00:27, 7295.36it/s] 50%|█████     | 200397/400000 [00:27<00:28, 7082.02it/s] 50%|█████     | 201108/400000 [00:27<00:29, 6821.47it/s] 50%|█████     | 201794/400000 [00:27<00:30, 6588.65it/s] 51%|█████     | 202480/400000 [00:27<00:29, 6665.94it/s] 51%|█████     | 203229/400000 [00:28<00:28, 6891.86it/s] 51%|█████     | 203966/400000 [00:28<00:27, 7027.69it/s] 51%|█████     | 204695/400000 [00:28<00:27, 7101.81it/s] 51%|█████▏    | 205408/400000 [00:28<00:27, 6956.05it/s] 52%|█████▏    | 206107/400000 [00:28<00:28, 6815.32it/s] 52%|█████▏    | 206791/400000 [00:28<00:28, 6767.70it/s] 52%|█████▏    | 207470/400000 [00:28<00:28, 6722.98it/s] 52%|█████▏    | 208232/400000 [00:28<00:27, 6968.23it/s] 52%|█████▏    | 209033/400000 [00:28<00:26, 7250.51it/s] 52%|█████▏    | 209816/400000 [00:28<00:25, 7413.61it/s] 53%|█████▎    | 210598/400000 [00:29<00:25, 7529.76it/s] 53%|█████▎    | 211355/400000 [00:29<00:25, 7413.99it/s] 53%|█████▎    | 212100/400000 [00:29<00:25, 7306.99it/s] 53%|█████▎    | 212834/400000 [00:29<00:25, 7241.25it/s] 53%|█████▎    | 213560/400000 [00:29<00:25, 7214.92it/s] 54%|█████▎    | 214357/400000 [00:29<00:25, 7425.19it/s] 54%|█████▍    | 215121/400000 [00:29<00:24, 7487.25it/s] 54%|█████▍    | 215918/400000 [00:29<00:24, 7625.47it/s] 54%|█████▍    | 216686/400000 [00:29<00:23, 7640.39it/s] 54%|█████▍    | 217462/400000 [00:29<00:23, 7675.04it/s] 55%|█████▍    | 218257/400000 [00:30<00:23, 7754.75it/s] 55%|█████▍    | 219034/400000 [00:30<00:23, 7741.42it/s] 55%|█████▍    | 219809/400000 [00:30<00:23, 7562.10it/s] 55%|█████▌    | 220567/400000 [00:30<00:24, 7374.40it/s] 55%|█████▌    | 221307/400000 [00:30<00:24, 7232.78it/s] 56%|█████▌    | 222033/400000 [00:30<00:25, 7082.71it/s] 56%|█████▌    | 222744/400000 [00:30<00:25, 6883.46it/s] 56%|█████▌    | 223436/400000 [00:30<00:26, 6761.69it/s] 56%|█████▌    | 224133/400000 [00:30<00:25, 6822.52it/s] 56%|█████▌    | 224878/400000 [00:31<00:25, 6998.65it/s] 56%|█████▋    | 225586/400000 [00:31<00:24, 7022.05it/s] 57%|█████▋    | 226300/400000 [00:31<00:24, 7055.35it/s] 57%|█████▋    | 227023/400000 [00:31<00:24, 7106.11it/s] 57%|█████▋    | 227772/400000 [00:31<00:23, 7215.72it/s] 57%|█████▋    | 228519/400000 [00:31<00:23, 7287.32it/s] 57%|█████▋    | 229269/400000 [00:31<00:23, 7349.13it/s] 58%|█████▊    | 230008/400000 [00:31<00:23, 7359.78it/s] 58%|█████▊    | 230745/400000 [00:31<00:23, 7319.95it/s] 58%|█████▊    | 231478/400000 [00:31<00:23, 7148.97it/s] 58%|█████▊    | 232196/400000 [00:32<00:23, 7157.45it/s] 58%|█████▊    | 232930/400000 [00:32<00:23, 7211.21it/s] 58%|█████▊    | 233656/400000 [00:32<00:23, 7223.74it/s] 59%|█████▊    | 234388/400000 [00:32<00:22, 7252.13it/s] 59%|█████▉    | 235143/400000 [00:32<00:22, 7338.17it/s] 59%|█████▉    | 235878/400000 [00:32<00:23, 7122.05it/s] 59%|█████▉    | 236592/400000 [00:32<00:23, 6974.86it/s] 59%|█████▉    | 237292/400000 [00:32<00:23, 6891.38it/s] 60%|█████▉    | 238063/400000 [00:32<00:22, 7117.80it/s] 60%|█████▉    | 238859/400000 [00:32<00:21, 7350.52it/s] 60%|█████▉    | 239666/400000 [00:33<00:21, 7551.72it/s] 60%|██████    | 240440/400000 [00:33<00:20, 7605.22it/s] 60%|██████    | 241204/400000 [00:33<00:21, 7352.54it/s] 60%|██████    | 241944/400000 [00:33<00:21, 7335.83it/s] 61%|██████    | 242689/400000 [00:33<00:21, 7368.56it/s] 61%|██████    | 243428/400000 [00:33<00:21, 7342.06it/s] 61%|██████    | 244164/400000 [00:33<00:21, 7198.19it/s] 61%|██████    | 244992/400000 [00:33<00:20, 7490.59it/s] 61%|██████▏   | 245750/400000 [00:33<00:20, 7516.14it/s] 62%|██████▏   | 246523/400000 [00:33<00:20, 7578.48it/s] 62%|██████▏   | 247315/400000 [00:34<00:19, 7675.13it/s] 62%|██████▏   | 248100/400000 [00:34<00:19, 7725.25it/s] 62%|██████▏   | 248874/400000 [00:34<00:19, 7575.41it/s] 62%|██████▏   | 249649/400000 [00:34<00:19, 7626.78it/s] 63%|██████▎   | 250432/400000 [00:34<00:19, 7686.30it/s] 63%|██████▎   | 251245/400000 [00:34<00:19, 7813.23it/s] 63%|██████▎   | 252042/400000 [00:34<00:18, 7858.18it/s] 63%|██████▎   | 252829/400000 [00:34<00:18, 7782.49it/s] 63%|██████▎   | 253609/400000 [00:34<00:19, 7683.26it/s] 64%|██████▎   | 254379/400000 [00:34<00:18, 7681.20it/s] 64%|██████▍   | 255148/400000 [00:35<00:18, 7640.16it/s] 64%|██████▍   | 255913/400000 [00:35<00:18, 7604.43it/s] 64%|██████▍   | 256674/400000 [00:35<00:18, 7588.02it/s] 64%|██████▍   | 257461/400000 [00:35<00:18, 7668.70it/s] 65%|██████▍   | 258282/400000 [00:35<00:18, 7820.99it/s] 65%|██████▍   | 259106/400000 [00:35<00:17, 7941.40it/s] 65%|██████▍   | 259902/400000 [00:35<00:19, 7193.36it/s] 65%|██████▌   | 260636/400000 [00:35<00:20, 6868.36it/s] 65%|██████▌   | 261336/400000 [00:35<00:21, 6540.87it/s] 66%|██████▌   | 262012/400000 [00:36<00:20, 6604.52it/s] 66%|██████▌   | 262685/400000 [00:36<00:20, 6639.73it/s] 66%|██████▌   | 263369/400000 [00:36<00:20, 6696.42it/s] 66%|██████▌   | 264159/400000 [00:36<00:19, 7017.10it/s] 66%|██████▌   | 264974/400000 [00:36<00:18, 7320.61it/s] 66%|██████▋   | 265715/400000 [00:36<00:18, 7335.96it/s] 67%|██████▋   | 266521/400000 [00:36<00:17, 7536.73it/s] 67%|██████▋   | 267281/400000 [00:36<00:17, 7495.89it/s] 67%|██████▋   | 268035/400000 [00:36<00:17, 7424.99it/s] 67%|██████▋   | 268781/400000 [00:36<00:18, 7237.47it/s] 67%|██████▋   | 269524/400000 [00:37<00:17, 7292.19it/s] 68%|██████▊   | 270256/400000 [00:37<00:17, 7249.56it/s] 68%|██████▊   | 270987/400000 [00:37<00:17, 7266.93it/s] 68%|██████▊   | 271733/400000 [00:37<00:17, 7323.69it/s] 68%|██████▊   | 272494/400000 [00:37<00:17, 7405.82it/s] 68%|██████▊   | 273260/400000 [00:37<00:16, 7477.65it/s] 69%|██████▊   | 274030/400000 [00:37<00:16, 7541.60it/s] 69%|██████▊   | 274799/400000 [00:37<00:16, 7583.25it/s] 69%|██████▉   | 275576/400000 [00:37<00:16, 7636.76it/s] 69%|██████▉   | 276341/400000 [00:37<00:16, 7597.08it/s] 69%|██████▉   | 277168/400000 [00:38<00:15, 7785.39it/s] 69%|██████▉   | 277948/400000 [00:38<00:16, 7329.63it/s] 70%|██████▉   | 278688/400000 [00:38<00:17, 6801.28it/s] 70%|██████▉   | 279381/400000 [00:38<00:18, 6514.30it/s] 70%|███████   | 280044/400000 [00:38<00:18, 6452.90it/s] 70%|███████   | 280728/400000 [00:38<00:18, 6562.95it/s] 70%|███████   | 281395/400000 [00:38<00:17, 6594.47it/s] 71%|███████   | 282113/400000 [00:38<00:17, 6758.41it/s] 71%|███████   | 282864/400000 [00:38<00:16, 6967.11it/s] 71%|███████   | 283651/400000 [00:39<00:16, 7214.39it/s] 71%|███████   | 284398/400000 [00:39<00:15, 7287.24it/s] 71%|███████▏  | 285147/400000 [00:39<00:15, 7345.98it/s] 71%|███████▏  | 285932/400000 [00:39<00:15, 7489.09it/s] 72%|███████▏  | 286733/400000 [00:39<00:14, 7637.85it/s] 72%|███████▏  | 287551/400000 [00:39<00:14, 7792.65it/s] 72%|███████▏  | 288347/400000 [00:39<00:14, 7841.99it/s] 72%|███████▏  | 289134/400000 [00:39<00:14, 7417.19it/s] 72%|███████▏  | 289882/400000 [00:39<00:15, 6967.59it/s] 73%|███████▎  | 290600/400000 [00:39<00:15, 7029.41it/s] 73%|███████▎  | 291395/400000 [00:40<00:14, 7280.24it/s] 73%|███████▎  | 292173/400000 [00:40<00:14, 7423.14it/s] 73%|███████▎  | 292975/400000 [00:40<00:14, 7591.85it/s] 73%|███████▎  | 293740/400000 [00:40<00:14, 7282.31it/s] 74%|███████▎  | 294475/400000 [00:40<00:14, 7044.77it/s] 74%|███████▍  | 295186/400000 [00:40<00:15, 6879.86it/s] 74%|███████▍  | 295921/400000 [00:40<00:14, 7012.90it/s] 74%|███████▍  | 296627/400000 [00:40<00:14, 6997.20it/s] 74%|███████▍  | 297330/400000 [00:40<00:15, 6790.46it/s] 75%|███████▍  | 298013/400000 [00:41<00:15, 6665.62it/s] 75%|███████▍  | 298722/400000 [00:41<00:14, 6785.83it/s] 75%|███████▍  | 299404/400000 [00:41<00:15, 6668.72it/s] 75%|███████▌  | 300074/400000 [00:41<00:15, 6579.29it/s] 75%|███████▌  | 300738/400000 [00:41<00:15, 6594.56it/s] 75%|███████▌  | 301399/400000 [00:41<00:14, 6588.84it/s] 76%|███████▌  | 302132/400000 [00:41<00:14, 6792.39it/s] 76%|███████▌  | 302903/400000 [00:41<00:13, 7042.45it/s] 76%|███████▌  | 303621/400000 [00:41<00:13, 7081.92it/s] 76%|███████▌  | 304332/400000 [00:41<00:13, 7081.74it/s] 76%|███████▋  | 305134/400000 [00:42<00:12, 7337.50it/s] 76%|███████▋  | 305883/400000 [00:42<00:12, 7380.96it/s] 77%|███████▋  | 306640/400000 [00:42<00:12, 7435.96it/s] 77%|███████▋  | 307453/400000 [00:42<00:12, 7630.29it/s] 77%|███████▋  | 308219/400000 [00:42<00:12, 7566.20it/s] 77%|███████▋  | 309003/400000 [00:42<00:11, 7645.18it/s] 77%|███████▋  | 309801/400000 [00:42<00:11, 7741.16it/s] 78%|███████▊  | 310603/400000 [00:42<00:11, 7820.69it/s] 78%|███████▊  | 311421/400000 [00:42<00:11, 7924.87it/s] 78%|███████▊  | 312215/400000 [00:42<00:11, 7813.05it/s] 78%|███████▊  | 313008/400000 [00:43<00:11, 7837.62it/s] 78%|███████▊  | 313815/400000 [00:43<00:10, 7904.83it/s] 79%|███████▊  | 314607/400000 [00:43<00:10, 7865.67it/s] 79%|███████▉  | 315395/400000 [00:43<00:10, 7830.80it/s] 79%|███████▉  | 316179/400000 [00:43<00:10, 7692.87it/s] 79%|███████▉  | 316950/400000 [00:43<00:11, 7277.85it/s] 79%|███████▉  | 317684/400000 [00:43<00:11, 6995.34it/s] 80%|███████▉  | 318433/400000 [00:43<00:11, 7126.72it/s] 80%|███████▉  | 319161/400000 [00:43<00:11, 7167.27it/s] 80%|███████▉  | 319882/400000 [00:44<00:11, 6909.72it/s] 80%|████████  | 320621/400000 [00:44<00:11, 7045.95it/s] 80%|████████  | 321361/400000 [00:44<00:11, 7147.41it/s] 81%|████████  | 322112/400000 [00:44<00:10, 7250.40it/s] 81%|████████  | 322880/400000 [00:44<00:10, 7371.71it/s] 81%|████████  | 323620/400000 [00:44<00:10, 7378.10it/s] 81%|████████  | 324402/400000 [00:44<00:10, 7503.94it/s] 81%|████████▏ | 325194/400000 [00:44<00:09, 7622.43it/s] 81%|████████▏ | 325988/400000 [00:44<00:09, 7708.00it/s] 82%|████████▏ | 326761/400000 [00:44<00:09, 7712.79it/s] 82%|████████▏ | 327534/400000 [00:45<00:09, 7698.04it/s] 82%|████████▏ | 328311/400000 [00:45<00:09, 7717.89it/s] 82%|████████▏ | 329121/400000 [00:45<00:09, 7827.40it/s] 82%|████████▏ | 329905/400000 [00:45<00:09, 7645.15it/s] 83%|████████▎ | 330671/400000 [00:45<00:09, 7624.79it/s] 83%|████████▎ | 331457/400000 [00:45<00:08, 7692.43it/s] 83%|████████▎ | 332228/400000 [00:45<00:08, 7572.38it/s] 83%|████████▎ | 333032/400000 [00:45<00:08, 7705.61it/s] 83%|████████▎ | 333804/400000 [00:45<00:08, 7691.28it/s] 84%|████████▎ | 334603/400000 [00:45<00:08, 7776.93it/s] 84%|████████▍ | 335382/400000 [00:46<00:08, 7610.89it/s] 84%|████████▍ | 336145/400000 [00:46<00:08, 7297.74it/s] 84%|████████▍ | 336879/400000 [00:46<00:08, 7034.98it/s] 84%|████████▍ | 337588/400000 [00:46<00:09, 6926.28it/s] 85%|████████▍ | 338321/400000 [00:46<00:08, 7041.17it/s] 85%|████████▍ | 339056/400000 [00:46<00:08, 7130.47it/s] 85%|████████▍ | 339772/400000 [00:46<00:08, 7071.31it/s] 85%|████████▌ | 340566/400000 [00:46<00:08, 7308.28it/s] 85%|████████▌ | 341338/400000 [00:46<00:07, 7425.56it/s] 86%|████████▌ | 342084/400000 [00:46<00:08, 7135.64it/s] 86%|████████▌ | 342802/400000 [00:47<00:08, 6939.88it/s] 86%|████████▌ | 343501/400000 [00:47<00:08, 6854.44it/s] 86%|████████▌ | 344190/400000 [00:47<00:08, 6812.90it/s] 86%|████████▌ | 344920/400000 [00:47<00:07, 6950.34it/s] 86%|████████▋ | 345707/400000 [00:47<00:07, 7201.00it/s] 87%|████████▋ | 346445/400000 [00:47<00:07, 7252.10it/s] 87%|████████▋ | 347219/400000 [00:47<00:07, 7389.72it/s] 87%|████████▋ | 347987/400000 [00:47<00:06, 7471.65it/s] 87%|████████▋ | 348750/400000 [00:47<00:06, 7517.93it/s] 87%|████████▋ | 349507/400000 [00:48<00:06, 7530.60it/s] 88%|████████▊ | 350284/400000 [00:48<00:06, 7598.94it/s] 88%|████████▊ | 351045/400000 [00:48<00:06, 7462.02it/s] 88%|████████▊ | 351845/400000 [00:48<00:06, 7615.40it/s] 88%|████████▊ | 352629/400000 [00:48<00:06, 7679.26it/s] 88%|████████▊ | 353399/400000 [00:48<00:06, 7683.56it/s] 89%|████████▊ | 354178/400000 [00:48<00:05, 7715.17it/s] 89%|████████▊ | 354951/400000 [00:48<00:05, 7604.99it/s] 89%|████████▉ | 355713/400000 [00:48<00:06, 7274.98it/s] 89%|████████▉ | 356445/400000 [00:48<00:06, 7013.79it/s] 89%|████████▉ | 357151/400000 [00:49<00:06, 6877.65it/s] 89%|████████▉ | 357843/400000 [00:49<00:06, 6781.14it/s] 90%|████████▉ | 358536/400000 [00:49<00:06, 6823.83it/s] 90%|████████▉ | 359221/400000 [00:49<00:06, 6772.95it/s] 90%|████████▉ | 359900/400000 [00:49<00:05, 6683.64it/s] 90%|█████████ | 360570/400000 [00:49<00:05, 6607.53it/s] 90%|█████████ | 361244/400000 [00:49<00:05, 6644.87it/s] 91%|█████████ | 362034/400000 [00:49<00:05, 6976.37it/s] 91%|█████████ | 362804/400000 [00:49<00:05, 7178.26it/s] 91%|█████████ | 363529/400000 [00:49<00:05, 7199.24it/s] 91%|█████████ | 364339/400000 [00:50<00:04, 7446.27it/s] 91%|█████████▏| 365115/400000 [00:50<00:04, 7534.59it/s] 91%|█████████▏| 365894/400000 [00:50<00:04, 7607.59it/s] 92%|█████████▏| 366729/400000 [00:50<00:04, 7815.39it/s] 92%|█████████▏| 367530/400000 [00:50<00:04, 7870.40it/s] 92%|█████████▏| 368320/400000 [00:50<00:04, 7878.95it/s] 92%|█████████▏| 369110/400000 [00:50<00:03, 7729.91it/s] 92%|█████████▏| 369885/400000 [00:50<00:03, 7673.94it/s] 93%|█████████▎| 370673/400000 [00:50<00:03, 7734.23it/s] 93%|█████████▎| 371448/400000 [00:50<00:03, 7603.44it/s] 93%|█████████▎| 372210/400000 [00:51<00:03, 7574.74it/s] 93%|█████████▎| 372969/400000 [00:51<00:03, 7559.07it/s] 93%|█████████▎| 373738/400000 [00:51<00:03, 7597.47it/s] 94%|█████████▎| 374499/400000 [00:51<00:03, 7594.22it/s] 94%|█████████▍| 375311/400000 [00:51<00:03, 7742.06it/s] 94%|█████████▍| 376087/400000 [00:51<00:03, 7363.77it/s] 94%|█████████▍| 376828/400000 [00:51<00:03, 7068.88it/s] 94%|█████████▍| 377541/400000 [00:51<00:03, 7075.43it/s] 95%|█████████▍| 378269/400000 [00:51<00:03, 7127.73it/s] 95%|█████████▍| 379011/400000 [00:52<00:02, 7208.88it/s] 95%|█████████▍| 379755/400000 [00:52<00:02, 7275.37it/s] 95%|█████████▌| 380485/400000 [00:52<00:02, 7140.10it/s] 95%|█████████▌| 381222/400000 [00:52<00:02, 7207.13it/s] 95%|█████████▌| 381984/400000 [00:52<00:02, 7325.22it/s] 96%|█████████▌| 382752/400000 [00:52<00:02, 7427.90it/s] 96%|█████████▌| 383540/400000 [00:52<00:02, 7555.74it/s] 96%|█████████▌| 384315/400000 [00:52<00:02, 7612.65it/s] 96%|█████████▋| 385078/400000 [00:52<00:01, 7514.37it/s] 96%|█████████▋| 385831/400000 [00:52<00:01, 7499.30it/s] 97%|█████████▋| 386582/400000 [00:53<00:01, 7498.14it/s] 97%|█████████▋| 387333/400000 [00:53<00:01, 7495.32it/s] 97%|█████████▋| 388119/400000 [00:53<00:01, 7600.17it/s] 97%|█████████▋| 388880/400000 [00:53<00:01, 7600.97it/s] 97%|█████████▋| 389641/400000 [00:53<00:01, 7390.41it/s] 98%|█████████▊| 390382/400000 [00:53<00:01, 7199.54it/s] 98%|█████████▊| 391128/400000 [00:53<00:01, 7270.05it/s] 98%|█████████▊| 391892/400000 [00:53<00:01, 7376.06it/s] 98%|█████████▊| 392632/400000 [00:53<00:00, 7371.67it/s] 98%|█████████▊| 393371/400000 [00:53<00:00, 7034.20it/s] 99%|█████████▊| 394079/400000 [00:54<00:00, 6889.69it/s] 99%|█████████▊| 394772/400000 [00:54<00:00, 6709.65it/s] 99%|█████████▉| 395447/400000 [00:54<00:00, 6681.48it/s] 99%|█████████▉| 396179/400000 [00:54<00:00, 6859.19it/s] 99%|█████████▉| 396965/400000 [00:54<00:00, 7129.71it/s] 99%|█████████▉| 397735/400000 [00:54<00:00, 7290.02it/s]100%|█████████▉| 398469/400000 [00:54<00:00, 7011.96it/s]100%|█████████▉| 399176/400000 [00:54<00:00, 6773.85it/s]100%|█████████▉| 399859/400000 [00:54<00:00, 6734.90it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7279.05it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f544c136160>, <torchtext.data.dataset.TabularDataset object at 0x7f544c1362b0>, <torchtext.vocab.Vocab object at 0x7f544c1361d0>), {}) 


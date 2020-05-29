
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5a4d4f2378> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5a4d4f2378> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5abf5400d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5abf5400d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5a56cd29d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5a56cd29d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5a6ce248c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5a6ce248c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5a6ce248c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:33, 297449.14it/s]  2%|▏         | 212992/9912422 [00:00<00:25, 384126.14it/s]  9%|▉         | 876544/9912422 [00:00<00:17, 531407.13it/s] 34%|███▎      | 3325952/9912422 [00:00<00:08, 748856.06it/s] 68%|██████▊   | 6717440/9912422 [00:00<00:03, 1057716.87it/s]9920512it [00:00, 10034309.67it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 105638.01it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:02, 765940.52it/s] 30%|███       | 499712/1648877 [00:00<00:01, 980683.17it/s]1654784it [00:00, 3369794.08it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 50367.70it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5a4f87dd68>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5a4f87dcc0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5a6ce24510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5a6ce24510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5a6ce24510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<02:23,  1.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<02:23,  1.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<02:22,  1.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<02:21,  1.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:01<01:40,  1.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:01<01:40,  1.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<01:39,  1.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<01:39,  1.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<01:38,  1.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:01<01:09,  2.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<01:09,  2.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<01:09,  2.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<01:08,  2.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<01:08,  2.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:01<00:48,  3.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:48,  3.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:48,  3.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:48,  3.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:47,  3.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:01<00:34,  4.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:34,  4.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:34,  4.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:33,  4.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:33,  4.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:01<00:24,  5.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:24,  5.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:24,  5.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:24,  5.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:24,  5.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:01<00:17,  7.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:17,  7.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:17,  7.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:17,  7.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:17,  7.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:01<00:13,  9.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:13,  9.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:13,  9.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:13,  9.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:13,  9.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:01<00:10, 12.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:10, 12.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:10, 12.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:10, 12.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:10, 12.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:08, 15.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:08, 15.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:08, 15.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:02<00:08, 15.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:02<00:07, 15.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:02<00:06, 18.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:02<00:06, 18.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:02<00:06, 18.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:02<00:06, 18.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:02<00:06, 18.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:02<00:05, 21.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:02<00:05, 21.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:02<00:05, 21.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:02<00:05, 21.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:02<00:05, 21.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:02<00:04, 24.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:02<00:04, 24.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:02<00:04, 24.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:04, 24.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:04, 24.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:04, 24.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:02<00:03, 27.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:03, 27.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:03, 27.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:03, 27.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:03, 27.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:03, 27.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:02<00:03, 31.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:02<00:02, 34.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:02, 34.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:02, 34.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:02, 34.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:02, 34.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:02, 34.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:02<00:02, 37.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:02, 37.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:02, 37.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 37.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:02, 37.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 37.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 39.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 39.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 39.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 39.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 39.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 39.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 40.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 40.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 40.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:03<00:02, 40.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:03<00:01, 40.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:03<00:01, 40.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:03<00:01, 42.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:03<00:01, 42.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:03<00:01, 42.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:03<00:01, 42.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:03<00:01, 42.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:03<00:01, 42.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:03<00:01, 42.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:03<00:01, 42.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:03<00:01, 42.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:03<00:01, 42.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:03<00:01, 42.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:03<00:01, 42.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:01, 43.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:01, 43.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:03<00:01, 43.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:03<00:01, 43.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:03<00:01, 43.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:03<00:01, 43.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:03<00:01, 44.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:03<00:01, 44.47 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:03<00:01, 44.47 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:03<00:01, 44.47 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:01, 44.47 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:03<00:01, 44.47 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 44.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 44.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:01, 44.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:01, 44.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 44.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 44.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 44.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 44.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 44.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 44.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 44.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 44.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 44.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 44.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:01, 44.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:01, 44.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:01, 44.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:01, 44.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 45.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 45.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 45.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 45.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 45.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 45.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 45.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 45.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 45.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:04<00:00, 45.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:04<00:00, 45.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:04<00:00, 45.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:04<00:00, 44.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:04<00:00, 44.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:04<00:00, 44.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:04<00:00, 44.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:04<00:00, 44.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:04<00:00, 44.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:04<00:00, 45.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:04<00:00, 45.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:04<00:00, 45.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:04<00:00, 45.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:04<00:00, 45.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:04<00:00, 45.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:04<00:00, 45.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:04<00:00, 45.72 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:04<00:00, 45.72 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:04<00:00, 45.72 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:04<00:00, 45.72 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:04<00:00, 45.72 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:04<00:00, 45.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:04<00:00, 45.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:04<00:00, 45.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:04<00:00, 45.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:04<00:00, 45.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:04<00:00, 45.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 45.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 45.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:04<00:00, 45.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:04<00:00, 45.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:04<00:00, 45.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:04<00:00, 45.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 45.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 45.35 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 45.35 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 45.35 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 45.35 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 45.35 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 44.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 44.80 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 44.80 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 44.80 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 44.80 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 44.80 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.82s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.82s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 44.80 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.82s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 44.80 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.69s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.82s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 44.80 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.69s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.69s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 24.22 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.69s/ url]
0 examples [00:00, ? examples/s]2020-05-29 18:08:37.727229: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-29 18:08:37.731341: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-29 18:08:37.731467: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5566ef056410 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-29 18:08:37.731481: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
83 examples [00:00, 829.33 examples/s]201 examples [00:00, 910.21 examples/s]312 examples [00:00, 960.65 examples/s]415 examples [00:00, 979.75 examples/s]527 examples [00:00, 1015.83 examples/s]634 examples [00:00, 1029.91 examples/s]743 examples [00:00, 1045.56 examples/s]853 examples [00:00, 1059.73 examples/s]967 examples [00:00, 1081.14 examples/s]1082 examples [00:01, 1099.01 examples/s]1192 examples [00:01, 1097.34 examples/s]1301 examples [00:01, 1087.56 examples/s]1418 examples [00:01, 1109.33 examples/s]1537 examples [00:01, 1132.21 examples/s]1651 examples [00:01, 1132.41 examples/s]1765 examples [00:01, 1111.29 examples/s]1877 examples [00:01, 1113.66 examples/s]1996 examples [00:01, 1132.44 examples/s]2112 examples [00:01, 1140.06 examples/s]2227 examples [00:02, 1128.47 examples/s]2340 examples [00:02, 1104.72 examples/s]2460 examples [00:02, 1129.68 examples/s]2580 examples [00:02, 1149.03 examples/s]2698 examples [00:02, 1155.56 examples/s]2814 examples [00:02, 1136.06 examples/s]2928 examples [00:02, 1120.45 examples/s]3045 examples [00:02, 1133.99 examples/s]3159 examples [00:02, 1115.68 examples/s]3275 examples [00:02, 1128.42 examples/s]3392 examples [00:03, 1138.39 examples/s]3506 examples [00:03, 1120.71 examples/s]3622 examples [00:03, 1130.50 examples/s]3736 examples [00:03, 1133.19 examples/s]3850 examples [00:03, 1133.74 examples/s]3964 examples [00:03, 1133.99 examples/s]4078 examples [00:03, 1098.81 examples/s]4196 examples [00:03, 1120.53 examples/s]4313 examples [00:03, 1134.30 examples/s]4431 examples [00:03, 1145.86 examples/s]4548 examples [00:04, 1152.16 examples/s]4664 examples [00:04, 1139.39 examples/s]4781 examples [00:04, 1145.83 examples/s]4899 examples [00:04, 1154.23 examples/s]5015 examples [00:04, 1144.79 examples/s]5132 examples [00:04, 1150.38 examples/s]5248 examples [00:04, 1103.79 examples/s]5367 examples [00:04, 1126.98 examples/s]5486 examples [00:04, 1142.57 examples/s]5604 examples [00:04, 1152.58 examples/s]5720 examples [00:05, 1130.18 examples/s]5834 examples [00:05, 1086.90 examples/s]5944 examples [00:05, 1082.04 examples/s]6059 examples [00:05, 1100.67 examples/s]6177 examples [00:05, 1120.45 examples/s]6291 examples [00:05, 1122.71 examples/s]6406 examples [00:05, 1129.19 examples/s]6526 examples [00:05, 1148.45 examples/s]6642 examples [00:05, 1126.82 examples/s]6759 examples [00:06, 1137.22 examples/s]6873 examples [00:06, 1123.52 examples/s]6986 examples [00:06, 1103.84 examples/s]7105 examples [00:06, 1125.85 examples/s]7223 examples [00:06, 1141.46 examples/s]7342 examples [00:06, 1153.80 examples/s]7461 examples [00:06, 1164.14 examples/s]7578 examples [00:06, 1141.47 examples/s]7693 examples [00:06, 1122.18 examples/s]7806 examples [00:06, 1115.17 examples/s]7918 examples [00:07, 1109.47 examples/s]8030 examples [00:07, 1105.79 examples/s]8141 examples [00:07, 1096.68 examples/s]8251 examples [00:07, 1093.30 examples/s]8361 examples [00:07, 1085.26 examples/s]8470 examples [00:07, 1071.86 examples/s]8582 examples [00:07, 1084.10 examples/s]8696 examples [00:07, 1100.21 examples/s]8813 examples [00:07, 1119.24 examples/s]8934 examples [00:07, 1144.58 examples/s]9052 examples [00:08, 1152.75 examples/s]9168 examples [00:08, 1143.91 examples/s]9286 examples [00:08, 1153.72 examples/s]9402 examples [00:08, 1147.59 examples/s]9517 examples [00:08, 1134.51 examples/s]9631 examples [00:08, 1122.04 examples/s]9744 examples [00:08, 1110.10 examples/s]9863 examples [00:08, 1132.04 examples/s]9977 examples [00:08, 1127.98 examples/s]10090 examples [00:09, 1040.10 examples/s]10203 examples [00:09, 1065.50 examples/s]10311 examples [00:09, 1065.14 examples/s]10425 examples [00:09, 1084.99 examples/s]10537 examples [00:09, 1093.64 examples/s]10651 examples [00:09, 1105.69 examples/s]10762 examples [00:09, 1106.02 examples/s]10873 examples [00:09, 1098.12 examples/s]10984 examples [00:09, 1099.11 examples/s]11097 examples [00:09, 1106.54 examples/s]11208 examples [00:10, 1107.26 examples/s]11319 examples [00:10, 1104.02 examples/s]11430 examples [00:10, 1084.37 examples/s]11539 examples [00:10, 1056.22 examples/s]11651 examples [00:10, 1071.64 examples/s]11763 examples [00:10, 1082.84 examples/s]11872 examples [00:10, 1082.92 examples/s]11984 examples [00:10, 1091.06 examples/s]12104 examples [00:10, 1121.11 examples/s]12222 examples [00:10, 1136.09 examples/s]12339 examples [00:11, 1144.95 examples/s]12454 examples [00:11, 1144.82 examples/s]12569 examples [00:11, 1130.40 examples/s]12689 examples [00:11, 1148.31 examples/s]12811 examples [00:11, 1166.31 examples/s]12931 examples [00:11, 1174.73 examples/s]13049 examples [00:11, 1172.31 examples/s]13167 examples [00:11, 1157.51 examples/s]13283 examples [00:11, 1121.90 examples/s]13396 examples [00:11, 1076.88 examples/s]13518 examples [00:12, 1114.56 examples/s]13635 examples [00:12, 1130.26 examples/s]13752 examples [00:12, 1140.14 examples/s]13867 examples [00:12, 1133.21 examples/s]13985 examples [00:12, 1146.68 examples/s]14103 examples [00:12, 1153.57 examples/s]14221 examples [00:12, 1160.04 examples/s]14338 examples [00:12, 1140.47 examples/s]14455 examples [00:12, 1146.74 examples/s]14573 examples [00:13, 1156.11 examples/s]14689 examples [00:13, 1153.97 examples/s]14807 examples [00:13, 1161.42 examples/s]14924 examples [00:13, 1154.75 examples/s]15040 examples [00:13, 1152.14 examples/s]15156 examples [00:13, 1130.43 examples/s]15270 examples [00:13, 1123.66 examples/s]15386 examples [00:13, 1133.47 examples/s]15500 examples [00:13, 1131.58 examples/s]15619 examples [00:13, 1146.33 examples/s]15737 examples [00:14, 1155.66 examples/s]15853 examples [00:14, 1151.48 examples/s]15970 examples [00:14, 1155.10 examples/s]16086 examples [00:14, 1138.39 examples/s]16202 examples [00:14, 1142.86 examples/s]16317 examples [00:14, 1139.79 examples/s]16432 examples [00:14, 1140.42 examples/s]16547 examples [00:14, 1137.03 examples/s]16661 examples [00:14, 1120.93 examples/s]16775 examples [00:14, 1126.40 examples/s]16888 examples [00:15, 1101.95 examples/s]16999 examples [00:15, 1087.84 examples/s]17108 examples [00:15, 1086.55 examples/s]17222 examples [00:15, 1099.49 examples/s]17340 examples [00:15, 1120.03 examples/s]17456 examples [00:15, 1130.82 examples/s]17573 examples [00:15, 1142.25 examples/s]17693 examples [00:15, 1156.41 examples/s]17809 examples [00:15, 1123.84 examples/s]17922 examples [00:15, 1080.32 examples/s]18037 examples [00:16, 1099.73 examples/s]18154 examples [00:16, 1117.89 examples/s]18271 examples [00:16, 1130.63 examples/s]18385 examples [00:16, 1112.54 examples/s]18498 examples [00:16, 1116.93 examples/s]18614 examples [00:16, 1128.38 examples/s]18728 examples [00:16, 1130.75 examples/s]18842 examples [00:16, 1129.01 examples/s]18955 examples [00:16, 1108.29 examples/s]19068 examples [00:16, 1110.93 examples/s]19180 examples [00:17, 1074.32 examples/s]19288 examples [00:17, 1074.30 examples/s]19399 examples [00:17, 1083.95 examples/s]19508 examples [00:17, 1066.30 examples/s]19622 examples [00:17, 1085.88 examples/s]19734 examples [00:17, 1093.09 examples/s]19846 examples [00:17, 1098.87 examples/s]19960 examples [00:17, 1109.25 examples/s]20072 examples [00:17, 1025.46 examples/s]20180 examples [00:18, 1041.12 examples/s]20286 examples [00:18, 1037.43 examples/s]20407 examples [00:18, 1082.30 examples/s]20521 examples [00:18, 1098.95 examples/s]20632 examples [00:18, 1074.59 examples/s]20741 examples [00:18, 1060.80 examples/s]20849 examples [00:18, 1064.43 examples/s]20959 examples [00:18, 1072.61 examples/s]21069 examples [00:18, 1079.29 examples/s]21178 examples [00:18, 1079.85 examples/s]21289 examples [00:19, 1088.22 examples/s]21404 examples [00:19, 1103.64 examples/s]21521 examples [00:19, 1122.70 examples/s]21637 examples [00:19, 1132.32 examples/s]21751 examples [00:19, 1112.92 examples/s]21869 examples [00:19, 1130.89 examples/s]21987 examples [00:19, 1145.03 examples/s]22103 examples [00:19, 1147.91 examples/s]22222 examples [00:19, 1159.92 examples/s]22339 examples [00:19, 1154.08 examples/s]22455 examples [00:20, 1147.22 examples/s]22570 examples [00:20, 1132.18 examples/s]22686 examples [00:20, 1138.09 examples/s]22800 examples [00:20, 1135.46 examples/s]22914 examples [00:20, 1132.96 examples/s]23028 examples [00:20, 1116.97 examples/s]23143 examples [00:20, 1125.92 examples/s]23262 examples [00:20, 1143.80 examples/s]23377 examples [00:20, 1139.28 examples/s]23494 examples [00:20, 1146.22 examples/s]23609 examples [00:21, 1147.01 examples/s]23724 examples [00:21, 1085.34 examples/s]23834 examples [00:21, 1079.66 examples/s]23943 examples [00:21, 1078.40 examples/s]24060 examples [00:21, 1103.13 examples/s]24178 examples [00:21, 1122.93 examples/s]24293 examples [00:21, 1128.79 examples/s]24407 examples [00:21, 1120.04 examples/s]24520 examples [00:21, 1109.99 examples/s]24637 examples [00:22, 1126.35 examples/s]24753 examples [00:22, 1133.59 examples/s]24867 examples [00:22, 1099.48 examples/s]24978 examples [00:22, 1095.52 examples/s]25088 examples [00:22, 1078.37 examples/s]25200 examples [00:22, 1089.92 examples/s]25311 examples [00:22, 1095.79 examples/s]25423 examples [00:22, 1102.50 examples/s]25534 examples [00:22, 1104.62 examples/s]25648 examples [00:22, 1112.23 examples/s]25765 examples [00:23, 1127.30 examples/s]25879 examples [00:23, 1128.36 examples/s]25992 examples [00:23, 1123.44 examples/s]26105 examples [00:23, 1119.57 examples/s]26217 examples [00:23, 1093.60 examples/s]26334 examples [00:23, 1111.93 examples/s]26450 examples [00:23, 1125.63 examples/s]26568 examples [00:23, 1140.40 examples/s]26684 examples [00:23, 1145.40 examples/s]26799 examples [00:23, 1142.62 examples/s]26914 examples [00:24, 1140.82 examples/s]27029 examples [00:24, 1132.19 examples/s]27143 examples [00:24, 1100.89 examples/s]27254 examples [00:24, 1086.26 examples/s]27363 examples [00:24, 1069.94 examples/s]27471 examples [00:24, 1069.03 examples/s]27579 examples [00:24, 1066.98 examples/s]27686 examples [00:24, 1061.83 examples/s]27793 examples [00:24, 1055.66 examples/s]27907 examples [00:25, 1077.82 examples/s]28017 examples [00:25, 1083.56 examples/s]28133 examples [00:25, 1103.22 examples/s]28244 examples [00:25, 1100.52 examples/s]28355 examples [00:25, 1070.89 examples/s]28469 examples [00:25, 1090.39 examples/s]28580 examples [00:25, 1093.66 examples/s]28691 examples [00:25, 1098.07 examples/s]28802 examples [00:25, 1099.13 examples/s]28913 examples [00:25, 1092.20 examples/s]29024 examples [00:26, 1096.31 examples/s]29137 examples [00:26, 1104.26 examples/s]29248 examples [00:26, 1099.02 examples/s]29362 examples [00:26, 1108.49 examples/s]29473 examples [00:26, 1087.70 examples/s]29582 examples [00:26, 1082.52 examples/s]29691 examples [00:26, 1078.09 examples/s]29799 examples [00:26, 1058.32 examples/s]29905 examples [00:26, 1052.49 examples/s]30011 examples [00:26, 993.82 examples/s] 30121 examples [00:27, 1023.13 examples/s]30231 examples [00:27, 1042.29 examples/s]30340 examples [00:27, 1054.91 examples/s]30446 examples [00:27, 1036.33 examples/s]30551 examples [00:27, 1030.34 examples/s]30655 examples [00:27, 1015.09 examples/s]30757 examples [00:27, 997.47 examples/s] 30861 examples [00:27, 1008.43 examples/s]30968 examples [00:27, 1025.42 examples/s]31078 examples [00:27, 1045.66 examples/s]31191 examples [00:28, 1067.57 examples/s]31304 examples [00:28, 1084.30 examples/s]31415 examples [00:28, 1089.43 examples/s]31525 examples [00:28, 1071.63 examples/s]31633 examples [00:28, 1073.52 examples/s]31745 examples [00:28, 1086.64 examples/s]31857 examples [00:28, 1095.89 examples/s]31967 examples [00:28, 1095.96 examples/s]32077 examples [00:28, 1096.99 examples/s]32187 examples [00:28, 1092.48 examples/s]32297 examples [00:29, 1087.81 examples/s]32409 examples [00:29, 1096.72 examples/s]32521 examples [00:29, 1102.66 examples/s]32632 examples [00:29, 1090.00 examples/s]32742 examples [00:29, 1091.48 examples/s]32863 examples [00:29, 1122.69 examples/s]32978 examples [00:29, 1129.43 examples/s]33092 examples [00:29, 1071.50 examples/s]33200 examples [00:29, 1026.19 examples/s]33304 examples [00:30, 956.75 examples/s] 33404 examples [00:30, 968.63 examples/s]33511 examples [00:30, 984.92 examples/s]33611 examples [00:30, 964.59 examples/s]33709 examples [00:30, 941.47 examples/s]33814 examples [00:30, 970.87 examples/s]33914 examples [00:30, 974.91 examples/s]34023 examples [00:30, 1004.09 examples/s]34136 examples [00:30, 1038.81 examples/s]34241 examples [00:31, 995.93 examples/s] 34353 examples [00:31, 1028.38 examples/s]34468 examples [00:31, 1062.05 examples/s]34577 examples [00:31, 1068.11 examples/s]34689 examples [00:31, 1079.42 examples/s]34801 examples [00:31, 1089.38 examples/s]34911 examples [00:31, 1082.84 examples/s]35020 examples [00:31, 1078.00 examples/s]35128 examples [00:31, 1075.55 examples/s]35240 examples [00:31, 1088.04 examples/s]35349 examples [00:32, 1078.25 examples/s]35458 examples [00:32, 1080.48 examples/s]35570 examples [00:32, 1091.33 examples/s]35682 examples [00:32, 1098.37 examples/s]35792 examples [00:32, 1098.81 examples/s]35902 examples [00:32, 1053.38 examples/s]36011 examples [00:32, 1061.68 examples/s]36119 examples [00:32, 1066.39 examples/s]36229 examples [00:32, 1074.54 examples/s]36350 examples [00:32, 1110.16 examples/s]36462 examples [00:33, 1099.28 examples/s]36577 examples [00:33, 1110.58 examples/s]36689 examples [00:33, 1110.05 examples/s]36801 examples [00:33, 1103.95 examples/s]36912 examples [00:33, 1103.69 examples/s]37028 examples [00:33, 1117.01 examples/s]37143 examples [00:33, 1126.39 examples/s]37260 examples [00:33, 1138.96 examples/s]37377 examples [00:33, 1147.05 examples/s]37492 examples [00:33, 1116.66 examples/s]37605 examples [00:34, 1119.39 examples/s]37718 examples [00:34, 1118.15 examples/s]37834 examples [00:34, 1126.94 examples/s]37950 examples [00:34, 1135.30 examples/s]38065 examples [00:34, 1138.24 examples/s]38179 examples [00:34, 1123.43 examples/s]38296 examples [00:34, 1136.13 examples/s]38410 examples [00:34, 1120.81 examples/s]38523 examples [00:34, 1098.00 examples/s]38633 examples [00:34, 1072.07 examples/s]38743 examples [00:35, 1078.85 examples/s]38856 examples [00:35, 1090.93 examples/s]38966 examples [00:35, 1090.71 examples/s]39080 examples [00:35, 1102.98 examples/s]39191 examples [00:35, 1063.21 examples/s]39312 examples [00:35, 1102.07 examples/s]39428 examples [00:35, 1118.62 examples/s]39546 examples [00:35, 1135.15 examples/s]39663 examples [00:35, 1144.25 examples/s]39778 examples [00:36, 1137.02 examples/s]39892 examples [00:36, 1137.04 examples/s]40006 examples [00:36, 1059.85 examples/s]40125 examples [00:36, 1095.16 examples/s]40236 examples [00:36, 1095.93 examples/s]40347 examples [00:36, 1063.25 examples/s]40456 examples [00:36, 1068.81 examples/s]40568 examples [00:36, 1083.52 examples/s]40677 examples [00:36, 1077.56 examples/s]40786 examples [00:36, 1053.45 examples/s]40900 examples [00:37, 1077.57 examples/s]41014 examples [00:37, 1094.21 examples/s]41124 examples [00:37, 1078.29 examples/s]41233 examples [00:37, 1079.68 examples/s]41343 examples [00:37, 1084.42 examples/s]41452 examples [00:37, 1062.87 examples/s]41559 examples [00:37, 1055.96 examples/s]41665 examples [00:37, 1056.83 examples/s]41779 examples [00:37, 1079.64 examples/s]41895 examples [00:37, 1101.49 examples/s]42006 examples [00:38, 1096.96 examples/s]42117 examples [00:38, 1100.50 examples/s]42228 examples [00:38, 1099.25 examples/s]42339 examples [00:38, 1091.54 examples/s]42454 examples [00:38, 1106.70 examples/s]42565 examples [00:38, 1104.32 examples/s]42681 examples [00:38, 1119.16 examples/s]42799 examples [00:38, 1135.37 examples/s]42915 examples [00:38, 1141.99 examples/s]43031 examples [00:38, 1144.58 examples/s]43146 examples [00:39, 1138.56 examples/s]43264 examples [00:39, 1149.40 examples/s]43380 examples [00:39, 1150.74 examples/s]43496 examples [00:39, 1146.49 examples/s]43613 examples [00:39, 1152.86 examples/s]43729 examples [00:39, 1117.03 examples/s]43841 examples [00:39, 1110.96 examples/s]43953 examples [00:39, 1102.46 examples/s]44064 examples [00:39, 1099.14 examples/s]44175 examples [00:40, 1068.23 examples/s]44283 examples [00:40, 1064.97 examples/s]44396 examples [00:40, 1082.56 examples/s]44509 examples [00:40, 1094.18 examples/s]44621 examples [00:40, 1101.45 examples/s]44732 examples [00:40, 1100.96 examples/s]44843 examples [00:40, 1087.03 examples/s]44957 examples [00:40, 1100.34 examples/s]45068 examples [00:40, 1085.13 examples/s]45179 examples [00:40, 1091.47 examples/s]45293 examples [00:41, 1103.69 examples/s]45404 examples [00:41, 1086.69 examples/s]45518 examples [00:41, 1100.54 examples/s]45630 examples [00:41, 1104.21 examples/s]45741 examples [00:41, 1078.23 examples/s]45850 examples [00:41, 1076.65 examples/s]45958 examples [00:41, 1074.16 examples/s]46066 examples [00:41, 1070.97 examples/s]46179 examples [00:41, 1087.49 examples/s]46290 examples [00:41, 1093.62 examples/s]46403 examples [00:42, 1102.63 examples/s]46514 examples [00:42, 1082.33 examples/s]46624 examples [00:42, 1085.17 examples/s]46735 examples [00:42, 1092.43 examples/s]46851 examples [00:42, 1110.31 examples/s]46969 examples [00:42, 1128.67 examples/s]47083 examples [00:42, 1118.92 examples/s]47196 examples [00:42, 1114.58 examples/s]47308 examples [00:42, 1101.70 examples/s]47425 examples [00:42, 1121.04 examples/s]47538 examples [00:43, 1099.81 examples/s]47649 examples [00:43, 1092.70 examples/s]47765 examples [00:43, 1109.62 examples/s]47882 examples [00:43, 1125.10 examples/s]47998 examples [00:43, 1133.66 examples/s]48115 examples [00:43, 1142.41 examples/s]48230 examples [00:43, 1129.95 examples/s]48344 examples [00:43, 1129.13 examples/s]48458 examples [00:43, 1132.37 examples/s]48573 examples [00:43, 1135.00 examples/s]48690 examples [00:44, 1145.11 examples/s]48805 examples [00:44, 1133.84 examples/s]48923 examples [00:44, 1144.61 examples/s]49041 examples [00:44, 1153.53 examples/s]49159 examples [00:44, 1160.27 examples/s]49276 examples [00:44, 1152.00 examples/s]49392 examples [00:44, 1144.93 examples/s]49507 examples [00:44, 1133.33 examples/s]49621 examples [00:44, 1133.52 examples/s]49736 examples [00:45, 1137.09 examples/s]49853 examples [00:45, 1145.53 examples/s]49968 examples [00:45, 1128.03 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 19%|█▉        | 9607/50000 [00:00<00:00, 96069.86 examples/s] 48%|████▊     | 23840/50000 [00:00<00:00, 106448.54 examples/s] 76%|███████▌  | 38118/50000 [00:00<00:00, 115245.88 examples/s]                                                                0 examples [00:00, ? examples/s]86 examples [00:00, 852.08 examples/s]203 examples [00:00, 926.51 examples/s]321 examples [00:00, 989.57 examples/s]439 examples [00:00, 1039.25 examples/s]543 examples [00:00, 1039.13 examples/s]653 examples [00:00, 1055.63 examples/s]750 examples [00:00, 882.17 examples/s] 862 examples [00:00, 940.44 examples/s]979 examples [00:00, 999.00 examples/s]1093 examples [00:01, 1035.59 examples/s]1212 examples [00:01, 1075.27 examples/s]1330 examples [00:01, 1103.65 examples/s]1450 examples [00:01, 1130.49 examples/s]1571 examples [00:01, 1152.68 examples/s]1687 examples [00:01, 1148.80 examples/s]1803 examples [00:01, 1150.48 examples/s]1920 examples [00:01, 1155.69 examples/s]2036 examples [00:01, 1147.11 examples/s]2154 examples [00:01, 1154.17 examples/s]2270 examples [00:02, 1143.58 examples/s]2387 examples [00:02, 1150.94 examples/s]2503 examples [00:02, 1143.51 examples/s]2618 examples [00:02, 1109.09 examples/s]2730 examples [00:02, 1098.66 examples/s]2841 examples [00:02, 1090.94 examples/s]2958 examples [00:02, 1112.08 examples/s]3079 examples [00:02, 1139.14 examples/s]3196 examples [00:02, 1145.53 examples/s]3313 examples [00:02, 1152.74 examples/s]3429 examples [00:03, 1132.81 examples/s]3543 examples [00:03, 1120.45 examples/s]3656 examples [00:03, 1120.77 examples/s]3769 examples [00:03, 1112.71 examples/s]3881 examples [00:03, 1103.24 examples/s]3992 examples [00:03, 1101.80 examples/s]4109 examples [00:03, 1119.18 examples/s]4222 examples [00:03, 1121.27 examples/s]4335 examples [00:03, 1112.75 examples/s]4447 examples [00:04, 1111.56 examples/s]4559 examples [00:04, 1082.23 examples/s]4669 examples [00:04, 1085.35 examples/s]4780 examples [00:04, 1092.13 examples/s]4891 examples [00:04, 1087.73 examples/s]5004 examples [00:04, 1098.59 examples/s]5114 examples [00:04, 1085.42 examples/s]5225 examples [00:04, 1092.28 examples/s]5338 examples [00:04, 1102.30 examples/s]5452 examples [00:04, 1109.97 examples/s]5564 examples [00:05, 1108.31 examples/s]5675 examples [00:05, 1086.87 examples/s]5787 examples [00:05, 1095.12 examples/s]5900 examples [00:05, 1103.61 examples/s]6011 examples [00:05, 1105.10 examples/s]6122 examples [00:05, 1105.54 examples/s]6233 examples [00:05, 1089.04 examples/s]6342 examples [00:05, 1024.52 examples/s]6455 examples [00:05, 1052.07 examples/s]6573 examples [00:05, 1086.23 examples/s]6693 examples [00:06, 1116.94 examples/s]6806 examples [00:06, 1103.92 examples/s]6920 examples [00:06, 1112.98 examples/s]7037 examples [00:06, 1127.51 examples/s]7153 examples [00:06, 1134.74 examples/s]7267 examples [00:06, 1133.94 examples/s]7381 examples [00:06, 1100.11 examples/s]7494 examples [00:06, 1106.53 examples/s]7605 examples [00:06, 1106.97 examples/s]7716 examples [00:06, 1106.20 examples/s]7828 examples [00:07, 1108.08 examples/s]7939 examples [00:07, 1094.60 examples/s]8051 examples [00:07, 1101.61 examples/s]8162 examples [00:07, 1079.24 examples/s]8271 examples [00:07, 1054.22 examples/s]8380 examples [00:07, 1064.12 examples/s]8487 examples [00:07, 1059.14 examples/s]8597 examples [00:07, 1069.05 examples/s]8715 examples [00:07, 1099.79 examples/s]8827 examples [00:08, 1105.43 examples/s]8941 examples [00:08, 1113.13 examples/s]9056 examples [00:08, 1122.48 examples/s]9170 examples [00:08, 1126.22 examples/s]9286 examples [00:08, 1135.42 examples/s]9404 examples [00:08, 1146.86 examples/s]9523 examples [00:08, 1158.05 examples/s]9639 examples [00:08, 1143.16 examples/s]9756 examples [00:08, 1150.21 examples/s]9872 examples [00:08, 1152.20 examples/s]9988 examples [00:09, 1114.64 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete3LSHW4/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete3LSHW4/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5a6ce248c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5a6ce248c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5a6ce248c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ad92c7da0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5a563dab38>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5a6ce248c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5a6ce248c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5a6ce248c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5a4e45fdd8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5a504d52e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f5a4c0597b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f5a4c0597b8> 

  function with postional parmater data_info <function split_train_valid at 0x7f5a4c0597b8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f5a4c0598c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f5a4c0598c8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f5a4c0598c8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=094bc4463f99422d16fc5cb1d65627ee99dc1fa9b9c8d842b3be841b856023b4
  Stored in directory: /tmp/pip-ephem-wheel-cache-a0mvsgsy/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:09<275:34:35, 869B/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:09<192:49:27, 1.24kB/s].vector_cache/glove.6B.zip:   1%|          | 4.55M/862M [00:09<134:20:14, 1.77kB/s].vector_cache/glove.6B.zip:   1%|          | 10.3M/862M [00:09<93:24:17, 2.53kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 16.6M/862M [00:09<64:54:19, 3.62kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.2M/862M [00:09<45:01:33, 5.17kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.8M/862M [00:10<31:11:43, 7.39kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.3M/862M [00:10<21:33:37, 10.6kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.3M/862M [00:10<14:57:47, 15.1kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:10<10:28:02, 21.5kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:10<7:17:53, 30.7kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:12<17:30:01, 12.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:12<12:14:53, 18.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:14<8:34:27, 26.0kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:14<6:01:35, 37.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.6M/862M [00:14<4:12:01, 52.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:16<4:02:42, 54.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:16<2:51:16, 77.7kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:18<2:01:14, 109kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:18<1:26:27, 153kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:20<1:02:07, 212kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:20<44:42, 294kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:22<33:04, 396kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:22<24:33, 533kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:24<19:00, 686kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:24<14:52, 876kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:26<12:14, 1.06MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:26<09:32, 1.36MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:28<08:34, 1.50MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:28<06:52, 1.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:29<06:46, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:30<05:36, 2.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:31<05:52, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:32<04:58, 2.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:33<05:25, 2.34MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:34<04:39, 2.72MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:35<05:11, 2.43MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:35<04:24, 2.86MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:37<05:00, 2.51MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:37<04:22, 2.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:39<04:57, 2.51MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:39<04:41, 2.66MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:40<03:21, 3.69MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:41<1:04:12, 193kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:41<45:45, 271kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:43<33:46, 365kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:43<24:26, 505kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:43<17:08, 716kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:45<26:43, 459kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:45<19:31, 628kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:47<15:28, 789kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:47<11:44, 1.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:49<10:00, 1.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:49<07:49, 1.55MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:51<07:18, 1.65MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:51<05:53, 2.05MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:52<04:31, 2.65MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:53<8:34:07, 23.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:53<5:59:35, 33.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:55<4:12:59, 47.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:55<2:57:46, 67.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:57<2:05:39, 94.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:57<1:28:43, 134kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:59<1:03:36, 186kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:59<45:17, 260kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:01<33:21, 352kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:01<24:19, 482kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [01:03<18:39, 625kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:03<13:51, 841kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:05<11:26, 1.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:05<08:47, 1.32MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:07<07:54, 1.46MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:07<06:16, 1.84MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:08<06:07, 1.87MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:09<05:14, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:09<03:44, 3.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:10<1:34:57, 120kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:11<1:07:11, 169kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:12<48:32, 233kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:13<34:43, 326kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:14<25:54, 434kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:15<18:53, 595kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:16<14:52, 752kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:16<11:42, 955kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:18<09:45, 1.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:18<07:35, 1.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:20<06:59, 1.58MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:20<05:34, 1.98MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:22<05:34, 1.97MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:22<05:01, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:24<05:05, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:24<04:27, 2.44MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:26<04:43, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:26<03:58, 2.73MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:28<04:25, 2.43MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:28<03:49, 2.81MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:30<04:19, 2.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:30<03:42, 2.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:31<02:57, 3.60MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:32<7:33:45, 23.5kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:32<5:17:10, 33.5kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:34<3:43:14, 47.4kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:34<2:36:54, 67.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:34<1:49:20, 96.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:36<1:35:43, 110kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:36<1:07:54, 155kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:38<48:47, 214kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:38<35:20, 295kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:40<26:05, 398kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:40<19:15, 539kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:40<13:30, 764kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:42<19:07, 539kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:42<14:22, 717kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:44<11:30, 890kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:44<08:44, 1.17MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:46<07:39, 1.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:46<06:01, 1.69MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:47<05:43, 1.76MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:48<04:38, 2.17MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:48<03:18, 3.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:50<1:21:02, 124kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:50<56:29, 177kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:51<47:42, 209kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:52<34:31, 288kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:52<24:05, 411kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:53<1:28:03, 112kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:54<1:02:20, 159kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:54<43:27, 226kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:55<54:53, 179kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:55<39:06, 251kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:57<28:40, 340kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:57<21:01, 464kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:59<16:02, 604kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:59<12:14, 791kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:01<09:54, 970kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:01<07:41, 1.25MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:03<06:46, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:03<05:24, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:05<05:13, 1.82MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:05<04:17, 2.21MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [02:07<04:26, 2.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:07<03:44, 2.51MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:07<02:42, 3.46MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:09<07:53, 1.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:09<06:10, 1.51MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:11<05:43, 1.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:11<04:37, 2.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:12<03:31, 2.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:13<6:42:35, 22.9kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:13<4:41:25, 32.7kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:15<3:17:43, 46.3kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:15<2:19:19, 65.6kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:17<1:38:10, 92.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:17<1:09:15, 131kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:19<49:32, 182kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:19<35:43, 252kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:19<24:55, 359kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:21<26:19, 340kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:21<19:26, 460kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:23<14:47, 600kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:23<11:03, 802kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:25<09:00, 978kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:25<07:05, 1.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:27<06:13, 1.40MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:27<05:16, 1.66MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:27<03:44, 2.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:29<14:33, 595kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:29<10:56, 791kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:30<08:52, 968kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:31<06:47, 1.26MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:32<06:02, 1.41MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:33<04:56, 1.73MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:33<03:30, 2.41MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:34<12:04, 701kB/s] .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:35<09:02, 934kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:36<07:34, 1.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:37<05:52, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:37<04:09, 2.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:38<16:13, 513kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:38<11:55, 698kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:40<09:33, 864kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:40<07:32, 1.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:42<06:26, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:42<05:03, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:42<03:34, 2.27MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:44<1:08:11, 119kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:44<48:13, 168kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:46<34:45, 232kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:46<24:50, 323kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:48<18:29, 431kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:48<13:28, 591kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:50<10:34, 748kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:50<07:56, 994kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:52<06:42, 1.17MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:52<05:13, 1.50MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:53<03:53, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:54<5:40:45, 22.8kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:54<3:58:04, 32.6kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:56<2:47:08, 46.2kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:56<1:57:23, 65.6kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:58<1:22:42, 92.4kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:58<58:20, 131kB/s]   .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [03:00<41:40, 182kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [03:00<29:44, 255kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:02<21:46, 345kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:02<15:51, 473kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:04<12:06, 614kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:04<09:21, 794kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:04<06:33, 1.12MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:06<23:40, 311kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [03:06<16:59, 433kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:08<12:55, 565kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:08<09:52, 739kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:10<07:54, 915kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:10<06:01, 1.20MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:11<05:17, 1.36MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:12<04:10, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:13<03:59, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:14<03:22, 2.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:15<03:23, 2.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:16<03:10, 2.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:17<03:12, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:18<02:43, 2.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:19<02:56, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:19<02:32, 2.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:21<02:47, 2.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:21<02:25, 2.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:23<02:43, 2.48MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:23<02:22, 2.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:25<02:40, 2.50MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:25<02:19, 2.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:27<02:38, 2.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:27<02:15, 2.92MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:29<02:34, 2.54MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:29<02:13, 2.93MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:31<02:32, 2.56MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:31<02:13, 2.90MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:33<02:31, 2.53MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:33<02:13, 2.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:34<01:45, 3.63MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:35<4:37:59, 22.8kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:35<3:13:55, 32.6kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:37<2:16:12, 46.1kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:37<1:35:46, 65.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:39<1:07:18, 92.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:39<47:35, 130kB/s]   .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:39<33:03, 186kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:41<32:36, 188kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:41<23:12, 264kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:43<17:01, 357kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:43<12:21, 491kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:45<09:27, 635kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:45<07:01, 854kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:47<05:46, 1.03MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:47<04:27, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:49<03:59, 1.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:49<03:09, 1.86MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:51<03:04, 1.89MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:51<02:32, 2.27MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:52<02:38, 2.16MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:53<02:24, 2.38MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:53<01:42, 3.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:54<17:09, 330kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:55<12:23, 457kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:56<09:26, 593kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:57<06:57, 803kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:58<05:39, 976kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:58<04:22, 1.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:59<03:04, 1.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:00<12:15, 445kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:00<08:56, 609kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:02<07:01, 768kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:02<05:16, 1.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:04<04:28, 1.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:04<03:31, 1.51MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:06<03:13, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:06<02:37, 1.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:08<02:36, 1.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:08<02:10, 2.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:10<02:17, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:10<01:56, 2.64MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:12<02:06, 2.39MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:12<02:03, 2.45MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:14<02:08, 2.32MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:14<01:50, 2.71MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:15<01:25, 3.43MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:16<3:34:10, 23.0kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:16<2:29:19, 32.8kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:18<1:44:32, 46.4kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:18<1:13:25, 66.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:20<51:27, 92.9kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:20<36:15, 132kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:22<25:46, 183kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:22<18:34, 253kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:24<13:29, 344kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:24<09:52, 470kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:26<07:29, 611kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:26<05:47, 789kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:26<04:01, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:28<1:01:04, 73.8kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:28<43:07, 104kB/s]   .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:30<30:25, 146kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:30<21:48, 203kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:32<15:40, 279kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:32<11:12, 389kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:33<08:24, 511kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:34<06:23, 672kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:35<05:01, 843kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:36<03:58, 1.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:37<03:20, 1.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:38<02:37, 1.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:39<02:26, 1.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:39<01:57, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:40<01:24, 2.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:41<02:38, 1.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:41<02:07, 1.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:42<01:30, 2.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:43<04:42, 840kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:43<03:34, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:44<02:30, 1.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:45<06:16, 620kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:45<04:38, 835kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:47<03:47, 1.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:47<02:54, 1.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:49<02:34, 1.45MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:49<02:01, 1.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:51<01:57, 1.87MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:51<01:35, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:53<01:39, 2.18MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:53<01:24, 2.55MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:55<01:30, 2.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:55<01:15, 2.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:56<00:59, 3.49MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:57<2:30:38, 23.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:57<1:44:53, 33.0kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:59<1:13:05, 46.7kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:59<51:15, 66.5kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:01<35:46, 93.6kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:01<25:16, 132kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:03<17:51, 184kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:03<12:41, 258kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:05<09:12, 348kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:05<06:39, 481kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:07<05:03, 622kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:07<03:45, 834kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:09<03:02, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:09<02:25, 1.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:11<02:06, 1.43MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:11<01:40, 1.78MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:13<01:36, 1.83MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:13<01:19, 2.22MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:14<01:20, 2.13MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:15<01:07, 2.56MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:16<01:11, 2.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:17<01:02, 2.66MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:18<01:07, 2.42MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:19<00:58, 2.78MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:20<01:04, 2.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:21<00:55, 2.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:22<01:01, 2.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:22<01:01, 2.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:23<00:43, 3.51MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:24<11:23, 221kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:24<08:07, 309kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:26<05:55, 414kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:26<04:18, 567kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:28<03:18, 721kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:28<02:28, 962kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:30<02:02, 1.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:30<01:34, 1.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:32<01:25, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:32<01:10, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:34<01:07, 1.94MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:34<00:55, 2.33MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:36<00:57, 2.20MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:36<00:49, 2.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:38<00:52, 2.35MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:38<00:45, 2.71MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:39<00:35, 3.38MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:40<1:25:01, 23.3kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:40<58:52, 33.3kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:42<40:38, 47.1kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:42<28:27, 66.9kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:44<19:34, 94.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:44<13:46, 133kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<09:35, 185kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<06:48, 260kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:48<04:51, 351kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:48<03:30, 485kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:50<02:37, 626kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:50<01:56, 842kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:52<01:32, 1.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:52<01:11, 1.32MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:54<01:01, 1.46MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:54<00:49, 1.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:56<00:46, 1.86MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:56<00:38, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:58<00:38, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:58<00:32, 2.50MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:00<00:33, 2.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:00<00:28, 2.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:01<00:30, 2.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:02<00:26, 2.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:03<00:28, 2.48MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:04<00:24, 2.84MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:05<00:26, 2.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:06<00:22, 2.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:07<00:24, 2.51MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:08<00:21, 2.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:09<00:22, 2.52MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:09<00:21, 2.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:11<00:21, 2.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:11<00:18, 2.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:13<00:19, 2.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:13<00:16, 2.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:15<00:17, 2.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:15<00:15, 2.81MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:17<00:16, 2.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:17<00:14, 2.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:19<00:14, 2.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:19<00:12, 2.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:21<00:12, 2.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:21<00:11, 2.86MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:23<00:11, 2.51MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:23<00:09, 2.96MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:24<00:06, 3.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:25<18:47, 21.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:25<12:22, 31.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:27<07:44, 44.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:27<05:17, 62.6kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:29<03:04, 88.2kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:29<02:06, 125kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:31<01:10, 174kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:31<00:47, 244kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:33<00:24, 331kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:33<00:16, 460kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:35<00:06, 597kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:35<00:04, 790kB/s].vector_cache/glove.6B.zip: 862MB [06:35, 2.18MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<17:13:59,  6.45it/s]  0%|          | 888/400000 [00:00<12:02:24,  9.21it/s]  0%|          | 1795/400000 [00:00<8:24:45, 13.15it/s]  1%|          | 2745/400000 [00:00<5:52:41, 18.77it/s]  1%|          | 3721/400000 [00:00<4:06:29, 26.80it/s]  1%|          | 4642/400000 [00:00<2:52:21, 38.23it/s]  1%|▏         | 5584/400000 [00:00<2:00:34, 54.52it/s]  2%|▏         | 6483/400000 [00:00<1:24:25, 77.69it/s]  2%|▏         | 7332/400000 [00:00<59:12, 110.52it/s]   2%|▏         | 8167/400000 [00:01<41:36, 156.95it/s]  2%|▏         | 9066/400000 [00:01<29:16, 222.55it/s]  2%|▏         | 9959/400000 [00:01<20:39, 314.56it/s]  3%|▎         | 10819/400000 [00:01<14:39, 442.36it/s]  3%|▎         | 11676/400000 [00:01<10:28, 617.89it/s]  3%|▎         | 12659/400000 [00:01<07:30, 859.51it/s]  3%|▎         | 13599/400000 [00:01<05:27, 1181.53it/s]  4%|▎         | 14521/400000 [00:01<04:00, 1600.01it/s]  4%|▍         | 15447/400000 [00:01<03:00, 2128.05it/s]  4%|▍         | 16362/400000 [00:01<02:19, 2752.61it/s]  4%|▍         | 17339/400000 [00:02<01:49, 3507.60it/s]  5%|▍         | 18285/400000 [00:02<01:28, 4323.71it/s]  5%|▍         | 19282/400000 [00:02<01:13, 5208.31it/s]  5%|▌         | 20233/400000 [00:02<01:03, 6020.02it/s]  5%|▌         | 21225/400000 [00:02<00:55, 6824.24it/s]  6%|▌         | 22216/400000 [00:02<00:50, 7525.22it/s]  6%|▌         | 23187/400000 [00:02<00:47, 8016.70it/s]  6%|▌         | 24160/400000 [00:02<00:44, 8463.61it/s]  6%|▋         | 25144/400000 [00:02<00:42, 8833.24it/s]  7%|▋         | 26116/400000 [00:02<00:41, 8964.62it/s]  7%|▋         | 27097/400000 [00:03<00:40, 9200.24it/s]  7%|▋         | 28074/400000 [00:03<00:39, 9362.30it/s]  7%|▋         | 29043/400000 [00:03<00:39, 9379.10it/s]  8%|▊         | 30027/400000 [00:03<00:38, 9510.66it/s]  8%|▊         | 30995/400000 [00:03<00:39, 9387.60it/s]  8%|▊         | 31946/400000 [00:03<00:39, 9327.07it/s]  8%|▊         | 32894/400000 [00:03<00:39, 9370.55it/s]  8%|▊         | 33861/400000 [00:03<00:38, 9456.57it/s]  9%|▊         | 34863/400000 [00:03<00:37, 9617.47it/s]  9%|▉         | 35855/400000 [00:03<00:37, 9705.95it/s]  9%|▉         | 36844/400000 [00:04<00:37, 9760.07it/s]  9%|▉         | 37823/400000 [00:04<00:37, 9685.70it/s] 10%|▉         | 38817/400000 [00:04<00:37, 9759.69it/s] 10%|▉         | 39801/400000 [00:04<00:36, 9781.53it/s] 10%|█         | 40781/400000 [00:04<00:36, 9727.56it/s] 10%|█         | 41755/400000 [00:04<00:36, 9704.08it/s] 11%|█         | 42726/400000 [00:04<00:37, 9628.35it/s] 11%|█         | 43700/400000 [00:04<00:36, 9660.65it/s] 11%|█         | 44667/400000 [00:04<00:36, 9631.20it/s] 11%|█▏        | 45631/400000 [00:05<00:37, 9479.89it/s] 12%|█▏        | 46582/400000 [00:05<00:37, 9488.50it/s] 12%|█▏        | 47532/400000 [00:05<00:38, 9088.91it/s] 12%|█▏        | 48490/400000 [00:05<00:38, 9228.95it/s] 12%|█▏        | 49466/400000 [00:05<00:37, 9381.48it/s] 13%|█▎        | 50425/400000 [00:05<00:37, 9440.37it/s] 13%|█▎        | 51389/400000 [00:05<00:36, 9498.69it/s] 13%|█▎        | 52341/400000 [00:05<00:36, 9443.43it/s] 13%|█▎        | 53305/400000 [00:05<00:36, 9499.58it/s] 14%|█▎        | 54274/400000 [00:05<00:36, 9554.10it/s] 14%|█▍        | 55231/400000 [00:06<00:36, 9346.60it/s] 14%|█▍        | 56194/400000 [00:06<00:36, 9427.29it/s] 14%|█▍        | 57138/400000 [00:06<00:36, 9349.24it/s] 15%|█▍        | 58112/400000 [00:06<00:36, 9461.54it/s] 15%|█▍        | 59087/400000 [00:06<00:35, 9546.22it/s] 15%|█▌        | 60082/400000 [00:06<00:35, 9661.79it/s] 15%|█▌        | 61082/400000 [00:06<00:34, 9760.65it/s] 16%|█▌        | 62059/400000 [00:06<00:34, 9699.53it/s] 16%|█▌        | 63030/400000 [00:06<00:34, 9639.86it/s] 16%|█▌        | 63995/400000 [00:06<00:35, 9599.66it/s] 16%|█▌        | 64995/400000 [00:07<00:34, 9713.07it/s] 17%|█▋        | 66007/400000 [00:07<00:33, 9830.84it/s] 17%|█▋        | 66991/400000 [00:07<00:34, 9781.56it/s] 17%|█▋        | 67970/400000 [00:07<00:33, 9771.28it/s] 17%|█▋        | 68951/400000 [00:07<00:33, 9780.53it/s] 17%|█▋        | 69950/400000 [00:07<00:33, 9840.86it/s] 18%|█▊        | 70940/400000 [00:07<00:33, 9857.09it/s] 18%|█▊        | 71926/400000 [00:07<00:34, 9605.79it/s] 18%|█▊        | 72889/400000 [00:07<00:34, 9603.48it/s] 18%|█▊        | 73876/400000 [00:07<00:33, 9680.12it/s] 19%|█▊        | 74885/400000 [00:08<00:33, 9797.04it/s] 19%|█▉        | 75876/400000 [00:08<00:33, 9821.23it/s] 19%|█▉        | 76859/400000 [00:08<00:33, 9729.88it/s] 19%|█▉        | 77853/400000 [00:08<00:32, 9790.67it/s] 20%|█▉        | 78833/400000 [00:08<00:32, 9765.66it/s] 20%|█▉        | 79810/400000 [00:08<00:33, 9426.98it/s] 20%|██        | 80756/400000 [00:08<00:34, 9376.27it/s] 20%|██        | 81705/400000 [00:08<00:33, 9409.11it/s] 21%|██        | 82699/400000 [00:08<00:33, 9560.64it/s] 21%|██        | 83696/400000 [00:08<00:32, 9677.19it/s] 21%|██        | 84666/400000 [00:09<00:32, 9619.51it/s] 21%|██▏       | 85638/400000 [00:09<00:32, 9648.10it/s] 22%|██▏       | 86604/400000 [00:09<00:32, 9649.78it/s] 22%|██▏       | 87586/400000 [00:09<00:32, 9700.07it/s] 22%|██▏       | 88596/400000 [00:09<00:31, 9814.19it/s] 22%|██▏       | 89583/400000 [00:09<00:31, 9828.46it/s] 23%|██▎       | 90571/400000 [00:09<00:31, 9842.56it/s] 23%|██▎       | 91573/400000 [00:09<00:31, 9892.65it/s] 23%|██▎       | 92563/400000 [00:09<00:31, 9786.35it/s] 23%|██▎       | 93563/400000 [00:09<00:31, 9848.46it/s] 24%|██▎       | 94549/400000 [00:10<00:31, 9592.03it/s] 24%|██▍       | 95510/400000 [00:10<00:32, 9511.92it/s] 24%|██▍       | 96502/400000 [00:10<00:31, 9628.14it/s] 24%|██▍       | 97475/400000 [00:10<00:31, 9658.23it/s] 25%|██▍       | 98442/400000 [00:10<00:31, 9620.92it/s] 25%|██▍       | 99444/400000 [00:10<00:30, 9736.38it/s] 25%|██▌       | 100449/400000 [00:10<00:30, 9827.57it/s] 25%|██▌       | 101433/400000 [00:10<00:30, 9812.30it/s] 26%|██▌       | 102415/400000 [00:10<00:31, 9530.45it/s] 26%|██▌       | 103413/400000 [00:11<00:30, 9660.40it/s] 26%|██▌       | 104381/400000 [00:11<00:30, 9660.59it/s] 26%|██▋       | 105380/400000 [00:11<00:30, 9754.99it/s] 27%|██▋       | 106366/400000 [00:11<00:30, 9784.53it/s] 27%|██▋       | 107346/400000 [00:11<00:30, 9611.30it/s] 27%|██▋       | 108334/400000 [00:11<00:30, 9689.34it/s] 27%|██▋       | 109320/400000 [00:11<00:29, 9738.72it/s] 28%|██▊       | 110324/400000 [00:11<00:29, 9824.55it/s] 28%|██▊       | 111308/400000 [00:11<00:30, 9586.00it/s] 28%|██▊       | 112269/400000 [00:11<00:30, 9466.87it/s] 28%|██▊       | 113250/400000 [00:12<00:29, 9566.73it/s] 29%|██▊       | 114258/400000 [00:12<00:29, 9714.58it/s] 29%|██▉       | 115249/400000 [00:12<00:29, 9772.08it/s] 29%|██▉       | 116240/400000 [00:12<00:28, 9809.23it/s] 29%|██▉       | 117222/400000 [00:12<00:29, 9710.86it/s] 30%|██▉       | 118215/400000 [00:12<00:28, 9774.92it/s] 30%|██▉       | 119194/400000 [00:12<00:28, 9757.14it/s] 30%|███       | 120171/400000 [00:12<00:28, 9711.45it/s] 30%|███       | 121173/400000 [00:12<00:28, 9801.94it/s] 31%|███       | 122154/400000 [00:12<00:28, 9748.27it/s] 31%|███       | 123162/400000 [00:13<00:28, 9843.95it/s] 31%|███       | 124147/400000 [00:13<00:28, 9645.53it/s] 31%|███▏      | 125119/400000 [00:13<00:28, 9664.95it/s] 32%|███▏      | 126100/400000 [00:13<00:28, 9704.12it/s] 32%|███▏      | 127072/400000 [00:13<00:28, 9511.44it/s] 32%|███▏      | 128025/400000 [00:13<00:28, 9396.21it/s] 32%|███▏      | 128996/400000 [00:13<00:28, 9487.35it/s] 32%|███▏      | 129946/400000 [00:13<00:28, 9457.55it/s] 33%|███▎      | 130893/400000 [00:13<00:28, 9437.70it/s] 33%|███▎      | 131838/400000 [00:13<00:28, 9424.18it/s] 33%|███▎      | 132797/400000 [00:14<00:28, 9471.40it/s] 33%|███▎      | 133765/400000 [00:14<00:27, 9531.02it/s] 34%|███▎      | 134737/400000 [00:14<00:27, 9584.74it/s] 34%|███▍      | 135726/400000 [00:14<00:27, 9672.46it/s] 34%|███▍      | 136694/400000 [00:14<00:27, 9606.34it/s] 34%|███▍      | 137672/400000 [00:14<00:27, 9657.55it/s] 35%|███▍      | 138668/400000 [00:14<00:26, 9744.11it/s] 35%|███▍      | 139645/400000 [00:14<00:26, 9750.31it/s] 35%|███▌      | 140644/400000 [00:14<00:26, 9820.14it/s] 35%|███▌      | 141627/400000 [00:14<00:26, 9677.79it/s] 36%|███▌      | 142596/400000 [00:15<00:26, 9609.94it/s] 36%|███▌      | 143558/400000 [00:15<00:26, 9597.95it/s] 36%|███▌      | 144565/400000 [00:15<00:26, 9734.22it/s] 36%|███▋      | 145545/400000 [00:15<00:26, 9752.05it/s] 37%|███▋      | 146521/400000 [00:15<00:26, 9659.69it/s] 37%|███▋      | 147519/400000 [00:15<00:25, 9753.10it/s] 37%|███▋      | 148527/400000 [00:15<00:25, 9848.39it/s] 37%|███▋      | 149532/400000 [00:15<00:25, 9905.50it/s] 38%|███▊      | 150524/400000 [00:15<00:25, 9873.10it/s] 38%|███▊      | 151512/400000 [00:15<00:25, 9612.42it/s] 38%|███▊      | 152511/400000 [00:16<00:25, 9720.73it/s] 38%|███▊      | 153509/400000 [00:16<00:25, 9795.69it/s] 39%|███▊      | 154490/400000 [00:16<00:25, 9754.91it/s] 39%|███▉      | 155497/400000 [00:16<00:24, 9845.97it/s] 39%|███▉      | 156483/400000 [00:16<00:25, 9685.86it/s] 39%|███▉      | 157469/400000 [00:16<00:24, 9734.73it/s] 40%|███▉      | 158444/400000 [00:16<00:25, 9545.38it/s] 40%|███▉      | 159400/400000 [00:16<00:25, 9539.24it/s] 40%|████      | 160387/400000 [00:16<00:24, 9634.68it/s] 40%|████      | 161352/400000 [00:17<00:25, 9518.04it/s] 41%|████      | 162328/400000 [00:17<00:24, 9586.74it/s] 41%|████      | 163317/400000 [00:17<00:24, 9674.57it/s] 41%|████      | 164305/400000 [00:17<00:24, 9734.99it/s] 41%|████▏     | 165288/400000 [00:17<00:24, 9763.14it/s] 42%|████▏     | 166265/400000 [00:17<00:24, 9653.32it/s] 42%|████▏     | 167231/400000 [00:17<00:24, 9626.34it/s] 42%|████▏     | 168212/400000 [00:17<00:23, 9680.65it/s] 42%|████▏     | 169210/400000 [00:17<00:23, 9766.53it/s] 43%|████▎     | 170215/400000 [00:17<00:23, 9849.32it/s] 43%|████▎     | 171201/400000 [00:18<00:23, 9654.82it/s] 43%|████▎     | 172195/400000 [00:18<00:23, 9737.27it/s] 43%|████▎     | 173187/400000 [00:18<00:23, 9789.39it/s] 44%|████▎     | 174167/400000 [00:18<00:23, 9775.11it/s] 44%|████▍     | 175146/400000 [00:18<00:23, 9683.30it/s] 44%|████▍     | 176115/400000 [00:18<00:23, 9419.91it/s] 44%|████▍     | 177059/400000 [00:18<00:24, 9130.49it/s] 45%|████▍     | 178050/400000 [00:18<00:23, 9351.03it/s] 45%|████▍     | 179004/400000 [00:18<00:23, 9406.50it/s] 45%|████▍     | 179948/400000 [00:18<00:23, 9322.21it/s] 45%|████▌     | 180901/400000 [00:19<00:23, 9382.86it/s] 45%|████▌     | 181897/400000 [00:19<00:22, 9548.13it/s] 46%|████▌     | 182885/400000 [00:19<00:22, 9644.83it/s] 46%|████▌     | 183866/400000 [00:19<00:22, 9691.97it/s] 46%|████▌     | 184849/400000 [00:19<00:22, 9732.63it/s] 46%|████▋     | 185824/400000 [00:19<00:22, 9603.11it/s] 47%|████▋     | 186818/400000 [00:19<00:21, 9701.33it/s] 47%|████▋     | 187804/400000 [00:19<00:21, 9747.47it/s] 47%|████▋     | 188780/400000 [00:19<00:21, 9743.67it/s] 47%|████▋     | 189755/400000 [00:19<00:21, 9680.60it/s] 48%|████▊     | 190724/400000 [00:20<00:22, 9451.31it/s] 48%|████▊     | 191671/400000 [00:20<00:22, 9262.50it/s] 48%|████▊     | 192608/400000 [00:20<00:22, 9293.69it/s] 48%|████▊     | 193583/400000 [00:20<00:21, 9423.72it/s] 49%|████▊     | 194543/400000 [00:20<00:21, 9473.33it/s] 49%|████▉     | 195492/400000 [00:20<00:21, 9380.85it/s] 49%|████▉     | 196476/400000 [00:20<00:21, 9511.51it/s] 49%|████▉     | 197429/400000 [00:20<00:21, 9469.55it/s] 50%|████▉     | 198403/400000 [00:20<00:21, 9548.98it/s] 50%|████▉     | 199371/400000 [00:20<00:20, 9586.77it/s] 50%|█████     | 200331/400000 [00:21<00:21, 9489.88it/s] 50%|█████     | 201281/400000 [00:21<00:21, 9423.58it/s] 51%|█████     | 202252/400000 [00:21<00:20, 9506.19it/s] 51%|█████     | 203236/400000 [00:21<00:20, 9603.12it/s] 51%|█████     | 204214/400000 [00:21<00:20, 9653.52it/s] 51%|█████▏    | 205180/400000 [00:21<00:20, 9539.36it/s] 52%|█████▏    | 206135/400000 [00:21<00:20, 9437.99it/s] 52%|█████▏    | 207085/400000 [00:21<00:20, 9455.90it/s] 52%|█████▏    | 208066/400000 [00:21<00:20, 9557.27it/s] 52%|█████▏    | 209037/400000 [00:21<00:19, 9600.29it/s] 52%|█████▏    | 209998/400000 [00:22<00:20, 9380.86it/s] 53%|█████▎    | 210994/400000 [00:22<00:19, 9545.66it/s] 53%|█████▎    | 211992/400000 [00:22<00:19, 9669.25it/s] 53%|█████▎    | 212984/400000 [00:22<00:19, 9740.72it/s] 53%|█████▎    | 213969/400000 [00:22<00:19, 9771.24it/s] 54%|█████▎    | 214948/400000 [00:22<00:19, 9646.54it/s] 54%|█████▍    | 215917/400000 [00:22<00:19, 9657.12it/s] 54%|█████▍    | 216898/400000 [00:22<00:18, 9698.26it/s] 54%|█████▍    | 217872/400000 [00:22<00:18, 9708.46it/s] 55%|█████▍    | 218851/400000 [00:22<00:18, 9731.90it/s] 55%|█████▍    | 219825/400000 [00:23<00:18, 9647.72it/s] 55%|█████▌    | 220811/400000 [00:23<00:18, 9709.49it/s] 55%|█████▌    | 221783/400000 [00:23<00:18, 9611.06it/s] 56%|█████▌    | 222745/400000 [00:23<00:18, 9454.80it/s] 56%|█████▌    | 223731/400000 [00:23<00:18, 9572.53it/s] 56%|█████▌    | 224690/400000 [00:23<00:18, 9551.47it/s] 56%|█████▋    | 225691/400000 [00:23<00:17, 9684.01it/s] 57%|█████▋    | 226671/400000 [00:23<00:17, 9717.73it/s] 57%|█████▋    | 227644/400000 [00:23<00:17, 9699.44it/s] 57%|█████▋    | 228615/400000 [00:24<00:17, 9596.01it/s] 57%|█████▋    | 229576/400000 [00:24<00:17, 9468.34it/s] 58%|█████▊    | 230569/400000 [00:24<00:17, 9600.54it/s] 58%|█████▊    | 231537/400000 [00:24<00:17, 9621.61it/s] 58%|█████▊    | 232531/400000 [00:24<00:17, 9712.71it/s] 58%|█████▊    | 233527/400000 [00:24<00:17, 9783.09it/s] 59%|█████▊    | 234506/400000 [00:24<00:17, 9703.14it/s] 59%|█████▉    | 235488/400000 [00:24<00:16, 9735.29it/s] 59%|█████▉    | 236493/400000 [00:24<00:16, 9827.08it/s] 59%|█████▉    | 237494/400000 [00:24<00:16, 9881.00it/s] 60%|█████▉    | 238483/400000 [00:25<00:16, 9772.85it/s] 60%|█████▉    | 239461/400000 [00:25<00:16, 9556.38it/s] 60%|██████    | 240431/400000 [00:25<00:16, 9596.17it/s] 60%|██████    | 241428/400000 [00:25<00:16, 9702.54it/s] 61%|██████    | 242432/400000 [00:25<00:16, 9799.26it/s] 61%|██████    | 243418/400000 [00:25<00:15, 9816.31it/s] 61%|██████    | 244401/400000 [00:25<00:16, 9604.28it/s] 61%|██████▏   | 245383/400000 [00:25<00:15, 9666.02it/s] 62%|██████▏   | 246377/400000 [00:25<00:15, 9746.27it/s] 62%|██████▏   | 247353/400000 [00:25<00:15, 9724.71it/s] 62%|██████▏   | 248337/400000 [00:26<00:15, 9758.19it/s] 62%|██████▏   | 249314/400000 [00:26<00:15, 9678.44it/s] 63%|██████▎   | 250310/400000 [00:26<00:15, 9759.94it/s] 63%|██████▎   | 251287/400000 [00:26<00:15, 9723.95it/s] 63%|██████▎   | 252274/400000 [00:26<00:15, 9764.59it/s] 63%|██████▎   | 253268/400000 [00:26<00:14, 9816.15it/s] 64%|██████▎   | 254250/400000 [00:26<00:15, 9524.97it/s] 64%|██████▍   | 255205/400000 [00:26<00:15, 9151.86it/s] 64%|██████▍   | 256157/400000 [00:26<00:15, 9256.65it/s] 64%|██████▍   | 257132/400000 [00:26<00:15, 9397.45it/s] 65%|██████▍   | 258127/400000 [00:27<00:14, 9556.22it/s] 65%|██████▍   | 259086/400000 [00:27<00:14, 9498.27it/s] 65%|██████▌   | 260090/400000 [00:27<00:14, 9651.99it/s] 65%|██████▌   | 261060/400000 [00:27<00:14, 9665.40it/s] 66%|██████▌   | 262058/400000 [00:27<00:14, 9756.88it/s] 66%|██████▌   | 263043/400000 [00:27<00:13, 9782.87it/s] 66%|██████▌   | 264023/400000 [00:27<00:14, 9686.53it/s] 66%|██████▌   | 264993/400000 [00:27<00:14, 9627.75it/s] 66%|██████▋   | 265972/400000 [00:27<00:13, 9674.02it/s] 67%|██████▋   | 266971/400000 [00:27<00:13, 9766.21it/s] 67%|██████▋   | 267967/400000 [00:28<00:13, 9822.22it/s] 67%|██████▋   | 268950/400000 [00:28<00:13, 9719.41it/s] 67%|██████▋   | 269923/400000 [00:28<00:13, 9702.77it/s] 68%|██████▊   | 270894/400000 [00:28<00:13, 9659.19it/s] 68%|██████▊   | 271861/400000 [00:28<00:13, 9646.31it/s] 68%|██████▊   | 272834/400000 [00:28<00:13, 9669.78it/s] 68%|██████▊   | 273802/400000 [00:28<00:13, 9531.54it/s] 69%|██████▊   | 274796/400000 [00:28<00:12, 9648.57it/s] 69%|██████▉   | 275791/400000 [00:28<00:12, 9735.95it/s] 69%|██████▉   | 276766/400000 [00:28<00:12, 9707.57it/s] 69%|██████▉   | 277756/400000 [00:29<00:12, 9763.27it/s] 70%|██████▉   | 278733/400000 [00:29<00:12, 9410.21it/s] 70%|██████▉   | 279707/400000 [00:29<00:12, 9504.72it/s] 70%|███████   | 280676/400000 [00:29<00:12, 9557.09it/s] 70%|███████   | 281680/400000 [00:29<00:12, 9696.17it/s] 71%|███████   | 282677/400000 [00:29<00:12, 9775.30it/s] 71%|███████   | 283656/400000 [00:29<00:12, 9655.16it/s] 71%|███████   | 284642/400000 [00:29<00:11, 9713.54it/s] 71%|███████▏  | 285623/400000 [00:29<00:11, 9738.75it/s] 72%|███████▏  | 286598/400000 [00:30<00:12, 9052.81it/s] 72%|███████▏  | 287520/400000 [00:30<00:12, 9102.13it/s] 72%|███████▏  | 288466/400000 [00:30<00:12, 9205.20it/s] 72%|███████▏  | 289427/400000 [00:30<00:11, 9320.83it/s] 73%|███████▎  | 290413/400000 [00:30<00:11, 9474.25it/s] 73%|███████▎  | 291396/400000 [00:30<00:11, 9576.06it/s] 73%|███████▎  | 292398/400000 [00:30<00:11, 9704.80it/s] 73%|███████▎  | 293371/400000 [00:30<00:11, 9581.07it/s] 74%|███████▎  | 294373/400000 [00:30<00:10, 9708.03it/s] 74%|███████▍  | 295346/400000 [00:30<00:10, 9613.00it/s] 74%|███████▍  | 296318/400000 [00:31<00:10, 9643.39it/s] 74%|███████▍  | 297305/400000 [00:31<00:10, 9708.80it/s] 75%|███████▍  | 298277/400000 [00:31<00:10, 9607.98it/s] 75%|███████▍  | 299259/400000 [00:31<00:10, 9667.39it/s] 75%|███████▌  | 300243/400000 [00:31<00:10, 9717.55it/s] 75%|███████▌  | 301237/400000 [00:31<00:10, 9783.11it/s] 76%|███████▌  | 302216/400000 [00:31<00:10, 9515.34it/s] 76%|███████▌  | 303170/400000 [00:31<00:10, 9311.71it/s] 76%|███████▌  | 304151/400000 [00:31<00:10, 9455.43it/s] 76%|███████▋  | 305143/400000 [00:31<00:09, 9586.54it/s] 77%|███████▋  | 306104/400000 [00:32<00:09, 9585.63it/s] 77%|███████▋  | 307092/400000 [00:32<00:09, 9670.25it/s] 77%|███████▋  | 308061/400000 [00:32<00:09, 9559.36it/s] 77%|███████▋  | 309049/400000 [00:32<00:09, 9652.19it/s] 78%|███████▊  | 310030/400000 [00:32<00:09, 9696.80it/s] 78%|███████▊  | 311025/400000 [00:32<00:09, 9770.64it/s] 78%|███████▊  | 312020/400000 [00:32<00:08, 9822.98it/s] 78%|███████▊  | 313003/400000 [00:32<00:09, 9587.34it/s] 78%|███████▊  | 313991/400000 [00:32<00:08, 9671.69it/s] 79%|███████▊  | 314960/400000 [00:32<00:08, 9674.19it/s] 79%|███████▉  | 315965/400000 [00:33<00:08, 9783.75it/s] 79%|███████▉  | 316945/400000 [00:33<00:08, 9768.09it/s] 79%|███████▉  | 317923/400000 [00:33<00:08, 9633.38it/s] 80%|███████▉  | 318888/400000 [00:33<00:08, 9450.42it/s] 80%|███████▉  | 319864/400000 [00:33<00:08, 9540.41it/s] 80%|████████  | 320838/400000 [00:33<00:08, 9597.46it/s] 80%|████████  | 321799/400000 [00:33<00:08, 9586.00it/s] 81%|████████  | 322759/400000 [00:33<00:08, 9577.51it/s] 81%|████████  | 323759/400000 [00:33<00:07, 9700.22it/s] 81%|████████  | 324755/400000 [00:33<00:07, 9775.24it/s] 81%|████████▏ | 325734/400000 [00:34<00:07, 9755.34it/s] 82%|████████▏ | 326710/400000 [00:34<00:07, 9730.82it/s] 82%|████████▏ | 327684/400000 [00:34<00:07, 9567.36it/s] 82%|████████▏ | 328642/400000 [00:34<00:07, 9543.18it/s] 82%|████████▏ | 329624/400000 [00:34<00:07, 9623.54it/s] 83%|████████▎ | 330598/400000 [00:34<00:07, 9658.01it/s] 83%|████████▎ | 331579/400000 [00:34<00:07, 9702.00it/s] 83%|████████▎ | 332550/400000 [00:34<00:07, 9599.90it/s] 83%|████████▎ | 333528/400000 [00:34<00:06, 9651.74it/s] 84%|████████▎ | 334494/400000 [00:35<00:06, 9479.86it/s] 84%|████████▍ | 335453/400000 [00:35<00:06, 9511.23it/s] 84%|████████▍ | 336450/400000 [00:35<00:06, 9643.98it/s] 84%|████████▍ | 337416/400000 [00:35<00:06, 9599.58it/s] 85%|████████▍ | 338387/400000 [00:35<00:06, 9631.37it/s] 85%|████████▍ | 339351/400000 [00:35<00:06, 9597.47it/s] 85%|████████▌ | 340337/400000 [00:35<00:06, 9673.56it/s] 85%|████████▌ | 341345/400000 [00:35<00:05, 9789.41it/s] 86%|████████▌ | 342325/400000 [00:35<00:05, 9619.36it/s] 86%|████████▌ | 343333/400000 [00:35<00:05, 9752.92it/s] 86%|████████▌ | 344339/400000 [00:36<00:05, 9840.63it/s] 86%|████████▋ | 345325/400000 [00:36<00:05, 9839.36it/s] 87%|████████▋ | 346310/400000 [00:36<00:05, 9841.23it/s] 87%|████████▋ | 347295/400000 [00:36<00:05, 9727.98it/s] 87%|████████▋ | 348298/400000 [00:36<00:05, 9816.61it/s] 87%|████████▋ | 349294/400000 [00:36<00:05, 9858.69it/s] 88%|████████▊ | 350281/400000 [00:36<00:05, 9785.52it/s] 88%|████████▊ | 351261/400000 [00:36<00:05, 9688.33it/s] 88%|████████▊ | 352231/400000 [00:36<00:04, 9629.17it/s] 88%|████████▊ | 353227/400000 [00:36<00:04, 9725.75it/s] 89%|████████▊ | 354228/400000 [00:37<00:04, 9807.82it/s] 89%|████████▉ | 355210/400000 [00:37<00:04, 9783.74it/s] 89%|████████▉ | 356205/400000 [00:37<00:04, 9832.21it/s] 89%|████████▉ | 357189/400000 [00:37<00:04, 9719.30it/s] 90%|████████▉ | 358196/400000 [00:37<00:04, 9820.09it/s] 90%|████████▉ | 359179/400000 [00:37<00:04, 9810.41it/s] 90%|█████████ | 360161/400000 [00:37<00:04, 9762.68it/s] 90%|█████████ | 361143/400000 [00:37<00:03, 9778.67it/s] 91%|█████████ | 362122/400000 [00:37<00:03, 9660.45it/s] 91%|█████████ | 363089/400000 [00:37<00:03, 9657.67it/s] 91%|█████████ | 364056/400000 [00:38<00:03, 9592.48it/s] 91%|█████████▏| 365040/400000 [00:38<00:03, 9664.41it/s] 92%|█████████▏| 366016/400000 [00:38<00:03, 9690.37it/s] 92%|█████████▏| 366986/400000 [00:38<00:03, 9378.60it/s] 92%|█████████▏| 367965/400000 [00:38<00:03, 9497.33it/s] 92%|█████████▏| 368933/400000 [00:38<00:03, 9550.39it/s] 92%|█████████▏| 369929/400000 [00:38<00:03, 9667.08it/s] 93%|█████████▎| 370934/400000 [00:38<00:02, 9778.14it/s] 93%|█████████▎| 371914/400000 [00:38<00:02, 9630.63it/s] 93%|█████████▎| 372894/400000 [00:38<00:02, 9679.64it/s] 93%|█████████▎| 373900/400000 [00:39<00:02, 9778.90it/s] 94%|█████████▎| 374894/400000 [00:39<00:02, 9824.25it/s] 94%|█████████▍| 375878/400000 [00:39<00:02, 9791.00it/s] 94%|█████████▍| 376858/400000 [00:39<00:02, 9601.81it/s] 94%|█████████▍| 377820/400000 [00:39<00:02, 9551.80it/s] 95%|█████████▍| 378777/400000 [00:39<00:02, 9549.39it/s] 95%|█████████▍| 379752/400000 [00:39<00:02, 9608.68it/s] 95%|█████████▌| 380714/400000 [00:39<00:02, 9609.38it/s] 95%|█████████▌| 381676/400000 [00:39<00:01, 9467.35it/s] 96%|█████████▌| 382624/400000 [00:39<00:01, 9444.33it/s] 96%|█████████▌| 383590/400000 [00:40<00:01, 9507.08it/s] 96%|█████████▌| 384557/400000 [00:40<00:01, 9553.20it/s] 96%|█████████▋| 385547/400000 [00:40<00:01, 9652.59it/s] 97%|█████████▋| 386513/400000 [00:40<00:01, 9555.72it/s] 97%|█████████▋| 387506/400000 [00:40<00:01, 9664.39it/s] 97%|█████████▋| 388485/400000 [00:40<00:01, 9699.68it/s] 97%|█████████▋| 389456/400000 [00:40<00:01, 9629.29it/s] 98%|█████████▊| 390445/400000 [00:40<00:00, 9704.03it/s] 98%|█████████▊| 391416/400000 [00:40<00:00, 9642.44it/s] 98%|█████████▊| 392412/400000 [00:40<00:00, 9734.82it/s] 98%|█████████▊| 393386/400000 [00:41<00:00, 9633.17it/s] 99%|█████████▊| 394359/400000 [00:41<00:00, 9659.89it/s] 99%|█████████▉| 395330/400000 [00:41<00:00, 9672.13it/s] 99%|█████████▉| 396298/400000 [00:41<00:00, 9628.93it/s] 99%|█████████▉| 397262/400000 [00:41<00:00, 9615.93it/s]100%|█████████▉| 398224/400000 [00:41<00:00, 9356.86it/s]100%|█████████▉| 399208/400000 [00:41<00:00, 9496.03it/s]100%|█████████▉| 399999/400000 [00:41<00:00, 9573.55it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f5a468e8160>, <torchtext.data.dataset.TabularDataset object at 0x7f5a468e82b0>, <torchtext.vocab.Vocab object at 0x7f5a468e81d0>), {}) 


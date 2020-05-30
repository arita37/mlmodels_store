
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7feae4d532f0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7feae4d532f0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7feb56da10d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7feb56da10d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7feaee5359d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7feaee5359d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feb046858c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feb046858c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feb046858c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 28%|██▊       | 2768896/9912422 [00:00<00:00, 27685686.01it/s]9920512it [00:00, 30234029.21it/s]                             
0it [00:00, ?it/s]32768it [00:00, 327737.35it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7719331.56it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 75121.64it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feae70dee80>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feae70dedd8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7feb04685510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7feb04685510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7feb04685510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:36,  1.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:36,  1.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 2/162 [00:00<01:16,  2.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:16,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 3/162 [00:00<00:59,  2.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:59,  2.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:01<00:47,  3.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:01<00:47,  3.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<00:46,  3.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:01<00:36,  4.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<00:36,  4.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<00:36,  4.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:01<00:28,  5.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:28,  5.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:28,  5.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:01<00:23,  6.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:23,  6.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:23,  6.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:01<00:19,  7.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:19,  7.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:19,  7.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:01<00:16,  8.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:16,  8.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:16,  8.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:01<00:14, 10.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:14, 10.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:14, 10.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:02<00:13, 10.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:02<00:11, 12.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:02<00:11, 12.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:02<00:11, 12.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:02<00:11, 12.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:02<00:09, 14.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:02<00:09, 14.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:02<00:09, 14.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:02<00:09, 14.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:02<00:08, 15.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:02<00:08, 15.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:02<00:08, 15.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:02<00:08, 15.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:02<00:07, 17.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:02<00:07, 17.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:02<00:07, 17.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:02<00:07, 17.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:02<00:07, 18.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:02<00:07, 18.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:02<00:06, 18.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:02<00:06, 18.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:02<00:06, 19.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:02<00:06, 19.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:02<00:06, 19.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:02<00:06, 19.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:02<00:06, 20.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:02<00:06, 20.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:02<00:06, 20.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:02<00:06, 20.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:03<00:05, 21.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:03<00:05, 21.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:03<00:05, 21.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:03<00:05, 21.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:03<00:05, 21.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:03<00:04, 24.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:03<00:04, 24.77 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:03<00:04, 24.77 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:03<00:04, 24.77 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:03<00:04, 24.77 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:03<00:04, 24.77 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:03<00:03, 28.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:03<00:03, 28.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:03<00:03, 28.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:03<00:03, 28.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:03<00:03, 28.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:03<00:03, 28.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:03<00:03, 31.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:03<00:03, 31.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:03<00:03, 31.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:03<00:03, 31.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:03<00:03, 31.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:03<00:03, 31.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:03<00:03, 31.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:03<00:02, 36.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:03<00:02, 36.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:03<00:02, 36.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:03<00:02, 36.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:03<00:02, 36.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:03<00:02, 36.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:03<00:02, 36.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:03<00:02, 36.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:03<00:02, 42.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:03<00:02, 42.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:03<00:02, 42.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:03<00:02, 42.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:03<00:02, 42.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:03<00:02, 42.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:03<00:02, 42.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:03<00:02, 42.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:03<00:01, 48.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:03<00:01, 52.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:03<00:01, 52.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:03<00:01, 52.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:03<00:01, 52.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:03<00:01, 52.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:03<00:01, 52.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:03<00:01, 52.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:03<00:01, 52.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:03<00:01, 52.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:03<00:01, 57.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:03<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:03<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:03<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:03<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:03<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:03<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:03<00:01, 60.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:03<00:01, 60.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:03<00:01, 60.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:03<00:01, 60.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:04<00:01, 60.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:04<00:01, 60.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:04<00:01, 60.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:04<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:04<00:00, 62.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:04<00:00, 62.85 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:04<00:00, 62.85 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:04<00:00, 62.85 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:04<00:00, 62.85 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:04<00:00, 62.85 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:04<00:00, 62.85 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:04<00:00, 62.85 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:04<00:00, 62.85 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:04<00:00, 65.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:04<00:00, 65.63 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:04<00:00, 65.63 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:04<00:00, 65.63 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:04<00:00, 65.63 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:04<00:00, 65.63 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:04<00:00, 65.63 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:04<00:00, 65.63 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:04<00:00, 64.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:04<00:00, 64.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:04<00:00, 64.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:04<00:00, 64.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:04<00:00, 64.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:04<00:00, 64.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:04<00:00, 64.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:04<00:00, 64.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:04<00:00, 64.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:04<00:00, 66.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:04<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:04<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:04<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:04<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:04<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:04<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:04<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:04<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:04<00:00, 67.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:04<00:00, 67.92 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:04<00:00, 67.92 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:04<00:00, 67.92 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:04<00:00, 67.92 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:04<00:00, 67.92 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:04<00:00, 67.92 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:04<00:00, 67.92 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:04<00:00, 67.92 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:04<00:00, 69.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:04<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:04<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:04<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:04<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:04<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:04<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:04<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:04<00:00, 70.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:04<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:04<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:04<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 70.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 70.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 70.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 70.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 70.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 70.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.90s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 70.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 70.90 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:07<00:00,  7.07s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:07<00:00,  4.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 70.90 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:07<00:00,  7.07s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:07<00:00,  7.07s/ file]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 22.90 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.07s/ url]
0 examples [00:00, ? examples/s]2020-05-30 06:09:05.414360: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-30 06:09:05.418357: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397215000 Hz
2020-05-30 06:09:05.418560: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561d4de4d8b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-30 06:09:05.418576: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
72 examples [00:00, 717.90 examples/s]160 examples [00:00, 759.33 examples/s]247 examples [00:00, 788.11 examples/s]332 examples [00:00, 801.84 examples/s]411 examples [00:00, 798.16 examples/s]493 examples [00:00, 802.59 examples/s]583 examples [00:00, 829.38 examples/s]674 examples [00:00, 850.08 examples/s]764 examples [00:00, 863.24 examples/s]858 examples [00:01, 882.86 examples/s]950 examples [00:01, 892.97 examples/s]1039 examples [00:01, 877.34 examples/s]1126 examples [00:01, 842.73 examples/s]1211 examples [00:01, 815.94 examples/s]1293 examples [00:01, 816.69 examples/s]1381 examples [00:01, 832.51 examples/s]1467 examples [00:01, 840.24 examples/s]1566 examples [00:01, 880.16 examples/s]1668 examples [00:01, 916.08 examples/s]1768 examples [00:02, 937.61 examples/s]1863 examples [00:02, 941.20 examples/s]1962 examples [00:02, 954.96 examples/s]2058 examples [00:02, 954.92 examples/s]2159 examples [00:02, 969.22 examples/s]2257 examples [00:02, 939.52 examples/s]2352 examples [00:02, 915.84 examples/s]2444 examples [00:02, 861.29 examples/s]2532 examples [00:02, 834.11 examples/s]2617 examples [00:03, 820.88 examples/s]2702 examples [00:03, 827.95 examples/s]2786 examples [00:03, 826.97 examples/s]2870 examples [00:03, 820.01 examples/s]2959 examples [00:03, 839.22 examples/s]3057 examples [00:03, 876.34 examples/s]3159 examples [00:03, 914.31 examples/s]3252 examples [00:03, 888.11 examples/s]3342 examples [00:03, 881.17 examples/s]3431 examples [00:03, 869.14 examples/s]3519 examples [00:04, 869.02 examples/s]3617 examples [00:04, 898.39 examples/s]3710 examples [00:04, 906.67 examples/s]3808 examples [00:04, 927.21 examples/s]3906 examples [00:04, 940.63 examples/s]4001 examples [00:04, 934.10 examples/s]4095 examples [00:04, 934.06 examples/s]4194 examples [00:04, 949.85 examples/s]4293 examples [00:04, 955.63 examples/s]4393 examples [00:04, 966.26 examples/s]4491 examples [00:05, 969.39 examples/s]4592 examples [00:05, 979.25 examples/s]4691 examples [00:05, 976.70 examples/s]4789 examples [00:05, 962.21 examples/s]4887 examples [00:05, 962.82 examples/s]4986 examples [00:05, 968.60 examples/s]5087 examples [00:05, 978.55 examples/s]5190 examples [00:05, 991.91 examples/s]5290 examples [00:05, 985.49 examples/s]5389 examples [00:05, 979.80 examples/s]5488 examples [00:06, 973.49 examples/s]5586 examples [00:06, 970.50 examples/s]5687 examples [00:06, 981.04 examples/s]5788 examples [00:06, 987.55 examples/s]5887 examples [00:06, 969.93 examples/s]5985 examples [00:06, 930.86 examples/s]6081 examples [00:06, 937.30 examples/s]6176 examples [00:06, 925.66 examples/s]6276 examples [00:06, 946.09 examples/s]6371 examples [00:07, 890.01 examples/s]6461 examples [00:07, 877.90 examples/s]6558 examples [00:07, 902.29 examples/s]6651 examples [00:07, 909.49 examples/s]6748 examples [00:07, 924.99 examples/s]6844 examples [00:07, 932.28 examples/s]6938 examples [00:07, 914.35 examples/s]7035 examples [00:07, 928.62 examples/s]7133 examples [00:07, 942.91 examples/s]7230 examples [00:07, 949.80 examples/s]7328 examples [00:08, 956.92 examples/s]7424 examples [00:08, 915.60 examples/s]7517 examples [00:08, 884.99 examples/s]7607 examples [00:08, 836.65 examples/s]7692 examples [00:08, 837.38 examples/s]7777 examples [00:08, 829.80 examples/s]7865 examples [00:08, 843.73 examples/s]7965 examples [00:08, 884.00 examples/s]8064 examples [00:08, 912.77 examples/s]8157 examples [00:09, 893.05 examples/s]8247 examples [00:09, 891.69 examples/s]8339 examples [00:09, 897.68 examples/s]8439 examples [00:09, 924.11 examples/s]8539 examples [00:09, 944.11 examples/s]8643 examples [00:09, 969.25 examples/s]8741 examples [00:09, 970.86 examples/s]8839 examples [00:09, 971.61 examples/s]8939 examples [00:09, 979.82 examples/s]9040 examples [00:09, 988.43 examples/s]9139 examples [00:10, 983.29 examples/s]9238 examples [00:10, 941.24 examples/s]9333 examples [00:10, 923.39 examples/s]9428 examples [00:10, 930.61 examples/s]9529 examples [00:10, 952.05 examples/s]9633 examples [00:10, 975.12 examples/s]9732 examples [00:10, 977.80 examples/s]9831 examples [00:10, 947.08 examples/s]9927 examples [00:10, 938.97 examples/s]10022 examples [00:10, 869.64 examples/s]10111 examples [00:11, 864.79 examples/s]10199 examples [00:11, 864.18 examples/s]10303 examples [00:11, 909.19 examples/s]10405 examples [00:11, 938.29 examples/s]10505 examples [00:11, 955.24 examples/s]10607 examples [00:11, 972.87 examples/s]10705 examples [00:11, 941.82 examples/s]10800 examples [00:11, 904.72 examples/s]10892 examples [00:11, 882.00 examples/s]10991 examples [00:12, 909.30 examples/s]11083 examples [00:12, 912.07 examples/s]11182 examples [00:12, 931.62 examples/s]11277 examples [00:12, 932.72 examples/s]11371 examples [00:12, 930.80 examples/s]11465 examples [00:12, 923.29 examples/s]11558 examples [00:12, 904.21 examples/s]11652 examples [00:12, 913.87 examples/s]11744 examples [00:12, 905.81 examples/s]11843 examples [00:12, 928.33 examples/s]11942 examples [00:13, 941.94 examples/s]12037 examples [00:13, 909.70 examples/s]12132 examples [00:13, 919.21 examples/s]12229 examples [00:13, 931.86 examples/s]12323 examples [00:13, 929.47 examples/s]12417 examples [00:13, 904.85 examples/s]12508 examples [00:13, 881.14 examples/s]12598 examples [00:13, 886.24 examples/s]12689 examples [00:13, 890.49 examples/s]12787 examples [00:13, 913.64 examples/s]12887 examples [00:14, 937.71 examples/s]12982 examples [00:14, 914.78 examples/s]13074 examples [00:14, 902.14 examples/s]13165 examples [00:14, 894.92 examples/s]13257 examples [00:14, 899.80 examples/s]13354 examples [00:14, 919.40 examples/s]13447 examples [00:14, 915.48 examples/s]13547 examples [00:14, 937.08 examples/s]13644 examples [00:14, 945.84 examples/s]13741 examples [00:15, 952.46 examples/s]13837 examples [00:15, 933.54 examples/s]13931 examples [00:15, 904.65 examples/s]14028 examples [00:15, 922.14 examples/s]14127 examples [00:15, 938.53 examples/s]14228 examples [00:15, 955.17 examples/s]14327 examples [00:15, 964.16 examples/s]14424 examples [00:15, 945.90 examples/s]14523 examples [00:15, 957.17 examples/s]14620 examples [00:15, 960.35 examples/s]14720 examples [00:16, 970.91 examples/s]14818 examples [00:16, 918.24 examples/s]14911 examples [00:16, 890.73 examples/s]15007 examples [00:16, 910.00 examples/s]15099 examples [00:16, 898.46 examples/s]15197 examples [00:16, 918.61 examples/s]15291 examples [00:16, 922.37 examples/s]15384 examples [00:16, 915.53 examples/s]15476 examples [00:16, 890.01 examples/s]15566 examples [00:16, 886.63 examples/s]15655 examples [00:17, 884.55 examples/s]15755 examples [00:17, 914.87 examples/s]15849 examples [00:17, 919.60 examples/s]15946 examples [00:17, 932.84 examples/s]16051 examples [00:17, 964.84 examples/s]16148 examples [00:17, 963.48 examples/s]16245 examples [00:17, 963.78 examples/s]16342 examples [00:17, 958.07 examples/s]16438 examples [00:17, 957.23 examples/s]16534 examples [00:18, 929.48 examples/s]16630 examples [00:18, 937.05 examples/s]16724 examples [00:18, 917.46 examples/s]16816 examples [00:18, 911.73 examples/s]16908 examples [00:18, 880.73 examples/s]17013 examples [00:18, 923.86 examples/s]17110 examples [00:18, 935.44 examples/s]17205 examples [00:18, 915.53 examples/s]17298 examples [00:18, 911.90 examples/s]17390 examples [00:18, 894.89 examples/s]17480 examples [00:19, 875.80 examples/s]17568 examples [00:19, 872.86 examples/s]17666 examples [00:19, 900.95 examples/s]17761 examples [00:19, 913.26 examples/s]17853 examples [00:19, 908.82 examples/s]17945 examples [00:19, 902.45 examples/s]18036 examples [00:19, 885.54 examples/s]18127 examples [00:19, 891.37 examples/s]18224 examples [00:19, 912.99 examples/s]18327 examples [00:19, 942.92 examples/s]18422 examples [00:20, 937.08 examples/s]18517 examples [00:20, 921.28 examples/s]18611 examples [00:20, 925.67 examples/s]18705 examples [00:20, 926.88 examples/s]18798 examples [00:20, 896.73 examples/s]18889 examples [00:20, 899.71 examples/s]18980 examples [00:20, 852.54 examples/s]19066 examples [00:20, 837.06 examples/s]19151 examples [00:20, 826.98 examples/s]19237 examples [00:21, 834.21 examples/s]19321 examples [00:21, 812.34 examples/s]19419 examples [00:21, 855.77 examples/s]19517 examples [00:21, 889.27 examples/s]19614 examples [00:21, 910.90 examples/s]19706 examples [00:21, 894.37 examples/s]19797 examples [00:21, 872.63 examples/s]19885 examples [00:21, 839.31 examples/s]19970 examples [00:21, 805.40 examples/s]20052 examples [00:22, 743.88 examples/s]20138 examples [00:22, 774.78 examples/s]20224 examples [00:22, 797.53 examples/s]20306 examples [00:22, 803.12 examples/s]20399 examples [00:22, 835.45 examples/s]20500 examples [00:22, 880.64 examples/s]20600 examples [00:22, 912.43 examples/s]20693 examples [00:22, 878.14 examples/s]20782 examples [00:22, 867.43 examples/s]20870 examples [00:22, 863.13 examples/s]20958 examples [00:23, 867.37 examples/s]21050 examples [00:23, 881.34 examples/s]21140 examples [00:23, 886.32 examples/s]21236 examples [00:23, 906.81 examples/s]21332 examples [00:23, 920.94 examples/s]21425 examples [00:23, 920.43 examples/s]21518 examples [00:23, 916.03 examples/s]21615 examples [00:23, 930.80 examples/s]21709 examples [00:23, 923.70 examples/s]21802 examples [00:23, 904.14 examples/s]21893 examples [00:24, 892.41 examples/s]21983 examples [00:24, 871.67 examples/s]22071 examples [00:24, 844.20 examples/s]22156 examples [00:24, 818.98 examples/s]22239 examples [00:24, 789.24 examples/s]22319 examples [00:24, 778.05 examples/s]22399 examples [00:24, 784.00 examples/s]22492 examples [00:24, 821.30 examples/s]22590 examples [00:24, 861.92 examples/s]22693 examples [00:25, 903.27 examples/s]22785 examples [00:25, 890.64 examples/s]22875 examples [00:25, 850.59 examples/s]22962 examples [00:25, 855.83 examples/s]23049 examples [00:25, 836.68 examples/s]23139 examples [00:25, 854.06 examples/s]23225 examples [00:25, 850.91 examples/s]23311 examples [00:25, 829.78 examples/s]23399 examples [00:25, 844.10 examples/s]23492 examples [00:25, 865.62 examples/s]23596 examples [00:26, 908.59 examples/s]23695 examples [00:26, 929.06 examples/s]23794 examples [00:26, 944.59 examples/s]23889 examples [00:26, 914.54 examples/s]23989 examples [00:26, 936.48 examples/s]24088 examples [00:26, 950.48 examples/s]24185 examples [00:26, 954.71 examples/s]24281 examples [00:26, 921.82 examples/s]24374 examples [00:26, 921.71 examples/s]24467 examples [00:27, 907.59 examples/s]24565 examples [00:27, 927.25 examples/s]24666 examples [00:27, 949.13 examples/s]24762 examples [00:27, 918.90 examples/s]24855 examples [00:27, 900.64 examples/s]24946 examples [00:27, 882.56 examples/s]25035 examples [00:27, 869.97 examples/s]25125 examples [00:27, 877.15 examples/s]25221 examples [00:27, 898.63 examples/s]25317 examples [00:27, 916.10 examples/s]25416 examples [00:28, 937.07 examples/s]25511 examples [00:28, 922.54 examples/s]25604 examples [00:28, 912.55 examples/s]25696 examples [00:28, 905.54 examples/s]25787 examples [00:28, 888.01 examples/s]25876 examples [00:28, 878.33 examples/s]25966 examples [00:28, 883.10 examples/s]26055 examples [00:28, 881.61 examples/s]26148 examples [00:28, 895.17 examples/s]26245 examples [00:28, 915.94 examples/s]26343 examples [00:29, 926.69 examples/s]26436 examples [00:29, 897.43 examples/s]26527 examples [00:29, 868.05 examples/s]26615 examples [00:29, 858.76 examples/s]26702 examples [00:29, 853.74 examples/s]26788 examples [00:29, 849.23 examples/s]26880 examples [00:29, 867.08 examples/s]26972 examples [00:29, 880.27 examples/s]27073 examples [00:29, 914.55 examples/s]27166 examples [00:30, 916.99 examples/s]27259 examples [00:30, 877.40 examples/s]27348 examples [00:30, 840.50 examples/s]27433 examples [00:30, 807.46 examples/s]27526 examples [00:30, 838.70 examples/s]27616 examples [00:30, 855.47 examples/s]27710 examples [00:30, 876.65 examples/s]27804 examples [00:30, 894.64 examples/s]27894 examples [00:30, 875.58 examples/s]27983 examples [00:30, 864.21 examples/s]28070 examples [00:31, 841.59 examples/s]28155 examples [00:31, 822.29 examples/s]28245 examples [00:31, 842.73 examples/s]28330 examples [00:31, 839.53 examples/s]28418 examples [00:31, 849.87 examples/s]28507 examples [00:31, 860.75 examples/s]28594 examples [00:31, 851.96 examples/s]28680 examples [00:31, 849.25 examples/s]28772 examples [00:31, 869.11 examples/s]28863 examples [00:32, 878.87 examples/s]28954 examples [00:32, 885.34 examples/s]29045 examples [00:32, 891.91 examples/s]29135 examples [00:32, 848.25 examples/s]29221 examples [00:32, 817.51 examples/s]29304 examples [00:32, 795.05 examples/s]29385 examples [00:32, 778.22 examples/s]29464 examples [00:32, 770.94 examples/s]29542 examples [00:32, 771.39 examples/s]29629 examples [00:32, 796.10 examples/s]29711 examples [00:33, 800.60 examples/s]29802 examples [00:33, 828.93 examples/s]29899 examples [00:33, 864.13 examples/s]29987 examples [00:33, 802.43 examples/s]30069 examples [00:33, 755.89 examples/s]30147 examples [00:33, 758.10 examples/s]30226 examples [00:33, 765.15 examples/s]30310 examples [00:33, 784.08 examples/s]30397 examples [00:33, 806.42 examples/s]30481 examples [00:34, 814.12 examples/s]30566 examples [00:34, 824.09 examples/s]30651 examples [00:34, 829.51 examples/s]30735 examples [00:34, 825.27 examples/s]30822 examples [00:34, 835.81 examples/s]30908 examples [00:34, 841.60 examples/s]31001 examples [00:34, 865.21 examples/s]31092 examples [00:34, 875.81 examples/s]31180 examples [00:34, 876.20 examples/s]31268 examples [00:34, 857.03 examples/s]31354 examples [00:35, 836.81 examples/s]31448 examples [00:35, 865.09 examples/s]31545 examples [00:35, 891.83 examples/s]31643 examples [00:35, 914.25 examples/s]31738 examples [00:35, 923.93 examples/s]31831 examples [00:35, 914.35 examples/s]31923 examples [00:35, 909.97 examples/s]32015 examples [00:35, 911.11 examples/s]32108 examples [00:35, 916.07 examples/s]32200 examples [00:35, 904.84 examples/s]32291 examples [00:36, 874.13 examples/s]32379 examples [00:36, 863.95 examples/s]32466 examples [00:36, 848.03 examples/s]32552 examples [00:36, 826.49 examples/s]32649 examples [00:36, 863.29 examples/s]32736 examples [00:36, 833.95 examples/s]32822 examples [00:36, 841.35 examples/s]32922 examples [00:36, 883.32 examples/s]33023 examples [00:36, 916.13 examples/s]33124 examples [00:37, 941.79 examples/s]33227 examples [00:37, 965.97 examples/s]33327 examples [00:37, 973.01 examples/s]33425 examples [00:37, 962.76 examples/s]33524 examples [00:37, 970.63 examples/s]33622 examples [00:37, 960.03 examples/s]33719 examples [00:37, 918.91 examples/s]33812 examples [00:37, 878.42 examples/s]33901 examples [00:37, 836.40 examples/s]33986 examples [00:37, 800.86 examples/s]34068 examples [00:38, 788.95 examples/s]34155 examples [00:38, 809.29 examples/s]34244 examples [00:38, 830.94 examples/s]34347 examples [00:38, 880.61 examples/s]34452 examples [00:38, 924.21 examples/s]34546 examples [00:38, 919.55 examples/s]34645 examples [00:38, 938.66 examples/s]34740 examples [00:38, 936.15 examples/s]34835 examples [00:38, 924.57 examples/s]34928 examples [00:39, 920.73 examples/s]35021 examples [00:39, 904.79 examples/s]35114 examples [00:39, 909.98 examples/s]35209 examples [00:39, 920.39 examples/s]35308 examples [00:39, 939.36 examples/s]35408 examples [00:39, 954.00 examples/s]35504 examples [00:39, 950.18 examples/s]35600 examples [00:39, 932.88 examples/s]35694 examples [00:39, 910.05 examples/s]35786 examples [00:39, 906.43 examples/s]35887 examples [00:40, 932.88 examples/s]35986 examples [00:40, 946.40 examples/s]36093 examples [00:40, 979.19 examples/s]36196 examples [00:40, 991.11 examples/s]36296 examples [00:40, 992.97 examples/s]36398 examples [00:40, 998.09 examples/s]36498 examples [00:40, 988.71 examples/s]36598 examples [00:40, 951.77 examples/s]36694 examples [00:40, 941.81 examples/s]36789 examples [00:40, 899.55 examples/s]36885 examples [00:41, 916.61 examples/s]36981 examples [00:41, 927.04 examples/s]37080 examples [00:41, 942.88 examples/s]37178 examples [00:41, 951.31 examples/s]37274 examples [00:41, 949.17 examples/s]37370 examples [00:41, 934.01 examples/s]37464 examples [00:41, 908.28 examples/s]37560 examples [00:41, 922.47 examples/s]37659 examples [00:41, 939.71 examples/s]37758 examples [00:42, 951.72 examples/s]37857 examples [00:42, 961.38 examples/s]37954 examples [00:42, 955.73 examples/s]38050 examples [00:42, 950.74 examples/s]38146 examples [00:42, 947.52 examples/s]38245 examples [00:42, 957.93 examples/s]38342 examples [00:42, 959.69 examples/s]38439 examples [00:42, 959.36 examples/s]38537 examples [00:42, 960.58 examples/s]38634 examples [00:42, 958.20 examples/s]38734 examples [00:43, 967.26 examples/s]38831 examples [00:43, 965.56 examples/s]38928 examples [00:43, 922.16 examples/s]39027 examples [00:43, 938.67 examples/s]39122 examples [00:43, 914.50 examples/s]39214 examples [00:43, 912.57 examples/s]39314 examples [00:43, 937.00 examples/s]39409 examples [00:43, 922.14 examples/s]39502 examples [00:43, 923.55 examples/s]39595 examples [00:43, 912.27 examples/s]39687 examples [00:44, 907.00 examples/s]39778 examples [00:44, 892.95 examples/s]39868 examples [00:44, 888.17 examples/s]39957 examples [00:44, 883.05 examples/s]40046 examples [00:44, 841.54 examples/s]40135 examples [00:44, 853.62 examples/s]40221 examples [00:44, 849.53 examples/s]40307 examples [00:44, 821.12 examples/s]40396 examples [00:44, 840.35 examples/s]40493 examples [00:45, 874.75 examples/s]40591 examples [00:45, 902.62 examples/s]40688 examples [00:45, 920.13 examples/s]40782 examples [00:45, 922.42 examples/s]40875 examples [00:45, 918.71 examples/s]40968 examples [00:45, 919.05 examples/s]41061 examples [00:45, 922.18 examples/s]41154 examples [00:45, 895.10 examples/s]41244 examples [00:45, 880.04 examples/s]41333 examples [00:45, 872.00 examples/s]41426 examples [00:46, 888.32 examples/s]41516 examples [00:46, 883.14 examples/s]41611 examples [00:46, 901.06 examples/s]41713 examples [00:46, 933.12 examples/s]41807 examples [00:46, 918.07 examples/s]41900 examples [00:46, 911.57 examples/s]41993 examples [00:46, 916.85 examples/s]42085 examples [00:46, 902.13 examples/s]42183 examples [00:46, 921.99 examples/s]42279 examples [00:46, 931.15 examples/s]42378 examples [00:47, 947.80 examples/s]42473 examples [00:47, 927.21 examples/s]42569 examples [00:47, 935.93 examples/s]42665 examples [00:47, 940.95 examples/s]42762 examples [00:47, 948.26 examples/s]42857 examples [00:47, 948.41 examples/s]42952 examples [00:47, 942.54 examples/s]43047 examples [00:47, 937.85 examples/s]43141 examples [00:47, 908.95 examples/s]43233 examples [00:48, 883.75 examples/s]43322 examples [00:48, 849.45 examples/s]43408 examples [00:48, 818.19 examples/s]43491 examples [00:48, 818.71 examples/s]43592 examples [00:48, 866.77 examples/s]43693 examples [00:48, 905.24 examples/s]43787 examples [00:48, 914.58 examples/s]43887 examples [00:48, 937.18 examples/s]43982 examples [00:48, 927.63 examples/s]44076 examples [00:48, 907.35 examples/s]44168 examples [00:49, 879.00 examples/s]44257 examples [00:49, 837.37 examples/s]44342 examples [00:49, 824.62 examples/s]44437 examples [00:49, 857.38 examples/s]44535 examples [00:49, 888.92 examples/s]44633 examples [00:49, 912.75 examples/s]44726 examples [00:49, 909.06 examples/s]44818 examples [00:49, 894.48 examples/s]44909 examples [00:49, 897.77 examples/s]45000 examples [00:50, 884.32 examples/s]45090 examples [00:50, 886.40 examples/s]45179 examples [00:50, 886.96 examples/s]45275 examples [00:50, 906.87 examples/s]45367 examples [00:50, 910.68 examples/s]45462 examples [00:50, 921.39 examples/s]45566 examples [00:50, 952.47 examples/s]45667 examples [00:50, 968.37 examples/s]45768 examples [00:50, 979.23 examples/s]45868 examples [00:50, 984.96 examples/s]45967 examples [00:51, 985.30 examples/s]46066 examples [00:51, 980.76 examples/s]46170 examples [00:51, 997.68 examples/s]46270 examples [00:51, 993.18 examples/s]46370 examples [00:51, 973.16 examples/s]46468 examples [00:51, 970.46 examples/s]46567 examples [00:51, 975.77 examples/s]46665 examples [00:51, 972.35 examples/s]46765 examples [00:51, 980.01 examples/s]46864 examples [00:51, 982.65 examples/s]46968 examples [00:52, 997.05 examples/s]47071 examples [00:52, 1004.42 examples/s]47172 examples [00:52, 979.00 examples/s] 47271 examples [00:52, 941.07 examples/s]47366 examples [00:52, 912.01 examples/s]47458 examples [00:52, 902.51 examples/s]47549 examples [00:52, 893.26 examples/s]47639 examples [00:52, 859.49 examples/s]47739 examples [00:52, 895.36 examples/s]47835 examples [00:52, 913.74 examples/s]47927 examples [00:53, 898.36 examples/s]48025 examples [00:53, 918.99 examples/s]48118 examples [00:53, 889.25 examples/s]48208 examples [00:53, 870.94 examples/s]48296 examples [00:53, 857.67 examples/s]48383 examples [00:53, 835.01 examples/s]48467 examples [00:53, 831.28 examples/s]48557 examples [00:53, 850.63 examples/s]48651 examples [00:53, 875.13 examples/s]48748 examples [00:54, 901.33 examples/s]48846 examples [00:54, 922.51 examples/s]48939 examples [00:54, 920.30 examples/s]49036 examples [00:54, 934.18 examples/s]49130 examples [00:54, 935.25 examples/s]49228 examples [00:54, 947.37 examples/s]49323 examples [00:54, 944.13 examples/s]49418 examples [00:54, 940.94 examples/s]49516 examples [00:54, 950.79 examples/s]49612 examples [00:54, 926.99 examples/s]49705 examples [00:55, 912.72 examples/s]49797 examples [00:55, 901.92 examples/s]49888 examples [00:55, 882.97 examples/s]49980 examples [00:55, 892.21 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5330/50000 [00:00<00:00, 53295.86 examples/s] 31%|███       | 15431/50000 [00:00<00:00, 62095.00 examples/s] 51%|█████     | 25425/50000 [00:00<00:00, 70052.55 examples/s] 71%|███████   | 35378/50000 [00:00<00:00, 76883.25 examples/s] 94%|█████████▍| 46940/50000 [00:00<00:00, 85473.61 examples/s]                                                               0 examples [00:00, ? examples/s]78 examples [00:00, 772.78 examples/s]166 examples [00:00, 801.69 examples/s]257 examples [00:00, 829.51 examples/s]335 examples [00:00, 811.75 examples/s]422 examples [00:00, 827.19 examples/s]518 examples [00:00, 861.62 examples/s]607 examples [00:00, 867.85 examples/s]697 examples [00:00, 690.99 examples/s]786 examples [00:00, 739.62 examples/s]878 examples [00:01, 784.43 examples/s]967 examples [00:01, 812.17 examples/s]1057 examples [00:01, 835.29 examples/s]1143 examples [00:01, 842.01 examples/s]1229 examples [00:01, 840.51 examples/s]1314 examples [00:01, 834.83 examples/s]1398 examples [00:01, 828.40 examples/s]1482 examples [00:01, 827.59 examples/s]1566 examples [00:01, 799.27 examples/s]1647 examples [00:02, 774.46 examples/s]1729 examples [00:02, 787.06 examples/s]1830 examples [00:02, 842.79 examples/s]1930 examples [00:02, 883.31 examples/s]2031 examples [00:02, 916.72 examples/s]2129 examples [00:02, 934.17 examples/s]2232 examples [00:02, 959.45 examples/s]2334 examples [00:02, 974.87 examples/s]2439 examples [00:02, 994.28 examples/s]2539 examples [00:02, 986.12 examples/s]2639 examples [00:03, 974.70 examples/s]2738 examples [00:03, 976.54 examples/s]2840 examples [00:03, 988.10 examples/s]2940 examples [00:03, 976.61 examples/s]3038 examples [00:03, 927.14 examples/s]3132 examples [00:03, 915.80 examples/s]3225 examples [00:03, 888.66 examples/s]3321 examples [00:03, 906.46 examples/s]3419 examples [00:03, 924.38 examples/s]3518 examples [00:03, 940.70 examples/s]3622 examples [00:04, 966.92 examples/s]3720 examples [00:04, 968.85 examples/s]3820 examples [00:04, 976.11 examples/s]3918 examples [00:04, 968.82 examples/s]4017 examples [00:04, 974.34 examples/s]4115 examples [00:04, 942.54 examples/s]4210 examples [00:04, 914.25 examples/s]4302 examples [00:04, 897.18 examples/s]4393 examples [00:04, 900.82 examples/s]4487 examples [00:05, 910.27 examples/s]4588 examples [00:05, 936.95 examples/s]4683 examples [00:05, 913.38 examples/s]4778 examples [00:05, 921.65 examples/s]4876 examples [00:05, 937.77 examples/s]4976 examples [00:05, 952.92 examples/s]5072 examples [00:05, 937.80 examples/s]5167 examples [00:05, 911.58 examples/s]5259 examples [00:05, 887.68 examples/s]5349 examples [00:05, 887.55 examples/s]5439 examples [00:06, 889.89 examples/s]5529 examples [00:06, 882.97 examples/s]5621 examples [00:06, 891.52 examples/s]5711 examples [00:06, 876.62 examples/s]5799 examples [00:06, 860.66 examples/s]5899 examples [00:06, 897.93 examples/s]5998 examples [00:06, 922.92 examples/s]6098 examples [00:06, 942.87 examples/s]6193 examples [00:06, 938.15 examples/s]6288 examples [00:07, 903.21 examples/s]6379 examples [00:07, 895.94 examples/s]6469 examples [00:07, 895.44 examples/s]6559 examples [00:07, 889.73 examples/s]6649 examples [00:07, 866.13 examples/s]6741 examples [00:07, 881.45 examples/s]6830 examples [00:07, 873.41 examples/s]6918 examples [00:07, 859.00 examples/s]7005 examples [00:07, 843.17 examples/s]7098 examples [00:07, 866.18 examples/s]7191 examples [00:08, 884.38 examples/s]7280 examples [00:08, 874.93 examples/s]7368 examples [00:08, 830.02 examples/s]7452 examples [00:08, 806.63 examples/s]7536 examples [00:08, 815.85 examples/s]7620 examples [00:08, 821.63 examples/s]7715 examples [00:08, 856.23 examples/s]7814 examples [00:08, 890.79 examples/s]7913 examples [00:08, 916.84 examples/s]8013 examples [00:08, 939.15 examples/s]8108 examples [00:09, 915.53 examples/s]8201 examples [00:09, 872.70 examples/s]8290 examples [00:09, 832.47 examples/s]8375 examples [00:09, 826.12 examples/s]8461 examples [00:09, 834.36 examples/s]8553 examples [00:09, 857.83 examples/s]8649 examples [00:09, 885.14 examples/s]8747 examples [00:09, 909.91 examples/s]8839 examples [00:09, 903.83 examples/s]8930 examples [00:10, 889.04 examples/s]9020 examples [00:10, 866.62 examples/s]9108 examples [00:10, 863.82 examples/s]9204 examples [00:10, 889.61 examples/s]9303 examples [00:10, 917.39 examples/s]9404 examples [00:10, 941.21 examples/s]9502 examples [00:10, 950.48 examples/s]9602 examples [00:10, 963.43 examples/s]9699 examples [00:10, 949.93 examples/s]9800 examples [00:10, 965.64 examples/s]9901 examples [00:11, 977.75 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteDK30IO/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteDK30IO/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feb046858c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feb046858c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feb046858c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feb70b28da0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feaede68400>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feb046858c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feb046858c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feb046858c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feae7d45470>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feae7d45320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7feade13d620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7feade13d620> 

  function with postional parmater data_info <function split_train_valid at 0x7feade13d620> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7feade13d730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7feade13d730> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7feade13d730> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=0ff26b7d39db59811b2e7762db9938cdaf7c53f25d9a53653fb6da414e76246b
  Stored in directory: /tmp/pip-ephem-wheel-cache-kf_qoljy/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:17:52, 10.7kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:50:26, 15.1kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:01<11:08:41, 21.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 868k/862M [00:01<7:48:36, 30.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.46M/862M [00:01<5:27:13, 43.7kB/s].vector_cache/glove.6B.zip:   1%|          | 7.06M/862M [00:01<3:48:12, 62.5kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:01<2:38:49, 89.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.4M/862M [00:01<1:50:55, 127kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:01<1:17:12, 182kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.0M/862M [00:01<54:00, 259kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.4M/862M [00:01<37:38, 369kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:01<26:22, 524kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.5M/862M [00:02<18:26, 745kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.1M/862M [00:02<12:58, 1.06MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.2M/862M [00:02<09:06, 1.49MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.2M/862M [00:02<06:34, 2.07MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:02<04:42, 2.87MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.8M/862M [00:03<05:12, 2.59MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<05:33, 2.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<08:14, 1.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:05<06:42, 2.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:05<04:55, 2.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<06:55, 1.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:07<06:24, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.8M/862M [00:07<04:52, 2.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<06:10, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<07:07, 1.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:09<05:35, 2.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:09<04:03, 3.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<11:49, 1.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:11<09:36, 1.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.3M/862M [00:11<07:00, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<07:59, 1.64MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<08:15, 1.59MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:13<06:20, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:13<04:35, 2.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<10:43, 1.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:15<08:51, 1.47MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.5M/862M [00:15<06:31, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<07:36, 1.71MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:17<06:38, 1.96MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.7M/862M [00:17<04:55, 2.64MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<06:31, 1.98MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:19<05:52, 2.20MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.8M/862M [00:19<04:26, 2.91MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<06:08, 2.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:21<06:55, 1.86MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:21<05:23, 2.38MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:21<03:57, 3.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<08:02, 1.59MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:23<06:57, 1.84MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.0M/862M [00:23<05:11, 2.46MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<06:36, 1.93MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<05:55, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<04:27, 2.85MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:06, 2.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:34, 2.27MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<04:12, 3.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:55, 2.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:25, 2.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<04:06, 3.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:50, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:21, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:03, 3.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:47, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:29, 1.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<05:10, 2.41MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:37, 2.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:33, 1.89MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<05:14, 2.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<03:48, 3.23MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<1:30:38, 136kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<1:04:43, 190kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<45:31, 270kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<34:39, 354kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<26:45, 458kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<19:20, 633kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<15:27, 788kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<12:04, 1.01MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<08:44, 1.39MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:56, 1.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:44, 1.38MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:44, 1.79MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<04:50, 2.49MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<49:50, 242kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<36:05, 334kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<25:31, 471kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<20:37, 581kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<15:40, 764kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<11:15, 1.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<10:38, 1.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:52, 1.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<07:24, 1.60MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<05:19, 2.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<10:31, 1.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:34, 1.38MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<06:14, 1.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:07, 1.65MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:11, 1.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:37, 2.54MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:59, 1.95MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:23, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<04:00, 2.91MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:35, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:16, 1.86MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:58, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:20, 2.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<04:55, 2.35MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:44, 3.09MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:18, 2.17MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:03, 1.90MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<04:49, 2.38MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<03:28, 3.28MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<11:04:21, 17.2kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<7:45:58, 24.5kB/s] .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<5:25:43, 35.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<3:49:56, 49.4kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<2:43:14, 69.6kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<1:54:37, 99.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<1:20:14, 141kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<59:49, 189kB/s]  .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<43:00, 262kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<30:19, 371kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<23:47, 472kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<17:48, 630kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<12:43, 880kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<11:30, 969kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<10:19, 1.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<07:47, 1.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:13, 1.53MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<07:25, 1.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:45, 1.92MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<04:07, 2.67MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<1:01:20, 180kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<44:03, 250kB/s]  .vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<31:03, 354kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<24:14, 452kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<18:04, 606kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<12:53, 846kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<11:35, 939kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<09:12, 1.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<06:40, 1.63MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:14, 1.49MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:10, 1.75MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:34, 2.35MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:44, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:06, 2.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:50, 2.79MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:13, 2.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:44, 1.86MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<04:33, 2.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:53, 2.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:40, 1.87MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<04:25, 2.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:14, 3.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:19, 1.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:14, 1.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<04:36, 2.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:40, 1.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:08, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:49, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:02, 2.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:36, 2.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<03:28, 2.99MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:50, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:29, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:22, 2.36MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<03:10, 3.23MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<1:26:08, 119kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<1:01:18, 167kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<43:04, 238kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<32:25, 314kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<24:46, 411kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<17:46, 573kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<12:32, 809kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<13:22, 758kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<10:24, 973kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<07:29, 1.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<07:35, 1.33MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<07:22, 1.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:36, 1.79MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<04:02, 2.48MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<09:05, 1.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<07:22, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<05:24, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:06, 1.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:18, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:50, 2.04MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<03:31, 2.80MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:54, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:50, 1.69MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:19, 2.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:19, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:41, 2.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:31, 2.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:45, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:19, 2.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:16, 2.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:33, 2.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:10, 1.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:02, 2.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<02:57, 3.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<06:19, 1.52MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:23, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:00, 2.38MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:01, 1.89MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:28, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:22, 2.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:35, 2.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:10, 2.26MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<03:09, 2.98MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:25, 2.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:02, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<03:57, 2.36MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<02:50, 3.27MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<30:45, 303kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<22:28, 414kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<15:53, 584kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<13:15, 696kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<11:09, 828kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<08:11, 1.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<05:50, 1.57MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<08:28, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<06:51, 1.34MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:01, 1.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:38, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:47, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<04:26, 2.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:15, 2.78MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:26, 1.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:44, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<03:32, 2.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:33, 1.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:02, 1.78MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<03:58, 2.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:13, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:51, 2.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<02:55, 3.03MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:06, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:47, 2.33MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<02:52, 3.06MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:03, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:37, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:40, 2.37MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:58, 2.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:40, 2.36MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<02:47, 3.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:58, 2.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:31, 1.90MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:32, 2.43MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<02:35, 3.31MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<05:38, 1.51MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:50, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<03:33, 2.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:28, 1.89MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:53, 1.73MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:50, 2.20MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<02:46, 3.04MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<8:08:08, 17.2kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<5:42:16, 24.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<3:59:04, 35.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<2:48:35, 49.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<1:59:39, 69.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<1:24:03, 99.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:30<58:34, 141kB/s]   .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<1:41:02, 81.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<1:11:30, 116kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<50:06, 165kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<36:50, 223kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<27:27, 299kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<19:32, 419kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<13:44, 594kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<13:21, 609kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<10:11, 798kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<07:18, 1.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<06:59, 1.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<06:36, 1.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:02, 1.60MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<03:36, 2.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<59:42, 134kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<42:35, 188kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<29:55, 266kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<22:41, 350kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<17:29, 453kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<12:37, 627kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<10:03, 782kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<07:49, 1.00MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<05:39, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<05:46, 1.35MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:38, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:16, 1.82MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<03:04, 2.51MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<05:59, 1.29MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:59, 1.55MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:39, 2.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<04:20, 1.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:35, 1.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:35, 2.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:44, 2.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:23, 2.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:32, 2.98MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:32, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:00, 1.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:06, 2.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<02:17, 3.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:42, 1.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<04:03, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:00, 2.46MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:49, 1.93MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:14, 1.74MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:17, 2.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<02:23, 3.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:01, 1.45MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<04:17, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:10, 2.29MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:54, 1.85MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:21, 2.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:31, 2.86MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<01:50, 3.89MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<30:14, 237kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<22:36, 317kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<16:09, 443kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<12:23, 574kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<09:23, 757kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<06:43, 1.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<06:20, 1.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<05:52, 1.20MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<04:23, 1.60MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<03:09, 2.21MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<05:57, 1.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<04:53, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<03:35, 1.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:07, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:17, 1.61MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:17, 2.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<02:23, 2.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:13, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:20, 1.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<03:10, 2.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:48, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:02, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:06, 2.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<02:15, 2.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:01, 1.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:12, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<03:05, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:42, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:56, 1.68MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:05, 2.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:12, 2.04MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:54, 2.25MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<02:12, 2.96MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:03, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:47, 2.32MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<02:07, 3.05MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:58, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:23, 1.89MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:39, 2.42MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<01:56, 3.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:29, 1.42MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:48, 1.67MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:48, 2.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:25, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:44, 1.68MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:56, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<02:06, 2.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<23:54, 260kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<17:21, 358kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<12:15, 505kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<09:58, 617kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<08:14, 746kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<06:04, 1.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<05:11, 1.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<04:16, 1.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:07, 1.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:34, 1.68MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:07, 1.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:19, 2.57MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:01, 1.96MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:43, 2.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:02, 2.89MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:49, 2.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:09, 1.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:27, 2.38MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<01:48, 3.23MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:43, 1.56MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:12, 1.81MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:22, 2.43MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:00, 1.91MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:15, 1.76MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:34, 2.23MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:42, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:29, 2.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<01:52, 3.00MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:36, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:57, 1.89MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:21, 2.37MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:31, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:19, 2.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<01:46, 3.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:30, 2.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:53, 1.89MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:15, 2.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<01:37, 3.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:30, 1.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<03:42, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:43, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:08, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:17, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:32, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<01:49, 2.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<06:32, 804kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<05:06, 1.03MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<03:41, 1.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:47, 1.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:42, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:50, 1.82MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:48, 1.82MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:29, 2.05MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<01:51, 2.73MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:24, 2.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:42, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:08, 2.35MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:17, 2.17MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:07, 2.34MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:36, 3.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:15, 2.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:04, 2.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<01:34, 3.11MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:14, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:33, 1.90MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:01, 2.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:10<01:27, 3.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<4:36:51, 17.2kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<3:14:01, 24.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<2:15:12, 35.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<1:35:00, 49.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<1:07:25, 69.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<47:16, 99.2kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<32:52, 141kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<25:38, 181kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<18:24, 252kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<12:55, 356kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<10:03, 454kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<07:57, 573kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<05:45, 791kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<04:03, 1.11MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<05:06, 881kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<04:01, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:54, 1.54MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:03, 1.45MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:30, 1.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:51, 2.36MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:19, 1.88MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:04, 2.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:32, 2.81MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:05, 2.05MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:20, 1.83MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:51, 2.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:58, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:49, 2.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:22, 3.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:55, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:45, 2.36MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:19, 3.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:53, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:44, 2.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:18, 3.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:51, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:07, 1.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:40, 2.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<01:48, 2.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:39, 2.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:15, 3.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:46, 2.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:34, 2.45MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:10, 3.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:38<00:52, 4.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<15:00, 254kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<10:52, 350kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<07:39, 494kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<06:11, 605kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<04:42, 794kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:22, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<03:12, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:59, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:14, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:37, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:20, 1.54MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:00, 1.79MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:29, 2.40MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:51, 1.91MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:00, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:35, 2.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:39, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:30, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:08, 3.02MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:35, 2.14MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:27, 2.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:05, 3.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:32, 2.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:24, 2.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:03, 3.14MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:30, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:23, 2.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:02, 3.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:28, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:39, 1.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:17, 2.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<00:56, 3.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:01, 1.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:44, 1.79MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:17, 2.40MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:36, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:26, 2.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:04, 2.84MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:26, 2.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:37, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:16, 2.33MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:21, 2.16MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:14, 2.34MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<00:55, 3.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:18, 2.18MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:29, 1.91MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:09, 2.44MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:07<00:50, 3.33MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:06, 1.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:45, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:17, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:31, 1.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:37, 1.67MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:16, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:17, 2.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:10, 2.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<00:52, 2.97MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:12, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:22, 1.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:05, 2.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:09, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:20, 1.86MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:02, 2.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<00:45, 3.25MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:42, 1.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:25, 1.70MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<01:03, 2.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:17, 1.84MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:08, 2.08MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<00:50, 2.76MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:07, 2.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:15, 1.83MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:58, 2.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:41, 3.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:06, 1.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:41, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:13, 1.81MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:20, 1.61MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:09, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:50, 2.51MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:05, 1.93MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:58, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:42, 2.89MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:58, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:05, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:50, 2.37MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<00:36, 3.27MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:50, 692kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:11, 895kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:33, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:30, 1.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:14, 1.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:54, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:03, 1.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:06, 1.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:51, 2.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:52, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:47, 2.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:35, 2.92MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:48, 2.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:43, 2.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:32, 3.04MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:45, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<00:51, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:40, 2.39MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:28, 3.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<02:06, 732kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:38, 943kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:09, 1.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:08, 1.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:56, 1.56MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:40, 2.12MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:48, 1.76MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:51, 1.66MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:39, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:39, 2.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:36, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:26, 2.94MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:36, 2.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:41, 1.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<00:32, 2.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:22, 3.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:17, 935kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:01, 1.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:43, 1.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:45, 1.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:38, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:28, 2.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:34, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:37, 1.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:28, 2.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:19, 3.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:57, 1.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:45, 1.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:32, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:35, 1.59MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:36, 1.52MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:28, 1.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:19, 2.71MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<03:20, 258kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<02:24, 355kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<01:39, 501kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:17, 612kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<01:03, 743kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:46, 1.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:37, 1.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:30, 1.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:21, 1.93MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:23, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:24, 1.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:18, 2.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:12, 2.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:42, 822kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:33, 1.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:23, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:22, 1.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:18, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:13, 2.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:14, 1.81MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 2.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:11, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:10, 2.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:07, 2.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:09, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:07, 2.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:04, 3.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:02, 3.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.38MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 2.19MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 2.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 3.12MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<11:20:07,  9.80it/s]  0%|          | 728/400000 [00:00<7:55:30, 13.99it/s]  0%|          | 1518/400000 [00:00<5:32:26, 19.98it/s]  1%|          | 2296/400000 [00:00<3:52:30, 28.51it/s]  1%|          | 3073/400000 [00:00<2:42:41, 40.66it/s]  1%|          | 3891/400000 [00:00<1:53:53, 57.96it/s]  1%|          | 4636/400000 [00:00<1:19:50, 82.53it/s]  1%|▏         | 5319/400000 [00:00<56:06, 117.23it/s]   2%|▏         | 6003/400000 [00:00<39:29, 166.25it/s]  2%|▏         | 6673/400000 [00:01<27:55, 234.80it/s]  2%|▏         | 7328/400000 [00:01<19:49, 330.03it/s]  2%|▏         | 8014/400000 [00:01<14:08, 461.94it/s]  2%|▏         | 8791/400000 [00:01<10:07, 643.50it/s]  2%|▏         | 9553/400000 [00:01<07:20, 887.18it/s]  3%|▎         | 10360/400000 [00:01<05:21, 1210.33it/s]  3%|▎         | 11157/400000 [00:01<03:59, 1623.29it/s]  3%|▎         | 11930/400000 [00:01<03:02, 2127.30it/s]  3%|▎         | 12692/400000 [00:01<02:25, 2654.71it/s]  3%|▎         | 13439/400000 [00:01<01:57, 3290.94it/s]  4%|▎         | 14169/400000 [00:02<01:38, 3929.23it/s]  4%|▎         | 14896/400000 [00:02<01:24, 4538.96it/s]  4%|▍         | 15683/400000 [00:02<01:13, 5198.76it/s]  4%|▍         | 16490/400000 [00:02<01:05, 5818.59it/s]  4%|▍         | 17273/400000 [00:02<01:00, 6302.88it/s]  5%|▍         | 18083/400000 [00:02<00:56, 6751.94it/s]  5%|▍         | 18876/400000 [00:02<00:53, 7066.79it/s]  5%|▍         | 19660/400000 [00:02<00:54, 7041.70it/s]  5%|▌         | 20419/400000 [00:02<00:56, 6670.66it/s]  5%|▌         | 21128/400000 [00:03<00:56, 6649.96it/s]  5%|▌         | 21900/400000 [00:03<00:54, 6937.83it/s]  6%|▌         | 22645/400000 [00:03<00:53, 7081.90it/s]  6%|▌         | 23371/400000 [00:03<00:55, 6817.90it/s]  6%|▌         | 24068/400000 [00:03<00:56, 6701.65it/s]  6%|▌         | 24749/400000 [00:03<00:56, 6682.58it/s]  6%|▋         | 25425/400000 [00:03<00:56, 6686.14it/s]  7%|▋         | 26190/400000 [00:03<00:53, 6948.28it/s]  7%|▋         | 26905/400000 [00:03<00:53, 7007.09it/s]  7%|▋         | 27667/400000 [00:03<00:51, 7178.35it/s]  7%|▋         | 28472/400000 [00:04<00:50, 7418.22it/s]  7%|▋         | 29237/400000 [00:04<00:49, 7484.25it/s]  8%|▊         | 30040/400000 [00:04<00:48, 7638.87it/s]  8%|▊         | 30845/400000 [00:04<00:47, 7756.88it/s]  8%|▊         | 31624/400000 [00:04<00:48, 7600.95it/s]  8%|▊         | 32397/400000 [00:04<00:48, 7636.42it/s]  8%|▊         | 33163/400000 [00:04<00:48, 7584.91it/s]  8%|▊         | 33945/400000 [00:04<00:47, 7639.86it/s]  9%|▊         | 34711/400000 [00:04<00:47, 7629.55it/s]  9%|▉         | 35475/400000 [00:04<00:47, 7619.83it/s]  9%|▉         | 36275/400000 [00:05<00:47, 7729.90it/s]  9%|▉         | 37070/400000 [00:05<00:46, 7794.04it/s]  9%|▉         | 37870/400000 [00:05<00:46, 7853.28it/s] 10%|▉         | 38664/400000 [00:05<00:45, 7879.04it/s] 10%|▉         | 39453/400000 [00:05<00:46, 7789.99it/s] 10%|█         | 40233/400000 [00:05<00:46, 7751.32it/s] 10%|█         | 41009/400000 [00:05<00:46, 7716.88it/s] 10%|█         | 41782/400000 [00:05<00:47, 7569.73it/s] 11%|█         | 42550/400000 [00:05<00:47, 7599.50it/s] 11%|█         | 43322/400000 [00:05<00:46, 7634.12it/s] 11%|█         | 44111/400000 [00:06<00:46, 7697.04it/s] 11%|█         | 44882/400000 [00:06<00:49, 7190.90it/s] 11%|█▏        | 45609/400000 [00:06<00:50, 7010.27it/s] 12%|█▏        | 46316/400000 [00:06<00:51, 6813.24it/s] 12%|█▏        | 47052/400000 [00:06<00:50, 6966.51it/s] 12%|█▏        | 47866/400000 [00:06<00:48, 7278.98it/s] 12%|█▏        | 48622/400000 [00:06<00:47, 7358.55it/s] 12%|█▏        | 49363/400000 [00:06<00:51, 6853.37it/s] 13%|█▎        | 50059/400000 [00:06<00:50, 6880.54it/s] 13%|█▎        | 50801/400000 [00:07<00:49, 7031.98it/s] 13%|█▎        | 51592/400000 [00:07<00:47, 7274.17it/s] 13%|█▎        | 52384/400000 [00:07<00:46, 7455.01it/s] 13%|█▎        | 53186/400000 [00:07<00:45, 7614.30it/s] 13%|█▎        | 53953/400000 [00:07<00:46, 7440.49it/s] 14%|█▎        | 54702/400000 [00:07<00:49, 6998.01it/s] 14%|█▍        | 55411/400000 [00:07<00:51, 6688.07it/s] 14%|█▍        | 56089/400000 [00:07<00:53, 6444.60it/s] 14%|█▍        | 56748/400000 [00:07<00:52, 6487.49it/s] 14%|█▍        | 57488/400000 [00:07<00:50, 6735.92it/s] 15%|█▍        | 58169/400000 [00:08<00:52, 6551.39it/s] 15%|█▍        | 58830/400000 [00:08<00:54, 6290.62it/s] 15%|█▍        | 59466/400000 [00:08<00:56, 6001.22it/s] 15%|█▌        | 60151/400000 [00:08<00:54, 6231.66it/s] 15%|█▌        | 60902/400000 [00:08<00:51, 6566.19it/s] 15%|█▌        | 61652/400000 [00:08<00:49, 6820.93it/s] 16%|█▌        | 62418/400000 [00:08<00:47, 7051.06it/s] 16%|█▌        | 63132/400000 [00:08<00:49, 6866.93it/s] 16%|█▌        | 63826/400000 [00:08<00:50, 6636.92it/s] 16%|█▌        | 64588/400000 [00:09<00:48, 6903.81it/s] 16%|█▋        | 65384/400000 [00:09<00:46, 7188.73it/s] 17%|█▋        | 66177/400000 [00:09<00:45, 7395.02it/s] 17%|█▋        | 66968/400000 [00:09<00:44, 7533.42it/s] 17%|█▋        | 67773/400000 [00:09<00:43, 7680.40it/s] 17%|█▋        | 68546/400000 [00:09<00:45, 7250.30it/s] 17%|█▋        | 69280/400000 [00:09<00:47, 6977.25it/s] 17%|█▋        | 69998/400000 [00:09<00:46, 7036.30it/s] 18%|█▊        | 70798/400000 [00:09<00:45, 7299.90it/s] 18%|█▊        | 71575/400000 [00:09<00:44, 7432.41it/s] 18%|█▊        | 72324/400000 [00:10<00:44, 7377.35it/s] 18%|█▊        | 73128/400000 [00:10<00:43, 7562.14it/s] 18%|█▊        | 73931/400000 [00:10<00:42, 7695.96it/s] 19%|█▊        | 74740/400000 [00:10<00:41, 7808.34it/s] 19%|█▉        | 75550/400000 [00:10<00:41, 7892.91it/s] 19%|█▉        | 76342/400000 [00:10<00:42, 7661.82it/s] 19%|█▉        | 77121/400000 [00:10<00:41, 7698.63it/s] 19%|█▉        | 77893/400000 [00:10<00:43, 7440.72it/s] 20%|█▉        | 78721/400000 [00:10<00:41, 7664.93it/s] 20%|█▉        | 79519/400000 [00:11<00:41, 7751.80it/s] 20%|██        | 80298/400000 [00:11<00:41, 7665.68it/s] 20%|██        | 81067/400000 [00:11<00:41, 7631.14it/s] 20%|██        | 81832/400000 [00:11<00:43, 7396.13it/s] 21%|██        | 82586/400000 [00:11<00:42, 7435.89it/s] 21%|██        | 83364/400000 [00:11<00:42, 7535.71it/s] 21%|██        | 84153/400000 [00:11<00:41, 7637.66it/s] 21%|██        | 84970/400000 [00:11<00:40, 7789.70it/s] 21%|██▏       | 85751/400000 [00:11<00:40, 7782.74it/s] 22%|██▏       | 86563/400000 [00:11<00:39, 7879.41it/s] 22%|██▏       | 87362/400000 [00:12<00:39, 7909.84it/s] 22%|██▏       | 88169/400000 [00:12<00:39, 7956.73it/s] 22%|██▏       | 88994/400000 [00:12<00:38, 8040.37it/s] 22%|██▏       | 89799/400000 [00:12<00:39, 7892.65it/s] 23%|██▎       | 90618/400000 [00:12<00:38, 7979.21it/s] 23%|██▎       | 91417/400000 [00:12<00:38, 7965.37it/s] 23%|██▎       | 92215/400000 [00:12<00:41, 7468.52it/s] 23%|██▎       | 92969/400000 [00:12<00:43, 7104.84it/s] 23%|██▎       | 93698/400000 [00:12<00:42, 7157.50it/s] 24%|██▎       | 94420/400000 [00:12<00:42, 7144.71it/s] 24%|██▍       | 95196/400000 [00:13<00:41, 7318.53it/s] 24%|██▍       | 96017/400000 [00:13<00:40, 7562.76it/s] 24%|██▍       | 96844/400000 [00:13<00:39, 7761.67it/s] 24%|██▍       | 97667/400000 [00:13<00:38, 7896.47it/s] 25%|██▍       | 98461/400000 [00:13<00:38, 7785.95it/s] 25%|██▍       | 99243/400000 [00:13<00:41, 7315.42it/s] 25%|██▍       | 99983/400000 [00:13<00:42, 6985.43it/s] 25%|██▌       | 100691/400000 [00:13<00:44, 6782.73it/s] 25%|██▌       | 101414/400000 [00:13<00:43, 6910.65it/s] 26%|██▌       | 102208/400000 [00:14<00:41, 7189.52it/s] 26%|██▌       | 102934/400000 [00:14<00:41, 7161.03it/s] 26%|██▌       | 103680/400000 [00:14<00:40, 7245.35it/s] 26%|██▌       | 104461/400000 [00:14<00:39, 7404.38it/s] 26%|██▋       | 105280/400000 [00:14<00:38, 7622.07it/s] 27%|██▋       | 106047/400000 [00:14<00:39, 7450.94it/s] 27%|██▋       | 106796/400000 [00:14<00:39, 7394.77it/s] 27%|██▋       | 107562/400000 [00:14<00:39, 7471.53it/s] 27%|██▋       | 108368/400000 [00:14<00:38, 7638.68it/s] 27%|██▋       | 109135/400000 [00:14<00:40, 7205.62it/s] 27%|██▋       | 109863/400000 [00:15<00:43, 6714.39it/s] 28%|██▊       | 110546/400000 [00:15<00:44, 6441.34it/s] 28%|██▊       | 111341/400000 [00:15<00:42, 6828.91it/s] 28%|██▊       | 112142/400000 [00:15<00:40, 7142.31it/s] 28%|██▊       | 112947/400000 [00:15<00:38, 7391.40it/s] 28%|██▊       | 113698/400000 [00:15<00:39, 7276.45it/s] 29%|██▊       | 114435/400000 [00:15<00:40, 7031.08it/s] 29%|██▉       | 115146/400000 [00:15<00:41, 6821.05it/s] 29%|██▉       | 115835/400000 [00:15<00:43, 6606.15it/s] 29%|██▉       | 116502/400000 [00:16<00:42, 6595.35it/s] 29%|██▉       | 117185/400000 [00:16<00:42, 6662.78it/s] 29%|██▉       | 117902/400000 [00:16<00:41, 6805.54it/s] 30%|██▉       | 118731/400000 [00:16<00:39, 7190.30it/s] 30%|██▉       | 119540/400000 [00:16<00:37, 7436.61it/s] 30%|███       | 120340/400000 [00:16<00:36, 7596.70it/s] 30%|███       | 121115/400000 [00:16<00:36, 7640.84it/s] 30%|███       | 121884/400000 [00:16<00:36, 7642.74it/s] 31%|███       | 122686/400000 [00:16<00:35, 7749.75it/s] 31%|███       | 123509/400000 [00:16<00:35, 7877.87it/s] 31%|███       | 124300/400000 [00:17<00:34, 7877.47it/s] 31%|███▏      | 125095/400000 [00:17<00:34, 7897.11it/s] 31%|███▏      | 125903/400000 [00:17<00:34, 7950.06it/s] 32%|███▏      | 126699/400000 [00:17<00:35, 7643.37it/s] 32%|███▏      | 127467/400000 [00:17<00:36, 7469.34it/s] 32%|███▏      | 128280/400000 [00:17<00:35, 7654.06it/s] 32%|███▏      | 129049/400000 [00:17<00:37, 7271.19it/s] 32%|███▏      | 129783/400000 [00:17<00:39, 6922.66it/s] 33%|███▎      | 130484/400000 [00:17<00:39, 6869.32it/s] 33%|███▎      | 131177/400000 [00:18<00:40, 6717.85it/s] 33%|███▎      | 131854/400000 [00:18<00:40, 6550.69it/s] 33%|███▎      | 132514/400000 [00:18<00:41, 6469.53it/s] 33%|███▎      | 133221/400000 [00:18<00:40, 6637.47it/s] 34%|███▎      | 134029/400000 [00:18<00:37, 7012.30it/s] 34%|███▎      | 134821/400000 [00:18<00:36, 7260.80it/s] 34%|███▍      | 135609/400000 [00:18<00:35, 7434.88it/s] 34%|███▍      | 136360/400000 [00:18<00:36, 7292.02it/s] 34%|███▍      | 137095/400000 [00:18<00:37, 6941.23it/s] 34%|███▍      | 137797/400000 [00:18<00:38, 6771.33it/s] 35%|███▍      | 138481/400000 [00:19<00:39, 6639.45it/s] 35%|███▍      | 139150/400000 [00:19<00:40, 6492.58it/s] 35%|███▍      | 139804/400000 [00:19<00:40, 6438.86it/s] 35%|███▌      | 140451/400000 [00:19<00:40, 6366.38it/s] 35%|███▌      | 141090/400000 [00:19<00:41, 6308.71it/s] 35%|███▌      | 141723/400000 [00:19<00:41, 6297.69it/s] 36%|███▌      | 142412/400000 [00:19<00:39, 6463.42it/s] 36%|███▌      | 143061/400000 [00:19<00:39, 6426.88it/s] 36%|███▌      | 143746/400000 [00:19<00:39, 6546.36it/s] 36%|███▌      | 144529/400000 [00:20<00:37, 6883.89it/s] 36%|███▋      | 145364/400000 [00:20<00:35, 7265.61it/s] 37%|███▋      | 146118/400000 [00:20<00:34, 7344.31it/s] 37%|███▋      | 146949/400000 [00:20<00:33, 7609.57it/s] 37%|███▋      | 147718/400000 [00:20<00:35, 7138.19it/s] 37%|███▋      | 148443/400000 [00:20<00:37, 6738.23it/s] 37%|███▋      | 149130/400000 [00:20<00:38, 6442.28it/s] 37%|███▋      | 149810/400000 [00:20<00:38, 6543.10it/s] 38%|███▊      | 150473/400000 [00:20<00:38, 6508.61it/s] 38%|███▊      | 151130/400000 [00:20<00:38, 6490.14it/s] 38%|███▊      | 151904/400000 [00:21<00:36, 6819.84it/s] 38%|███▊      | 152620/400000 [00:21<00:35, 6917.58it/s] 38%|███▊      | 153379/400000 [00:21<00:34, 7104.90it/s] 39%|███▊      | 154171/400000 [00:21<00:33, 7331.21it/s] 39%|███▊      | 154965/400000 [00:21<00:32, 7503.29it/s] 39%|███▉      | 155777/400000 [00:21<00:31, 7676.71it/s] 39%|███▉      | 156549/400000 [00:21<00:32, 7412.14it/s] 39%|███▉      | 157296/400000 [00:21<00:32, 7379.60it/s] 40%|███▉      | 158088/400000 [00:21<00:32, 7531.86it/s] 40%|███▉      | 158865/400000 [00:21<00:31, 7600.75it/s] 40%|███▉      | 159688/400000 [00:22<00:30, 7777.58it/s] 40%|████      | 160499/400000 [00:22<00:30, 7870.72it/s] 40%|████      | 161289/400000 [00:22<00:31, 7633.30it/s] 41%|████      | 162113/400000 [00:22<00:30, 7805.29it/s] 41%|████      | 162903/400000 [00:22<00:30, 7832.05it/s] 41%|████      | 163689/400000 [00:22<00:30, 7795.65it/s] 41%|████      | 164493/400000 [00:22<00:29, 7866.28it/s] 41%|████▏     | 165296/400000 [00:22<00:29, 7913.46it/s] 42%|████▏     | 166131/400000 [00:22<00:29, 8038.91it/s] 42%|████▏     | 166937/400000 [00:23<00:29, 7920.74it/s] 42%|████▏     | 167761/400000 [00:23<00:28, 8012.06it/s] 42%|████▏     | 168576/400000 [00:23<00:28, 8051.89it/s] 42%|████▏     | 169382/400000 [00:23<00:30, 7520.30it/s] 43%|████▎     | 170142/400000 [00:23<00:32, 7176.41it/s] 43%|████▎     | 170869/400000 [00:23<00:33, 6929.98it/s] 43%|████▎     | 171597/400000 [00:23<00:32, 7031.00it/s] 43%|████▎     | 172334/400000 [00:23<00:31, 7128.21it/s] 43%|████▎     | 173160/400000 [00:23<00:30, 7426.11it/s] 43%|████▎     | 173997/400000 [00:23<00:29, 7684.69it/s] 44%|████▎     | 174826/400000 [00:24<00:28, 7855.63it/s] 44%|████▍     | 175618/400000 [00:24<00:28, 7868.41it/s] 44%|████▍     | 176409/400000 [00:24<00:28, 7872.63it/s] 44%|████▍     | 177237/400000 [00:24<00:27, 7989.28it/s] 45%|████▍     | 178074/400000 [00:24<00:27, 8098.46it/s] 45%|████▍     | 178891/400000 [00:24<00:27, 8117.86it/s] 45%|████▍     | 179705/400000 [00:24<00:27, 8037.44it/s] 45%|████▌     | 180510/400000 [00:24<00:28, 7774.36it/s] 45%|████▌     | 181307/400000 [00:24<00:27, 7830.41it/s] 46%|████▌     | 182093/400000 [00:24<00:28, 7679.38it/s] 46%|████▌     | 182863/400000 [00:25<00:29, 7295.91it/s] 46%|████▌     | 183599/400000 [00:25<00:30, 7010.69it/s] 46%|████▌     | 184369/400000 [00:25<00:29, 7203.90it/s] 46%|████▋     | 185194/400000 [00:25<00:28, 7487.61it/s] 46%|████▋     | 185993/400000 [00:25<00:28, 7629.78it/s] 47%|████▋     | 186823/400000 [00:25<00:27, 7818.24it/s] 47%|████▋     | 187650/400000 [00:25<00:26, 7947.78it/s] 47%|████▋     | 188449/400000 [00:25<00:28, 7376.17it/s] 47%|████▋     | 189198/400000 [00:25<00:30, 6851.65it/s] 47%|████▋     | 189899/400000 [00:26<00:31, 6758.00it/s] 48%|████▊     | 190654/400000 [00:26<00:30, 6976.06it/s] 48%|████▊     | 191362/400000 [00:26<00:29, 6985.81it/s] 48%|████▊     | 192068/400000 [00:26<00:29, 6953.77it/s] 48%|████▊     | 192776/400000 [00:26<00:29, 6990.61it/s] 48%|████▊     | 193542/400000 [00:26<00:28, 7176.87it/s] 49%|████▊     | 194264/400000 [00:26<00:29, 6954.39it/s] 49%|████▊     | 194964/400000 [00:26<00:30, 6717.11it/s] 49%|████▉     | 195641/400000 [00:26<00:30, 6620.21it/s] 49%|████▉     | 196307/400000 [00:27<00:31, 6546.14it/s] 49%|████▉     | 197035/400000 [00:27<00:30, 6748.50it/s] 49%|████▉     | 197714/400000 [00:27<00:30, 6654.33it/s] 50%|████▉     | 198383/400000 [00:27<00:30, 6525.22it/s] 50%|████▉     | 199038/400000 [00:27<00:31, 6480.88it/s] 50%|████▉     | 199688/400000 [00:27<00:31, 6429.14it/s] 50%|█████     | 200469/400000 [00:27<00:29, 6788.35it/s] 50%|█████     | 201215/400000 [00:27<00:28, 6976.10it/s] 50%|█████     | 201988/400000 [00:27<00:27, 7184.11it/s] 51%|█████     | 202805/400000 [00:27<00:26, 7453.02it/s] 51%|█████     | 203557/400000 [00:28<00:26, 7349.32it/s] 51%|█████     | 204385/400000 [00:28<00:25, 7605.07it/s] 51%|█████▏    | 205200/400000 [00:28<00:25, 7758.34it/s] 51%|█████▏    | 205981/400000 [00:28<00:25, 7704.62it/s] 52%|█████▏    | 206813/400000 [00:28<00:24, 7878.93it/s] 52%|█████▏    | 207637/400000 [00:28<00:24, 7981.56it/s] 52%|█████▏    | 208438/400000 [00:28<00:24, 7872.08it/s] 52%|█████▏    | 209238/400000 [00:28<00:24, 7908.86it/s] 53%|█████▎    | 210031/400000 [00:28<00:24, 7914.72it/s] 53%|█████▎    | 210860/400000 [00:28<00:23, 8022.66it/s] 53%|█████▎    | 211679/400000 [00:29<00:23, 8069.33it/s] 53%|█████▎    | 212492/400000 [00:29<00:23, 8085.30it/s] 53%|█████▎    | 213304/400000 [00:29<00:23, 8093.35it/s] 54%|█████▎    | 214114/400000 [00:29<00:23, 8028.38it/s] 54%|█████▎    | 214932/400000 [00:29<00:22, 8072.50it/s] 54%|█████▍    | 215740/400000 [00:29<00:23, 7833.45it/s] 54%|█████▍    | 216526/400000 [00:29<00:24, 7370.73it/s] 54%|█████▍    | 217270/400000 [00:29<00:26, 6986.21it/s] 54%|█████▍    | 217978/400000 [00:29<00:26, 6825.41it/s] 55%|█████▍    | 218744/400000 [00:30<00:25, 7054.18it/s] 55%|█████▍    | 219552/400000 [00:30<00:24, 7330.93it/s] 55%|█████▌    | 220347/400000 [00:30<00:23, 7505.25it/s] 55%|█████▌    | 221170/400000 [00:30<00:23, 7707.87it/s] 55%|█████▌    | 221947/400000 [00:30<00:23, 7646.51it/s] 56%|█████▌    | 222762/400000 [00:30<00:22, 7788.47it/s] 56%|█████▌    | 223593/400000 [00:30<00:22, 7937.31it/s] 56%|█████▌    | 224390/400000 [00:30<00:22, 7902.16it/s] 56%|█████▋    | 225189/400000 [00:30<00:22, 7927.07it/s] 56%|█████▋    | 225998/400000 [00:30<00:21, 7973.60it/s] 57%|█████▋    | 226818/400000 [00:31<00:21, 8039.88it/s] 57%|█████▋    | 227623/400000 [00:31<00:22, 7834.84it/s] 57%|█████▋    | 228409/400000 [00:31<00:22, 7788.50it/s] 57%|█████▋    | 229190/400000 [00:31<00:23, 7261.26it/s] 57%|█████▋    | 229925/400000 [00:31<00:23, 7204.15it/s] 58%|█████▊    | 230652/400000 [00:31<00:23, 7190.61it/s] 58%|█████▊    | 231376/400000 [00:31<00:23, 7113.06it/s] 58%|█████▊    | 232188/400000 [00:31<00:22, 7385.77it/s] 58%|█████▊    | 232955/400000 [00:31<00:22, 7468.64it/s] 58%|█████▊    | 233706/400000 [00:31<00:22, 7428.74it/s] 59%|█████▊    | 234525/400000 [00:32<00:21, 7640.99it/s] 59%|█████▉    | 235329/400000 [00:32<00:21, 7754.46it/s] 59%|█████▉    | 236120/400000 [00:32<00:21, 7798.78it/s] 59%|█████▉    | 236902/400000 [00:32<00:22, 7297.40it/s] 59%|█████▉    | 237640/400000 [00:32<00:23, 6834.27it/s] 60%|█████▉    | 238335/400000 [00:32<00:24, 6527.63it/s] 60%|█████▉    | 238999/400000 [00:32<00:25, 6309.16it/s] 60%|█████▉    | 239642/400000 [00:32<00:25, 6344.66it/s] 60%|██████    | 240337/400000 [00:32<00:24, 6514.26it/s] 60%|██████    | 241075/400000 [00:33<00:23, 6751.81it/s] 60%|██████    | 241757/400000 [00:33<00:23, 6650.90it/s] 61%|██████    | 242427/400000 [00:33<00:24, 6562.83it/s] 61%|██████    | 243087/400000 [00:33<00:24, 6531.73it/s] 61%|██████    | 243743/400000 [00:33<00:24, 6433.42it/s] 61%|██████    | 244389/400000 [00:33<00:24, 6378.12it/s] 61%|██████▏   | 245204/400000 [00:33<00:22, 6822.87it/s] 62%|██████▏   | 246025/400000 [00:33<00:21, 7185.15it/s] 62%|██████▏   | 246816/400000 [00:33<00:20, 7387.17it/s] 62%|██████▏   | 247565/400000 [00:33<00:20, 7339.52it/s] 62%|██████▏   | 248306/400000 [00:34<00:21, 7015.57it/s] 62%|██████▏   | 249016/400000 [00:34<00:22, 6767.85it/s] 62%|██████▏   | 249701/400000 [00:34<00:22, 6654.93it/s] 63%|██████▎   | 250515/400000 [00:34<00:21, 7033.45it/s] 63%|██████▎   | 251308/400000 [00:34<00:20, 7279.25it/s] 63%|██████▎   | 252095/400000 [00:34<00:19, 7444.84it/s] 63%|██████▎   | 252847/400000 [00:34<00:19, 7462.60it/s] 63%|██████▎   | 253639/400000 [00:34<00:19, 7593.17it/s] 64%|██████▎   | 254464/400000 [00:34<00:18, 7773.78it/s] 64%|██████▍   | 255256/400000 [00:35<00:18, 7816.18it/s] 64%|██████▍   | 256084/400000 [00:35<00:18, 7949.24it/s] 64%|██████▍   | 256882/400000 [00:35<00:18, 7923.41it/s] 64%|██████▍   | 257677/400000 [00:35<00:18, 7785.41it/s] 65%|██████▍   | 258488/400000 [00:35<00:17, 7879.30it/s] 65%|██████▍   | 259282/400000 [00:35<00:17, 7894.86it/s] 65%|██████▌   | 260073/400000 [00:35<00:18, 7441.64it/s] 65%|██████▌   | 260824/400000 [00:35<00:19, 7079.93it/s] 65%|██████▌   | 261540/400000 [00:35<00:19, 6929.14it/s] 66%|██████▌   | 262349/400000 [00:35<00:19, 7238.87it/s] 66%|██████▌   | 263117/400000 [00:36<00:18, 7364.74it/s] 66%|██████▌   | 263934/400000 [00:36<00:17, 7587.69it/s] 66%|██████▌   | 264752/400000 [00:36<00:17, 7753.77it/s] 66%|██████▋   | 265533/400000 [00:36<00:18, 7321.59it/s] 67%|██████▋   | 266314/400000 [00:36<00:17, 7460.77it/s] 67%|██████▋   | 267130/400000 [00:36<00:17, 7655.41it/s] 67%|██████▋   | 267936/400000 [00:36<00:16, 7771.97it/s] 67%|██████▋   | 268739/400000 [00:36<00:16, 7846.22it/s] 67%|██████▋   | 269543/400000 [00:36<00:16, 7903.27it/s] 68%|██████▊   | 270336/400000 [00:36<00:16, 7793.82it/s] 68%|██████▊   | 271118/400000 [00:37<00:17, 7382.96it/s] 68%|██████▊   | 271917/400000 [00:37<00:16, 7554.82it/s] 68%|██████▊   | 272678/400000 [00:37<00:16, 7517.34it/s] 68%|██████▊   | 273434/400000 [00:37<00:17, 7181.73it/s] 69%|██████▊   | 274158/400000 [00:37<00:18, 6912.23it/s] 69%|██████▊   | 274969/400000 [00:37<00:17, 7232.18it/s] 69%|██████▉   | 275788/400000 [00:37<00:16, 7493.91it/s] 69%|██████▉   | 276605/400000 [00:37<00:16, 7683.64it/s] 69%|██████▉   | 277381/400000 [00:37<00:16, 7586.63it/s] 70%|██████▉   | 278145/400000 [00:38<00:16, 7426.18it/s] 70%|██████▉   | 278958/400000 [00:38<00:15, 7623.50it/s] 70%|██████▉   | 279777/400000 [00:38<00:15, 7784.06it/s] 70%|███████   | 280617/400000 [00:38<00:15, 7958.59it/s] 70%|███████   | 281438/400000 [00:38<00:14, 8028.02it/s] 71%|███████   | 282244/400000 [00:38<00:14, 7998.12it/s] 71%|███████   | 283057/400000 [00:38<00:14, 8035.36it/s] 71%|███████   | 283868/400000 [00:38<00:14, 8056.24it/s] 71%|███████   | 284707/400000 [00:38<00:14, 8152.21it/s] 71%|███████▏  | 285524/400000 [00:38<00:14, 7821.07it/s] 72%|███████▏  | 286310/400000 [00:39<00:15, 7334.08it/s] 72%|███████▏  | 287053/400000 [00:39<00:15, 7340.21it/s] 72%|███████▏  | 287878/400000 [00:39<00:14, 7591.00it/s] 72%|███████▏  | 288683/400000 [00:39<00:14, 7722.86it/s] 72%|███████▏  | 289511/400000 [00:39<00:14, 7879.32it/s] 73%|███████▎  | 290304/400000 [00:39<00:13, 7859.51it/s] 73%|███████▎  | 291093/400000 [00:39<00:14, 7542.47it/s] 73%|███████▎  | 291882/400000 [00:39<00:14, 7641.86it/s] 73%|███████▎  | 292650/400000 [00:39<00:14, 7629.18it/s] 73%|███████▎  | 293466/400000 [00:40<00:13, 7778.49it/s] 74%|███████▎  | 294247/400000 [00:40<00:13, 7759.14it/s] 74%|███████▍  | 295080/400000 [00:40<00:13, 7921.37it/s] 74%|███████▍  | 295878/400000 [00:40<00:13, 7938.20it/s] 74%|███████▍  | 296698/400000 [00:40<00:12, 8014.76it/s] 74%|███████▍  | 297518/400000 [00:40<00:12, 8069.35it/s] 75%|███████▍  | 298326/400000 [00:40<00:12, 7893.59it/s] 75%|███████▍  | 299133/400000 [00:40<00:12, 7945.22it/s] 75%|███████▍  | 299929/400000 [00:40<00:12, 7895.74it/s] 75%|███████▌  | 300755/400000 [00:40<00:12, 7998.67it/s] 75%|███████▌  | 301586/400000 [00:41<00:12, 8087.09it/s] 76%|███████▌  | 302396/400000 [00:41<00:12, 8047.53it/s] 76%|███████▌  | 303223/400000 [00:41<00:11, 8110.95it/s] 76%|███████▌  | 304035/400000 [00:41<00:12, 7477.20it/s] 76%|███████▌  | 304793/400000 [00:41<00:13, 7104.59it/s] 76%|███████▋  | 305564/400000 [00:41<00:12, 7274.16it/s] 77%|███████▋  | 306360/400000 [00:41<00:12, 7464.14it/s] 77%|███████▋  | 307185/400000 [00:41<00:12, 7683.59it/s] 77%|███████▋  | 307990/400000 [00:41<00:11, 7788.18it/s] 77%|███████▋  | 308816/400000 [00:41<00:11, 7922.76it/s] 77%|███████▋  | 309634/400000 [00:42<00:11, 7997.53it/s] 78%|███████▊  | 310437/400000 [00:42<00:11, 7972.29it/s] 78%|███████▊  | 311237/400000 [00:42<00:11, 7929.92it/s] 78%|███████▊  | 312038/400000 [00:42<00:11, 7951.90it/s] 78%|███████▊  | 312845/400000 [00:42<00:10, 7985.52it/s] 78%|███████▊  | 313653/400000 [00:42<00:10, 8010.98it/s] 79%|███████▊  | 314455/400000 [00:42<00:10, 7946.22it/s] 79%|███████▉  | 315252/400000 [00:42<00:10, 7951.26it/s] 79%|███████▉  | 316048/400000 [00:42<00:10, 7898.01it/s] 79%|███████▉  | 316839/400000 [00:42<00:10, 7868.02it/s] 79%|███████▉  | 317633/400000 [00:43<00:10, 7885.79it/s] 80%|███████▉  | 318422/400000 [00:43<00:10, 7816.77it/s] 80%|███████▉  | 319249/400000 [00:43<00:10, 7947.34it/s] 80%|████████  | 320063/400000 [00:43<00:09, 8002.42it/s] 80%|████████  | 320876/400000 [00:43<00:09, 8037.49it/s] 80%|████████  | 321695/400000 [00:43<00:09, 8080.74it/s] 81%|████████  | 322504/400000 [00:43<00:09, 8010.56it/s] 81%|████████  | 323319/400000 [00:43<00:09, 8049.46it/s] 81%|████████  | 324140/400000 [00:43<00:09, 8096.81it/s] 81%|████████  | 324964/400000 [00:43<00:09, 8138.81it/s] 81%|████████▏ | 325779/400000 [00:44<00:09, 8108.37it/s] 82%|████████▏ | 326591/400000 [00:44<00:09, 8059.51it/s] 82%|████████▏ | 327398/400000 [00:44<00:09, 8047.58it/s] 82%|████████▏ | 328220/400000 [00:44<00:08, 8096.14it/s] 82%|████████▏ | 329039/400000 [00:44<00:08, 8122.80it/s] 82%|████████▏ | 329852/400000 [00:44<00:08, 8019.40it/s] 83%|████████▎ | 330655/400000 [00:44<00:09, 7659.07it/s] 83%|████████▎ | 331425/400000 [00:44<00:09, 7244.70it/s] 83%|████████▎ | 332157/400000 [00:44<00:09, 6958.85it/s] 83%|████████▎ | 332861/400000 [00:45<00:09, 6754.09it/s] 83%|████████▎ | 333565/400000 [00:45<00:09, 6836.07it/s] 84%|████████▎ | 334339/400000 [00:45<00:09, 7082.88it/s] 84%|████████▍ | 335133/400000 [00:45<00:08, 7316.24it/s] 84%|████████▍ | 335951/400000 [00:45<00:08, 7553.52it/s] 84%|████████▍ | 336777/400000 [00:45<00:08, 7750.24it/s] 84%|████████▍ | 337577/400000 [00:45<00:07, 7820.09it/s] 85%|████████▍ | 338364/400000 [00:45<00:07, 7806.64it/s] 85%|████████▍ | 339179/400000 [00:45<00:07, 7905.66it/s] 85%|████████▍ | 339972/400000 [00:45<00:07, 7716.65it/s] 85%|████████▌ | 340755/400000 [00:46<00:07, 7747.19it/s] 85%|████████▌ | 341532/400000 [00:46<00:07, 7463.88it/s] 86%|████████▌ | 342282/400000 [00:46<00:07, 7445.15it/s] 86%|████████▌ | 343050/400000 [00:46<00:07, 7512.18it/s] 86%|████████▌ | 343861/400000 [00:46<00:07, 7681.25it/s] 86%|████████▌ | 344687/400000 [00:46<00:07, 7845.76it/s] 86%|████████▋ | 345482/400000 [00:46<00:06, 7874.19it/s] 87%|████████▋ | 346272/400000 [00:46<00:06, 7827.29it/s] 87%|████████▋ | 347066/400000 [00:46<00:06, 7858.81it/s] 87%|████████▋ | 347853/400000 [00:46<00:06, 7797.08it/s] 87%|████████▋ | 348653/400000 [00:47<00:06, 7855.31it/s] 87%|████████▋ | 349464/400000 [00:47<00:06, 7929.27it/s] 88%|████████▊ | 350258/400000 [00:47<00:06, 7857.19it/s] 88%|████████▊ | 351045/400000 [00:47<00:06, 7687.30it/s] 88%|████████▊ | 351815/400000 [00:47<00:06, 7596.18it/s] 88%|████████▊ | 352648/400000 [00:47<00:06, 7800.49it/s] 88%|████████▊ | 353456/400000 [00:47<00:05, 7881.06it/s] 89%|████████▊ | 354246/400000 [00:47<00:06, 7596.79it/s] 89%|████████▉ | 355010/400000 [00:47<00:06, 7338.85it/s] 89%|████████▉ | 355789/400000 [00:48<00:05, 7468.53it/s] 89%|████████▉ | 356604/400000 [00:48<00:05, 7658.85it/s] 89%|████████▉ | 357387/400000 [00:48<00:05, 7707.81it/s] 90%|████████▉ | 358161/400000 [00:48<00:05, 7713.67it/s] 90%|████████▉ | 358987/400000 [00:48<00:05, 7869.36it/s] 90%|████████▉ | 359784/400000 [00:48<00:05, 7898.07it/s] 90%|█████████ | 360576/400000 [00:48<00:05, 7569.31it/s] 90%|█████████ | 361337/400000 [00:48<00:05, 7072.79it/s] 91%|█████████ | 362054/400000 [00:48<00:05, 6858.74it/s] 91%|█████████ | 362825/400000 [00:48<00:05, 7093.67it/s] 91%|█████████ | 363575/400000 [00:49<00:05, 7208.41it/s] 91%|█████████ | 364320/400000 [00:49<00:04, 7277.34it/s] 91%|█████████▏| 365100/400000 [00:49<00:04, 7424.95it/s] 91%|█████████▏| 365915/400000 [00:49<00:04, 7627.48it/s] 92%|█████████▏| 366702/400000 [00:49<00:04, 7694.39it/s] 92%|█████████▏| 367475/400000 [00:49<00:04, 7554.91it/s] 92%|█████████▏| 368257/400000 [00:49<00:04, 7630.93it/s] 92%|█████████▏| 369023/400000 [00:49<00:04, 7415.00it/s] 92%|█████████▏| 369844/400000 [00:49<00:03, 7635.24it/s] 93%|█████████▎| 370661/400000 [00:50<00:03, 7787.72it/s] 93%|█████████▎| 371469/400000 [00:50<00:03, 7871.73it/s] 93%|█████████▎| 372272/400000 [00:50<00:03, 7916.90it/s] 93%|█████████▎| 373066/400000 [00:50<00:03, 7840.12it/s] 93%|█████████▎| 373902/400000 [00:50<00:03, 7989.12it/s] 94%|█████████▎| 374720/400000 [00:50<00:03, 8044.08it/s] 94%|█████████▍| 375551/400000 [00:50<00:03, 8119.49it/s] 94%|█████████▍| 376365/400000 [00:50<00:02, 8016.65it/s] 94%|█████████▍| 377168/400000 [00:50<00:02, 7863.48it/s] 94%|█████████▍| 377977/400000 [00:50<00:02, 7928.01it/s] 95%|█████████▍| 378804/400000 [00:51<00:02, 8027.13it/s] 95%|█████████▍| 379612/400000 [00:51<00:02, 8041.24it/s] 95%|█████████▌| 380417/400000 [00:51<00:02, 7701.87it/s] 95%|█████████▌| 381191/400000 [00:51<00:02, 7659.73it/s] 95%|█████████▌| 381975/400000 [00:51<00:02, 7711.48it/s] 96%|█████████▌| 382749/400000 [00:51<00:02, 7684.22it/s] 96%|█████████▌| 383533/400000 [00:51<00:02, 7728.54it/s] 96%|█████████▌| 384356/400000 [00:51<00:01, 7870.62it/s] 96%|█████████▋| 385145/400000 [00:51<00:01, 7779.26it/s] 96%|█████████▋| 385943/400000 [00:51<00:01, 7836.63it/s] 97%|█████████▋| 386766/400000 [00:52<00:01, 7950.33it/s] 97%|█████████▋| 387577/400000 [00:52<00:01, 7996.34it/s] 97%|█████████▋| 388378/400000 [00:52<00:01, 7802.39it/s] 97%|█████████▋| 389160/400000 [00:52<00:01, 7397.15it/s] 97%|█████████▋| 389979/400000 [00:52<00:01, 7618.09it/s] 98%|█████████▊| 390797/400000 [00:52<00:01, 7776.95it/s] 98%|█████████▊| 391604/400000 [00:52<00:01, 7862.53it/s] 98%|█████████▊| 392397/400000 [00:52<00:00, 7879.70it/s] 98%|█████████▊| 393188/400000 [00:52<00:00, 7703.36it/s] 98%|█████████▊| 393961/400000 [00:52<00:00, 7469.96it/s] 99%|█████████▊| 394712/400000 [00:53<00:00, 7031.16it/s] 99%|█████████▉| 395434/400000 [00:53<00:00, 7084.97it/s] 99%|█████████▉| 396263/400000 [00:53<00:00, 7406.61it/s] 99%|█████████▉| 397012/400000 [00:53<00:00, 7224.31it/s] 99%|█████████▉| 397741/400000 [00:53<00:00, 6944.92it/s]100%|█████████▉| 398443/400000 [00:53<00:00, 6868.20it/s]100%|█████████▉| 399135/400000 [00:53<00:00, 6710.92it/s]100%|█████████▉| 399811/400000 [00:53<00:00, 6673.22it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7423.77it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7feade166198>, <torchtext.data.dataset.TabularDataset object at 0x7feade1662e8>, <torchtext.vocab.Vocab object at 0x7feade166208>), {}) 



  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/f99d563c757626e8c9e87b4375b62b1f0138fa5f', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'f99d563c757626e8c9e87b4375b62b1f0138fa5f', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/f99d563c757626e8c9e87b4375b62b1f0138fa5f

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/f99d563c757626e8c9e87b4375b62b1f0138fa5f

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/f99d563c757626e8c9e87b4375b62b1f0138fa5f

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f157ef482f0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f157ef482f0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f15f0f970d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f15f0f970d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f15887299d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f15887299d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f159e87b8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f159e87b8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f159e87b8c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▊      | 3825664/9912422 [00:00<00:00, 37468325.69it/s]9920512it [00:00, 36102934.73it/s]                             
0it [00:00, ?it/s]32768it [00:00, 436503.98it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 159478.94it/s]1654784it [00:00, 11478969.62it/s]                         
0it [00:00, ?it/s]8192it [00:00, 151084.95it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1587e31ba8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f157f152cc0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f159e87b510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f159e87b510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f159e87b510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:58,  2.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:58,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:57,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:57,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:57,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:56,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:56,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:55,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:39,  3.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:39,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:39,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:38,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:38,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:37,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:37,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:37,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:36,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:25,  5.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:25,  5.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:25,  5.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:24,  5.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:24,  5.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:00<00:11, 10.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:11, 10.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:11, 10.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:11, 10.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:11, 10.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:10, 10.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:10, 10.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:10, 10.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:10, 10.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:10, 10.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:10, 10.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:10, 10.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 14.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 14.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:07, 14.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:07, 14.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:07, 14.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:07, 14.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:07, 14.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:00<00:07, 14.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:00<00:07, 14.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:00<00:06, 14.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:00<00:06, 14.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:00<00:06, 14.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:00<00:05, 19.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:00<00:05, 19.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:00<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 33.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 33.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 33.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 33.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 33.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 33.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 33.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 41.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 41.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 41.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 50.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 60.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 69.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 69.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 69.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 69.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 69.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 69.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 69.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 69.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 69.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 69.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 69.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 69.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 76.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 76.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 76.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 76.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 76.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 76.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 76.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 76.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 76.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 76.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 76.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 76.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 83.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:01<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:01<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:01<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:01<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:01<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:01<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:01<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:01<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:01<00:00, 89.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:01<00:00, 89.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:01<00:00,  1.93s/ url]Dl Completed...: 100%|██████████| 1/1 [00:01<00:00,  1.93s/ url]
Dl Size...: 100%|██████████| 162/162 [00:01<00:00, 89.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:01<00:00,  1.93s/ url]
Dl Size...: 100%|██████████| 162/162 [00:01<00:00, 89.74 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:01<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.06s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  1.93s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 89.74 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.06s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.06s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 39.86 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.06s/ url]
0 examples [00:00, ? examples/s]2020-05-29 06:11:40.310173: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-29 06:11:40.327819: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-05-29 06:11:40.327986: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555fd3cb59b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-29 06:11:40.328001: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
7 examples [00:00, 69.68 examples/s]111 examples [00:00, 96.74 examples/s]213 examples [00:00, 132.80 examples/s]317 examples [00:00, 179.85 examples/s]424 examples [00:00, 239.66 examples/s]517 examples [00:00, 308.31 examples/s]615 examples [00:00, 387.73 examples/s]714 examples [00:00, 473.62 examples/s]811 examples [00:00, 559.38 examples/s]912 examples [00:01, 644.55 examples/s]1014 examples [00:01, 723.39 examples/s]1117 examples [00:01, 793.46 examples/s]1216 examples [00:01, 831.07 examples/s]1322 examples [00:01, 886.75 examples/s]1427 examples [00:01, 927.96 examples/s]1532 examples [00:01, 961.32 examples/s]1640 examples [00:01, 992.65 examples/s]1744 examples [00:01, 979.24 examples/s]1847 examples [00:01, 991.66 examples/s]1953 examples [00:02, 1009.44 examples/s]2060 examples [00:02, 1025.08 examples/s]2167 examples [00:02, 1037.24 examples/s]2272 examples [00:02, 1036.78 examples/s]2377 examples [00:02, 1027.99 examples/s]2481 examples [00:02, 985.44 examples/s] 2588 examples [00:02, 1007.81 examples/s]2694 examples [00:02, 1021.97 examples/s]2801 examples [00:02, 1033.85 examples/s]2905 examples [00:02, 1007.51 examples/s]3009 examples [00:03, 1016.04 examples/s]3114 examples [00:03, 1022.80 examples/s]3218 examples [00:03, 1026.26 examples/s]3321 examples [00:03, 1016.72 examples/s]3423 examples [00:03, 1017.42 examples/s]3525 examples [00:03, 983.24 examples/s] 3629 examples [00:03, 998.29 examples/s]3735 examples [00:03, 1014.76 examples/s]3837 examples [00:03, 1005.58 examples/s]3940 examples [00:03, 1011.59 examples/s]4044 examples [00:04, 1018.14 examples/s]4148 examples [00:04, 1023.10 examples/s]4253 examples [00:04, 1028.24 examples/s]4356 examples [00:04, 1023.43 examples/s]4464 examples [00:04, 1039.51 examples/s]4569 examples [00:04, 1041.22 examples/s]4674 examples [00:04, 1014.63 examples/s]4781 examples [00:04, 1028.54 examples/s]4889 examples [00:04, 1042.35 examples/s]4996 examples [00:04, 1048.52 examples/s]5101 examples [00:05, 1038.11 examples/s]5211 examples [00:05, 1053.03 examples/s]5317 examples [00:05, 1040.75 examples/s]5422 examples [00:05, 1009.99 examples/s]5524 examples [00:05, 965.30 examples/s] 5622 examples [00:05, 944.69 examples/s]5717 examples [00:05, 936.74 examples/s]5817 examples [00:05, 953.76 examples/s]5919 examples [00:05, 972.24 examples/s]6023 examples [00:06, 990.22 examples/s]6123 examples [00:06, 986.01 examples/s]6225 examples [00:06, 994.22 examples/s]6325 examples [00:06, 983.69 examples/s]6432 examples [00:06, 1007.95 examples/s]6539 examples [00:06, 1024.52 examples/s]6642 examples [00:06, 1023.52 examples/s]6745 examples [00:06, 1015.10 examples/s]6847 examples [00:06, 984.67 examples/s] 6949 examples [00:06, 992.62 examples/s]7049 examples [00:07, 985.84 examples/s]7150 examples [00:07, 991.43 examples/s]7254 examples [00:07, 1004.38 examples/s]7361 examples [00:07, 1020.82 examples/s]7464 examples [00:07, 993.47 examples/s] 7567 examples [00:07, 1003.00 examples/s]7670 examples [00:07, 1008.83 examples/s]7776 examples [00:07, 1023.05 examples/s]7879 examples [00:07, 991.74 examples/s] 7982 examples [00:07, 1000.49 examples/s]8084 examples [00:08, 1004.53 examples/s]8190 examples [00:08, 1019.82 examples/s]8296 examples [00:08, 1031.21 examples/s]8400 examples [00:08, 1019.37 examples/s]8507 examples [00:08, 1033.48 examples/s]8611 examples [00:08, 971.90 examples/s] 8710 examples [00:08, 940.50 examples/s]8805 examples [00:08, 943.27 examples/s]8905 examples [00:08, 958.60 examples/s]9007 examples [00:09, 975.21 examples/s]9105 examples [00:09, 969.01 examples/s]9203 examples [00:09, 962.00 examples/s]9305 examples [00:09, 977.15 examples/s]9404 examples [00:09, 979.74 examples/s]9503 examples [00:09, 965.97 examples/s]9606 examples [00:09, 981.78 examples/s]9710 examples [00:09, 996.66 examples/s]9810 examples [00:09, 990.78 examples/s]9910 examples [00:09, 980.98 examples/s]10009 examples [00:10, 883.91 examples/s]10114 examples [00:10, 926.24 examples/s]10209 examples [00:10, 924.68 examples/s]10305 examples [00:10, 933.09 examples/s]10411 examples [00:10, 966.21 examples/s]10518 examples [00:10, 995.15 examples/s]10626 examples [00:10, 1018.93 examples/s]10732 examples [00:10, 1030.88 examples/s]10836 examples [00:10, 1032.54 examples/s]10940 examples [00:11, 1005.94 examples/s]11042 examples [00:11, 962.34 examples/s] 11144 examples [00:11, 977.77 examples/s]11249 examples [00:11, 997.08 examples/s]11351 examples [00:11, 1002.61 examples/s]11453 examples [00:11, 1005.92 examples/s]11554 examples [00:11, 972.47 examples/s] 11652 examples [00:11, 970.85 examples/s]11755 examples [00:11, 986.01 examples/s]11854 examples [00:11, 983.04 examples/s]11955 examples [00:12, 990.86 examples/s]12060 examples [00:12, 1007.63 examples/s]12161 examples [00:12, 1007.15 examples/s]12262 examples [00:12, 1003.86 examples/s]12365 examples [00:12, 1009.18 examples/s]12471 examples [00:12, 1023.59 examples/s]12574 examples [00:12, 1015.49 examples/s]12676 examples [00:12, 982.37 examples/s] 12781 examples [00:12, 999.78 examples/s]12889 examples [00:12, 1022.18 examples/s]12997 examples [00:13, 1037.36 examples/s]13102 examples [00:13, 998.29 examples/s] 13206 examples [00:13, 1009.35 examples/s]13314 examples [00:13, 1028.75 examples/s]13420 examples [00:13, 1035.29 examples/s]13524 examples [00:13, 1030.21 examples/s]13628 examples [00:13, 1027.17 examples/s]13734 examples [00:13, 1036.67 examples/s]13840 examples [00:13, 1040.33 examples/s]13945 examples [00:13, 1031.49 examples/s]14052 examples [00:14, 1041.63 examples/s]14157 examples [00:14, 1035.33 examples/s]14261 examples [00:14, 1029.57 examples/s]14365 examples [00:14, 1030.16 examples/s]14469 examples [00:14, 1026.32 examples/s]14572 examples [00:14, 1024.66 examples/s]14675 examples [00:14, 997.47 examples/s] 14782 examples [00:14, 1017.77 examples/s]14889 examples [00:14, 1032.68 examples/s]14993 examples [00:15, 1033.24 examples/s]15097 examples [00:15, 1028.33 examples/s]15201 examples [00:15, 1031.54 examples/s]15305 examples [00:15, 1015.15 examples/s]15410 examples [00:15, 1024.71 examples/s]15513 examples [00:15, 1018.00 examples/s]15619 examples [00:15, 1029.99 examples/s]15726 examples [00:15, 1041.50 examples/s]15833 examples [00:15, 1048.61 examples/s]15938 examples [00:15, 1047.94 examples/s]16043 examples [00:16, 1037.88 examples/s]16147 examples [00:16, 1030.21 examples/s]16251 examples [00:16, 1024.51 examples/s]16356 examples [00:16, 1029.05 examples/s]16459 examples [00:16, 1001.37 examples/s]16561 examples [00:16, 1004.65 examples/s]16663 examples [00:16, 1007.28 examples/s]16767 examples [00:16, 1014.74 examples/s]16869 examples [00:16, 1014.35 examples/s]16972 examples [00:16, 1016.31 examples/s]17074 examples [00:17, 1011.07 examples/s]17176 examples [00:17, 1013.60 examples/s]17278 examples [00:17, 1010.20 examples/s]17381 examples [00:17, 1014.94 examples/s]17485 examples [00:17, 1021.24 examples/s]17588 examples [00:17, 1021.70 examples/s]17691 examples [00:17, 1017.58 examples/s]17793 examples [00:17, 979.30 examples/s] 17896 examples [00:17, 991.56 examples/s]18000 examples [00:17, 1005.45 examples/s]18103 examples [00:18, 1011.35 examples/s]18209 examples [00:18, 1022.88 examples/s]18315 examples [00:18, 1032.84 examples/s]18421 examples [00:18, 1038.80 examples/s]18529 examples [00:18, 1049.83 examples/s]18635 examples [00:18, 1024.33 examples/s]18738 examples [00:18, 1019.55 examples/s]18847 examples [00:18, 1038.22 examples/s]18954 examples [00:18, 1046.84 examples/s]19062 examples [00:18, 1054.26 examples/s]19168 examples [00:19, 1055.55 examples/s]19275 examples [00:19, 1058.43 examples/s]19384 examples [00:19, 1066.58 examples/s]19493 examples [00:19, 1070.58 examples/s]19601 examples [00:19, 1030.68 examples/s]19705 examples [00:19, 1022.83 examples/s]19808 examples [00:19, 1012.19 examples/s]19912 examples [00:19, 1013.90 examples/s]20014 examples [00:19, 964.03 examples/s] 20112 examples [00:20, 950.84 examples/s]20214 examples [00:20, 969.77 examples/s]20316 examples [00:20, 983.33 examples/s]20418 examples [00:20, 993.32 examples/s]20519 examples [00:20, 994.22 examples/s]20619 examples [00:20, 985.17 examples/s]20720 examples [00:20, 990.96 examples/s]20827 examples [00:20, 1012.78 examples/s]20929 examples [00:20, 991.80 examples/s] 21032 examples [00:20, 1001.61 examples/s]21137 examples [00:21, 1013.80 examples/s]21239 examples [00:21, 993.78 examples/s] 21339 examples [00:21, 964.52 examples/s]21436 examples [00:21, 927.74 examples/s]21542 examples [00:21, 962.64 examples/s]21647 examples [00:21, 986.42 examples/s]21752 examples [00:21, 1003.06 examples/s]21857 examples [00:21, 1014.85 examples/s]21961 examples [00:21, 1022.14 examples/s]22064 examples [00:21, 1016.79 examples/s]22167 examples [00:22, 1019.70 examples/s]22275 examples [00:22, 1036.29 examples/s]22381 examples [00:22, 1042.46 examples/s]22488 examples [00:22, 1049.38 examples/s]22596 examples [00:22, 1054.59 examples/s]22702 examples [00:22, 1049.61 examples/s]22808 examples [00:22, 1032.15 examples/s]22913 examples [00:22, 1036.47 examples/s]23017 examples [00:22, 1026.58 examples/s]23120 examples [00:23, 1021.44 examples/s]23223 examples [00:23, 1022.60 examples/s]23326 examples [00:23, 1006.91 examples/s]23427 examples [00:23, 997.67 examples/s] 23527 examples [00:23, 969.96 examples/s]23625 examples [00:23, 969.83 examples/s]23723 examples [00:23, 948.83 examples/s]23823 examples [00:23, 961.53 examples/s]23928 examples [00:23, 983.80 examples/s]24027 examples [00:23, 961.12 examples/s]24128 examples [00:24, 970.86 examples/s]24230 examples [00:24, 984.82 examples/s]24338 examples [00:24, 1010.71 examples/s]24440 examples [00:24, 996.16 examples/s] 24540 examples [00:24, 989.61 examples/s]24646 examples [00:24, 1008.37 examples/s]24748 examples [00:24, 1009.64 examples/s]24850 examples [00:24, 1010.96 examples/s]24955 examples [00:24, 1019.76 examples/s]25061 examples [00:24, 1029.53 examples/s]25165 examples [00:25, 1022.73 examples/s]25268 examples [00:25, 1009.05 examples/s]25375 examples [00:25, 1024.58 examples/s]25483 examples [00:25, 1038.34 examples/s]25588 examples [00:25, 1041.45 examples/s]25693 examples [00:25, 1026.79 examples/s]25798 examples [00:25, 1033.13 examples/s]25904 examples [00:25, 1039.05 examples/s]26011 examples [00:25, 1047.30 examples/s]26118 examples [00:25, 1052.38 examples/s]26224 examples [00:26, 1051.06 examples/s]26330 examples [00:26, 1041.43 examples/s]26435 examples [00:26, 1017.74 examples/s]26540 examples [00:26, 1026.51 examples/s]26646 examples [00:26, 1036.10 examples/s]26753 examples [00:26, 1043.77 examples/s]26858 examples [00:26, 990.00 examples/s] 26958 examples [00:26, 991.53 examples/s]27060 examples [00:26, 998.47 examples/s]27161 examples [00:27, 991.53 examples/s]27261 examples [00:27, 991.66 examples/s]27363 examples [00:27, 998.13 examples/s]27463 examples [00:27, 988.81 examples/s]27565 examples [00:27, 996.94 examples/s]27665 examples [00:27, 985.44 examples/s]27766 examples [00:27, 992.09 examples/s]27868 examples [00:27, 1000.02 examples/s]27977 examples [00:27, 1023.67 examples/s]28082 examples [00:27, 1030.98 examples/s]28186 examples [00:28, 1028.79 examples/s]28290 examples [00:28, 1031.82 examples/s]28395 examples [00:28, 1034.75 examples/s]28499 examples [00:28, 990.68 examples/s] 28599 examples [00:28, 933.60 examples/s]28697 examples [00:28, 944.79 examples/s]28793 examples [00:28, 939.27 examples/s]28897 examples [00:28, 964.98 examples/s]29003 examples [00:28, 990.09 examples/s]29109 examples [00:28, 1008.62 examples/s]29218 examples [00:29, 1030.90 examples/s]29326 examples [00:29, 1042.99 examples/s]29435 examples [00:29, 1054.74 examples/s]29544 examples [00:29, 1063.63 examples/s]29651 examples [00:29, 1039.38 examples/s]29759 examples [00:29, 1049.53 examples/s]29865 examples [00:29, 1049.91 examples/s]29971 examples [00:29, 1047.86 examples/s]30076 examples [00:29, 959.48 examples/s] 30178 examples [00:30, 972.99 examples/s]30277 examples [00:30, 975.20 examples/s]30381 examples [00:30, 993.05 examples/s]30481 examples [00:30, 966.41 examples/s]30579 examples [00:30, 964.84 examples/s]30676 examples [00:30, 953.96 examples/s]30778 examples [00:30, 972.67 examples/s]30880 examples [00:30, 985.96 examples/s]30983 examples [00:30, 997.73 examples/s]31083 examples [00:30, 997.61 examples/s]31191 examples [00:31, 1020.10 examples/s]31294 examples [00:31, 1011.02 examples/s]31398 examples [00:31, 1019.45 examples/s]31501 examples [00:31, 1012.65 examples/s]31606 examples [00:31, 1021.69 examples/s]31714 examples [00:31, 1035.99 examples/s]31818 examples [00:31, 1033.47 examples/s]31923 examples [00:31, 1036.71 examples/s]32027 examples [00:31, 1033.66 examples/s]32133 examples [00:31, 1039.80 examples/s]32238 examples [00:32, 1036.46 examples/s]32342 examples [00:32, 1032.98 examples/s]32446 examples [00:32, 1027.10 examples/s]32550 examples [00:32, 1029.83 examples/s]32658 examples [00:32, 1042.79 examples/s]32763 examples [00:32, 1035.85 examples/s]32867 examples [00:32, 1021.01 examples/s]32970 examples [00:32, 1011.84 examples/s]33072 examples [00:32, 998.67 examples/s] 33174 examples [00:32, 1003.16 examples/s]33275 examples [00:33, 986.45 examples/s] 33374 examples [00:33, 920.64 examples/s]33468 examples [00:33, 891.32 examples/s]33572 examples [00:33, 930.19 examples/s]33668 examples [00:33, 936.01 examples/s]33763 examples [00:33, 930.65 examples/s]33857 examples [00:33, 924.89 examples/s]33958 examples [00:33, 947.10 examples/s]34054 examples [00:33, 932.91 examples/s]34159 examples [00:34, 965.04 examples/s]34266 examples [00:34, 991.76 examples/s]34366 examples [00:34, 972.44 examples/s]34470 examples [00:34, 989.87 examples/s]34575 examples [00:34, 1006.67 examples/s]34680 examples [00:34, 1017.53 examples/s]34784 examples [00:34, 1021.12 examples/s]34887 examples [00:34, 1019.10 examples/s]34990 examples [00:34, 1010.82 examples/s]35092 examples [00:34, 1012.63 examples/s]35196 examples [00:35, 1020.22 examples/s]35299 examples [00:35, 1018.18 examples/s]35401 examples [00:35, 1003.47 examples/s]35502 examples [00:35, 983.55 examples/s] 35606 examples [00:35, 999.65 examples/s]35712 examples [00:35, 1016.97 examples/s]35816 examples [00:35, 1021.62 examples/s]35920 examples [00:35, 1026.34 examples/s]36023 examples [00:35, 988.84 examples/s] 36123 examples [00:35, 984.18 examples/s]36222 examples [00:36, 973.89 examples/s]36320 examples [00:36, 938.83 examples/s]36415 examples [00:36, 928.86 examples/s]36512 examples [00:36, 939.49 examples/s]36607 examples [00:36, 927.36 examples/s]36707 examples [00:36, 946.08 examples/s]36807 examples [00:36, 959.96 examples/s]36913 examples [00:36, 985.34 examples/s]37018 examples [00:36, 1001.20 examples/s]37123 examples [00:37, 1013.57 examples/s]37225 examples [00:37, 1007.51 examples/s]37326 examples [00:37, 1006.69 examples/s]37430 examples [00:37, 1014.23 examples/s]37532 examples [00:37, 1014.08 examples/s]37634 examples [00:37, 994.28 examples/s] 37739 examples [00:37, 1008.22 examples/s]37842 examples [00:37, 1012.50 examples/s]37944 examples [00:37, 1014.28 examples/s]38046 examples [00:37, 1006.96 examples/s]38150 examples [00:38, 1013.84 examples/s]38254 examples [00:38, 1019.74 examples/s]38357 examples [00:38, 1018.84 examples/s]38460 examples [00:38, 1018.91 examples/s]38562 examples [00:38, 997.61 examples/s] 38664 examples [00:38, 1003.34 examples/s]38767 examples [00:38, 1009.56 examples/s]38869 examples [00:38, 988.68 examples/s] 38969 examples [00:38, 984.10 examples/s]39074 examples [00:38, 1001.60 examples/s]39178 examples [00:39, 1010.43 examples/s]39280 examples [00:39, 1006.02 examples/s]39381 examples [00:39, 990.05 examples/s] 39484 examples [00:39, 1001.59 examples/s]39586 examples [00:39, 1005.67 examples/s]39687 examples [00:39, 1004.62 examples/s]39788 examples [00:39, 994.50 examples/s] 39888 examples [00:39, 989.87 examples/s]39988 examples [00:39, 987.64 examples/s]40087 examples [00:39, 912.92 examples/s]40191 examples [00:40, 947.61 examples/s]40288 examples [00:40, 954.07 examples/s]40389 examples [00:40, 968.58 examples/s]40491 examples [00:40, 981.75 examples/s]40597 examples [00:40, 1002.03 examples/s]40700 examples [00:40, 1007.98 examples/s]40803 examples [00:40, 1011.77 examples/s]40909 examples [00:40, 1023.92 examples/s]41012 examples [00:40, 1019.00 examples/s]41120 examples [00:41, 1034.19 examples/s]41227 examples [00:41, 1044.56 examples/s]41332 examples [00:41, 1026.89 examples/s]41435 examples [00:41, 998.74 examples/s] 41539 examples [00:41, 1010.22 examples/s]41647 examples [00:41, 1029.22 examples/s]41751 examples [00:41, 1001.87 examples/s]41852 examples [00:41, 978.98 examples/s] 41951 examples [00:41, 969.03 examples/s]42050 examples [00:41, 974.72 examples/s]42159 examples [00:42, 1005.65 examples/s]42260 examples [00:42, 1001.20 examples/s]42365 examples [00:42, 1013.84 examples/s]42467 examples [00:42, 1005.14 examples/s]42570 examples [00:42, 1011.91 examples/s]42674 examples [00:42, 1017.83 examples/s]42776 examples [00:42, 1013.00 examples/s]42878 examples [00:42, 1005.02 examples/s]42979 examples [00:42, 997.11 examples/s] 43080 examples [00:42, 999.08 examples/s]43182 examples [00:43, 1005.23 examples/s]43286 examples [00:43, 1013.63 examples/s]43388 examples [00:43, 1014.03 examples/s]43490 examples [00:43, 1011.13 examples/s]43592 examples [00:43, 1012.32 examples/s]43695 examples [00:43, 1015.10 examples/s]43799 examples [00:43, 1020.30 examples/s]43902 examples [00:43, 1010.32 examples/s]44005 examples [00:43, 1014.41 examples/s]44107 examples [00:43, 997.51 examples/s] 44207 examples [00:44, 994.92 examples/s]44310 examples [00:44, 1003.34 examples/s]44411 examples [00:44, 1001.04 examples/s]44517 examples [00:44, 1017.32 examples/s]44620 examples [00:44, 1020.33 examples/s]44723 examples [00:44, 1010.44 examples/s]44826 examples [00:44, 1013.68 examples/s]44930 examples [00:44, 1020.76 examples/s]45033 examples [00:44, 1017.40 examples/s]45138 examples [00:44, 1025.34 examples/s]45241 examples [00:45, 1021.13 examples/s]45344 examples [00:45, 1005.72 examples/s]45448 examples [00:45, 1015.16 examples/s]45552 examples [00:45, 1020.84 examples/s]45656 examples [00:45, 1025.27 examples/s]45759 examples [00:45, 1003.65 examples/s]45860 examples [00:45, 942.02 examples/s] 45959 examples [00:45, 953.41 examples/s]46060 examples [00:45, 968.33 examples/s]46158 examples [00:46, 957.00 examples/s]46263 examples [00:46, 981.85 examples/s]46371 examples [00:46, 1006.75 examples/s]46477 examples [00:46, 1021.05 examples/s]46584 examples [00:46, 1034.58 examples/s]46688 examples [00:46, 1034.85 examples/s]46795 examples [00:46, 1042.46 examples/s]46902 examples [00:46, 1048.61 examples/s]47007 examples [00:46, 1045.17 examples/s]47114 examples [00:46, 1050.10 examples/s]47221 examples [00:47, 1052.91 examples/s]47327 examples [00:47, 1053.07 examples/s]47433 examples [00:47, 1052.55 examples/s]47541 examples [00:47, 1058.32 examples/s]47648 examples [00:47, 1060.24 examples/s]47755 examples [00:47, 1046.07 examples/s]47863 examples [00:47, 1054.53 examples/s]47971 examples [00:47, 1061.27 examples/s]48078 examples [00:47, 1032.29 examples/s]48182 examples [00:47, 1031.33 examples/s]48286 examples [00:48, 1027.61 examples/s]48389 examples [00:48, 1018.31 examples/s]48491 examples [00:48, 1018.31 examples/s]48593 examples [00:48, 984.75 examples/s] 48692 examples [00:48, 983.91 examples/s]48795 examples [00:48, 995.40 examples/s]48900 examples [00:48, 1008.77 examples/s]49002 examples [00:48, 958.04 examples/s] 49103 examples [00:48, 971.70 examples/s]49206 examples [00:48, 987.79 examples/s]49310 examples [00:49, 1000.83 examples/s]49411 examples [00:49, 986.18 examples/s] 49512 examples [00:49, 991.89 examples/s]49612 examples [00:49, 984.34 examples/s]49721 examples [00:49, 1011.51 examples/s]49829 examples [00:49, 1029.31 examples/s]49933 examples [00:49, 1032.41 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7169/50000 [00:00<00:00, 71686.65 examples/s] 40%|████      | 20139/50000 [00:00<00:00, 82796.33 examples/s] 65%|██████▌   | 32662/50000 [00:00<00:00, 92164.96 examples/s] 91%|█████████ | 45471/50000 [00:00<00:00, 100631.40 examples/s]                                                                0 examples [00:00, ? examples/s]82 examples [00:00, 814.51 examples/s]188 examples [00:00, 874.88 examples/s]278 examples [00:00, 880.26 examples/s]383 examples [00:00, 922.87 examples/s]485 examples [00:00, 948.50 examples/s]588 examples [00:00, 970.94 examples/s]689 examples [00:00, 980.06 examples/s]794 examples [00:00, 998.95 examples/s]899 examples [00:00, 1012.22 examples/s]1003 examples [00:01, 1019.82 examples/s]1108 examples [00:01, 1027.99 examples/s]1210 examples [00:01, 1024.80 examples/s]1312 examples [00:01, 1016.87 examples/s]1414 examples [00:01, 800.90 examples/s] 1509 examples [00:01, 838.13 examples/s]1616 examples [00:01, 894.44 examples/s]1720 examples [00:01, 931.33 examples/s]1821 examples [00:01, 951.19 examples/s]1927 examples [00:02, 980.94 examples/s]2032 examples [00:02, 999.94 examples/s]2139 examples [00:02, 1018.42 examples/s]2243 examples [00:02, 1010.12 examples/s]2352 examples [00:02, 1030.50 examples/s]2456 examples [00:02, 1024.50 examples/s]2564 examples [00:02, 1037.81 examples/s]2669 examples [00:02, 1039.44 examples/s]2778 examples [00:02, 1052.53 examples/s]2886 examples [00:02, 1058.52 examples/s]2993 examples [00:03, 1061.59 examples/s]3100 examples [00:03, 1063.57 examples/s]3207 examples [00:03, 1064.45 examples/s]3314 examples [00:03, 1045.29 examples/s]3419 examples [00:03, 1043.37 examples/s]3524 examples [00:03, 980.59 examples/s] 3623 examples [00:03, 974.03 examples/s]3721 examples [00:03, 974.20 examples/s]3819 examples [00:03, 964.26 examples/s]3918 examples [00:03, 971.31 examples/s]4025 examples [00:04, 996.38 examples/s]4125 examples [00:04, 987.87 examples/s]4225 examples [00:04, 983.19 examples/s]4326 examples [00:04, 988.52 examples/s]4425 examples [00:04, 987.65 examples/s]4526 examples [00:04, 992.97 examples/s]4626 examples [00:04, 962.09 examples/s]4723 examples [00:04, 961.09 examples/s]4820 examples [00:04, 954.84 examples/s]4925 examples [00:04, 979.44 examples/s]5029 examples [00:05, 994.44 examples/s]5131 examples [00:05, 999.08 examples/s]5233 examples [00:05, 1003.48 examples/s]5334 examples [00:05, 977.76 examples/s] 5433 examples [00:05, 961.87 examples/s]5533 examples [00:05, 971.78 examples/s]5637 examples [00:05, 988.34 examples/s]5740 examples [00:05, 999.49 examples/s]5842 examples [00:05, 1004.57 examples/s]5943 examples [00:06, 988.89 examples/s] 6043 examples [00:06, 990.19 examples/s]6146 examples [00:06, 999.10 examples/s]6248 examples [00:06, 1003.97 examples/s]6349 examples [00:06, 1002.09 examples/s]6451 examples [00:06, 1006.85 examples/s]6553 examples [00:06, 1010.17 examples/s]6655 examples [00:06, 1010.84 examples/s]6757 examples [00:06, 1013.39 examples/s]6859 examples [00:06, 995.51 examples/s] 6959 examples [00:07, 957.30 examples/s]7056 examples [00:07, 945.52 examples/s]7160 examples [00:07, 971.59 examples/s]7258 examples [00:07, 956.59 examples/s]7357 examples [00:07, 965.24 examples/s]7457 examples [00:07, 973.92 examples/s]7559 examples [00:07, 986.92 examples/s]7658 examples [00:07, 980.03 examples/s]7766 examples [00:07, 1006.55 examples/s]7870 examples [00:07, 1016.21 examples/s]7977 examples [00:08, 1030.66 examples/s]8082 examples [00:08, 1034.33 examples/s]8186 examples [00:08, 1024.13 examples/s]8289 examples [00:08, 1011.16 examples/s]8391 examples [00:08, 1012.91 examples/s]8493 examples [00:08, 975.28 examples/s] 8595 examples [00:08, 987.87 examples/s]8695 examples [00:08, 989.57 examples/s]8797 examples [00:08, 998.43 examples/s]8899 examples [00:08, 1002.62 examples/s]9001 examples [00:09, 1006.50 examples/s]9102 examples [00:09, 1001.06 examples/s]9209 examples [00:09, 1018.20 examples/s]9311 examples [00:09, 1018.55 examples/s]9413 examples [00:09, 976.74 examples/s] 9515 examples [00:09, 987.38 examples/s]9622 examples [00:09, 1009.93 examples/s]9729 examples [00:09, 1026.69 examples/s]9837 examples [00:09, 1039.41 examples/s]9942 examples [00:09, 1035.40 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteCMKWXT/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteCMKWXT/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f159e87b8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f159e87b8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f159e87b8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f160ad1eda0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f157c07aef0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f159e87b8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f159e87b8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f159e87b8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f157feac438>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f157feac2e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f1574036620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f1574036620> 

  function with postional parmater data_info <function split_train_valid at 0x7f1574036620> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f1574036730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f1574036730> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f1574036730> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=f61edc8d4cc2fbb6fa2a08252df9f736551d6fec013c33038510ec7c2a0a1fcf
  Stored in directory: /tmp/pip-ephem-wheel-cache-f_82kah0/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:08<252:03:06, 950B/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:08<176:40:04, 1.36kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:08<123:41:56, 1.94kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:09<86:32:06, 2.76kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:09<60:23:06, 3.95kB/s].vector_cache/glove.6B.zip:   1%|          | 8.77M/862M [00:09<42:01:06, 5.64kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.9M/862M [00:09<29:18:21, 8.06kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.0M/862M [00:09<20:23:37, 11.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.3M/862M [00:09<14:12:16, 16.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.7M/862M [00:09<9:53:35, 23.5kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:09<6:53:31, 33.5kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:09<4:47:43, 47.9kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.4M/862M [00:09<3:20:44, 68.4kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.7M/862M [00:10<2:19:41, 97.7kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.0M/862M [00:10<1:37:31, 139kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:10<1:08:04, 198kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:12<49:19, 273kB/s]  .vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:12<36:21, 370kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:12<25:52, 519kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:14<20:37, 649kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:14<17:11, 777kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:14<12:36, 1.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:14<09:04, 1.47MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:16<10:07, 1.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:16<08:29, 1.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:16<06:16, 2.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:18<07:28, 1.77MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:18<06:36, 2.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:18<04:54, 2.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:20<06:32, 2.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:20<05:55, 2.22MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:20<04:28, 2.93MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:22<06:12, 2.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:22<05:43, 2.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:22<04:17, 3.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:24<06:07, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:24<06:59, 1.86MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:24<05:26, 2.39MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.8M/862M [00:24<04:03, 3.20MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:26<06:43, 1.93MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:26<06:04, 2.13MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:26<04:35, 2.82MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:28<06:12, 2.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:28<06:59, 1.84MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:28<05:33, 2.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:28<04:01, 3.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:30<12:19:24, 17.3kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:30<8:38:38, 24.7kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:30<6:02:36, 35.3kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:32<4:16:06, 49.8kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:32<3:00:29, 70.6kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:32<2:06:24, 101kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:33<1:31:14, 139kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:34<1:05:08, 195kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:34<45:46, 276kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:35<34:57, 361kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:36<25:48, 489kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:36<18:18, 688kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:37<15:39, 802kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:38<13:38, 920kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:38<10:05, 1.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:38<07:13, 1.73MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:39<10:19, 1.21MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:40<08:33, 1.46MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:40<06:16, 1.99MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:41<07:12, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:41<07:42, 1.61MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:42<05:56, 2.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:42<04:19, 2.85MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:43<08:06, 1.52MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:43<06:57, 1.77MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:44<05:11, 2.37MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:45<06:26, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:45<05:51, 2.10MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:46<04:22, 2.80MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:47<05:51, 2.08MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:47<06:42, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:47<05:15, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:48<03:51, 3.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:49<07:35, 1.60MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:49<06:36, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:49<04:52, 2.48MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:51<06:12, 1.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:51<06:55, 1.74MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:51<05:24, 2.23MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:51<03:56, 3.05MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:53<08:48, 1.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:53<07:27, 1.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:53<05:31, 2.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:55<06:36, 1.81MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:55<05:52, 2.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:55<04:25, 2.69MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:57<05:48, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:57<05:18, 2.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:57<04:01, 2.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:59<05:31, 2.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:59<06:23, 1.84MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:59<05:06, 2.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:59<03:43, 3.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:01<37:45, 311kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:01<27:38, 424kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:01<19:37, 596kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:03<16:22, 712kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:03<12:43, 916kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:03<09:08, 1.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:05<09:01, 1.28MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:05<07:33, 1.53MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:05<05:32, 2.09MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:07<06:28, 1.78MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:07<06:59, 1.65MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:07<05:23, 2.13MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:07<04:00, 2.87MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:09<06:04, 1.89MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:09<05:26, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:09<04:06, 2.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:11<05:29, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:11<04:51, 2.35MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:11<03:48, 2.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:11<02:46, 4.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:13<23:15, 486kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:13<18:42, 605kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:13<13:34, 832kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:13<09:42, 1.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:15<10:10, 1.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:15<08:19, 1.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:15<06:03, 1.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:17<06:50, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:17<07:10, 1.56MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:17<05:36, 1.99MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:17<04:03, 2.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:19<38:00, 292kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:19<27:45, 400kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:19<19:40, 563kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:21<16:16, 679kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:21<13:44, 803kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:21<10:12, 1.08MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:21<07:15, 1.51MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:23<37:59, 289kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:23<27:43, 396kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:23<19:36, 558kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:25<16:11, 673kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:25<13:40, 798kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:25<10:03, 1.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:25<07:10, 1.51MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:26<09:45, 1.11MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:27<07:58, 1.36MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:27<05:51, 1.84MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:28<06:34, 1.64MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:29<05:45, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:29<04:18, 2.50MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:30<05:27, 1.96MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:31<04:57, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:31<03:44, 2.85MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:32<05:03, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:32<04:39, 2.28MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:33<03:32, 3.00MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:34<04:53, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:34<04:32, 2.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:35<03:26, 3.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:36<04:49, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:36<04:28, 2.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:37<03:24, 3.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:38<04:46, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:38<04:25, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:39<03:21, 3.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:40<04:44, 2.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:40<04:25, 2.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:40<03:19, 3.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:42<04:41, 2.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:42<05:29, 1.88MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:42<04:18, 2.38MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:43<03:08, 3.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:44<06:54, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:44<05:56, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:44<04:24, 2.31MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:46<05:24, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:46<05:57, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:46<04:38, 2.19MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:46<03:22, 2.99MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:48<07:02, 1.43MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:48<05:58, 1.69MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:48<04:24, 2.28MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:50<05:23, 1.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:50<04:49, 2.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:50<03:35, 2.78MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:52<04:46, 2.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:52<04:23, 2.27MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:52<03:18, 3.00MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:54<04:32, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:54<04:14, 2.33MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:54<03:13, 3.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:56<04:30, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:56<04:01, 2.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:56<03:04, 3.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:58<04:24, 2.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:58<04:07, 2.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:58<03:08, 3.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [02:00<04:24, 2.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [02:00<04:06, 2.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:00<03:05, 3.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [02:02<04:21, 2.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [02:02<05:01, 1.91MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:02<03:56, 2.44MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:02<02:52, 3.32MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:04<06:42, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:04<05:41, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:04<04:13, 2.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:06<05:07, 1.85MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:06<04:34, 2.07MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:06<03:24, 2.77MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:08<04:33, 2.06MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:08<05:12, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:08<04:03, 2.31MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:08<02:59, 3.12MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:10<05:07, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:10<04:25, 2.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:10<03:23, 2.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:12<04:25, 2.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:12<05:04, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:12<04:02, 2.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:12<02:56, 3.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:14<29:37, 310kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:14<21:42, 423kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:14<15:21, 597kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:16<12:48, 712kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:16<09:55, 919kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:16<07:09, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:17<07:04, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:18<06:54, 1.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:18<05:17, 1.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:18<03:48, 2.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:19<25:01, 359kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:20<18:26, 487kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:20<13:06, 683kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:21<11:11, 797kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:22<09:45, 913kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:22<07:14, 1.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:22<05:09, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:23<08:31, 1.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:23<06:55, 1.28MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:24<05:03, 1.74MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:25<05:32, 1.58MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:25<04:48, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:26<03:34, 2.44MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:27<04:31, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:27<05:01, 1.73MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:28<03:58, 2.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:28<02:51, 3.02MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:29<33:13, 260kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:29<24:09, 357kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:29<17:04, 504kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:31<13:50, 619kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:31<10:36, 808kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:31<07:35, 1.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:33<07:13, 1.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:33<06:52, 1.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:33<05:10, 1.64MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:34<03:44, 2.26MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:35<05:55, 1.43MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:35<04:52, 1.73MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:35<03:34, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:35<02:37, 3.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:37<18:08, 461kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:37<14:28, 578kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:37<10:29, 796kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:37<07:30, 1.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:39<07:30, 1.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:39<06:08, 1.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:39<04:29, 1.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:41<05:01, 1.64MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:41<04:22, 1.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:41<03:16, 2.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:43<04:10, 1.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:43<03:47, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<02:50, 2.87MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:45<03:50, 2.11MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:45<04:25, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:45<03:27, 2.33MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:45<02:34, 3.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:47<04:14, 1.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:47<03:49, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:47<02:53, 2.77MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:49<03:48, 2.09MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:49<03:29, 2.27MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:49<02:38, 2.99MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:51<03:39, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:51<03:24, 2.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:51<02:35, 3.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:53<03:36, 2.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:53<04:11, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:53<03:20, 2.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:53<02:25, 3.20MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:55<56:58, 136kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:55<40:41, 190kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:55<28:35, 270kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:57<21:40, 354kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:57<16:49, 456kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:57<12:06, 633kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:57<08:34, 891kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:59<08:49, 862kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:59<06:59, 1.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:59<05:04, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:01<05:16, 1.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:01<04:30, 1.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:01<03:20, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:03<04:03, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:03<04:27, 1.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:03<03:31, 2.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:03<02:32, 2.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:05<25:10, 294kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:05<18:23, 402kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:05<13:01, 566kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:07<10:46, 681kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:07<08:19, 881kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:07<05:58, 1.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:09<05:49, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:09<04:50, 1.50MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:09<03:32, 2.05MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:09<02:33, 2.81MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:10<2:23:33, 50.2kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:11<1:41:55, 70.6kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:11<1:11:32, 100kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:11<50:03, 143kB/s]  .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:12<36:53, 193kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:13<26:32, 268kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:13<18:42, 380kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:14<14:39, 482kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:15<11:00, 641kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:15<07:50, 897kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:16<07:03, 992kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:16<05:40, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:17<04:08, 1.68MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:18<04:28, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:18<03:52, 1.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:19<02:53, 2.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:20<03:35, 1.91MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:20<03:13, 2.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:21<02:25, 2.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:22<03:15, 2.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:22<02:59, 2.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:22<02:15, 2.98MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:25<02:57, 2.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:25<04:50, 1.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:25<04:06, 1.63MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:25<03:00, 2.21MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:27<03:32, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:27<03:10, 2.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:27<02:21, 2.80MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:29<03:09, 2.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:29<03:36, 1.82MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:29<02:52, 2.28MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:29<02:04, 3.13MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:31<47:44, 136kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:31<34:05, 191kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:31<23:56, 270kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:32<18:08, 355kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:33<13:21, 481kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:33<09:28, 675kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:34<08:04, 788kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:35<06:19, 1.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:35<04:34, 1.38MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:36<04:38, 1.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:37<03:55, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:37<02:54, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:38<03:27, 1.80MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:38<03:04, 2.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:39<02:17, 2.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:40<03:01, 2.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:40<03:25, 1.80MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:41<02:40, 2.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:41<01:57, 3.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:42<03:50, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:42<03:19, 1.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:43<02:27, 2.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:44<03:06, 1.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:44<02:48, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:44<02:05, 2.86MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:46<02:49, 2.10MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:46<03:20, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:46<02:39, 2.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:47<01:55, 3.07MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:48<18:53, 312kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:48<13:49, 425kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:48<09:47, 598kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:50<08:12, 708kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:50<06:59, 832kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:50<05:08, 1.13MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:51<03:39, 1.58MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:52<05:55, 969kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:52<04:45, 1.21MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:52<03:26, 1.66MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:54<03:43, 1.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:54<03:49, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:54<02:58, 1.91MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:54<02:08, 2.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:56<18:19, 306kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:56<13:24, 418kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:56<09:28, 589kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:58<07:52, 704kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:58<06:41, 827kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:58<04:58, 1.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:58<03:30, 1.56MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:00<15:07, 362kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:00<11:09, 490kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:00<07:54, 688kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:02<06:44, 802kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:02<05:16, 1.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:02<03:49, 1.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:04<03:53, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:04<03:51, 1.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:04<02:58, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:04<02:08, 2.46MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:06<17:17, 305kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:06<12:38, 416kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:06<08:55, 586kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:08<07:24, 702kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:08<05:37, 923kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:08<04:08, 1.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:08<02:55, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:10<12:02, 426kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:10<08:57, 573kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:10<06:22, 801kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:12<05:35, 905kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:12<04:59, 1.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:12<03:45, 1.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:12<02:40, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:14<05:54, 844kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:14<04:40, 1.07MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:14<03:23, 1.47MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:16<03:28, 1.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:16<02:57, 1.66MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:16<02:11, 2.23MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:18<02:38, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:18<02:21, 2.06MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:18<01:46, 2.73MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:20<02:19, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:20<02:39, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:20<02:06, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:20<01:30, 3.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:22<12:24, 380kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:22<09:10, 514kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:22<06:30, 720kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:23<05:35, 831kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:24<04:23, 1.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:24<03:10, 1.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:25<03:15, 1.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:26<03:15, 1.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:26<02:28, 1.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:26<01:47, 2.53MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:27<02:59, 1.51MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:28<02:34, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:28<01:54, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:29<02:20, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:29<02:01, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:30<01:31, 2.90MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:30<01:06, 3.93MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:31<11:47, 371kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:31<08:41, 502kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:32<06:10, 704kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:33<05:16, 816kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:33<04:08, 1.04MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:34<03:00, 1.43MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:35<03:03, 1.39MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:35<02:34, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:36<01:54, 2.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:37<02:17, 1.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:37<02:30, 1.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:37<01:56, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:38<01:24, 2.91MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:39<02:36, 1.57MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:39<02:15, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:39<01:40, 2.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:41<02:05, 1.93MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:41<02:19, 1.73MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:41<01:48, 2.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:41<01:18, 3.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:43<02:45, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:43<02:20, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:43<01:44, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:45<02:06, 1.85MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:45<02:18, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:45<01:48, 2.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:45<01:17, 2.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:47<09:44, 393kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:47<07:13, 529kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:47<05:07, 740kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:49<04:23, 855kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:49<03:27, 1.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:49<02:29, 1.49MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:51<02:34, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:51<02:11, 1.68MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:51<01:37, 2.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:53<01:57, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:53<02:08, 1.68MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:53<01:39, 2.17MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:53<01:12, 2.97MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:55<02:20, 1.52MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:55<02:00, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:55<01:29, 2.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:57<01:49, 1.92MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:57<01:38, 2.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:57<01:13, 2.84MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:59<01:38, 2.09MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:59<01:50, 1.85MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:59<01:28, 2.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:59<01:02, 3.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:01<18:18, 183kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:01<13:08, 254kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:01<09:13, 360kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:03<07:08, 459kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:03<05:19, 614kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:03<03:46, 861kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:05<03:21, 954kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:05<02:41, 1.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:05<01:57, 1.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:07<02:04, 1.52MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:07<01:46, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:07<01:18, 2.37MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:09<01:36, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:09<01:27, 2.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:09<01:04, 2.82MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:11<01:26, 2.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:11<01:38, 1.82MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:11<01:16, 2.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:11<00:56, 3.16MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:13<01:43, 1.70MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:13<01:31, 1.93MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:13<01:07, 2.57MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:15<01:24, 2.04MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:15<01:34, 1.82MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:15<01:14, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:17<01:18, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:17<01:12, 2.31MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:17<00:53, 3.07MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:19<01:14, 2.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:19<01:10, 2.33MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:19<00:52, 3.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:21<01:13, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:21<01:25, 1.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:21<01:06, 2.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:21<00:48, 3.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:23<01:38, 1.57MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:23<01:25, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:23<01:03, 2.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:25<01:18, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:25<01:10, 2.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:25<00:52, 2.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:26<01:10, 2.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:27<01:04, 2.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:27<00:48, 2.97MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:28<01:06, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:29<01:01, 2.31MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:29<00:46, 3.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:30<01:04, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:31<00:59, 2.33MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:31<00:44, 3.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:32<01:01, 2.19MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:32<00:57, 2.35MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:33<00:42, 3.13MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:34<00:59, 2.18MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:34<01:08, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:35<00:54, 2.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:35<00:38, 3.26MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:36<11:28, 184kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:36<08:13, 255kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:37<05:44, 362kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:38<04:24, 462kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:38<03:30, 581kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:38<02:31, 800kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:39<01:46, 1.12MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:40<01:53, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:40<01:29, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:40<01:04, 1.81MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:41<00:45, 2.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:42<04:14, 449kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:42<03:22, 563kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:42<02:26, 771kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:43<01:41, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:44<05:34, 328kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:44<04:05, 447kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:44<02:52, 627kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<02:22, 743kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<02:02, 865kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:46<01:29, 1.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:46<01:03, 1.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:48<01:25, 1.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:48<01:10, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:48<00:50, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:50<00:57, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:50<01:00, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:50<00:47, 2.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:50<00:33, 2.82MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:52<07:52, 198kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:52<05:39, 274kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:52<03:55, 388kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:54<03:01, 492kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:54<02:16, 654kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:54<01:35, 915kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:56<01:24, 1.00MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:56<01:08, 1.25MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:56<00:49, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:58<00:52, 1.55MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:58<00:43, 1.86MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:58<00:31, 2.50MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:58<00:22, 3.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:00<02:45, 464kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:00<02:03, 620kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:00<01:26, 866kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:02<01:16, 954kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:02<01:09, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:02<00:51, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:02<00:35, 1.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:04<03:51, 298kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:04<02:48, 407kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:04<01:56, 573kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:06<01:34, 688kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:06<01:12, 890kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:06<00:50, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:08<00:48, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:08<00:46, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:08<00:34, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:08<00:24, 2.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:10<00:35, 1.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:10<00:30, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:10<00:22, 2.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:12<00:27, 1.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:12<00:30, 1.73MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:12<00:23, 2.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:12<00:16, 3.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:14<00:26, 1.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:14<00:23, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:14<00:17, 2.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:16<00:21, 2.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:16<00:24, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:16<00:18, 2.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:16<00:13, 3.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:18<00:22, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:18<00:19, 2.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:18<00:14, 2.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:20<00:17, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:20<00:19, 1.80MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:20<00:15, 2.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:20<00:10, 3.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:21<00:17, 1.86MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:22<00:15, 2.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:22<00:10, 2.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:23<00:13, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:24<00:15, 1.81MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:24<00:11, 2.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:24<00:08, 3.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:25<00:10, 2.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:26<00:09, 2.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:26<00:06, 3.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:27<00:08, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:27<00:10, 1.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:28<00:07, 2.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:28<00:05, 3.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:29<00:08, 1.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:29<00:07, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:30<00:05, 2.62MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:31<00:05, 2.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:31<00:06, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:32<00:04, 2.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:32<00:02, 3.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:33<00:47, 151kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:33<00:31, 211kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:33<00:17, 299kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:35<00:07, 389kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:35<00:05, 499kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:35<00:02, 687kB/s].vector_cache/glove.6B.zip: 862MB [06:36, 2.18MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 758/400000 [00:00<00:52, 7575.71it/s]  0%|          | 1583/400000 [00:00<00:51, 7763.59it/s]  1%|          | 2289/400000 [00:00<00:52, 7535.39it/s]  1%|          | 3121/400000 [00:00<00:51, 7754.54it/s]  1%|          | 3965/400000 [00:00<00:49, 7946.72it/s]  1%|          | 4811/400000 [00:00<00:48, 8092.68it/s]  1%|▏         | 5600/400000 [00:00<00:49, 8029.12it/s]  2%|▏         | 6391/400000 [00:00<00:49, 7990.58it/s]  2%|▏         | 7173/400000 [00:00<00:49, 7937.78it/s]  2%|▏         | 7973/400000 [00:01<00:49, 7955.05it/s]  2%|▏         | 8748/400000 [00:01<00:49, 7882.20it/s]  2%|▏         | 9564/400000 [00:01<00:49, 7961.14it/s]  3%|▎         | 10413/400000 [00:01<00:48, 8112.50it/s]  3%|▎         | 11275/400000 [00:01<00:47, 8257.85it/s]  3%|▎         | 12118/400000 [00:01<00:46, 8307.25it/s]  3%|▎         | 12982/400000 [00:01<00:46, 8402.53it/s]  3%|▎         | 13844/400000 [00:01<00:45, 8465.43it/s]  4%|▎         | 14704/400000 [00:01<00:45, 8504.33it/s]  4%|▍         | 15554/400000 [00:01<00:45, 8428.02it/s]  4%|▍         | 16397/400000 [00:02<00:45, 8372.60it/s]  4%|▍         | 17238/400000 [00:02<00:45, 8381.28it/s]  5%|▍         | 18096/400000 [00:02<00:45, 8439.49it/s]  5%|▍         | 18955/400000 [00:02<00:44, 8483.30it/s]  5%|▍         | 19804/400000 [00:02<00:45, 8371.68it/s]  5%|▌         | 20642/400000 [00:02<00:45, 8332.65it/s]  5%|▌         | 21499/400000 [00:02<00:45, 8399.83it/s]  6%|▌         | 22340/400000 [00:02<00:45, 8339.36it/s]  6%|▌         | 23181/400000 [00:02<00:45, 8359.42it/s]  6%|▌         | 24019/400000 [00:02<00:44, 8365.31it/s]  6%|▌         | 24856/400000 [00:03<00:44, 8341.54it/s]  6%|▋         | 25700/400000 [00:03<00:44, 8368.51it/s]  7%|▋         | 26537/400000 [00:03<00:45, 8293.27it/s]  7%|▋         | 27367/400000 [00:03<00:45, 8209.08it/s]  7%|▋         | 28189/400000 [00:03<00:46, 8076.87it/s]  7%|▋         | 29038/400000 [00:03<00:45, 8195.38it/s]  7%|▋         | 29896/400000 [00:03<00:44, 8305.80it/s]  8%|▊         | 30750/400000 [00:03<00:44, 8373.28it/s]  8%|▊         | 31589/400000 [00:03<00:45, 8180.92it/s]  8%|▊         | 32409/400000 [00:03<00:45, 8102.57it/s]  8%|▊         | 33221/400000 [00:04<00:45, 8043.73it/s]  9%|▊         | 34059/400000 [00:04<00:44, 8140.45it/s]  9%|▊         | 34887/400000 [00:04<00:44, 8181.36it/s]  9%|▉         | 35706/400000 [00:04<00:44, 8171.57it/s]  9%|▉         | 36546/400000 [00:04<00:44, 8237.12it/s]  9%|▉         | 37388/400000 [00:04<00:43, 8289.25it/s] 10%|▉         | 38233/400000 [00:04<00:43, 8335.49it/s] 10%|▉         | 39078/400000 [00:04<00:43, 8368.49it/s] 10%|▉         | 39919/400000 [00:04<00:42, 8378.71it/s] 10%|█         | 40762/400000 [00:04<00:42, 8392.86it/s] 10%|█         | 41626/400000 [00:05<00:42, 8464.58it/s] 11%|█         | 42473/400000 [00:05<00:42, 8378.92it/s] 11%|█         | 43325/400000 [00:05<00:42, 8420.45it/s] 11%|█         | 44168/400000 [00:05<00:42, 8321.75it/s] 11%|█▏        | 45024/400000 [00:05<00:42, 8389.28it/s] 11%|█▏        | 45873/400000 [00:05<00:42, 8419.09it/s] 12%|█▏        | 46722/400000 [00:05<00:41, 8438.56it/s] 12%|█▏        | 47567/400000 [00:05<00:42, 8371.19it/s] 12%|█▏        | 48405/400000 [00:05<00:42, 8257.59it/s] 12%|█▏        | 49232/400000 [00:05<00:43, 8110.45it/s] 13%|█▎        | 50065/400000 [00:06<00:42, 8175.03it/s] 13%|█▎        | 50884/400000 [00:06<00:43, 8069.82it/s] 13%|█▎        | 51692/400000 [00:06<00:43, 8028.55it/s] 13%|█▎        | 52537/400000 [00:06<00:42, 8149.81it/s] 13%|█▎        | 53374/400000 [00:06<00:42, 8214.63it/s] 14%|█▎        | 54197/400000 [00:06<00:42, 8158.93it/s] 14%|█▍        | 55014/400000 [00:06<00:42, 8035.19it/s] 14%|█▍        | 55819/400000 [00:06<00:42, 8018.75it/s] 14%|█▍        | 56622/400000 [00:06<00:42, 8015.05it/s] 14%|█▍        | 57425/400000 [00:06<00:42, 8017.57it/s] 15%|█▍        | 58228/400000 [00:07<00:42, 8017.87it/s] 15%|█▍        | 59031/400000 [00:07<00:42, 8020.49it/s] 15%|█▍        | 59834/400000 [00:07<00:42, 7926.99it/s] 15%|█▌        | 60652/400000 [00:07<00:42, 8000.57it/s] 15%|█▌        | 61453/400000 [00:07<00:42, 7874.70it/s] 16%|█▌        | 62242/400000 [00:07<00:42, 7855.48it/s] 16%|█▌        | 63061/400000 [00:07<00:42, 7952.04it/s] 16%|█▌        | 63857/400000 [00:07<00:43, 7786.65it/s] 16%|█▌        | 64637/400000 [00:07<00:43, 7773.83it/s] 16%|█▋        | 65440/400000 [00:07<00:42, 7846.80it/s] 17%|█▋        | 66292/400000 [00:08<00:41, 8035.98it/s] 17%|█▋        | 67140/400000 [00:08<00:40, 8162.71it/s] 17%|█▋        | 67958/400000 [00:08<00:41, 8096.37it/s] 17%|█▋        | 68769/400000 [00:08<00:41, 7973.84it/s] 17%|█▋        | 69575/400000 [00:08<00:41, 7998.06it/s] 18%|█▊        | 70411/400000 [00:08<00:40, 8101.35it/s] 18%|█▊        | 71223/400000 [00:08<00:40, 8036.83it/s] 18%|█▊        | 72028/400000 [00:08<00:41, 7997.17it/s] 18%|█▊        | 72843/400000 [00:08<00:40, 8040.65it/s] 18%|█▊        | 73693/400000 [00:09<00:39, 8171.69it/s] 19%|█▊        | 74524/400000 [00:09<00:39, 8208.70it/s] 19%|█▉        | 75346/400000 [00:09<00:39, 8188.21it/s] 19%|█▉        | 76179/400000 [00:09<00:39, 8228.38it/s] 19%|█▉        | 77017/400000 [00:09<00:39, 8272.98it/s] 19%|█▉        | 77851/400000 [00:09<00:38, 8292.37it/s] 20%|█▉        | 78710/400000 [00:09<00:38, 8377.66it/s] 20%|█▉        | 79549/400000 [00:09<00:38, 8262.91it/s] 20%|██        | 80376/400000 [00:09<00:39, 8164.08it/s] 20%|██        | 81213/400000 [00:09<00:38, 8224.40it/s] 21%|██        | 82055/400000 [00:10<00:38, 8280.49it/s] 21%|██        | 82897/400000 [00:10<00:38, 8319.82it/s] 21%|██        | 83730/400000 [00:10<00:38, 8238.56it/s] 21%|██        | 84556/400000 [00:10<00:38, 8242.32it/s] 21%|██▏       | 85381/400000 [00:10<00:38, 8140.23it/s] 22%|██▏       | 86239/400000 [00:10<00:37, 8265.39it/s] 22%|██▏       | 87094/400000 [00:10<00:37, 8347.76it/s] 22%|██▏       | 87930/400000 [00:10<00:37, 8238.35it/s] 22%|██▏       | 88755/400000 [00:10<00:38, 8020.21it/s] 22%|██▏       | 89559/400000 [00:10<00:39, 7847.42it/s] 23%|██▎       | 90353/400000 [00:11<00:39, 7874.13it/s] 23%|██▎       | 91160/400000 [00:11<00:38, 7931.83it/s] 23%|██▎       | 91959/400000 [00:11<00:38, 7948.04it/s] 23%|██▎       | 92771/400000 [00:11<00:38, 7996.77it/s] 23%|██▎       | 93609/400000 [00:11<00:37, 8105.35it/s] 24%|██▎       | 94425/400000 [00:11<00:37, 8120.90it/s] 24%|██▍       | 95251/400000 [00:11<00:37, 8160.49it/s] 24%|██▍       | 96068/400000 [00:11<00:37, 8098.09it/s] 24%|██▍       | 96903/400000 [00:11<00:37, 8171.07it/s] 24%|██▍       | 97741/400000 [00:11<00:36, 8231.93it/s] 25%|██▍       | 98565/400000 [00:12<00:36, 8230.85it/s] 25%|██▍       | 99423/400000 [00:12<00:36, 8332.44it/s] 25%|██▌       | 100280/400000 [00:12<00:35, 8400.19it/s] 25%|██▌       | 101121/400000 [00:12<00:35, 8392.21it/s] 25%|██▌       | 101961/400000 [00:12<00:35, 8364.08it/s] 26%|██▌       | 102813/400000 [00:12<00:35, 8409.82it/s] 26%|██▌       | 103681/400000 [00:12<00:34, 8487.35it/s] 26%|██▌       | 104540/400000 [00:12<00:34, 8514.99it/s] 26%|██▋       | 105392/400000 [00:12<00:35, 8380.08it/s] 27%|██▋       | 106231/400000 [00:12<00:35, 8354.31it/s] 27%|██▋       | 107076/400000 [00:13<00:34, 8381.03it/s] 27%|██▋       | 107915/400000 [00:13<00:35, 8209.06it/s] 27%|██▋       | 108772/400000 [00:13<00:35, 8311.54it/s] 27%|██▋       | 109609/400000 [00:13<00:34, 8328.64it/s] 28%|██▊       | 110459/400000 [00:13<00:34, 8378.16it/s] 28%|██▊       | 111298/400000 [00:13<00:34, 8355.11it/s] 28%|██▊       | 112169/400000 [00:13<00:34, 8456.07it/s] 28%|██▊       | 113023/400000 [00:13<00:33, 8479.98it/s] 28%|██▊       | 113885/400000 [00:13<00:33, 8518.77it/s] 29%|██▊       | 114738/400000 [00:13<00:33, 8472.60it/s] 29%|██▉       | 115586/400000 [00:14<00:33, 8429.40it/s] 29%|██▉       | 116457/400000 [00:14<00:33, 8511.15it/s] 29%|██▉       | 117319/400000 [00:14<00:33, 8541.78it/s] 30%|██▉       | 118178/400000 [00:14<00:32, 8555.28it/s] 30%|██▉       | 119034/400000 [00:14<00:32, 8515.37it/s] 30%|██▉       | 119886/400000 [00:14<00:32, 8492.52it/s] 30%|███       | 120736/400000 [00:14<00:33, 8344.83it/s] 30%|███       | 121593/400000 [00:14<00:33, 8410.70it/s] 31%|███       | 122454/400000 [00:14<00:32, 8467.47it/s] 31%|███       | 123302/400000 [00:14<00:32, 8468.30it/s] 31%|███       | 124150/400000 [00:15<00:33, 8263.42it/s] 31%|███       | 124987/400000 [00:15<00:33, 8294.58it/s] 31%|███▏      | 125847/400000 [00:15<00:32, 8381.11it/s] 32%|███▏      | 126694/400000 [00:15<00:32, 8405.73it/s] 32%|███▏      | 127564/400000 [00:15<00:32, 8489.69it/s] 32%|███▏      | 128414/400000 [00:15<00:32, 8415.48it/s] 32%|███▏      | 129269/400000 [00:15<00:32, 8454.66it/s] 33%|███▎      | 130115/400000 [00:15<00:32, 8345.57it/s] 33%|███▎      | 130959/400000 [00:15<00:32, 8371.61it/s] 33%|███▎      | 131799/400000 [00:16<00:32, 8378.75it/s] 33%|███▎      | 132644/400000 [00:16<00:31, 8398.25it/s] 33%|███▎      | 133485/400000 [00:16<00:31, 8401.55it/s] 34%|███▎      | 134326/400000 [00:16<00:31, 8377.99it/s] 34%|███▍      | 135188/400000 [00:16<00:31, 8448.93it/s] 34%|███▍      | 136057/400000 [00:16<00:30, 8517.95it/s] 34%|███▍      | 136910/400000 [00:16<00:30, 8518.56it/s] 34%|███▍      | 137769/400000 [00:16<00:30, 8539.50it/s] 35%|███▍      | 138633/400000 [00:16<00:30, 8568.01it/s] 35%|███▍      | 139503/400000 [00:16<00:30, 8606.71it/s] 35%|███▌      | 140364/400000 [00:17<00:30, 8560.05it/s] 35%|███▌      | 141221/400000 [00:17<00:30, 8505.69it/s] 36%|███▌      | 142072/400000 [00:17<00:30, 8438.82it/s] 36%|███▌      | 142919/400000 [00:17<00:30, 8445.63it/s] 36%|███▌      | 143770/400000 [00:17<00:30, 8462.20it/s] 36%|███▌      | 144620/400000 [00:17<00:30, 8472.38it/s] 36%|███▋      | 145468/400000 [00:17<00:30, 8369.94it/s] 37%|███▋      | 146306/400000 [00:17<00:31, 8048.30it/s] 37%|███▋      | 147161/400000 [00:17<00:30, 8190.06it/s] 37%|███▋      | 148010/400000 [00:17<00:30, 8275.21it/s] 37%|███▋      | 148870/400000 [00:18<00:30, 8368.12it/s] 37%|███▋      | 149722/400000 [00:18<00:29, 8412.71it/s] 38%|███▊      | 150565/400000 [00:18<00:29, 8416.06it/s] 38%|███▊      | 151432/400000 [00:18<00:29, 8487.79it/s] 38%|███▊      | 152282/400000 [00:18<00:29, 8471.56it/s] 38%|███▊      | 153141/400000 [00:18<00:29, 8506.09it/s] 38%|███▊      | 153993/400000 [00:18<00:29, 8323.97it/s] 39%|███▊      | 154827/400000 [00:18<00:29, 8210.50it/s] 39%|███▉      | 155650/400000 [00:18<00:30, 8144.90it/s] 39%|███▉      | 156466/400000 [00:18<00:29, 8148.75it/s] 39%|███▉      | 157282/400000 [00:19<00:29, 8131.95it/s] 40%|███▉      | 158096/400000 [00:19<00:30, 8051.88it/s] 40%|███▉      | 158902/400000 [00:19<00:30, 8015.27it/s] 40%|███▉      | 159706/400000 [00:19<00:29, 8021.32it/s] 40%|████      | 160509/400000 [00:19<00:29, 8023.34it/s] 40%|████      | 161320/400000 [00:19<00:29, 8047.71it/s] 41%|████      | 162136/400000 [00:19<00:29, 8079.51it/s] 41%|████      | 162945/400000 [00:19<00:29, 8037.69it/s] 41%|████      | 163761/400000 [00:19<00:29, 8073.99it/s] 41%|████      | 164574/400000 [00:19<00:29, 8090.16it/s] 41%|████▏     | 165384/400000 [00:20<00:29, 8082.49it/s] 42%|████▏     | 166209/400000 [00:20<00:28, 8130.39it/s] 42%|████▏     | 167023/400000 [00:20<00:28, 8121.05it/s] 42%|████▏     | 167843/400000 [00:20<00:28, 8141.86it/s] 42%|████▏     | 168658/400000 [00:20<00:28, 8144.04it/s] 42%|████▏     | 169473/400000 [00:20<00:28, 8120.63it/s] 43%|████▎     | 170308/400000 [00:20<00:28, 8186.17it/s] 43%|████▎     | 171166/400000 [00:20<00:27, 8298.39it/s] 43%|████▎     | 172011/400000 [00:20<00:27, 8342.63it/s] 43%|████▎     | 172876/400000 [00:20<00:26, 8432.12it/s] 43%|████▎     | 173738/400000 [00:21<00:26, 8486.94it/s] 44%|████▎     | 174590/400000 [00:21<00:26, 8493.47it/s] 44%|████▍     | 175447/400000 [00:21<00:26, 8514.56it/s] 44%|████▍     | 176299/400000 [00:21<00:26, 8466.70it/s] 44%|████▍     | 177146/400000 [00:21<00:26, 8467.20it/s] 44%|████▍     | 177993/400000 [00:21<00:26, 8434.16it/s] 45%|████▍     | 178837/400000 [00:21<00:27, 8139.77it/s] 45%|████▍     | 179654/400000 [00:21<00:27, 8102.80it/s] 45%|████▌     | 180485/400000 [00:21<00:26, 8161.66it/s] 45%|████▌     | 181330/400000 [00:21<00:26, 8243.21it/s] 46%|████▌     | 182188/400000 [00:22<00:26, 8341.25it/s] 46%|████▌     | 183051/400000 [00:22<00:25, 8424.81it/s] 46%|████▌     | 183895/400000 [00:22<00:25, 8322.50it/s] 46%|████▌     | 184729/400000 [00:22<00:26, 8269.02it/s] 46%|████▋     | 185557/400000 [00:22<00:25, 8258.35it/s] 47%|████▋     | 186395/400000 [00:22<00:25, 8293.90it/s] 47%|████▋     | 187225/400000 [00:22<00:25, 8270.90it/s] 47%|████▋     | 188086/400000 [00:22<00:25, 8367.63it/s] 47%|████▋     | 188944/400000 [00:22<00:25, 8428.10it/s] 47%|████▋     | 189811/400000 [00:22<00:24, 8498.16it/s] 48%|████▊     | 190665/400000 [00:23<00:24, 8508.37it/s] 48%|████▊     | 191526/400000 [00:23<00:24, 8537.73it/s] 48%|████▊     | 192385/400000 [00:23<00:24, 8551.32it/s] 48%|████▊     | 193241/400000 [00:23<00:24, 8470.46it/s] 49%|████▊     | 194089/400000 [00:23<00:24, 8380.67it/s] 49%|████▊     | 194948/400000 [00:23<00:24, 8440.55it/s] 49%|████▉     | 195793/400000 [00:23<00:24, 8375.57it/s] 49%|████▉     | 196631/400000 [00:23<00:24, 8283.98it/s] 49%|████▉     | 197460/400000 [00:23<00:24, 8277.60it/s] 50%|████▉     | 198299/400000 [00:23<00:24, 8309.95it/s] 50%|████▉     | 199144/400000 [00:24<00:24, 8348.94it/s] 50%|████▉     | 199984/400000 [00:24<00:23, 8363.78it/s] 50%|█████     | 200836/400000 [00:24<00:23, 8408.16it/s] 50%|█████     | 201677/400000 [00:24<00:24, 8259.47it/s] 51%|█████     | 202533/400000 [00:24<00:23, 8345.35it/s] 51%|█████     | 203402/400000 [00:24<00:23, 8444.16it/s] 51%|█████     | 204262/400000 [00:24<00:23, 8489.95it/s] 51%|█████▏    | 205132/400000 [00:24<00:22, 8550.74it/s] 51%|█████▏    | 205991/400000 [00:24<00:22, 8561.81it/s] 52%|█████▏    | 206848/400000 [00:25<00:22, 8456.64it/s] 52%|█████▏    | 207695/400000 [00:25<00:23, 8160.24it/s] 52%|█████▏    | 208534/400000 [00:25<00:23, 8225.57it/s] 52%|█████▏    | 209401/400000 [00:25<00:22, 8351.42it/s] 53%|█████▎    | 210258/400000 [00:25<00:22, 8415.51it/s] 53%|█████▎    | 211121/400000 [00:25<00:22, 8478.46it/s] 53%|█████▎    | 211977/400000 [00:25<00:22, 8499.88it/s] 53%|█████▎    | 212828/400000 [00:25<00:22, 8488.15it/s] 53%|█████▎    | 213678/400000 [00:25<00:22, 8426.65it/s] 54%|█████▎    | 214522/400000 [00:25<00:22, 8428.55it/s] 54%|█████▍    | 215366/400000 [00:26<00:21, 8406.79it/s] 54%|█████▍    | 216229/400000 [00:26<00:21, 8469.75it/s] 54%|█████▍    | 217084/400000 [00:26<00:21, 8490.78it/s] 54%|█████▍    | 217951/400000 [00:26<00:21, 8542.84it/s] 55%|█████▍    | 218806/400000 [00:26<00:21, 8500.96it/s] 55%|█████▍    | 219666/400000 [00:26<00:21, 8528.02it/s] 55%|█████▌    | 220519/400000 [00:26<00:21, 8354.18it/s] 55%|█████▌    | 221356/400000 [00:26<00:21, 8289.34it/s] 56%|█████▌    | 222219/400000 [00:26<00:21, 8388.30it/s] 56%|█████▌    | 223059/400000 [00:26<00:21, 8170.64it/s] 56%|█████▌    | 223878/400000 [00:27<00:21, 8076.36it/s] 56%|█████▌    | 224696/400000 [00:27<00:21, 8105.10it/s] 56%|█████▋    | 225508/400000 [00:27<00:21, 8052.95it/s] 57%|█████▋    | 226359/400000 [00:27<00:21, 8184.20it/s] 57%|█████▋    | 227205/400000 [00:27<00:20, 8263.82it/s] 57%|█████▋    | 228068/400000 [00:27<00:20, 8368.19it/s] 57%|█████▋    | 228915/400000 [00:27<00:20, 8398.21it/s] 57%|█████▋    | 229761/400000 [00:27<00:20, 8416.20it/s] 58%|█████▊    | 230613/400000 [00:27<00:20, 8447.04it/s] 58%|█████▊    | 231459/400000 [00:27<00:20, 8132.51it/s] 58%|█████▊    | 232276/400000 [00:28<00:20, 8104.21it/s] 58%|█████▊    | 233104/400000 [00:28<00:20, 8154.69it/s] 58%|█████▊    | 233949/400000 [00:28<00:20, 8239.97it/s] 59%|█████▊    | 234802/400000 [00:28<00:19, 8323.99it/s] 59%|█████▉    | 235636/400000 [00:28<00:19, 8312.38it/s] 59%|█████▉    | 236468/400000 [00:28<00:19, 8230.88it/s] 59%|█████▉    | 237298/400000 [00:28<00:19, 8249.84it/s] 60%|█████▉    | 238148/400000 [00:28<00:19, 8322.58it/s] 60%|█████▉    | 238985/400000 [00:28<00:19, 8335.39it/s] 60%|█████▉    | 239834/400000 [00:28<00:19, 8380.89it/s] 60%|██████    | 240691/400000 [00:29<00:18, 8434.65it/s] 60%|██████    | 241535/400000 [00:29<00:18, 8394.18it/s] 61%|██████    | 242398/400000 [00:29<00:18, 8463.43it/s] 61%|██████    | 243250/400000 [00:29<00:18, 8479.07it/s] 61%|██████    | 244099/400000 [00:29<00:18, 8465.47it/s] 61%|██████    | 244959/400000 [00:29<00:18, 8502.98it/s] 61%|██████▏   | 245826/400000 [00:29<00:18, 8549.60it/s] 62%|██████▏   | 246683/400000 [00:29<00:17, 8555.19it/s] 62%|██████▏   | 247539/400000 [00:29<00:17, 8506.78it/s] 62%|██████▏   | 248390/400000 [00:29<00:17, 8507.41it/s] 62%|██████▏   | 249241/400000 [00:30<00:17, 8483.50it/s] 63%|██████▎   | 250108/400000 [00:30<00:17, 8534.70it/s] 63%|██████▎   | 250962/400000 [00:30<00:17, 8464.93it/s] 63%|██████▎   | 251809/400000 [00:30<00:17, 8305.97it/s] 63%|██████▎   | 252666/400000 [00:30<00:17, 8382.01it/s] 63%|██████▎   | 253524/400000 [00:30<00:17, 8439.61it/s] 64%|██████▎   | 254369/400000 [00:30<00:17, 8388.02it/s] 64%|██████▍   | 255209/400000 [00:30<00:17, 8345.22it/s] 64%|██████▍   | 256044/400000 [00:30<00:17, 8243.90it/s] 64%|██████▍   | 256898/400000 [00:30<00:17, 8327.77it/s] 64%|██████▍   | 257768/400000 [00:31<00:16, 8434.13it/s] 65%|██████▍   | 258628/400000 [00:31<00:16, 8481.54it/s] 65%|██████▍   | 259495/400000 [00:31<00:16, 8535.47it/s] 65%|██████▌   | 260350/400000 [00:31<00:16, 8381.02it/s] 65%|██████▌   | 261190/400000 [00:31<00:16, 8353.52it/s] 66%|██████▌   | 262044/400000 [00:31<00:16, 8406.18it/s] 66%|██████▌   | 262904/400000 [00:31<00:16, 8462.82it/s] 66%|██████▌   | 263756/400000 [00:31<00:16, 8411.37it/s] 66%|██████▌   | 264598/400000 [00:31<00:16, 8379.34it/s] 66%|██████▋   | 265437/400000 [00:32<00:16, 8358.06it/s] 67%|██████▋   | 266274/400000 [00:32<00:16, 8285.93it/s] 67%|██████▋   | 267103/400000 [00:32<00:16, 8264.51it/s] 67%|██████▋   | 267930/400000 [00:32<00:16, 8243.50it/s] 67%|██████▋   | 268757/400000 [00:32<00:15, 8248.73it/s] 67%|██████▋   | 269622/400000 [00:32<00:15, 8363.49it/s] 68%|██████▊   | 270469/400000 [00:32<00:15, 8395.13it/s] 68%|██████▊   | 271317/400000 [00:32<00:15, 8419.45it/s] 68%|██████▊   | 272160/400000 [00:32<00:15, 8274.34it/s] 68%|██████▊   | 273013/400000 [00:32<00:15, 8346.71it/s] 68%|██████▊   | 273880/400000 [00:33<00:14, 8440.75it/s] 69%|██████▊   | 274725/400000 [00:33<00:14, 8390.33it/s] 69%|██████▉   | 275565/400000 [00:33<00:14, 8382.42it/s] 69%|██████▉   | 276433/400000 [00:33<00:14, 8467.59it/s] 69%|██████▉   | 277282/400000 [00:33<00:14, 8472.27it/s] 70%|██████▉   | 278130/400000 [00:33<00:14, 8468.16it/s] 70%|██████▉   | 278980/400000 [00:33<00:14, 8475.29it/s] 70%|██████▉   | 279828/400000 [00:33<00:14, 8428.46it/s] 70%|███████   | 280675/400000 [00:33<00:14, 8440.56it/s] 70%|███████   | 281520/400000 [00:33<00:14, 8441.02it/s] 71%|███████   | 282365/400000 [00:34<00:14, 8389.00it/s] 71%|███████   | 283220/400000 [00:34<00:13, 8436.13it/s] 71%|███████   | 284082/400000 [00:34<00:13, 8490.43it/s] 71%|███████   | 284953/400000 [00:34<00:13, 8553.28it/s] 71%|███████▏  | 285809/400000 [00:34<00:13, 8508.25it/s] 72%|███████▏  | 286662/400000 [00:34<00:13, 8513.58it/s] 72%|███████▏  | 287514/400000 [00:34<00:13, 8500.07it/s] 72%|███████▏  | 288374/400000 [00:34<00:13, 8528.38it/s] 72%|███████▏  | 289227/400000 [00:34<00:13, 8204.44it/s] 73%|███████▎  | 290084/400000 [00:34<00:13, 8308.18it/s] 73%|███████▎  | 290951/400000 [00:35<00:12, 8412.53it/s] 73%|███████▎  | 291795/400000 [00:35<00:12, 8385.89it/s] 73%|███████▎  | 292663/400000 [00:35<00:12, 8469.79it/s] 73%|███████▎  | 293519/400000 [00:35<00:12, 8493.58it/s] 74%|███████▎  | 294370/400000 [00:35<00:12, 8491.16it/s] 74%|███████▍  | 295220/400000 [00:35<00:12, 8449.96it/s] 74%|███████▍  | 296075/400000 [00:35<00:12, 8476.22it/s] 74%|███████▍  | 296944/400000 [00:35<00:12, 8536.56it/s] 74%|███████▍  | 297798/400000 [00:35<00:11, 8517.74it/s] 75%|███████▍  | 298651/400000 [00:35<00:11, 8517.99it/s] 75%|███████▍  | 299516/400000 [00:36<00:11, 8556.10it/s] 75%|███████▌  | 300372/400000 [00:36<00:11, 8552.05it/s] 75%|███████▌  | 301241/400000 [00:36<00:11, 8591.80it/s] 76%|███████▌  | 302101/400000 [00:36<00:11, 8508.67it/s] 76%|███████▌  | 302953/400000 [00:36<00:11, 8408.74it/s] 76%|███████▌  | 303795/400000 [00:36<00:11, 8378.30it/s] 76%|███████▌  | 304634/400000 [00:36<00:11, 8275.81it/s] 76%|███████▋  | 305463/400000 [00:36<00:11, 8162.62it/s] 77%|███████▋  | 306280/400000 [00:36<00:11, 8062.86it/s] 77%|███████▋  | 307097/400000 [00:36<00:11, 8092.03it/s] 77%|███████▋  | 307961/400000 [00:37<00:11, 8246.39it/s] 77%|███████▋  | 308787/400000 [00:37<00:11, 8225.69it/s] 77%|███████▋  | 309645/400000 [00:37<00:10, 8327.55it/s] 78%|███████▊  | 310488/400000 [00:37<00:10, 8356.35it/s] 78%|███████▊  | 311325/400000 [00:37<00:10, 8326.74it/s] 78%|███████▊  | 312196/400000 [00:37<00:10, 8435.83it/s] 78%|███████▊  | 313061/400000 [00:37<00:10, 8498.34it/s] 78%|███████▊  | 313925/400000 [00:37<00:10, 8539.19it/s] 79%|███████▊  | 314796/400000 [00:37<00:09, 8588.02it/s] 79%|███████▉  | 315664/400000 [00:37<00:09, 8614.18it/s] 79%|███████▉  | 316526/400000 [00:38<00:09, 8610.78it/s] 79%|███████▉  | 317388/400000 [00:38<00:09, 8570.96it/s] 80%|███████▉  | 318246/400000 [00:38<00:09, 8557.40it/s] 80%|███████▉  | 319102/400000 [00:38<00:09, 8411.97it/s] 80%|███████▉  | 319960/400000 [00:38<00:09, 8459.99it/s] 80%|████████  | 320807/400000 [00:38<00:09, 8439.42it/s] 80%|████████  | 321652/400000 [00:38<00:09, 8411.01it/s] 81%|████████  | 322497/400000 [00:38<00:09, 8421.56it/s] 81%|████████  | 323340/400000 [00:38<00:09, 8398.84it/s] 81%|████████  | 324181/400000 [00:38<00:09, 8301.18it/s] 81%|████████▏ | 325037/400000 [00:39<00:08, 8375.54it/s] 81%|████████▏ | 325880/400000 [00:39<00:08, 8389.10it/s] 82%|████████▏ | 326735/400000 [00:39<00:08, 8433.97it/s] 82%|████████▏ | 327579/400000 [00:39<00:08, 8398.89it/s] 82%|████████▏ | 328440/400000 [00:39<00:08, 8459.00it/s] 82%|████████▏ | 329287/400000 [00:39<00:08, 8448.30it/s] 83%|████████▎ | 330142/400000 [00:39<00:08, 8476.34it/s] 83%|████████▎ | 331013/400000 [00:39<00:08, 8542.99it/s] 83%|████████▎ | 331868/400000 [00:39<00:08, 8417.02it/s] 83%|████████▎ | 332731/400000 [00:39<00:07, 8477.72it/s] 83%|████████▎ | 333595/400000 [00:40<00:07, 8524.80it/s] 84%|████████▎ | 334448/400000 [00:40<00:07, 8513.53it/s] 84%|████████▍ | 335318/400000 [00:40<00:07, 8566.51it/s] 84%|████████▍ | 336175/400000 [00:40<00:07, 8431.11it/s] 84%|████████▍ | 337039/400000 [00:40<00:07, 8490.78it/s] 84%|████████▍ | 337889/400000 [00:40<00:07, 8361.67it/s] 85%|████████▍ | 338727/400000 [00:40<00:07, 8300.54it/s] 85%|████████▍ | 339582/400000 [00:40<00:07, 8373.36it/s] 85%|████████▌ | 340451/400000 [00:40<00:07, 8464.76it/s] 85%|████████▌ | 341299/400000 [00:41<00:07, 8343.51it/s] 86%|████████▌ | 342165/400000 [00:41<00:06, 8434.19it/s] 86%|████████▌ | 343015/400000 [00:41<00:06, 8451.28it/s] 86%|████████▌ | 343873/400000 [00:41<00:06, 8489.38it/s] 86%|████████▌ | 344726/400000 [00:41<00:06, 8499.00it/s] 86%|████████▋ | 345581/400000 [00:41<00:06, 8512.54it/s] 87%|████████▋ | 346433/400000 [00:41<00:06, 8498.67it/s] 87%|████████▋ | 347284/400000 [00:41<00:06, 8472.63it/s] 87%|████████▋ | 348132/400000 [00:41<00:06, 8447.79it/s] 87%|████████▋ | 348977/400000 [00:41<00:06, 8364.09it/s] 87%|████████▋ | 349814/400000 [00:42<00:06, 8305.80it/s] 88%|████████▊ | 350675/400000 [00:42<00:05, 8392.68it/s] 88%|████████▊ | 351527/400000 [00:42<00:05, 8428.69it/s] 88%|████████▊ | 352371/400000 [00:42<00:05, 8285.28it/s] 88%|████████▊ | 353201/400000 [00:42<00:05, 8274.83it/s] 89%|████████▊ | 354030/400000 [00:42<00:05, 8201.25it/s] 89%|████████▊ | 354851/400000 [00:42<00:05, 8175.51it/s] 89%|████████▉ | 355669/400000 [00:42<00:05, 8138.82it/s] 89%|████████▉ | 356490/400000 [00:42<00:05, 8157.43it/s] 89%|████████▉ | 357308/400000 [00:42<00:05, 8163.85it/s] 90%|████████▉ | 358125/400000 [00:43<00:05, 8113.21it/s] 90%|████████▉ | 358943/400000 [00:43<00:05, 8130.95it/s] 90%|████████▉ | 359757/400000 [00:43<00:04, 8121.04it/s] 90%|█████████ | 360573/400000 [00:43<00:04, 8131.87it/s] 90%|█████████ | 361387/400000 [00:43<00:04, 8133.30it/s] 91%|█████████ | 362203/400000 [00:43<00:04, 8140.00it/s] 91%|█████████ | 363023/400000 [00:43<00:04, 8157.04it/s] 91%|█████████ | 363862/400000 [00:43<00:04, 8224.89it/s] 91%|█████████ | 364685/400000 [00:43<00:04, 8179.09it/s] 91%|█████████▏| 365506/400000 [00:43<00:04, 8188.17it/s] 92%|█████████▏| 366325/400000 [00:44<00:04, 8050.71it/s] 92%|█████████▏| 367162/400000 [00:44<00:04, 8142.48it/s] 92%|█████████▏| 367977/400000 [00:44<00:03, 8132.11it/s] 92%|█████████▏| 368811/400000 [00:44<00:03, 8191.63it/s] 92%|█████████▏| 369644/400000 [00:44<00:03, 8232.21it/s] 93%|█████████▎| 370475/400000 [00:44<00:03, 8253.17it/s] 93%|█████████▎| 371325/400000 [00:44<00:03, 8324.06it/s] 93%|█████████▎| 372181/400000 [00:44<00:03, 8393.29it/s] 93%|█████████▎| 373050/400000 [00:44<00:03, 8479.18it/s] 93%|█████████▎| 373915/400000 [00:44<00:03, 8529.67it/s] 94%|█████████▎| 374780/400000 [00:45<00:02, 8562.67it/s] 94%|█████████▍| 375637/400000 [00:45<00:02, 8498.17it/s] 94%|█████████▍| 376488/400000 [00:45<00:02, 8437.30it/s] 94%|█████████▍| 377333/400000 [00:45<00:02, 8414.11it/s] 95%|█████████▍| 378178/400000 [00:45<00:02, 8424.76it/s] 95%|█████████▍| 379027/400000 [00:45<00:02, 8443.01it/s] 95%|█████████▍| 379878/400000 [00:45<00:02, 8462.65it/s] 95%|█████████▌| 380725/400000 [00:45<00:02, 8443.86it/s] 95%|█████████▌| 381580/400000 [00:45<00:02, 8473.27it/s] 96%|█████████▌| 382437/400000 [00:45<00:02, 8501.42it/s] 96%|█████████▌| 383293/400000 [00:46<00:01, 8516.59it/s] 96%|█████████▌| 384145/400000 [00:46<00:01, 8508.18it/s] 96%|█████████▌| 384996/400000 [00:46<00:01, 8500.14it/s] 96%|█████████▋| 385847/400000 [00:46<00:01, 8411.33it/s] 97%|█████████▋| 386689/400000 [00:46<00:01, 8276.25it/s] 97%|█████████▋| 387518/400000 [00:46<00:01, 8232.59it/s] 97%|█████████▋| 388381/400000 [00:46<00:01, 8347.91it/s] 97%|█████████▋| 389221/400000 [00:46<00:01, 8363.47it/s] 98%|█████████▊| 390085/400000 [00:46<00:01, 8442.56it/s] 98%|█████████▊| 390930/400000 [00:46<00:01, 8405.30it/s] 98%|█████████▊| 391782/400000 [00:47<00:00, 8437.22it/s] 98%|█████████▊| 392627/400000 [00:47<00:00, 8427.87it/s] 98%|█████████▊| 393471/400000 [00:47<00:00, 8418.17it/s] 99%|█████████▊| 394313/400000 [00:47<00:00, 8406.89it/s] 99%|█████████▉| 395164/400000 [00:47<00:00, 8436.57it/s] 99%|█████████▉| 396008/400000 [00:47<00:00, 8437.37it/s] 99%|█████████▉| 396852/400000 [00:47<00:00, 8282.64it/s] 99%|█████████▉| 397681/400000 [00:47<00:00, 8153.32it/s]100%|█████████▉| 398500/400000 [00:47<00:00, 8153.20it/s]100%|█████████▉| 399331/400000 [00:47<00:00, 8197.40it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8322.91it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f15740600f0>, <torchtext.data.dataset.TabularDataset object at 0x7f1574060240>, <torchtext.vocab.Vocab object at 0x7f1574060160>), {}) 



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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f0d84f02378> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f0d84f02378> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f0df6f4f0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f0df6f4f0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f0d8e6e19d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f0d8e6e19d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0da48338c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0da48338c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0da48338c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:13, 135407.04it/s] 81%|████████  | 8011776/9912422 [00:00<00:09, 193298.30it/s]9920512it [00:00, 42016319.12it/s]                           
0it [00:00, ?it/s]32768it [00:00, 438859.52it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7898448.86it/s]          
0it [00:00, ?it/s]8192it [00:00, 182995.26it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d84f06978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d8724ada0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f0da4833510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f0da4833510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f0da4833510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:14,  2.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:14,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:13,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:13,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:13,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:12,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:12,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:11,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:50,  3.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:50,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:50,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:49,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:49,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:49,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:48,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:48,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:48,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:47,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:33,  4.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:33,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:33,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:33,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:33,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:32,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:32,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:32,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:32,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:31,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:22,  5.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:22,  5.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:22,  5.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  5.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:21,  5.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:21,  5.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:21,  5.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:00<00:10, 11.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:10, 11.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:10, 11.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 11.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 11.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 15.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 15.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 15.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 15.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 15.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 20.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 20.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 32.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 40.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 40.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 40.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 40.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 40.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 40.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 40.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 40.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 40.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 40.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 47.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 47.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 47.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 47.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 47.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 47.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 47.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 47.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 47.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 47.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 54.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 54.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 54.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 54.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 54.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 54.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 54.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 54.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 54.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 60.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 60.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 60.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 60.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 60.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 60.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 60.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 60.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 60.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 60.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 66.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 66.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 66.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 66.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 66.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 66.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 66.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 70.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 70.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 70.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 70.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 70.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 70.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 73.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 75.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 75.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 75.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.98 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.90s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 77.98 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.90s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.90s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.07 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.90s/ url]
0 examples [00:00, ? examples/s]2020-05-29 12:11:45.550321: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-29 12:11:45.554799: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-29 12:11:45.554971: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564932196e20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-29 12:11:45.554989: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
56 examples [00:00, 556.37 examples/s]145 examples [00:00, 626.84 examples/s]234 examples [00:00, 687.65 examples/s]319 examples [00:00, 728.00 examples/s]403 examples [00:00, 757.42 examples/s]491 examples [00:00, 789.42 examples/s]580 examples [00:00, 814.76 examples/s]667 examples [00:00, 828.37 examples/s]757 examples [00:00, 848.22 examples/s]846 examples [00:01, 859.73 examples/s]934 examples [00:01, 864.49 examples/s]1024 examples [00:01, 872.79 examples/s]1113 examples [00:01, 877.12 examples/s]1202 examples [00:01, 879.02 examples/s]1290 examples [00:01, 878.93 examples/s]1378 examples [00:01, 875.50 examples/s]1466 examples [00:01, 862.99 examples/s]1553 examples [00:01, 849.36 examples/s]1643 examples [00:01, 863.75 examples/s]1733 examples [00:02, 873.17 examples/s]1824 examples [00:02, 882.94 examples/s]1918 examples [00:02, 897.15 examples/s]2008 examples [00:02, 894.84 examples/s]2098 examples [00:02, 872.24 examples/s]2186 examples [00:02, 869.22 examples/s]2274 examples [00:02, 868.44 examples/s]2370 examples [00:02, 892.58 examples/s]2463 examples [00:02, 900.90 examples/s]2554 examples [00:02, 880.20 examples/s]2643 examples [00:03, 876.69 examples/s]2732 examples [00:03, 880.09 examples/s]2821 examples [00:03, 878.46 examples/s]2909 examples [00:03, 873.68 examples/s]2997 examples [00:03, 854.76 examples/s]3087 examples [00:03, 865.34 examples/s]3174 examples [00:03, 861.28 examples/s]3264 examples [00:03, 870.74 examples/s]3356 examples [00:03, 883.88 examples/s]3445 examples [00:03, 885.57 examples/s]3534 examples [00:04, 868.27 examples/s]3622 examples [00:04, 871.42 examples/s]3710 examples [00:04, 829.96 examples/s]3798 examples [00:04, 843.88 examples/s]3885 examples [00:04, 850.31 examples/s]3971 examples [00:04, 851.07 examples/s]4058 examples [00:04, 855.11 examples/s]4144 examples [00:04, 842.66 examples/s]4232 examples [00:04, 850.89 examples/s]4321 examples [00:04, 861.40 examples/s]4410 examples [00:05, 867.90 examples/s]4500 examples [00:05, 876.30 examples/s]4588 examples [00:05, 872.41 examples/s]4676 examples [00:05, 870.26 examples/s]4765 examples [00:05, 874.76 examples/s]4854 examples [00:05, 878.05 examples/s]4942 examples [00:05, 875.73 examples/s]5032 examples [00:05, 881.19 examples/s]5123 examples [00:05, 886.52 examples/s]5212 examples [00:06, 885.66 examples/s]5302 examples [00:06, 887.43 examples/s]5392 examples [00:06, 888.77 examples/s]5482 examples [00:06, 891.14 examples/s]5573 examples [00:06, 894.16 examples/s]5663 examples [00:06, 894.36 examples/s]5753 examples [00:06, 887.55 examples/s]5846 examples [00:06, 899.27 examples/s]5939 examples [00:06, 906.34 examples/s]6030 examples [00:06, 898.02 examples/s]6120 examples [00:07, 891.85 examples/s]6210 examples [00:07, 885.37 examples/s]6299 examples [00:07, 883.24 examples/s]6390 examples [00:07, 890.25 examples/s]6480 examples [00:07, 881.75 examples/s]6569 examples [00:07, 881.40 examples/s]6659 examples [00:07, 884.63 examples/s]6753 examples [00:07, 897.69 examples/s]6843 examples [00:07, 887.79 examples/s]6932 examples [00:07, 871.33 examples/s]7025 examples [00:08, 885.81 examples/s]7120 examples [00:08, 902.30 examples/s]7212 examples [00:08, 907.30 examples/s]7305 examples [00:08, 913.55 examples/s]7397 examples [00:08, 909.80 examples/s]7489 examples [00:08, 903.75 examples/s]7580 examples [00:08, 894.23 examples/s]7670 examples [00:08, 895.52 examples/s]7763 examples [00:08, 903.20 examples/s]7854 examples [00:08, 905.15 examples/s]7945 examples [00:09, 905.05 examples/s]8038 examples [00:09, 912.36 examples/s]8130 examples [00:09, 896.21 examples/s]8221 examples [00:09, 898.29 examples/s]8311 examples [00:09, 891.61 examples/s]8401 examples [00:09, 880.73 examples/s]8490 examples [00:09, 883.37 examples/s]8583 examples [00:09, 895.93 examples/s]8673 examples [00:09, 890.21 examples/s]8763 examples [00:09, 882.89 examples/s]8852 examples [00:10, 875.78 examples/s]8942 examples [00:10, 880.31 examples/s]9034 examples [00:10, 891.68 examples/s]9124 examples [00:10, 892.52 examples/s]9214 examples [00:10, 886.86 examples/s]9303 examples [00:10, 878.71 examples/s]9392 examples [00:10, 880.28 examples/s]9483 examples [00:10, 888.85 examples/s]9572 examples [00:10, 877.68 examples/s]9660 examples [00:11, 867.34 examples/s]9751 examples [00:11, 879.50 examples/s]9842 examples [00:11, 886.14 examples/s]9936 examples [00:11, 901.55 examples/s]10027 examples [00:11, 848.28 examples/s]10121 examples [00:11, 872.96 examples/s]10210 examples [00:11, 877.27 examples/s]10299 examples [00:11, 873.22 examples/s]10390 examples [00:11, 882.18 examples/s]10481 examples [00:11, 887.85 examples/s]10572 examples [00:12, 891.80 examples/s]10662 examples [00:12, 888.91 examples/s]10755 examples [00:12, 899.68 examples/s]10846 examples [00:12, 895.30 examples/s]10936 examples [00:12, 893.90 examples/s]11026 examples [00:12, 886.09 examples/s]11115 examples [00:12, 879.18 examples/s]11206 examples [00:12, 885.36 examples/s]11298 examples [00:12, 893.08 examples/s]11388 examples [00:12, 894.59 examples/s]11478 examples [00:13, 895.24 examples/s]11569 examples [00:13, 897.63 examples/s]11662 examples [00:13, 905.25 examples/s]11753 examples [00:13, 883.20 examples/s]11842 examples [00:13, 884.17 examples/s]11933 examples [00:13, 891.31 examples/s]12027 examples [00:13, 904.52 examples/s]12118 examples [00:13, 892.07 examples/s]12208 examples [00:13, 879.48 examples/s]12297 examples [00:13, 849.57 examples/s]12383 examples [00:14, 841.48 examples/s]12468 examples [00:14, 842.26 examples/s]12553 examples [00:14, 837.29 examples/s]12643 examples [00:14, 854.15 examples/s]12731 examples [00:14, 861.54 examples/s]12821 examples [00:14, 870.22 examples/s]12909 examples [00:14, 870.11 examples/s]12997 examples [00:14, 862.47 examples/s]13084 examples [00:14, 850.30 examples/s]13171 examples [00:15, 855.21 examples/s]13263 examples [00:15, 872.75 examples/s]13351 examples [00:15, 872.69 examples/s]13443 examples [00:15, 884.41 examples/s]13532 examples [00:15, 881.42 examples/s]13621 examples [00:15, 876.45 examples/s]13709 examples [00:15, 871.92 examples/s]13797 examples [00:15, 867.66 examples/s]13885 examples [00:15, 871.07 examples/s]13975 examples [00:15, 878.24 examples/s]14064 examples [00:16, 880.12 examples/s]14158 examples [00:16, 894.58 examples/s]14248 examples [00:16, 891.90 examples/s]14338 examples [00:16, 892.22 examples/s]14428 examples [00:16, 890.04 examples/s]14519 examples [00:16, 892.99 examples/s]14609 examples [00:16, 894.83 examples/s]14699 examples [00:16, 888.85 examples/s]14789 examples [00:16, 891.41 examples/s]14879 examples [00:16, 893.87 examples/s]14969 examples [00:17, 884.70 examples/s]15062 examples [00:17, 896.28 examples/s]15152 examples [00:17, 892.62 examples/s]15246 examples [00:17, 903.41 examples/s]15339 examples [00:17, 908.97 examples/s]15430 examples [00:17, 890.70 examples/s]15520 examples [00:17, 888.01 examples/s]15609 examples [00:17, 885.42 examples/s]15699 examples [00:17, 887.91 examples/s]15789 examples [00:17, 889.17 examples/s]15878 examples [00:18, 886.69 examples/s]15967 examples [00:18, 884.17 examples/s]16057 examples [00:18, 888.21 examples/s]16148 examples [00:18, 893.22 examples/s]16238 examples [00:18, 888.77 examples/s]16327 examples [00:18, 874.31 examples/s]16418 examples [00:18, 882.96 examples/s]16509 examples [00:18, 889.92 examples/s]16602 examples [00:18, 900.62 examples/s]16695 examples [00:18, 907.49 examples/s]16786 examples [00:19, 893.25 examples/s]16876 examples [00:19, 884.15 examples/s]16973 examples [00:19, 907.71 examples/s]17065 examples [00:19, 910.48 examples/s]17157 examples [00:19, 911.25 examples/s]17249 examples [00:19, 899.26 examples/s]17340 examples [00:19, 880.49 examples/s]17432 examples [00:19, 889.31 examples/s]17524 examples [00:19, 896.98 examples/s]17615 examples [00:19, 898.30 examples/s]17708 examples [00:20, 906.14 examples/s]17799 examples [00:20, 890.96 examples/s]17889 examples [00:20, 891.03 examples/s]17979 examples [00:20, 888.95 examples/s]18068 examples [00:20, 889.22 examples/s]18157 examples [00:20, 884.56 examples/s]18246 examples [00:20, 882.48 examples/s]18335 examples [00:20, 883.82 examples/s]18424 examples [00:20, 853.92 examples/s]18514 examples [00:21, 865.89 examples/s]18606 examples [00:21, 879.06 examples/s]18695 examples [00:21, 873.05 examples/s]18787 examples [00:21, 886.50 examples/s]18878 examples [00:21, 890.98 examples/s]18968 examples [00:21, 880.95 examples/s]19057 examples [00:21, 874.70 examples/s]19145 examples [00:21, 872.80 examples/s]19234 examples [00:21, 877.37 examples/s]19322 examples [00:21, 870.07 examples/s]19410 examples [00:22, 872.00 examples/s]19500 examples [00:22, 877.37 examples/s]19588 examples [00:22, 875.81 examples/s]19676 examples [00:22, 874.16 examples/s]19764 examples [00:22, 874.99 examples/s]19855 examples [00:22, 884.06 examples/s]19944 examples [00:22, 881.77 examples/s]20033 examples [00:22, 822.69 examples/s]20120 examples [00:22, 836.25 examples/s]20213 examples [00:22, 862.21 examples/s]20301 examples [00:23, 864.77 examples/s]20390 examples [00:23, 871.42 examples/s]20478 examples [00:23, 857.43 examples/s]20571 examples [00:23, 875.63 examples/s]20659 examples [00:23, 875.78 examples/s]20747 examples [00:23, 861.42 examples/s]20834 examples [00:23, 844.07 examples/s]20920 examples [00:23, 846.85 examples/s]21006 examples [00:23, 849.75 examples/s]21092 examples [00:23, 843.21 examples/s]21181 examples [00:24, 854.82 examples/s]21267 examples [00:24, 855.60 examples/s]21355 examples [00:24, 862.15 examples/s]21445 examples [00:24, 873.08 examples/s]21533 examples [00:24, 871.01 examples/s]21621 examples [00:24, 870.71 examples/s]21709 examples [00:24, 868.83 examples/s]21798 examples [00:24, 873.42 examples/s]21886 examples [00:24, 875.07 examples/s]21975 examples [00:24, 877.34 examples/s]22063 examples [00:25, 860.41 examples/s]22151 examples [00:25, 864.74 examples/s]22240 examples [00:25, 870.16 examples/s]22329 examples [00:25, 874.47 examples/s]22419 examples [00:25, 880.83 examples/s]22508 examples [00:25, 856.71 examples/s]22594 examples [00:25, 854.14 examples/s]22680 examples [00:25, 854.23 examples/s]22766 examples [00:25, 850.67 examples/s]22856 examples [00:26, 863.59 examples/s]22944 examples [00:26, 865.37 examples/s]23031 examples [00:26, 860.74 examples/s]23118 examples [00:26, 838.48 examples/s]23203 examples [00:26, 838.04 examples/s]23289 examples [00:26, 841.91 examples/s]23374 examples [00:26, 808.59 examples/s]23462 examples [00:26, 827.27 examples/s]23553 examples [00:26, 848.61 examples/s]23642 examples [00:26, 858.99 examples/s]23730 examples [00:27, 864.90 examples/s]23818 examples [00:27, 866.95 examples/s]23906 examples [00:27, 869.77 examples/s]23994 examples [00:27, 843.29 examples/s]24082 examples [00:27, 851.60 examples/s]24168 examples [00:27, 850.19 examples/s]24254 examples [00:27, 850.95 examples/s]24340 examples [00:27, 833.07 examples/s]24424 examples [00:27, 827.64 examples/s]24507 examples [00:27, 825.47 examples/s]24590 examples [00:28, 817.54 examples/s]24675 examples [00:28, 825.01 examples/s]24764 examples [00:28, 841.80 examples/s]24849 examples [00:28, 828.68 examples/s]24937 examples [00:28, 842.03 examples/s]25025 examples [00:28, 850.87 examples/s]25112 examples [00:28, 854.51 examples/s]25199 examples [00:28, 859.03 examples/s]25285 examples [00:28, 852.41 examples/s]25372 examples [00:29, 855.64 examples/s]25460 examples [00:29, 862.52 examples/s]25549 examples [00:29, 868.18 examples/s]25637 examples [00:29, 870.09 examples/s]25725 examples [00:29, 842.29 examples/s]25815 examples [00:29, 850.49 examples/s]25902 examples [00:29, 855.23 examples/s]25989 examples [00:29, 858.60 examples/s]26075 examples [00:29, 851.57 examples/s]26163 examples [00:29, 858.12 examples/s]26250 examples [00:30, 860.63 examples/s]26341 examples [00:30, 871.78 examples/s]26429 examples [00:30, 864.63 examples/s]26516 examples [00:30, 856.51 examples/s]26602 examples [00:30, 845.73 examples/s]26687 examples [00:30, 842.17 examples/s]26774 examples [00:30, 850.21 examples/s]26860 examples [00:30, 847.57 examples/s]26946 examples [00:30, 850.98 examples/s]27032 examples [00:30, 850.47 examples/s]27118 examples [00:31, 833.94 examples/s]27204 examples [00:31, 840.69 examples/s]27289 examples [00:31, 839.62 examples/s]27374 examples [00:31, 839.91 examples/s]27468 examples [00:31, 867.59 examples/s]27558 examples [00:31, 875.21 examples/s]27646 examples [00:31, 866.82 examples/s]27733 examples [00:31, 847.22 examples/s]27818 examples [00:31, 847.68 examples/s]27903 examples [00:31, 837.83 examples/s]27987 examples [00:32, 837.48 examples/s]28071 examples [00:32, 829.66 examples/s]28157 examples [00:32, 835.79 examples/s]28242 examples [00:32, 839.55 examples/s]28328 examples [00:32, 843.23 examples/s]28413 examples [00:32, 843.67 examples/s]28498 examples [00:32, 810.97 examples/s]28580 examples [00:32, 812.23 examples/s]28664 examples [00:32, 820.24 examples/s]28750 examples [00:32, 829.51 examples/s]28834 examples [00:33, 823.86 examples/s]28917 examples [00:33, 803.69 examples/s]29003 examples [00:33, 819.48 examples/s]29094 examples [00:33, 844.18 examples/s]29184 examples [00:33, 858.59 examples/s]29272 examples [00:33, 862.28 examples/s]29359 examples [00:33, 849.24 examples/s]29447 examples [00:33, 857.94 examples/s]29536 examples [00:33, 864.52 examples/s]29626 examples [00:34, 873.11 examples/s]29714 examples [00:34, 873.48 examples/s]29804 examples [00:34, 879.35 examples/s]29892 examples [00:34, 870.78 examples/s]29982 examples [00:34, 877.18 examples/s]30070 examples [00:34, 818.28 examples/s]30162 examples [00:34, 843.89 examples/s]30254 examples [00:34, 865.08 examples/s]30342 examples [00:34, 861.95 examples/s]30435 examples [00:34, 878.79 examples/s]30524 examples [00:35, 876.26 examples/s]30612 examples [00:35, 874.99 examples/s]30700 examples [00:35, 832.40 examples/s]30790 examples [00:35, 850.00 examples/s]30876 examples [00:35, 850.52 examples/s]30968 examples [00:35, 870.04 examples/s]31062 examples [00:35, 886.47 examples/s]31154 examples [00:35, 895.60 examples/s]31244 examples [00:35, 896.04 examples/s]31335 examples [00:35, 898.99 examples/s]31426 examples [00:36, 900.18 examples/s]31519 examples [00:36, 905.76 examples/s]31610 examples [00:36, 895.73 examples/s]31705 examples [00:36, 908.55 examples/s]31796 examples [00:36, 905.47 examples/s]31887 examples [00:36, 905.68 examples/s]31978 examples [00:36, 902.28 examples/s]32069 examples [00:36, 888.51 examples/s]32158 examples [00:36, 882.37 examples/s]32247 examples [00:36, 882.43 examples/s]32336 examples [00:37, 873.25 examples/s]32424 examples [00:37, 869.93 examples/s]32512 examples [00:37, 872.11 examples/s]32603 examples [00:37, 881.42 examples/s]32692 examples [00:37, 880.04 examples/s]32783 examples [00:37, 887.15 examples/s]32872 examples [00:37, 883.63 examples/s]32962 examples [00:37, 886.86 examples/s]33051 examples [00:37, 876.44 examples/s]33141 examples [00:38, 882.59 examples/s]33233 examples [00:38, 891.60 examples/s]33328 examples [00:38, 906.65 examples/s]33423 examples [00:38, 918.74 examples/s]33515 examples [00:38, 894.96 examples/s]33611 examples [00:38, 912.42 examples/s]33706 examples [00:38, 920.76 examples/s]33800 examples [00:38, 925.30 examples/s]33893 examples [00:38, 921.40 examples/s]33986 examples [00:38, 909.32 examples/s]34078 examples [00:39, 900.28 examples/s]34169 examples [00:39, 896.76 examples/s]34259 examples [00:39, 870.44 examples/s]34348 examples [00:39, 873.97 examples/s]34442 examples [00:39, 891.59 examples/s]34532 examples [00:39, 882.38 examples/s]34622 examples [00:39, 884.67 examples/s]34711 examples [00:39, 882.03 examples/s]34800 examples [00:39, 867.59 examples/s]34888 examples [00:39, 868.98 examples/s]34975 examples [00:40, 852.67 examples/s]35062 examples [00:40, 856.06 examples/s]35152 examples [00:40, 867.40 examples/s]35243 examples [00:40, 878.40 examples/s]35331 examples [00:40, 874.90 examples/s]35419 examples [00:40, 860.18 examples/s]35511 examples [00:40, 876.15 examples/s]35604 examples [00:40, 889.48 examples/s]35694 examples [00:40, 880.68 examples/s]35783 examples [00:41, 842.31 examples/s]35873 examples [00:41, 857.01 examples/s]35962 examples [00:41, 864.74 examples/s]36052 examples [00:41, 873.56 examples/s]36141 examples [00:41, 877.00 examples/s]36229 examples [00:41, 874.04 examples/s]36318 examples [00:41, 875.91 examples/s]36406 examples [00:41, 872.53 examples/s]36496 examples [00:41, 877.84 examples/s]36584 examples [00:41, 874.98 examples/s]36672 examples [00:42, 830.42 examples/s]36757 examples [00:42, 835.63 examples/s]36843 examples [00:42, 841.04 examples/s]36928 examples [00:42, 828.56 examples/s]37017 examples [00:42, 844.30 examples/s]37105 examples [00:42, 854.34 examples/s]37195 examples [00:42, 867.47 examples/s]37282 examples [00:42, 868.04 examples/s]37369 examples [00:42, 863.66 examples/s]37456 examples [00:42, 857.04 examples/s]37542 examples [00:43, 837.90 examples/s]37626 examples [00:43, 833.53 examples/s]37710 examples [00:43, 829.52 examples/s]37798 examples [00:43, 841.07 examples/s]37883 examples [00:43, 839.20 examples/s]37969 examples [00:43, 844.22 examples/s]38055 examples [00:43, 847.78 examples/s]38140 examples [00:43, 821.57 examples/s]38227 examples [00:43, 833.74 examples/s]38315 examples [00:43, 843.96 examples/s]38401 examples [00:44, 847.91 examples/s]38486 examples [00:44, 841.03 examples/s]38571 examples [00:44, 831.41 examples/s]38655 examples [00:44, 831.97 examples/s]38741 examples [00:44, 837.37 examples/s]38828 examples [00:44, 844.95 examples/s]38914 examples [00:44, 846.77 examples/s]38999 examples [00:44, 837.03 examples/s]39083 examples [00:44, 826.36 examples/s]39170 examples [00:44, 837.35 examples/s]39259 examples [00:45, 851.85 examples/s]39347 examples [00:45, 859.55 examples/s]39434 examples [00:45, 858.58 examples/s]39520 examples [00:45, 818.63 examples/s]39605 examples [00:45, 826.68 examples/s]39693 examples [00:45, 839.73 examples/s]39778 examples [00:45, 831.99 examples/s]39862 examples [00:45, 831.25 examples/s]39950 examples [00:45, 844.05 examples/s]40035 examples [00:46, 800.00 examples/s]40125 examples [00:46, 825.01 examples/s]40210 examples [00:46, 830.81 examples/s]40301 examples [00:46, 852.08 examples/s]40387 examples [00:46, 841.24 examples/s]40476 examples [00:46, 852.83 examples/s]40562 examples [00:46, 851.52 examples/s]40652 examples [00:46, 862.89 examples/s]40743 examples [00:46, 874.17 examples/s]40831 examples [00:46, 872.74 examples/s]40920 examples [00:47, 877.32 examples/s]41008 examples [00:47, 873.22 examples/s]41096 examples [00:47, 868.16 examples/s]41185 examples [00:47, 874.21 examples/s]41273 examples [00:47, 872.45 examples/s]41362 examples [00:47, 875.21 examples/s]41450 examples [00:47, 876.05 examples/s]41538 examples [00:47, 868.95 examples/s]41628 examples [00:47, 876.18 examples/s]41716 examples [00:47, 874.00 examples/s]41804 examples [00:48, 872.57 examples/s]41892 examples [00:48, 871.13 examples/s]41980 examples [00:48, 870.88 examples/s]42068 examples [00:48, 866.15 examples/s]42155 examples [00:48, 822.04 examples/s]42239 examples [00:48, 825.81 examples/s]42325 examples [00:48, 834.76 examples/s]42413 examples [00:48, 846.26 examples/s]42498 examples [00:48, 839.40 examples/s]42589 examples [00:48, 857.35 examples/s]42675 examples [00:49, 838.30 examples/s]42760 examples [00:49, 839.67 examples/s]42848 examples [00:49, 849.52 examples/s]42934 examples [00:49, 848.97 examples/s]43021 examples [00:49, 853.69 examples/s]43107 examples [00:49, 849.09 examples/s]43192 examples [00:49, 834.12 examples/s]43279 examples [00:49, 844.04 examples/s]43364 examples [00:49, 827.28 examples/s]43448 examples [00:50, 829.14 examples/s]43532 examples [00:50, 832.03 examples/s]43616 examples [00:50, 833.48 examples/s]43702 examples [00:50, 838.80 examples/s]43788 examples [00:50, 843.04 examples/s]43874 examples [00:50, 846.48 examples/s]43959 examples [00:50, 816.21 examples/s]44041 examples [00:50, 775.79 examples/s]44128 examples [00:50, 800.43 examples/s]44210 examples [00:50, 804.07 examples/s]44297 examples [00:51, 820.74 examples/s]44383 examples [00:51, 829.50 examples/s]44470 examples [00:51, 839.56 examples/s]44560 examples [00:51, 854.04 examples/s]44646 examples [00:51, 852.79 examples/s]44732 examples [00:51, 826.57 examples/s]44815 examples [00:51, 820.08 examples/s]44899 examples [00:51, 825.33 examples/s]44982 examples [00:51, 813.88 examples/s]45067 examples [00:51, 822.82 examples/s]45150 examples [00:52, 820.44 examples/s]45235 examples [00:52, 826.61 examples/s]45321 examples [00:52, 834.72 examples/s]45405 examples [00:52, 830.80 examples/s]45494 examples [00:52, 846.07 examples/s]45579 examples [00:52, 845.47 examples/s]45664 examples [00:52, 842.85 examples/s]45751 examples [00:52, 850.61 examples/s]45840 examples [00:52, 861.53 examples/s]45928 examples [00:52, 865.71 examples/s]46015 examples [00:53, 861.23 examples/s]46102 examples [00:53, 862.77 examples/s]46189 examples [00:53, 851.75 examples/s]46278 examples [00:53, 860.94 examples/s]46365 examples [00:53, 839.55 examples/s]46450 examples [00:53, 823.30 examples/s]46533 examples [00:53, 816.69 examples/s]46621 examples [00:53, 833.71 examples/s]46710 examples [00:53, 848.67 examples/s]46796 examples [00:54, 841.95 examples/s]46881 examples [00:54, 837.70 examples/s]46966 examples [00:54, 838.98 examples/s]47057 examples [00:54, 857.23 examples/s]47146 examples [00:54, 866.24 examples/s]47233 examples [00:54, 865.39 examples/s]47320 examples [00:54, 820.78 examples/s]47406 examples [00:54, 830.49 examples/s]47492 examples [00:54, 837.16 examples/s]47579 examples [00:54, 845.61 examples/s]47665 examples [00:55, 848.87 examples/s]47752 examples [00:55, 854.29 examples/s]47841 examples [00:55, 862.67 examples/s]47931 examples [00:55, 873.39 examples/s]48019 examples [00:55, 872.72 examples/s]48107 examples [00:55, 871.47 examples/s]48195 examples [00:55, 863.06 examples/s]48282 examples [00:55, 856.86 examples/s]48370 examples [00:55, 861.42 examples/s]48457 examples [00:55, 863.78 examples/s]48546 examples [00:56, 871.17 examples/s]48634 examples [00:56, 868.29 examples/s]48723 examples [00:56, 872.50 examples/s]48811 examples [00:56, 874.18 examples/s]48899 examples [00:56, 865.48 examples/s]48986 examples [00:56, 851.00 examples/s]49072 examples [00:56, 848.29 examples/s]49159 examples [00:56, 853.48 examples/s]49250 examples [00:56, 869.41 examples/s]49338 examples [00:56, 860.57 examples/s]49426 examples [00:57, 863.58 examples/s]49513 examples [00:57, 853.22 examples/s]49599 examples [00:57, 853.13 examples/s]49686 examples [00:57, 858.08 examples/s]49777 examples [00:57, 872.56 examples/s]49865 examples [00:57, 857.68 examples/s]49951 examples [00:57, 840.96 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6530/50000 [00:00<00:00, 65295.55 examples/s] 36%|███▋      | 18191/50000 [00:00<00:00, 75225.23 examples/s] 59%|█████▉    | 29495/50000 [00:00<00:00, 83605.55 examples/s] 82%|████████▏ | 40947/50000 [00:00<00:00, 90971.93 examples/s]                                                               0 examples [00:00, ? examples/s]71 examples [00:00, 709.84 examples/s]163 examples [00:00, 760.81 examples/s]254 examples [00:00, 800.04 examples/s]342 examples [00:00, 821.53 examples/s]434 examples [00:00, 846.27 examples/s]528 examples [00:00, 870.28 examples/s]618 examples [00:00, 877.66 examples/s]701 examples [00:00, 687.52 examples/s]788 examples [00:00, 731.90 examples/s]877 examples [00:01, 771.71 examples/s]967 examples [00:01, 803.67 examples/s]1057 examples [00:01, 828.91 examples/s]1144 examples [00:01, 839.61 examples/s]1231 examples [00:01, 848.28 examples/s]1318 examples [00:01, 850.99 examples/s]1409 examples [00:01, 866.12 examples/s]1500 examples [00:01, 878.63 examples/s]1589 examples [00:01, 875.11 examples/s]1677 examples [00:01, 872.65 examples/s]1765 examples [00:02, 864.70 examples/s]1853 examples [00:02, 868.11 examples/s]1940 examples [00:02, 854.48 examples/s]2026 examples [00:02, 855.13 examples/s]2112 examples [00:02, 837.80 examples/s]2197 examples [00:02, 839.01 examples/s]2285 examples [00:02, 850.01 examples/s]2376 examples [00:02, 867.08 examples/s]2464 examples [00:02, 867.88 examples/s]2554 examples [00:03, 875.09 examples/s]2642 examples [00:03, 865.46 examples/s]2729 examples [00:03, 862.43 examples/s]2816 examples [00:03, 858.59 examples/s]2904 examples [00:03, 863.84 examples/s]2991 examples [00:03, 861.98 examples/s]3078 examples [00:03, 861.17 examples/s]3165 examples [00:03, 853.27 examples/s]3251 examples [00:03, 847.25 examples/s]3336 examples [00:03, 843.90 examples/s]3421 examples [00:04, 839.94 examples/s]3506 examples [00:04, 840.51 examples/s]3594 examples [00:04, 851.43 examples/s]3682 examples [00:04, 858.67 examples/s]3768 examples [00:04, 858.28 examples/s]3854 examples [00:04, 843.14 examples/s]3939 examples [00:04, 844.82 examples/s]4024 examples [00:04, 824.41 examples/s]4108 examples [00:04, 825.85 examples/s]4195 examples [00:04, 836.17 examples/s]4281 examples [00:05, 842.06 examples/s]4366 examples [00:05, 841.55 examples/s]4451 examples [00:05, 835.10 examples/s]4535 examples [00:05, 833.41 examples/s]4620 examples [00:05, 835.53 examples/s]4704 examples [00:05, 814.93 examples/s]4790 examples [00:05, 827.07 examples/s]4874 examples [00:05, 822.94 examples/s]4959 examples [00:05, 829.98 examples/s]5047 examples [00:05, 842.01 examples/s]5132 examples [00:06, 835.54 examples/s]5216 examples [00:06, 826.48 examples/s]5301 examples [00:06, 831.47 examples/s]5385 examples [00:06, 820.72 examples/s]5475 examples [00:06, 840.46 examples/s]5563 examples [00:06, 850.06 examples/s]5650 examples [00:06, 853.11 examples/s]5739 examples [00:06, 861.27 examples/s]5829 examples [00:06, 870.03 examples/s]5917 examples [00:07, 851.07 examples/s]6003 examples [00:07, 853.64 examples/s]6092 examples [00:07, 862.26 examples/s]6179 examples [00:07, 860.33 examples/s]6266 examples [00:07, 837.23 examples/s]6351 examples [00:07, 838.99 examples/s]6436 examples [00:07, 813.45 examples/s]6521 examples [00:07, 822.00 examples/s]6605 examples [00:07, 825.60 examples/s]6689 examples [00:07, 828.87 examples/s]6780 examples [00:08, 850.40 examples/s]6866 examples [00:08, 841.18 examples/s]6953 examples [00:08, 849.34 examples/s]7039 examples [00:08, 848.40 examples/s]7124 examples [00:08, 842.65 examples/s]7209 examples [00:08, 843.40 examples/s]7294 examples [00:08, 824.86 examples/s]7385 examples [00:08, 847.88 examples/s]7472 examples [00:08, 851.90 examples/s]7558 examples [00:08, 852.12 examples/s]7649 examples [00:09, 867.24 examples/s]7737 examples [00:09, 869.67 examples/s]7825 examples [00:09, 865.18 examples/s]7912 examples [00:09, 865.77 examples/s]8000 examples [00:09, 869.59 examples/s]8088 examples [00:09, 853.21 examples/s]8174 examples [00:09, 851.83 examples/s]8260 examples [00:09, 827.09 examples/s]8347 examples [00:09, 837.91 examples/s]8431 examples [00:09, 837.19 examples/s]8517 examples [00:10, 841.99 examples/s]8607 examples [00:10, 856.96 examples/s]8699 examples [00:10, 874.69 examples/s]8789 examples [00:10, 880.93 examples/s]8878 examples [00:10, 874.41 examples/s]8966 examples [00:10, 867.00 examples/s]9055 examples [00:10, 871.60 examples/s]9143 examples [00:10, 871.05 examples/s]9231 examples [00:10, 871.82 examples/s]9319 examples [00:10, 865.62 examples/s]9406 examples [00:11, 853.38 examples/s]9493 examples [00:11, 856.52 examples/s]9581 examples [00:11, 860.92 examples/s]9668 examples [00:11, 857.06 examples/s]9757 examples [00:11, 865.84 examples/s]9844 examples [00:11, 860.43 examples/s]9931 examples [00:11, 857.50 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteJ9ZEK8/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteJ9ZEK8/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0da48338c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0da48338c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0da48338c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0e106d0470>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d85e43ac8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0da48338c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0da48338c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0da48338c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d85dc0908>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d87ea52b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f0d840427b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f0d840427b8> 

  function with postional parmater data_info <function split_train_valid at 0x7f0d840427b8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f0d840428c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f0d840428c8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f0d840428c8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=75ed5c73ed55590aad59fd95794e59b5b217ac2c233cb484dbfdb142e952f38f
  Stored in directory: /tmp/pip-ephem-wheel-cache-0lihwjqa/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:08<255:46:31, 936B/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:08<179:16:05, 1.34kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:09<125:31:07, 1.91kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:09<87:48:23, 2.72kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.68M/862M [00:09<61:20:27, 3.89kB/s].vector_cache/glove.6B.zip:   1%|          | 6.65M/862M [00:09<42:44:31, 5.56kB/s].vector_cache/glove.6B.zip:   1%|          | 10.0M/862M [00:09<29:48:12, 7.94kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.8M/862M [00:09<20:44:47, 11.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.3M/862M [00:09<14:27:55, 16.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.3M/862M [00:09<10:04:02, 23.1kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.5M/862M [00:09<7:00:48, 33.1kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:09<4:52:51, 47.2kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.0M/862M [00:10<3:24:16, 67.4kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.8M/862M [00:10<2:22:14, 96.2kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.6M/862M [00:10<1:39:13, 137kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 49.2M/862M [00:10<1:09:09, 196kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:10<48:46, 277kB/s]  .vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:12<35:53, 374kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:12<29:00, 463kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:12<21:05, 637kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:13<14:56, 896kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:14<14:34, 917kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:14<11:47, 1.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:14<08:34, 1.55MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:16<08:44, 1.52MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:16<08:50, 1.50MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:16<06:44, 1.97MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.1M/862M [00:17<04:55, 2.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:18<08:08, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:18<06:49, 1.94MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.2M/862M [00:18<05:03, 2.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:18<03:42, 3.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:20<55:34, 237kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:20<40:12, 327kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:20<28:22, 463kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:22<22:56, 571kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:22<17:23, 753kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:22<12:25, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:24<11:47, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:24<10:54, 1.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:24<08:12, 1.58MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.0M/862M [00:24<05:54, 2.20MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:26<10:47, 1.20MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:26<08:52, 1.46MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:26<06:31, 1.98MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:28<07:34, 1.70MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:28<06:36, 1.95MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:28<04:56, 2.60MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:30<06:29, 1.97MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:30<05:51, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:30<04:22, 2.93MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:32<06:05, 2.09MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:32<06:51, 1.86MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:32<05:20, 2.38MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:32<04:00, 3.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:34<06:20, 2.00MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:34<05:46, 2.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:34<04:21, 2.90MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:36<05:59, 2.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:36<06:46, 1.86MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:36<05:16, 2.39MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:36<03:52, 3.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:38<07:37, 1.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:38<06:37, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:38<04:54, 2.55MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:40<06:21, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:40<06:59, 1.78MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:40<05:26, 2.29MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:40<03:57, 3.14MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:42<09:12, 1.35MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:42<07:42, 1.61MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:42<05:39, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:44<06:50, 1.80MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:44<07:18, 1.69MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:44<05:38, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:44<04:06, 2.99MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:46<08:52, 1.38MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:46<07:30, 1.63MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:46<05:33, 2.20MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:48<06:44, 1.81MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:48<07:12, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:48<05:40, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:48<04:05, 2.96MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:50<11:40:27, 17.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:50<8:11:07, 24.7kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:50<5:43:23, 35.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:50<3:59:50, 50.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:51<3:19:31, 60.5kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:52<2:20:49, 85.6kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:52<1:38:37, 122kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:53<1:11:39, 167kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:54<51:21, 233kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:54<36:10, 331kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:55<28:04, 425kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:56<20:51, 571kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:56<14:52, 799kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:57<13:11, 898kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:57<10:26, 1.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:58<07:35, 1.56MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:59<08:05, 1.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:59<06:40, 1.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [01:00<04:54, 2.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:00<03:35, 3.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:01<46:48, 250kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:01<35:08, 333kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:02<25:10, 465kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:03<19:26, 600kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:03<14:47, 787kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:03<10:35, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:05<10:06, 1.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:05<09:31, 1.22MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:05<07:13, 1.60MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:06<05:16, 2.19MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:07<07:00, 1.64MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:07<06:25, 1.79MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:07<04:52, 2.36MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:09<05:37, 2.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:09<05:18, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:09<04:03, 2.82MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:11<05:11, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:11<04:51, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:11<03:42, 3.06MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:13<05:06, 2.21MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:13<05:33, 2.03MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:13<04:28, 2.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:13<03:17, 3.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:15<06:55, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:15<06:01, 1.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:15<04:26, 2.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:17<05:43, 1.95MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:17<06:17, 1.78MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:17<04:53, 2.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:17<03:32, 3.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:19<10:22, 1.07MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:19<08:14, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:19<06:02, 1.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:21<06:49, 1.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:21<07:03, 1.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:21<05:27, 2.02MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:21<03:56, 2.78MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:23<1:21:31, 134kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:23<58:11, 188kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:23<40:55, 267kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:25<31:05, 350kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:25<24:00, 454kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:25<17:15, 630kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:25<12:11, 889kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:27<13:56, 777kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:27<10:40, 1.01MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:27<07:57, 1.36MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:27<05:39, 1.90MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:29<44:36, 241kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:29<33:26, 322kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:29<23:51, 450kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:29<16:47, 637kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:31<17:07, 624kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:31<13:05, 816kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:31<09:25, 1.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:33<09:03, 1.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:33<08:31, 1.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:33<06:30, 1.63MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:33<04:38, 2.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:35<22:26, 470kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:35<16:48, 627kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:35<12:00, 876kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:37<10:50, 967kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:37<09:45, 1.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:37<07:21, 1.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:39<06:48, 1.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:39<05:50, 1.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:39<04:20, 2.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:41<05:27, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:41<04:52, 2.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:41<03:40, 2.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:43<04:58, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:43<04:31, 2.27MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:43<03:25, 2.99MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:44<04:48, 2.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:45<04:14, 2.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:45<03:10, 3.21MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:45<02:21, 4.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:46<42:30, 239kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:47<31:49, 319kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:47<22:42, 446kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:47<15:55, 633kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:48<22:33, 447kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:49<16:50, 598kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:49<12:00, 836kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:50<10:43, 933kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:51<09:33, 1.05MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:51<07:08, 1.40MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:51<05:06, 1.95MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:52<1:15:07, 132kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:52<53:34, 185kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:53<37:39, 263kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:54<28:34, 345kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:54<20:50, 473kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:54<14:45, 667kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:55<10:25, 940kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:56<46:46, 209kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:56<33:43, 290kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:56<23:44, 411kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:58<18:53, 515kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:58<14:12, 684kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:58<10:10, 954kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [02:00<09:22, 1.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [02:00<08:32, 1.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [02:00<06:28, 1.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:02<06:03, 1.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:02<05:13, 1.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [02:02<03:53, 2.46MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:04<04:56, 1.93MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:04<05:24, 1.76MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:04<04:11, 2.27MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:04<03:02, 3.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:06<07:30, 1.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:06<06:14, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:06<04:35, 2.05MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:08<05:24, 1.74MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:08<05:42, 1.64MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:08<04:24, 2.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:08<03:11, 2.93MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:10<07:54, 1.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:10<06:29, 1.44MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:10<04:45, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:12<05:29, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:12<04:47, 1.93MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:12<03:34, 2.57MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:14<04:40, 1.97MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:14<04:12, 2.18MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:14<03:10, 2.89MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:16<04:22, 2.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:16<03:59, 2.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:16<03:00, 3.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:18<04:14, 2.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:18<04:48, 1.88MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:18<03:45, 2.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:18<02:44, 3.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:20<07:33, 1.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:20<06:13, 1.44MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:20<04:34, 1.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:22<05:16, 1.69MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:22<04:36, 1.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:22<03:26, 2.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:24<04:29, 1.97MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:24<04:57, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:24<03:55, 2.25MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:26<04:09, 2.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:26<03:47, 2.31MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:26<02:52, 3.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:28<04:02, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:28<04:36, 1.89MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:28<03:35, 2.42MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:28<02:41, 3.22MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:30<04:21, 1.98MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:30<03:56, 2.19MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:30<02:58, 2.89MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:32<04:05, 2.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:32<04:36, 1.86MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:32<03:39, 2.34MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:34<03:55, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:34<03:36, 2.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:34<02:44, 3.09MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:35<03:53, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:36<04:26, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:36<03:27, 2.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:36<02:32, 3.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:37<05:11, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:38<04:30, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:38<03:19, 2.50MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:39<04:15, 1.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:40<03:49, 2.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:40<02:50, 2.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:41<03:57, 2.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:41<03:28, 2.37MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:42<02:35, 3.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:42<01:55, 4.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:43<34:12, 238kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:43<24:45, 329kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:44<17:26, 465kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:45<14:03, 575kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:45<11:29, 703kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:46<08:22, 963kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:46<05:57, 1.35MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:47<07:57, 1.01MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:47<06:23, 1.25MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:48<04:40, 1.71MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:49<05:07, 1.55MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<05:12, 1.53MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<03:59, 1.99MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:50<02:53, 2.73MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:51<05:48, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:51<04:43, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:51<03:28, 2.26MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:51<02:31, 3.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:53<33:15, 235kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:53<24:52, 314kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:53<17:43, 440kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:53<12:26, 623kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:55<13:16, 583kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:55<10:05, 767kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:55<07:12, 1.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:57<06:49, 1.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:57<05:33, 1.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:57<04:04, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:59<04:38, 1.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:59<04:01, 1.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:59<02:57, 2.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:01<03:51, 1.95MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:01<04:14, 1.78MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:01<03:17, 2.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:01<02:23, 3.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:03<06:07, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:03<05:03, 1.47MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:03<03:43, 2.00MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:05<04:19, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:05<04:36, 1.60MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:05<03:36, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:05<02:36, 2.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:07<54:07, 135kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:07<38:37, 190kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:07<27:07, 269kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:09<20:34, 353kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:09<15:01, 483kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:09<10:40, 678kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:11<09:08, 787kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:11<07:06, 1.01MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:11<05:06, 1.40MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:13<05:15, 1.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:13<05:08, 1.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:13<03:57, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:13<02:49, 2.50MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:15<47:40, 148kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:15<34:03, 207kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:15<23:56, 293kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:17<18:18, 382kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:17<14:14, 490kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:17<10:14, 680kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:17<07:18, 951kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:19<06:56, 997kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:19<05:33, 1.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:19<04:01, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:21<04:24, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:21<03:47, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:21<02:47, 2.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:23<03:33, 1.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:23<03:53, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:23<03:03, 2.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:23<02:12, 3.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:25<50:04, 134kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:25<35:41, 188kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:25<25:03, 267kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:26<19:00, 349kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:27<14:39, 453kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:27<10:32, 629kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:27<07:25, 888kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:28<08:18, 791kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:29<06:29, 1.01MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:29<04:40, 1.40MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:30<04:46, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:31<04:40, 1.39MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:31<03:36, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:32<03:32, 1.82MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:33<03:09, 2.04MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:33<02:21, 2.71MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:34<03:08, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:34<02:50, 2.24MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:35<02:08, 2.95MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:36<02:58, 2.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:36<03:22, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:37<02:40, 2.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:38<02:52, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:38<02:38, 2.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:39<02:00, 3.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:40<02:50, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:40<03:14, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:40<02:31, 2.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:41<01:52, 3.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:42<03:16, 1.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:42<02:55, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:42<02:10, 2.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:44<02:55, 2.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:44<03:16, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:44<02:35, 2.31MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:46<02:46, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:46<02:33, 2.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:46<01:54, 3.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:48<02:43, 2.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:48<03:03, 1.92MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:48<02:23, 2.45MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:48<01:45, 3.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:50<03:44, 1.56MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:50<03:13, 1.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:50<02:22, 2.44MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:52<03:00, 1.92MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:52<03:17, 1.74MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:52<02:33, 2.25MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:52<01:51, 3.08MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:54<04:17, 1.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:54<03:29, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:54<02:34, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:56<03:07, 1.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:56<03:20, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:56<02:34, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:56<01:52, 2.97MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:58<03:29, 1.59MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:58<03:01, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:58<02:14, 2.47MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:00<02:50, 1.93MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:00<03:06, 1.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:00<02:24, 2.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:00<01:46, 3.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:02<03:00, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:02<02:39, 2.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:02<01:58, 2.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:04<02:37, 2.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:04<02:55, 1.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:04<02:16, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:04<01:43, 3.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:06<02:31, 2.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:06<02:18, 2.29MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:06<01:43, 3.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:08<02:25, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:08<02:45, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:08<02:11, 2.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:08<01:34, 3.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:10<13:34, 378kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:10<10:01, 512kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:10<07:05, 719kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:12<06:07, 828kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:12<05:19, 952kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:12<03:58, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:12<02:48, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:14<14:06, 354kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:14<10:22, 481kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:14<07:21, 675kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:16<06:16, 787kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:16<05:24, 911kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:16<04:00, 1.23MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:16<02:51, 1.71MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:18<03:48, 1.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:18<03:11, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:18<02:20, 2.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:19<02:44, 1.75MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:20<02:24, 1.98MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:20<01:48, 2.64MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:21<02:22, 1.99MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:22<02:08, 2.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:22<01:36, 2.90MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:23<02:13, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:24<02:02, 2.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:24<01:32, 3.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:25<02:09, 2.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:25<01:58, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:26<01:28, 3.08MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:27<02:06, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:27<01:55, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:28<01:26, 3.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:29<02:03, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:29<01:53, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:30<01:24, 3.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:31<02:01, 2.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:31<01:51, 2.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:32<01:24, 3.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:33<01:59, 2.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:33<02:16, 1.89MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:33<01:46, 2.42MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:34<01:18, 3.27MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:35<02:29, 1.71MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:35<02:10, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:35<01:37, 2.60MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:37<02:05, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:37<01:54, 2.19MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:37<01:25, 2.91MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:39<01:56, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:39<01:49, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:39<01:22, 2.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:41<01:49, 2.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:41<01:41, 2.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:41<01:16, 3.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:43<01:48, 2.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:43<01:40, 2.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:43<01:14, 3.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:45<01:47, 2.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:45<02:04, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:45<01:37, 2.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:45<01:10, 3.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:47<03:10, 1.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:47<02:36, 1.47MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:47<01:54, 1.99MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:49<02:12, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:49<02:20, 1.60MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:49<01:48, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:49<01:18, 2.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:51<02:32, 1.45MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:51<02:09, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:51<01:35, 2.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:53<01:57, 1.86MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:53<01:43, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:53<01:16, 2.80MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:55<01:44, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:55<01:34, 2.25MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:55<01:11, 2.97MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:57<01:38, 2.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:57<01:52, 1.86MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:57<01:27, 2.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:57<01:03, 3.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:59<02:51, 1.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:59<02:21, 1.45MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:59<01:42, 1.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:01<01:58, 1.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:01<02:04, 1.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:01<01:35, 2.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:01<01:09, 2.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:03<02:05, 1.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:03<01:47, 1.82MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:03<01:19, 2.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:05<01:40, 1.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:05<01:49, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:05<01:23, 2.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:05<01:03, 2.98MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:07<01:28, 2.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:07<01:21, 2.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:07<01:00, 3.07MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:07<00:44, 4.15MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:09<11:57, 257kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:09<08:40, 354kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:09<06:05, 499kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:10<04:55, 611kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:11<04:03, 740kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:11<02:57, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:11<02:06, 1.40MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:12<02:20, 1.26MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:13<01:56, 1.52MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:13<01:25, 2.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:14<01:39, 1.74MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:15<01:27, 1.97MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:15<01:04, 2.65MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:16<01:23, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:16<01:33, 1.81MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:17<01:12, 2.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:17<00:52, 3.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:18<01:49, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:18<01:33, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:19<01:09, 2.35MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:20<01:25, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:20<01:32, 1.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:21<01:11, 2.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:21<00:51, 3.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:22<02:40, 974kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:22<02:05, 1.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:22<01:31, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:23<01:04, 2.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:24<10:56, 231kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:24<07:53, 320kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:24<05:31, 452kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:26<04:23, 560kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:26<03:19, 739kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:26<02:21, 1.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:28<02:11, 1.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:28<01:46, 1.34MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:28<01:17, 1.83MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:30<01:26, 1.61MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:30<01:14, 1.87MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:30<00:55, 2.50MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:32<01:10, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:32<01:16, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:32<01:00, 2.23MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:32<00:42, 3.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:34<2:06:54, 17.2kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:34<1:28:47, 24.6kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:34<1:01:28, 35.1kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:36<42:48, 49.5kB/s]  .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:36<30:21, 69.7kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:36<21:13, 99.0kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:36<14:31, 141kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:38<18:03, 114kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:38<12:48, 160kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:38<08:54, 227kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:40<06:35, 301kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:40<05:01, 394kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:40<03:34, 549kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:40<02:29, 775kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:42<02:33, 749kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:42<01:58, 962kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:42<01:25, 1.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:44<01:24, 1.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:44<01:21, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:44<01:01, 1.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:44<00:43, 2.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<01:40, 1.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<01:21, 1.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:46<00:58, 1.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:48<01:04, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:48<00:55, 1.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:48<00:40, 2.50MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:50<00:51, 1.92MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:50<00:45, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:50<00:33, 2.87MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:52<00:45, 2.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:52<00:41, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:52<00:30, 3.00MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:54<00:42, 2.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:54<00:48, 1.87MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:54<00:37, 2.35MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:56<00:39, 2.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:56<00:36, 2.36MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:56<00:27, 3.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:58<00:37, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:58<00:43, 1.90MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:58<00:33, 2.42MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:58<00:24, 3.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:00<00:42, 1.81MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:00<00:38, 2.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:00<00:27, 2.73MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:01<00:36, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:02<00:40, 1.82MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:02<00:31, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:02<00:22, 3.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:03<00:40, 1.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:04<00:35, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:04<00:25, 2.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:05<00:32, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:06<00:36, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:06<00:28, 2.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:06<00:19, 3.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:07<08:33, 119kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:08<06:03, 168kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:08<04:09, 238kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:09<03:01, 315kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:09<02:18, 412kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:10<01:38, 571kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:10<01:06, 808kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:11<01:45, 503kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:11<01:18, 668kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:12<00:54, 932kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:13<00:48, 1.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:13<00:43, 1.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:14<00:32, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:14<00:21, 2.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:15<05:39, 132kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:15<04:00, 185kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:15<02:43, 263kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:17<01:58, 345kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:17<01:25, 469kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:17<00:58, 659kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:19<00:47, 770kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:19<00:40, 897kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:19<00:29, 1.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:19<00:19, 1.69MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:21<00:26, 1.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:21<00:21, 1.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:21<00:14, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:23<00:16, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:23<00:17, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:23<00:13, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:25<00:12, 2.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:25<00:10, 2.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:25<00:07, 3.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:25<00:04, 4.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:27<00:41, 485kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:27<00:30, 654kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:27<00:20, 916kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:27<00:12, 1.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:29<01:09, 231kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:29<00:48, 320kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:29<00:31, 452kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:31<00:21, 560kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:31<00:17, 687kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:31<00:11, 935kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:33<00:07, 1.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:33<00:05, 1.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:33<00:03, 1.83MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:35<00:02, 1.63MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:35<00:02, 1.57MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:35<00:01, 2.02MB/s].vector_cache/glove.6B.zip: 862MB [06:35, 2.18MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<14:30:04,  7.66it/s]  0%|          | 728/400000 [00:00<10:08:13, 10.94it/s]  0%|          | 1462/400000 [00:00<7:05:14, 15.62it/s]  1%|          | 2215/400000 [00:00<4:57:22, 22.29it/s]  1%|          | 2910/400000 [00:00<3:28:05, 31.81it/s]  1%|          | 3645/400000 [00:00<2:25:39, 45.35it/s]  1%|          | 4402/400000 [00:00<1:42:01, 64.62it/s]  1%|▏         | 5134/400000 [00:00<1:11:33, 91.97it/s]  1%|▏         | 5886/400000 [00:00<50:15, 130.70it/s]   2%|▏         | 6624/400000 [00:01<35:22, 185.31it/s]  2%|▏         | 7393/400000 [00:01<24:58, 262.01it/s]  2%|▏         | 8122/400000 [00:01<17:43, 368.62it/s]  2%|▏         | 8850/400000 [00:01<12:39, 514.84it/s]  2%|▏         | 9611/400000 [00:01<09:06, 714.75it/s]  3%|▎         | 10367/400000 [00:01<06:37, 981.20it/s]  3%|▎         | 11105/400000 [00:01<04:53, 1324.58it/s]  3%|▎         | 11838/400000 [00:01<03:42, 1745.59it/s]  3%|▎         | 12554/400000 [00:01<02:51, 2252.86it/s]  3%|▎         | 13265/400000 [00:01<02:16, 2827.95it/s]  3%|▎         | 13973/400000 [00:02<01:51, 3448.77it/s]  4%|▎         | 14681/400000 [00:02<01:35, 4054.36it/s]  4%|▍         | 15383/400000 [00:02<01:23, 4606.78it/s]  4%|▍         | 16089/400000 [00:02<01:14, 5142.55it/s]  4%|▍         | 16849/400000 [00:02<01:07, 5694.92it/s]  4%|▍         | 17586/400000 [00:02<01:02, 6110.08it/s]  5%|▍         | 18339/400000 [00:02<00:58, 6475.91it/s]  5%|▍         | 19071/400000 [00:02<00:57, 6662.24it/s]  5%|▍         | 19798/400000 [00:02<00:55, 6800.49it/s]  5%|▌         | 20521/400000 [00:02<00:55, 6853.73it/s]  5%|▌         | 21237/400000 [00:03<00:55, 6837.67it/s]  5%|▌         | 21980/400000 [00:03<00:53, 7003.14it/s]  6%|▌         | 22703/400000 [00:03<00:53, 7068.22it/s]  6%|▌         | 23445/400000 [00:03<00:52, 7169.64it/s]  6%|▌         | 24171/400000 [00:03<00:52, 7181.62it/s]  6%|▌         | 24900/400000 [00:03<00:52, 7211.17it/s]  6%|▋         | 25650/400000 [00:03<00:51, 7291.63it/s]  7%|▋         | 26383/400000 [00:03<00:52, 7143.31it/s]  7%|▋         | 27134/400000 [00:03<00:51, 7249.00it/s]  7%|▋         | 27887/400000 [00:03<00:50, 7328.33it/s]  7%|▋         | 28622/400000 [00:04<00:51, 7220.47it/s]  7%|▋         | 29355/400000 [00:04<00:51, 7252.51it/s]  8%|▊         | 30105/400000 [00:04<00:50, 7320.04it/s]  8%|▊         | 30838/400000 [00:04<00:50, 7302.90it/s]  8%|▊         | 31612/400000 [00:04<00:49, 7428.08it/s]  8%|▊         | 32356/400000 [00:04<00:49, 7411.80it/s]  8%|▊         | 33098/400000 [00:04<00:49, 7354.11it/s]  8%|▊         | 33835/400000 [00:04<00:49, 7329.96it/s]  9%|▊         | 34569/400000 [00:04<00:50, 7300.56it/s]  9%|▉         | 35312/400000 [00:04<00:49, 7338.12it/s]  9%|▉         | 36047/400000 [00:05<00:49, 7280.27it/s]  9%|▉         | 36785/400000 [00:05<00:49, 7308.52it/s]  9%|▉         | 37517/400000 [00:05<00:49, 7289.60it/s] 10%|▉         | 38247/400000 [00:05<00:50, 7112.57it/s] 10%|▉         | 39007/400000 [00:05<00:49, 7250.23it/s] 10%|▉         | 39734/400000 [00:05<00:49, 7236.32it/s] 10%|█         | 40459/400000 [00:05<00:50, 7186.66it/s] 10%|█         | 41207/400000 [00:05<00:49, 7270.81it/s] 10%|█         | 41935/400000 [00:05<00:50, 7063.12it/s] 11%|█         | 42671/400000 [00:06<00:49, 7148.15it/s] 11%|█         | 43391/400000 [00:06<00:49, 7162.76it/s] 11%|█         | 44109/400000 [00:06<00:50, 7097.21it/s] 11%|█         | 44820/400000 [00:06<00:50, 7073.63it/s] 11%|█▏        | 45547/400000 [00:06<00:49, 7129.70it/s] 12%|█▏        | 46261/400000 [00:06<00:50, 7068.18it/s] 12%|█▏        | 46969/400000 [00:06<00:50, 7052.86it/s] 12%|█▏        | 47675/400000 [00:06<00:50, 7038.58it/s] 12%|█▏        | 48388/400000 [00:06<00:49, 7064.39it/s] 12%|█▏        | 49095/400000 [00:06<00:49, 7020.73it/s] 12%|█▏        | 49841/400000 [00:07<00:49, 7144.67it/s] 13%|█▎        | 50560/400000 [00:07<00:48, 7155.62it/s] 13%|█▎        | 51302/400000 [00:07<00:48, 7231.72it/s] 13%|█▎        | 52041/400000 [00:07<00:47, 7276.35it/s] 13%|█▎        | 52770/400000 [00:07<00:47, 7278.15it/s] 13%|█▎        | 53528/400000 [00:07<00:47, 7364.96it/s] 14%|█▎        | 54293/400000 [00:07<00:46, 7447.28it/s] 14%|█▍        | 55039/400000 [00:07<00:46, 7423.62it/s] 14%|█▍        | 55782/400000 [00:07<00:46, 7342.37it/s] 14%|█▍        | 56538/400000 [00:07<00:46, 7406.01it/s] 14%|█▍        | 57292/400000 [00:08<00:46, 7444.84it/s] 15%|█▍        | 58037/400000 [00:08<00:46, 7411.24it/s] 15%|█▍        | 58801/400000 [00:08<00:45, 7475.91it/s] 15%|█▍        | 59549/400000 [00:08<00:46, 7390.73it/s] 15%|█▌        | 60305/400000 [00:08<00:45, 7440.28it/s] 15%|█▌        | 61057/400000 [00:08<00:45, 7463.07it/s] 15%|█▌        | 61804/400000 [00:08<00:45, 7438.78it/s] 16%|█▌        | 62549/400000 [00:08<00:46, 7309.85it/s] 16%|█▌        | 63308/400000 [00:08<00:45, 7389.95it/s] 16%|█▌        | 64066/400000 [00:08<00:45, 7445.70it/s] 16%|█▌        | 64812/400000 [00:09<00:45, 7417.71it/s] 16%|█▋        | 65555/400000 [00:09<00:45, 7373.19it/s] 17%|█▋        | 66293/400000 [00:09<00:46, 7227.63it/s] 17%|█▋        | 67035/400000 [00:09<00:45, 7284.23it/s] 17%|█▋        | 67765/400000 [00:09<00:46, 7133.54it/s] 17%|█▋        | 68480/400000 [00:09<00:47, 7036.35it/s] 17%|█▋        | 69185/400000 [00:09<00:47, 7021.65it/s] 17%|█▋        | 69916/400000 [00:09<00:46, 7104.57it/s] 18%|█▊        | 70673/400000 [00:09<00:45, 7235.88it/s] 18%|█▊        | 71398/400000 [00:09<00:45, 7184.25it/s] 18%|█▊        | 72118/400000 [00:10<00:45, 7183.54it/s] 18%|█▊        | 72838/400000 [00:10<00:45, 7184.06it/s] 18%|█▊        | 73557/400000 [00:10<00:45, 7179.81it/s] 19%|█▊        | 74330/400000 [00:10<00:44, 7335.58it/s] 19%|█▉        | 75076/400000 [00:10<00:44, 7370.89it/s] 19%|█▉        | 75844/400000 [00:10<00:43, 7458.48it/s] 19%|█▉        | 76591/400000 [00:10<00:44, 7336.36it/s] 19%|█▉        | 77326/400000 [00:10<00:44, 7301.49it/s] 20%|█▉        | 78057/400000 [00:10<00:44, 7244.84it/s] 20%|█▉        | 78791/400000 [00:10<00:44, 7271.12it/s] 20%|█▉        | 79544/400000 [00:11<00:43, 7344.40it/s] 20%|██        | 80279/400000 [00:11<00:43, 7330.57it/s] 20%|██        | 81013/400000 [00:11<00:43, 7294.18it/s] 20%|██        | 81766/400000 [00:11<00:43, 7363.16it/s] 21%|██        | 82517/400000 [00:11<00:42, 7404.76it/s] 21%|██        | 83260/400000 [00:11<00:42, 7408.47it/s] 21%|██        | 84004/400000 [00:11<00:42, 7416.14it/s] 21%|██        | 84750/400000 [00:11<00:42, 7425.92it/s] 21%|██▏       | 85493/400000 [00:11<00:42, 7382.76it/s] 22%|██▏       | 86253/400000 [00:11<00:42, 7446.55it/s] 22%|██▏       | 86998/400000 [00:12<00:42, 7291.62it/s] 22%|██▏       | 87729/400000 [00:12<00:42, 7277.07it/s] 22%|██▏       | 88458/400000 [00:12<00:43, 7168.45it/s] 22%|██▏       | 89176/400000 [00:12<00:43, 7122.44it/s] 22%|██▏       | 89889/400000 [00:12<00:43, 7118.60it/s] 23%|██▎       | 90609/400000 [00:12<00:43, 7140.28it/s] 23%|██▎       | 91338/400000 [00:12<00:42, 7181.74it/s] 23%|██▎       | 92072/400000 [00:12<00:42, 7226.06it/s] 23%|██▎       | 92795/400000 [00:12<00:42, 7199.19it/s] 23%|██▎       | 93530/400000 [00:13<00:42, 7241.71it/s] 24%|██▎       | 94255/400000 [00:13<00:42, 7141.68it/s] 24%|██▎       | 94970/400000 [00:13<00:42, 7143.74it/s] 24%|██▍       | 95685/400000 [00:13<00:42, 7141.48it/s] 24%|██▍       | 96400/400000 [00:13<00:42, 7113.95it/s] 24%|██▍       | 97132/400000 [00:13<00:42, 7173.56it/s] 24%|██▍       | 97880/400000 [00:13<00:41, 7260.93it/s] 25%|██▍       | 98607/400000 [00:13<00:41, 7225.81it/s] 25%|██▍       | 99339/400000 [00:13<00:41, 7251.62it/s] 25%|██▌       | 100065/400000 [00:13<00:41, 7240.95it/s] 25%|██▌       | 100839/400000 [00:14<00:40, 7381.09it/s] 25%|██▌       | 101597/400000 [00:14<00:40, 7439.57it/s] 26%|██▌       | 102342/400000 [00:14<00:40, 7400.95it/s] 26%|██▌       | 103083/400000 [00:14<00:40, 7359.09it/s] 26%|██▌       | 103840/400000 [00:14<00:39, 7419.35it/s] 26%|██▌       | 104583/400000 [00:14<00:40, 7270.06it/s] 26%|██▋       | 105328/400000 [00:14<00:40, 7321.00it/s] 27%|██▋       | 106120/400000 [00:14<00:39, 7490.70it/s] 27%|██▋       | 106871/400000 [00:14<00:40, 7323.51it/s] 27%|██▋       | 107652/400000 [00:14<00:39, 7461.23it/s] 27%|██▋       | 108421/400000 [00:15<00:38, 7527.80it/s] 27%|██▋       | 109202/400000 [00:15<00:38, 7608.88it/s] 28%|██▊       | 110013/400000 [00:15<00:37, 7751.79it/s] 28%|██▊       | 110790/400000 [00:15<00:37, 7688.58it/s] 28%|██▊       | 111561/400000 [00:15<00:38, 7493.76it/s] 28%|██▊       | 112317/400000 [00:15<00:38, 7511.33it/s] 28%|██▊       | 113070/400000 [00:15<00:38, 7419.47it/s] 28%|██▊       | 113814/400000 [00:15<00:39, 7305.60it/s] 29%|██▊       | 114561/400000 [00:15<00:38, 7352.21it/s] 29%|██▉       | 115298/400000 [00:15<00:38, 7332.00it/s] 29%|██▉       | 116032/400000 [00:16<00:39, 7223.70it/s] 29%|██▉       | 116766/400000 [00:16<00:39, 7256.42it/s] 29%|██▉       | 117493/400000 [00:16<00:39, 7241.63it/s] 30%|██▉       | 118226/400000 [00:16<00:38, 7265.33it/s] 30%|██▉       | 118986/400000 [00:16<00:38, 7362.28it/s] 30%|██▉       | 119723/400000 [00:16<00:38, 7274.57it/s] 30%|███       | 120475/400000 [00:16<00:38, 7344.21it/s] 30%|███       | 121211/400000 [00:16<00:38, 7334.48it/s] 30%|███       | 121945/400000 [00:16<00:37, 7332.65it/s] 31%|███       | 122679/400000 [00:16<00:38, 7272.59it/s] 31%|███       | 123431/400000 [00:17<00:37, 7343.45it/s] 31%|███       | 124166/400000 [00:17<00:37, 7291.47it/s] 31%|███       | 124896/400000 [00:17<00:38, 7096.19it/s] 31%|███▏      | 125633/400000 [00:17<00:38, 7174.04it/s] 32%|███▏      | 126375/400000 [00:17<00:37, 7243.42it/s] 32%|███▏      | 127108/400000 [00:17<00:37, 7268.24it/s] 32%|███▏      | 127845/400000 [00:17<00:37, 7297.80it/s] 32%|███▏      | 128576/400000 [00:17<00:37, 7193.39it/s] 32%|███▏      | 129297/400000 [00:17<00:37, 7171.30it/s] 33%|███▎      | 130037/400000 [00:17<00:37, 7236.49it/s] 33%|███▎      | 130770/400000 [00:18<00:37, 7262.85it/s] 33%|███▎      | 131497/400000 [00:18<00:37, 7214.24it/s] 33%|███▎      | 132219/400000 [00:18<00:38, 7019.48it/s] 33%|███▎      | 132929/400000 [00:18<00:37, 7041.48it/s] 33%|███▎      | 133635/400000 [00:18<00:37, 7043.70it/s] 34%|███▎      | 134348/400000 [00:18<00:37, 7066.90it/s] 34%|███▍      | 135064/400000 [00:18<00:37, 7092.06it/s] 34%|███▍      | 135774/400000 [00:18<00:37, 6953.46it/s] 34%|███▍      | 136482/400000 [00:18<00:37, 6989.90it/s] 34%|███▍      | 137188/400000 [00:19<00:37, 7009.98it/s] 34%|███▍      | 137935/400000 [00:19<00:36, 7141.16it/s] 35%|███▍      | 138663/400000 [00:19<00:36, 7180.42it/s] 35%|███▍      | 139428/400000 [00:19<00:35, 7312.94it/s] 35%|███▌      | 140161/400000 [00:19<00:35, 7279.90it/s] 35%|███▌      | 140900/400000 [00:19<00:35, 7309.93it/s] 35%|███▌      | 141632/400000 [00:19<00:35, 7251.39it/s] 36%|███▌      | 142387/400000 [00:19<00:35, 7337.76it/s] 36%|███▌      | 143148/400000 [00:19<00:34, 7416.16it/s] 36%|███▌      | 143891/400000 [00:19<00:34, 7379.63it/s] 36%|███▌      | 144630/400000 [00:20<00:34, 7366.73it/s] 36%|███▋      | 145371/400000 [00:20<00:34, 7377.08it/s] 37%|███▋      | 146109/400000 [00:20<00:34, 7283.31it/s] 37%|███▋      | 146853/400000 [00:20<00:34, 7327.59it/s] 37%|███▋      | 147604/400000 [00:20<00:34, 7380.05it/s] 37%|███▋      | 148350/400000 [00:20<00:33, 7403.72it/s] 37%|███▋      | 149091/400000 [00:20<00:34, 7375.35it/s] 37%|███▋      | 149829/400000 [00:20<00:34, 7265.42it/s] 38%|███▊      | 150562/400000 [00:20<00:34, 7284.59it/s] 38%|███▊      | 151325/400000 [00:20<00:33, 7384.58it/s] 38%|███▊      | 152070/400000 [00:21<00:33, 7403.99it/s] 38%|███▊      | 152811/400000 [00:21<00:33, 7400.23it/s] 38%|███▊      | 153557/400000 [00:21<00:33, 7416.15it/s] 39%|███▊      | 154299/400000 [00:21<00:33, 7336.61it/s] 39%|███▉      | 155047/400000 [00:21<00:33, 7377.29it/s] 39%|███▉      | 155786/400000 [00:21<00:33, 7355.53it/s] 39%|███▉      | 156522/400000 [00:21<00:33, 7351.84it/s] 39%|███▉      | 157273/400000 [00:21<00:32, 7397.56it/s] 40%|███▉      | 158013/400000 [00:21<00:32, 7390.70it/s] 40%|███▉      | 158753/400000 [00:21<00:33, 7290.15it/s] 40%|███▉      | 159493/400000 [00:22<00:32, 7320.05it/s] 40%|████      | 160226/400000 [00:22<00:33, 7263.35it/s] 40%|████      | 160984/400000 [00:22<00:32, 7355.24it/s] 40%|████      | 161721/400000 [00:22<00:32, 7347.25it/s] 41%|████      | 162457/400000 [00:22<00:32, 7317.26it/s] 41%|████      | 163190/400000 [00:22<00:32, 7254.69it/s] 41%|████      | 163916/400000 [00:22<00:32, 7197.63it/s] 41%|████      | 164686/400000 [00:22<00:32, 7341.26it/s] 41%|████▏     | 165455/400000 [00:22<00:31, 7440.33it/s] 42%|████▏     | 166201/400000 [00:22<00:31, 7395.70it/s] 42%|████▏     | 166942/400000 [00:23<00:31, 7378.93it/s] 42%|████▏     | 167703/400000 [00:23<00:31, 7446.50it/s] 42%|████▏     | 168455/400000 [00:23<00:31, 7466.33it/s] 42%|████▏     | 169208/400000 [00:23<00:30, 7483.32it/s] 42%|████▏     | 169957/400000 [00:23<00:31, 7295.59it/s] 43%|████▎     | 170709/400000 [00:23<00:31, 7358.89it/s] 43%|████▎     | 171446/400000 [00:23<00:31, 7314.12it/s] 43%|████▎     | 172193/400000 [00:23<00:30, 7359.13it/s] 43%|████▎     | 172948/400000 [00:23<00:30, 7413.90it/s] 43%|████▎     | 173710/400000 [00:23<00:30, 7474.43it/s] 44%|████▎     | 174466/400000 [00:24<00:30, 7498.19it/s] 44%|████▍     | 175217/400000 [00:24<00:30, 7478.73it/s] 44%|████▍     | 175966/400000 [00:24<00:30, 7391.75it/s] 44%|████▍     | 176723/400000 [00:24<00:29, 7443.09it/s] 44%|████▍     | 177488/400000 [00:24<00:29, 7502.79it/s] 45%|████▍     | 178244/400000 [00:24<00:29, 7519.01it/s] 45%|████▍     | 178997/400000 [00:24<00:29, 7452.77it/s] 45%|████▍     | 179743/400000 [00:24<00:29, 7394.65it/s] 45%|████▌     | 180483/400000 [00:24<00:29, 7335.66it/s] 45%|████▌     | 181217/400000 [00:24<00:29, 7331.41it/s] 45%|████▌     | 181964/400000 [00:25<00:29, 7371.14it/s] 46%|████▌     | 182718/400000 [00:25<00:29, 7418.96it/s] 46%|████▌     | 183472/400000 [00:25<00:29, 7452.70it/s] 46%|████▌     | 184218/400000 [00:25<00:29, 7376.54it/s] 46%|████▌     | 184956/400000 [00:25<00:29, 7355.63it/s] 46%|████▋     | 185716/400000 [00:25<00:28, 7424.82it/s] 47%|████▋     | 186470/400000 [00:25<00:28, 7458.72it/s] 47%|████▋     | 187223/400000 [00:25<00:28, 7477.83it/s] 47%|████▋     | 187995/400000 [00:25<00:28, 7548.75it/s] 47%|████▋     | 188751/400000 [00:25<00:28, 7446.20it/s] 47%|████▋     | 189502/400000 [00:26<00:28, 7464.92it/s] 48%|████▊     | 190249/400000 [00:26<00:28, 7458.41it/s] 48%|████▊     | 190999/400000 [00:26<00:27, 7469.98it/s] 48%|████▊     | 191747/400000 [00:26<00:28, 7331.51it/s] 48%|████▊     | 192481/400000 [00:26<00:28, 7330.15it/s] 48%|████▊     | 193216/400000 [00:26<00:28, 7334.27it/s] 48%|████▊     | 193950/400000 [00:26<00:28, 7333.84it/s] 49%|████▊     | 194684/400000 [00:26<00:28, 7311.22it/s] 49%|████▉     | 195454/400000 [00:26<00:27, 7422.52it/s] 49%|████▉     | 196197/400000 [00:27<00:27, 7365.32it/s] 49%|████▉     | 196935/400000 [00:27<00:27, 7327.59it/s] 49%|████▉     | 197704/400000 [00:27<00:27, 7430.40it/s] 50%|████▉     | 198477/400000 [00:27<00:26, 7516.63it/s] 50%|████▉     | 199244/400000 [00:27<00:26, 7560.80it/s] 50%|█████     | 200001/400000 [00:27<00:26, 7536.73it/s] 50%|█████     | 200787/400000 [00:27<00:26, 7628.79it/s] 50%|█████     | 201551/400000 [00:27<00:26, 7570.35it/s] 51%|█████     | 202309/400000 [00:27<00:26, 7520.19it/s] 51%|█████     | 203080/400000 [00:27<00:25, 7574.28it/s] 51%|█████     | 203838/400000 [00:28<00:26, 7532.02it/s] 51%|█████     | 204592/400000 [00:28<00:26, 7506.51it/s] 51%|█████▏    | 205343/400000 [00:28<00:26, 7481.17it/s] 52%|█████▏    | 206095/400000 [00:28<00:25, 7492.75it/s] 52%|█████▏    | 206869/400000 [00:28<00:25, 7563.63it/s] 52%|█████▏    | 207626/400000 [00:28<00:25, 7544.06it/s] 52%|█████▏    | 208381/400000 [00:28<00:25, 7419.31it/s] 52%|█████▏    | 209124/400000 [00:28<00:25, 7390.82it/s] 52%|█████▏    | 209864/400000 [00:28<00:25, 7357.44it/s] 53%|█████▎    | 210615/400000 [00:28<00:25, 7400.10it/s] 53%|█████▎    | 211356/400000 [00:29<00:25, 7291.42it/s] 53%|█████▎    | 212100/400000 [00:29<00:25, 7335.23it/s] 53%|█████▎    | 212850/400000 [00:29<00:25, 7382.50it/s] 53%|█████▎    | 213621/400000 [00:29<00:24, 7477.33it/s] 54%|█████▎    | 214372/400000 [00:29<00:24, 7485.64it/s] 54%|█████▍    | 215153/400000 [00:29<00:24, 7579.64it/s] 54%|█████▍    | 215916/400000 [00:29<00:24, 7594.22it/s] 54%|█████▍    | 216676/400000 [00:29<00:24, 7551.70it/s] 54%|█████▍    | 217432/400000 [00:29<00:24, 7538.84it/s] 55%|█████▍    | 218187/400000 [00:29<00:24, 7528.47it/s] 55%|█████▍    | 218941/400000 [00:30<00:24, 7318.53it/s] 55%|█████▍    | 219675/400000 [00:30<00:24, 7255.52it/s] 55%|█████▌    | 220428/400000 [00:30<00:24, 7333.97it/s] 55%|█████▌    | 221197/400000 [00:30<00:24, 7435.18it/s] 55%|█████▌    | 221942/400000 [00:30<00:23, 7423.93it/s] 56%|█████▌    | 222686/400000 [00:30<00:24, 7331.06it/s] 56%|█████▌    | 223433/400000 [00:30<00:23, 7371.70it/s] 56%|█████▌    | 224176/400000 [00:30<00:23, 7386.52it/s] 56%|█████▌    | 224916/400000 [00:30<00:23, 7385.57it/s] 56%|█████▋    | 225655/400000 [00:30<00:23, 7373.62it/s] 57%|█████▋    | 226393/400000 [00:31<00:23, 7283.52it/s] 57%|█████▋    | 227122/400000 [00:31<00:23, 7206.94it/s] 57%|█████▋    | 227855/400000 [00:31<00:23, 7241.43it/s] 57%|█████▋    | 228591/400000 [00:31<00:23, 7275.59it/s] 57%|█████▋    | 229319/400000 [00:31<00:23, 7266.19it/s] 58%|█████▊    | 230049/400000 [00:31<00:23, 7275.09it/s] 58%|█████▊    | 230782/400000 [00:31<00:23, 7289.80it/s] 58%|█████▊    | 231512/400000 [00:31<00:23, 7162.62it/s] 58%|█████▊    | 232232/400000 [00:31<00:23, 7173.24it/s] 58%|█████▊    | 232950/400000 [00:31<00:23, 7155.85it/s] 58%|█████▊    | 233666/400000 [00:32<00:23, 7148.52it/s] 59%|█████▊    | 234382/400000 [00:32<00:23, 7137.26it/s] 59%|█████▉    | 235096/400000 [00:32<00:23, 6982.09it/s] 59%|█████▉    | 235807/400000 [00:32<00:23, 7016.60it/s] 59%|█████▉    | 236514/400000 [00:32<00:23, 7032.24it/s] 59%|█████▉    | 237233/400000 [00:32<00:23, 7076.77it/s] 59%|█████▉    | 237965/400000 [00:32<00:22, 7147.85it/s] 60%|█████▉    | 238681/400000 [00:32<00:22, 7136.37it/s] 60%|█████▉    | 239395/400000 [00:32<00:22, 7017.54it/s] 60%|██████    | 240108/400000 [00:32<00:22, 7049.85it/s] 60%|██████    | 240814/400000 [00:33<00:23, 6913.42it/s] 60%|██████    | 241527/400000 [00:33<00:22, 6975.81it/s] 61%|██████    | 242234/400000 [00:33<00:22, 7003.35it/s] 61%|██████    | 242942/400000 [00:33<00:22, 7023.70it/s] 61%|██████    | 243661/400000 [00:33<00:22, 7070.43it/s] 61%|██████    | 244369/400000 [00:33<00:22, 6773.24it/s] 61%|██████▏   | 245053/400000 [00:33<00:22, 6792.50it/s] 61%|██████▏   | 245752/400000 [00:33<00:22, 6848.87it/s] 62%|██████▏   | 246448/400000 [00:33<00:22, 6881.16it/s] 62%|██████▏   | 247172/400000 [00:34<00:21, 6983.80it/s] 62%|██████▏   | 247872/400000 [00:34<00:21, 6931.62it/s] 62%|██████▏   | 248571/400000 [00:34<00:21, 6948.72it/s] 62%|██████▏   | 249297/400000 [00:34<00:21, 7038.32it/s] 63%|██████▎   | 250029/400000 [00:34<00:21, 7118.16it/s] 63%|██████▎   | 250750/400000 [00:34<00:20, 7143.38it/s] 63%|██████▎   | 251465/400000 [00:34<00:20, 7106.41it/s] 63%|██████▎   | 252177/400000 [00:34<00:20, 7073.31it/s] 63%|██████▎   | 252885/400000 [00:34<00:21, 6946.58it/s] 63%|██████▎   | 253601/400000 [00:34<00:20, 7006.94it/s] 64%|██████▎   | 254342/400000 [00:35<00:20, 7119.57it/s] 64%|██████▍   | 255055/400000 [00:35<00:20, 7016.10it/s] 64%|██████▍   | 255758/400000 [00:35<00:20, 6949.23it/s] 64%|██████▍   | 256502/400000 [00:35<00:20, 7088.82it/s] 64%|██████▍   | 257213/400000 [00:35<00:20, 6987.29it/s] 64%|██████▍   | 257951/400000 [00:35<00:20, 7099.37it/s] 65%|██████▍   | 258663/400000 [00:35<00:20, 6934.24it/s] 65%|██████▍   | 259359/400000 [00:35<00:20, 6916.44it/s] 65%|██████▌   | 260052/400000 [00:35<00:20, 6861.18it/s] 65%|██████▌   | 260771/400000 [00:35<00:20, 6956.60it/s] 65%|██████▌   | 261481/400000 [00:36<00:19, 6998.65it/s] 66%|██████▌   | 262182/400000 [00:36<00:20, 6836.40it/s] 66%|██████▌   | 262915/400000 [00:36<00:19, 6975.29it/s] 66%|██████▌   | 263662/400000 [00:36<00:19, 7116.00it/s] 66%|██████▌   | 264376/400000 [00:36<00:19, 7062.06it/s] 66%|██████▋   | 265133/400000 [00:36<00:18, 7205.19it/s] 66%|██████▋   | 265856/400000 [00:36<00:18, 7146.84it/s] 67%|██████▋   | 266572/400000 [00:36<00:19, 6963.10it/s] 67%|██████▋   | 267311/400000 [00:36<00:18, 7084.61it/s] 67%|██████▋   | 268059/400000 [00:36<00:18, 7198.05it/s] 67%|██████▋   | 268792/400000 [00:37<00:18, 7235.67it/s] 67%|██████▋   | 269543/400000 [00:37<00:17, 7314.86it/s] 68%|██████▊   | 270287/400000 [00:37<00:17, 7351.20it/s] 68%|██████▊   | 271033/400000 [00:37<00:17, 7383.04it/s] 68%|██████▊   | 271790/400000 [00:37<00:17, 7435.98it/s] 68%|██████▊   | 272557/400000 [00:37<00:16, 7504.35it/s] 68%|██████▊   | 273345/400000 [00:37<00:16, 7611.47it/s] 69%|██████▊   | 274107/400000 [00:37<00:16, 7453.53it/s] 69%|██████▊   | 274867/400000 [00:37<00:16, 7495.78it/s] 69%|██████▉   | 275626/400000 [00:37<00:16, 7520.49it/s] 69%|██████▉   | 276379/400000 [00:38<00:16, 7522.60it/s] 69%|██████▉   | 277132/400000 [00:38<00:16, 7389.72it/s] 69%|██████▉   | 277893/400000 [00:38<00:16, 7452.71it/s] 70%|██████▉   | 278639/400000 [00:38<00:16, 7323.38it/s] 70%|██████▉   | 279377/400000 [00:38<00:16, 7339.48it/s] 70%|███████   | 280147/400000 [00:38<00:16, 7441.97it/s] 70%|███████   | 280893/400000 [00:38<00:16, 7431.56it/s] 70%|███████   | 281637/400000 [00:38<00:15, 7401.62it/s] 71%|███████   | 282378/400000 [00:38<00:16, 7170.45it/s] 71%|███████   | 283097/400000 [00:39<00:16, 7158.68it/s] 71%|███████   | 283854/400000 [00:39<00:15, 7274.94it/s] 71%|███████   | 284618/400000 [00:39<00:15, 7380.14it/s] 71%|███████▏  | 285367/400000 [00:39<00:15, 7410.82it/s] 72%|███████▏  | 286110/400000 [00:39<00:15, 7413.29it/s] 72%|███████▏  | 286853/400000 [00:39<00:15, 7299.75it/s] 72%|███████▏  | 287610/400000 [00:39<00:15, 7377.50it/s] 72%|███████▏  | 288358/400000 [00:39<00:15, 7406.35it/s] 72%|███████▏  | 289109/400000 [00:39<00:14, 7436.54it/s] 72%|███████▏  | 289873/400000 [00:39<00:14, 7495.37it/s] 73%|███████▎  | 290633/400000 [00:40<00:14, 7525.36it/s] 73%|███████▎  | 291386/400000 [00:40<00:15, 7192.80it/s] 73%|███████▎  | 292136/400000 [00:40<00:14, 7281.68it/s] 73%|███████▎  | 292867/400000 [00:40<00:14, 7287.07it/s] 73%|███████▎  | 293622/400000 [00:40<00:14, 7360.99it/s] 74%|███████▎  | 294376/400000 [00:40<00:14, 7412.46it/s] 74%|███████▍  | 295119/400000 [00:40<00:14, 7387.30it/s] 74%|███████▍  | 295859/400000 [00:40<00:14, 7322.68it/s] 74%|███████▍  | 296595/400000 [00:40<00:14, 7333.34it/s] 74%|███████▍  | 297329/400000 [00:40<00:14, 7272.94it/s] 75%|███████▍  | 298057/400000 [00:41<00:14, 6949.64it/s] 75%|███████▍  | 298778/400000 [00:41<00:14, 7023.66it/s] 75%|███████▍  | 299503/400000 [00:41<00:14, 7089.90it/s] 75%|███████▌  | 300246/400000 [00:41<00:13, 7187.87it/s] 75%|███████▌  | 300989/400000 [00:41<00:13, 7256.75it/s] 75%|███████▌  | 301740/400000 [00:41<00:13, 7328.15it/s] 76%|███████▌  | 302501/400000 [00:41<00:13, 7409.49it/s] 76%|███████▌  | 303243/400000 [00:41<00:13, 7398.19it/s] 76%|███████▌  | 303984/400000 [00:41<00:13, 7347.37it/s] 76%|███████▌  | 304731/400000 [00:41<00:12, 7383.49it/s] 76%|███████▋  | 305476/400000 [00:42<00:12, 7401.49it/s] 77%|███████▋  | 306232/400000 [00:42<00:12, 7448.20it/s] 77%|███████▋  | 306978/400000 [00:42<00:12, 7406.83it/s] 77%|███████▋  | 307736/400000 [00:42<00:12, 7457.83it/s] 77%|███████▋  | 308483/400000 [00:42<00:12, 7403.45it/s] 77%|███████▋  | 309224/400000 [00:42<00:12, 7346.40it/s] 77%|███████▋  | 309959/400000 [00:42<00:12, 7253.63it/s] 78%|███████▊  | 310685/400000 [00:42<00:12, 7236.68it/s] 78%|███████▊  | 311410/400000 [00:42<00:12, 7208.08it/s] 78%|███████▊  | 312132/400000 [00:42<00:12, 7190.48it/s] 78%|███████▊  | 312852/400000 [00:43<00:12, 7041.79it/s] 78%|███████▊  | 313567/400000 [00:43<00:12, 7072.30it/s] 79%|███████▊  | 314275/400000 [00:43<00:12, 7051.42it/s] 79%|███████▉  | 315007/400000 [00:43<00:11, 7127.79it/s] 79%|███████▉  | 315721/400000 [00:43<00:11, 7097.60it/s] 79%|███████▉  | 316442/400000 [00:43<00:11, 7129.84it/s] 79%|███████▉  | 317156/400000 [00:43<00:11, 7111.16it/s] 79%|███████▉  | 317887/400000 [00:43<00:11, 7169.64it/s] 80%|███████▉  | 318628/400000 [00:43<00:11, 7237.81it/s] 80%|███████▉  | 319371/400000 [00:43<00:11, 7292.11it/s] 80%|████████  | 320101/400000 [00:44<00:10, 7292.58it/s] 80%|████████  | 320831/400000 [00:44<00:11, 7123.39it/s] 80%|████████  | 321562/400000 [00:44<00:10, 7176.95it/s] 81%|████████  | 322306/400000 [00:44<00:10, 7252.21it/s] 81%|████████  | 323057/400000 [00:44<00:10, 7326.38it/s] 81%|████████  | 323794/400000 [00:44<00:10, 7338.14it/s] 81%|████████  | 324529/400000 [00:44<00:10, 7298.82it/s] 81%|████████▏ | 325260/400000 [00:44<00:10, 7094.61it/s] 82%|████████▏ | 326015/400000 [00:44<00:10, 7224.18it/s] 82%|████████▏ | 326743/400000 [00:44<00:10, 7239.50it/s] 82%|████████▏ | 327484/400000 [00:45<00:09, 7287.11it/s] 82%|████████▏ | 328228/400000 [00:45<00:09, 7331.41it/s] 82%|████████▏ | 329003/400000 [00:45<00:09, 7451.67it/s] 82%|████████▏ | 329750/400000 [00:45<00:09, 7452.73it/s] 83%|████████▎ | 330510/400000 [00:45<00:09, 7494.60it/s] 83%|████████▎ | 331260/400000 [00:45<00:09, 7395.78it/s] 83%|████████▎ | 332001/400000 [00:45<00:09, 7366.66it/s] 83%|████████▎ | 332739/400000 [00:45<00:09, 7337.54it/s] 83%|████████▎ | 333483/400000 [00:45<00:09, 7366.38it/s] 84%|████████▎ | 334220/400000 [00:46<00:08, 7351.02it/s] 84%|████████▎ | 334986/400000 [00:46<00:08, 7439.37it/s] 84%|████████▍ | 335731/400000 [00:46<00:08, 7314.38it/s] 84%|████████▍ | 336464/400000 [00:46<00:08, 7288.91it/s] 84%|████████▍ | 337194/400000 [00:46<00:08, 7071.16it/s] 84%|████████▍ | 337903/400000 [00:46<00:09, 6819.88it/s] 85%|████████▍ | 338637/400000 [00:46<00:08, 6966.21it/s] 85%|████████▍ | 339361/400000 [00:46<00:08, 7045.16it/s] 85%|████████▌ | 340077/400000 [00:46<00:08, 7077.95it/s] 85%|████████▌ | 340787/400000 [00:46<00:08, 7021.74it/s] 85%|████████▌ | 341491/400000 [00:47<00:08, 6872.90it/s] 86%|████████▌ | 342206/400000 [00:47<00:08, 6953.28it/s] 86%|████████▌ | 342914/400000 [00:47<00:08, 6990.02it/s] 86%|████████▌ | 343632/400000 [00:47<00:08, 7045.33it/s] 86%|████████▌ | 344365/400000 [00:47<00:07, 7127.73it/s] 86%|████████▋ | 345110/400000 [00:47<00:07, 7219.28it/s] 86%|████████▋ | 345864/400000 [00:47<00:07, 7310.88it/s] 87%|████████▋ | 346598/400000 [00:47<00:07, 7317.17it/s] 87%|████████▋ | 347336/400000 [00:47<00:07, 7333.42it/s] 87%|████████▋ | 348070/400000 [00:47<00:07, 7246.42it/s] 87%|████████▋ | 348796/400000 [00:48<00:07, 7214.95it/s] 87%|████████▋ | 349535/400000 [00:48<00:06, 7264.65it/s] 88%|████████▊ | 350262/400000 [00:48<00:06, 7223.78it/s] 88%|████████▊ | 350985/400000 [00:48<00:06, 7098.60it/s] 88%|████████▊ | 351724/400000 [00:48<00:06, 7183.03it/s] 88%|████████▊ | 352491/400000 [00:48<00:06, 7321.08it/s] 88%|████████▊ | 353257/400000 [00:48<00:06, 7419.39it/s] 89%|████████▊ | 354003/400000 [00:48<00:06, 7430.47it/s] 89%|████████▊ | 354755/400000 [00:48<00:06, 7456.54it/s] 89%|████████▉ | 355502/400000 [00:48<00:06, 7331.40it/s] 89%|████████▉ | 356246/400000 [00:49<00:05, 7361.04it/s] 89%|████████▉ | 357001/400000 [00:49<00:05, 7416.38it/s] 89%|████████▉ | 357745/400000 [00:49<00:05, 7423.08it/s] 90%|████████▉ | 358488/400000 [00:49<00:05, 7410.53it/s] 90%|████████▉ | 359230/400000 [00:49<00:05, 7365.28it/s] 90%|████████▉ | 359997/400000 [00:49<00:05, 7453.79it/s] 90%|█████████ | 360748/400000 [00:49<00:05, 7467.92it/s] 90%|█████████ | 361496/400000 [00:49<00:05, 7460.71it/s] 91%|█████████ | 362243/400000 [00:49<00:05, 7381.72it/s] 91%|█████████ | 362982/400000 [00:49<00:05, 7333.98it/s] 91%|█████████ | 363716/400000 [00:50<00:05, 7175.75it/s] 91%|█████████ | 364472/400000 [00:50<00:04, 7283.81it/s] 91%|█████████▏| 365208/400000 [00:50<00:04, 7304.92it/s] 91%|█████████▏| 365940/400000 [00:50<00:04, 7294.37it/s] 92%|█████████▏| 366703/400000 [00:50<00:04, 7390.72it/s] 92%|█████████▏| 367443/400000 [00:50<00:04, 7388.72it/s] 92%|█████████▏| 368183/400000 [00:50<00:04, 7159.69it/s] 92%|█████████▏| 368928/400000 [00:50<00:04, 7243.65it/s] 92%|█████████▏| 369699/400000 [00:50<00:04, 7375.18it/s] 93%|█████████▎| 370472/400000 [00:50<00:03, 7477.29it/s] 93%|█████████▎| 371222/400000 [00:51<00:03, 7438.55it/s] 93%|█████████▎| 371967/400000 [00:51<00:03, 7350.17it/s] 93%|█████████▎| 372704/400000 [00:51<00:03, 7337.36it/s] 93%|█████████▎| 373457/400000 [00:51<00:03, 7393.05it/s] 94%|█████████▎| 374197/400000 [00:51<00:03, 7358.11it/s] 94%|█████████▎| 374938/400000 [00:51<00:03, 7372.17it/s] 94%|█████████▍| 375676/400000 [00:51<00:03, 7373.34it/s] 94%|█████████▍| 376414/400000 [00:51<00:03, 7266.05it/s] 94%|█████████▍| 377142/400000 [00:51<00:03, 7093.35it/s] 94%|█████████▍| 377876/400000 [00:52<00:03, 7164.39it/s] 95%|█████████▍| 378602/400000 [00:52<00:02, 7190.38it/s] 95%|█████████▍| 379322/400000 [00:52<00:02, 7182.14it/s] 95%|█████████▌| 380046/400000 [00:52<00:02, 7197.72it/s] 95%|█████████▌| 380802/400000 [00:52<00:02, 7300.61it/s] 95%|█████████▌| 381584/400000 [00:52<00:02, 7446.77it/s] 96%|█████████▌| 382351/400000 [00:52<00:02, 7512.18it/s] 96%|█████████▌| 383137/400000 [00:52<00:02, 7612.58it/s] 96%|█████████▌| 383930/400000 [00:52<00:02, 7704.75it/s] 96%|█████████▌| 384702/400000 [00:52<00:02, 7598.22it/s] 96%|█████████▋| 385464/400000 [00:53<00:01, 7604.58it/s] 97%|█████████▋| 386226/400000 [00:53<00:01, 7526.11it/s] 97%|█████████▋| 387030/400000 [00:53<00:01, 7670.36it/s] 97%|█████████▋| 387799/400000 [00:53<00:01, 7676.10it/s] 97%|█████████▋| 388568/400000 [00:53<00:01, 7585.00it/s] 97%|█████████▋| 389328/400000 [00:53<00:01, 7456.73it/s] 98%|█████████▊| 390075/400000 [00:53<00:01, 7457.72it/s] 98%|█████████▊| 390844/400000 [00:53<00:01, 7521.99it/s] 98%|█████████▊| 391619/400000 [00:53<00:01, 7586.42it/s] 98%|█████████▊| 392379/400000 [00:53<00:01, 7546.63it/s] 98%|█████████▊| 393135/400000 [00:54<00:00, 7519.33it/s] 98%|█████████▊| 393888/400000 [00:54<00:00, 7489.21it/s] 99%|█████████▊| 394638/400000 [00:54<00:00, 7486.57it/s] 99%|█████████▉| 395388/400000 [00:54<00:00, 7490.29it/s] 99%|█████████▉| 396151/400000 [00:54<00:00, 7529.61it/s] 99%|█████████▉| 396905/400000 [00:54<00:00, 7464.89it/s] 99%|█████████▉| 397652/400000 [00:54<00:00, 7364.90it/s]100%|█████████▉| 398417/400000 [00:54<00:00, 7447.91it/s]100%|█████████▉| 399184/400000 [00:54<00:00, 7511.05it/s]100%|█████████▉| 399936/400000 [00:54<00:00, 7470.07it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7277.37it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f0d7e0e7128>, <torchtext.data.dataset.TabularDataset object at 0x7f0d7e0e7278>, <torchtext.vocab.Vocab object at 0x7f0d7e0e7198>), {}) 


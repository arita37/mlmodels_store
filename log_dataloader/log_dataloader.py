
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '097d79840b302a3e2f9efdf341ce6b4d32d7a46c', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/097d79840b302a3e2f9efdf341ce6b4d32d7a46c

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fb237a242f0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fb237a242f0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fb2a9a720d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fb2a9a720d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fb2412049d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fb2412049d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb2573568c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb2573568c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb2573568c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3637248/9912422 [00:00<00:00, 36360378.09it/s]9920512it [00:00, 35541166.66it/s]                             
0it [00:00, ?it/s]32768it [00:00, 429862.11it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 8009152.12it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 72789.58it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb239dac518>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb239daccc0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fb257356510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fb257356510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fb257356510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:53,  3.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:53,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:53,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:52,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:52,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:52,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:51,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:36,  4.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:36,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:36,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:35,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:35,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:25,  5.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:25,  5.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:24,  5.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:23,  5.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:17,  8.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:17,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:16,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:16,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:16,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:11, 11.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:11, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:11, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:11, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:11, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:11, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:08, 14.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:08, 14.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:08, 14.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 14.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 14.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:05, 19.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:05, 19.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:05, 19.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:05, 19.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:05, 19.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 19.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 25.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 25.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 25.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 25.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 25.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 25.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 25.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 25.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 31.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 38.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 38.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 38.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 38.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 38.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 38.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 38.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 38.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 38.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 45.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 45.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 45.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 45.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 45.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 45.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 45.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 45.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 45.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 51.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 51.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 51.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 51.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 51.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 51.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 51.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 51.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 55.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 55.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 55.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 55.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 55.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 55.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 55.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 55.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 55.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:00, 60.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:00, 60.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 60.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 60.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 60.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 60.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 60.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 60.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 60.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 65.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 65.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 65.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 65.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 65.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 65.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 65.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 65.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 65.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 68.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 71.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 71.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 71.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 71.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 71.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 71.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 71.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 71.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 71.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 73.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 73.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 73.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 73.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 73.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 73.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 73.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 75.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 75.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 75.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 75.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 75.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 75.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 75.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 75.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 76.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 76.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 76.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 76.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 76.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 76.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 76.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 76.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 76.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.66 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.99s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 76.66 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.99s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.99s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.48 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.99s/ url]
0 examples [00:00, ? examples/s]2020-05-28 18:09:55.508520: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-28 18:09:55.513740: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-28 18:09:55.513970: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56477b25c0f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-28 18:09:55.513990: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
56 examples [00:00, 558.83 examples/s]141 examples [00:00, 621.19 examples/s]221 examples [00:00, 664.22 examples/s]307 examples [00:00, 711.52 examples/s]391 examples [00:00, 744.60 examples/s]476 examples [00:00, 772.78 examples/s]561 examples [00:00, 794.06 examples/s]648 examples [00:00, 814.28 examples/s]735 examples [00:00, 828.34 examples/s]822 examples [00:01, 838.02 examples/s]907 examples [00:01, 838.77 examples/s]993 examples [00:01, 844.15 examples/s]1078 examples [00:01, 843.09 examples/s]1162 examples [00:01, 831.47 examples/s]1249 examples [00:01, 841.93 examples/s]1336 examples [00:01, 849.49 examples/s]1424 examples [00:01, 857.02 examples/s]1510 examples [00:01, 855.88 examples/s]1599 examples [00:01, 863.42 examples/s]1687 examples [00:02, 867.22 examples/s]1774 examples [00:02, 863.89 examples/s]1862 examples [00:02, 865.95 examples/s]1949 examples [00:02, 852.41 examples/s]2035 examples [00:02, 852.45 examples/s]2122 examples [00:02, 856.35 examples/s]2208 examples [00:02, 854.30 examples/s]2294 examples [00:02, 851.82 examples/s]2380 examples [00:02, 840.43 examples/s]2465 examples [00:02, 840.97 examples/s]2551 examples [00:03, 844.18 examples/s]2637 examples [00:03, 846.35 examples/s]2722 examples [00:03, 836.75 examples/s]2809 examples [00:03, 845.02 examples/s]2897 examples [00:03, 854.16 examples/s]2983 examples [00:03, 849.94 examples/s]3069 examples [00:03, 841.34 examples/s]3154 examples [00:03, 826.50 examples/s]3238 examples [00:03, 830.23 examples/s]3322 examples [00:03, 827.47 examples/s]3405 examples [00:04, 816.57 examples/s]3487 examples [00:04, 814.77 examples/s]3570 examples [00:04, 817.90 examples/s]3652 examples [00:04, 797.32 examples/s]3736 examples [00:04, 808.39 examples/s]3819 examples [00:04, 814.21 examples/s]3903 examples [00:04, 820.10 examples/s]3989 examples [00:04, 830.19 examples/s]4073 examples [00:04, 831.09 examples/s]4160 examples [00:04, 841.34 examples/s]4246 examples [00:05, 844.65 examples/s]4331 examples [00:05, 835.59 examples/s]4422 examples [00:05, 855.67 examples/s]4508 examples [00:05, 826.37 examples/s]4592 examples [00:05, 828.97 examples/s]4677 examples [00:05, 834.48 examples/s]4761 examples [00:05, 794.46 examples/s]4841 examples [00:05, 796.09 examples/s]4923 examples [00:05, 803.05 examples/s]5004 examples [00:06, 801.53 examples/s]5087 examples [00:06, 808.36 examples/s]5173 examples [00:06, 822.81 examples/s]5259 examples [00:06, 831.43 examples/s]5347 examples [00:06, 845.01 examples/s]5434 examples [00:06, 851.79 examples/s]5520 examples [00:06, 846.41 examples/s]5605 examples [00:06, 828.33 examples/s]5689 examples [00:06, 830.55 examples/s]5779 examples [00:06, 848.28 examples/s]5867 examples [00:07, 857.31 examples/s]5954 examples [00:07, 859.59 examples/s]6041 examples [00:07, 855.50 examples/s]6127 examples [00:07, 856.09 examples/s]6214 examples [00:07, 859.20 examples/s]6300 examples [00:07, 858.68 examples/s]6386 examples [00:07, 844.46 examples/s]6471 examples [00:07, 839.44 examples/s]6556 examples [00:07, 835.58 examples/s]6643 examples [00:07, 843.96 examples/s]6729 examples [00:08, 847.57 examples/s]6817 examples [00:08, 854.25 examples/s]6903 examples [00:08, 850.34 examples/s]6991 examples [00:08, 856.34 examples/s]7077 examples [00:08, 844.21 examples/s]7162 examples [00:08, 838.46 examples/s]7251 examples [00:08, 851.20 examples/s]7337 examples [00:08, 844.92 examples/s]7423 examples [00:08, 849.27 examples/s]7510 examples [00:08, 853.97 examples/s]7598 examples [00:09, 861.10 examples/s]7685 examples [00:09, 860.72 examples/s]7772 examples [00:09, 849.35 examples/s]7861 examples [00:09, 861.05 examples/s]7948 examples [00:09, 860.38 examples/s]8035 examples [00:09, 847.75 examples/s]8120 examples [00:09, 841.65 examples/s]8205 examples [00:09, 821.17 examples/s]8288 examples [00:09, 801.65 examples/s]8369 examples [00:10, 800.75 examples/s]8452 examples [00:10, 807.70 examples/s]8534 examples [00:10, 810.61 examples/s]8616 examples [00:10, 803.51 examples/s]8697 examples [00:10, 803.71 examples/s]8778 examples [00:10, 803.63 examples/s]8860 examples [00:10, 805.97 examples/s]8947 examples [00:10, 821.91 examples/s]9033 examples [00:10, 832.62 examples/s]9117 examples [00:10, 810.93 examples/s]9203 examples [00:11, 823.70 examples/s]9292 examples [00:11, 840.53 examples/s]9377 examples [00:11, 839.83 examples/s]9462 examples [00:11, 842.68 examples/s]9548 examples [00:11, 846.16 examples/s]9633 examples [00:11, 820.02 examples/s]9719 examples [00:11, 831.44 examples/s]9804 examples [00:11, 835.93 examples/s]9892 examples [00:11, 846.30 examples/s]9977 examples [00:11, 821.70 examples/s]10060 examples [00:12, 766.76 examples/s]10144 examples [00:12, 785.86 examples/s]10227 examples [00:12, 798.34 examples/s]10312 examples [00:12, 811.10 examples/s]10402 examples [00:12, 833.57 examples/s]10488 examples [00:12, 838.87 examples/s]10575 examples [00:12, 847.64 examples/s]10661 examples [00:12, 839.55 examples/s]10746 examples [00:12, 832.55 examples/s]10831 examples [00:12, 836.07 examples/s]10915 examples [00:13, 833.75 examples/s]10999 examples [00:13, 810.66 examples/s]11081 examples [00:13, 802.26 examples/s]11162 examples [00:13, 803.62 examples/s]11243 examples [00:13, 785.82 examples/s]11327 examples [00:13, 801.14 examples/s]11408 examples [00:13, 790.59 examples/s]11488 examples [00:13, 765.27 examples/s]11565 examples [00:13, 745.16 examples/s]11644 examples [00:14, 757.99 examples/s]11727 examples [00:14, 777.37 examples/s]11815 examples [00:14, 805.11 examples/s]11903 examples [00:14, 825.79 examples/s]11992 examples [00:14, 841.42 examples/s]12077 examples [00:14, 828.21 examples/s]12161 examples [00:14, 803.83 examples/s]12245 examples [00:14, 813.50 examples/s]12331 examples [00:14, 825.69 examples/s]12418 examples [00:14, 837.76 examples/s]12503 examples [00:15, 841.07 examples/s]12590 examples [00:15, 847.39 examples/s]12677 examples [00:15, 852.12 examples/s]12763 examples [00:15, 828.29 examples/s]12848 examples [00:15, 833.26 examples/s]12938 examples [00:15, 851.28 examples/s]13024 examples [00:15, 853.23 examples/s]13110 examples [00:15, 846.90 examples/s]13197 examples [00:15, 851.62 examples/s]13283 examples [00:15, 838.78 examples/s]13367 examples [00:16, 833.98 examples/s]13451 examples [00:16, 833.02 examples/s]13535 examples [00:16, 827.75 examples/s]13618 examples [00:16, 813.32 examples/s]13700 examples [00:16, 804.24 examples/s]13784 examples [00:16, 813.99 examples/s]13866 examples [00:16, 811.28 examples/s]13953 examples [00:16, 825.26 examples/s]14040 examples [00:16, 837.84 examples/s]14124 examples [00:17, 833.27 examples/s]14208 examples [00:17, 819.65 examples/s]14291 examples [00:17, 809.99 examples/s]14378 examples [00:17, 826.30 examples/s]14462 examples [00:17, 827.93 examples/s]14549 examples [00:17, 837.41 examples/s]14634 examples [00:17, 838.80 examples/s]14718 examples [00:17, 806.41 examples/s]14799 examples [00:17, 791.52 examples/s]14881 examples [00:17, 797.13 examples/s]14967 examples [00:18, 813.44 examples/s]15052 examples [00:18, 823.08 examples/s]15138 examples [00:18, 832.44 examples/s]15224 examples [00:18, 840.05 examples/s]15310 examples [00:18, 845.49 examples/s]15395 examples [00:18, 833.80 examples/s]15482 examples [00:18, 841.38 examples/s]15567 examples [00:18, 837.24 examples/s]15652 examples [00:18, 838.59 examples/s]15736 examples [00:18, 830.77 examples/s]15820 examples [00:19, 821.37 examples/s]15903 examples [00:19, 806.28 examples/s]15984 examples [00:19, 795.55 examples/s]16067 examples [00:19, 804.76 examples/s]16150 examples [00:19, 810.81 examples/s]16232 examples [00:19, 805.18 examples/s]16313 examples [00:19, 793.91 examples/s]16393 examples [00:19, 787.10 examples/s]16477 examples [00:19, 799.05 examples/s]16562 examples [00:19, 811.90 examples/s]16651 examples [00:20, 830.96 examples/s]16735 examples [00:20, 831.25 examples/s]16819 examples [00:20, 832.36 examples/s]16903 examples [00:20, 831.74 examples/s]16987 examples [00:20, 832.90 examples/s]17071 examples [00:20, 834.49 examples/s]17157 examples [00:20, 839.16 examples/s]17241 examples [00:20, 822.52 examples/s]17325 examples [00:20, 827.03 examples/s]17408 examples [00:21, 823.37 examples/s]17495 examples [00:21, 834.27 examples/s]17580 examples [00:21, 837.49 examples/s]17664 examples [00:21, 820.47 examples/s]17748 examples [00:21, 825.06 examples/s]17833 examples [00:21, 831.91 examples/s]17919 examples [00:21, 838.97 examples/s]18009 examples [00:21, 854.07 examples/s]18095 examples [00:21, 848.44 examples/s]18180 examples [00:21, 837.68 examples/s]18266 examples [00:22, 843.69 examples/s]18351 examples [00:22, 840.86 examples/s]18437 examples [00:22, 845.17 examples/s]18523 examples [00:22, 848.59 examples/s]18608 examples [00:22, 839.31 examples/s]18692 examples [00:22, 838.73 examples/s]18777 examples [00:22, 840.86 examples/s]18862 examples [00:22, 838.54 examples/s]18948 examples [00:22, 842.29 examples/s]19033 examples [00:22, 820.47 examples/s]19116 examples [00:23, 817.21 examples/s]19199 examples [00:23, 815.75 examples/s]19281 examples [00:23, 816.64 examples/s]19363 examples [00:23, 816.63 examples/s]19448 examples [00:23, 824.23 examples/s]19534 examples [00:23, 834.56 examples/s]19618 examples [00:23, 828.46 examples/s]19701 examples [00:23, 816.93 examples/s]19783 examples [00:23, 804.17 examples/s]19864 examples [00:23, 800.10 examples/s]19949 examples [00:24, 812.65 examples/s]20031 examples [00:24, 769.93 examples/s]20115 examples [00:24, 787.93 examples/s]20199 examples [00:24, 799.56 examples/s]20281 examples [00:24, 805.00 examples/s]20367 examples [00:24, 820.63 examples/s]20450 examples [00:24, 821.71 examples/s]20535 examples [00:24, 829.45 examples/s]20619 examples [00:24, 822.91 examples/s]20702 examples [00:25, 808.43 examples/s]20783 examples [00:25, 806.46 examples/s]20869 examples [00:25, 819.41 examples/s]20953 examples [00:25, 825.10 examples/s]21039 examples [00:25, 833.92 examples/s]21124 examples [00:25, 836.49 examples/s]21209 examples [00:25, 839.45 examples/s]21296 examples [00:25, 845.62 examples/s]21384 examples [00:25, 853.70 examples/s]21470 examples [00:25, 852.16 examples/s]21556 examples [00:26, 832.58 examples/s]21640 examples [00:26, 816.36 examples/s]21726 examples [00:26, 826.59 examples/s]21813 examples [00:26, 839.11 examples/s]21898 examples [00:26, 804.46 examples/s]21979 examples [00:26, 805.19 examples/s]22065 examples [00:26, 818.44 examples/s]22151 examples [00:26, 828.09 examples/s]22238 examples [00:26, 839.89 examples/s]22324 examples [00:26, 843.52 examples/s]22409 examples [00:27, 840.02 examples/s]22494 examples [00:27, 842.62 examples/s]22579 examples [00:27, 843.78 examples/s]22666 examples [00:27, 849.42 examples/s]22751 examples [00:27, 846.14 examples/s]22836 examples [00:27, 817.02 examples/s]22918 examples [00:27, 810.10 examples/s]23000 examples [00:27, 796.48 examples/s]23083 examples [00:27, 803.62 examples/s]23164 examples [00:27, 783.63 examples/s]23247 examples [00:28, 795.84 examples/s]23332 examples [00:28, 809.09 examples/s]23417 examples [00:28, 819.77 examples/s]23505 examples [00:28, 836.66 examples/s]23589 examples [00:28, 832.74 examples/s]23673 examples [00:28, 809.72 examples/s]23755 examples [00:28, 811.78 examples/s]23838 examples [00:28, 815.99 examples/s]23920 examples [00:28, 814.75 examples/s]24004 examples [00:29, 820.51 examples/s]24091 examples [00:29, 833.55 examples/s]24175 examples [00:29, 799.16 examples/s]24256 examples [00:29, 791.94 examples/s]24336 examples [00:29, 790.09 examples/s]24416 examples [00:29, 777.51 examples/s]24497 examples [00:29, 785.06 examples/s]24583 examples [00:29, 803.98 examples/s]24664 examples [00:29, 803.67 examples/s]24748 examples [00:29, 812.82 examples/s]24833 examples [00:30, 821.30 examples/s]24917 examples [00:30, 824.39 examples/s]25000 examples [00:30, 820.05 examples/s]25092 examples [00:30, 847.17 examples/s]25177 examples [00:30, 844.16 examples/s]25262 examples [00:30, 844.16 examples/s]25348 examples [00:30, 847.06 examples/s]25438 examples [00:30, 860.20 examples/s]25527 examples [00:30, 866.80 examples/s]25614 examples [00:30, 843.82 examples/s]25699 examples [00:31, 844.05 examples/s]25784 examples [00:31, 838.70 examples/s]25871 examples [00:31, 846.06 examples/s]25956 examples [00:31, 841.48 examples/s]26043 examples [00:31, 849.20 examples/s]26129 examples [00:31, 849.58 examples/s]26215 examples [00:31, 828.18 examples/s]26304 examples [00:31, 845.07 examples/s]26391 examples [00:31, 850.16 examples/s]26478 examples [00:31, 855.49 examples/s]26567 examples [00:32, 863.48 examples/s]26654 examples [00:32, 851.69 examples/s]26740 examples [00:32, 847.04 examples/s]26825 examples [00:32, 837.59 examples/s]26909 examples [00:32, 813.10 examples/s]26991 examples [00:32, 811.25 examples/s]27075 examples [00:32, 818.83 examples/s]27162 examples [00:32, 832.06 examples/s]27246 examples [00:32, 829.80 examples/s]27334 examples [00:33, 842.42 examples/s]27419 examples [00:33, 827.80 examples/s]27504 examples [00:33, 832.40 examples/s]27592 examples [00:33, 845.17 examples/s]27677 examples [00:33, 843.06 examples/s]27762 examples [00:33, 843.56 examples/s]27847 examples [00:33, 840.92 examples/s]27932 examples [00:33, 841.71 examples/s]28018 examples [00:33, 846.56 examples/s]28103 examples [00:33, 843.51 examples/s]28188 examples [00:34, 827.90 examples/s]28272 examples [00:34, 831.07 examples/s]28356 examples [00:34, 800.54 examples/s]28439 examples [00:34, 808.91 examples/s]28521 examples [00:34, 808.80 examples/s]28605 examples [00:34, 815.89 examples/s]28690 examples [00:34, 824.24 examples/s]28773 examples [00:34, 813.56 examples/s]28857 examples [00:34, 819.02 examples/s]28939 examples [00:34, 800.93 examples/s]29023 examples [00:35, 809.24 examples/s]29107 examples [00:35, 817.52 examples/s]29189 examples [00:35, 809.09 examples/s]29273 examples [00:35, 816.32 examples/s]29356 examples [00:35, 818.84 examples/s]29441 examples [00:35, 825.28 examples/s]29525 examples [00:35, 828.86 examples/s]29608 examples [00:35, 826.89 examples/s]29691 examples [00:35, 827.75 examples/s]29776 examples [00:35, 834.26 examples/s]29860 examples [00:36, 835.10 examples/s]29944 examples [00:36, 831.32 examples/s]30028 examples [00:36, 778.47 examples/s]30110 examples [00:36, 788.58 examples/s]30196 examples [00:36, 807.23 examples/s]30278 examples [00:36, 808.68 examples/s]30365 examples [00:36, 824.62 examples/s]30455 examples [00:36, 843.77 examples/s]30542 examples [00:36, 850.62 examples/s]30628 examples [00:36, 843.34 examples/s]30713 examples [00:37, 821.30 examples/s]30796 examples [00:37, 822.54 examples/s]30880 examples [00:37, 827.39 examples/s]30965 examples [00:37, 833.26 examples/s]31051 examples [00:37, 840.43 examples/s]31136 examples [00:37, 826.10 examples/s]31225 examples [00:37, 843.85 examples/s]31313 examples [00:37, 854.24 examples/s]31402 examples [00:37, 862.26 examples/s]31489 examples [00:38, 861.05 examples/s]31576 examples [00:38, 857.74 examples/s]31662 examples [00:38, 855.66 examples/s]31748 examples [00:38, 819.00 examples/s]31835 examples [00:38, 832.85 examples/s]31924 examples [00:38, 847.18 examples/s]32011 examples [00:38, 852.44 examples/s]32099 examples [00:38, 859.45 examples/s]32187 examples [00:38, 864.29 examples/s]32275 examples [00:38, 867.71 examples/s]32362 examples [00:39, 867.99 examples/s]32449 examples [00:39, 858.10 examples/s]32537 examples [00:39, 862.13 examples/s]32624 examples [00:39, 861.11 examples/s]32715 examples [00:39, 873.74 examples/s]32804 examples [00:39, 876.79 examples/s]32892 examples [00:39, 872.96 examples/s]32982 examples [00:39, 877.93 examples/s]33070 examples [00:39, 875.88 examples/s]33158 examples [00:39, 868.73 examples/s]33245 examples [00:40, 862.72 examples/s]33332 examples [00:40, 859.28 examples/s]33418 examples [00:40, 859.03 examples/s]33504 examples [00:40, 854.92 examples/s]33591 examples [00:40, 857.70 examples/s]33678 examples [00:40, 861.06 examples/s]33765 examples [00:40, 856.17 examples/s]33852 examples [00:40, 859.32 examples/s]33938 examples [00:40, 855.09 examples/s]34026 examples [00:40, 860.97 examples/s]34114 examples [00:41, 863.54 examples/s]34202 examples [00:41, 868.21 examples/s]34293 examples [00:41, 877.68 examples/s]34382 examples [00:41, 879.11 examples/s]34472 examples [00:41, 883.99 examples/s]34561 examples [00:41, 868.22 examples/s]34648 examples [00:41, 862.95 examples/s]34735 examples [00:41, 864.28 examples/s]34822 examples [00:41, 865.06 examples/s]34909 examples [00:41, 856.42 examples/s]34995 examples [00:42, 855.27 examples/s]35082 examples [00:42, 859.52 examples/s]35170 examples [00:42, 864.26 examples/s]35258 examples [00:42, 866.24 examples/s]35346 examples [00:42, 867.21 examples/s]35434 examples [00:42, 868.48 examples/s]35522 examples [00:42, 870.47 examples/s]35610 examples [00:42, 871.20 examples/s]35698 examples [00:42, 806.27 examples/s]35781 examples [00:43, 811.90 examples/s]35863 examples [00:43, 811.54 examples/s]35945 examples [00:43, 778.58 examples/s]36028 examples [00:43, 793.06 examples/s]36108 examples [00:43, 781.56 examples/s]36191 examples [00:43, 794.64 examples/s]36279 examples [00:43, 816.34 examples/s]36364 examples [00:43, 825.98 examples/s]36450 examples [00:43, 834.44 examples/s]36535 examples [00:43, 837.43 examples/s]36619 examples [00:44, 836.36 examples/s]36705 examples [00:44, 842.46 examples/s]36791 examples [00:44, 845.24 examples/s]36876 examples [00:44, 839.80 examples/s]36961 examples [00:44, 836.25 examples/s]37048 examples [00:44, 844.08 examples/s]37135 examples [00:44, 849.06 examples/s]37224 examples [00:44, 858.42 examples/s]37310 examples [00:44, 858.72 examples/s]37396 examples [00:44, 857.83 examples/s]37482 examples [00:45, 809.47 examples/s]37566 examples [00:45, 815.81 examples/s]37653 examples [00:45, 831.19 examples/s]37738 examples [00:45, 836.69 examples/s]37822 examples [00:45, 835.96 examples/s]37908 examples [00:45, 842.12 examples/s]37993 examples [00:45, 841.88 examples/s]38079 examples [00:45, 847.16 examples/s]38166 examples [00:45, 851.30 examples/s]38252 examples [00:45, 820.36 examples/s]38335 examples [00:46, 817.64 examples/s]38419 examples [00:46, 822.36 examples/s]38502 examples [00:46, 823.18 examples/s]38587 examples [00:46, 829.46 examples/s]38671 examples [00:46, 829.86 examples/s]38758 examples [00:46, 841.30 examples/s]38847 examples [00:46, 852.79 examples/s]38933 examples [00:46, 851.29 examples/s]39019 examples [00:46, 847.07 examples/s]39104 examples [00:47, 821.25 examples/s]39187 examples [00:47, 809.98 examples/s]39271 examples [00:47, 816.78 examples/s]39355 examples [00:47, 821.80 examples/s]39439 examples [00:47, 825.06 examples/s]39523 examples [00:47, 827.42 examples/s]39606 examples [00:47, 825.41 examples/s]39689 examples [00:47, 819.76 examples/s]39772 examples [00:47, 819.76 examples/s]39855 examples [00:47, 811.47 examples/s]39939 examples [00:48, 819.04 examples/s]40021 examples [00:48, 753.87 examples/s]40111 examples [00:48, 791.06 examples/s]40192 examples [00:48, 793.67 examples/s]40276 examples [00:48, 806.56 examples/s]40362 examples [00:48, 821.46 examples/s]40451 examples [00:48, 838.23 examples/s]40536 examples [00:48, 815.24 examples/s]40621 examples [00:48, 824.14 examples/s]40704 examples [00:48, 821.46 examples/s]40790 examples [00:49, 830.57 examples/s]40876 examples [00:49, 838.31 examples/s]40960 examples [00:49, 812.90 examples/s]41045 examples [00:49, 821.16 examples/s]41131 examples [00:49, 830.70 examples/s]41215 examples [00:49, 814.36 examples/s]41299 examples [00:49, 821.47 examples/s]41382 examples [00:49, 812.68 examples/s]41464 examples [00:49, 808.17 examples/s]41546 examples [00:50, 809.50 examples/s]41631 examples [00:50, 820.48 examples/s]41714 examples [00:50, 822.02 examples/s]41798 examples [00:50, 825.86 examples/s]41881 examples [00:50, 825.71 examples/s]41965 examples [00:50, 828.59 examples/s]42049 examples [00:50, 830.51 examples/s]42133 examples [00:50, 832.57 examples/s]42220 examples [00:50, 842.21 examples/s]42307 examples [00:50, 847.82 examples/s]42394 examples [00:51, 852.43 examples/s]42483 examples [00:51, 863.31 examples/s]42573 examples [00:51, 872.05 examples/s]42661 examples [00:51, 851.72 examples/s]42747 examples [00:51, 853.39 examples/s]42833 examples [00:51, 849.64 examples/s]42919 examples [00:51, 851.42 examples/s]43005 examples [00:51, 853.27 examples/s]43091 examples [00:51, 839.40 examples/s]43176 examples [00:51, 832.20 examples/s]43260 examples [00:52, 830.75 examples/s]43344 examples [00:52, 829.42 examples/s]43427 examples [00:52, 829.15 examples/s]43510 examples [00:52, 828.33 examples/s]43594 examples [00:52, 829.88 examples/s]43680 examples [00:52, 837.06 examples/s]43766 examples [00:52, 843.49 examples/s]43853 examples [00:52, 849.86 examples/s]43939 examples [00:52, 850.04 examples/s]44025 examples [00:52, 818.18 examples/s]44110 examples [00:53, 827.30 examples/s]44198 examples [00:53, 839.71 examples/s]44284 examples [00:53, 844.21 examples/s]44370 examples [00:53, 847.34 examples/s]44455 examples [00:53, 844.82 examples/s]44540 examples [00:53, 831.19 examples/s]44624 examples [00:53, 832.02 examples/s]44713 examples [00:53, 846.38 examples/s]44798 examples [00:53, 835.93 examples/s]44885 examples [00:53, 844.02 examples/s]44970 examples [00:54, 844.81 examples/s]45056 examples [00:54, 847.70 examples/s]45143 examples [00:54, 849.40 examples/s]45230 examples [00:54, 854.70 examples/s]45316 examples [00:54, 826.28 examples/s]45400 examples [00:54, 828.40 examples/s]45488 examples [00:54, 841.32 examples/s]45575 examples [00:54, 849.39 examples/s]45661 examples [00:54, 852.08 examples/s]45750 examples [00:54, 861.97 examples/s]45837 examples [00:55, 859.95 examples/s]45924 examples [00:55, 856.36 examples/s]46012 examples [00:55, 860.60 examples/s]46099 examples [00:55, 862.92 examples/s]46186 examples [00:55, 823.58 examples/s]46269 examples [00:55, 824.60 examples/s]46355 examples [00:55, 833.45 examples/s]46443 examples [00:55, 846.73 examples/s]46531 examples [00:55, 855.08 examples/s]46617 examples [00:56, 839.07 examples/s]46704 examples [00:56, 845.53 examples/s]46791 examples [00:56, 850.77 examples/s]46879 examples [00:56, 858.46 examples/s]46965 examples [00:56, 852.73 examples/s]47051 examples [00:56, 854.70 examples/s]47137 examples [00:56, 841.85 examples/s]47225 examples [00:56, 850.12 examples/s]47311 examples [00:56, 848.13 examples/s]47397 examples [00:56, 849.98 examples/s]47483 examples [00:57, 848.58 examples/s]47568 examples [00:57, 834.96 examples/s]47656 examples [00:57, 847.46 examples/s]47744 examples [00:57, 856.28 examples/s]47833 examples [00:57, 864.73 examples/s]47920 examples [00:57, 859.17 examples/s]48006 examples [00:57, 858.67 examples/s]48094 examples [00:57, 863.37 examples/s]48181 examples [00:57, 858.64 examples/s]48268 examples [00:57, 860.99 examples/s]48355 examples [00:58, 860.81 examples/s]48442 examples [00:58, 856.98 examples/s]48530 examples [00:58, 861.49 examples/s]48617 examples [00:58, 843.25 examples/s]48702 examples [00:58, 824.11 examples/s]48787 examples [00:58, 831.66 examples/s]48872 examples [00:58, 835.34 examples/s]48956 examples [00:58, 822.38 examples/s]49039 examples [00:58, 819.23 examples/s]49122 examples [00:58, 813.08 examples/s]49204 examples [00:59, 814.16 examples/s]49287 examples [00:59, 817.84 examples/s]49369 examples [00:59, 817.56 examples/s]49451 examples [00:59, 814.42 examples/s]49534 examples [00:59, 817.56 examples/s]49622 examples [00:59, 834.02 examples/s]49706 examples [00:59, 820.48 examples/s]49792 examples [00:59, 829.51 examples/s]49881 examples [00:59, 845.01 examples/s]49966 examples [01:00, 845.23 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5494/50000 [00:00<00:00, 54937.69 examples/s] 31%|███▏      | 15728/50000 [00:00<00:00, 63803.46 examples/s] 53%|█████▎    | 26423/50000 [00:00<00:00, 72588.62 examples/s] 74%|███████▍  | 37164/50000 [00:00<00:00, 80408.66 examples/s] 95%|█████████▍| 47300/50000 [00:00<00:00, 85723.79 examples/s]                                                               0 examples [00:00, ? examples/s]66 examples [00:00, 656.75 examples/s]152 examples [00:00, 705.48 examples/s]237 examples [00:00, 743.24 examples/s]323 examples [00:00, 774.26 examples/s]411 examples [00:00, 801.77 examples/s]499 examples [00:00, 822.56 examples/s]585 examples [00:00, 833.26 examples/s]674 examples [00:00, 848.30 examples/s]763 examples [00:00, 857.68 examples/s]850 examples [00:01, 860.36 examples/s]938 examples [00:01, 864.67 examples/s]1024 examples [00:01, 841.49 examples/s]1113 examples [00:01, 852.11 examples/s]1199 examples [00:01, 853.34 examples/s]1286 examples [00:01, 856.13 examples/s]1373 examples [00:01, 857.63 examples/s]1459 examples [00:01, 670.09 examples/s]1546 examples [00:01, 718.72 examples/s]1632 examples [00:02, 755.30 examples/s]1718 examples [00:02, 782.72 examples/s]1805 examples [00:02, 805.06 examples/s]1892 examples [00:02, 822.31 examples/s]1977 examples [00:02, 824.28 examples/s]2061 examples [00:02, 816.66 examples/s]2144 examples [00:02, 816.01 examples/s]2229 examples [00:02, 825.21 examples/s]2317 examples [00:02, 839.15 examples/s]2402 examples [00:02, 812.45 examples/s]2484 examples [00:03, 804.76 examples/s]2572 examples [00:03, 822.92 examples/s]2659 examples [00:03, 835.23 examples/s]2744 examples [00:03, 838.72 examples/s]2832 examples [00:03, 848.98 examples/s]2918 examples [00:03, 849.21 examples/s]3004 examples [00:03, 842.92 examples/s]3093 examples [00:03, 854.16 examples/s]3181 examples [00:03, 861.09 examples/s]3270 examples [00:03, 869.56 examples/s]3358 examples [00:04, 869.91 examples/s]3446 examples [00:04, 857.23 examples/s]3533 examples [00:04, 859.06 examples/s]3619 examples [00:04, 851.36 examples/s]3707 examples [00:04, 858.80 examples/s]3795 examples [00:04, 864.93 examples/s]3882 examples [00:04, 852.65 examples/s]3969 examples [00:04, 856.76 examples/s]4057 examples [00:04, 863.29 examples/s]4145 examples [00:04, 867.47 examples/s]4232 examples [00:05, 861.08 examples/s]4319 examples [00:05, 862.20 examples/s]4407 examples [00:05, 864.75 examples/s]4495 examples [00:05, 866.66 examples/s]4582 examples [00:05, 864.36 examples/s]4669 examples [00:05, 849.51 examples/s]4755 examples [00:05, 851.30 examples/s]4843 examples [00:05, 857.91 examples/s]4929 examples [00:05, 856.12 examples/s]5017 examples [00:05, 861.77 examples/s]5104 examples [00:06, 859.77 examples/s]5191 examples [00:06, 856.20 examples/s]5278 examples [00:06, 858.20 examples/s]5364 examples [00:06, 840.55 examples/s]5449 examples [00:06, 823.24 examples/s]5532 examples [00:06, 813.03 examples/s]5614 examples [00:06, 801.16 examples/s]5698 examples [00:06, 811.53 examples/s]5786 examples [00:06, 829.47 examples/s]5870 examples [00:07, 824.49 examples/s]5957 examples [00:07, 835.77 examples/s]6048 examples [00:07, 854.04 examples/s]6136 examples [00:07, 860.67 examples/s]6223 examples [00:07, 850.59 examples/s]6310 examples [00:07, 855.88 examples/s]6396 examples [00:07, 856.15 examples/s]6482 examples [00:07, 853.23 examples/s]6570 examples [00:07, 858.14 examples/s]6656 examples [00:07, 857.71 examples/s]6742 examples [00:08, 847.96 examples/s]6829 examples [00:08, 852.82 examples/s]6915 examples [00:08, 853.01 examples/s]7004 examples [00:08, 862.76 examples/s]7092 examples [00:08, 864.99 examples/s]7179 examples [00:08, 861.18 examples/s]7266 examples [00:08, 832.21 examples/s]7353 examples [00:08, 842.30 examples/s]7441 examples [00:08, 852.39 examples/s]7527 examples [00:08, 853.44 examples/s]7616 examples [00:09, 863.08 examples/s]7704 examples [00:09, 865.63 examples/s]7792 examples [00:09, 869.36 examples/s]7882 examples [00:09, 877.22 examples/s]7970 examples [00:09, 874.67 examples/s]8058 examples [00:09, 875.34 examples/s]8146 examples [00:09, 852.59 examples/s]8232 examples [00:09, 814.12 examples/s]8316 examples [00:09, 821.00 examples/s]8404 examples [00:09, 837.74 examples/s]8490 examples [00:10, 843.08 examples/s]8578 examples [00:10, 850.94 examples/s]8664 examples [00:10, 831.72 examples/s]8748 examples [00:10, 787.04 examples/s]8828 examples [00:10, 776.83 examples/s]8916 examples [00:10, 803.30 examples/s]9005 examples [00:10, 824.88 examples/s]9094 examples [00:10, 841.03 examples/s]9180 examples [00:10, 844.89 examples/s]9266 examples [00:11, 846.64 examples/s]9354 examples [00:11, 854.33 examples/s]9442 examples [00:11, 861.14 examples/s]9529 examples [00:11, 862.96 examples/s]9616 examples [00:11, 859.64 examples/s]9703 examples [00:11, 857.43 examples/s]9791 examples [00:11, 861.87 examples/s]9878 examples [00:11, 837.73 examples/s]9964 examples [00:11, 843.07 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]100%|█████████▉| 9954/10000 [00:00<00:00, 99534.16 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKW9YV0/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKW9YV0/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb2573568c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb2573568c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb2573568c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb2c31f3470>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb24090cb00>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb2573568c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb2573568c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb2573568c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb23aa08438>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb23aa082e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fb22f5316a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fb22f5316a8> 

  function with postional parmater data_info <function split_train_valid at 0x7fb22f5316a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fb22f5317b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fb22f5317b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fb22f5317b8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=a3d9a4275cb3381343c5631a36a961458b5321d2d2ac596e87815cda22a7499d
  Stored in directory: /tmp/pip-ephem-wheel-cache-txom75u6/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:07<227:46:14, 1.05kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:07<159:39:55, 1.50kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:08<111:47:59, 2.14kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:08<78:12:44, 3.06kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.64M/862M [00:08<54:38:32, 4.37kB/s].vector_cache/glove.6B.zip:   1%|          | 6.32M/862M [00:08<38:05:15, 6.24kB/s].vector_cache/glove.6B.zip:   1%|          | 10.6M/862M [00:08<26:31:42, 8.92kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.9M/862M [00:08<18:28:41, 12.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.3M/862M [00:08<12:52:11, 18.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.5M/862M [00:08<8:57:56, 26.0kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 27.8M/862M [00:08<6:14:44, 37.1kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.0M/862M [00:09<4:21:04, 53.0kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.3M/862M [00:09<3:01:54, 75.7kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.7M/862M [00:09<2:06:45, 108kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 45.0M/862M [00:09<1:28:22, 154kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.2M/862M [00:09<1:01:38, 220kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:09<43:37, 309kB/s]  .vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:09<30:32, 440kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:11<30:51, 435kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:11<25:58, 517kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:12<19:05, 703kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:12<13:40, 980kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:13<12:20, 1.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:13<10:49, 1.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:14<08:06, 1.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:15<08:00, 1.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:15<07:46, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:16<05:58, 2.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:16<04:18, 3.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:17<46:03, 287kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:17<34:20, 385kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:18<24:33, 538kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:19<19:26, 677kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:19<15:46, 834kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:20<11:33, 1.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:21<10:22, 1.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:21<11:33, 1.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:21<08:57, 1.46MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:22<06:40, 1.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:23<07:07, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:23<07:06, 1.83MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:23<05:29, 2.36MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:25<06:07, 2.11MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:25<08:14, 1.57MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:25<06:49, 1.90MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.8M/862M [00:26<05:01, 2.57MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:27<07:32, 1.71MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:27<07:18, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:27<05:37, 2.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:29<06:11, 2.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:29<06:26, 1.99MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:29<04:56, 2.59MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.0M/862M [00:30<03:37, 3.53MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:31<13:29, 944kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:31<11:31, 1.11MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:31<08:34, 1.48MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:33<08:13, 1.54MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:33<07:44, 1.64MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:33<05:50, 2.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:33<04:14, 2.97MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:35<12:13, 1.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:35<12:39, 996kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:35<09:52, 1.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:35<07:05, 1.77MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:37<08:43, 1.44MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:37<08:09, 1.54MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:37<06:13, 2.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:39<06:32, 1.91MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:39<06:33, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:39<04:59, 2.49MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:39<03:38, 3.41MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:41<16:24, 756kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:41<13:30, 918kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:41<09:56, 1.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:43<09:06, 1.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:43<08:18, 1.48MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:43<06:18, 1.95MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:45<06:33, 1.87MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:45<06:37, 1.85MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:45<05:07, 2.39MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:47<05:43, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:47<06:00, 2.03MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:47<04:41, 2.60MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:49<05:24, 2.24MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:49<05:42, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:49<04:28, 2.70MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:51<05:15, 2.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:51<07:36, 1.58MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:51<06:10, 1.95MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:51<04:37, 2.60MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:53<05:40, 2.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:53<05:57, 2.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:53<04:38, 2.57MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:55<05:20, 2.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:55<05:36, 2.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:55<04:24, 2.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:55<03:12, 3.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:57<33:50, 350kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:57<25:37, 462kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:57<18:23, 643kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:57<12:57, 909kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:59<49:38, 237kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:59<36:40, 321kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:59<26:07, 450kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:01<20:16, 577kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:01<16:06, 727kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:01<11:44, 996kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:03<10:14, 1.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:03<09:00, 1.29MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:03<06:45, 1.72MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:05<06:45, 1.71MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:05<06:36, 1.75MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:05<05:02, 2.29MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:05<03:38, 3.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:07<32:06, 358kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:07<24:20, 472kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:07<17:24, 659kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:07<12:21, 927kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:09<13:26, 850kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:09<11:14, 1.02MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:09<08:19, 1.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:09<05:56, 1.91MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:11<16:14, 700kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:11<13:13, 859kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:11<09:37, 1.18MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:11<06:53, 1.64MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:13<10:30, 1.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:13<09:11, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:13<06:52, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:15<06:47, 1.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:15<06:32, 1.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:15<04:57, 2.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:15<03:37, 3.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:16<08:36, 1.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:17<07:50, 1.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:17<05:52, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:17<04:16, 2.60MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:18<08:25, 1.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:19<07:42, 1.44MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:19<05:50, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:20<06:00, 1.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:21<05:36, 1.96MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:21<04:15, 2.58MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:22<05:10, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:23<07:01, 1.56MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:23<05:40, 1.93MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:23<04:16, 2.56MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:23<03:22, 3.23MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:24<05:29, 1.98MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:25<06:21, 1.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:25<05:00, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:25<03:37, 2.98MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:26<07:54, 1.37MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:27<08:04, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:27<06:13, 1.74MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:27<04:32, 2.37MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:28<06:15, 1.72MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:28<06:52, 1.56MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:29<05:25, 1.98MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:29<03:55, 2.72MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:30<11:11, 953kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:30<10:19, 1.03MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:31<07:44, 1.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:31<05:35, 1.90MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:32<07:01, 1.51MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:32<07:22, 1.44MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:33<05:46, 1.83MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:33<04:11, 2.52MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:34<11:32, 912kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:34<10:32, 1.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:35<07:59, 1.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:35<05:43, 1.83MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:36<12:33, 833kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:36<10:55, 958kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:37<08:10, 1.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:37<06:07, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:38<06:34, 1.58MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:38<08:15, 1.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:39<06:41, 1.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:39<04:51, 2.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:40<06:12, 1.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:40<07:42, 1.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:41<06:16, 1.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:41<04:35, 2.24MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:42<06:09, 1.66MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:42<07:41, 1.33MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:42<06:12, 1.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:43<04:30, 2.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:44<06:11, 1.64MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:44<07:40, 1.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:44<06:11, 1.64MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:45<04:29, 2.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:46<06:09, 1.64MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:46<07:26, 1.36MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:46<05:59, 1.69MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:47<04:21, 2.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:48<06:09, 1.63MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:48<07:24, 1.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:48<05:56, 1.69MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:49<04:19, 2.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:50<06:08, 1.62MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:50<07:24, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:50<05:48, 1.72MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:50<04:13, 2.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:51<03:09, 3.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:52<57:08, 173kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:52<42:52, 231kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:52<30:36, 323kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:52<21:33, 458kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:53<15:14, 646kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:54<1:18:47, 125kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:54<58:09, 169kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:54<41:18, 238kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:54<29:00, 338kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:56<23:26, 417kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:56<19:16, 507kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:56<14:04, 693kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:56<10:03, 968kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:57<07:09, 1.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:58<4:37:23, 35.0kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:58<3:16:56, 49.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:58<2:18:21, 70.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:58<1:36:43, 99.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:59<1:07:40, 142kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:00<5:17:35, 30.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:00<3:45:04, 42.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:00<2:37:58, 60.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [02:00<1:50:29, 86.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:02<1:19:26, 120kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:02<58:22, 164kB/s]  .vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:02<41:30, 230kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:02<29:07, 327kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:04<23:25, 405kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:04<19:08, 496kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:04<13:57, 679kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:04<09:55, 952kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:06<10:04, 936kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:06<09:45, 966kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:06<07:25, 1.27MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:06<05:19, 1.76MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:08<06:46, 1.38MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:08<07:25, 1.26MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:08<05:51, 1.60MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:08<04:16, 2.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:10<06:05, 1.52MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:10<06:47, 1.37MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:10<05:18, 1.75MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:10<03:53, 2.38MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:12<05:48, 1.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:12<06:55, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:12<05:31, 1.66MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:12<04:02, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:14<05:44, 1.59MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:14<07:21, 1.24MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:14<05:47, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:14<04:11, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:16<05:54, 1.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:16<06:44, 1.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:16<05:22, 1.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:16<03:53, 2.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:18<05:38, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:18<06:34, 1.37MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:18<05:15, 1.71MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:18<03:51, 2.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:20<05:34, 1.60MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:20<06:28, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:20<05:09, 1.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:20<03:46, 2.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:22<05:41, 1.56MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:22<06:24, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:22<05:06, 1.74MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:22<03:43, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:24<05:40, 1.55MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:24<06:31, 1.35MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:24<05:04, 1.73MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:24<03:41, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:26<05:03, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:26<06:04, 1.44MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:26<04:53, 1.78MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:26<03:33, 2.45MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:28<05:19, 1.63MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:28<06:14, 1.39MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:28<04:54, 1.76MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:28<03:35, 2.40MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:30<05:28, 1.57MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:30<06:11, 1.39MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:30<04:56, 1.74MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:30<03:36, 2.37MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:32<05:28, 1.56MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:32<06:19, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:32<05:02, 1.69MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:32<03:41, 2.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:34<05:19, 1.59MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:34<06:10, 1.37MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:34<04:55, 1.71MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:34<03:36, 2.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:36<05:20, 1.57MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:36<06:00, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:36<04:42, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:36<03:26, 2.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:38<05:14, 1.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:38<06:03, 1.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:38<04:46, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:38<03:28, 2.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:38<02:34, 3.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:40<1:34:36, 87.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:40<1:08:27, 120kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:40<48:19, 170kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:40<33:50, 242kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:42<25:57, 315kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:42<20:23, 401kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:42<14:43, 554kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:42<10:25, 780kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:44<10:17, 788kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:44<09:02, 897kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:44<06:51, 1.18MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:44<04:55, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:46<05:42, 1.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:46<06:04, 1.32MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:46<04:45, 1.69MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:46<03:27, 2.31MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:48<05:48, 1.37MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:48<06:07, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:48<04:43, 1.69MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:48<03:26, 2.30MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:50<05:19, 1.49MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:50<05:38, 1.40MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:50<04:25, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:50<03:13, 2.43MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:52<05:26, 1.44MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:52<05:42, 1.37MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:52<04:28, 1.75MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:52<03:15, 2.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:54<05:27, 1.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:54<05:55, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:54<04:36, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:54<03:19, 2.32MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:56<05:09, 1.49MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:56<05:24, 1.42MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:56<04:10, 1.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:56<03:04, 2.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:58<04:06, 1.85MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:58<04:38, 1.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:58<03:37, 2.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:58<02:37, 2.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:00<05:21, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:00<05:25, 1.39MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:00<04:12, 1.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:00<03:01, 2.47MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:02<07:10, 1.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:02<06:36, 1.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:02<05:00, 1.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:02<03:35, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:04<07:52, 942kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:04<07:05, 1.05MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:04<05:16, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:04<03:47, 1.95MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:05<05:53, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:06<05:36, 1.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:06<04:17, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:06<03:05, 2.36MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:07<17:02, 427kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:08<13:31, 538kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:08<09:52, 736kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:08<06:59, 1.03MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:09<09:20, 772kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:10<07:53, 914kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:10<05:51, 1.23MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:10<04:10, 1.71MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:11<17:00, 420kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:12<13:15, 538kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:12<09:33, 746kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:12<06:43, 1.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:13<19:05, 370kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:14<15:05, 469kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:14<10:54, 647kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:14<07:41, 912kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:15<08:10, 856kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:15<07:23, 946kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:16<05:34, 1.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:16<03:58, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:17<07:36, 910kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:17<06:56, 998kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:18<05:15, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:18<03:45, 1.83MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:19<07:29, 916kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:19<06:51, 1.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:20<05:07, 1.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:20<03:41, 1.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:21<04:44, 1.43MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:21<04:23, 1.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:22<03:19, 2.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:23<03:30, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:23<03:53, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:24<03:04, 2.18MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:24<02:12, 3.01MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:25<15:49, 420kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:25<11:55, 558kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:26<08:34, 774kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:26<06:02, 1.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:27<13:47, 478kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:27<10:56, 602kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:27<07:55, 830kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:28<05:38, 1.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:29<06:06, 1.07MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:29<05:36, 1.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:29<04:11, 1.55MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:30<03:00, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:31<06:20, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:31<05:42, 1.13MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:31<04:15, 1.51MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:32<03:03, 2.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:33<05:43, 1.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:33<05:16, 1.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:33<03:59, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:33<02:50, 2.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:35<10:11, 619kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:35<08:28, 745kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:35<06:10, 1.02MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:35<04:27, 1.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:37<04:41, 1.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:37<04:31, 1.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:37<03:26, 1.81MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:37<02:28, 2.51MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:39<06:16, 984kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:39<05:40, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:39<04:13, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:39<03:02, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:41<04:21, 1.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:41<04:20, 1.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:41<03:16, 1.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:41<02:25, 2.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:43<03:18, 1.82MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:43<03:29, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:43<02:44, 2.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:45<02:52, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:45<03:05, 1.94MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:45<02:46, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:45<02:04, 2.87MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:47<02:45, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:47<02:56, 2.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:47<02:25, 2.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:47<01:50, 3.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:49<02:40, 2.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:49<03:01, 1.93MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:49<02:23, 2.43MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:51<02:36, 2.21MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:51<02:59, 1.92MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:51<02:22, 2.42MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:53<02:34, 2.21MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:53<02:56, 1.93MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:53<02:20, 2.42MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:55<02:32, 2.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:55<02:55, 1.92MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:55<02:19, 2.41MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:57<02:31, 2.20MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:57<02:41, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:57<02:09, 2.56MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:57<01:37, 3.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:59<02:42, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:59<02:59, 1.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:59<02:21, 2.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:01<02:31, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:01<02:30, 2.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:01<01:55, 2.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:01<01:24, 3.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:03<08:20, 641kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:03<06:48, 785kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:03<04:59, 1.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:03<03:33, 1.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:05<04:45, 1.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:05<04:16, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:05<03:13, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:05<02:17, 2.28MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:06<18:33, 281kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:07<13:53, 375kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:07<09:54, 525kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:07<06:57, 742kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:08<07:43, 666kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:09<06:21, 809kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:09<04:37, 1.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:09<03:18, 1.54MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:10<04:15, 1.19MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:11<03:54, 1.30MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:11<02:56, 1.72MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:12<02:53, 1.73MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:13<02:40, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:13<02:01, 2.46MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:13<01:29, 3.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:14<03:55, 1.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:15<03:37, 1.36MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:15<02:44, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:16<02:44, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:16<02:46, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:17<02:06, 2.29MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:17<01:33, 3.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:18<02:57, 1.62MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:18<02:54, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:19<02:12, 2.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:19<01:36, 2.95MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:20<03:15, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:20<03:06, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:21<02:22, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:22<02:26, 1.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:22<02:31, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:23<01:57, 2.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:24<02:08, 2.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:24<02:18, 1.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:24<01:47, 2.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:25<01:17, 3.49MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:26<03:55, 1.16MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:26<03:30, 1.29MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:26<02:38, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:28<02:35, 1.72MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:28<03:22, 1.32MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:28<02:41, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:29<01:57, 2.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:30<02:32, 1.73MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:30<02:32, 1.72MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:30<01:55, 2.26MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:30<01:23, 3.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:32<03:46, 1.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:32<03:21, 1.28MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:32<02:31, 1.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:34<02:28, 1.71MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:34<02:28, 1.71MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:34<01:53, 2.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:34<01:22, 3.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:36<03:42, 1.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:36<03:18, 1.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:36<02:28, 1.68MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:38<02:25, 1.70MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:38<02:24, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:38<01:49, 2.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:38<01:19, 3.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:40<02:57, 1.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:40<02:58, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:40<02:18, 1.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:40<01:43, 2.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:42<02:00, 1.98MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:42<02:05, 1.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:42<01:38, 2.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:44<01:47, 2.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:44<01:56, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:44<01:29, 2.60MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:44<01:05, 3.53MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:46<02:54, 1.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:46<02:41, 1.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:46<02:00, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:46<01:26, 2.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:48<03:33, 1.06MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:48<03:08, 1.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:48<02:19, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:48<01:40, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:50<02:42, 1.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:50<02:32, 1.45MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:50<01:56, 1.90MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:52<01:57, 1.85MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:52<02:00, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:52<01:33, 2.32MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:54<01:41, 2.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:54<01:48, 1.97MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:54<01:24, 2.50MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:54<01:00, 3.45MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:56<11:27, 305kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:56<08:35, 406kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:56<06:07, 567kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:56<04:17, 800kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:58<04:55, 696kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:58<04:02, 847kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:58<02:56, 1.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:58<02:05, 1.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:00<03:19, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:00<02:55, 1.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:00<02:09, 1.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:00<01:32, 2.14MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:02<02:40, 1.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:02<02:26, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:02<01:50, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:04<01:49, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:04<01:49, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:04<01:23, 2.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:04<01:00, 3.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:06<03:09, 996kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:06<02:44, 1.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:06<02:02, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:08<01:56, 1.59MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:08<01:53, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:08<01:27, 2.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:10<01:31, 1.99MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:10<01:35, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:10<01:13, 2.43MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:11<01:21, 2.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:12<01:58, 1.49MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:12<01:37, 1.80MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:12<01:11, 2.45MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:13<01:36, 1.78MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:14<01:37, 1.76MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:14<01:14, 2.30MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:14<00:53, 3.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:15<02:40, 1.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:16<02:20, 1.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:16<01:44, 1.61MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:16<01:14, 2.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:17<02:36, 1.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:18<02:17, 1.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:18<01:43, 1.58MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:18<01:12, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:19<52:55, 50.4kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:19<37:27, 71.1kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:20<26:10, 101kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:21<18:25, 141kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:21<13:19, 195kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:22<09:23, 275kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:23<06:52, 368kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:23<05:14, 481kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:24<03:44, 670kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:24<02:36, 946kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:25<03:40, 671kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:25<02:58, 825kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:26<02:09, 1.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:26<01:30, 1.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:27<03:12, 745kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:27<02:39, 897kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:28<01:57, 1.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:29<01:44, 1.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:29<01:37, 1.43MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:29<01:13, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:31<01:13, 1.84MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:31<01:15, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:31<00:56, 2.36MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:32<00:41, 3.20MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:33<01:27, 1.51MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:33<01:23, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:33<01:04, 2.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:34<00:47, 2.72MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:35<01:30, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:35<02:05, 1.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:35<01:44, 1.22MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:36<01:16, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:36<00:55, 2.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:37<01:43, 1.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:37<02:13, 921kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:37<01:48, 1.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:37<01:19, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:38<00:56, 2.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:39<01:50, 1.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:39<02:15, 877kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:39<01:48, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:39<01:19, 1.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:40<00:57, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:41<01:55, 994kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:41<02:11, 869kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:41<01:45, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:41<01:16, 1.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:42<00:54, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:43<02:01, 910kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:43<02:18, 795kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:43<01:47, 1.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:43<01:17, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:43<00:57, 1.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:45<01:11, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:45<01:37, 1.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:45<01:19, 1.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:45<00:58, 1.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:46<00:42, 2.43MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:47<02:02, 835kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:47<02:10, 780kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:47<01:40, 1.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:47<01:12, 1.38MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:47<00:51, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:49<01:37, 1.00MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:49<01:51, 875kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:49<01:28, 1.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:49<01:04, 1.50MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:49<00:46, 2.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:51<01:57, 798kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:51<02:04, 756kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:51<01:36, 964kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:51<01:09, 1.32MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:51<00:49, 1.82MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:53<01:55, 776kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:53<02:00, 743kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:53<01:34, 949kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:53<01:07, 1.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:53<00:48, 1.79MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:55<01:58, 722kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:55<02:00, 707kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:55<01:33, 909kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:55<01:07, 1.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:55<00:47, 1.72MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:57<02:00, 674kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:57<02:04, 655kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:57<01:35, 848kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:57<01:08, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:57<00:47, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:59<01:46, 726kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:59<01:48, 710kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:59<01:24, 911kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:59<01:00, 1.26MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:59<00:42, 1.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:01<01:07, 1.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:01<01:19, 918kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:01<01:03, 1.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:01<00:45, 1.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:01<00:32, 2.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:03<01:26, 794kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:03<01:31, 755kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:03<01:11, 960kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:03<00:51, 1.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:03<00:36, 1.81MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:05<01:29, 723kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:05<01:31, 709kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:05<01:10, 909kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:05<00:50, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:05<00:36, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:07<00:47, 1.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:07<01:10, 858kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:07<00:58, 1.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:07<00:42, 1.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:07<00:30, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:09<00:38, 1.46MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:09<01:06, 850kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:09<00:53, 1.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:09<00:39, 1.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:09<00:28, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:09<00:21, 2.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:11<00:51, 1.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:11<00:51, 1.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:11<00:39, 1.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:11<00:27, 1.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:11<00:20, 2.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:13<00:57, 842kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:13<01:06, 728kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:13<00:52, 908kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:13<00:37, 1.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:13<00:26, 1.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:15<00:35, 1.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:15<00:48, 904kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:15<00:40, 1.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:15<00:28, 1.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:15<00:20, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:15<00:15, 2.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:17<01:02, 639kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:17<00:54, 731kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:17<00:39, 986kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:17<00:27, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:17<00:20, 1.80MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:19<00:28, 1.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:19<00:39, 912kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:19<00:31, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:19<00:22, 1.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:19<00:15, 2.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:19<00:11, 2.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:21<01:17, 409kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:21<01:11, 444kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:21<00:52, 593kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:21<00:36, 822kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:21<00:25, 1.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:21<00:17, 1.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:22<01:19, 347kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:23<01:10, 390kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:23<00:51, 524kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:23<00:35, 730kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:23<00:24, 1.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:23<00:16, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:24<00:54, 431kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:25<00:49, 473kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:25<00:37, 621kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:25<00:25, 864kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:25<00:17, 1.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:25<00:12, 1.63MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:26<00:56, 340kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:27<00:49, 393kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:27<00:36, 522kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:27<00:24, 730kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:27<00:16, 1.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:27<00:10, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:28<00:39, 381kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:29<00:36, 420kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:29<00:26, 556kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:29<00:17, 776kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:29<00:11, 1.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:29<00:07, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:30<00:33, 331kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:31<00:29, 375kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:31<00:21, 502kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:31<00:13, 699kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:31<00:08, 974kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:32<00:07, 869kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:32<00:09, 738kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:33<00:07, 927kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:33<00:04, 1.26MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:33<00:02, 1.72MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:34<00:02, 1.18MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:34<00:02, 1.18MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:35<00:01, 1.52MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:35<00:00, 2.05MB/s].vector_cache/glove.6B.zip: 862MB [06:35, 2.18MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 545/400000 [00:00<01:13, 5445.67it/s]  0%|          | 1197/400000 [00:00<01:09, 5727.72it/s]  0%|          | 1894/400000 [00:00<01:05, 6050.01it/s]  1%|          | 2570/400000 [00:00<01:03, 6245.73it/s]  1%|          | 3180/400000 [00:00<01:04, 6200.13it/s]  1%|          | 3870/400000 [00:00<01:01, 6393.43it/s]  1%|          | 4585/400000 [00:00<00:59, 6602.01it/s]  1%|▏         | 5294/400000 [00:00<00:58, 6739.82it/s]  2%|▏         | 6006/400000 [00:00<00:57, 6849.05it/s]  2%|▏         | 6715/400000 [00:01<00:56, 6917.32it/s]  2%|▏         | 7398/400000 [00:01<00:56, 6889.07it/s]  2%|▏         | 8078/400000 [00:01<00:57, 6807.81it/s]  2%|▏         | 8758/400000 [00:01<00:57, 6805.45it/s]  2%|▏         | 9435/400000 [00:01<00:58, 6625.91it/s]  3%|▎         | 10133/400000 [00:01<00:57, 6725.95it/s]  3%|▎         | 10805/400000 [00:01<00:59, 6518.77it/s]  3%|▎         | 11458/400000 [00:01<01:01, 6341.38it/s]  3%|▎         | 12159/400000 [00:01<00:59, 6527.14it/s]  3%|▎         | 12855/400000 [00:01<00:58, 6650.29it/s]  3%|▎         | 13550/400000 [00:02<00:57, 6736.51it/s]  4%|▎         | 14239/400000 [00:02<00:56, 6779.14it/s]  4%|▎         | 14919/400000 [00:02<00:57, 6663.83it/s]  4%|▍         | 15587/400000 [00:02<00:58, 6564.60it/s]  4%|▍         | 16245/400000 [00:02<00:58, 6511.12it/s]  4%|▍         | 16931/400000 [00:02<00:57, 6609.28it/s]  4%|▍         | 17614/400000 [00:02<00:57, 6670.05it/s]  5%|▍         | 18294/400000 [00:02<00:56, 6708.13it/s]  5%|▍         | 18993/400000 [00:02<00:56, 6788.75it/s]  5%|▍         | 19673/400000 [00:02<00:56, 6728.84it/s]  5%|▌         | 20347/400000 [00:03<00:57, 6621.80it/s]  5%|▌         | 21010/400000 [00:03<00:57, 6587.72it/s]  5%|▌         | 21670/400000 [00:03<00:57, 6554.66it/s]  6%|▌         | 22326/400000 [00:03<00:57, 6537.96it/s]  6%|▌         | 22981/400000 [00:03<00:58, 6493.41it/s]  6%|▌         | 23643/400000 [00:03<00:57, 6529.38it/s]  6%|▌         | 24297/400000 [00:03<00:57, 6493.93it/s]  6%|▌         | 24968/400000 [00:03<00:57, 6555.01it/s]  6%|▋         | 25676/400000 [00:03<00:55, 6701.83it/s]  7%|▋         | 26376/400000 [00:03<00:55, 6786.12it/s]  7%|▋         | 27076/400000 [00:04<00:54, 6846.83it/s]  7%|▋         | 27771/400000 [00:04<00:54, 6876.39it/s]  7%|▋         | 28460/400000 [00:04<00:54, 6779.99it/s]  7%|▋         | 29139/400000 [00:04<00:54, 6749.77it/s]  7%|▋         | 29815/400000 [00:04<00:56, 6589.98it/s]  8%|▊         | 30526/400000 [00:04<00:54, 6736.61it/s]  8%|▊         | 31220/400000 [00:04<00:54, 6796.11it/s]  8%|▊         | 31902/400000 [00:04<00:54, 6803.21it/s]  8%|▊         | 32601/400000 [00:04<00:53, 6857.06it/s]  8%|▊         | 33290/400000 [00:04<00:53, 6866.73it/s]  8%|▊         | 33979/400000 [00:05<00:53, 6870.94it/s]  9%|▊         | 34686/400000 [00:05<00:52, 6926.77it/s]  9%|▉         | 35380/400000 [00:05<00:52, 6908.01it/s]  9%|▉         | 36072/400000 [00:05<00:52, 6889.38it/s]  9%|▉         | 36778/400000 [00:05<00:52, 6937.19it/s]  9%|▉         | 37472/400000 [00:05<00:52, 6903.92it/s] 10%|▉         | 38163/400000 [00:05<00:52, 6875.58it/s] 10%|▉         | 38871/400000 [00:05<00:52, 6933.06it/s] 10%|▉         | 39585/400000 [00:05<00:51, 6992.93it/s] 10%|█         | 40285/400000 [00:05<00:52, 6819.42it/s] 10%|█         | 40969/400000 [00:06<00:53, 6700.43it/s] 10%|█         | 41641/400000 [00:06<00:53, 6687.52it/s] 11%|█         | 42326/400000 [00:06<00:53, 6734.36it/s] 11%|█         | 43023/400000 [00:06<00:52, 6801.56it/s] 11%|█         | 43717/400000 [00:06<00:52, 6841.21it/s] 11%|█         | 44418/400000 [00:06<00:51, 6888.18it/s] 11%|█▏        | 45134/400000 [00:06<00:50, 6966.78it/s] 11%|█▏        | 45834/400000 [00:06<00:50, 6975.95it/s] 12%|█▏        | 46532/400000 [00:06<00:52, 6763.85it/s] 12%|█▏        | 47211/400000 [00:07<00:52, 6701.53it/s] 12%|█▏        | 47883/400000 [00:07<00:53, 6641.88it/s] 12%|█▏        | 48549/400000 [00:07<00:53, 6598.42it/s] 12%|█▏        | 49214/400000 [00:07<00:53, 6611.69it/s] 12%|█▏        | 49881/400000 [00:07<00:52, 6625.30it/s] 13%|█▎        | 50556/400000 [00:07<00:52, 6660.54it/s] 13%|█▎        | 51263/400000 [00:07<00:51, 6776.80it/s] 13%|█▎        | 51962/400000 [00:07<00:50, 6838.22it/s] 13%|█▎        | 52665/400000 [00:07<00:50, 6893.30it/s] 13%|█▎        | 53355/400000 [00:07<00:50, 6801.62it/s] 14%|█▎        | 54036/400000 [00:08<00:51, 6721.22it/s] 14%|█▎        | 54709/400000 [00:08<00:52, 6637.13it/s] 14%|█▍        | 55374/400000 [00:08<00:52, 6585.46it/s] 14%|█▍        | 56050/400000 [00:08<00:51, 6634.56it/s] 14%|█▍        | 56743/400000 [00:08<00:51, 6719.33it/s] 14%|█▍        | 57431/400000 [00:08<00:50, 6765.59it/s] 15%|█▍        | 58129/400000 [00:08<00:50, 6828.26it/s] 15%|█▍        | 58842/400000 [00:08<00:49, 6913.60it/s] 15%|█▍        | 59542/400000 [00:08<00:49, 6938.99it/s] 15%|█▌        | 60237/400000 [00:08<00:49, 6866.69it/s] 15%|█▌        | 60937/400000 [00:09<00:49, 6903.59it/s] 15%|█▌        | 61651/400000 [00:09<00:48, 6971.10it/s] 16%|█▌        | 62351/400000 [00:09<00:48, 6976.55it/s] 16%|█▌        | 63049/400000 [00:09<00:48, 6918.86it/s] 16%|█▌        | 63742/400000 [00:09<00:49, 6849.49it/s] 16%|█▌        | 64428/400000 [00:09<00:49, 6800.03it/s] 16%|█▋        | 65110/400000 [00:09<00:49, 6805.05it/s] 16%|█▋        | 65803/400000 [00:09<00:48, 6841.40it/s] 17%|█▋        | 66498/400000 [00:09<00:48, 6871.70it/s] 17%|█▋        | 67188/400000 [00:09<00:48, 6878.01it/s] 17%|█▋        | 67876/400000 [00:10<00:48, 6861.29it/s] 17%|█▋        | 68588/400000 [00:10<00:47, 6934.96it/s] 17%|█▋        | 69294/400000 [00:10<00:47, 6969.85it/s] 18%|█▊        | 70008/400000 [00:10<00:47, 7018.89it/s] 18%|█▊        | 70713/400000 [00:10<00:46, 7028.17it/s] 18%|█▊        | 71420/400000 [00:10<00:46, 7038.35it/s] 18%|█▊        | 72124/400000 [00:10<00:47, 6969.52it/s] 18%|█▊        | 72825/400000 [00:10<00:46, 6978.90it/s] 18%|█▊        | 73524/400000 [00:10<00:46, 6979.33it/s] 19%|█▊        | 74223/400000 [00:10<00:46, 6944.41it/s] 19%|█▊        | 74918/400000 [00:11<00:47, 6894.43it/s] 19%|█▉        | 75608/400000 [00:11<00:47, 6804.68it/s] 19%|█▉        | 76307/400000 [00:11<00:47, 6859.20it/s] 19%|█▉        | 76994/400000 [00:11<00:47, 6844.99it/s] 19%|█▉        | 77705/400000 [00:11<00:46, 6921.20it/s] 20%|█▉        | 78403/400000 [00:11<00:46, 6936.38it/s] 20%|█▉        | 79097/400000 [00:11<00:46, 6900.03it/s] 20%|█▉        | 79789/400000 [00:11<00:46, 6884.76it/s] 20%|██        | 80486/400000 [00:11<00:46, 6909.75it/s] 20%|██        | 81178/400000 [00:11<00:46, 6898.75it/s] 20%|██        | 81875/400000 [00:12<00:45, 6917.78it/s] 21%|██        | 82580/400000 [00:12<00:45, 6955.05it/s] 21%|██        | 83276/400000 [00:12<00:45, 6944.64it/s] 21%|██        | 83991/400000 [00:12<00:45, 7004.60it/s] 21%|██        | 84692/400000 [00:12<00:45, 7003.00it/s] 21%|██▏       | 85393/400000 [00:12<00:45, 6885.41it/s] 22%|██▏       | 86084/400000 [00:12<00:45, 6890.14it/s] 22%|██▏       | 86782/400000 [00:12<00:45, 6915.46it/s] 22%|██▏       | 87482/400000 [00:12<00:45, 6939.28it/s] 22%|██▏       | 88177/400000 [00:12<00:44, 6939.41it/s] 22%|██▏       | 88886/400000 [00:13<00:44, 6983.41it/s] 22%|██▏       | 89585/400000 [00:13<00:45, 6888.76it/s] 23%|██▎       | 90296/400000 [00:13<00:44, 6953.23it/s] 23%|██▎       | 90992/400000 [00:13<00:44, 6921.32it/s] 23%|██▎       | 91690/400000 [00:13<00:44, 6936.12it/s] 23%|██▎       | 92397/400000 [00:13<00:44, 6972.33it/s] 23%|██▎       | 93095/400000 [00:13<00:44, 6933.97it/s] 23%|██▎       | 93791/400000 [00:13<00:44, 6939.57it/s] 24%|██▎       | 94505/400000 [00:13<00:43, 6996.28it/s] 24%|██▍       | 95205/400000 [00:13<00:44, 6912.16it/s] 24%|██▍       | 95897/400000 [00:14<00:44, 6867.37it/s] 24%|██▍       | 96585/400000 [00:14<00:44, 6867.97it/s] 24%|██▍       | 97273/400000 [00:14<00:44, 6848.01it/s] 24%|██▍       | 97971/400000 [00:14<00:43, 6884.73it/s] 25%|██▍       | 98660/400000 [00:14<00:44, 6737.58it/s] 25%|██▍       | 99360/400000 [00:14<00:44, 6813.94it/s] 25%|██▌       | 100058/400000 [00:14<00:43, 6861.68it/s] 25%|██▌       | 100777/400000 [00:14<00:43, 6956.36it/s] 25%|██▌       | 101493/400000 [00:14<00:42, 7014.02it/s] 26%|██▌       | 102209/400000 [00:15<00:42, 7054.53it/s] 26%|██▌       | 102915/400000 [00:15<00:42, 7049.31it/s] 26%|██▌       | 103621/400000 [00:15<00:42, 7043.13it/s] 26%|██▌       | 104326/400000 [00:15<00:42, 7039.01it/s] 26%|██▋       | 105031/400000 [00:15<00:42, 6975.76it/s] 26%|██▋       | 105734/400000 [00:15<00:42, 6990.67it/s] 27%|██▋       | 106442/400000 [00:15<00:41, 7016.05it/s] 27%|██▋       | 107144/400000 [00:15<00:42, 6946.26it/s] 27%|██▋       | 107850/400000 [00:15<00:41, 6979.12it/s] 27%|██▋       | 108560/400000 [00:15<00:41, 7014.34it/s] 27%|██▋       | 109266/400000 [00:16<00:41, 7025.77it/s] 27%|██▋       | 109975/400000 [00:16<00:41, 7044.60it/s] 28%|██▊       | 110680/400000 [00:16<00:41, 7026.34it/s] 28%|██▊       | 111383/400000 [00:16<00:43, 6692.88it/s] 28%|██▊       | 112056/400000 [00:16<00:43, 6636.61it/s] 28%|██▊       | 112725/400000 [00:16<00:43, 6649.70it/s] 28%|██▊       | 113392/400000 [00:16<00:43, 6627.91it/s] 29%|██▊       | 114057/400000 [00:16<00:43, 6600.39it/s] 29%|██▊       | 114718/400000 [00:16<00:43, 6557.41it/s] 29%|██▉       | 115375/400000 [00:16<00:43, 6525.59it/s] 29%|██▉       | 116029/400000 [00:17<00:44, 6420.65it/s] 29%|██▉       | 116721/400000 [00:17<00:43, 6561.62it/s] 29%|██▉       | 117404/400000 [00:17<00:42, 6638.75it/s] 30%|██▉       | 118069/400000 [00:17<00:42, 6564.40it/s] 30%|██▉       | 118759/400000 [00:17<00:42, 6661.08it/s] 30%|██▉       | 119469/400000 [00:17<00:41, 6786.46it/s] 30%|███       | 120162/400000 [00:17<00:40, 6828.73it/s] 30%|███       | 120860/400000 [00:17<00:40, 6871.10it/s] 30%|███       | 121548/400000 [00:17<00:41, 6738.46it/s] 31%|███       | 122232/400000 [00:17<00:41, 6765.26it/s] 31%|███       | 122910/400000 [00:18<00:41, 6648.74it/s] 31%|███       | 123603/400000 [00:18<00:41, 6728.25it/s] 31%|███       | 124285/400000 [00:18<00:40, 6753.90it/s] 31%|███       | 124992/400000 [00:18<00:40, 6844.50it/s] 31%|███▏      | 125692/400000 [00:18<00:39, 6888.26it/s] 32%|███▏      | 126388/400000 [00:18<00:39, 6908.40it/s] 32%|███▏      | 127092/400000 [00:18<00:39, 6945.82it/s] 32%|███▏      | 127788/400000 [00:18<00:39, 6948.59it/s] 32%|███▏      | 128488/400000 [00:18<00:38, 6962.77it/s] 32%|███▏      | 129185/400000 [00:18<00:38, 6959.58it/s] 32%|███▏      | 129882/400000 [00:19<00:39, 6870.42it/s] 33%|███▎      | 130589/400000 [00:19<00:38, 6928.74it/s] 33%|███▎      | 131286/400000 [00:19<00:38, 6938.67it/s] 33%|███▎      | 131990/400000 [00:19<00:38, 6967.64it/s] 33%|███▎      | 132687/400000 [00:19<00:38, 6941.98it/s] 33%|███▎      | 133382/400000 [00:19<00:38, 6943.56it/s] 34%|███▎      | 134083/400000 [00:19<00:38, 6960.29it/s] 34%|███▎      | 134780/400000 [00:19<00:38, 6913.37it/s] 34%|███▍      | 135491/400000 [00:19<00:37, 6970.30it/s] 34%|███▍      | 136189/400000 [00:19<00:38, 6856.87it/s] 34%|███▍      | 136876/400000 [00:20<00:38, 6859.16it/s] 34%|███▍      | 137580/400000 [00:20<00:37, 6912.32it/s] 35%|███▍      | 138277/400000 [00:20<00:37, 6927.69it/s] 35%|███▍      | 138972/400000 [00:20<00:37, 6933.68it/s] 35%|███▍      | 139686/400000 [00:20<00:37, 6993.97it/s] 35%|███▌      | 140386/400000 [00:20<00:37, 6863.99it/s] 35%|███▌      | 141074/400000 [00:20<00:38, 6807.61it/s] 35%|███▌      | 141756/400000 [00:20<00:38, 6665.59it/s] 36%|███▌      | 142424/400000 [00:20<00:38, 6634.29it/s] 36%|███▌      | 143089/400000 [00:20<00:38, 6635.28it/s] 36%|███▌      | 143757/400000 [00:21<00:38, 6647.22it/s] 36%|███▌      | 144423/400000 [00:21<00:38, 6574.57it/s] 36%|███▋      | 145107/400000 [00:21<00:38, 6648.27it/s] 36%|███▋      | 145775/400000 [00:21<00:38, 6655.32it/s] 37%|███▋      | 146469/400000 [00:21<00:37, 6737.06it/s] 37%|███▋      | 147176/400000 [00:21<00:37, 6831.72it/s] 37%|███▋      | 147883/400000 [00:21<00:36, 6900.76it/s] 37%|███▋      | 148581/400000 [00:21<00:36, 6921.79it/s] 37%|███▋      | 149288/400000 [00:21<00:35, 6964.95it/s] 37%|███▋      | 149985/400000 [00:22<00:36, 6887.58it/s] 38%|███▊      | 150691/400000 [00:22<00:35, 6937.90it/s] 38%|███▊      | 151408/400000 [00:22<00:35, 7004.68it/s] 38%|███▊      | 152111/400000 [00:22<00:35, 7012.08it/s] 38%|███▊      | 152813/400000 [00:22<00:35, 6971.57it/s] 38%|███▊      | 153511/400000 [00:22<00:35, 6878.09it/s] 39%|███▊      | 154200/400000 [00:22<00:36, 6750.66it/s] 39%|███▊      | 154876/400000 [00:22<00:36, 6702.61it/s] 39%|███▉      | 155547/400000 [00:22<00:36, 6635.10it/s] 39%|███▉      | 156212/400000 [00:22<00:37, 6519.60it/s] 39%|███▉      | 156865/400000 [00:23<00:38, 6391.51it/s] 39%|███▉      | 157509/400000 [00:23<00:37, 6404.94it/s] 40%|███▉      | 158171/400000 [00:23<00:37, 6466.35it/s] 40%|███▉      | 158819/400000 [00:23<00:37, 6419.02it/s] 40%|███▉      | 159469/400000 [00:23<00:37, 6442.46it/s] 40%|████      | 160173/400000 [00:23<00:36, 6609.82it/s] 40%|████      | 160883/400000 [00:23<00:35, 6749.37it/s] 40%|████      | 161589/400000 [00:23<00:34, 6838.61it/s] 41%|████      | 162290/400000 [00:23<00:34, 6886.85it/s] 41%|████      | 162982/400000 [00:23<00:34, 6896.14it/s] 41%|████      | 163686/400000 [00:24<00:34, 6938.37it/s] 41%|████      | 164389/400000 [00:24<00:33, 6962.27it/s] 41%|████▏     | 165098/400000 [00:24<00:33, 6998.66it/s] 41%|████▏     | 165799/400000 [00:24<00:33, 6974.64it/s] 42%|████▏     | 166497/400000 [00:24<00:33, 6944.80it/s] 42%|████▏     | 167192/400000 [00:24<00:33, 6927.31it/s] 42%|████▏     | 167885/400000 [00:24<00:33, 6907.33it/s] 42%|████▏     | 168576/400000 [00:24<00:33, 6891.67it/s] 42%|████▏     | 169277/400000 [00:24<00:33, 6924.82it/s] 42%|████▏     | 169979/400000 [00:24<00:33, 6951.31it/s] 43%|████▎     | 170693/400000 [00:25<00:32, 7006.83it/s] 43%|████▎     | 171394/400000 [00:25<00:32, 6952.37it/s] 43%|████▎     | 172110/400000 [00:25<00:32, 7010.74it/s] 43%|████▎     | 172818/400000 [00:25<00:32, 7030.34it/s] 43%|████▎     | 173524/400000 [00:25<00:32, 7037.94it/s] 44%|████▎     | 174228/400000 [00:25<00:32, 6986.32it/s] 44%|████▎     | 174927/400000 [00:25<00:32, 6879.09it/s] 44%|████▍     | 175616/400000 [00:25<00:33, 6723.54it/s] 44%|████▍     | 176290/400000 [00:25<00:33, 6670.65it/s] 44%|████▍     | 176958/400000 [00:25<00:33, 6656.82it/s] 44%|████▍     | 177625/400000 [00:26<00:33, 6639.94it/s] 45%|████▍     | 178290/400000 [00:26<00:33, 6616.51it/s] 45%|████▍     | 178952/400000 [00:26<00:33, 6605.73it/s] 45%|████▍     | 179613/400000 [00:26<00:35, 6222.84it/s] 45%|████▌     | 180241/400000 [00:26<00:35, 6218.70it/s] 45%|████▌     | 180889/400000 [00:26<00:34, 6292.36it/s] 45%|████▌     | 181550/400000 [00:26<00:34, 6384.35it/s] 46%|████▌     | 182212/400000 [00:26<00:33, 6451.78it/s] 46%|████▌     | 182871/400000 [00:26<00:33, 6490.24it/s] 46%|████▌     | 183522/400000 [00:26<00:33, 6434.42it/s] 46%|████▌     | 184167/400000 [00:27<00:33, 6375.39it/s] 46%|████▌     | 184820/400000 [00:27<00:33, 6418.89it/s] 46%|████▋     | 185486/400000 [00:27<00:33, 6488.60it/s] 47%|████▋     | 186136/400000 [00:27<00:33, 6472.08it/s] 47%|████▋     | 186787/400000 [00:27<00:32, 6482.91it/s] 47%|████▋     | 187446/400000 [00:27<00:32, 6513.29it/s] 47%|████▋     | 188117/400000 [00:27<00:32, 6568.88it/s] 47%|████▋     | 188775/400000 [00:27<00:32, 6501.90it/s] 47%|████▋     | 189472/400000 [00:27<00:31, 6633.80it/s] 48%|████▊     | 190137/400000 [00:28<00:31, 6572.23it/s] 48%|████▊     | 190795/400000 [00:28<00:31, 6566.97it/s] 48%|████▊     | 191453/400000 [00:28<00:32, 6459.12it/s] 48%|████▊     | 192119/400000 [00:28<00:31, 6517.16it/s] 48%|████▊     | 192772/400000 [00:28<00:32, 6434.61it/s] 48%|████▊     | 193452/400000 [00:28<00:31, 6537.95it/s] 49%|████▊     | 194151/400000 [00:28<00:30, 6666.08it/s] 49%|████▊     | 194819/400000 [00:28<00:31, 6505.80it/s] 49%|████▉     | 195479/400000 [00:28<00:31, 6532.58it/s] 49%|████▉     | 196136/400000 [00:28<00:31, 6540.82it/s] 49%|████▉     | 196794/400000 [00:29<00:31, 6552.14it/s] 49%|████▉     | 197477/400000 [00:29<00:30, 6631.39it/s] 50%|████▉     | 198167/400000 [00:29<00:30, 6706.67it/s] 50%|████▉     | 198839/400000 [00:29<00:31, 6407.75it/s] 50%|████▉     | 199497/400000 [00:29<00:31, 6457.53it/s] 50%|█████     | 200159/400000 [00:29<00:30, 6502.76it/s] 50%|█████     | 200863/400000 [00:29<00:29, 6652.98it/s] 50%|█████     | 201547/400000 [00:29<00:29, 6707.32it/s] 51%|█████     | 202227/400000 [00:29<00:29, 6733.98it/s] 51%|█████     | 202922/400000 [00:29<00:28, 6796.14it/s] 51%|█████     | 203622/400000 [00:30<00:28, 6853.61it/s] 51%|█████     | 204313/400000 [00:30<00:28, 6870.23it/s] 51%|█████▏    | 205030/400000 [00:30<00:28, 6956.32it/s] 51%|█████▏    | 205730/400000 [00:30<00:27, 6969.22it/s] 52%|█████▏    | 206433/400000 [00:30<00:27, 6987.30it/s] 52%|█████▏    | 207152/400000 [00:30<00:27, 7043.59it/s] 52%|█████▏    | 207864/400000 [00:30<00:27, 7063.34it/s] 52%|█████▏    | 208583/400000 [00:30<00:26, 7100.45it/s] 52%|█████▏    | 209294/400000 [00:30<00:26, 7088.01it/s] 53%|█████▎    | 210003/400000 [00:30<00:27, 6978.57it/s] 53%|█████▎    | 210713/400000 [00:31<00:26, 7012.68it/s] 53%|█████▎    | 211420/400000 [00:31<00:26, 7028.12it/s] 53%|█████▎    | 212124/400000 [00:31<00:26, 6992.89it/s] 53%|█████▎    | 212824/400000 [00:31<00:27, 6868.51it/s] 53%|█████▎    | 213512/400000 [00:31<00:27, 6671.73it/s] 54%|█████▎    | 214181/400000 [00:31<00:29, 6392.37it/s] 54%|█████▎    | 214828/400000 [00:31<00:28, 6413.08it/s] 54%|█████▍    | 215479/400000 [00:31<00:28, 6441.21it/s] 54%|█████▍    | 216131/400000 [00:31<00:28, 6464.27it/s] 54%|█████▍    | 216779/400000 [00:31<00:28, 6459.99it/s] 54%|█████▍    | 217442/400000 [00:32<00:28, 6509.11it/s] 55%|█████▍    | 218099/400000 [00:32<00:27, 6527.07it/s] 55%|█████▍    | 218753/400000 [00:32<00:28, 6452.49it/s] 55%|█████▍    | 219399/400000 [00:32<00:28, 6365.98it/s] 55%|█████▌    | 220101/400000 [00:32<00:27, 6548.41it/s] 55%|█████▌    | 220808/400000 [00:32<00:26, 6696.33it/s] 55%|█████▌    | 221522/400000 [00:32<00:26, 6823.36it/s] 56%|█████▌    | 222234/400000 [00:32<00:25, 6909.31it/s] 56%|█████▌    | 222927/400000 [00:32<00:25, 6899.94it/s] 56%|█████▌    | 223624/400000 [00:32<00:25, 6919.10it/s] 56%|█████▌    | 224338/400000 [00:33<00:25, 6982.87it/s] 56%|█████▋    | 225038/400000 [00:33<00:25, 6981.14it/s] 56%|█████▋    | 225737/400000 [00:33<00:24, 6982.65it/s] 57%|█████▋    | 226446/400000 [00:33<00:24, 7012.97it/s] 57%|█████▋    | 227148/400000 [00:33<00:24, 6935.64it/s] 57%|█████▋    | 227856/400000 [00:33<00:24, 6977.14it/s] 57%|█████▋    | 228563/400000 [00:33<00:24, 7004.55it/s] 57%|█████▋    | 229273/400000 [00:33<00:24, 7031.61it/s] 57%|█████▋    | 229980/400000 [00:33<00:24, 7040.55it/s] 58%|█████▊    | 230685/400000 [00:33<00:24, 7036.93it/s] 58%|█████▊    | 231389/400000 [00:34<00:24, 7013.94it/s] 58%|█████▊    | 232103/400000 [00:34<00:23, 7049.74it/s] 58%|█████▊    | 232814/400000 [00:34<00:23, 7067.28it/s] 58%|█████▊    | 233521/400000 [00:34<00:23, 7050.54it/s] 59%|█████▊    | 234227/400000 [00:34<00:23, 7030.55it/s] 59%|█████▊    | 234931/400000 [00:34<00:23, 7022.17it/s] 59%|█████▉    | 235634/400000 [00:34<00:23, 6985.94it/s] 59%|█████▉    | 236333/400000 [00:34<00:23, 6973.69it/s] 59%|█████▉    | 237042/400000 [00:34<00:23, 7006.58it/s] 59%|█████▉    | 237743/400000 [00:35<00:23, 6997.50it/s] 60%|█████▉    | 238447/400000 [00:35<00:23, 7009.24it/s] 60%|█████▉    | 239157/400000 [00:35<00:22, 7033.35it/s] 60%|█████▉    | 239861/400000 [00:35<00:22, 7007.61it/s] 60%|██████    | 240562/400000 [00:35<00:22, 7000.07it/s] 60%|██████    | 241263/400000 [00:35<00:22, 6992.02it/s] 60%|██████    | 241969/400000 [00:35<00:22, 7011.68it/s] 61%|██████    | 242673/400000 [00:35<00:22, 7019.96it/s] 61%|██████    | 243390/400000 [00:35<00:22, 7063.67it/s] 61%|██████    | 244097/400000 [00:35<00:22, 6997.90it/s] 61%|██████    | 244811/400000 [00:36<00:22, 7037.72it/s] 61%|██████▏   | 245519/400000 [00:36<00:21, 7050.10it/s] 62%|██████▏   | 246225/400000 [00:36<00:21, 7045.78it/s] 62%|██████▏   | 246937/400000 [00:36<00:21, 7065.49it/s] 62%|██████▏   | 247650/400000 [00:36<00:21, 7083.13it/s] 62%|██████▏   | 248359/400000 [00:36<00:22, 6828.09it/s] 62%|██████▏   | 249044/400000 [00:36<00:22, 6708.84it/s] 62%|██████▏   | 249756/400000 [00:36<00:22, 6824.56it/s] 63%|██████▎   | 250468/400000 [00:36<00:21, 6909.61it/s] 63%|██████▎   | 251174/400000 [00:36<00:21, 6951.92it/s] 63%|██████▎   | 251871/400000 [00:37<00:21, 6949.92it/s] 63%|██████▎   | 252567/400000 [00:37<00:21, 6773.58it/s] 63%|██████▎   | 253246/400000 [00:37<00:21, 6734.94it/s] 63%|██████▎   | 253921/400000 [00:37<00:21, 6695.16it/s] 64%|██████▎   | 254592/400000 [00:37<00:21, 6674.50it/s] 64%|██████▍   | 255294/400000 [00:37<00:21, 6773.82it/s] 64%|██████▍   | 255998/400000 [00:37<00:21, 6850.01it/s] 64%|██████▍   | 256720/400000 [00:37<00:20, 6955.54it/s] 64%|██████▍   | 257431/400000 [00:37<00:20, 6998.25it/s] 65%|██████▍   | 258132/400000 [00:37<00:20, 6969.81it/s] 65%|██████▍   | 258840/400000 [00:38<00:20, 7002.15it/s] 65%|██████▍   | 259544/400000 [00:38<00:20, 7012.09it/s] 65%|██████▌   | 260246/400000 [00:38<00:20, 6977.41it/s] 65%|██████▌   | 260944/400000 [00:38<00:20, 6872.01it/s] 65%|██████▌   | 261632/400000 [00:38<00:20, 6838.85it/s] 66%|██████▌   | 262328/400000 [00:38<00:20, 6873.94it/s] 66%|██████▌   | 263031/400000 [00:38<00:19, 6917.86it/s] 66%|██████▌   | 263749/400000 [00:38<00:19, 6993.09it/s] 66%|██████▌   | 264468/400000 [00:38<00:19, 7049.40it/s] 66%|██████▋   | 265179/400000 [00:38<00:19, 7064.70it/s] 66%|██████▋   | 265888/400000 [00:39<00:18, 7071.08it/s] 67%|██████▋   | 266605/400000 [00:39<00:18, 7098.12it/s] 67%|██████▋   | 267315/400000 [00:39<00:18, 7084.14it/s] 67%|██████▋   | 268034/400000 [00:39<00:18, 7113.16it/s] 67%|██████▋   | 268749/400000 [00:39<00:18, 7123.12it/s] 67%|██████▋   | 269470/400000 [00:39<00:18, 7147.09it/s] 68%|██████▊   | 270185/400000 [00:39<00:18, 7073.44it/s] 68%|██████▊   | 270893/400000 [00:39<00:18, 7047.35it/s] 68%|██████▊   | 271599/400000 [00:39<00:18, 7049.43it/s] 68%|██████▊   | 272309/400000 [00:39<00:18, 7062.63it/s] 68%|██████▊   | 273016/400000 [00:40<00:18, 7041.26it/s] 68%|██████▊   | 273721/400000 [00:40<00:17, 7024.30it/s] 69%|██████▊   | 274424/400000 [00:40<00:17, 7022.06it/s] 69%|██████▉   | 275138/400000 [00:40<00:17, 7055.01it/s] 69%|██████▉   | 275844/400000 [00:40<00:17, 6997.55it/s] 69%|██████▉   | 276554/400000 [00:40<00:17, 7026.27it/s] 69%|██████▉   | 277257/400000 [00:40<00:17, 7014.94it/s] 69%|██████▉   | 277960/400000 [00:40<00:17, 7018.38it/s] 70%|██████▉   | 278662/400000 [00:40<00:17, 7004.42it/s] 70%|██████▉   | 279363/400000 [00:40<00:17, 6998.01it/s] 70%|███████   | 280070/400000 [00:41<00:17, 7018.20it/s] 70%|███████   | 280772/400000 [00:41<00:17, 7001.76it/s] 70%|███████   | 281482/400000 [00:41<00:16, 7029.08it/s] 71%|███████   | 282188/400000 [00:41<00:16, 7036.38it/s] 71%|███████   | 282892/400000 [00:41<00:16, 6982.18it/s] 71%|███████   | 283599/400000 [00:41<00:16, 7007.96it/s] 71%|███████   | 284300/400000 [00:41<00:16, 6984.74it/s] 71%|███████   | 284999/400000 [00:41<00:16, 6974.26it/s] 71%|███████▏  | 285707/400000 [00:41<00:16, 7003.37it/s] 72%|███████▏  | 286413/400000 [00:41<00:16, 7018.40it/s] 72%|███████▏  | 287115/400000 [00:42<00:16, 6733.05it/s] 72%|███████▏  | 287815/400000 [00:42<00:16, 6810.53it/s] 72%|███████▏  | 288519/400000 [00:42<00:16, 6875.07it/s] 72%|███████▏  | 289222/400000 [00:42<00:16, 6918.01it/s] 72%|███████▏  | 289923/400000 [00:42<00:15, 6945.20it/s] 73%|███████▎  | 290625/400000 [00:42<00:15, 6965.97it/s] 73%|███████▎  | 291323/400000 [00:42<00:15, 6955.77it/s] 73%|███████▎  | 292032/400000 [00:42<00:15, 6995.18it/s] 73%|███████▎  | 292751/400000 [00:42<00:15, 7052.31it/s] 73%|███████▎  | 293461/400000 [00:42<00:15, 7065.12it/s] 74%|███████▎  | 294168/400000 [00:43<00:15, 6984.38it/s] 74%|███████▎  | 294867/400000 [00:43<00:15, 6844.76it/s] 74%|███████▍  | 295553/400000 [00:43<00:15, 6702.36it/s] 74%|███████▍  | 296251/400000 [00:43<00:15, 6781.93it/s] 74%|███████▍  | 296951/400000 [00:43<00:15, 6845.15it/s] 74%|███████▍  | 297662/400000 [00:43<00:14, 6921.17it/s] 75%|███████▍  | 298355/400000 [00:43<00:14, 6923.48it/s] 75%|███████▍  | 299055/400000 [00:43<00:14, 6944.48it/s] 75%|███████▍  | 299750/400000 [00:43<00:14, 6891.38it/s] 75%|███████▌  | 300440/400000 [00:44<00:14, 6812.10it/s] 75%|███████▌  | 301137/400000 [00:44<00:14, 6858.34it/s] 75%|███████▌  | 301824/400000 [00:44<00:14, 6854.30it/s] 76%|███████▌  | 302526/400000 [00:44<00:14, 6903.08it/s] 76%|███████▌  | 303221/400000 [00:44<00:13, 6914.23it/s] 76%|███████▌  | 303913/400000 [00:44<00:13, 6914.18it/s] 76%|███████▌  | 304611/400000 [00:44<00:13, 6931.69it/s] 76%|███████▋  | 305305/400000 [00:44<00:13, 6915.89it/s] 77%|███████▋  | 306017/400000 [00:44<00:13, 6975.35it/s] 77%|███████▋  | 306736/400000 [00:44<00:13, 7036.04it/s] 77%|███████▋  | 307458/400000 [00:45<00:13, 7087.65it/s] 77%|███████▋  | 308168/400000 [00:45<00:13, 6989.98it/s] 77%|███████▋  | 308882/400000 [00:45<00:12, 7032.19it/s] 77%|███████▋  | 309603/400000 [00:45<00:12, 7083.15it/s] 78%|███████▊  | 310322/400000 [00:45<00:12, 7114.13it/s] 78%|███████▊  | 311034/400000 [00:45<00:12, 7075.34it/s] 78%|███████▊  | 311742/400000 [00:45<00:12, 7058.87it/s] 78%|███████▊  | 312449/400000 [00:45<00:12, 6966.63it/s] 78%|███████▊  | 313157/400000 [00:45<00:12, 7000.21it/s] 78%|███████▊  | 313869/400000 [00:45<00:12, 7034.39it/s] 79%|███████▊  | 314575/400000 [00:46<00:12, 7040.26it/s] 79%|███████▉  | 315280/400000 [00:46<00:12, 7023.72it/s] 79%|███████▉  | 315983/400000 [00:46<00:12, 7000.29it/s] 79%|███████▉  | 316684/400000 [00:46<00:12, 6885.38it/s] 79%|███████▉  | 317374/400000 [00:46<00:12, 6803.58it/s] 80%|███████▉  | 318055/400000 [00:46<00:12, 6724.62it/s] 80%|███████▉  | 318729/400000 [00:46<00:12, 6700.99it/s] 80%|███████▉  | 319400/400000 [00:46<00:12, 6540.99it/s] 80%|████████  | 320112/400000 [00:46<00:11, 6702.23it/s] 80%|████████  | 320807/400000 [00:46<00:11, 6771.37it/s] 80%|████████  | 321514/400000 [00:47<00:11, 6857.59it/s] 81%|████████  | 322225/400000 [00:47<00:11, 6928.93it/s] 81%|████████  | 322922/400000 [00:47<00:11, 6940.99it/s] 81%|████████  | 323627/400000 [00:47<00:10, 6972.26it/s] 81%|████████  | 324325/400000 [00:47<00:10, 6964.69it/s] 81%|████████▏ | 325035/400000 [00:47<00:10, 7004.15it/s] 81%|████████▏ | 325736/400000 [00:47<00:10, 6972.44it/s] 82%|████████▏ | 326434/400000 [00:47<00:10, 6965.21it/s] 82%|████████▏ | 327140/400000 [00:47<00:10, 6992.35it/s] 82%|████████▏ | 327845/400000 [00:47<00:10, 7008.06it/s] 82%|████████▏ | 328546/400000 [00:48<00:10, 6999.48it/s] 82%|████████▏ | 329247/400000 [00:48<00:10, 6996.62it/s] 82%|████████▏ | 329947/400000 [00:48<00:10, 6901.17it/s] 83%|████████▎ | 330644/400000 [00:48<00:10, 6920.24it/s] 83%|████████▎ | 331337/400000 [00:48<00:10, 6821.79it/s] 83%|████████▎ | 332039/400000 [00:48<00:09, 6878.42it/s] 83%|████████▎ | 332744/400000 [00:48<00:09, 6928.82it/s] 83%|████████▎ | 333443/400000 [00:48<00:09, 6943.51it/s] 84%|████████▎ | 334138/400000 [00:48<00:09, 6866.05it/s] 84%|████████▎ | 334832/400000 [00:48<00:09, 6886.56it/s] 84%|████████▍ | 335544/400000 [00:49<00:09, 6952.61it/s] 84%|████████▍ | 336244/400000 [00:49<00:09, 6966.33it/s] 84%|████████▍ | 336941/400000 [00:49<00:09, 6927.03it/s] 84%|████████▍ | 337656/400000 [00:49<00:08, 6991.09it/s] 85%|████████▍ | 338356/400000 [00:49<00:08, 6972.35it/s] 85%|████████▍ | 339067/400000 [00:49<00:08, 7012.20it/s] 85%|████████▍ | 339782/400000 [00:49<00:08, 7052.41it/s] 85%|████████▌ | 340491/400000 [00:49<00:08, 7063.47it/s] 85%|████████▌ | 341199/400000 [00:49<00:08, 7066.22it/s] 85%|████████▌ | 341906/400000 [00:49<00:08, 7046.01it/s] 86%|████████▌ | 342611/400000 [00:50<00:08, 6956.88it/s] 86%|████████▌ | 343319/400000 [00:50<00:08, 6992.21it/s] 86%|████████▌ | 344043/400000 [00:50<00:07, 7063.60it/s] 86%|████████▌ | 344756/400000 [00:50<00:07, 7082.00it/s] 86%|████████▋ | 345469/400000 [00:50<00:07, 7096.11it/s] 87%|████████▋ | 346179/400000 [00:50<00:07, 7054.37it/s] 87%|████████▋ | 346885/400000 [00:50<00:07, 6942.07it/s] 87%|████████▋ | 347580/400000 [00:50<00:07, 6940.54it/s] 87%|████████▋ | 348289/400000 [00:50<00:07, 6982.46it/s] 87%|████████▋ | 349001/400000 [00:50<00:07, 7022.38it/s] 87%|████████▋ | 349710/400000 [00:51<00:07, 7039.35it/s] 88%|████████▊ | 350425/400000 [00:51<00:07, 7069.87it/s] 88%|████████▊ | 351133/400000 [00:51<00:07, 6944.61it/s] 88%|████████▊ | 351840/400000 [00:51<00:06, 6979.04it/s] 88%|████████▊ | 352541/400000 [00:51<00:06, 6987.52it/s] 88%|████████▊ | 353241/400000 [00:51<00:06, 6973.58it/s] 88%|████████▊ | 353945/400000 [00:51<00:06, 6991.79it/s] 89%|████████▊ | 354645/400000 [00:51<00:06, 6824.50it/s] 89%|████████▉ | 355329/400000 [00:51<00:06, 6776.36it/s] 89%|████████▉ | 356030/400000 [00:52<00:06, 6844.48it/s] 89%|████████▉ | 356737/400000 [00:52<00:06, 6908.88it/s] 89%|████████▉ | 357461/400000 [00:52<00:06, 7003.60it/s] 90%|████████▉ | 358169/400000 [00:52<00:05, 7023.92it/s] 90%|████████▉ | 358878/400000 [00:52<00:05, 7043.56it/s] 90%|████████▉ | 359583/400000 [00:52<00:05, 6998.20it/s] 90%|█████████ | 360285/400000 [00:52<00:05, 7004.62it/s] 90%|█████████ | 360986/400000 [00:52<00:05, 6998.01it/s] 90%|█████████ | 361694/400000 [00:52<00:05, 7021.76it/s] 91%|█████████ | 362397/400000 [00:52<00:05, 6982.63it/s] 91%|█████████ | 363113/400000 [00:53<00:05, 7034.20it/s] 91%|█████████ | 363817/400000 [00:53<00:05, 7005.66it/s] 91%|█████████ | 364523/400000 [00:53<00:05, 7021.42it/s] 91%|█████████▏| 365233/400000 [00:53<00:04, 7043.64it/s] 91%|█████████▏| 365938/400000 [00:53<00:05, 6685.06it/s] 92%|█████████▏| 366635/400000 [00:53<00:04, 6767.64it/s] 92%|█████████▏| 367333/400000 [00:53<00:04, 6828.73it/s] 92%|█████████▏| 368019/400000 [00:53<00:04, 6757.82it/s] 92%|█████████▏| 368728/400000 [00:53<00:04, 6853.78it/s] 92%|█████████▏| 369433/400000 [00:53<00:04, 6910.42it/s] 93%|█████████▎| 370141/400000 [00:54<00:04, 6958.29it/s] 93%|█████████▎| 370842/400000 [00:54<00:04, 6972.63it/s] 93%|█████████▎| 371544/400000 [00:54<00:04, 6986.10it/s] 93%|█████████▎| 372244/400000 [00:54<00:04, 6924.00it/s] 93%|█████████▎| 372941/400000 [00:54<00:03, 6936.47it/s] 93%|█████████▎| 373635/400000 [00:54<00:03, 6866.98it/s] 94%|█████████▎| 374323/400000 [00:54<00:03, 6818.75it/s] 94%|█████████▍| 375027/400000 [00:54<00:03, 6881.60it/s] 94%|█████████▍| 375719/400000 [00:54<00:03, 6892.49it/s] 94%|█████████▍| 376409/400000 [00:54<00:03, 6873.50it/s] 94%|█████████▍| 377097/400000 [00:55<00:03, 6842.86it/s] 94%|█████████▍| 377782/400000 [00:55<00:03, 6819.55it/s] 95%|█████████▍| 378465/400000 [00:55<00:03, 6804.54it/s] 95%|█████████▍| 379158/400000 [00:55<00:03, 6840.15it/s] 95%|█████████▍| 379856/400000 [00:55<00:02, 6879.20it/s] 95%|█████████▌| 380545/400000 [00:55<00:02, 6846.75it/s] 95%|█████████▌| 381230/400000 [00:55<00:02, 6646.75it/s] 95%|█████████▌| 381937/400000 [00:55<00:02, 6765.61it/s] 96%|█████████▌| 382640/400000 [00:55<00:02, 6839.02it/s] 96%|█████████▌| 383332/400000 [00:55<00:02, 6862.51it/s] 96%|█████████▌| 384031/400000 [00:56<00:02, 6900.07it/s] 96%|█████████▌| 384741/400000 [00:56<00:02, 6957.37it/s] 96%|█████████▋| 385466/400000 [00:56<00:02, 7041.59it/s] 97%|█████████▋| 386179/400000 [00:56<00:01, 7065.93it/s] 97%|█████████▋| 386887/400000 [00:56<00:01, 6964.90it/s] 97%|█████████▋| 387585/400000 [00:56<00:01, 6816.25it/s] 97%|█████████▋| 388268/400000 [00:56<00:01, 6696.64it/s] 97%|█████████▋| 388939/400000 [00:56<00:01, 6439.91it/s] 97%|█████████▋| 389586/400000 [00:56<00:01, 6211.48it/s] 98%|█████████▊| 390233/400000 [00:57<00:01, 6285.41it/s] 98%|█████████▊| 390927/400000 [00:57<00:01, 6466.96it/s] 98%|█████████▊| 391631/400000 [00:57<00:01, 6628.14it/s] 98%|█████████▊| 392339/400000 [00:57<00:01, 6756.25it/s] 98%|█████████▊| 393030/400000 [00:57<00:01, 6797.51it/s] 98%|█████████▊| 393727/400000 [00:57<00:00, 6845.69it/s] 99%|█████████▊| 394436/400000 [00:57<00:00, 6915.00it/s] 99%|█████████▉| 395149/400000 [00:57<00:00, 6977.46it/s] 99%|█████████▉| 395864/400000 [00:57<00:00, 7027.22it/s] 99%|█████████▉| 396568/400000 [00:57<00:00, 6989.34it/s] 99%|█████████▉| 397268/400000 [00:58<00:00, 6904.31it/s] 99%|█████████▉| 397960/400000 [00:58<00:00, 6855.58it/s]100%|█████████▉| 398656/400000 [00:58<00:00, 6885.75it/s]100%|█████████▉| 399345/400000 [00:58<00:00, 6853.34it/s]100%|█████████▉| 399999/400000 [00:58<00:00, 6846.83it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fb22f54c0f0>, <torchtext.data.dataset.TabularDataset object at 0x7fb22f54c240>, <torchtext.vocab.Vocab object at 0x7fb22f54c160>), {}) 



  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/af5a6c5b0d843e4b9ea61430a3320e1d19adf413', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'af5a6c5b0d843e4b9ea61430a3320e1d19adf413', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/af5a6c5b0d843e4b9ea61430a3320e1d19adf413

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/af5a6c5b0d843e4b9ea61430a3320e1d19adf413

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/af5a6c5b0d843e4b9ea61430a3320e1d19adf413

 ************************************************************************************************************************

  ############Check model ################################ 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info": {
    "dataset": "mlmodels/dataset/text/ag_news_csv",
    "train": true,
    "alphabet_size": 69,
    "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size": 1014,
    "num_of_classes": 4
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels.preprocess.generic:pandasDataset",
      "args": {
        "colX": [
          "colX"
        ],
        "coly": [
          "coly"
        ],
        "encoding": "'ISO-8859-1'",
        "read_csv_parm": {
          "usecols": [
            0,
            1
          ],
          "names": [
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name": "tokenizer",
      "uri": "mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args": {
        "data_source": "",
        "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size": 1014,
        "num_of_classes": 4
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
  "data_info": {
    "dataset": "mlmodels/dataset/text/ag_news_csv",
    "train": true,
    "alphabet_size": 69,
    "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size": 1014,
    "num_of_classes": 4
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels.preprocess.generic:pandasDataset",
      "args": {
        "colX": [
          "colX"
        ],
        "coly": [
          "coly"
        ],
        "encoding": "'ISO-8859-1'",
        "read_csv_parm": {
          "usecols": [
            0,
            1
          ],
          "names": [
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name": "tokenizer",
      "uri": "mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args": {
        "data_source": "",
        "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size": 1014,
        "num_of_classes": 4
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
  "data_info": {
    "dataset": "mlmodels/dataset/text/imdb",
    "pass_data_pars": false,
    "train": true,
    "maxlen": 40,
    "max_features": 5
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels.preprocess.generic:NumpyDataset",
      "args": {
        "numpy_loader_args": {
          "allow_pickle": true
        },
        "encoding": "'ISO-8859-1'"
      }
    },
    {
      "name": "imdb_process",
      "uri": "mlmodels.preprocess.text_keras:IMDBDataset",
      "args": {
        "num_words": 5
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
  "data_info": {
    "data_path": "dataset/text/",
    "dataset": "ner_dataset.csv",
    "pass_data_pars": false,
    "train": true
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels.preprocess.generic:pandasDataset",
      "args": {
        "read_csv_parm": {
          "encoding": "ISO-8859-1"
        },
        "colX": [],
        "coly": []
      }
    },
    {
      "uri": "mlmodels.preprocess.text_keras:Preprocess_namentity",
      "args": {
        "max_len": 75
      },
      "internal_states": [
        "word_count"
      ]
    },
    {
      "name": "split_xy",
      "uri": "mlmodels.dataloader:split_xy_from_dict",
      "args": {
        "col_Xinput": [
          "X"
        ],
        "col_yinput": [
          "y"
        ]
      }
    },
    {
      "name": "split_train_test",
      "uri": "sklearn.model_selection:train_test_split",
      "args": {
        "test_size": 0.5
      }
    },
    {
      "name": "saver",
      "uri": "mlmodels.dataloader:pickle_dump",
      "args": {
        "path": "mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl"
      }
    }
  ],
  "output": {
    "shape": [
      [
        75
      ],
      [
        75,
        18
      ]
    ],
    "max_len": 75
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fca06deb6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fca06deb6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fca6ff47ae8> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fca6ff47ae8> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fca1c91a378> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fca1c91a378> 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/torchhub_cnn_dataloader.json 
 

  #####  Load JSON data_pars 

  {
  "data_info": {
    "data_path": "dataset/vision/MNIST",
    "dataset": "MNIST",
    "data_type": "tch_dataset",
    "batch_size": 10,
    "train": true
  },
  "preprocessors": [
    {
      "name": "tch_dataset_start",
      "uri": "mlmodels.preprocess.generic:get_dataset_torch",
      "args": {
        "dataloader": "torchvision.datasets:MNIST",
        "to_image": true,
        "transform": {
          "uri": "mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars": false,
          "arg": {
            "fixed_size": 256,
            "path": "dataset/vision/MNIST/"
          }
        },
        "shuffle": true,
        "download": true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fca1d7d0268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fca1d7d0268> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fca1d7d0268> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 27%|██▋       | 2719744/9912422 [00:00<00:00, 27150470.63it/s]9920512it [00:00, 34595191.35it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 281218.83it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7748822.88it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 75816.17it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fca06db49b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc9f51e1be0>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/model_list_CIFAR.json 
 

  #####  Load JSON data_pars 

  {
  "data_info": {
    "data_path": "dataset/vision/cifar10/",
    "dataset": "CIFAR10",
    "data_type": "tf_dataset",
    "batch_size": 10,
    "train": true
  },
  "preprocessors": [
    {
      "name": "tf_dataset_start",
      "uri": "mlmodels.preprocess.generic:tf_dataset_download",
      "arg": {
        "train_samples": 2000,
        "test_samples": 500,
        "shuffle": true,
        "download": true
      }
    },
    {
      "uri": "mlmodels.preprocess.generic:get_dataset_torch",
      "args": {
        "dataloader": "mlmodels.preprocess.generic:NumpyDataset",
        "to_image": true,
        "transform": {
          "uri": "mlmodels.preprocess.image:torch_transform_generic",
          "pass_data_pars": false,
          "arg": {
            "fixed_size": 256
          }
        },
        "shuffle": true,
        "download": true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:tf_dataset_download {} 

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fca1d7c8e18> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fca1d7c8e18> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fca1d7c8e18> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:19,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:18,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:17,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:17,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:16,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:53,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:53,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:52,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:52,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:51,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:51,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:50,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:35,  4.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:35,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:34,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:34,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:34,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:33,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:33,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:22,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:22,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:22,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:22,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:22,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  7.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:14,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:14,  7.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:14,  7.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 25.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 25.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 25.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 41.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 41.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 41.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 41.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 57.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 57.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 57.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 57.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 57.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 57.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 57.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 57.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 57.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 57.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 57.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 65.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 71.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 76.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 76.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 76.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 76.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 76.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 76.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 76.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 81.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 81.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 81.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 81.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 81.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.24s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.25 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.10s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 81.25 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.10s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.11s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 39.44 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.11s/ url]
0 examples [00:00, ? examples/s]2020-05-27 05:55:09.378987: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-27 05:55:09.382872: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-05-27 05:55:09.383085: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55aee8dce000 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-27 05:55:09.383105: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
86 examples [00:00, 854.18 examples/s]181 examples [00:00, 879.37 examples/s]296 examples [00:00, 944.94 examples/s]396 examples [00:00, 958.50 examples/s]511 examples [00:00, 1008.40 examples/s]626 examples [00:00, 1046.79 examples/s]740 examples [00:00, 1071.67 examples/s]854 examples [00:00, 1090.67 examples/s]967 examples [00:00, 1101.18 examples/s]1086 examples [00:01, 1125.70 examples/s]1203 examples [00:01, 1138.10 examples/s]1318 examples [00:01, 1141.52 examples/s]1435 examples [00:01, 1148.86 examples/s]1550 examples [00:01, 1135.88 examples/s]1664 examples [00:01, 1120.75 examples/s]1780 examples [00:01, 1132.12 examples/s]1896 examples [00:01, 1139.54 examples/s]2013 examples [00:01, 1147.44 examples/s]2128 examples [00:01, 1140.01 examples/s]2242 examples [00:02, 1134.71 examples/s]2361 examples [00:02, 1148.73 examples/s]2481 examples [00:02, 1161.45 examples/s]2602 examples [00:02, 1174.91 examples/s]2720 examples [00:02, 1158.28 examples/s]2839 examples [00:02, 1165.63 examples/s]2958 examples [00:02, 1170.99 examples/s]3077 examples [00:02, 1173.69 examples/s]3195 examples [00:02, 1175.20 examples/s]3313 examples [00:02, 1159.04 examples/s]3429 examples [00:03, 1156.72 examples/s]3545 examples [00:03, 1153.64 examples/s]3661 examples [00:03, 1115.17 examples/s]3780 examples [00:03, 1134.69 examples/s]3894 examples [00:03, 1132.89 examples/s]4008 examples [00:03, 1133.61 examples/s]4128 examples [00:03, 1152.21 examples/s]4245 examples [00:03, 1156.61 examples/s]4362 examples [00:03, 1159.05 examples/s]4478 examples [00:03, 1147.65 examples/s]4596 examples [00:04, 1156.74 examples/s]4712 examples [00:04, 1156.45 examples/s]4828 examples [00:04, 1155.50 examples/s]4945 examples [00:04, 1159.77 examples/s]5062 examples [00:04, 1150.52 examples/s]5178 examples [00:04, 1146.56 examples/s]5294 examples [00:04, 1149.14 examples/s]5409 examples [00:04, 918.85 examples/s] 5521 examples [00:04, 970.83 examples/s]5626 examples [00:05, 991.25 examples/s]5741 examples [00:05, 1033.21 examples/s]5856 examples [00:05, 1063.92 examples/s]5968 examples [00:05, 1079.14 examples/s]6078 examples [00:05, 1079.87 examples/s]6194 examples [00:05, 1101.03 examples/s]6310 examples [00:05, 1117.72 examples/s]6428 examples [00:05, 1134.50 examples/s]6544 examples [00:05, 1141.70 examples/s]6661 examples [00:05, 1148.27 examples/s]6781 examples [00:06, 1161.75 examples/s]6899 examples [00:06, 1166.68 examples/s]7017 examples [00:06, 1167.82 examples/s]7134 examples [00:06, 1130.63 examples/s]7249 examples [00:06, 1134.69 examples/s]7363 examples [00:06, 1104.00 examples/s]7478 examples [00:06, 1116.37 examples/s]7590 examples [00:06, 1116.06 examples/s]7703 examples [00:06, 1118.69 examples/s]7816 examples [00:06, 1119.71 examples/s]7934 examples [00:07, 1135.46 examples/s]8051 examples [00:07, 1144.91 examples/s]8166 examples [00:07, 1133.31 examples/s]8280 examples [00:07, 1077.99 examples/s]8392 examples [00:07, 1089.75 examples/s]8506 examples [00:07, 1101.98 examples/s]8620 examples [00:07, 1111.65 examples/s]8737 examples [00:07, 1127.14 examples/s]8853 examples [00:07, 1135.05 examples/s]8967 examples [00:07, 1131.91 examples/s]9081 examples [00:08, 1124.99 examples/s]9194 examples [00:08, 1091.25 examples/s]9308 examples [00:08, 1103.07 examples/s]9425 examples [00:08, 1119.73 examples/s]9538 examples [00:08, 1110.35 examples/s]9651 examples [00:08, 1115.59 examples/s]9765 examples [00:08, 1121.34 examples/s]9880 examples [00:08, 1127.80 examples/s]9993 examples [00:08, 1114.12 examples/s]10105 examples [00:09, 1062.22 examples/s]10224 examples [00:09, 1096.83 examples/s]10343 examples [00:09, 1122.13 examples/s]10456 examples [00:09, 1110.61 examples/s]10572 examples [00:09, 1124.71 examples/s]10685 examples [00:09, 1097.09 examples/s]10802 examples [00:09, 1115.44 examples/s]10917 examples [00:09, 1123.21 examples/s]11030 examples [00:09, 1084.10 examples/s]11139 examples [00:09, 1073.06 examples/s]11252 examples [00:10, 1088.76 examples/s]11364 examples [00:10, 1095.95 examples/s]11478 examples [00:10, 1107.29 examples/s]11589 examples [00:10, 1094.90 examples/s]11705 examples [00:10, 1111.71 examples/s]11817 examples [00:10, 1114.00 examples/s]11932 examples [00:10, 1123.12 examples/s]12051 examples [00:10, 1140.01 examples/s]12167 examples [00:10, 1144.29 examples/s]12284 examples [00:10, 1149.87 examples/s]12400 examples [00:11, 1142.44 examples/s]12516 examples [00:11, 1147.45 examples/s]12634 examples [00:11, 1156.02 examples/s]12750 examples [00:11, 1125.03 examples/s]12863 examples [00:11, 1125.03 examples/s]12976 examples [00:11, 1111.79 examples/s]13088 examples [00:11, 1114.17 examples/s]13200 examples [00:11, 1109.32 examples/s]13319 examples [00:11, 1130.10 examples/s]13435 examples [00:12, 1137.38 examples/s]13552 examples [00:12, 1144.02 examples/s]13672 examples [00:12, 1158.22 examples/s]13792 examples [00:12, 1169.73 examples/s]13910 examples [00:12, 1138.02 examples/s]14029 examples [00:12, 1151.48 examples/s]14145 examples [00:12, 1150.34 examples/s]14261 examples [00:12, 1152.49 examples/s]14377 examples [00:12, 1151.41 examples/s]14493 examples [00:12, 1144.51 examples/s]14610 examples [00:13, 1150.70 examples/s]14726 examples [00:13, 1142.48 examples/s]14841 examples [00:13, 1128.96 examples/s]14954 examples [00:13, 1115.99 examples/s]15071 examples [00:13, 1130.31 examples/s]15187 examples [00:13, 1137.92 examples/s]15302 examples [00:13, 1141.16 examples/s]15417 examples [00:13, 1141.80 examples/s]15532 examples [00:13, 1132.47 examples/s]15649 examples [00:13, 1143.24 examples/s]15767 examples [00:14, 1153.85 examples/s]15883 examples [00:14, 1150.71 examples/s]15999 examples [00:14, 1148.65 examples/s]16116 examples [00:14, 1154.95 examples/s]16234 examples [00:14, 1160.54 examples/s]16351 examples [00:14, 1139.63 examples/s]16466 examples [00:14, 1128.87 examples/s]16580 examples [00:14, 1130.32 examples/s]16694 examples [00:14, 1117.75 examples/s]16806 examples [00:14, 1090.33 examples/s]16916 examples [00:15, 1091.42 examples/s]17027 examples [00:15, 1095.41 examples/s]17144 examples [00:15, 1115.54 examples/s]17260 examples [00:15, 1126.35 examples/s]17373 examples [00:15, 1101.88 examples/s]17490 examples [00:15, 1120.67 examples/s]17603 examples [00:15, 1119.51 examples/s]17720 examples [00:15, 1131.04 examples/s]17834 examples [00:15, 1126.07 examples/s]17950 examples [00:15, 1135.88 examples/s]18069 examples [00:16, 1150.82 examples/s]18185 examples [00:16, 1121.50 examples/s]18299 examples [00:16, 1126.61 examples/s]18414 examples [00:16, 1131.58 examples/s]18528 examples [00:16, 1130.82 examples/s]18642 examples [00:16, 1120.00 examples/s]18755 examples [00:16, 1116.21 examples/s]18869 examples [00:16, 1122.17 examples/s]18986 examples [00:16, 1134.74 examples/s]19103 examples [00:16, 1143.23 examples/s]19221 examples [00:17, 1153.56 examples/s]19337 examples [00:17, 1148.28 examples/s]19455 examples [00:17, 1156.50 examples/s]19574 examples [00:17, 1164.78 examples/s]19691 examples [00:17, 1140.75 examples/s]19809 examples [00:17, 1149.80 examples/s]19925 examples [00:17, 1104.45 examples/s]20036 examples [00:17, 1052.77 examples/s]20149 examples [00:17, 1074.06 examples/s]20265 examples [00:18, 1098.23 examples/s]20376 examples [00:18, 1090.76 examples/s]20486 examples [00:18, 1089.78 examples/s]20601 examples [00:18, 1106.76 examples/s]20716 examples [00:18, 1117.95 examples/s]20829 examples [00:18, 1073.61 examples/s]20944 examples [00:18, 1093.06 examples/s]21059 examples [00:18, 1107.45 examples/s]21171 examples [00:18, 1038.98 examples/s]21284 examples [00:18, 1063.40 examples/s]21398 examples [00:19, 1084.93 examples/s]21510 examples [00:19, 1093.24 examples/s]21623 examples [00:19, 1101.65 examples/s]21737 examples [00:19, 1111.25 examples/s]21852 examples [00:19, 1120.77 examples/s]21965 examples [00:19, 1117.75 examples/s]22077 examples [00:19, 1109.27 examples/s]22189 examples [00:19, 1093.38 examples/s]22300 examples [00:19, 1095.66 examples/s]22410 examples [00:20, 1090.43 examples/s]22523 examples [00:20, 1101.67 examples/s]22635 examples [00:20, 1103.43 examples/s]22746 examples [00:20, 1095.11 examples/s]22860 examples [00:20, 1106.19 examples/s]22975 examples [00:20, 1117.16 examples/s]23090 examples [00:20, 1124.13 examples/s]23203 examples [00:20, 1121.68 examples/s]23316 examples [00:20, 1120.55 examples/s]23431 examples [00:20, 1127.14 examples/s]23548 examples [00:21, 1138.13 examples/s]23666 examples [00:21, 1147.69 examples/s]23781 examples [00:21, 1139.03 examples/s]23895 examples [00:21, 1113.81 examples/s]24011 examples [00:21, 1124.12 examples/s]24124 examples [00:21, 1106.26 examples/s]24235 examples [00:21, 1099.31 examples/s]24349 examples [00:21, 1109.03 examples/s]24463 examples [00:21, 1117.80 examples/s]24581 examples [00:21, 1134.02 examples/s]24695 examples [00:22, 1127.36 examples/s]24811 examples [00:22, 1134.15 examples/s]24930 examples [00:22, 1147.89 examples/s]25045 examples [00:22, 1138.36 examples/s]25162 examples [00:22, 1146.34 examples/s]25279 examples [00:22, 1153.17 examples/s]25396 examples [00:22, 1155.78 examples/s]25512 examples [00:22, 1154.05 examples/s]25628 examples [00:22, 1144.69 examples/s]25746 examples [00:22, 1152.37 examples/s]25865 examples [00:23, 1160.58 examples/s]25982 examples [00:23, 1117.14 examples/s]26096 examples [00:23, 1121.46 examples/s]26209 examples [00:23, 1099.21 examples/s]26325 examples [00:23, 1115.93 examples/s]26441 examples [00:23, 1127.70 examples/s]26560 examples [00:23, 1143.35 examples/s]26678 examples [00:23, 1152.48 examples/s]26794 examples [00:23, 1142.67 examples/s]26912 examples [00:23, 1152.49 examples/s]27030 examples [00:24, 1158.44 examples/s]27149 examples [00:24, 1167.64 examples/s]27268 examples [00:24, 1171.83 examples/s]27386 examples [00:24, 1163.47 examples/s]27505 examples [00:24, 1171.13 examples/s]27623 examples [00:24, 1138.65 examples/s]27738 examples [00:24, 1127.42 examples/s]27852 examples [00:24, 1127.22 examples/s]27968 examples [00:24, 1136.11 examples/s]28082 examples [00:25, 1107.45 examples/s]28193 examples [00:25, 1097.88 examples/s]28308 examples [00:25, 1110.20 examples/s]28422 examples [00:25, 1118.86 examples/s]28535 examples [00:25, 1108.25 examples/s]28648 examples [00:25, 1113.69 examples/s]28766 examples [00:25, 1130.39 examples/s]28883 examples [00:25, 1139.53 examples/s]28999 examples [00:25, 1143.69 examples/s]29114 examples [00:25, 1134.04 examples/s]29228 examples [00:26, 1116.30 examples/s]29340 examples [00:26, 1101.26 examples/s]29451 examples [00:26, 1097.72 examples/s]29561 examples [00:26, 1083.25 examples/s]29670 examples [00:26, 1084.34 examples/s]29781 examples [00:26, 1091.34 examples/s]29891 examples [00:26, 1081.09 examples/s]30001 examples [00:26, 1036.05 examples/s]30112 examples [00:26, 1056.33 examples/s]30230 examples [00:26, 1087.75 examples/s]30340 examples [00:27, 1088.95 examples/s]30457 examples [00:27, 1111.68 examples/s]30574 examples [00:27, 1126.33 examples/s]30689 examples [00:27, 1132.38 examples/s]30807 examples [00:27, 1145.58 examples/s]30927 examples [00:27, 1160.14 examples/s]31044 examples [00:27, 1156.43 examples/s]31160 examples [00:27, 1133.08 examples/s]31275 examples [00:27, 1137.21 examples/s]31391 examples [00:27, 1143.71 examples/s]31508 examples [00:28, 1149.42 examples/s]31624 examples [00:28, 1123.54 examples/s]31737 examples [00:28, 1120.69 examples/s]31850 examples [00:28, 1119.64 examples/s]31963 examples [00:28, 1120.58 examples/s]32080 examples [00:28, 1132.83 examples/s]32196 examples [00:28, 1138.68 examples/s]32314 examples [00:28, 1148.78 examples/s]32429 examples [00:28, 1138.53 examples/s]32543 examples [00:28, 1115.95 examples/s]32656 examples [00:29, 1118.12 examples/s]32773 examples [00:29, 1131.76 examples/s]32887 examples [00:29, 1116.56 examples/s]33001 examples [00:29, 1119.38 examples/s]33116 examples [00:29, 1127.20 examples/s]33233 examples [00:29, 1138.73 examples/s]33349 examples [00:29, 1142.68 examples/s]33465 examples [00:29, 1147.21 examples/s]33580 examples [00:29, 1131.33 examples/s]33694 examples [00:30, 1123.99 examples/s]33807 examples [00:30, 1101.25 examples/s]33919 examples [00:30, 1104.45 examples/s]34036 examples [00:30, 1120.86 examples/s]34149 examples [00:30, 1116.08 examples/s]34264 examples [00:30, 1123.43 examples/s]34377 examples [00:30, 1124.25 examples/s]34490 examples [00:30, 1079.74 examples/s]34599 examples [00:30, 1082.40 examples/s]34708 examples [00:30, 1075.83 examples/s]34826 examples [00:31, 1102.48 examples/s]34939 examples [00:31, 1110.11 examples/s]35051 examples [00:31, 1084.75 examples/s]35162 examples [00:31, 1090.18 examples/s]35272 examples [00:31, 1079.43 examples/s]35386 examples [00:31, 1094.55 examples/s]35496 examples [00:31, 1086.04 examples/s]35605 examples [00:31, 1068.71 examples/s]35719 examples [00:31, 1087.83 examples/s]35831 examples [00:31, 1095.33 examples/s]35942 examples [00:32, 1097.99 examples/s]36057 examples [00:32, 1112.70 examples/s]36173 examples [00:32, 1124.28 examples/s]36286 examples [00:32, 1124.17 examples/s]36399 examples [00:32, 1114.28 examples/s]36514 examples [00:32, 1122.15 examples/s]36630 examples [00:32, 1130.73 examples/s]36744 examples [00:32, 1131.05 examples/s]36858 examples [00:32, 1110.11 examples/s]36970 examples [00:32, 1106.20 examples/s]37081 examples [00:33, 1104.61 examples/s]37193 examples [00:33, 1107.02 examples/s]37308 examples [00:33, 1118.41 examples/s]37420 examples [00:33, 1117.72 examples/s]37532 examples [00:33, 1099.54 examples/s]37647 examples [00:33, 1113.04 examples/s]37762 examples [00:33, 1122.44 examples/s]37875 examples [00:33, 1102.00 examples/s]37988 examples [00:33, 1108.84 examples/s]38100 examples [00:34, 1110.48 examples/s]38212 examples [00:34, 1109.22 examples/s]38324 examples [00:34, 1110.32 examples/s]38436 examples [00:34, 1094.15 examples/s]38546 examples [00:34, 1089.90 examples/s]38657 examples [00:34, 1094.99 examples/s]38770 examples [00:34, 1103.01 examples/s]38881 examples [00:34, 1085.12 examples/s]38995 examples [00:34, 1099.17 examples/s]39106 examples [00:34, 1095.90 examples/s]39221 examples [00:35, 1110.12 examples/s]39334 examples [00:35, 1115.96 examples/s]39446 examples [00:35, 1117.06 examples/s]39558 examples [00:35, 1084.48 examples/s]39667 examples [00:35, 1081.06 examples/s]39780 examples [00:35, 1093.93 examples/s]39895 examples [00:35, 1108.56 examples/s]40007 examples [00:35, 1030.19 examples/s]40124 examples [00:35, 1066.60 examples/s]40234 examples [00:35, 1075.12 examples/s]40351 examples [00:36, 1101.44 examples/s]40470 examples [00:36, 1124.53 examples/s]40588 examples [00:36, 1137.75 examples/s]40703 examples [00:36, 1135.26 examples/s]40817 examples [00:36, 1129.38 examples/s]40937 examples [00:36, 1149.28 examples/s]41054 examples [00:36, 1152.95 examples/s]41170 examples [00:36, 1153.67 examples/s]41286 examples [00:36, 1139.69 examples/s]41401 examples [00:36, 1130.46 examples/s]41515 examples [00:37, 1119.11 examples/s]41632 examples [00:37, 1133.78 examples/s]41751 examples [00:37, 1149.33 examples/s]41867 examples [00:37, 1117.81 examples/s]41980 examples [00:37, 1119.63 examples/s]42096 examples [00:37, 1129.61 examples/s]42213 examples [00:37, 1140.06 examples/s]42329 examples [00:37, 1144.28 examples/s]42446 examples [00:37, 1151.85 examples/s]42562 examples [00:37, 1145.82 examples/s]42680 examples [00:38, 1155.12 examples/s]42799 examples [00:38, 1164.76 examples/s]42916 examples [00:38, 1155.65 examples/s]43036 examples [00:38, 1166.40 examples/s]43153 examples [00:38, 1140.86 examples/s]43268 examples [00:38, 1141.47 examples/s]43383 examples [00:38, 1142.89 examples/s]43498 examples [00:38, 1141.20 examples/s]43618 examples [00:38, 1157.03 examples/s]43735 examples [00:39, 1159.30 examples/s]43854 examples [00:39, 1166.69 examples/s]43971 examples [00:39, 1163.90 examples/s]44089 examples [00:39, 1167.55 examples/s]44206 examples [00:39, 1162.49 examples/s]44323 examples [00:39, 1145.96 examples/s]44439 examples [00:39, 1147.91 examples/s]44555 examples [00:39, 1148.90 examples/s]44670 examples [00:39, 1147.66 examples/s]44788 examples [00:39, 1155.00 examples/s]44904 examples [00:40, 1150.16 examples/s]45020 examples [00:40, 1139.79 examples/s]45135 examples [00:40, 1109.90 examples/s]45250 examples [00:40, 1119.64 examples/s]45363 examples [00:40, 1091.65 examples/s]45473 examples [00:40, 1093.61 examples/s]45592 examples [00:40, 1120.64 examples/s]45708 examples [00:40, 1130.80 examples/s]45822 examples [00:40, 1122.13 examples/s]45935 examples [00:40, 1117.17 examples/s]46050 examples [00:41, 1125.72 examples/s]46170 examples [00:41, 1144.58 examples/s]46287 examples [00:41, 1152.06 examples/s]46404 examples [00:41, 1156.21 examples/s]46520 examples [00:41, 1134.82 examples/s]46634 examples [00:41, 1123.54 examples/s]46747 examples [00:41, 1123.88 examples/s]46860 examples [00:41, 1121.66 examples/s]46973 examples [00:41, 1118.32 examples/s]47085 examples [00:41, 1114.15 examples/s]47197 examples [00:42, 1110.09 examples/s]47309 examples [00:42, 1101.34 examples/s]47423 examples [00:42, 1112.52 examples/s]47539 examples [00:42, 1126.02 examples/s]47656 examples [00:42, 1137.43 examples/s]47770 examples [00:42, 1119.71 examples/s]47886 examples [00:42, 1129.48 examples/s]48003 examples [00:42, 1140.76 examples/s]48119 examples [00:42, 1146.35 examples/s]48236 examples [00:42, 1153.11 examples/s]48352 examples [00:43, 1145.23 examples/s]48467 examples [00:43, 1140.16 examples/s]48584 examples [00:43, 1148.55 examples/s]48699 examples [00:43, 1132.24 examples/s]48813 examples [00:43, 1084.36 examples/s]48922 examples [00:43, 1070.15 examples/s]49036 examples [00:43, 1088.51 examples/s]49148 examples [00:43, 1096.52 examples/s]49258 examples [00:43, 1095.07 examples/s]49375 examples [00:44, 1115.02 examples/s]49487 examples [00:44, 1105.41 examples/s]49599 examples [00:44, 1109.71 examples/s]49714 examples [00:44, 1120.79 examples/s]49830 examples [00:44, 1130.82 examples/s]49944 examples [00:44, 1131.29 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 8058/50000 [00:00<00:00, 80579.50 examples/s] 44%|████▍     | 22182/50000 [00:00<00:00, 92496.75 examples/s] 73%|███████▎  | 36387/50000 [00:00<00:00, 103306.53 examples/s] 97%|█████████▋| 48635/50000 [00:00<00:00, 108396.47 examples/s]                                                                0 examples [00:00, ? examples/s]86 examples [00:00, 859.13 examples/s]196 examples [00:00, 918.29 examples/s]311 examples [00:00, 975.81 examples/s]424 examples [00:00, 1016.26 examples/s]543 examples [00:00, 1062.42 examples/s]662 examples [00:00, 1096.40 examples/s]778 examples [00:00, 1112.45 examples/s]896 examples [00:00, 1130.02 examples/s]1013 examples [00:00, 1141.20 examples/s]1132 examples [00:01, 1153.36 examples/s]1248 examples [00:01, 1153.50 examples/s]1362 examples [00:01, 1134.01 examples/s]1477 examples [00:01, 1137.55 examples/s]1592 examples [00:01, 1141.10 examples/s]1706 examples [00:01, 1111.42 examples/s]1818 examples [00:01, 1109.75 examples/s]1931 examples [00:01, 1113.70 examples/s]2046 examples [00:01, 1124.28 examples/s]2159 examples [00:01, 1096.35 examples/s]2277 examples [00:02, 1117.61 examples/s]2389 examples [00:02, 1117.42 examples/s]2504 examples [00:02, 1125.12 examples/s]2622 examples [00:02, 1139.42 examples/s]2740 examples [00:02, 1149.28 examples/s]2858 examples [00:02, 1157.40 examples/s]2974 examples [00:02, 1157.87 examples/s]3090 examples [00:02, 1150.13 examples/s]3206 examples [00:02, 1152.31 examples/s]3325 examples [00:02, 1161.49 examples/s]3442 examples [00:03, 1147.22 examples/s]3557 examples [00:03, 1137.63 examples/s]3673 examples [00:03, 1143.04 examples/s]3791 examples [00:03, 1151.17 examples/s]3909 examples [00:03, 1159.02 examples/s]4025 examples [00:03, 1156.64 examples/s]4141 examples [00:03, 1135.53 examples/s]4257 examples [00:03, 1139.57 examples/s]4372 examples [00:03, 1124.02 examples/s]4485 examples [00:03, 1125.62 examples/s]4602 examples [00:04, 1137.15 examples/s]4716 examples [00:04, 1134.47 examples/s]4830 examples [00:04, 1129.60 examples/s]4946 examples [00:04, 1138.16 examples/s]5061 examples [00:04, 1140.81 examples/s]5176 examples [00:04, 1113.45 examples/s]5288 examples [00:04, 1113.95 examples/s]5400 examples [00:04, 1093.69 examples/s]5515 examples [00:04, 1108.02 examples/s]5631 examples [00:04, 1121.94 examples/s]5748 examples [00:05, 1134.43 examples/s]5862 examples [00:05, 1127.25 examples/s]5975 examples [00:05, 1118.69 examples/s]6087 examples [00:05, 1101.74 examples/s]6198 examples [00:05, 1097.98 examples/s]6315 examples [00:05, 1116.15 examples/s]6430 examples [00:05, 1124.41 examples/s]6545 examples [00:05, 1130.66 examples/s]6659 examples [00:05, 1128.86 examples/s]6772 examples [00:06, 1106.17 examples/s]6883 examples [00:06, 1093.18 examples/s]6995 examples [00:06, 1099.56 examples/s]7113 examples [00:06, 1122.22 examples/s]7232 examples [00:06, 1139.54 examples/s]7350 examples [00:06, 1149.15 examples/s]7469 examples [00:06, 1159.45 examples/s]7586 examples [00:06, 1119.22 examples/s]7703 examples [00:06, 1133.11 examples/s]7817 examples [00:06, 1121.56 examples/s]7930 examples [00:07, 1123.53 examples/s]8043 examples [00:07, 1111.34 examples/s]8155 examples [00:07, 1113.28 examples/s]8270 examples [00:07, 1123.13 examples/s]8385 examples [00:07, 1130.81 examples/s]8503 examples [00:07, 1144.05 examples/s]8618 examples [00:07, 1128.64 examples/s]8731 examples [00:07, 1126.21 examples/s]8847 examples [00:07, 1134.68 examples/s]8964 examples [00:07, 1143.71 examples/s]9082 examples [00:08, 1154.34 examples/s]9198 examples [00:08, 1153.15 examples/s]9314 examples [00:08, 1147.03 examples/s]9430 examples [00:08, 1148.38 examples/s]9545 examples [00:08, 1147.71 examples/s]9660 examples [00:08, 1127.35 examples/s]9773 examples [00:08, 1084.40 examples/s]9882 examples [00:08, 1078.92 examples/s]9991 examples [00:08, 1076.32 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1PZYP7/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1PZYP7/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fca1d7d0268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fca1d7d0268> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fca1d7d0268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fca1c907eb8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fca06dda1d0>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/resnet34_benchmark_mnist.json 
 

  #####  Load JSON data_pars 

  {
  "data_info": {
    "data_path": "dataset/vision/MNIST/",
    "dataset": "MNIST",
    "data_type": "tch_dataset",
    "batch_size": 10,
    "train": true
  },
  "preprocessors": [
    {
      "name": "tch_dataset_start",
      "uri": "mlmodels.preprocess.generic:get_dataset_torch",
      "args": {
        "dataloader": "torchvision.datasets:MNIST",
        "to_image": true,
        "transform": {
          "uri": "mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars": false,
          "arg": {}
        },
        "shuffle": true,
        "download": true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fca1d7d0268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fca1d7d0268> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fca1d7d0268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc9fe874278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc9f51e1940>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info": {
    "data_path": "dataset/recommender/",
    "dataset": "IMDB_sample.txt",
    "data_type": "csv_dataset",
    "batch_size": 64,
    "train": true
  },
  "preprocessors": [
    {
      "uri": "mlmodels.model_tch.textcnn:split_train_valid",
      "args": {
        "frac": 0.99
      }
    },
    {
      "uri": "mlmodels.model_tch.textcnn:create_tabular_dataset",
      "args": {
        "lang": "en",
        "pretrained_emb": "glove.6B.300d"
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fc9f7816ae8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fc9f7816ae8> 

  function with postional parmater data_info <function split_train_valid at 0x7fc9f7816ae8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fc9f7816bf8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fc9f7816bf8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fc9f7816bf8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=edb266e16b9c9486474bab25d582ccba5396d42847f84c44805b0a877f8c1065
  Stored in directory: /tmp/pip-ephem-wheel-cache-v39hdeul/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:09<281:31:08, 851B/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:09<197:18:02, 1.21kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:09<138:08:31, 1.73kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:10<96:38:11, 2.48kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.93M/862M [00:10<67:34:14, 3.54kB/s].vector_cache/glove.6B.zip:   1%|          | 6.24M/862M [00:10<47:03:51, 5.05kB/s].vector_cache/glove.6B.zip:   1%|          | 9.64M/862M [00:10<32:49:00, 7.22kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.3M/862M [00:10<22:49:16, 10.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:10<15:52:12, 14.7kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.7M/862M [00:10<11:02:04, 21.0kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.2M/862M [00:10<7:40:29, 30.0kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 35.5M/862M [00:11<5:21:13, 42.9kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.1M/862M [00:11<3:43:40, 61.3kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:11<2:35:57, 87.4kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.7M/862M [00:11<1:48:37, 125kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:11<1:16:24, 177kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:12<53:19, 252kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:13<2:19:01, 96.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:14<1:39:47, 135kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:14<1:10:22, 190kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:15<51:22, 260kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:16<37:40, 354kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:16<26:42, 499kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:16<18:48, 707kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:17<59:44, 222kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:18<43:25, 306kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:18<30:40, 432kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:19<24:16, 545kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:20<18:24, 718kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:20<13:13, 997kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:21<12:18, 1.07MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:21<09:48, 1.34MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:22<07:37, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:22<05:27, 2.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:23<35:10, 372kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:23<26:03, 502kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:24<18:33, 703kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:25<15:58, 814kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:25<12:35, 1.03MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:26<09:08, 1.42MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:27<09:25, 1.37MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:27<07:59, 1.62MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:28<05:56, 2.17MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:29<07:09, 1.80MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:29<06:23, 2.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:29<04:49, 2.66MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:31<06:22, 2.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:31<05:51, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:31<04:22, 2.92MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:33<06:04, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:33<05:35, 2.27MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:33<04:15, 2.98MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:35<05:56, 2.13MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:35<05:31, 2.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:35<04:12, 3.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:37<05:53, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:37<05:29, 2.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:37<04:10, 3.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:39<05:50, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:39<05:27, 2.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:39<04:07, 3.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:39<03:04, 4.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:41<14:28, 862kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:41<11:52, 1.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:41<08:44, 1.42MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:43<08:27, 1.46MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:43<07:43, 1.60MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:43<05:45, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:43<04:13, 2.92MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:45<09:29, 1.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:45<08:26, 1.46MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:45<06:19, 1.94MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:47<06:46, 1.81MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:47<06:28, 1.89MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:47<04:57, 2.47MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:49<05:47, 2.10MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:49<05:48, 2.10MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:49<04:29, 2.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:51<05:27, 2.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:51<05:30, 2.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:51<04:17, 2.82MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:53<05:18, 2.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:53<05:24, 2.23MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:53<04:12, 2.86MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:55<05:13, 2.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:55<05:22, 2.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:55<04:11, 2.85MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:57<05:12, 2.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:57<05:18, 2.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:57<04:03, 2.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:57<03:03, 3.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:59<07:23, 1.60MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:59<06:52, 1.72MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:59<05:09, 2.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:59<03:44, 3.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [01:01<25:06, 469kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [01:01<19:12, 612kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [01:01<13:50, 849kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:03<11:54, 983kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:03<09:58, 1.17MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:03<07:23, 1.58MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:05<07:22, 1.58MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:05<06:50, 1.70MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:05<05:08, 2.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:05<03:43, 3.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:07<21:05, 548kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:07<16:22, 706kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:07<11:46, 980kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:07<08:26, 1.36MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:09<11:18, 1.02MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:09<09:31, 1.21MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:09<06:59, 1.64MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:09<05:05, 2.24MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:11<08:40, 1.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:11<07:42, 1.48MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:11<05:47, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:13<06:12, 1.83MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:13<05:59, 1.89MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:13<04:31, 2.50MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:13<03:18, 3.42MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:15<19:43, 573kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:15<15:28, 729kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:15<11:10, 1.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:15<07:56, 1.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:17<18:20, 612kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:17<14:27, 776kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:17<10:29, 1.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:19<09:27, 1.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:19<08:10, 1.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:19<06:06, 1.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:21<06:23, 1.73MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:21<06:01, 1.84MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:21<04:36, 2.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:22<05:19, 2.07MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:23<05:16, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:23<04:00, 2.74MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:23<02:56, 3.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:24<12:02, 909kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:25<10:00, 1.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:25<07:22, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:26<07:14, 1.50MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:27<06:35, 1.65MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:27<04:55, 2.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:27<03:35, 3.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:28<12:22, 874kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:29<10:10, 1.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:29<07:29, 1.44MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:30<07:17, 1.47MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:31<06:36, 1.62MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:31<04:59, 2.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:32<05:31, 1.93MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:32<05:24, 1.97MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:33<04:09, 2.56MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:34<04:55, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:34<04:56, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:35<03:49, 2.76MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:36<04:41, 2.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:36<04:45, 2.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:37<03:41, 2.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:38<04:34, 2.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:38<04:44, 2.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:39<03:40, 2.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:40<04:33, 2.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:40<04:38, 2.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:41<03:38, 2.85MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:42<04:30, 2.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:42<04:38, 2.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:43<03:36, 2.86MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:44<04:28, 2.29MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:44<04:34, 2.24MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:44<03:29, 2.93MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:45<02:36, 3.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:46<07:08, 1.43MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:46<06:25, 1.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:46<04:51, 2.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:48<05:18, 1.90MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:48<05:08, 1.97MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:48<03:56, 2.56MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:50<04:41, 2.14MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:50<04:41, 2.14MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:50<03:37, 2.76MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:52<04:26, 2.25MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:52<04:33, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:52<03:28, 2.86MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:53<02:34, 3.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:54<09:43, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:54<08:12, 1.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:54<06:05, 1.62MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:56<06:07, 1.61MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:56<05:42, 1.72MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:56<04:16, 2.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:56<03:07, 3.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:58<10:07, 965kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:58<08:27, 1.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:58<06:15, 1.56MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:00<06:12, 1.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [02:00<05:42, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [02:00<04:16, 2.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:00<03:08, 3.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:02<08:08, 1.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:02<07:04, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [02:02<05:17, 1.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:04<05:31, 1.73MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [02:04<05:12, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:04<03:55, 2.43MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:04<02:51, 3.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:06<35:00, 271kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:06<25:53, 367kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:06<18:24, 514kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:08<14:38, 644kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:08<11:33, 815kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:08<08:21, 1.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:08<05:57, 1.57MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:10<12:37, 741kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:10<10:11, 918kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:10<07:23, 1.26MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:10<05:19, 1.75MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:12<08:21, 1.11MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:12<07:08, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:12<05:15, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:12<03:48, 2.42MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:14<10:06, 912kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:14<08:22, 1.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:14<06:10, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:16<06:02, 1.51MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:16<05:31, 1.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:16<04:07, 2.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:16<03:03, 2.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:18<06:10, 1.47MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:18<05:37, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:18<04:15, 2.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:20<05:04, 1.78MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:22<04:33, 1.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:22<04:26, 2.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:22<03:21, 2.65MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:22<02:29, 3.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:24<07:39, 1.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:24<06:35, 1.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:24<04:55, 1.80MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:26<05:07, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:26<04:50, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:26<03:39, 2.40MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:26<02:39, 3.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:28<17:01, 513kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:28<13:08, 664kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:28<09:29, 918kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:30<08:15, 1.05MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:30<07:01, 1.23MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:30<05:12, 1.66MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:32<05:17, 1.63MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:32<04:56, 1.74MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:32<03:45, 2.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:34<04:14, 2.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:34<04:10, 2.04MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:34<03:10, 2.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:34<02:19, 3.63MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:36<08:53, 951kB/s] .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:36<07:26, 1.13MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:36<05:30, 1.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:38<05:26, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:38<04:59, 1.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:38<03:44, 2.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:38<02:44, 3.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:40<07:59, 1.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:40<06:45, 1.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:40<04:58, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:40<03:37, 2.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:42<06:22, 1.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:42<05:36, 1.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:42<04:09, 1.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:42<03:02, 2.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:44<05:57, 1.37MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:44<05:05, 1.60MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:44<04:10, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:44<03:01, 2.68MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:46<05:43, 1.42MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:46<05:08, 1.58MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:46<03:52, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:48<04:14, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:48<04:07, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:48<03:08, 2.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:48<02:20, 3.41MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:50<05:20, 1.49MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:50<04:43, 1.68MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:50<03:32, 2.24MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:52<04:07, 1.92MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:52<04:00, 1.97MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:52<03:02, 2.60MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:52<02:14, 3.51MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:54<07:10, 1.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:54<06:07, 1.28MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:54<04:30, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:54<03:17, 2.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:56<05:56, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:56<05:15, 1.48MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:56<03:53, 1.99MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:56<02:51, 2.70MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:58<05:32, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:58<04:59, 1.54MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:58<03:45, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:59<04:04, 1.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [03:00<03:55, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [03:00<02:58, 2.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [03:00<02:11, 3.45MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:01<05:48, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:02<05:07, 1.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:02<03:50, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:03<04:06, 1.82MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:04<03:55, 1.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:04<02:58, 2.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:04<02:10, 3.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:05<07:20, 1.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:06<06:10, 1.20MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:06<04:31, 1.63MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:06<03:15, 2.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:07<07:56, 927kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:08<06:34, 1.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:08<04:51, 1.51MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:09<04:46, 1.53MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:09<04:23, 1.66MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:10<03:16, 2.21MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:10<02:24, 3.01MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:11<05:57, 1.21MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:11<05:12, 1.39MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:12<03:53, 1.85MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:13<04:04, 1.75MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:13<03:51, 1.85MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:14<02:56, 2.42MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:15<03:24, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:15<03:23, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:16<02:37, 2.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:17<03:09, 2.21MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:17<03:13, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:18<02:30, 2.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:19<03:04, 2.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:19<02:58, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:20<02:17, 3.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:21<03:02, 2.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:21<03:06, 2.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:21<02:22, 2.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:22<01:46, 3.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:23<04:35, 1.48MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:23<04:11, 1.62MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:23<03:07, 2.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:24<02:16, 2.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:25<06:27, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:25<05:27, 1.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:25<04:00, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:26<02:54, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:27<06:03, 1.10MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:27<05:10, 1.29MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:27<03:49, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:27<02:44, 2.41MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:29<44:35, 148kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:29<32:07, 205kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:29<22:36, 290kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:29<15:50, 412kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:31<16:57, 385kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:31<12:46, 510kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:31<09:08, 710kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:33<07:35, 849kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:33<06:13, 1.04MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:33<04:34, 1.41MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:35<04:24, 1.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:35<03:59, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:35<02:58, 2.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:35<02:10, 2.91MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:37<05:11, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:37<04:30, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:37<03:22, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:39<03:32, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:39<03:25, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:39<02:36, 2.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:41<02:59, 2.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:41<02:58, 2.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:41<02:17, 2.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:43<02:45, 2.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:43<02:47, 2.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:43<02:09, 2.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:45<02:39, 2.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:45<02:43, 2.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:45<02:05, 2.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:47<02:36, 2.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:47<02:39, 2.24MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:47<02:02, 2.92MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:47<01:30, 3.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:49<06:06, 965kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:49<05:07, 1.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:49<03:46, 1.56MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:51<03:44, 1.56MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:51<03:26, 1.70MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:51<02:34, 2.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:51<01:52, 3.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:53<05:22, 1.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:53<04:34, 1.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:53<03:23, 1.69MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:55<03:26, 1.65MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:55<03:06, 1.83MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:55<02:20, 2.42MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:57<02:48, 2.00MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:57<02:45, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:57<02:07, 2.64MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:59<02:32, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:59<02:35, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:59<02:00, 2.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:01<02:26, 2.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:01<02:29, 2.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:01<01:53, 2.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:01<01:25, 3.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:03<03:30, 1.55MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:03<03:12, 1.69MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [04:03<02:24, 2.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:03<01:44, 3.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:05<12:03, 444kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:05<09:11, 581kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:05<06:34, 810kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:05<04:40, 1.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:07<06:01, 877kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:07<04:57, 1.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:07<03:36, 1.46MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:07<02:36, 2.01MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:09<04:09, 1.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:09<03:38, 1.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:09<02:43, 1.90MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:11<02:52, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:11<02:43, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:11<02:03, 2.48MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:11<01:29, 3.40MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:13<1:41:00, 50.2kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:13<1:11:21, 71.0kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:13<49:56, 101kB/s]   .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:15<35:36, 140kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:15<25:35, 195kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:15<17:59, 277kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:15<12:34, 393kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:17<13:25, 367kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:17<10:05, 488kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:17<07:12, 680kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:19<05:56, 819kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:19<04:51, 1.00MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:19<03:32, 1.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:21<03:22, 1.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:21<03:02, 1.58MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:21<02:15, 2.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:21<01:38, 2.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:22<03:39, 1.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:23<03:14, 1.46MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:23<02:24, 1.96MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:23<01:44, 2.69MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:24<05:14, 888kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:25<04:20, 1.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:25<03:11, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:26<03:05, 1.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:27<02:48, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:27<02:05, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:27<01:30, 2.99MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:28<05:55, 763kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:29<04:47, 942kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:29<03:29, 1.29MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:29<02:30, 1.78MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:31<04:03, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:31<02:51, 1.54MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:32<03:51, 1.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:32<03:19, 1.31MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:33<02:28, 1.76MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 604M/862M [04:34<02:32, 1.69MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:34<02:16, 1.89MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:35<01:50, 2.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:35<01:20, 3.18MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:36<02:48, 1.51MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:36<02:33, 1.65MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:37<01:56, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:38<02:08, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:38<02:05, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:39<01:34, 2.65MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:39<01:10, 3.50MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:40<02:44, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:40<02:30, 1.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:40<01:51, 2.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:41<01:22, 2.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:42<02:45, 1.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:42<02:29, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:42<01:51, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:43<01:21, 2.94MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:44<03:15, 1.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:44<02:49, 1.40MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:44<02:05, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:45<01:29, 2.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:46<09:54, 393kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:46<07:29, 520kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:46<05:21, 723kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:48<04:26, 862kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:48<03:38, 1.05MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:48<02:40, 1.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:50<02:34, 1.46MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:50<02:20, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:50<01:45, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:52<01:55, 1.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:52<01:51, 1.98MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:52<01:24, 2.61MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:52<01:01, 3.53MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:54<03:40, 984kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:54<03:04, 1.17MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:54<02:14, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:54<01:37, 2.20MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:56<02:53, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:56<02:31, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:56<01:51, 1.89MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:56<01:20, 2.60MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:58<04:13, 823kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:58<03:27, 1.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:58<02:30, 1.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:58<01:47, 1.91MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [05:00<03:18, 1.03MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [05:00<02:43, 1.25MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:00<01:59, 1.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:00<01:25, 2.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:02<15:11, 220kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:02<11:01, 303kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:02<07:45, 428kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:04<06:03, 541kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:04<04:42, 695kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:04<03:22, 965kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:04<02:22, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:06<07:13, 444kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:06<05:29, 582kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:06<03:55, 812kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:06<02:46, 1.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:08<03:20, 941kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:08<02:46, 1.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:08<02:00, 1.54MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:08<01:27, 2.12MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:10<02:31, 1.22MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:10<02:11, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:10<01:38, 1.86MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:12<01:42, 1.76MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:12<01:36, 1.86MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:12<01:13, 2.43MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:14<01:24, 2.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:14<01:24, 2.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:14<01:03, 2.75MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:14<00:47, 3.62MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:16<01:52, 1.52MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:16<01:38, 1.73MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:16<01:16, 2.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:16<00:55, 3.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:18<01:52, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:18<01:42, 1.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:18<01:16, 2.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:18<00:55, 2.97MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:20<02:45, 984kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:20<02:18, 1.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:20<01:41, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:20<01:12, 2.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:22<03:10, 833kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:22<02:35, 1.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:22<01:53, 1.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:24<01:48, 1.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:24<01:37, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:24<01:13, 2.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:26<01:19, 1.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:26<01:17, 1.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:26<00:58, 2.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:28<01:08, 2.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:28<01:08, 2.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:28<00:51, 2.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:28<00:37, 3.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:30<02:09, 1.10MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:30<01:50, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:30<01:21, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:32<01:22, 1.68MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:32<01:17, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:32<00:57, 2.37MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:32<00:41, 3.25MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:34<08:14, 271kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:34<06:04, 367kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:34<04:17, 515kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:36<03:21, 645kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:36<02:39, 815kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:36<01:54, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:38<01:42, 1.22MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:38<01:27, 1.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:38<01:04, 1.92MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:40<01:09, 1.74MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:40<01:06, 1.83MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:40<00:49, 2.43MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:40<00:36, 3.28MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:42<01:26, 1.35MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:42<01:17, 1.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:42<00:56, 2.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:42<00:40, 2.81MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:44<03:40, 515kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:44<02:50, 665kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:44<02:01, 924kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:44<01:24, 1.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:46<03:01, 602kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:46<02:19, 781kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:46<01:45, 1.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<01:13, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:48<01:37, 1.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:48<01:22, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:48<01:01, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:50<01:01, 1.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:50<00:57, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:50<00:42, 2.34MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:50<00:30, 3.18MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:52<01:30, 1.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:52<01:17, 1.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:52<00:56, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:53<00:56, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:54<00:52, 1.75MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:54<00:39, 2.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:55<00:43, 2.02MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:56<00:43, 2.05MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:56<00:32, 2.65MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:57<00:38, 2.20MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:58<00:38, 2.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:58<00:29, 2.78MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:59<00:35, 2.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [06:00<00:36, 2.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:00<00:27, 2.85MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:01<00:33, 2.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:02<00:32, 2.34MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:02<00:24, 3.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:03<00:31, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:04<00:32, 2.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:04<00:24, 2.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:05<00:29, 2.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:05<00:28, 2.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:06<00:21, 3.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:07<00:28, 2.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:07<00:28, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:08<00:21, 2.83MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:09<00:26, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:09<00:25, 2.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:10<00:19, 3.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:11<00:24, 2.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:11<00:24, 2.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:12<00:18, 2.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:12<00:13, 3.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:13<00:51, 1.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:13<00:41, 1.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:14<00:29, 1.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:15<00:29, 1.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:15<00:27, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:16<00:20, 2.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:17<00:21, 1.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:17<00:21, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:18<00:15, 2.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:18<00:11, 3.56MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:19<00:28, 1.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:19<00:25, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:20<00:18, 2.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:21<00:18, 1.86MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:21<00:17, 2.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:21<00:12, 2.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:22<00:08, 3.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:23<00:30, 1.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:23<00:24, 1.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:24<00:17, 1.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:25<00:16, 1.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:25<00:15, 1.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:25<00:10, 2.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:27<00:11, 1.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:27<00:10, 2.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:27<00:07, 2.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:29<00:08, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:29<00:07, 2.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:29<00:05, 2.92MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:31<00:06, 2.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:31<00:06, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:31<00:04, 2.82MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:33<00:04, 2.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:33<00:03, 2.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:33<00:02, 3.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:35<00:02, 2.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:35<00:02, 2.19MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:35<00:01, 2.87MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:36<00:00, 3.80MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:37<00:00, 1.56MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:37<00:00, 1.69MB/s].vector_cache/glove.6B.zip: 862MB [06:37, 2.17MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 621/400000 [00:00<01:04, 6204.53it/s]  0%|          | 1502/400000 [00:00<00:58, 6807.50it/s]  1%|          | 2378/400000 [00:00<00:54, 7294.69it/s]  1%|          | 3248/400000 [00:00<00:51, 7665.82it/s]  1%|          | 4154/400000 [00:00<00:49, 8036.09it/s]  1%|▏         | 5000/400000 [00:00<00:48, 8157.63it/s]  1%|▏         | 5856/400000 [00:00<00:47, 8273.29it/s]  2%|▏         | 6727/400000 [00:00<00:46, 8399.54it/s]  2%|▏         | 7584/400000 [00:00<00:46, 8448.61it/s]  2%|▏         | 8490/400000 [00:01<00:45, 8622.31it/s]  2%|▏         | 9417/400000 [00:01<00:44, 8806.70it/s]  3%|▎         | 10290/400000 [00:01<00:44, 8707.01it/s]  3%|▎         | 11156/400000 [00:01<00:44, 8666.79it/s]  3%|▎         | 12019/400000 [00:01<00:44, 8636.41it/s]  3%|▎         | 12891/400000 [00:01<00:44, 8661.25it/s]  3%|▎         | 13831/400000 [00:01<00:43, 8869.31it/s]  4%|▎         | 14718/400000 [00:01<00:44, 8753.42it/s]  4%|▍         | 15619/400000 [00:01<00:43, 8826.12it/s]  4%|▍         | 16522/400000 [00:01<00:43, 8884.30it/s]  4%|▍         | 17450/400000 [00:02<00:42, 8998.77it/s]  5%|▍         | 18385/400000 [00:02<00:41, 9098.95it/s]  5%|▍         | 19321/400000 [00:02<00:41, 9175.70it/s]  5%|▌         | 20275/400000 [00:02<00:40, 9281.53it/s]  5%|▌         | 21204/400000 [00:02<00:41, 9118.71it/s]  6%|▌         | 22118/400000 [00:02<00:41, 9083.74it/s]  6%|▌         | 23028/400000 [00:02<00:42, 8904.22it/s]  6%|▌         | 23924/400000 [00:02<00:42, 8919.03it/s]  6%|▌         | 24817/400000 [00:02<00:42, 8893.79it/s]  6%|▋         | 25708/400000 [00:02<00:42, 8868.27it/s]  7%|▋         | 26596/400000 [00:03<00:42, 8792.55it/s]  7%|▋         | 27484/400000 [00:03<00:42, 8817.19it/s]  7%|▋         | 28409/400000 [00:03<00:41, 8942.40it/s]  7%|▋         | 29309/400000 [00:03<00:41, 8941.81it/s]  8%|▊         | 30204/400000 [00:03<00:42, 8802.41it/s]  8%|▊         | 31086/400000 [00:03<00:42, 8665.16it/s]  8%|▊         | 31995/400000 [00:03<00:41, 8787.23it/s]  8%|▊         | 32907/400000 [00:03<00:41, 8883.41it/s]  8%|▊         | 33797/400000 [00:03<00:41, 8760.96it/s]  9%|▊         | 34675/400000 [00:03<00:42, 8693.38it/s]  9%|▉         | 35584/400000 [00:04<00:41, 8807.78it/s]  9%|▉         | 36477/400000 [00:04<00:41, 8841.64it/s]  9%|▉         | 37380/400000 [00:04<00:40, 8896.36it/s] 10%|▉         | 38294/400000 [00:04<00:40, 8965.69it/s] 10%|▉         | 39192/400000 [00:04<00:40, 8909.63it/s] 10%|█         | 40103/400000 [00:04<00:40, 8968.01it/s] 10%|█         | 41001/400000 [00:04<00:40, 8891.20it/s] 10%|█         | 41896/400000 [00:04<00:40, 8907.86it/s] 11%|█         | 42801/400000 [00:04<00:39, 8947.50it/s] 11%|█         | 43697/400000 [00:04<00:40, 8895.57it/s] 11%|█         | 44587/400000 [00:05<00:39, 8889.56it/s] 11%|█▏        | 45486/400000 [00:05<00:39, 8918.07it/s] 12%|█▏        | 46408/400000 [00:05<00:39, 9005.31it/s] 12%|█▏        | 47329/400000 [00:05<00:38, 9063.68it/s] 12%|█▏        | 48236/400000 [00:05<00:38, 9041.88it/s] 12%|█▏        | 49141/400000 [00:05<00:39, 8782.21it/s] 13%|█▎        | 50022/400000 [00:05<00:40, 8692.66it/s] 13%|█▎        | 50942/400000 [00:05<00:39, 8838.40it/s] 13%|█▎        | 51860/400000 [00:05<00:38, 8935.56it/s] 13%|█▎        | 52757/400000 [00:05<00:38, 8944.37it/s] 13%|█▎        | 53679/400000 [00:06<00:38, 9022.33it/s] 14%|█▎        | 54583/400000 [00:06<00:38, 8878.64it/s] 14%|█▍        | 55482/400000 [00:06<00:38, 8910.84it/s] 14%|█▍        | 56374/400000 [00:06<00:38, 8839.39it/s] 14%|█▍        | 57259/400000 [00:06<00:38, 8813.00it/s] 15%|█▍        | 58177/400000 [00:06<00:38, 8917.64it/s] 15%|█▍        | 59104/400000 [00:06<00:37, 9019.84it/s] 15%|█▌        | 60007/400000 [00:06<00:37, 8956.35it/s] 15%|█▌        | 60904/400000 [00:06<00:38, 8920.19it/s] 15%|█▌        | 61797/400000 [00:06<00:38, 8834.82it/s] 16%|█▌        | 62687/400000 [00:07<00:38, 8852.74it/s] 16%|█▌        | 63603/400000 [00:07<00:37, 8940.78it/s] 16%|█▌        | 64516/400000 [00:07<00:37, 8994.31it/s] 16%|█▋        | 65430/400000 [00:07<00:37, 9036.95it/s] 17%|█▋        | 66335/400000 [00:07<00:37, 8978.76it/s] 17%|█▋        | 67250/400000 [00:07<00:36, 9027.40it/s] 17%|█▋        | 68154/400000 [00:07<00:36, 9025.55it/s] 17%|█▋        | 69064/400000 [00:07<00:36, 9046.84it/s] 17%|█▋        | 69969/400000 [00:07<00:36, 9014.47it/s] 18%|█▊        | 70871/400000 [00:07<00:36, 8904.76it/s] 18%|█▊        | 71762/400000 [00:08<00:37, 8763.27it/s] 18%|█▊        | 72640/400000 [00:08<00:38, 8602.63it/s] 18%|█▊        | 73537/400000 [00:08<00:37, 8709.15it/s] 19%|█▊        | 74410/400000 [00:08<00:37, 8662.70it/s] 19%|█▉        | 75286/400000 [00:08<00:37, 8688.98it/s] 19%|█▉        | 76157/400000 [00:08<00:37, 8693.43it/s] 19%|█▉        | 77027/400000 [00:08<00:37, 8570.90it/s] 19%|█▉        | 77885/400000 [00:08<00:39, 8236.01it/s] 20%|█▉        | 78744/400000 [00:08<00:38, 8337.56it/s] 20%|█▉        | 79581/400000 [00:09<00:39, 8185.16it/s] 20%|██        | 80411/400000 [00:09<00:38, 8216.76it/s] 20%|██        | 81301/400000 [00:09<00:37, 8408.82it/s] 21%|██        | 82214/400000 [00:09<00:36, 8611.05it/s] 21%|██        | 83114/400000 [00:09<00:36, 8722.90it/s] 21%|██        | 84005/400000 [00:09<00:36, 8777.41it/s] 21%|██        | 84885/400000 [00:09<00:36, 8745.90it/s] 21%|██▏       | 85761/400000 [00:09<00:36, 8632.32it/s] 22%|██▏       | 86653/400000 [00:09<00:35, 8716.17it/s] 22%|██▏       | 87564/400000 [00:09<00:35, 8829.36it/s] 22%|██▏       | 88449/400000 [00:10<00:35, 8724.52it/s] 22%|██▏       | 89367/400000 [00:10<00:35, 8854.08it/s] 23%|██▎       | 90264/400000 [00:10<00:34, 8886.59it/s] 23%|██▎       | 91154/400000 [00:10<00:34, 8881.82it/s] 23%|██▎       | 92053/400000 [00:10<00:34, 8912.62it/s] 23%|██▎       | 92945/400000 [00:10<00:34, 8873.64it/s] 23%|██▎       | 93833/400000 [00:10<00:35, 8645.74it/s] 24%|██▎       | 94743/400000 [00:10<00:34, 8776.21it/s] 24%|██▍       | 95640/400000 [00:10<00:34, 8832.41it/s] 24%|██▍       | 96538/400000 [00:10<00:34, 8876.03it/s] 24%|██▍       | 97427/400000 [00:11<00:34, 8840.60it/s] 25%|██▍       | 98332/400000 [00:11<00:33, 8899.61it/s] 25%|██▍       | 99232/400000 [00:11<00:33, 8928.45it/s] 25%|██▌       | 100126/400000 [00:11<00:33, 8850.13it/s] 25%|██▌       | 101012/400000 [00:11<00:34, 8786.41it/s] 25%|██▌       | 101892/400000 [00:11<00:34, 8665.92it/s] 26%|██▌       | 102760/400000 [00:11<00:35, 8477.53it/s] 26%|██▌       | 103656/400000 [00:11<00:34, 8615.25it/s] 26%|██▌       | 104573/400000 [00:11<00:33, 8772.56it/s] 26%|██▋       | 105466/400000 [00:11<00:33, 8817.26it/s] 27%|██▋       | 106356/400000 [00:12<00:33, 8841.25it/s] 27%|██▋       | 107280/400000 [00:12<00:32, 8954.87it/s] 27%|██▋       | 108200/400000 [00:12<00:32, 9024.12it/s] 27%|██▋       | 109143/400000 [00:12<00:31, 9141.56it/s] 28%|██▊       | 110059/400000 [00:12<00:31, 9137.10it/s] 28%|██▊       | 110974/400000 [00:12<00:31, 9136.24it/s] 28%|██▊       | 111907/400000 [00:12<00:31, 9193.34it/s] 28%|██▊       | 112833/400000 [00:12<00:31, 9213.01it/s] 28%|██▊       | 113781/400000 [00:12<00:30, 9289.71it/s] 29%|██▊       | 114711/400000 [00:12<00:30, 9258.39it/s] 29%|██▉       | 115638/400000 [00:13<00:31, 9116.14it/s] 29%|██▉       | 116551/400000 [00:13<00:31, 8944.36it/s] 29%|██▉       | 117466/400000 [00:13<00:31, 9002.61it/s] 30%|██▉       | 118368/400000 [00:13<00:32, 8761.39it/s] 30%|██▉       | 119271/400000 [00:13<00:31, 8839.08it/s] 30%|███       | 120157/400000 [00:13<00:31, 8751.42it/s] 30%|███       | 121071/400000 [00:13<00:31, 8864.25it/s] 30%|███       | 121992/400000 [00:13<00:31, 8964.73it/s] 31%|███       | 122924/400000 [00:13<00:30, 9067.59it/s] 31%|███       | 123839/400000 [00:14<00:30, 9092.09it/s] 31%|███       | 124749/400000 [00:14<00:30, 9045.73it/s] 31%|███▏      | 125685/400000 [00:14<00:30, 9134.96it/s] 32%|███▏      | 126605/400000 [00:14<00:29, 9150.75it/s] 32%|███▏      | 127521/400000 [00:14<00:29, 9085.42it/s] 32%|███▏      | 128450/400000 [00:14<00:29, 9144.35it/s] 32%|███▏      | 129381/400000 [00:14<00:29, 9190.46it/s] 33%|███▎      | 130301/400000 [00:14<00:30, 8967.76it/s] 33%|███▎      | 131200/400000 [00:14<00:30, 8834.99it/s] 33%|███▎      | 132085/400000 [00:14<00:31, 8629.39it/s] 33%|███▎      | 133011/400000 [00:15<00:30, 8808.05it/s] 33%|███▎      | 133895/400000 [00:15<00:30, 8803.54it/s] 34%|███▎      | 134806/400000 [00:15<00:29, 8891.60it/s] 34%|███▍      | 135733/400000 [00:15<00:29, 8999.59it/s] 34%|███▍      | 136635/400000 [00:15<00:29, 8897.65it/s] 34%|███▍      | 137526/400000 [00:15<00:29, 8896.43it/s] 35%|███▍      | 138452/400000 [00:15<00:29, 8995.02it/s] 35%|███▍      | 139353/400000 [00:15<00:29, 8818.82it/s] 35%|███▌      | 140241/400000 [00:15<00:29, 8836.90it/s] 35%|███▌      | 141126/400000 [00:15<00:29, 8746.64it/s] 36%|███▌      | 142032/400000 [00:16<00:29, 8836.58it/s] 36%|███▌      | 142934/400000 [00:16<00:28, 8890.69it/s] 36%|███▌      | 143854/400000 [00:16<00:28, 8978.24it/s] 36%|███▌      | 144786/400000 [00:16<00:28, 9076.81it/s] 36%|███▋      | 145695/400000 [00:16<00:28, 9007.22it/s] 37%|███▋      | 146603/400000 [00:16<00:28, 9027.82it/s] 37%|███▋      | 147507/400000 [00:16<00:27, 9021.94it/s] 37%|███▋      | 148430/400000 [00:16<00:27, 9081.90it/s] 37%|███▋      | 149354/400000 [00:16<00:27, 9128.59it/s] 38%|███▊      | 150287/400000 [00:16<00:27, 9187.63it/s] 38%|███▊      | 151221/400000 [00:17<00:26, 9231.38it/s] 38%|███▊      | 152145/400000 [00:17<00:27, 9091.51it/s] 38%|███▊      | 153077/400000 [00:17<00:26, 9156.76it/s] 39%|███▊      | 154013/400000 [00:17<00:26, 9216.21it/s] 39%|███▊      | 154936/400000 [00:17<00:26, 9213.80it/s] 39%|███▉      | 155860/400000 [00:17<00:26, 9220.76it/s] 39%|███▉      | 156783/400000 [00:17<00:26, 9047.73it/s] 39%|███▉      | 157689/400000 [00:17<00:26, 9036.60it/s] 40%|███▉      | 158594/400000 [00:17<00:26, 9037.69it/s] 40%|███▉      | 159507/400000 [00:17<00:26, 9064.97it/s] 40%|████      | 160434/400000 [00:18<00:26, 9122.44it/s] 40%|████      | 161347/400000 [00:18<00:26, 8913.80it/s] 41%|████      | 162240/400000 [00:18<00:26, 8851.73it/s] 41%|████      | 163172/400000 [00:18<00:26, 8986.96it/s] 41%|████      | 164097/400000 [00:18<00:26, 9063.23it/s] 41%|████▏     | 165033/400000 [00:18<00:25, 9148.76it/s] 41%|████▏     | 165949/400000 [00:18<00:25, 9093.45it/s] 42%|████▏     | 166860/400000 [00:18<00:26, 8930.93it/s] 42%|████▏     | 167792/400000 [00:18<00:25, 9044.14it/s] 42%|████▏     | 168714/400000 [00:18<00:25, 9094.67it/s] 42%|████▏     | 169638/400000 [00:19<00:25, 9135.16it/s] 43%|████▎     | 170553/400000 [00:19<00:25, 9084.04it/s] 43%|████▎     | 171462/400000 [00:19<00:25, 9028.57it/s] 43%|████▎     | 172366/400000 [00:19<00:25, 8845.93it/s] 43%|████▎     | 173262/400000 [00:19<00:25, 8878.73it/s] 44%|████▎     | 174180/400000 [00:19<00:25, 8966.36it/s] 44%|████▍     | 175078/400000 [00:19<00:25, 8914.06it/s] 44%|████▍     | 175971/400000 [00:19<00:25, 8870.78it/s] 44%|████▍     | 176860/400000 [00:19<00:25, 8875.71it/s] 44%|████▍     | 177777/400000 [00:19<00:24, 8959.60it/s] 45%|████▍     | 178674/400000 [00:20<00:24, 8941.07it/s] 45%|████▍     | 179569/400000 [00:20<00:24, 8923.18it/s] 45%|████▌     | 180462/400000 [00:20<00:24, 8904.67it/s] 45%|████▌     | 181389/400000 [00:20<00:24, 9009.49it/s] 46%|████▌     | 182305/400000 [00:20<00:24, 9052.28it/s] 46%|████▌     | 183239/400000 [00:20<00:23, 9134.46it/s] 46%|████▌     | 184153/400000 [00:20<00:23, 9022.29it/s] 46%|████▋     | 185056/400000 [00:20<00:24, 8761.66it/s] 46%|████▋     | 185987/400000 [00:20<00:23, 8919.21it/s] 47%|████▋     | 186882/400000 [00:21<00:23, 8908.91it/s] 47%|████▋     | 187787/400000 [00:21<00:23, 8949.19it/s] 47%|████▋     | 188684/400000 [00:21<00:24, 8700.16it/s] 47%|████▋     | 189584/400000 [00:21<00:23, 8785.66it/s] 48%|████▊     | 190491/400000 [00:21<00:23, 8866.55it/s] 48%|████▊     | 191380/400000 [00:21<00:23, 8802.78it/s] 48%|████▊     | 192301/400000 [00:21<00:23, 8920.38it/s] 48%|████▊     | 193195/400000 [00:21<00:23, 8748.23it/s] 49%|████▊     | 194101/400000 [00:21<00:23, 8839.16it/s] 49%|████▉     | 195026/400000 [00:21<00:22, 8956.26it/s] 49%|████▉     | 195958/400000 [00:22<00:22, 9060.86it/s] 49%|████▉     | 196885/400000 [00:22<00:22, 9120.98it/s] 49%|████▉     | 197799/400000 [00:22<00:22, 9056.67it/s] 50%|████▉     | 198719/400000 [00:22<00:22, 9098.11it/s] 50%|████▉     | 199632/400000 [00:22<00:22, 9107.23it/s] 50%|█████     | 200544/400000 [00:22<00:22, 9033.82it/s] 50%|█████     | 201459/400000 [00:22<00:21, 9067.21it/s] 51%|█████     | 202367/400000 [00:22<00:22, 8885.20it/s] 51%|█████     | 203271/400000 [00:22<00:22, 8929.90it/s] 51%|█████     | 204178/400000 [00:22<00:21, 8970.06it/s] 51%|█████▏    | 205076/400000 [00:23<00:21, 8951.09it/s] 51%|█████▏    | 205972/400000 [00:23<00:21, 8857.10it/s] 52%|█████▏    | 206859/400000 [00:23<00:22, 8764.54it/s] 52%|█████▏    | 207737/400000 [00:23<00:22, 8591.15it/s] 52%|█████▏    | 208645/400000 [00:23<00:21, 8731.66it/s] 52%|█████▏    | 209532/400000 [00:23<00:21, 8772.47it/s] 53%|█████▎    | 210429/400000 [00:23<00:21, 8829.70it/s] 53%|█████▎    | 211313/400000 [00:23<00:21, 8801.28it/s] 53%|█████▎    | 212233/400000 [00:23<00:21, 8915.88it/s] 53%|█████▎    | 213163/400000 [00:23<00:20, 9027.51it/s] 54%|█████▎    | 214098/400000 [00:24<00:20, 9121.80it/s] 54%|█████▍    | 215032/400000 [00:24<00:20, 9183.17it/s] 54%|█████▍    | 215952/400000 [00:24<00:20, 9145.39it/s] 54%|█████▍    | 216882/400000 [00:24<00:19, 9190.02it/s] 54%|█████▍    | 217814/400000 [00:24<00:19, 9226.64it/s] 55%|█████▍    | 218756/400000 [00:24<00:19, 9281.36it/s] 55%|█████▍    | 219685/400000 [00:24<00:19, 9248.99it/s] 55%|█████▌    | 220611/400000 [00:24<00:19, 9180.61it/s] 55%|█████▌    | 221530/400000 [00:24<00:20, 8906.20it/s] 56%|█████▌    | 222423/400000 [00:24<00:20, 8757.87it/s] 56%|█████▌    | 223331/400000 [00:25<00:19, 8851.94it/s] 56%|█████▌    | 224218/400000 [00:25<00:20, 8596.55it/s] 56%|█████▋    | 225088/400000 [00:25<00:20, 8624.34it/s] 56%|█████▋    | 225972/400000 [00:25<00:20, 8685.57it/s] 57%|█████▋    | 226877/400000 [00:25<00:19, 8789.91it/s] 57%|█████▋    | 227758/400000 [00:25<00:20, 8518.53it/s] 57%|█████▋    | 228642/400000 [00:25<00:19, 8610.04it/s] 57%|█████▋    | 229508/400000 [00:25<00:19, 8623.78it/s] 58%|█████▊    | 230375/400000 [00:25<00:19, 8635.32it/s] 58%|█████▊    | 231312/400000 [00:26<00:19, 8841.38it/s] 58%|█████▊    | 232233/400000 [00:26<00:18, 8948.57it/s] 58%|█████▊    | 233159/400000 [00:26<00:18, 9038.08it/s] 59%|█████▊    | 234065/400000 [00:26<00:18, 9021.70it/s] 59%|█████▊    | 234972/400000 [00:26<00:18, 9034.78it/s] 59%|█████▉    | 235895/400000 [00:26<00:18, 9089.94it/s] 59%|█████▉    | 236818/400000 [00:26<00:17, 9131.45it/s] 59%|█████▉    | 237732/400000 [00:26<00:18, 9007.27it/s] 60%|█████▉    | 238634/400000 [00:26<00:18, 8691.70it/s] 60%|█████▉    | 239507/400000 [00:26<00:18, 8605.23it/s] 60%|██████    | 240415/400000 [00:27<00:18, 8739.97it/s] 60%|██████    | 241324/400000 [00:27<00:17, 8841.93it/s] 61%|██████    | 242210/400000 [00:27<00:17, 8796.68it/s] 61%|██████    | 243095/400000 [00:27<00:17, 8811.17it/s] 61%|██████    | 243991/400000 [00:27<00:17, 8853.21it/s] 61%|██████    | 244877/400000 [00:27<00:17, 8702.72it/s] 61%|██████▏   | 245755/400000 [00:27<00:17, 8723.20it/s] 62%|██████▏   | 246632/400000 [00:27<00:17, 8735.77it/s] 62%|██████▏   | 247507/400000 [00:27<00:17, 8692.73it/s] 62%|██████▏   | 248378/400000 [00:27<00:17, 8695.55it/s] 62%|██████▏   | 249270/400000 [00:28<00:17, 8761.34it/s] 63%|██████▎   | 250150/400000 [00:28<00:17, 8771.46it/s] 63%|██████▎   | 251034/400000 [00:28<00:16, 8791.67it/s] 63%|██████▎   | 251914/400000 [00:28<00:17, 8647.57it/s] 63%|██████▎   | 252780/400000 [00:28<00:17, 8530.03it/s] 63%|██████▎   | 253669/400000 [00:28<00:16, 8634.20it/s] 64%|██████▎   | 254560/400000 [00:28<00:16, 8714.05it/s] 64%|██████▍   | 255436/400000 [00:28<00:16, 8726.24it/s] 64%|██████▍   | 256310/400000 [00:28<00:16, 8633.15it/s] 64%|██████▍   | 257191/400000 [00:28<00:16, 8683.00it/s] 65%|██████▍   | 258074/400000 [00:29<00:16, 8725.43it/s] 65%|██████▍   | 258956/400000 [00:29<00:16, 8751.50it/s] 65%|██████▍   | 259832/400000 [00:29<00:16, 8687.63it/s] 65%|██████▌   | 260708/400000 [00:29<00:15, 8706.85it/s] 65%|██████▌   | 261632/400000 [00:29<00:15, 8857.97it/s] 66%|██████▌   | 262560/400000 [00:29<00:15, 8978.15it/s] 66%|██████▌   | 263485/400000 [00:29<00:15, 9055.30it/s] 66%|██████▌   | 264392/400000 [00:29<00:15, 8971.60it/s] 66%|██████▋   | 265290/400000 [00:29<00:15, 8919.06it/s] 67%|██████▋   | 266204/400000 [00:29<00:14, 8983.58it/s] 67%|██████▋   | 267103/400000 [00:30<00:15, 8758.98it/s] 67%|██████▋   | 267982/400000 [00:30<00:15, 8766.35it/s] 67%|██████▋   | 268880/400000 [00:30<00:14, 8826.57it/s] 67%|██████▋   | 269764/400000 [00:30<00:14, 8812.87it/s] 68%|██████▊   | 270656/400000 [00:30<00:14, 8843.03it/s] 68%|██████▊   | 271541/400000 [00:30<00:15, 8494.75it/s] 68%|██████▊   | 272394/400000 [00:30<00:15, 8455.07it/s] 68%|██████▊   | 273290/400000 [00:30<00:14, 8598.89it/s] 69%|██████▊   | 274160/400000 [00:30<00:14, 8626.57it/s] 69%|██████▉   | 275025/400000 [00:30<00:14, 8623.49it/s] 69%|██████▉   | 275891/400000 [00:31<00:14, 8634.30it/s] 69%|██████▉   | 276764/400000 [00:31<00:14, 8660.17it/s] 69%|██████▉   | 277644/400000 [00:31<00:14, 8700.56it/s] 70%|██████▉   | 278515/400000 [00:31<00:14, 8578.08it/s] 70%|██████▉   | 279388/400000 [00:31<00:13, 8619.75it/s] 70%|███████   | 280275/400000 [00:31<00:13, 8691.31it/s] 70%|███████   | 281145/400000 [00:31<00:14, 8141.48it/s] 70%|███████   | 282000/400000 [00:31<00:14, 8258.95it/s] 71%|███████   | 282832/400000 [00:31<00:14, 8176.21it/s] 71%|███████   | 283654/400000 [00:32<00:14, 8073.62it/s] 71%|███████   | 284515/400000 [00:32<00:14, 8226.51it/s] 71%|███████▏  | 285358/400000 [00:32<00:13, 8284.33it/s] 72%|███████▏  | 286189/400000 [00:32<00:13, 8281.46it/s] 72%|███████▏  | 287055/400000 [00:32<00:13, 8390.56it/s] 72%|███████▏  | 287934/400000 [00:32<00:13, 8504.59it/s] 72%|███████▏  | 288815/400000 [00:32<00:12, 8590.22it/s] 72%|███████▏  | 289676/400000 [00:32<00:12, 8595.46it/s] 73%|███████▎  | 290537/400000 [00:32<00:12, 8579.78it/s] 73%|███████▎  | 291408/400000 [00:32<00:12, 8616.16it/s] 73%|███████▎  | 292301/400000 [00:33<00:12, 8705.38it/s] 73%|███████▎  | 293219/400000 [00:33<00:12, 8840.37it/s] 74%|███████▎  | 294104/400000 [00:33<00:12, 8469.21it/s] 74%|███████▎  | 294961/400000 [00:33<00:12, 8498.30it/s] 74%|███████▍  | 295814/400000 [00:33<00:12, 8499.66it/s] 74%|███████▍  | 296687/400000 [00:33<00:12, 8565.57it/s] 74%|███████▍  | 297554/400000 [00:33<00:11, 8564.76it/s] 75%|███████▍  | 298412/400000 [00:33<00:11, 8508.92it/s] 75%|███████▍  | 299264/400000 [00:33<00:11, 8404.19it/s] 75%|███████▌  | 300106/400000 [00:33<00:11, 8351.70it/s] 75%|███████▌  | 300981/400000 [00:34<00:11, 8465.81it/s] 75%|███████▌  | 301829/400000 [00:34<00:11, 8327.04it/s] 76%|███████▌  | 302669/400000 [00:34<00:11, 8347.64it/s] 76%|███████▌  | 303527/400000 [00:34<00:11, 8415.55it/s] 76%|███████▌  | 304370/400000 [00:34<00:11, 8386.38it/s] 76%|███████▋  | 305214/400000 [00:34<00:11, 8398.99it/s] 77%|███████▋  | 306055/400000 [00:34<00:11, 8115.30it/s] 77%|███████▋  | 306869/400000 [00:34<00:11, 8004.40it/s] 77%|███████▋  | 307707/400000 [00:34<00:11, 8112.05it/s] 77%|███████▋  | 308520/400000 [00:34<00:11, 8112.99it/s] 77%|███████▋  | 309370/400000 [00:35<00:11, 8224.10it/s] 78%|███████▊  | 310262/400000 [00:35<00:10, 8419.43it/s] 78%|███████▊  | 311125/400000 [00:35<00:10, 8480.97it/s] 78%|███████▊  | 311977/400000 [00:35<00:10, 8492.61it/s] 78%|███████▊  | 312828/400000 [00:35<00:10, 8234.61it/s] 78%|███████▊  | 313703/400000 [00:35<00:10, 8380.26it/s] 79%|███████▊  | 314544/400000 [00:35<00:10, 8347.87it/s] 79%|███████▉  | 315381/400000 [00:35<00:10, 7986.27it/s] 79%|███████▉  | 316248/400000 [00:35<00:10, 8177.56it/s] 79%|███████▉  | 317071/400000 [00:36<00:10, 7945.42it/s] 79%|███████▉  | 317932/400000 [00:36<00:10, 8133.53it/s] 80%|███████▉  | 318824/400000 [00:36<00:09, 8352.55it/s] 80%|███████▉  | 319664/400000 [00:36<00:09, 8345.62it/s] 80%|████████  | 320580/400000 [00:36<00:09, 8573.20it/s] 80%|████████  | 321450/400000 [00:36<00:09, 8609.49it/s] 81%|████████  | 322374/400000 [00:36<00:08, 8786.75it/s] 81%|████████  | 323256/400000 [00:36<00:08, 8716.63it/s] 81%|████████  | 324130/400000 [00:36<00:08, 8607.61it/s] 81%|████████  | 324993/400000 [00:36<00:09, 8310.89it/s] 81%|████████▏ | 325838/400000 [00:37<00:08, 8349.62it/s] 82%|████████▏ | 326681/400000 [00:37<00:08, 8371.90it/s] 82%|████████▏ | 327553/400000 [00:37<00:08, 8471.76it/s] 82%|████████▏ | 328442/400000 [00:37<00:08, 8592.43it/s] 82%|████████▏ | 329324/400000 [00:37<00:08, 8656.39it/s] 83%|████████▎ | 330191/400000 [00:37<00:08, 8649.28it/s] 83%|████████▎ | 331061/400000 [00:37<00:07, 8664.16it/s] 83%|████████▎ | 331928/400000 [00:37<00:07, 8644.23it/s] 83%|████████▎ | 332812/400000 [00:37<00:07, 8699.21it/s] 83%|████████▎ | 333683/400000 [00:37<00:07, 8417.60it/s] 84%|████████▎ | 334536/400000 [00:38<00:07, 8449.97it/s] 84%|████████▍ | 335387/400000 [00:38<00:07, 8465.73it/s] 84%|████████▍ | 336240/400000 [00:38<00:07, 8483.07it/s] 84%|████████▍ | 337090/400000 [00:38<00:07, 8482.43it/s] 84%|████████▍ | 337964/400000 [00:38<00:07, 8557.75it/s] 85%|████████▍ | 338821/400000 [00:38<00:07, 8540.63it/s] 85%|████████▍ | 339717/400000 [00:38<00:06, 8660.37it/s] 85%|████████▌ | 340607/400000 [00:38<00:06, 8728.51it/s] 85%|████████▌ | 341487/400000 [00:38<00:06, 8748.46it/s] 86%|████████▌ | 342398/400000 [00:38<00:06, 8851.80it/s] 86%|████████▌ | 343284/400000 [00:39<00:06, 8739.62it/s] 86%|████████▌ | 344159/400000 [00:39<00:06, 8711.34it/s] 86%|████████▋ | 345031/400000 [00:39<00:06, 8674.34it/s] 86%|████████▋ | 345899/400000 [00:39<00:06, 8458.43it/s] 87%|████████▋ | 346753/400000 [00:39<00:06, 8481.21it/s] 87%|████████▋ | 347603/400000 [00:39<00:06, 8432.95it/s] 87%|████████▋ | 348511/400000 [00:39<00:05, 8616.58it/s] 87%|████████▋ | 349402/400000 [00:39<00:05, 8702.41it/s] 88%|████████▊ | 350305/400000 [00:39<00:05, 8797.38it/s] 88%|████████▊ | 351224/400000 [00:39<00:05, 8910.30it/s] 88%|████████▊ | 352117/400000 [00:40<00:05, 8876.18it/s] 88%|████████▊ | 353030/400000 [00:40<00:05, 8950.69it/s] 88%|████████▊ | 353939/400000 [00:40<00:05, 8989.62it/s] 89%|████████▊ | 354845/400000 [00:40<00:05, 9009.12it/s] 89%|████████▉ | 355747/400000 [00:40<00:04, 8864.78it/s] 89%|████████▉ | 356635/400000 [00:40<00:04, 8774.51it/s] 89%|████████▉ | 357546/400000 [00:40<00:04, 8871.41it/s] 90%|████████▉ | 358453/400000 [00:40<00:04, 8929.55it/s] 90%|████████▉ | 359385/400000 [00:40<00:04, 9040.77it/s] 90%|█████████ | 360309/400000 [00:40<00:04, 9097.25it/s] 90%|█████████ | 361220/400000 [00:41<00:04, 9020.03it/s] 91%|█████████ | 362123/400000 [00:41<00:04, 8909.49it/s] 91%|█████████ | 363020/400000 [00:41<00:04, 8926.92it/s] 91%|█████████ | 363919/400000 [00:41<00:04, 8943.79it/s] 91%|█████████ | 364814/400000 [00:41<00:03, 8887.31it/s] 91%|█████████▏| 365704/400000 [00:41<00:03, 8844.39it/s] 92%|█████████▏| 366600/400000 [00:41<00:03, 8878.42it/s] 92%|█████████▏| 367489/400000 [00:41<00:03, 8830.75it/s] 92%|█████████▏| 368373/400000 [00:41<00:03, 8582.41it/s] 92%|█████████▏| 369250/400000 [00:42<00:03, 8635.83it/s] 93%|█████████▎| 370115/400000 [00:42<00:03, 8383.61it/s] 93%|█████████▎| 370956/400000 [00:42<00:03, 8358.79it/s] 93%|█████████▎| 371794/400000 [00:42<00:03, 8361.65it/s] 93%|█████████▎| 372694/400000 [00:42<00:03, 8542.13it/s] 93%|█████████▎| 373591/400000 [00:42<00:03, 8664.67it/s] 94%|█████████▎| 374485/400000 [00:42<00:02, 8743.61it/s] 94%|█████████▍| 375371/400000 [00:42<00:02, 8776.60it/s] 94%|█████████▍| 376252/400000 [00:42<00:02, 8784.65it/s] 94%|█████████▍| 377133/400000 [00:42<00:02, 8790.61it/s] 95%|█████████▍| 378013/400000 [00:43<00:02, 8542.41it/s] 95%|█████████▍| 378870/400000 [00:43<00:02, 8492.68it/s] 95%|█████████▍| 379731/400000 [00:43<00:02, 8526.07it/s] 95%|█████████▌| 380631/400000 [00:43<00:02, 8660.98it/s] 95%|█████████▌| 381509/400000 [00:43<00:02, 8694.72it/s] 96%|█████████▌| 382418/400000 [00:43<00:01, 8809.05it/s] 96%|█████████▌| 383300/400000 [00:43<00:01, 8695.96it/s] 96%|█████████▌| 384171/400000 [00:43<00:01, 8691.81it/s] 96%|█████████▋| 385068/400000 [00:43<00:01, 8771.82it/s] 96%|█████████▋| 385946/400000 [00:43<00:01, 8694.15it/s] 97%|█████████▋| 386838/400000 [00:44<00:01, 8758.68it/s] 97%|█████████▋| 387715/400000 [00:44<00:01, 8742.53it/s] 97%|█████████▋| 388590/400000 [00:44<00:01, 8539.69it/s] 97%|█████████▋| 389490/400000 [00:44<00:01, 8671.73it/s] 98%|█████████▊| 390377/400000 [00:44<00:01, 8728.22it/s] 98%|█████████▊| 391262/400000 [00:44<00:00, 8763.08it/s] 98%|█████████▊| 392140/400000 [00:44<00:00, 8744.94it/s] 98%|█████████▊| 393021/400000 [00:44<00:00, 8763.96it/s] 98%|█████████▊| 393898/400000 [00:44<00:00, 8763.73it/s] 99%|█████████▊| 394775/400000 [00:44<00:00, 8751.46it/s] 99%|█████████▉| 395651/400000 [00:45<00:00, 8752.78it/s] 99%|█████████▉| 396527/400000 [00:45<00:00, 8676.79it/s] 99%|█████████▉| 397395/400000 [00:45<00:00, 8668.17it/s]100%|█████████▉| 398262/400000 [00:45<00:00, 8509.40it/s]100%|█████████▉| 399114/400000 [00:45<00:00, 8399.40it/s]100%|█████████▉| 399967/400000 [00:45<00:00, 8436.31it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8778.68it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fca897d0860>, <torchtext.data.dataset.TabularDataset object at 0x7fca897d09b0>, <torchtext.vocab.Vocab object at 0x7fca897d08d0>), {}) 


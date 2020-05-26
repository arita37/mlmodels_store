
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/efc0e3fdb2ef28f9272e1435d71c61f048b27c99', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'efc0e3fdb2ef28f9272e1435d71c61f048b27c99', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/efc0e3fdb2ef28f9272e1435d71c61f048b27c99

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/efc0e3fdb2ef28f9272e1435d71c61f048b27c99

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/efc0e3fdb2ef28f9272e1435d71c61f048b27c99

 ************************************************************************************************************************

  ############Check model ################################ 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  
['/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet34_benchmark_mnist.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//matchzoo_models_new.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//model_list_CIFAR.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//model_list_KMNIST.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//keras_reuters_charcnn.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//armdn_dataloader.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet18_benchmark_mnist.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//textcnn.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//torchhub_cnn_dataloader.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet18_benchmark_FashionMNIST.json']





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

###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f2d4980b0d0>

 ######### postional parameteres :  ['out']

 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f2d4980b0d0>

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

###### load_callable_from_uri LOADED <function train_test_split at 0x7f2dba968ea0>

 ######### postional parameteres :  []

 ######### Execute : preprocessor_func <function train_test_split at 0x7f2dba968ea0>

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

###### load_callable_from_uri LOADED <function pickle_dump at 0x7f2d5182c1e0>

 ######### postional parameteres :  ['t']

 ######### Execute : preprocessor_func <function pickle_dump at 0x7f2d5182c1e0>
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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2d681d6488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2d681d6488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2d681d6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 500, in test_dataloader
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 322, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 75, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:06, 149457.75it/s] 75%|███████▌  | 7454720/9912422 [00:00<00:11, 213325.69it/s]9920512it [00:00, 43465307.54it/s]                           
0it [00:00, ?it/s]32768it [00:00, 532315.04it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 158406.60it/s]1654784it [00:00, 9887145.17it/s]                          
0it [00:00, ?it/s]8192it [00:00, 179245.44it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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
((<torch.utils.data.dataloader.DataLoader object at 0x7f2d5179cfd0>, <torch.utils.data.dataloader.DataLoader object at 0x7f2d48972400>), {})





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

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f2db3cb22f0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f2db3cb22f0>

  function with postional parmater data_info <function tf_dataset_download at 0x7f2db3cb22f0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:54,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:53,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:53,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:53,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:38,  4.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:38,  4.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:37,  4.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:37,  4.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:37,  4.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:36,  4.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:26,  5.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:26,  5.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:26,  5.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:26,  5.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:25,  5.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:25,  5.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:25,  5.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:18,  7.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:18,  7.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:18,  7.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:18,  7.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:18,  7.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:17,  7.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:17,  7.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:13, 10.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:13, 10.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:13, 10.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:12, 10.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:12, 10.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:12, 10.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:12, 10.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:09, 13.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:09, 13.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:09, 13.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:09, 13.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:09, 13.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:09, 13.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:09, 13.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:00<00:07, 17.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:07, 17.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:07, 17.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:06, 17.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:06, 17.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:06, 17.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:06, 17.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:05, 22.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:05, 22.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:05, 22.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:05, 22.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:05, 22.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:05, 22.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 22.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 26.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 26.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 26.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 26.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 26.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 26.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 26.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 26.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 26.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 33.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:02, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 41.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 41.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 41.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 41.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 41.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 41.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 41.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 41.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 41.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 41.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 41.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 50.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 50.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 58.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 65.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 65.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:00, 65.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:00, 65.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:00, 65.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:00, 65.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:00, 65.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:00, 65.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:00, 65.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 65.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 65.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 72.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 72.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 72.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 72.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 72.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 72.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 72.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 72.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 72.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 72.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 72.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 78.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 78.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 78.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 78.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 78.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 78.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 78.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 78.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 78.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 78.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 78.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 82.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 82.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 82.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 82.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 82.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 82.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 82.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 82.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 82.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 82.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 82.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 85.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 85.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 85.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 85.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 85.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 85.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 85.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 85.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 85.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 85.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 85.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 87.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 87.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 87.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 87.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 87.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 87.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 87.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 87.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 87.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 87.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 87.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 90.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 90.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 90.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 90.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 90.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 90.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 90.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 90.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.44s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.44s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 90.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.44s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 90.08 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.66s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.44s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 90.08 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.66s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.66s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.78 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.66s/ url]
0 examples [00:00, ? examples/s]2020-05-26 09:44:13.453037: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-26 09:44:13.464893: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-26 09:44:13.465130: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562a6a8532e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-26 09:44:13.465147: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
19 examples [00:00, 189.79 examples/s]139 examples [00:00, 253.90 examples/s]265 examples [00:00, 333.68 examples/s]375 examples [00:00, 421.48 examples/s]498 examples [00:00, 524.90 examples/s]619 examples [00:00, 631.80 examples/s]734 examples [00:00, 729.96 examples/s]847 examples [00:00, 815.81 examples/s]967 examples [00:00, 902.03 examples/s]1091 examples [00:01, 981.57 examples/s]1217 examples [00:01, 1049.95 examples/s]1336 examples [00:01, 1087.04 examples/s]1455 examples [00:01, 1114.84 examples/s]1574 examples [00:01, 1136.31 examples/s]1693 examples [00:01, 1125.95 examples/s]1814 examples [00:01, 1149.62 examples/s]1934 examples [00:01, 1163.28 examples/s]2055 examples [00:01, 1175.45 examples/s]2180 examples [00:01, 1195.90 examples/s]2307 examples [00:02, 1214.89 examples/s]2431 examples [00:02, 1221.37 examples/s]2556 examples [00:02, 1228.72 examples/s]2682 examples [00:02, 1237.68 examples/s]2810 examples [00:02, 1248.66 examples/s]2938 examples [00:02, 1255.07 examples/s]3064 examples [00:02, 1256.20 examples/s]3190 examples [00:02, 1230.39 examples/s]3314 examples [00:02, 1227.00 examples/s]3437 examples [00:02, 1197.55 examples/s]3559 examples [00:03, 1202.04 examples/s]3688 examples [00:03, 1224.91 examples/s]3813 examples [00:03, 1229.25 examples/s]3940 examples [00:03, 1239.72 examples/s]4065 examples [00:03, 1242.27 examples/s]4190 examples [00:03, 1226.28 examples/s]4313 examples [00:03, 1215.57 examples/s]4435 examples [00:03, 1214.95 examples/s]4557 examples [00:03, 1158.96 examples/s]4681 examples [00:03, 1181.21 examples/s]4801 examples [00:04, 1186.57 examples/s]4929 examples [00:04, 1213.11 examples/s]5051 examples [00:04, 1202.96 examples/s]5176 examples [00:04, 1216.46 examples/s]5298 examples [00:04, 1209.45 examples/s]5420 examples [00:04, 1186.15 examples/s]5539 examples [00:04, 1178.59 examples/s]5658 examples [00:04, 1178.77 examples/s]5783 examples [00:04, 1197.67 examples/s]5905 examples [00:04, 1201.95 examples/s]6026 examples [00:05, 1193.28 examples/s]6148 examples [00:05, 1199.41 examples/s]6269 examples [00:05, 1196.25 examples/s]6392 examples [00:05, 1205.01 examples/s]6517 examples [00:05, 1216.11 examples/s]6639 examples [00:05, 1216.53 examples/s]6764 examples [00:05, 1224.27 examples/s]6887 examples [00:05, 1220.59 examples/s]7010 examples [00:05, 1207.98 examples/s]7131 examples [00:05, 1201.87 examples/s]7259 examples [00:06, 1222.90 examples/s]7382 examples [00:06, 1193.96 examples/s]7502 examples [00:06, 1173.72 examples/s]7621 examples [00:06, 1176.64 examples/s]7748 examples [00:06, 1200.40 examples/s]7870 examples [00:06, 1205.24 examples/s]7991 examples [00:06, 1198.79 examples/s]8112 examples [00:06, 1156.69 examples/s]8231 examples [00:06, 1164.53 examples/s]8348 examples [00:07, 1163.11 examples/s]8473 examples [00:07, 1185.66 examples/s]8593 examples [00:07, 1187.54 examples/s]8712 examples [00:07, 1187.09 examples/s]8838 examples [00:07, 1207.54 examples/s]8959 examples [00:07, 1205.92 examples/s]9082 examples [00:07, 1212.19 examples/s]9207 examples [00:07, 1220.62 examples/s]9330 examples [00:07, 1218.64 examples/s]9454 examples [00:07, 1224.54 examples/s]9577 examples [00:08, 1183.17 examples/s]9697 examples [00:08, 1187.48 examples/s]9821 examples [00:08, 1200.79 examples/s]9942 examples [00:08, 1200.24 examples/s]10063 examples [00:08, 1160.84 examples/s]10188 examples [00:08, 1185.33 examples/s]10313 examples [00:08, 1202.01 examples/s]10434 examples [00:08, 1195.98 examples/s]10554 examples [00:08, 1196.41 examples/s]10675 examples [00:08, 1197.79 examples/s]10796 examples [00:09, 1199.05 examples/s]10926 examples [00:09, 1227.55 examples/s]11049 examples [00:09, 1216.01 examples/s]11171 examples [00:09, 1211.33 examples/s]11295 examples [00:09, 1219.10 examples/s]11422 examples [00:09, 1231.68 examples/s]11546 examples [00:09, 1210.44 examples/s]11668 examples [00:09, 1205.13 examples/s]11789 examples [00:09, 1194.33 examples/s]11909 examples [00:09, 1178.68 examples/s]12033 examples [00:10, 1194.75 examples/s]12161 examples [00:10, 1218.46 examples/s]12286 examples [00:10, 1226.64 examples/s]12409 examples [00:10, 1215.84 examples/s]12535 examples [00:10, 1226.96 examples/s]12658 examples [00:10, 1223.70 examples/s]12785 examples [00:10, 1236.77 examples/s]12909 examples [00:10, 1215.51 examples/s]13033 examples [00:10, 1220.20 examples/s]13156 examples [00:11, 1218.02 examples/s]13280 examples [00:11, 1223.56 examples/s]13409 examples [00:11, 1241.00 examples/s]13534 examples [00:11, 1235.58 examples/s]13658 examples [00:11, 1204.82 examples/s]13784 examples [00:11, 1219.36 examples/s]13908 examples [00:11, 1223.51 examples/s]14031 examples [00:11, 1221.38 examples/s]14161 examples [00:11, 1241.00 examples/s]14286 examples [00:11, 1221.08 examples/s]14413 examples [00:12, 1233.14 examples/s]14542 examples [00:12, 1247.11 examples/s]14669 examples [00:12, 1251.47 examples/s]14795 examples [00:12, 1252.91 examples/s]14921 examples [00:12, 1241.61 examples/s]15046 examples [00:12, 1211.25 examples/s]15170 examples [00:12, 1219.21 examples/s]15294 examples [00:12, 1223.75 examples/s]15417 examples [00:12, 1218.90 examples/s]15541 examples [00:12, 1224.09 examples/s]15664 examples [00:13, 1186.09 examples/s]15789 examples [00:13, 1201.60 examples/s]15917 examples [00:13, 1221.46 examples/s]16044 examples [00:13, 1234.11 examples/s]16168 examples [00:13, 1198.22 examples/s]16289 examples [00:13, 1195.28 examples/s]16409 examples [00:13, 1168.60 examples/s]16527 examples [00:13, 1159.38 examples/s]16644 examples [00:13, 1147.27 examples/s]16759 examples [00:14, 1075.15 examples/s]16877 examples [00:14, 1103.01 examples/s]16995 examples [00:14, 1124.30 examples/s]17109 examples [00:14, 1122.06 examples/s]17231 examples [00:14, 1149.33 examples/s]17358 examples [00:14, 1182.24 examples/s]17477 examples [00:14, 1182.05 examples/s]17603 examples [00:14, 1203.91 examples/s]17727 examples [00:14, 1214.20 examples/s]17849 examples [00:14, 1213.06 examples/s]17971 examples [00:15, 1194.55 examples/s]18091 examples [00:15, 1151.47 examples/s]18215 examples [00:15, 1176.54 examples/s]18342 examples [00:15, 1200.79 examples/s]18467 examples [00:15, 1212.65 examples/s]18592 examples [00:15, 1221.88 examples/s]18718 examples [00:15, 1230.81 examples/s]18842 examples [00:15, 1232.96 examples/s]18966 examples [00:15, 1234.56 examples/s]19092 examples [00:15, 1239.37 examples/s]19217 examples [00:16, 1221.29 examples/s]19340 examples [00:16, 1216.43 examples/s]19462 examples [00:16, 1210.50 examples/s]19586 examples [00:16, 1217.66 examples/s]19708 examples [00:16, 1213.03 examples/s]19833 examples [00:16, 1222.83 examples/s]19956 examples [00:16, 1219.09 examples/s]20078 examples [00:16, 1142.58 examples/s]20194 examples [00:16, 1143.48 examples/s]20316 examples [00:16, 1164.71 examples/s]20434 examples [00:17, 1165.73 examples/s]20562 examples [00:17, 1196.57 examples/s]20689 examples [00:17, 1217.37 examples/s]20815 examples [00:17, 1228.84 examples/s]20944 examples [00:17, 1245.29 examples/s]21069 examples [00:17, 1246.64 examples/s]21197 examples [00:17, 1254.17 examples/s]21325 examples [00:17, 1258.65 examples/s]21453 examples [00:17, 1264.24 examples/s]21580 examples [00:17, 1255.98 examples/s]21706 examples [00:18, 1254.05 examples/s]21836 examples [00:18, 1267.24 examples/s]21964 examples [00:18, 1270.81 examples/s]22092 examples [00:18, 1251.06 examples/s]22218 examples [00:18, 1235.59 examples/s]22344 examples [00:18, 1241.20 examples/s]22470 examples [00:18, 1243.80 examples/s]22596 examples [00:18, 1246.12 examples/s]22721 examples [00:18, 1225.60 examples/s]22844 examples [00:19, 1188.53 examples/s]22964 examples [00:19, 1138.04 examples/s]23079 examples [00:19, 1127.40 examples/s]23193 examples [00:19, 1102.69 examples/s]23309 examples [00:19, 1119.25 examples/s]23422 examples [00:19, 1119.83 examples/s]23538 examples [00:19, 1128.90 examples/s]23654 examples [00:19, 1134.91 examples/s]23775 examples [00:19, 1154.61 examples/s]23900 examples [00:19, 1180.02 examples/s]24020 examples [00:20, 1184.21 examples/s]24139 examples [00:20, 1169.31 examples/s]24265 examples [00:20, 1194.15 examples/s]24392 examples [00:20, 1214.80 examples/s]24517 examples [00:20, 1225.09 examples/s]24641 examples [00:20, 1227.31 examples/s]24764 examples [00:20, 1219.59 examples/s]24888 examples [00:20, 1223.68 examples/s]25011 examples [00:20, 1218.35 examples/s]25133 examples [00:20, 1214.47 examples/s]25255 examples [00:21, 1214.73 examples/s]25378 examples [00:21, 1218.36 examples/s]25508 examples [00:21, 1240.89 examples/s]25633 examples [00:21, 1232.97 examples/s]25757 examples [00:21, 1213.10 examples/s]25883 examples [00:21, 1225.27 examples/s]26007 examples [00:21, 1228.42 examples/s]26130 examples [00:21, 1192.61 examples/s]26250 examples [00:21, 1176.88 examples/s]26373 examples [00:21, 1191.30 examples/s]26495 examples [00:22, 1197.52 examples/s]26615 examples [00:22, 1178.64 examples/s]26739 examples [00:22, 1194.84 examples/s]26864 examples [00:22, 1210.09 examples/s]26994 examples [00:22, 1232.20 examples/s]27118 examples [00:22, 1201.32 examples/s]27242 examples [00:22, 1211.96 examples/s]27368 examples [00:22, 1225.39 examples/s]27494 examples [00:22, 1234.44 examples/s]27620 examples [00:22, 1240.35 examples/s]27745 examples [00:23, 1231.48 examples/s]27870 examples [00:23, 1235.52 examples/s]27997 examples [00:23, 1243.27 examples/s]28122 examples [00:23, 1234.15 examples/s]28246 examples [00:23, 1211.66 examples/s]28368 examples [00:23, 1202.91 examples/s]28489 examples [00:23, 1203.43 examples/s]28614 examples [00:23, 1215.12 examples/s]28736 examples [00:23, 1209.32 examples/s]28857 examples [00:24, 1202.37 examples/s]28980 examples [00:24, 1208.95 examples/s]29101 examples [00:24, 1157.21 examples/s]29225 examples [00:24, 1179.67 examples/s]29346 examples [00:24, 1186.08 examples/s]29474 examples [00:24, 1210.32 examples/s]29596 examples [00:24, 1212.19 examples/s]29718 examples [00:24, 1213.37 examples/s]29840 examples [00:24, 1194.71 examples/s]29960 examples [00:24, 1183.47 examples/s]30079 examples [00:25, 1138.56 examples/s]30194 examples [00:25, 1118.31 examples/s]30308 examples [00:25, 1124.42 examples/s]30433 examples [00:25, 1158.04 examples/s]30556 examples [00:25, 1177.12 examples/s]30681 examples [00:25, 1196.87 examples/s]30802 examples [00:25, 1191.36 examples/s]30924 examples [00:25, 1198.07 examples/s]31051 examples [00:25, 1215.05 examples/s]31173 examples [00:25, 1205.83 examples/s]31295 examples [00:26, 1207.59 examples/s]31416 examples [00:26, 1206.67 examples/s]31539 examples [00:26, 1213.17 examples/s]31663 examples [00:26, 1219.64 examples/s]31793 examples [00:26, 1240.41 examples/s]31918 examples [00:26, 1223.50 examples/s]32041 examples [00:26, 1193.39 examples/s]32161 examples [00:26, 1139.95 examples/s]32276 examples [00:26, 1129.22 examples/s]32394 examples [00:27, 1142.93 examples/s]32514 examples [00:27, 1157.31 examples/s]32636 examples [00:27, 1175.31 examples/s]32754 examples [00:27, 1160.31 examples/s]32871 examples [00:27, 1155.92 examples/s]32996 examples [00:27, 1181.65 examples/s]33120 examples [00:27, 1198.38 examples/s]33241 examples [00:27, 1179.64 examples/s]33368 examples [00:27, 1204.45 examples/s]33494 examples [00:27, 1217.77 examples/s]33617 examples [00:28, 1221.38 examples/s]33746 examples [00:28, 1239.46 examples/s]33871 examples [00:28, 1236.05 examples/s]33996 examples [00:28, 1238.39 examples/s]34120 examples [00:28, 1214.26 examples/s]34242 examples [00:28, 1206.09 examples/s]34365 examples [00:28, 1210.30 examples/s]34489 examples [00:28, 1216.44 examples/s]34615 examples [00:28, 1227.83 examples/s]34738 examples [00:28, 1191.79 examples/s]34858 examples [00:29, 1169.45 examples/s]34981 examples [00:29, 1186.15 examples/s]35100 examples [00:29, 1172.70 examples/s]35224 examples [00:29, 1191.81 examples/s]35344 examples [00:29, 1185.42 examples/s]35469 examples [00:29, 1203.31 examples/s]35594 examples [00:29, 1216.52 examples/s]35716 examples [00:29, 1217.18 examples/s]35838 examples [00:29, 1216.35 examples/s]35960 examples [00:29, 1166.77 examples/s]36078 examples [00:30, 1125.03 examples/s]36192 examples [00:30, 1112.37 examples/s]36307 examples [00:30, 1123.17 examples/s]36431 examples [00:30, 1153.61 examples/s]36560 examples [00:30, 1189.63 examples/s]36686 examples [00:30, 1207.95 examples/s]36808 examples [00:30, 1181.18 examples/s]36931 examples [00:30, 1193.69 examples/s]37051 examples [00:30, 1192.19 examples/s]37175 examples [00:31, 1205.24 examples/s]37296 examples [00:31, 1186.55 examples/s]37418 examples [00:31, 1193.91 examples/s]37540 examples [00:31, 1201.50 examples/s]37666 examples [00:31, 1217.63 examples/s]37798 examples [00:31, 1244.11 examples/s]37925 examples [00:31, 1251.19 examples/s]38051 examples [00:31, 1223.87 examples/s]38174 examples [00:31, 1199.14 examples/s]38295 examples [00:31, 1177.41 examples/s]38421 examples [00:32, 1200.69 examples/s]38542 examples [00:32, 1184.69 examples/s]38661 examples [00:32, 1177.87 examples/s]38779 examples [00:32, 1164.53 examples/s]38901 examples [00:32, 1180.12 examples/s]39027 examples [00:32, 1200.75 examples/s]39148 examples [00:32, 1190.97 examples/s]39270 examples [00:32, 1197.26 examples/s]39392 examples [00:32, 1203.87 examples/s]39515 examples [00:32, 1210.66 examples/s]39639 examples [00:33, 1217.63 examples/s]39767 examples [00:33, 1233.36 examples/s]39891 examples [00:33, 1234.60 examples/s]40015 examples [00:34, 325.68 examples/s] 40137 examples [00:34, 417.16 examples/s]40260 examples [00:34, 520.29 examples/s]40381 examples [00:34, 627.40 examples/s]40503 examples [00:34, 733.61 examples/s]40625 examples [00:34, 831.86 examples/s]40748 examples [00:34, 921.29 examples/s]40867 examples [00:35, 973.08 examples/s]40984 examples [00:35, 1010.44 examples/s]41101 examples [00:35, 1052.84 examples/s]41217 examples [00:35, 1080.41 examples/s]41333 examples [00:35, 1098.52 examples/s]41455 examples [00:35, 1129.15 examples/s]41579 examples [00:35, 1160.06 examples/s]41698 examples [00:35, 1167.68 examples/s]41817 examples [00:35, 1170.21 examples/s]41940 examples [00:35, 1186.12 examples/s]42062 examples [00:36, 1195.01 examples/s]42190 examples [00:36, 1217.64 examples/s]42319 examples [00:36, 1235.78 examples/s]42444 examples [00:36, 1213.76 examples/s]42573 examples [00:36, 1234.40 examples/s]42704 examples [00:36, 1255.46 examples/s]42831 examples [00:36, 1257.07 examples/s]42964 examples [00:36, 1276.70 examples/s]43092 examples [00:36, 1252.27 examples/s]43218 examples [00:36, 1207.09 examples/s]43340 examples [00:37, 1205.25 examples/s]43469 examples [00:37, 1229.35 examples/s]43595 examples [00:37, 1238.06 examples/s]43720 examples [00:37, 1233.25 examples/s]43849 examples [00:37, 1247.05 examples/s]43976 examples [00:37, 1252.93 examples/s]44102 examples [00:37, 1254.10 examples/s]44231 examples [00:37, 1263.72 examples/s]44358 examples [00:37, 1256.83 examples/s]44484 examples [00:37, 1253.62 examples/s]44614 examples [00:38, 1265.26 examples/s]44741 examples [00:38, 1265.31 examples/s]44868 examples [00:38, 1264.51 examples/s]44995 examples [00:38, 1253.12 examples/s]45121 examples [00:38, 1203.23 examples/s]45242 examples [00:38, 1175.27 examples/s]45360 examples [00:38, 1170.71 examples/s]45482 examples [00:38, 1184.26 examples/s]45602 examples [00:38, 1187.42 examples/s]45721 examples [00:38, 1176.35 examples/s]45845 examples [00:39, 1192.39 examples/s]45965 examples [00:39, 1179.31 examples/s]46084 examples [00:39, 1159.52 examples/s]46201 examples [00:39, 1143.95 examples/s]46317 examples [00:39, 1146.57 examples/s]46433 examples [00:39, 1147.60 examples/s]46548 examples [00:39, 1143.46 examples/s]46663 examples [00:39, 1139.99 examples/s]46778 examples [00:39, 1134.28 examples/s]46904 examples [00:40, 1166.98 examples/s]47028 examples [00:40, 1184.14 examples/s]47147 examples [00:40, 1180.32 examples/s]47266 examples [00:40, 1179.29 examples/s]47386 examples [00:40, 1181.88 examples/s]47511 examples [00:40, 1199.02 examples/s]47635 examples [00:40, 1210.49 examples/s]47757 examples [00:40, 1204.59 examples/s]47882 examples [00:40, 1214.98 examples/s]48004 examples [00:40, 1209.20 examples/s]48125 examples [00:41, 1204.16 examples/s]48251 examples [00:41, 1217.98 examples/s]48377 examples [00:41, 1229.18 examples/s]48503 examples [00:41, 1236.29 examples/s]48627 examples [00:41, 1221.52 examples/s]48750 examples [00:41, 1202.07 examples/s]48871 examples [00:41, 1191.03 examples/s]48993 examples [00:41, 1197.45 examples/s]49113 examples [00:41, 1183.41 examples/s]49232 examples [00:41, 1173.72 examples/s]49354 examples [00:42, 1185.00 examples/s]49479 examples [00:42, 1201.23 examples/s]49606 examples [00:42, 1219.90 examples/s]49732 examples [00:42, 1230.87 examples/s]49856 examples [00:42, 1223.06 examples/s]49984 examples [00:42, 1238.99 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 9065/50000 [00:00<00:00, 90644.25 examples/s] 49%|████▉     | 24564/50000 [00:00<00:00, 103539.10 examples/s] 80%|███████▉  | 39997/50000 [00:00<00:00, 114881.15 examples/s]                                                                0 examples [00:00, ? examples/s]100 examples [00:00, 996.92 examples/s]229 examples [00:00, 1069.61 examples/s]352 examples [00:00, 1110.86 examples/s]466 examples [00:00, 1119.09 examples/s]577 examples [00:00, 1113.79 examples/s]696 examples [00:00, 1133.44 examples/s]809 examples [00:00, 1129.39 examples/s]922 examples [00:00, 1128.96 examples/s]1047 examples [00:00, 1162.40 examples/s]1162 examples [00:01, 1156.56 examples/s]1287 examples [00:01, 1181.46 examples/s]1404 examples [00:01, 942.85 examples/s] 1525 examples [00:01, 1007.72 examples/s]1634 examples [00:01, 1030.57 examples/s]1747 examples [00:01, 1056.94 examples/s]1856 examples [00:01, 1062.19 examples/s]1983 examples [00:01, 1115.54 examples/s]2103 examples [00:01, 1136.26 examples/s]2220 examples [00:01, 1144.19 examples/s]2336 examples [00:02, 1124.64 examples/s]2450 examples [00:02, 1124.73 examples/s]2569 examples [00:02, 1141.34 examples/s]2694 examples [00:02, 1169.34 examples/s]2812 examples [00:02, 1168.82 examples/s]2931 examples [00:02, 1172.61 examples/s]3053 examples [00:02, 1185.71 examples/s]3183 examples [00:02, 1216.47 examples/s]3310 examples [00:02, 1230.58 examples/s]3436 examples [00:03, 1237.57 examples/s]3560 examples [00:03, 1215.76 examples/s]3682 examples [00:03, 1208.56 examples/s]3804 examples [00:03, 1198.65 examples/s]3926 examples [00:03, 1204.29 examples/s]4054 examples [00:03, 1224.54 examples/s]4178 examples [00:03, 1227.66 examples/s]4301 examples [00:03, 1191.59 examples/s]4422 examples [00:03, 1196.05 examples/s]4542 examples [00:03, 1173.86 examples/s]4660 examples [00:04, 1155.68 examples/s]4788 examples [00:04, 1189.87 examples/s]4911 examples [00:04, 1198.58 examples/s]5032 examples [00:04, 1197.42 examples/s]5152 examples [00:04, 1174.94 examples/s]5270 examples [00:04, 1126.46 examples/s]5384 examples [00:04, 1102.03 examples/s]5495 examples [00:04, 1060.95 examples/s]5615 examples [00:04, 1098.11 examples/s]5726 examples [00:04, 1093.90 examples/s]5845 examples [00:05, 1120.84 examples/s]5972 examples [00:05, 1161.46 examples/s]6098 examples [00:05, 1187.23 examples/s]6218 examples [00:05, 1183.78 examples/s]6337 examples [00:05, 1158.77 examples/s]6454 examples [00:05, 1152.20 examples/s]6576 examples [00:05, 1170.17 examples/s]6698 examples [00:05, 1184.66 examples/s]6817 examples [00:05, 1174.44 examples/s]6936 examples [00:06, 1166.87 examples/s]7053 examples [00:06, 1141.89 examples/s]7176 examples [00:06, 1164.72 examples/s]7304 examples [00:06, 1195.58 examples/s]7432 examples [00:06, 1216.97 examples/s]7555 examples [00:06, 1207.66 examples/s]7679 examples [00:06, 1215.19 examples/s]7806 examples [00:06, 1230.40 examples/s]7930 examples [00:06, 1232.25 examples/s]8057 examples [00:06, 1241.16 examples/s]8182 examples [00:07, 1239.37 examples/s]8307 examples [00:07, 1233.37 examples/s]8431 examples [00:07, 1199.51 examples/s]8553 examples [00:07, 1204.87 examples/s]8674 examples [00:07, 1202.50 examples/s]8795 examples [00:07, 1190.82 examples/s]8917 examples [00:07, 1197.12 examples/s]9040 examples [00:07, 1206.55 examples/s]9161 examples [00:07, 1170.17 examples/s]9284 examples [00:07, 1187.44 examples/s]9404 examples [00:08, 1178.69 examples/s]9523 examples [00:08, 1181.90 examples/s]9645 examples [00:08, 1190.46 examples/s]9771 examples [00:08, 1208.21 examples/s]9892 examples [00:08, 1208.52 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteI65V7Z/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteI65V7Z/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2d681d6488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2d681d6488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2d681d6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

 #####  get_Data DataLoader 
((<torch.utils.data.dataloader.DataLoader object at 0x7f2db3ba5240>, <torch.utils.data.dataloader.DataLoader object at 0x7f2d48cee080>), {})





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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2d681d6488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2d681d6488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2d681d6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

 #####  get_Data DataLoader 
((<torch.utils.data.dataloader.DataLoader object at 0x7f2d48cfaac8>, <torch.utils.data.dataloader.DataLoader object at 0x7f2d48cb4978>), {})





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

###### load_callable_from_uri LOADED <function split_train_valid at 0x7f2d3fafc488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function split_train_valid at 0x7f2d3fafc488>

  function with postional parmater data_info <function split_train_valid at 0x7f2d3fafc488> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f2d3fafc598>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f2d3fafc598>

  function with postional parmater data_info <function create_tabular_dataset at 0x7f2d3fafc598> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=855f3dafe82a2b195f101e3d150e992ca5f0b4831fa0b816a6af5e142c0856c8
  Stored in directory: /tmp/pip-ephem-wheel-cache-ciin873a/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<17:37:13, 13.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<12:34:01, 19.1kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<8:51:03, 27.1kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:12:17, 38.6kB/s].vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<4:19:57, 55.0kB/s].vector_cache/glove.6B.zip:   1%|          | 7.41M/862M [00:01<3:01:17, 78.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<2:06:16, 112kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 16.0M/862M [00:01<1:28:06, 160kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.6M/862M [00:01<1:01:25, 228kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.2M/862M [00:01<42:56, 325kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 28.9M/862M [00:01<29:58, 463kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:01<21:00, 658kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.0M/862M [00:01<14:43, 934kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.0M/862M [00:02<10:21, 1.32MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.3M/862M [00:02<07:18, 1.86MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.6M/862M [00:02<05:11, 2.61MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:02<04:30, 2.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<05:03, 2.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:04<05:53, 2.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<04:41, 2.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<05:34, 2.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<05:34, 2.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<04:15, 3.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:07<03:09, 4.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<18:26, 720kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:08<14:34, 912kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:09<10:36, 1.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<10:02, 1.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<10:25, 1.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:11<08:07, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:11<05:49, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<11:34, 1.14MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:12<09:26, 1.39MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:13<06:55, 1.89MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<07:56, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<06:51, 1.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:14<05:07, 2.54MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:39, 1.95MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<05:59, 2.17MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:16<04:27, 2.91MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<06:12, 2.08MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<05:39, 2.28MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:18<04:14, 3.05MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<06:01, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<05:31, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:20<04:11, 3.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<05:57, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<06:46, 1.89MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<05:17, 2.42MB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:22<03:51, 3.30MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<09:33, 1.33MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<07:47, 1.64MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:24<05:45, 2.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:01, 1.80MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:12, 2.04MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:36, 2.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:13, 2.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:39, 2.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:16, 2.94MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:56, 2.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:28, 2.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:08, 3.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:44, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:33, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:06, 2.43MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<03:43, 3.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<11:26, 1.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<09:17, 1.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<06:49, 1.81MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:35, 1.62MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:49, 1.57MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<06:06, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:14, 1.96MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:24, 2.27MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<04:02, 3.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<02:59, 4.08MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<51:13, 238kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<37:05, 328kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<26:10, 464kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<21:06, 574kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<17:21, 698kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<12:40, 955kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<08:59, 1.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<13:04, 921kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<10:24, 1.16MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<07:34, 1.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:06, 1.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:07, 1.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:13, 1.92MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<04:27, 2.67MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<21:58, 542kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<16:35, 718kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<11:53, 999kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<11:04, 1.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<10:12, 1.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<07:37, 1.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<05:31, 2.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:58, 1.48MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:46, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:59, 2.36MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:14, 1.88MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:33, 2.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<04:08, 2.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:39, 2.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:09, 2.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:53, 2.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:27, 2.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<04:59, 2.31MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:46, 3.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:21, 2.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<04:55, 2.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:43, 3.07MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:18, 2.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:02, 1.89MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<04:43, 2.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:27, 3.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<08:24, 1.35MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<07:04, 1.61MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:13, 2.17MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:16, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<06:41, 1.69MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<05:10, 2.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<03:45, 2.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<09:45, 1.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<07:58, 1.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<05:51, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:42, 1.67MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:57, 1.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<05:26, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:35, 1.98MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:03, 2.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<03:49, 2.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:15, 2.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:47, 2.30MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<03:35, 3.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:04, 2.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:40, 2.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<03:30, 3.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:02, 2.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:40, 2.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<03:29, 3.11MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:58, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:41, 1.90MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:31, 2.38MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<03:17, 3.27MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<10:19:33, 17.3kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<7:14:32, 24.7kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<5:03:44, 35.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<3:34:22, 49.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<2:32:11, 70.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<1:46:52, 99.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<1:14:44, 142kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<56:40, 187kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<40:35, 261kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<28:34, 370kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:25<20:04, 525kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<43:00, 245kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<31:09, 338kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<22:01, 477kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<17:51, 587kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<14:37, 716kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<10:45, 973kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<09:10, 1.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<07:29, 1.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<05:30, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:16, 1.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:29, 1.59MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:59, 2.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:37, 2.84MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<08:15, 1.24MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:49, 1.50MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<04:59, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:53, 1.73MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:09, 1.98MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:49, 2.66MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<05:05, 1.99MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:25, 2.29MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:21, 3.01MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<04:43, 2.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:20, 1.88MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:14, 2.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:34, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<04:14, 2.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:11, 3.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:32, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:11, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:08, 2.40MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:29, 2.20MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:08, 2.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:06, 3.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:28, 2.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:50, 2.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:53, 2.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<02:51, 3.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:47, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:03, 1.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:44, 2.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:53, 1.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:24, 2.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:19, 2.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:35, 2.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<05:05, 1.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<03:58, 2.41MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<02:57, 3.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:53, 1.94MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:25, 2.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:20, 2.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:31, 2.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:07, 2.28MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<03:07, 3.01MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:23, 2.13MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:53, 2.40MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:00<02:57, 3.15MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:16, 2.17MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:53, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:49, 2.43MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<02:47, 3.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:55, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:39, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<04:16, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:04<03:06, 2.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<24:51, 369kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<18:20, 499kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<13:00, 703kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<11:13, 811kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<09:41, 938kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<07:10, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<05:09, 1.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:49, 1.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<05:33, 1.63MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<04:05, 2.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<02:58, 3.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<35:55, 250kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<26:57, 332kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<19:18, 464kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<13:32, 657kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<1:13:39, 121kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<52:26, 170kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<36:49, 241kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<27:48, 317kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<21:21, 413kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<15:19, 575kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<10:51, 809kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<10:15, 853kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<08:05, 1.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<05:50, 1.50MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<06:07, 1.42MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:09, 1.68MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<03:49, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:42, 1.83MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<05:02, 1.71MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:58, 2.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:09, 2.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<03:48, 2.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<02:50, 2.99MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:56, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:29, 1.89MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:30, 2.42MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<02:35, 3.25MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:30, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<04:01, 2.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<02:59, 2.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:03, 2.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:32, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:33, 2.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<02:33, 3.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<23:59, 345kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<17:38, 469kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<12:29, 660kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<10:37, 773kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<09:06, 902kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<06:46, 1.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<06:02, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<05:04, 1.60MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:43, 2.18MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:29, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:47, 1.69MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:41, 2.18MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<02:41, 2.99MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:33, 1.44MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:41, 1.70MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<03:27, 2.31MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:16, 1.86MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:36, 1.72MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:34, 2.22MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<02:34, 3.05MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<07:59, 984kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<06:24, 1.23MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:40, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<05:05, 1.53MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:21, 1.79MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:12, 2.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:04, 1.90MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:37, 2.13MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<02:43, 2.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:42, 2.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<04:09, 1.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:14, 2.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<02:21, 3.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<07:10, 1.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<05:40, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:09, 1.82MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<02:59, 2.52MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<17:18, 435kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<13:23, 561kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<09:44, 770kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<06:52, 1.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<09:42, 768kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<07:33, 985kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<05:26, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:31, 1.34MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:21, 1.38MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<04:03, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<02:57, 2.48MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:28, 1.63MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:53, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:52, 2.54MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:42, 1.96MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:03, 1.78MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:12, 2.25MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<02:19, 3.10MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<6:53:21, 17.4kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<4:49:42, 24.8kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<3:22:16, 35.3kB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:03<2:20:59, 50.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<2:14:00, 53.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<1:34:26, 75.2kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<1:06:01, 107kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<47:39, 148kB/s]  .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<34:46, 202kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<24:39, 285kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:07<17:12, 405kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<6:55:32, 16.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<4:51:20, 23.9kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<3:23:19, 34.1kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<2:23:13, 48.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<1:40:51, 68.4kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<1:10:30, 97.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<50:46, 135kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<36:55, 185kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<26:05, 261kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<18:16, 371kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<15:52, 426kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<11:48, 573kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<08:22, 804kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<07:24, 904kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<05:51, 1.14MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<04:14, 1.57MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:31, 1.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:50, 1.72MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:49, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:29, 1.88MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:44, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:56, 2.23MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:06, 2.09MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:51, 2.26MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:08, 3.01MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:58, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:43, 2.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:02, 3.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:56, 2.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:42, 2.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:01, 3.12MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:54, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:40, 2.34MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:00, 3.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:52, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:19, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:36, 2.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<01:53, 3.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<06:05, 1.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:54, 1.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:34, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:54, 1.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:57, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:01, 2.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<02:11, 2.75MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:28, 1.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:45, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:46, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:19, 1.79MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:32, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:43, 2.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<01:58, 2.98MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:08, 1.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:29, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<02:35, 2.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:09, 1.83MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:21, 1.73MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:35, 2.23MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<01:53, 3.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:55, 1.46MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:20, 1.71MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:28, 2.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:03, 1.86MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:20, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:37, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<01:53, 2.96MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<41:18, 136kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<29:26, 190kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<20:39, 269kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<15:40, 353kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<11:31, 479kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<08:08, 675kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<06:57, 784kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<05:25, 1.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:55, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<04:02, 1.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<03:55, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:58, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<02:08, 2.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:41, 1.44MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:02, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<02:15, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:49, 1.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:30, 2.10MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<01:52, 2.78MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:32, 2.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:49, 1.83MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:14, 2.31MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:23, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:12, 2.32MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<01:39, 3.06MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:19, 2.17MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:08, 2.35MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:35, 3.14MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:15, 2.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:33, 1.95MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:59, 2.49MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<01:27, 3.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:01, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:37, 1.87MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:56, 2.52MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:28, 1.96MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:42, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:08, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:15, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:36, 1.83MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:03, 2.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:12<01:29, 3.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<31:08, 151kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<22:15, 211kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<15:37, 299kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<11:55, 389kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<09:17, 499kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<06:43, 688kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<05:23, 847kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<04:14, 1.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<03:04, 1.48MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:10, 1.42MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:10, 1.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:24, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:44, 2.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:47, 1.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:06, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:17, 1.93MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:27, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:36, 1.67MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:02, 2.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<01:27, 2.96MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<32:18, 133kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<23:01, 186kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<16:07, 264kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<12:09, 347kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<08:56, 471kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<06:19, 662kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<05:21, 775kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<04:10, 995kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<03:00, 1.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:03, 1.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:32, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:52, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:15, 1.78MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:54, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:25, 2.81MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:34<01:02, 3.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<16:39, 237kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<12:27, 316kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<08:52, 443kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<06:12, 627kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<06:00, 646kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<04:35, 842kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:17, 1.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<03:10, 1.20MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:36, 1.46MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:53, 1.99MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:11, 1.70MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:16, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:44, 2.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:16, 2.91MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:23, 1.53MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:59, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:28, 2.47MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:04, 3.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<07:32, 477kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<05:38, 637kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<04:01, 889kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<03:36, 978kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<03:14, 1.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:26, 1.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:14, 1.54MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:55, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:25, 2.41MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:47, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:56, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:30, 2.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:05, 3.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:16, 1.46MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:55, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:24, 2.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:44, 1.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:55, 1.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:30, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<01:04, 2.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<23:52, 134kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<17:00, 187kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<11:52, 266kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<08:56, 349kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<06:53, 453kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<04:56, 629kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<03:27, 889kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<04:17, 712kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<03:18, 920kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<02:22, 1.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:20, 1.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:14, 1.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:43, 1.72MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:13, 2.39MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<21:41, 135kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<15:27, 188kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<10:47, 267kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<08:07, 351kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<05:57, 477kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<04:11, 671kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<03:33, 780kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<03:02, 915kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<02:14, 1.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:35, 1.72MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:22, 1.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:56, 1.39MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:24, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:35, 1.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:20, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:00, 2.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:13<00:43, 3.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<10:53, 236kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<08:08, 316kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<05:46, 443kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<04:01, 627kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<03:53, 643kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:58, 840kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<02:07, 1.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<02:02, 1.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:55, 1.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:27, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:23, 1.71MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:12, 1.95MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<00:53, 2.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:09, 1.99MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:02, 2.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:46, 2.95MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:03, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:10, 1.89MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:54, 2.42MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:40, 3.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:07, 1.93MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:00, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:44, 2.88MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:00, 2.08MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:07, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<00:53, 2.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:29<00:37, 3.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<1:57:37, 17.2kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<1:22:14, 24.6kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<56:55, 35.1kB/s]  .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<39:09, 50.0kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<35:16, 55.5kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<25:02, 78.0kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<17:29, 111kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<12:03, 158kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<09:11, 205kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<06:36, 285kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<04:36, 402kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<03:35, 507kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<02:49, 644kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<02:03, 878kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<01:26, 1.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:38, 1.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:19, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:57, 1.80MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:02, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:54, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:39, 2.48MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:50, 1.93MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:44, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:32, 2.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:44, 2.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:50, 1.84MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:38, 2.36MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:28, 3.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:48, 1.84MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:41, 2.12MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:33, 2.60MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:23, 3.57MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:39, 851kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:26, 970kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:04, 1.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:44, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<10:04, 133kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<07:09, 186kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<04:56, 264kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<03:39, 347kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<02:46, 457kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<01:59, 631kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<01:22, 888kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:23, 868kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:05, 1.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:46, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:47, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:39, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:28, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:34, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:37, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:28, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:58<00:19, 3.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:17, 767kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:00, 984kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:42, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:41, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:40, 1.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:30, 1.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:21, 2.49MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:43, 1.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:35, 1.45MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:25, 1.98MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:27, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:24, 1.95MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:17, 2.59MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:21, 1.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:24, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:18, 2.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:08<00:12, 3.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:03, 615kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:48, 805kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:33, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:30, 1.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:28, 1.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:20, 1.64MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:13, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:32, 957kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:25, 1.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:17, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:17, 1.51MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:18, 1.48MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:13, 1.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:09, 2.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:14, 1.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:12, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:08, 2.49MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:09, 1.95MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:10, 1.78MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:07, 2.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:04, 3.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:15, 937kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:11, 1.20MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:07, 1.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:04, 2.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:51, 205kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:37, 276kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:24, 386kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:12, 506kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:08, 673kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:04, 938kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 1.02MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.11MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.47MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<13:33:48,  8.19it/s]  0%|          | 1010/400000 [00:00<9:28:25, 11.70it/s]  1%|          | 2071/400000 [00:00<6:37:01, 16.70it/s]  1%|          | 3087/400000 [00:00<4:37:24, 23.85it/s]  1%|          | 4069/400000 [00:00<3:13:54, 34.03it/s]  1%|▏         | 5065/400000 [00:00<2:15:35, 48.55it/s]  2%|▏         | 6091/400000 [00:00<1:34:51, 69.21it/s]  2%|▏         | 7120/400000 [00:00<1:06:25, 98.59it/s]  2%|▏         | 8203/400000 [00:00<46:32, 140.29it/s]   2%|▏         | 9187/400000 [00:01<32:41, 199.20it/s]  3%|▎         | 10184/400000 [00:01<23:01, 282.15it/s]  3%|▎         | 11197/400000 [00:01<16:16, 398.31it/s]  3%|▎         | 12224/400000 [00:01<11:32, 559.71it/s]  3%|▎         | 13282/400000 [00:01<08:14, 781.85it/s]  4%|▎         | 14313/400000 [00:01<05:56, 1081.77it/s]  4%|▍         | 15336/400000 [00:01<04:20, 1477.63it/s]  4%|▍         | 16356/400000 [00:01<03:13, 1984.54it/s]  4%|▍         | 17368/400000 [00:01<02:26, 2613.89it/s]  5%|▍         | 18397/400000 [00:01<01:53, 3367.43it/s]  5%|▍         | 19413/400000 [00:02<01:31, 4174.32it/s]  5%|▌         | 20408/400000 [00:02<01:15, 5040.74it/s]  5%|▌         | 21398/400000 [00:02<01:04, 5866.07it/s]  6%|▌         | 22434/400000 [00:02<00:56, 6742.13it/s]  6%|▌         | 23450/400000 [00:02<00:50, 7497.70it/s]  6%|▌         | 24478/400000 [00:02<00:46, 8159.11it/s]  6%|▋         | 25492/400000 [00:02<00:43, 8666.70it/s]  7%|▋         | 26503/400000 [00:02<00:41, 9034.73it/s]  7%|▋         | 27541/400000 [00:02<00:39, 9399.67it/s]  7%|▋         | 28572/400000 [00:02<00:38, 9652.98it/s]  7%|▋         | 29606/400000 [00:03<00:37, 9847.53it/s]  8%|▊         | 30669/400000 [00:03<00:36, 10068.28it/s]  8%|▊         | 31729/400000 [00:03<00:36, 10220.97it/s]  8%|▊         | 32799/400000 [00:03<00:35, 10359.67it/s]  8%|▊         | 33918/400000 [00:03<00:34, 10594.90it/s]  9%|▊         | 34990/400000 [00:03<00:34, 10622.65it/s]  9%|▉         | 36061/400000 [00:03<00:34, 10574.77it/s]  9%|▉         | 37147/400000 [00:03<00:34, 10658.11it/s] 10%|▉         | 38218/400000 [00:03<00:35, 10276.64it/s] 10%|▉         | 39326/400000 [00:03<00:34, 10504.99it/s] 10%|█         | 40382/400000 [00:04<00:34, 10406.87it/s] 10%|█         | 41427/400000 [00:04<00:34, 10377.59it/s] 11%|█         | 42507/400000 [00:04<00:34, 10497.84it/s] 11%|█         | 43560/400000 [00:04<00:34, 10461.33it/s] 11%|█         | 44654/400000 [00:04<00:33, 10598.67it/s] 11%|█▏        | 45729/400000 [00:04<00:33, 10641.63it/s] 12%|█▏        | 46795/400000 [00:04<00:33, 10582.41it/s] 12%|█▏        | 47855/400000 [00:04<00:33, 10528.79it/s] 12%|█▏        | 48909/400000 [00:04<00:33, 10451.29it/s] 12%|█▏        | 49955/400000 [00:04<00:33, 10449.62it/s] 13%|█▎        | 51024/400000 [00:05<00:33, 10518.64it/s] 13%|█▎        | 52108/400000 [00:05<00:32, 10611.22it/s] 13%|█▎        | 53178/400000 [00:05<00:32, 10636.66it/s] 14%|█▎        | 54243/400000 [00:05<00:32, 10563.62it/s] 14%|█▍        | 55327/400000 [00:05<00:32, 10644.51it/s] 14%|█▍        | 56395/400000 [00:05<00:32, 10652.63it/s] 14%|█▍        | 57461/400000 [00:05<00:32, 10381.82it/s] 15%|█▍        | 58501/400000 [00:05<00:33, 10196.12it/s] 15%|█▍        | 59569/400000 [00:05<00:32, 10334.13it/s] 15%|█▌        | 60607/400000 [00:05<00:32, 10346.29it/s] 15%|█▌        | 61664/400000 [00:06<00:32, 10409.70it/s] 16%|█▌        | 62709/400000 [00:06<00:32, 10419.66it/s] 16%|█▌        | 63781/400000 [00:06<00:31, 10507.58it/s] 16%|█▌        | 64833/400000 [00:06<00:32, 10407.54it/s] 16%|█▋        | 65893/400000 [00:06<00:31, 10463.09it/s] 17%|█▋        | 66940/400000 [00:06<00:33, 9957.07it/s]  17%|█▋        | 67942/400000 [00:06<00:33, 9810.06it/s] 17%|█▋        | 68928/400000 [00:06<00:34, 9603.86it/s] 17%|█▋        | 69893/400000 [00:06<00:34, 9433.71it/s] 18%|█▊        | 70840/400000 [00:07<00:35, 9344.36it/s] 18%|█▊        | 71808/400000 [00:07<00:34, 9441.14it/s] 18%|█▊        | 72755/400000 [00:07<00:35, 9338.51it/s] 18%|█▊        | 73691/400000 [00:07<00:35, 9273.51it/s] 19%|█▊        | 74625/400000 [00:07<00:35, 9292.88it/s] 19%|█▉        | 75656/400000 [00:07<00:33, 9575.63it/s] 19%|█▉        | 76709/400000 [00:07<00:32, 9840.74it/s] 19%|█▉        | 77738/400000 [00:07<00:32, 9969.49it/s] 20%|█▉        | 78739/400000 [00:07<00:32, 9946.69it/s] 20%|█▉        | 79779/400000 [00:07<00:31, 10077.42it/s] 20%|██        | 80809/400000 [00:08<00:31, 10141.16it/s] 20%|██        | 81869/400000 [00:08<00:30, 10274.16it/s] 21%|██        | 82945/400000 [00:08<00:30, 10411.31it/s] 21%|██        | 83988/400000 [00:08<00:31, 10150.23it/s] 21%|██▏       | 85006/400000 [00:08<00:31, 10118.59it/s] 22%|██▏       | 86051/400000 [00:08<00:30, 10213.92it/s] 22%|██▏       | 87085/400000 [00:08<00:30, 10249.75it/s] 22%|██▏       | 88137/400000 [00:08<00:30, 10327.15it/s] 22%|██▏       | 89186/400000 [00:08<00:29, 10374.20it/s] 23%|██▎       | 90252/400000 [00:08<00:29, 10457.29it/s] 23%|██▎       | 91299/400000 [00:09<00:30, 10141.30it/s] 23%|██▎       | 92316/400000 [00:09<00:30, 10074.46it/s] 23%|██▎       | 93360/400000 [00:09<00:30, 10180.17it/s] 24%|██▎       | 94380/400000 [00:09<00:30, 10146.12it/s] 24%|██▍       | 95408/400000 [00:09<00:29, 10173.57it/s] 24%|██▍       | 96428/400000 [00:09<00:29, 10181.31it/s] 24%|██▍       | 97447/400000 [00:09<00:30, 10058.32it/s] 25%|██▍       | 98525/400000 [00:09<00:29, 10263.57it/s] 25%|██▍       | 99585/400000 [00:09<00:28, 10361.89it/s] 25%|██▌       | 100623/400000 [00:09<00:28, 10331.84it/s] 25%|██▌       | 101658/400000 [00:10<00:28, 10336.58it/s] 26%|██▌       | 102748/400000 [00:10<00:28, 10497.79it/s] 26%|██▌       | 103827/400000 [00:10<00:27, 10581.68it/s] 26%|██▌       | 104887/400000 [00:10<00:28, 10510.36it/s] 26%|██▋       | 105939/400000 [00:10<00:28, 10432.41it/s] 27%|██▋       | 106983/400000 [00:10<00:28, 10336.66it/s] 27%|██▋       | 108018/400000 [00:10<00:28, 10296.63it/s] 27%|██▋       | 109049/400000 [00:10<00:29, 9969.56it/s]  28%|██▊       | 110053/400000 [00:10<00:29, 9990.21it/s] 28%|██▊       | 111086/400000 [00:10<00:28, 10088.90it/s] 28%|██▊       | 112131/400000 [00:11<00:28, 10194.00it/s] 28%|██▊       | 113187/400000 [00:11<00:27, 10300.10it/s] 29%|██▊       | 114232/400000 [00:11<00:27, 10344.50it/s] 29%|██▉       | 115268/400000 [00:11<00:27, 10297.00it/s] 29%|██▉       | 116308/400000 [00:11<00:27, 10327.12it/s] 29%|██▉       | 117342/400000 [00:11<00:27, 10212.96it/s] 30%|██▉       | 118364/400000 [00:11<00:27, 10132.51it/s] 30%|██▉       | 119400/400000 [00:11<00:27, 10199.58it/s] 30%|███       | 120452/400000 [00:11<00:27, 10291.37it/s] 30%|███       | 121482/400000 [00:12<00:27, 10077.27it/s] 31%|███       | 122533/400000 [00:12<00:27, 10202.15it/s] 31%|███       | 123605/400000 [00:12<00:26, 10350.16it/s] 31%|███       | 124642/400000 [00:12<00:26, 10338.48it/s] 31%|███▏      | 125677/400000 [00:12<00:26, 10179.01it/s] 32%|███▏      | 126697/400000 [00:12<00:27, 9820.19it/s]  32%|███▏      | 127752/400000 [00:12<00:27, 10025.98it/s] 32%|███▏      | 128807/400000 [00:12<00:26, 10176.65it/s] 32%|███▏      | 129865/400000 [00:12<00:26, 10292.33it/s] 33%|███▎      | 130897/400000 [00:12<00:26, 10297.49it/s] 33%|███▎      | 131929/400000 [00:13<00:26, 10287.97it/s] 33%|███▎      | 132960/400000 [00:13<00:26, 10105.63it/s] 34%|███▎      | 134001/400000 [00:13<00:26, 10192.47it/s] 34%|███▍      | 135022/400000 [00:13<00:25, 10194.81it/s] 34%|███▍      | 136076/400000 [00:13<00:25, 10295.07it/s] 34%|███▍      | 137107/400000 [00:13<00:26, 10017.99it/s] 35%|███▍      | 138148/400000 [00:13<00:25, 10130.52it/s] 35%|███▍      | 139204/400000 [00:13<00:25, 10255.34it/s] 35%|███▌      | 140232/400000 [00:13<00:25, 10258.60it/s] 35%|███▌      | 141260/400000 [00:13<00:25, 10132.40it/s] 36%|███▌      | 142275/400000 [00:14<00:25, 9973.19it/s]  36%|███▌      | 143274/400000 [00:14<00:25, 9881.12it/s] 36%|███▌      | 144330/400000 [00:14<00:25, 10073.64it/s] 36%|███▋      | 145382/400000 [00:14<00:24, 10201.78it/s] 37%|███▋      | 146430/400000 [00:14<00:24, 10280.17it/s] 37%|███▋      | 147460/400000 [00:14<00:24, 10190.68it/s] 37%|███▋      | 148500/400000 [00:14<00:24, 10251.17it/s] 37%|███▋      | 149560/400000 [00:14<00:24, 10350.93it/s] 38%|███▊      | 150638/400000 [00:14<00:23, 10474.58it/s] 38%|███▊      | 151687/400000 [00:14<00:23, 10458.62it/s] 38%|███▊      | 152734/400000 [00:15<00:24, 10153.20it/s] 38%|███▊      | 153791/400000 [00:15<00:23, 10272.24it/s] 39%|███▊      | 154861/400000 [00:15<00:23, 10394.44it/s] 39%|███▉      | 155942/400000 [00:15<00:23, 10513.99it/s] 39%|███▉      | 157004/400000 [00:15<00:23, 10544.45it/s] 40%|███▉      | 158060/400000 [00:15<00:23, 10507.79it/s] 40%|███▉      | 159129/400000 [00:15<00:22, 10559.64it/s] 40%|████      | 160186/400000 [00:15<00:23, 10189.16it/s] 40%|████      | 161209/400000 [00:15<00:24, 9794.91it/s]  41%|████      | 162248/400000 [00:15<00:23, 9963.27it/s] 41%|████      | 163314/400000 [00:16<00:23, 10162.01it/s] 41%|████      | 164335/400000 [00:16<00:23, 10040.46it/s] 41%|████▏     | 165343/400000 [00:16<00:23, 9798.19it/s]  42%|████▏     | 166348/400000 [00:16<00:23, 9870.94it/s] 42%|████▏     | 167345/400000 [00:16<00:23, 9899.23it/s] 42%|████▏     | 168400/400000 [00:16<00:22, 10083.30it/s] 42%|████▏     | 169411/400000 [00:16<00:22, 10067.60it/s] 43%|████▎     | 170450/400000 [00:16<00:22, 10162.10it/s] 43%|████▎     | 171538/400000 [00:16<00:22, 10367.10it/s] 43%|████▎     | 172577/400000 [00:17<00:22, 10211.95it/s] 43%|████▎     | 173601/400000 [00:17<00:22, 10171.52it/s] 44%|████▎     | 174626/400000 [00:17<00:22, 10192.01it/s] 44%|████▍     | 175664/400000 [00:17<00:21, 10245.96it/s] 44%|████▍     | 176695/400000 [00:17<00:21, 10264.97it/s] 44%|████▍     | 177723/400000 [00:17<00:22, 10066.73it/s] 45%|████▍     | 178731/400000 [00:17<00:22, 9990.89it/s]  45%|████▍     | 179756/400000 [00:17<00:21, 10066.18it/s] 45%|████▌     | 180829/400000 [00:17<00:21, 10254.03it/s] 45%|████▌     | 181893/400000 [00:17<00:21, 10365.85it/s] 46%|████▌     | 182931/400000 [00:18<00:21, 10295.96it/s] 46%|████▌     | 183983/400000 [00:18<00:20, 10359.93it/s] 46%|████▋     | 185020/400000 [00:18<00:20, 10251.85it/s] 47%|████▋     | 186077/400000 [00:18<00:20, 10344.49it/s] 47%|████▋     | 187113/400000 [00:18<00:20, 10341.74it/s] 47%|████▋     | 188148/400000 [00:18<00:20, 10289.57it/s] 47%|████▋     | 189224/400000 [00:18<00:20, 10424.71it/s] 48%|████▊     | 190288/400000 [00:18<00:19, 10488.07it/s] 48%|████▊     | 191338/400000 [00:18<00:20, 10358.99it/s] 48%|████▊     | 192434/400000 [00:18<00:19, 10530.40it/s] 48%|████▊     | 193489/400000 [00:19<00:19, 10528.06it/s] 49%|████▊     | 194543/400000 [00:19<00:20, 10234.95it/s] 49%|████▉     | 195569/400000 [00:19<00:20, 10157.04it/s] 49%|████▉     | 196626/400000 [00:19<00:19, 10275.08it/s] 49%|████▉     | 197689/400000 [00:19<00:19, 10378.37it/s] 50%|████▉     | 198729/400000 [00:19<00:19, 10200.94it/s] 50%|████▉     | 199759/400000 [00:19<00:19, 10229.08it/s] 50%|█████     | 200784/400000 [00:19<00:19, 10207.85it/s] 50%|█████     | 201831/400000 [00:19<00:19, 10284.43it/s] 51%|█████     | 202861/400000 [00:19<00:19, 10221.62it/s] 51%|█████     | 203884/400000 [00:20<00:19, 9976.08it/s]  51%|█████     | 204990/400000 [00:20<00:18, 10277.12it/s] 52%|█████▏    | 206055/400000 [00:20<00:18, 10384.99it/s] 52%|█████▏    | 207150/400000 [00:20<00:18, 10546.57it/s] 52%|█████▏    | 208208/400000 [00:20<00:18, 10188.44it/s] 52%|█████▏    | 209232/400000 [00:20<00:18, 10190.23it/s] 53%|█████▎    | 210277/400000 [00:20<00:18, 10266.68it/s] 53%|█████▎    | 211306/400000 [00:20<00:18, 10037.31it/s] 53%|█████▎    | 212313/400000 [00:20<00:18, 10008.99it/s] 53%|█████▎    | 213375/400000 [00:20<00:18, 10182.40it/s] 54%|█████▎    | 214396/400000 [00:21<00:18, 10184.21it/s] 54%|█████▍    | 215476/400000 [00:21<00:17, 10359.74it/s] 54%|█████▍    | 216538/400000 [00:21<00:17, 10436.40it/s] 54%|█████▍    | 217584/400000 [00:21<00:17, 10261.70it/s] 55%|█████▍    | 218612/400000 [00:21<00:18, 9809.07it/s]  55%|█████▍    | 219599/400000 [00:21<00:18, 9740.41it/s] 55%|█████▌    | 220577/400000 [00:21<00:18, 9562.42it/s] 55%|█████▌    | 221563/400000 [00:21<00:18, 9647.83it/s] 56%|█████▌    | 222565/400000 [00:21<00:18, 9756.16it/s] 56%|█████▌    | 223569/400000 [00:22<00:17, 9837.87it/s] 56%|█████▌    | 224578/400000 [00:22<00:17, 9910.97it/s] 56%|█████▋    | 225683/400000 [00:22<00:17, 10226.98it/s] 57%|█████▋    | 226710/400000 [00:22<00:17, 10142.46it/s] 57%|█████▋    | 227727/400000 [00:22<00:17, 9929.39it/s]  57%|█████▋    | 228723/400000 [00:22<00:17, 9790.48it/s] 57%|█████▋    | 229727/400000 [00:22<00:17, 9861.05it/s] 58%|█████▊    | 230773/400000 [00:22<00:16, 10032.48it/s] 58%|█████▊    | 231807/400000 [00:22<00:16, 10121.52it/s] 58%|█████▊    | 232881/400000 [00:22<00:16, 10298.26it/s] 58%|█████▊    | 233913/400000 [00:23<00:16, 10296.98it/s] 59%|█████▊    | 234944/400000 [00:23<00:16, 10129.80it/s] 59%|█████▉    | 236031/400000 [00:23<00:15, 10339.36it/s] 59%|█████▉    | 237089/400000 [00:23<00:15, 10408.32it/s] 60%|█████▉    | 238145/400000 [00:23<00:15, 10452.37it/s] 60%|█████▉    | 239192/400000 [00:23<00:15, 10305.18it/s] 60%|██████    | 240244/400000 [00:23<00:15, 10367.09it/s] 60%|██████    | 241282/400000 [00:23<00:15, 10335.97it/s] 61%|██████    | 242324/400000 [00:23<00:15, 10360.62it/s] 61%|██████    | 243418/400000 [00:23<00:14, 10524.84it/s] 61%|██████    | 244472/400000 [00:24<00:15, 10364.79it/s] 61%|██████▏   | 245510/400000 [00:24<00:15, 10197.14it/s] 62%|██████▏   | 246532/400000 [00:24<00:15, 10187.06it/s] 62%|██████▏   | 247601/400000 [00:24<00:14, 10331.04it/s] 62%|██████▏   | 248680/400000 [00:24<00:14, 10463.55it/s] 62%|██████▏   | 249728/400000 [00:24<00:14, 10430.00it/s] 63%|██████▎   | 250772/400000 [00:24<00:14, 10257.96it/s] 63%|██████▎   | 251800/400000 [00:24<00:14, 10092.43it/s] 63%|██████▎   | 252898/400000 [00:24<00:14, 10340.96it/s] 63%|██████▎   | 253998/400000 [00:24<00:13, 10529.25it/s] 64%|██████▍   | 255054/400000 [00:25<00:13, 10537.99it/s] 64%|██████▍   | 256110/400000 [00:25<00:13, 10495.47it/s] 64%|██████▍   | 257180/400000 [00:25<00:13, 10553.70it/s] 65%|██████▍   | 258265/400000 [00:25<00:13, 10640.05it/s] 65%|██████▍   | 259337/400000 [00:25<00:13, 10662.67it/s] 65%|██████▌   | 260404/400000 [00:25<00:13, 10526.34it/s] 65%|██████▌   | 261458/400000 [00:25<00:13, 10486.09it/s] 66%|██████▌   | 262508/400000 [00:25<00:13, 9873.32it/s]  66%|██████▌   | 263504/400000 [00:25<00:13, 9805.97it/s] 66%|██████▌   | 264491/400000 [00:26<00:13, 9817.13it/s] 66%|██████▋   | 265510/400000 [00:26<00:13, 9925.94it/s] 67%|██████▋   | 266506/400000 [00:26<00:13, 9618.46it/s] 67%|██████▋   | 267564/400000 [00:26<00:13, 9886.24it/s] 67%|██████▋   | 268644/400000 [00:26<00:12, 10141.56it/s] 67%|██████▋   | 269708/400000 [00:26<00:12, 10284.23it/s] 68%|██████▊   | 270765/400000 [00:26<00:12, 10367.01it/s] 68%|██████▊   | 271805/400000 [00:26<00:12, 10115.44it/s] 68%|██████▊   | 272881/400000 [00:26<00:12, 10298.72it/s] 68%|██████▊   | 273936/400000 [00:26<00:12, 10370.22it/s] 69%|██████▉   | 275018/400000 [00:27<00:11, 10499.85it/s] 69%|██████▉   | 276071/400000 [00:27<00:11, 10451.81it/s] 69%|██████▉   | 277118/400000 [00:27<00:11, 10400.60it/s] 70%|██████▉   | 278175/400000 [00:27<00:11, 10448.12it/s] 70%|██████▉   | 279221/400000 [00:27<00:11, 10444.93it/s] 70%|███████   | 280267/400000 [00:27<00:11, 10124.79it/s] 70%|███████   | 281293/400000 [00:27<00:11, 10163.05it/s] 71%|███████   | 282312/400000 [00:27<00:11, 10029.89it/s] 71%|███████   | 283397/400000 [00:27<00:11, 10262.24it/s] 71%|███████   | 284426/400000 [00:27<00:11, 10165.61it/s] 71%|███████▏  | 285468/400000 [00:28<00:11, 10240.33it/s] 72%|███████▏  | 286563/400000 [00:28<00:10, 10440.53it/s] 72%|███████▏  | 287610/400000 [00:28<00:10, 10384.41it/s] 72%|███████▏  | 288650/400000 [00:28<00:10, 10357.94it/s] 72%|███████▏  | 289708/400000 [00:28<00:10, 10421.03it/s] 73%|███████▎  | 290751/400000 [00:28<00:10, 10363.48it/s] 73%|███████▎  | 291844/400000 [00:28<00:10, 10526.42it/s] 73%|███████▎  | 292898/400000 [00:28<00:10, 10450.94it/s] 73%|███████▎  | 293944/400000 [00:28<00:10, 10404.72it/s] 74%|███████▍  | 295044/400000 [00:28<00:09, 10575.37it/s] 74%|███████▍  | 296123/400000 [00:29<00:09, 10636.46it/s] 74%|███████▍  | 297188/400000 [00:29<00:10, 10174.77it/s] 75%|███████▍  | 298211/400000 [00:29<00:10, 9832.78it/s]  75%|███████▍  | 299268/400000 [00:29<00:10, 10040.93it/s] 75%|███████▌  | 300306/400000 [00:29<00:09, 10138.13it/s] 75%|███████▌  | 301324/400000 [00:29<00:09, 10048.87it/s] 76%|███████▌  | 302341/400000 [00:29<00:09, 10083.90it/s] 76%|███████▌  | 303352/400000 [00:29<00:09, 10002.48it/s] 76%|███████▌  | 304412/400000 [00:29<00:09, 10172.56it/s] 76%|███████▋  | 305495/400000 [00:29<00:09, 10360.50it/s] 77%|███████▋  | 306544/400000 [00:30<00:08, 10397.09it/s] 77%|███████▋  | 307646/400000 [00:30<00:08, 10574.80it/s] 77%|███████▋  | 308706/400000 [00:30<00:08, 10460.95it/s] 77%|███████▋  | 309777/400000 [00:30<00:08, 10531.96it/s] 78%|███████▊  | 310852/400000 [00:30<00:08, 10594.00it/s] 78%|███████▊  | 311913/400000 [00:30<00:08, 10577.64it/s] 78%|███████▊  | 312977/400000 [00:30<00:08, 10589.97it/s] 79%|███████▊  | 314037/400000 [00:30<00:08, 10327.06it/s] 79%|███████▉  | 315072/400000 [00:30<00:08, 10192.44it/s] 79%|███████▉  | 316123/400000 [00:31<00:08, 10283.57it/s] 79%|███████▉  | 317164/400000 [00:31<00:08, 10320.20it/s] 80%|███████▉  | 318214/400000 [00:31<00:07, 10372.31it/s] 80%|███████▉  | 319252/400000 [00:31<00:07, 10301.49it/s] 80%|████████  | 320314/400000 [00:31<00:07, 10394.28it/s] 80%|████████  | 321387/400000 [00:31<00:07, 10490.53it/s] 81%|████████  | 322460/400000 [00:31<00:07, 10560.20it/s] 81%|████████  | 323517/400000 [00:31<00:07, 10502.27it/s] 81%|████████  | 324568/400000 [00:31<00:07, 10234.44it/s] 81%|████████▏ | 325680/400000 [00:31<00:07, 10483.07it/s] 82%|████████▏ | 326741/400000 [00:32<00:06, 10519.47it/s] 82%|████████▏ | 327795/400000 [00:32<00:06, 10496.59it/s] 82%|████████▏ | 328847/400000 [00:32<00:06, 10410.87it/s] 82%|████████▏ | 329890/400000 [00:32<00:06, 10188.86it/s] 83%|████████▎ | 330911/400000 [00:32<00:06, 10036.56it/s] 83%|████████▎ | 331917/400000 [00:32<00:07, 9696.65it/s]  83%|████████▎ | 332891/400000 [00:32<00:07, 9424.03it/s] 83%|████████▎ | 333848/400000 [00:32<00:06, 9467.35it/s] 84%|████████▎ | 334897/400000 [00:32<00:06, 9751.63it/s] 84%|████████▍ | 335978/400000 [00:32<00:06, 10044.36it/s] 84%|████████▍ | 336988/400000 [00:33<00:06, 9959.53it/s]  85%|████████▍ | 338043/400000 [00:33<00:06, 10128.14it/s] 85%|████████▍ | 339076/400000 [00:33<00:05, 10186.86it/s] 85%|████████▌ | 340115/400000 [00:33<00:05, 10246.79it/s] 85%|████████▌ | 341197/400000 [00:33<00:05, 10410.34it/s] 86%|████████▌ | 342240/400000 [00:33<00:05, 10375.61it/s] 86%|████████▌ | 343329/400000 [00:33<00:05, 10524.30it/s] 86%|████████▌ | 344383/400000 [00:33<00:05, 10442.39it/s] 86%|████████▋ | 345455/400000 [00:33<00:05, 10521.93it/s] 87%|████████▋ | 346537/400000 [00:33<00:05, 10609.06it/s] 87%|████████▋ | 347599/400000 [00:34<00:04, 10576.71it/s] 87%|████████▋ | 348658/400000 [00:34<00:04, 10426.51it/s] 87%|████████▋ | 349702/400000 [00:34<00:04, 10201.84it/s] 88%|████████▊ | 350727/400000 [00:34<00:04, 10213.11it/s] 88%|████████▊ | 351817/400000 [00:34<00:04, 10407.97it/s] 88%|████████▊ | 352860/400000 [00:34<00:04, 10142.54it/s] 88%|████████▊ | 353897/400000 [00:34<00:04, 10208.22it/s] 89%|████████▊ | 354925/400000 [00:34<00:04, 10228.54it/s] 89%|████████▉ | 355987/400000 [00:34<00:04, 10342.13it/s] 89%|████████▉ | 357023/400000 [00:35<00:04, 10274.66it/s] 90%|████████▉ | 358100/400000 [00:35<00:04, 10416.90it/s] 90%|████████▉ | 359157/400000 [00:35<00:03, 10461.18it/s] 90%|█████████ | 360204/400000 [00:35<00:03, 10389.82it/s] 90%|█████████ | 361244/400000 [00:35<00:03, 10355.92it/s] 91%|█████████ | 362281/400000 [00:35<00:03, 10346.30it/s] 91%|█████████ | 363317/400000 [00:35<00:03, 10284.04it/s] 91%|█████████ | 364365/400000 [00:35<00:03, 10340.90it/s] 91%|█████████▏| 365400/400000 [00:35<00:03, 10246.97it/s] 92%|█████████▏| 366426/400000 [00:35<00:03, 10240.56it/s] 92%|█████████▏| 367451/400000 [00:36<00:03, 9864.49it/s]  92%|█████████▏| 368500/400000 [00:36<00:03, 10041.99it/s] 92%|█████████▏| 369560/400000 [00:36<00:02, 10202.52it/s] 93%|█████████▎| 370583/400000 [00:36<00:02, 10175.91it/s] 93%|█████████▎| 371637/400000 [00:36<00:02, 10281.83it/s] 93%|█████████▎| 372688/400000 [00:36<00:02, 10347.52it/s] 93%|█████████▎| 373724/400000 [00:36<00:02, 10340.23it/s] 94%|█████████▎| 374764/400000 [00:36<00:02, 10357.89it/s] 94%|█████████▍| 375801/400000 [00:36<00:02, 10266.09it/s] 94%|█████████▍| 376874/400000 [00:36<00:02, 10399.49it/s] 94%|█████████▍| 377954/400000 [00:37<00:02, 10515.15it/s] 95%|█████████▍| 379007/400000 [00:37<00:02, 10416.64it/s] 95%|█████████▌| 380115/400000 [00:37<00:01, 10606.24it/s] 95%|█████████▌| 381178/400000 [00:37<00:01, 10430.60it/s] 96%|█████████▌| 382243/400000 [00:37<00:01, 10491.93it/s] 96%|█████████▌| 383304/400000 [00:37<00:01, 10525.47it/s] 96%|█████████▌| 384358/400000 [00:37<00:01, 10461.44it/s] 96%|█████████▋| 385405/400000 [00:37<00:01, 10247.13it/s] 97%|█████████▋| 386432/400000 [00:37<00:01, 9862.64it/s]  97%|█████████▋| 387455/400000 [00:37<00:01, 9967.52it/s] 97%|█████████▋| 388504/400000 [00:38<00:01, 10117.01it/s] 97%|█████████▋| 389547/400000 [00:38<00:01, 10208.31it/s] 98%|█████████▊| 390570/400000 [00:38<00:00, 10178.99it/s] 98%|█████████▊| 391590/400000 [00:38<00:00, 10113.76it/s] 98%|█████████▊| 392603/400000 [00:38<00:00, 10074.34it/s] 98%|█████████▊| 393612/400000 [00:38<00:00, 10030.53it/s] 99%|█████████▊| 394633/400000 [00:38<00:00, 10082.54it/s] 99%|█████████▉| 395668/400000 [00:38<00:00, 10160.38it/s] 99%|█████████▉| 396685/400000 [00:38<00:00, 10128.51it/s] 99%|█████████▉| 397758/400000 [00:38<00:00, 10298.41it/s]100%|█████████▉| 398817/400000 [00:39<00:00, 10383.70it/s]100%|█████████▉| 399857/400000 [00:39<00:00, 10151.76it/s]100%|█████████▉| 399999/400000 [00:39<00:00, 10206.95it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

 #####  get_Data DataLoader 
((<torchtext.data.dataset.TabularDataset object at 0x7f2d3fcc7278>, <torchtext.data.dataset.TabularDataset object at 0x7f2d3fcc73c8>, <torchtext.vocab.Vocab object at 0x7f2d3fcc72e8>), {})



###### Test_run_model  #############################################################



 ####################################################################################################
model_keras/charcnn.json
{
  "test": {
    "model_pars": {
      "model_uri": "model_keras.charcnn.py",
      "embedding_size": 128,
      "conv_layers": [
        [
          256,
          10
        ],
        [
          256,
          7
        ],
        [
          256,
          5
        ],
        [
          256,
          3
        ]
      ],
      "fully_connected_layers": [
        1024,
        1024
      ],
      "threshold": 1e-06,
      "dropout_p": 0.1,
      "optimizer": "adam",
      "loss": "categorical_crossentropy"
    },
    "data_pars": {
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
    },
    "compute_pars": {
      "epochs": 1,
      "batch_size": 128
    },
    "out_pars": {
      "path": "ztest/ml_keras/charcnn/charcnn.h5",
      "data_type": "pandas",
      "size": [
        0,
        0,
        6
      ],
      "output_size": [
        0,
        6
      ]
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.charcnn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:3313: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:3313: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
CharCNNKim model built: 
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
sent_input (InputLayer)         (None, 1014)         0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 1014, 128)    8960        sent_input[0][0]                 
__________________________________________________________________________________________________
Conv1D_256_10 (Conv1D)          (None, 1005, 256)    327936      embedding_1[0][0]                
__________________________________________________________________________________________________
Conv1D_256_7 (Conv1D)           (None, 1008, 256)    229632      embedding_1[0][0]                
__________________________________________________________________________________________________
Conv1D_256_5 (Conv1D)           (None, 1010, 256)    164096      embedding_1[0][0]                
__________________________________________________________________________________________________
Conv1D_256_3 (Conv1D)           (None, 1012, 256)    98560       embedding_1[0][0]                
__________________________________________________________________________________________________
MaxPoolingOverTime_256_10 (Glob (None, 256)          0           Conv1D_256_10[0][0]              
__________________________________________________________________________________________________
MaxPoolingOverTime_256_7 (Globa (None, 256)          0           Conv1D_256_7[0][0]               
__________________________________________________________________________________________________
MaxPoolingOverTime_256_5 (Globa (None, 256)          0           Conv1D_256_5[0][0]               
__________________________________________________________________________________________________
MaxPoolingOverTime_256_3 (Globa (None, 256)          0           Conv1D_256_3[0][0]               
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 1024)         0           MaxPoolingOverTime_256_10[0][0]  
                                                                 MaxPoolingOverTime_256_7[0][0]   
                                                                 MaxPoolingOverTime_256_5[0][0]   
                                                                 MaxPoolingOverTime_256_3[0][0]   
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1024)         1049600     concatenate_1[0][0]              
__________________________________________________________________________________________________
alpha_dropout_1 (AlphaDropout)  (None, 1024)         0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1024)         1049600     alpha_dropout_1[0][0]            
__________________________________________________________________________________________________
alpha_dropout_2 (AlphaDropout)  (None, 1024)         0           dense_2[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 4)            4100        alpha_dropout_2[0][0]            
==================================================================================================
Total params: 2,932,484
Trainable params: 2,932,484
Non-trainable params: 0
__________________________________________________________________________________________________

  <mlmodels.model_keras.charcnn.Model object at 0x7f2cfc749b38> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.


 Object Creation

 Object Compute

 Object get_data
Train on 354 samples, validate on 354 samples
Epoch 1/1

128/354 [=========>....................] - ETA: 7s - loss: 1.6815
256/354 [====================>.........] - ETA: 3s - loss: 1.2638
354/354 [==============================] - 14s 40ms/step - loss: 1.1883 - val_loss: 0.6620

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data
[[3.0134526e-05 3.4378132e-05 1.5815482e-01 8.4178066e-01]
 [2.6729727e-05 2.7237877e-05 1.3207312e-01 8.6787295e-01]
 [2.9390103e-05 3.2133958e-05 1.6103455e-01 8.3890390e-01]
 ...
 [5.3935921e-05 5.6254343e-05 1.5494387e-01 8.4494591e-01]
 [4.5704910e-05 5.1216270e-05 1.6853644e-01 8.3136660e-01]
 [4.2138316e-05 4.2360287e-05 1.5095581e-01 8.4895968e-01]]

  #### Get  metrics   ################################################ 

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
model_keras/charcnn_zhang.json
{
  "test": {
    "model_pars": {
      "model_uri": "model_keras.charcnn_zhang.py",
      "embedding_size": 128,
      "conv_layers": [
        [
          256,
          7,
          3
        ],
        [
          256,
          7,
          3
        ],
        [
          256,
          3,
          -1
        ],
        [
          256,
          3,
          -1
        ],
        [
          256,
          3,
          -1
        ],
        [
          256,
          3,
          3
        ]
      ],
      "fully_connected_layers": [
        1024,
        1024
      ],
      "threshold": 1e-06,
      "dropout_p": 0.5,
      "optimizer": "adam",
      "loss": "categorical_crossentropy"
    },
    "data_pars": {
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
    },
    "compute_pars": {
      "epochs": 1,
      "batch_size": 128
    },
    "out_pars": {
      "path": "ztest/ml_keras/charcnn_zhang/",
      "data_type": "pandas",
      "size": [
        0,
        0,
        6
      ],
      "output_size": [
        0,
        6
      ]
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset

  <module 'mlmodels.model_keras.charcnn_zhang' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/callbacks/callbacks.py:846: RuntimeWarning: Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: val_loss,loss
  (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

CharCNNZhang model built: 
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
sent_input (InputLayer)      (None, 1014)              0         
_________________________________________________________________
embedding_2 (Embedding)      (None, 1014, 128)         8960      
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 1008, 256)         229632    
_________________________________________________________________
thresholded_re_lu_1 (Thresho (None, 1008, 256)         0         
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 336, 256)          0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 330, 256)          459008    
_________________________________________________________________
thresholded_re_lu_2 (Thresho (None, 330, 256)          0         
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 110, 256)          0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 108, 256)          196864    
_________________________________________________________________
thresholded_re_lu_3 (Thresho (None, 108, 256)          0         
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 106, 256)          196864    
_________________________________________________________________
thresholded_re_lu_4 (Thresho (None, 106, 256)          0         
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 104, 256)          196864    
_________________________________________________________________
thresholded_re_lu_5 (Thresho (None, 104, 256)          0         
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 102, 256)          196864    
_________________________________________________________________
thresholded_re_lu_6 (Thresho (None, 102, 256)          0         
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 34, 256)           0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8704)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 1024)              8913920   
_________________________________________________________________
thresholded_re_lu_7 (Thresho (None, 1024)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
thresholded_re_lu_8 (Thresho (None, 1024)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_6 (Dense)              (None, 4)                 4100      
=================================================================
Total params: 11,452,676
Trainable params: 11,452,676
Non-trainable params: 0
_________________________________________________________________

  <mlmodels.model_keras.charcnn_zhang.Model object at 0x7f2cfc6dceb8> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data
Train on 354 samples, validate on 354 samples
Epoch 1/1

128/354 [=========>....................] - ETA: 5s - loss: 1.3847
256/354 [====================>.........] - ETA: 2s - loss: 1.1880
354/354 [==============================] - 9s 26ms/step - loss: 1.5808 - val_loss: 0.6247

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data
[[0.01587212 0.02135341 0.22289534 0.73987913]
 [0.01585889 0.02134489 0.22291614 0.7398801 ]
 [0.01584145 0.02132432 0.22283413 0.7400001 ]
 ...
 [0.01606157 0.02158009 0.22354737 0.73881096]
 [0.01609111 0.02161225 0.22363888 0.7386577 ]
 [0.01608789 0.02160837 0.22362082 0.738683  ]]

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
model_keras/textcnn.json
{
  "test": {
    "model_pars": {
      "model_uri": "model_keras.textcnn.py",
      "maxlen": 40,
      "max_features": 5,
      "embedding_dims": 50
    },
    "data_pars": {
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
    },
    "compute_pars": {
      "engine": "adam",
      "loss": "binary_crossentropy",
      "metrics": [
        "accuracy"
      ],
      "batch_size": 1000,
      "epochs": 1
    },
    "out_pars": {
      "path": "mlmodels/output/textcnn_keras//model.h5",
      "model_path": "mlmodels/output/textcnn_keras/model.h5"
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.textcnn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 
Model: "model_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_3 (Embedding)         (None, 40, 50)       250         input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, 38, 128)      19328       embedding_3[0][0]                
__________________________________________________________________________________________________
conv1d_8 (Conv1D)               (None, 37, 128)      25728       embedding_3[0][0]                
__________________________________________________________________________________________________
conv1d_9 (Conv1D)               (None, 36, 128)      32128       embedding_3[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 128)          0           conv1d_7[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 128)          0           conv1d_8[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalM (None, 128)          0           conv1d_9[0][0]                   
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 384)          0           global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
                                                                 global_max_pooling1d_3[0][0]     
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 1)            385         concatenate_2[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________

  <mlmodels.model_keras.textcnn.Model object at 0x7f2cfc6d02e8> 

  #### Fit   ######################################################## 
Loading data...

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
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 8.0653 - accuracy: 0.4740
 2000/25000 [=>............................] - ETA: 7s - loss: 8.0270 - accuracy: 0.4765 
 3000/25000 [==>...........................] - ETA: 6s - loss: 8.0040 - accuracy: 0.4780
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.9388 - accuracy: 0.4823
 5000/25000 [=====>........................] - ETA: 4s - loss: 7.8844 - accuracy: 0.4858
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7816 - accuracy: 0.4925
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7827 - accuracy: 0.4924
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.7586 - accuracy: 0.4940
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7518 - accuracy: 0.4944
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7080 - accuracy: 0.4973
11000/25000 [============>.................] - ETA: 3s - loss: 7.7294 - accuracy: 0.4959
12000/25000 [=============>................] - ETA: 2s - loss: 7.7228 - accuracy: 0.4963
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7350 - accuracy: 0.4955
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7039 - accuracy: 0.4976
15000/25000 [=================>............] - ETA: 2s - loss: 7.6942 - accuracy: 0.4982
16000/25000 [==================>...........] - ETA: 1s - loss: 7.7107 - accuracy: 0.4971
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7207 - accuracy: 0.4965
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7067 - accuracy: 0.4974
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7102 - accuracy: 0.4972
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7073 - accuracy: 0.4974
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7148 - accuracy: 0.4969
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6868 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6793 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 6s 248us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Predict   #################################################### 
Loading data...

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
(array([[1.],
       [1.],
       [1.],
       ...,
       [1.],
       [1.],
       [1.]], dtype=float32), None)

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
model_keras/namentity_crm_bilstm.json
{
  "test": {
    "model_pars": {
      "model_uri": "model_keras.namentity_crm_bilstm.py",
      "embedding": 40,
      "optimizer": "rmsprop"
    },
    "data_pars": {
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
    },
    "compute_pars": {
      "epochs": 1,
      "batch_size": 64
    },
    "out_pars": {
      "path": "ztest/ml_keras/namentity_crm_bilstm/"
    }
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.namentity_crm_bilstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

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

###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f2d4980b0d0>

 ######### postional parameteres :  ['out']

 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f2d4980b0d0>

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

###### load_callable_from_uri LOADED <function train_test_split at 0x7f2dba968ea0>

 ######### postional parameteres :  []

 ######### Execute : preprocessor_func <function train_test_split at 0x7f2dba968ea0>

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

###### load_callable_from_uri LOADED <function pickle_dump at 0x7f2d5182c1e0>

 ######### postional parameteres :  ['t']

 ######### Execute : preprocessor_func <function pickle_dump at 0x7f2d5182c1e0>
######## Error model_keras/namentity_crm_bilstm.json [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



 ####################################################################################################
dataset/json/refactor/resnet18_benchmark_mnist.json
{
  "test": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub.py",
      "repo_uri": "pytorch/vision",
      "model": "resnet18",
      "num_classes": 10,
      "pretrained": 0,
      "_comment": "0: False, 1: True",
      "num_layers": 1,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
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
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 10,
      "epochs": 5,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/MNIST/restnet18/checkpoints/",
      "path": "ztest/model_tch/torchhub/MNIST/restnet18/"
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_tch.torchhub' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

  <mlmodels.model_tch.torchhub.Model object at 0x7f2cfc7465f8> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2d681d6488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2d681d6488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2d681d6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0030845001141230266 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.020037840366363524 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.0018036277045806249 	 Accuracy: 1
Train Epoch: 2 	 Loss: 0.025346020221710206 	 Accuracy: 2
Train Epoch: 3 	 Loss: 0.0023436088164647422 	 Accuracy: 1
Train Epoch: 3 	 Loss: 0.034677846431732176 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 4 	 Loss: 0.0023876243333021798 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.013331442594528198 	 Accuracy: 6
model saves at 6 accuracy
Train Epoch: 5 	 Loss: 0.0012223627157509328 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.01830833050608635 	 Accuracy: 5

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2d681d6488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2d681d6488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2d681d6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([9, 8, 8, 6, 1, 1, 1, 1, 9, 5]), array([9, 9, 2, 6, 1, 7, 1, 1, 9, 5]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
dataset/json/refactor/resnet34_benchmark_mnist.json
{
  "test": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub.py",
      "repo_uri": "pytorch/vision",
      "model": "resnet34",
      "num_classes": 10,
      "pretrained": 0,
      "_comment": "0: False, 1: True",
      "num_layers": 1,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
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
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 5,
      "epochs": 2,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/MNIST/restnet34/checkpoints/",
      "path": "ztest/model_tch/torchhub/MNIST/restnet34/"
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_tch.torchhub' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

  <mlmodels.model_tch.torchhub.Model object at 0x7f2cfbf99ac8> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2d681d6488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2d681d6488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2d681d6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0019677653114000956 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.018176914691925047 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0019242128531138103 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.05953568935394287 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2d681d6488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2d681d6488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2d681d6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8]), array([1, 7, 7, 8, 3, 6, 7, 9, 7, 5]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
dataset/json/refactor/model_list_CIFAR.json
{
  "test": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub.py",
      "repo_uri": "pytorch/vision",
      "model": "resnet18",
      "num_classes": 10,
      "pretrained": 1,
      "_comment": "0: False, 1: True",
      "num_layers": 5,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
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
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 10,
      "epochs": 2,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/cifar10/resnet18/checkpoints/",
      "path": "ztest/model_tch/torchhub/cifar10/resnet18/"
    }
  },
  "resnet152": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub.py",
      "repo_uri": "pytorch/vision",
      "model": "resnet152",
      "num_classes": 10,
      "pretrained": 1,
      "_comment": "0: False, 1: True",
      "num_layers": 5,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
      "dataset": "torchvision.datasets:CIFAR",
      "data_path": "dataset/vision/",
      "train_batch_size": 100,
      "test_batch_size": 10,
      "transform_uri": "mlmodels.preprocess.image:torch_transform_data_augment"
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 10,
      "epochs": 5,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/resnet152/checkpoints/",
      "path": "ztest/model_tch/torchhub/resnet152/"
    }
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_tch.torchhub' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/callbacks/callbacks.py:846: RuntimeWarning: Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: val_loss,loss
  (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 392, in test_run_model
    test_module(config['test']['model_pars']['model_uri'], param_pars, fittable = False if x in not_fittable_models else True)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 260, in test_module
    model = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.py", line 66, in __init__
    data_set, internal_states = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.py", line 183, in get_dataset
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 322, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 75, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth" to /home/runner/.cache/torch/checkpoints/resnet18-5c106cde.pth
  0%|          | 0.00/44.7M [00:00<?, ?B/s] 20%|█▉        | 8.73M/44.7M [00:00<00:00, 91.6MB/s] 41%|████▏     | 18.4M/44.7M [00:00<00:00, 94.4MB/s] 65%|██████▍   | 28.8M/44.7M [00:00<00:00, 98.4MB/s] 87%|████████▋ | 38.8M/44.7M [00:00<00:00, 99.9MB/s]100%|██████████| 44.7M/44.7M [00:00<00:00, 102MB/s] 
  <mlmodels.model_tch.torchhub.Model object at 0x7f2cfbfb5860> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f2db3cb22f0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f2db3cb22f0>

  function with postional parmater data_info <function tf_dataset_download at 0x7f2db3cb22f0> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2d681d6488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2d681d6488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2d681d6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.47286725759506226 	 Accuracy: 62
Train Epoch: 1 	 Loss: 15.643813323974609 	 Accuracy: 200
model saves at 200 accuracy
Train Epoch: 2 	 Loss: 0.47062973260879515 	 Accuracy: 50
Train Epoch: 2 	 Loss: 15.159478759765625 	 Accuracy: 100

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f2db3cb22f0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f2db3cb22f0>

  function with postional parmater data_info <function tf_dataset_download at 0x7f2db3cb22f0> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2d681d6488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2d681d6488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2d681d6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([1, 1, 1, 1, 1, 1, 9, 1, 1, 1]), array([5, 2, 5, 7, 3, 2, 7, 0, 0, 0]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
dataset/json/refactor/torchhub_cnn_dataloader.json
{
  "test": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub",
      "repo_uri": "facebookresearch/pytorch_GAN_zoo:hub",
      "model": "PGAN",
      "model_name": "celebAHQ-512",
      "num_classes": 1000,
      "pretrained": 0,
      "_comment": "0: False, 1: True",
      "num_layers": 1,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
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
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 5,
      "epochs": 2,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/pytorch_GAN_zoo/checkpoints/",
      "path": "ztest/model_tch/torchhub/pytorch_GAN_zoo/"
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_tch.torchhub' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

  <mlmodels.model_tch.torchhub.Model object at 0x7f2cf0eea400> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
img_01.png
0

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
dataset/json/refactor/resnet18_benchmark_FashionMNIST.json
{
  "test": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub.py",
      "repo_uri": "pytorch/vision",
      "model": "resnet18",
      "num_classes": 10,
      "pretrained": 0,
      "_comment": "0: False, 1: True",
      "num_layers": 5,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
      "data_info": {
        "data_path": "dataset/vision/FashionMNIST",
        "dataset": "FashionMNIST",
        "data_type": "tch_dataset",
        "batch_size": 10,
        "train": true
      },
      "preprocessors": [
        {
          "name": "tch_dataset_start",
          "uri": "mlmodels.preprocess.generic:get_dataset_torch",
          "args": {
            "dataloader": "torchvision.datasets:FashionMNIST",
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
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 10,
      "epochs": 5,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/FashionMNIST/resnet18/checkpoints/",
      "path": "ztest/model_tch/torchhub/FashionMNIST/resnet18/"
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_tch.torchhub' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

  <mlmodels.model_tch.torchhub.Model object at 0x7f2cfbf24fd0> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2d681d6488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2d681d6488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2d681d6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 

Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 49152/26421880 [00:00<01:39, 265228.20it/s]  1%|          | 172032/26421880 [00:00<01:17, 340706.61it/s]  2%|▏         | 450560/26421880 [00:00<00:57, 449704.10it/s]  6%|▌         | 1474560/26421880 [00:00<00:39, 628038.78it/s] 14%|█▍        | 3710976/26421880 [00:00<00:25, 882793.68it/s] 21%|██        | 5455872/26421880 [00:01<00:17, 1228516.59it/s] 37%|███▋      | 9773056/26421880 [00:01<00:09, 1733877.12it/s] 52%|█████▏    | 13811712/26421880 [00:01<00:05, 2397555.80it/s] 68%|██████▊   | 17940480/26421880 [00:01<00:02, 3341900.80it/s] 85%|████████▌ | 22585344/26421880 [00:01<00:00, 4631327.34it/s] 99%|█████████▊| 26034176/26421880 [00:01<00:00, 6019813.58it/s]26427392it [00:01, 15000916.34it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 100726.40it/s]           
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 40960/4422102 [00:00<00:12, 359365.63it/s]  1%|▏         | 65536/4422102 [00:00<00:17, 256130.32it/s]  2%|▏         | 106496/4422102 [00:00<00:15, 280948.90it/s]  3%|▎         | 139264/4422102 [00:00<00:16, 252613.52it/s]  4%|▍         | 188416/4422102 [00:00<00:16, 257538.72it/s]  5%|▌         | 237568/4422102 [00:01<00:16, 261199.08it/s]  6%|▋         | 286720/4422102 [00:01<00:13, 296875.38it/s]  7%|▋         | 319488/4422102 [00:01<00:15, 261262.88it/s]  9%|▊         | 376832/4422102 [00:01<00:14, 275326.86it/s] 10%|▉         | 434176/4422102 [00:01<00:13, 286080.78it/s] 11%|█         | 491520/4422102 [00:01<00:13, 294136.12it/s] 12%|█▏        | 548864/4422102 [00:02<00:12, 300094.46it/s] 14%|█▎        | 606208/4422102 [00:02<00:11, 342837.79it/s] 15%|█▍        | 647168/4422102 [00:02<00:12, 309796.62it/s] 16%|█▌        | 712704/4422102 [00:02<00:11, 323204.23it/s] 18%|█▊        | 778240/4422102 [00:02<00:10, 333326.99it/s] 19%|█▉        | 843776/4422102 [00:02<00:09, 383566.78it/s] 20%|██        | 892928/4422102 [00:03<00:09, 354615.16it/s] 22%|██▏       | 958464/4422102 [00:03<00:09, 356187.27it/s] 23%|██▎       | 1032192/4422102 [00:03<00:08, 414651.87it/s] 24%|██▍       | 1081344/4422102 [00:03<00:08, 371960.31it/s] 26%|██▌       | 1155072/4422102 [00:03<00:07, 430045.11it/s] 27%|██▋       | 1212416/4422102 [00:03<00:07, 401964.91it/s] 29%|██▉       | 1286144/4422102 [00:04<00:07, 402649.19it/s] 31%|███       | 1376256/4422102 [00:04<00:07, 426346.57it/s] 33%|███▎      | 1466368/4422102 [00:04<00:06, 444870.15it/s] 35%|███▌      | 1556480/4422102 [00:04<00:05, 514858.42it/s] 36%|███▋      | 1613824/4422102 [00:04<00:06, 452338.62it/s] 39%|███▊      | 1703936/4422102 [00:04<00:05, 463748.37it/s] 41%|████      | 1794048/4422102 [00:04<00:04, 533346.49it/s] 42%|████▏     | 1859584/4422102 [00:05<00:05, 485646.53it/s] 44%|████▍     | 1949696/4422102 [00:05<00:05, 488159.52it/s] 46%|████▋     | 2048000/4422102 [00:05<00:04, 565136.54it/s] 48%|████▊     | 2113536/4422102 [00:05<00:04, 502986.22it/s] 50%|█████     | 2211840/4422102 [00:05<00:03, 579206.44it/s] 52%|█████▏    | 2285568/4422102 [00:05<00:04, 530933.05it/s] 54%|█████▎    | 2375680/4422102 [00:05<00:03, 597037.45it/s] 55%|█████▌    | 2449408/4422102 [00:06<00:03, 541537.39it/s] 58%|█████▊    | 2547712/4422102 [00:06<00:03, 617334.10it/s] 59%|█████▉    | 2621440/4422102 [00:06<00:03, 552605.39it/s] 62%|██████▏   | 2719744/4422102 [00:06<00:02, 627281.85it/s] 63%|██████▎   | 2793472/4422102 [00:06<00:02, 557860.54it/s] 66%|██████▌   | 2908160/4422102 [00:06<00:02, 651745.88it/s] 68%|██████▊   | 2990080/4422102 [00:06<00:02, 594455.20it/s] 70%|███████   | 3104768/4422102 [00:07<00:01, 685717.58it/s] 72%|███████▏  | 3186688/4422102 [00:07<00:02, 614147.37it/s] 75%|███████▌  | 3317760/4422102 [00:07<00:01, 722027.12it/s] 77%|███████▋  | 3416064/4422102 [00:07<00:01, 676558.73it/s] 81%|████████  | 3563520/4422102 [00:07<00:01, 799391.29it/s] 83%|████████▎ | 3661824/4422102 [00:07<00:00, 777580.94it/s] 85%|████████▍ | 3751936/4422102 [00:07<00:00, 760835.45it/s] 89%|████████▉ | 3940352/4422102 [00:07<00:00, 919363.29it/s] 92%|█████████▏| 4055040/4422102 [00:08<00:00, 832378.90it/s] 96%|█████████▌| 4251648/4422102 [00:08<00:00, 999073.94it/s] 99%|█████████▉| 4390912/4422102 [00:08<00:00, 940360.07it/s]4423680it [00:08, 527779.53it/s]                             
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 38072.66it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0035425148804982503 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.018184763073921202 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 2 	 Loss: 0.002937537133693695 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.014144305944442749 	 Accuracy: 4
Train Epoch: 3 	 Loss: 0.0024778268237908682 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.01565396761894226 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 4 	 Loss: 0.0023685145477453868 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.016384138405323028 	 Accuracy: 5
Train Epoch: 5 	 Loss: 0.002145238806804021 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.011041519463062286 	 Accuracy: 6
model saves at 6 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2d681d6488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2d681d6488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2d681d6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([9, 4, 0, 4, 5, 1, 4, 1, 8, 7]), array([9, 6, 6, 2, 5, 1, 2, 1, 8, 7]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
dataset/json/refactor/model_list_KMNIST.json
{
  "test": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub.py",
      "repo_uri": "pytorch/vision",
      "model": "resnet18",
      "num_classes": 10,
      "pretrained": 1,
      "_comment": "0: False, 1: True",
      "num_layers": 5,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
      "data_info": {
        "data_path": "dataset/vision/KMNIST",
        "dataset": "KMNIST",
        "data_type": "tch_dataset",
        "batch_size": 10,
        "train": true
      },
      "preprocessors": [
        {
          "name": "tch_dataset_start",
          "uri": "mlmodels.preprocess.generic:get_dataset_torch",
          "args": {
            "dataloader": "torchvision.datasets:KMNIST",
            "to_image": true,
            "transform": {
              "uri": "mlmodels.preprocess.image:torch_transform_mnist",
              "pass_data_pars": false,
              "arg": {}
            },
            "shuffle": true,
            "download": true,
            "_comment": "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "
          }
        }
      ]
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 10,
      "epochs": 5,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/KMNIST/resnet18/checkpoints/",
      "path": "ztest/model_tch/torchhub/KMNIST/resnet18/"
    }
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_tch.torchhub' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

  <mlmodels.model_tch.torchhub.Model object at 0x7f2cfbe529b0> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2d681d6488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2d681d6488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2d681d6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:44, 110155.96it/s]  0%|          | 40960/18165135 [00:00<02:28, 122327.15it/s]  1%|          | 98304/18165135 [00:00<01:57, 153760.20it/s]  1%|          | 212992/18165135 [00:01<01:28, 202350.75it/s]  2%|▏         | 425984/18165135 [00:01<01:05, 272565.96it/s]  5%|▍         | 860160/18165135 [00:01<00:46, 374361.36it/s] 10%|▉         | 1736704/18165135 [00:01<00:31, 520599.21it/s] 19%|█▉        | 3481600/18165135 [00:01<00:20, 729775.60it/s] 32%|███▏      | 5832704/18165135 [00:01<00:12, 1009953.90it/s] 40%|████      | 7348224/18165135 [00:02<00:07, 1402675.82it/s] 46%|████▌     | 8273920/18165135 [00:03<00:08, 1161486.18it/s] 55%|█████▌    | 10043392/18165135 [00:03<00:05, 1593859.59it/s] 60%|█████▉    | 10878976/18165135 [00:03<00:05, 1457132.28it/s] 74%|███████▎  | 13361152/18165135 [00:04<00:02, 2006972.21it/s] 79%|███████▊  | 14303232/18165135 [00:04<00:01, 2231072.34it/s] 83%|████████▎ | 15056896/18165135 [00:04<00:01, 2663619.36it/s] 87%|████████▋ | 15728640/18165135 [00:04<00:00, 3063410.45it/s] 90%|████████▉ | 16343040/18165135 [00:04<00:00, 3325848.23it/s] 93%|█████████▎| 16900096/18165135 [00:05<00:00, 3481576.19it/s] 96%|█████████▋| 17498112/18165135 [00:05<00:00, 3655105.29it/s]18169856it [00:05, 3479418.09it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 110222.57it/s]32768it [00:00, 48676.60it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 110376.60it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 122545.79it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 154008.50it/s]  7%|▋         | 212992/3041136 [00:01<00:13, 202596.91it/s] 14%|█▍        | 425984/3041136 [00:01<00:09, 272830.18it/s] 28%|██▊       | 860160/3041136 [00:01<00:05, 374710.78it/s] 49%|████▉     | 1482752/3041136 [00:01<00:03, 515478.79it/s] 70%|███████   | 2138112/3041136 [00:01<00:01, 701093.82it/s] 92%|█████████▏| 2809856/3041136 [00:01<00:00, 938900.35it/s]3047424it [00:01, 1668505.20it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 16983.81it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.004862343827883402 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.028892497420310973 	 Accuracy: 1
model saves at 1 accuracy
Train Epoch: 2 	 Loss: 0.00362255330880483 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.041465123295783994 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 3 	 Loss: 0.0029459330042203268 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.04615710568428039 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 4 	 Loss: 0.002792078673839569 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.019110018134117127 	 Accuracy: 3
Train Epoch: 5 	 Loss: 0.003436989506085714 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.017120248138904573 	 Accuracy: 4
model saves at 4 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2d681d6488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2d681d6488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2d681d6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([2, 3, 1, 7, 9, 3, 7, 6, 0, 8]), array([1, 8, 2, 4, 9, 8, 7, 2, 4, 9]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
dataset/json/refactor/textcnn.json
{
  "test": {
    "hypermodel_pars": {},
    "data_pars": {
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
    },
    "model_pars": {
      "model_uri": "model_tch.textcnn.py",
      "dim_channel": 100,
      "kernel_height": [
        3,
        4,
        5
      ],
      "dropout_rate": 0.5,
      "num_class": 2
    },
    "compute_pars": {
      "learning_rate": 0.001,
      "epochs": 1,
      "checkpointdir": "./output/text_cnn_tch/checkpoint/"
    },
    "out_pars": {
      "path": "./output/text_cnn_tch/model.h5",
      "checkpointdir": "./output/text_cnn_tch/checkpoint/"
    }
  },
  "prod": {}
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_tch.textcnn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 
{'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}

  <mlmodels.model_tch.textcnn.Model object at 0x7f2cf92e0048> 

  #### Fit   ######################################################## 

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

###### load_callable_from_uri LOADED <function split_train_valid at 0x7f2d3fafc488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function split_train_valid at 0x7f2d3fafc488>

  function with postional parmater data_info <function split_train_valid at 0x7f2d3fafc488> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f2d3fafc598>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f2d3fafc598>

  function with postional parmater data_info <function create_tabular_dataset at 0x7f2d3fafc598> , (data_info, **args) 

  Download en 
Requirement already satisfied: en_core_web_sm==2.2.5 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (2.2.5)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011261403439981253 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.06264539957046508 	 Accuracy: 60

  model saves at 60% accuracy 

  #### Predict   #################################################### 
{'data_info': {'data_path': 'dataset/recommender/', 'dataset': 'IMDB_sample.txt', 'data_type': 'csv_dataset', 'batch_size': 64, 'train': True}, 'preprocessors': [{'uri': 'mlmodels.model_tch.textcnn:split_train_valid', 'args': {'frac': 0.99}}, {'uri': 'mlmodels.model_tch.textcnn:create_tabular_dataset', 'args': {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'}}], 'frac': 1}

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

###### load_callable_from_uri LOADED <function split_train_valid at 0x7f2d3fafc488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function split_train_valid at 0x7f2d3fafc488>

  function with postional parmater data_info <function split_train_valid at 0x7f2d3fafc488> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f2d3fafc598>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f2d3fafc598>

  function with postional parmater data_info <function create_tabular_dataset at 0x7f2d3fafc598> , (data_info, **args) 

  Download en 
Requirement already satisfied: en_core_web_sm==2.2.5 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (2.2.5)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
(tensor([1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
        1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1]), tensor([1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2,
        1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
        2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1]))

  #### Get  metrics   ################################################ 

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

###### load_callable_from_uri LOADED <function split_train_valid at 0x7f2d3fafc488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function split_train_valid at 0x7f2d3fafc488>

  function with postional parmater data_info <function split_train_valid at 0x7f2d3fafc488> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f2d3fafc598>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f2d3fafc598>

  function with postional parmater data_info <function create_tabular_dataset at 0x7f2d3fafc598> , (data_info, **args) 

  Download en 
Requirement already satisfied: en_core_web_sm==2.2.5 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (2.2.5)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  #### Save   ######################################################## 

  #### Load   ######################################################## 


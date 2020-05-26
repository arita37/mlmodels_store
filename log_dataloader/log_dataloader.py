
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/03d0494536615c7835c9ae84728e28a524b489dd', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '03d0494536615c7835c9ae84728e28a524b489dd', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/03d0494536615c7835c9ae84728e28a524b489dd

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/03d0494536615c7835c9ae84728e28a524b489dd

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/03d0494536615c7835c9ae84728e28a524b489dd

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

###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f60ccd8a8c8>

 ######### postional parameters :  ['out']

 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f60ccd8a8c8>

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

###### load_callable_from_uri LOADED <function train_test_split at 0x7f6135ee9bf8>

 ######### postional parameters :  []

 ######### Execute : preprocessor_func <function train_test_split at 0x7f6135ee9bf8>

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

###### load_callable_from_uri LOADED <function pickle_dump at 0x7f60e28b1598>

 ######### postional parameters :  ['t']

 ######### Execute : preprocessor_func <function pickle_dump at 0x7f60e28b1598>
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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f60e3766268>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f60e3766268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f60e3766268> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 154844.74it/s] 82%|████████▏ | 8134656/9912422 [00:00<00:08, 221026.09it/s]9920512it [00:00, 44588369.17it/s]                           
0it [00:00, ?it/s]32768it [00:00, 456089.60it/s]
0it [00:00, ?it/s]1654784it [00:00, 21263842.84it/s]
0it [00:00, ?it/s]8192it [00:00, 177321.37it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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
((<torch.utils.data.dataloader.DataLoader object at 0x7f60ccd5a748>, <torch.utils.data.dataloader.DataLoader object at 0x7f60bb163a58>), {})





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

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f612f232048>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f612f232048>

  function with postional parmater data_info <function tf_dataset_download at 0x7f612f232048> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:12,  2.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:12,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:11,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:11,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:11,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:10,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:50,  3.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:50,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:49,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:49,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:49,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:48,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:00<00:34,  4.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:34,  4.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:34,  4.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:34,  4.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:34,  4.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:33,  4.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:33,  4.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:23,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:23,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:17,  7.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:17,  7.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:17,  7.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:17,  7.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:17,  7.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  7.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:17,  7.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:01<00:12, 10.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:12, 10.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:12, 10.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:12, 10.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:12, 10.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:12, 10.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:12, 10.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:01<00:09, 13.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:09, 13.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:09, 13.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:09, 13.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:09, 13.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:08, 13.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:07, 17.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:07, 17.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:07, 17.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:07, 17.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:06, 17.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:06, 17.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 17.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:05, 21.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:05, 21.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:05, 21.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:05, 21.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 21.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 21.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 21.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 25.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 25.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 25.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 25.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 25.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 25.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 29.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 29.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 29.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 29.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 29.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 29.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 29.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 33.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 33.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 33.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 33.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 33.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 33.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 33.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 37.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 37.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 37.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 37.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 37.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 37.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 37.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 37.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 37.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 37.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 37.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 37.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 37.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 40.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 40.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 40.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 40.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:02, 40.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 43.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 43.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 43.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 43.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 43.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 43.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 42.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 42.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 42.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 42.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 42.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 42.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 44.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 44.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 44.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 44.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 44.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 44.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 45.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 44.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 44.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 44.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 44.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 44.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 44.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 44.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 47.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 47.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 47.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:01, 47.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:01, 47.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:01, 47.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:02<00:01, 44.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:01, 44.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:01, 44.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:01, 44.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 44.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 44.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 45.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 45.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 45.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 45.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 45.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 45.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 46.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 47.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 47.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 47.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 47.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 47.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 47.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 47.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 47.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 47.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 47.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 53.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 53.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 53.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 53.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 53.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 53.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 53.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 53.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 53.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 57.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 57.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 60.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 60.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 60.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 60.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 60.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 60.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 60.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 60.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 60.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.65s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.65s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 60.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.65s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 60.98 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.69s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.65s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 60.98 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.69s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.69s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 28.44 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.70s/ url]
0 examples [00:00, ? examples/s]2020-05-26 13:21:32.810050: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-26 13:21:32.814464: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-26 13:21:32.814605: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5575ac04ba40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-26 13:21:32.814619: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
67 examples [00:00, 664.97 examples/s]166 examples [00:00, 735.79 examples/s]261 examples [00:00, 786.84 examples/s]359 examples [00:00, 835.17 examples/s]457 examples [00:00, 872.04 examples/s]554 examples [00:00, 898.99 examples/s]657 examples [00:00, 933.08 examples/s]758 examples [00:00, 954.64 examples/s]861 examples [00:00, 975.02 examples/s]961 examples [00:01, 981.82 examples/s]1060 examples [00:01, 982.93 examples/s]1161 examples [00:01, 988.96 examples/s]1260 examples [00:01, 981.51 examples/s]1366 examples [00:01, 1001.58 examples/s]1466 examples [00:01, 993.17 examples/s] 1566 examples [00:01, 974.74 examples/s]1664 examples [00:01, 942.14 examples/s]1759 examples [00:01, 938.42 examples/s]1855 examples [00:01, 944.14 examples/s]1963 examples [00:02, 980.90 examples/s]2070 examples [00:02, 1004.22 examples/s]2176 examples [00:02, 1018.72 examples/s]2279 examples [00:02, 1002.08 examples/s]2380 examples [00:02, 1000.00 examples/s]2481 examples [00:02, 968.94 examples/s] 2579 examples [00:02, 965.18 examples/s]2680 examples [00:02, 974.89 examples/s]2779 examples [00:02, 976.54 examples/s]2877 examples [00:02, 959.38 examples/s]2974 examples [00:03, 960.30 examples/s]3071 examples [00:03, 957.11 examples/s]3167 examples [00:03, 950.30 examples/s]3263 examples [00:03, 950.88 examples/s]3361 examples [00:03, 959.29 examples/s]3469 examples [00:03, 990.85 examples/s]3576 examples [00:03, 1012.09 examples/s]3684 examples [00:03, 1029.64 examples/s]3788 examples [00:03, 983.35 examples/s] 3887 examples [00:03, 976.24 examples/s]3987 examples [00:04, 982.76 examples/s]4089 examples [00:04, 991.86 examples/s]4192 examples [00:04, 1002.87 examples/s]4302 examples [00:04, 1027.77 examples/s]4406 examples [00:04, 1024.27 examples/s]4513 examples [00:04, 1037.44 examples/s]4617 examples [00:04, 1033.37 examples/s]4721 examples [00:04, 1028.42 examples/s]4825 examples [00:04, 1030.69 examples/s]4932 examples [00:05, 1040.81 examples/s]5037 examples [00:05, 1003.00 examples/s]5138 examples [00:05, 972.57 examples/s] 5236 examples [00:05, 971.29 examples/s]5334 examples [00:05, 765.10 examples/s]5438 examples [00:05, 830.81 examples/s]5545 examples [00:05, 889.19 examples/s]5648 examples [00:05, 926.03 examples/s]5746 examples [00:05, 933.91 examples/s]5846 examples [00:06, 951.05 examples/s]5945 examples [00:06, 960.05 examples/s]6045 examples [00:06, 971.34 examples/s]6144 examples [00:06, 963.52 examples/s]6250 examples [00:06, 988.37 examples/s]6354 examples [00:06, 1000.50 examples/s]6463 examples [00:06, 1023.68 examples/s]6566 examples [00:06, 1021.01 examples/s]6669 examples [00:06, 1006.17 examples/s]6770 examples [00:06, 1005.66 examples/s]6872 examples [00:07, 1009.10 examples/s]6974 examples [00:07, 1007.82 examples/s]7076 examples [00:07, 1010.32 examples/s]7178 examples [00:07, 995.39 examples/s] 7278 examples [00:07, 986.64 examples/s]7378 examples [00:07, 989.33 examples/s]7477 examples [00:07, 968.60 examples/s]7575 examples [00:07, 968.86 examples/s]7672 examples [00:07, 959.00 examples/s]7770 examples [00:07, 964.82 examples/s]7875 examples [00:08, 988.34 examples/s]7975 examples [00:08, 972.19 examples/s]8078 examples [00:08, 985.62 examples/s]8177 examples [00:08, 966.29 examples/s]8274 examples [00:08, 948.39 examples/s]8374 examples [00:08, 962.08 examples/s]8478 examples [00:08, 981.17 examples/s]8582 examples [00:08, 997.43 examples/s]8682 examples [00:08, 984.55 examples/s]8781 examples [00:08, 984.56 examples/s]8881 examples [00:09, 987.02 examples/s]8980 examples [00:09, 981.71 examples/s]9079 examples [00:09, 979.40 examples/s]9177 examples [00:09, 953.59 examples/s]9282 examples [00:09, 979.50 examples/s]9384 examples [00:09, 990.12 examples/s]9486 examples [00:09, 998.12 examples/s]9586 examples [00:09, 986.59 examples/s]9685 examples [00:09, 978.05 examples/s]9793 examples [00:10, 1005.59 examples/s]9895 examples [00:10, 1006.89 examples/s]9996 examples [00:10, 997.97 examples/s] 10096 examples [00:10, 911.19 examples/s]10189 examples [00:10, 913.60 examples/s]10282 examples [00:10, 907.38 examples/s]10374 examples [00:10, 903.63 examples/s]10465 examples [00:10, 901.28 examples/s]10556 examples [00:10, 900.31 examples/s]10662 examples [00:10, 942.37 examples/s]10768 examples [00:11, 972.35 examples/s]10872 examples [00:11, 990.23 examples/s]10979 examples [00:11, 1010.60 examples/s]11083 examples [00:11, 1016.51 examples/s]11186 examples [00:11, 1016.24 examples/s]11288 examples [00:11, 1002.84 examples/s]11393 examples [00:11, 1016.02 examples/s]11495 examples [00:11, 994.88 examples/s] 11597 examples [00:11, 1000.25 examples/s]11698 examples [00:11, 981.76 examples/s] 11805 examples [00:12, 1005.02 examples/s]11914 examples [00:12, 1026.37 examples/s]12021 examples [00:12, 1037.02 examples/s]12125 examples [00:12, 1016.98 examples/s]12227 examples [00:12, 979.20 examples/s] 12326 examples [00:12, 960.31 examples/s]12425 examples [00:12, 968.73 examples/s]12531 examples [00:12, 993.55 examples/s]12636 examples [00:12, 1006.56 examples/s]12737 examples [00:13, 979.63 examples/s] 12836 examples [00:13, 981.93 examples/s]12935 examples [00:13, 979.63 examples/s]13042 examples [00:13, 1004.10 examples/s]13143 examples [00:13, 978.61 examples/s] 13242 examples [00:13, 974.70 examples/s]13349 examples [00:13, 999.35 examples/s]13450 examples [00:13, 1001.95 examples/s]13551 examples [00:13, 1002.92 examples/s]13652 examples [00:13, 987.33 examples/s] 13751 examples [00:14, 984.31 examples/s]13852 examples [00:14, 991.37 examples/s]13952 examples [00:14, 990.64 examples/s]14052 examples [00:14, 980.62 examples/s]14151 examples [00:14, 956.40 examples/s]14247 examples [00:14, 956.25 examples/s]14359 examples [00:14, 998.56 examples/s]14461 examples [00:14, 1003.13 examples/s]14564 examples [00:14, 1009.99 examples/s]14666 examples [00:14, 1001.58 examples/s]14767 examples [00:15, 969.19 examples/s] 14865 examples [00:15, 971.86 examples/s]14963 examples [00:15, 970.35 examples/s]15061 examples [00:15, 969.43 examples/s]15159 examples [00:15, 933.47 examples/s]15255 examples [00:15, 939.21 examples/s]15352 examples [00:15, 948.23 examples/s]15450 examples [00:15, 954.76 examples/s]15547 examples [00:15, 957.94 examples/s]15643 examples [00:16, 943.22 examples/s]15743 examples [00:16, 957.26 examples/s]15844 examples [00:16, 970.82 examples/s]15946 examples [00:16, 982.50 examples/s]16046 examples [00:16, 984.43 examples/s]16145 examples [00:16, 981.26 examples/s]16244 examples [00:16, 982.65 examples/s]16343 examples [00:16, 972.98 examples/s]16441 examples [00:16, 974.69 examples/s]16539 examples [00:16, 975.10 examples/s]16637 examples [00:17, 966.29 examples/s]16734 examples [00:17, 963.21 examples/s]16836 examples [00:17, 976.59 examples/s]16934 examples [00:17, 969.59 examples/s]17032 examples [00:17, 972.19 examples/s]17130 examples [00:17, 942.56 examples/s]17228 examples [00:17, 952.34 examples/s]17326 examples [00:17, 957.48 examples/s]17423 examples [00:17, 958.75 examples/s]17519 examples [00:17, 953.65 examples/s]17615 examples [00:18, 940.86 examples/s]17710 examples [00:18, 925.24 examples/s]17804 examples [00:18, 925.80 examples/s]17898 examples [00:18, 927.27 examples/s]17991 examples [00:18, 900.25 examples/s]18082 examples [00:18, 879.69 examples/s]18182 examples [00:18, 912.62 examples/s]18278 examples [00:18, 924.21 examples/s]18373 examples [00:18, 930.23 examples/s]18467 examples [00:18, 930.69 examples/s]18561 examples [00:19, 916.53 examples/s]18658 examples [00:19, 931.63 examples/s]18752 examples [00:19, 913.50 examples/s]18849 examples [00:19, 927.58 examples/s]18945 examples [00:19, 935.20 examples/s]19039 examples [00:19, 891.91 examples/s]19131 examples [00:19, 898.55 examples/s]19224 examples [00:19, 906.69 examples/s]19316 examples [00:19, 910.05 examples/s]19410 examples [00:20, 916.94 examples/s]19502 examples [00:20, 910.98 examples/s]19596 examples [00:20, 917.06 examples/s]19691 examples [00:20, 926.57 examples/s]19785 examples [00:20, 928.08 examples/s]19883 examples [00:20, 942.97 examples/s]19978 examples [00:20, 943.93 examples/s]20073 examples [00:20, 896.71 examples/s]20170 examples [00:20, 916.13 examples/s]20263 examples [00:20, 906.60 examples/s]20357 examples [00:21, 915.16 examples/s]20451 examples [00:21, 921.66 examples/s]20556 examples [00:21, 956.50 examples/s]20653 examples [00:21, 954.97 examples/s]20749 examples [00:21, 944.07 examples/s]20847 examples [00:21, 952.07 examples/s]20943 examples [00:21, 919.56 examples/s]21037 examples [00:21, 924.78 examples/s]21130 examples [00:21, 913.87 examples/s]21226 examples [00:21, 925.97 examples/s]21319 examples [00:22, 920.19 examples/s]21413 examples [00:22, 924.45 examples/s]21510 examples [00:22, 935.36 examples/s]21604 examples [00:22, 925.80 examples/s]21699 examples [00:22, 931.49 examples/s]21794 examples [00:22, 936.69 examples/s]21888 examples [00:22, 930.57 examples/s]21984 examples [00:22, 938.24 examples/s]22082 examples [00:22, 947.39 examples/s]22178 examples [00:22, 950.52 examples/s]22274 examples [00:23, 934.34 examples/s]22368 examples [00:23, 933.73 examples/s]22462 examples [00:23, 931.51 examples/s]22557 examples [00:23, 934.58 examples/s]22651 examples [00:23, 919.24 examples/s]22745 examples [00:23, 923.48 examples/s]22838 examples [00:23, 917.92 examples/s]22937 examples [00:23, 937.16 examples/s]23042 examples [00:23, 966.00 examples/s]23143 examples [00:24, 978.07 examples/s]23245 examples [00:24, 988.17 examples/s]23345 examples [00:24, 991.53 examples/s]23447 examples [00:24, 998.72 examples/s]23547 examples [00:24, 993.53 examples/s]23653 examples [00:24, 1011.21 examples/s]23765 examples [00:24, 1041.36 examples/s]23870 examples [00:24, 998.18 examples/s] 23971 examples [00:24, 990.21 examples/s]24071 examples [00:24, 963.06 examples/s]24168 examples [00:25, 954.85 examples/s]24264 examples [00:25, 955.36 examples/s]24360 examples [00:25, 954.39 examples/s]24469 examples [00:25, 989.61 examples/s]24574 examples [00:25, 1004.75 examples/s]24676 examples [00:25, 1007.22 examples/s]24777 examples [00:25, 994.40 examples/s] 24877 examples [00:25, 975.49 examples/s]24978 examples [00:25, 985.30 examples/s]25077 examples [00:25, 984.76 examples/s]25180 examples [00:26, 996.68 examples/s]25285 examples [00:26, 1009.80 examples/s]25388 examples [00:26, 1012.96 examples/s]25494 examples [00:26, 1026.35 examples/s]25597 examples [00:26, 1014.03 examples/s]25699 examples [00:26, 1006.35 examples/s]25805 examples [00:26, 1021.10 examples/s]25908 examples [00:26, 992.30 examples/s] 26008 examples [00:26, 954.19 examples/s]26104 examples [00:26, 941.02 examples/s]26199 examples [00:27, 940.37 examples/s]26297 examples [00:27, 949.36 examples/s]26393 examples [00:27, 917.85 examples/s]26486 examples [00:27, 918.41 examples/s]26579 examples [00:27, 921.34 examples/s]26678 examples [00:27, 939.24 examples/s]26773 examples [00:27, 939.50 examples/s]26868 examples [00:27, 935.55 examples/s]26971 examples [00:27, 959.81 examples/s]27068 examples [00:28, 962.67 examples/s]27168 examples [00:28, 969.54 examples/s]27266 examples [00:28, 945.52 examples/s]27362 examples [00:28, 949.07 examples/s]27458 examples [00:28, 943.01 examples/s]27553 examples [00:28, 933.27 examples/s]27647 examples [00:28, 916.57 examples/s]27751 examples [00:28, 947.10 examples/s]27847 examples [00:28, 949.13 examples/s]27945 examples [00:28, 956.61 examples/s]28041 examples [00:29, 950.77 examples/s]28142 examples [00:29, 965.45 examples/s]28243 examples [00:29, 975.91 examples/s]28346 examples [00:29, 988.64 examples/s]28454 examples [00:29, 1012.74 examples/s]28556 examples [00:29, 1011.65 examples/s]28661 examples [00:29, 1021.14 examples/s]28764 examples [00:29, 993.17 examples/s] 28864 examples [00:29, 975.58 examples/s]28962 examples [00:29, 926.99 examples/s]29056 examples [00:30, 919.23 examples/s]29155 examples [00:30, 938.05 examples/s]29252 examples [00:30, 946.01 examples/s]29348 examples [00:30, 948.68 examples/s]29456 examples [00:30, 984.58 examples/s]29556 examples [00:30, 987.85 examples/s]29656 examples [00:30, 991.37 examples/s]29756 examples [00:30, 975.69 examples/s]29854 examples [00:30, 959.86 examples/s]29951 examples [00:31, 959.86 examples/s]30048 examples [00:31, 912.77 examples/s]30150 examples [00:31, 941.61 examples/s]30245 examples [00:31, 933.30 examples/s]30353 examples [00:31, 970.99 examples/s]30451 examples [00:31, 971.53 examples/s]30549 examples [00:31, 962.67 examples/s]30646 examples [00:31, 964.53 examples/s]30743 examples [00:31, 949.96 examples/s]30839 examples [00:31, 950.97 examples/s]30935 examples [00:32, 919.84 examples/s]31035 examples [00:32, 942.03 examples/s]31138 examples [00:32, 965.19 examples/s]31235 examples [00:32, 947.43 examples/s]31333 examples [00:32, 955.63 examples/s]31429 examples [00:32, 952.15 examples/s]31525 examples [00:32, 946.06 examples/s]31625 examples [00:32, 961.39 examples/s]31722 examples [00:32, 959.81 examples/s]31819 examples [00:32, 961.10 examples/s]31916 examples [00:33, 958.15 examples/s]32012 examples [00:33, 956.33 examples/s]32108 examples [00:33, 954.96 examples/s]32210 examples [00:33, 973.52 examples/s]32308 examples [00:33, 962.98 examples/s]32414 examples [00:33, 989.65 examples/s]32515 examples [00:33, 992.78 examples/s]32615 examples [00:33, 963.37 examples/s]32712 examples [00:33, 956.48 examples/s]32808 examples [00:34, 895.11 examples/s]32901 examples [00:34, 904.07 examples/s]33008 examples [00:34, 945.86 examples/s]33108 examples [00:34, 958.93 examples/s]33205 examples [00:34, 955.74 examples/s]33306 examples [00:34, 968.47 examples/s]33410 examples [00:34, 986.34 examples/s]33518 examples [00:34, 1010.48 examples/s]33624 examples [00:34, 1024.57 examples/s]33727 examples [00:34, 1017.94 examples/s]33833 examples [00:35, 1027.81 examples/s]33936 examples [00:35, 1023.95 examples/s]34039 examples [00:35, 1017.99 examples/s]34143 examples [00:35, 1022.43 examples/s]34246 examples [00:35, 1002.39 examples/s]34348 examples [00:35, 1006.09 examples/s]34449 examples [00:35, 993.60 examples/s] 34549 examples [00:35, 990.54 examples/s]34650 examples [00:35, 992.30 examples/s]34750 examples [00:35, 974.82 examples/s]34848 examples [00:36, 972.65 examples/s]34946 examples [00:36, 974.77 examples/s]35052 examples [00:36, 997.53 examples/s]35152 examples [00:36, 972.66 examples/s]35252 examples [00:36, 980.37 examples/s]35360 examples [00:36, 1006.27 examples/s]35462 examples [00:36, 1010.06 examples/s]35569 examples [00:36, 1025.98 examples/s]35672 examples [00:36, 1012.45 examples/s]35774 examples [00:36, 996.76 examples/s] 35874 examples [00:37, 962.70 examples/s]35975 examples [00:37, 975.50 examples/s]36073 examples [00:37, 971.11 examples/s]36171 examples [00:37, 921.62 examples/s]36265 examples [00:37, 925.63 examples/s]36363 examples [00:37, 938.98 examples/s]36458 examples [00:37, 937.46 examples/s]36553 examples [00:37, 934.64 examples/s]36647 examples [00:37, 923.87 examples/s]36746 examples [00:38, 941.22 examples/s]36849 examples [00:38, 965.90 examples/s]36946 examples [00:38, 938.39 examples/s]37041 examples [00:38, 935.55 examples/s]37135 examples [00:38, 932.11 examples/s]37229 examples [00:38, 922.26 examples/s]37322 examples [00:38, 901.02 examples/s]37424 examples [00:38, 931.99 examples/s]37524 examples [00:38, 949.25 examples/s]37620 examples [00:38, 936.26 examples/s]37716 examples [00:39, 940.53 examples/s]37815 examples [00:39, 953.37 examples/s]37919 examples [00:39, 977.22 examples/s]38027 examples [00:39, 1003.96 examples/s]38129 examples [00:39, 1007.31 examples/s]38231 examples [00:39, 1008.97 examples/s]38338 examples [00:39, 1026.22 examples/s]38441 examples [00:39, 1019.46 examples/s]38544 examples [00:39, 1014.23 examples/s]38646 examples [00:39, 993.01 examples/s] 38748 examples [00:40, 998.02 examples/s]38848 examples [00:40, 957.60 examples/s]38946 examples [00:40, 963.63 examples/s]39043 examples [00:40, 965.27 examples/s]39140 examples [00:40, 957.57 examples/s]39241 examples [00:40, 970.84 examples/s]39350 examples [00:40, 1002.00 examples/s]39454 examples [00:40, 1011.31 examples/s]39562 examples [00:40, 1030.30 examples/s]39666 examples [00:41, 1000.67 examples/s]39767 examples [00:41, 1002.51 examples/s]39872 examples [00:41, 1013.66 examples/s]39974 examples [00:41, 996.82 examples/s] 40074 examples [00:41, 911.72 examples/s]40170 examples [00:41, 924.66 examples/s]40271 examples [00:41, 946.47 examples/s]40372 examples [00:41, 963.92 examples/s]40474 examples [00:41, 979.75 examples/s]40573 examples [00:41, 972.28 examples/s]40671 examples [00:42, 970.31 examples/s]40775 examples [00:42, 988.85 examples/s]40877 examples [00:42, 997.33 examples/s]40977 examples [00:42, 984.64 examples/s]41078 examples [00:42, 991.95 examples/s]41178 examples [00:42, 971.51 examples/s]41276 examples [00:42, 960.08 examples/s]41373 examples [00:42, 947.34 examples/s]41468 examples [00:42, 940.90 examples/s]41563 examples [00:42, 930.95 examples/s]41657 examples [00:43, 927.47 examples/s]41753 examples [00:43, 936.53 examples/s]41850 examples [00:43, 943.97 examples/s]41948 examples [00:43, 952.28 examples/s]42047 examples [00:43, 963.16 examples/s]42144 examples [00:43, 935.47 examples/s]42238 examples [00:43, 934.80 examples/s]42335 examples [00:43, 943.59 examples/s]42435 examples [00:43, 956.42 examples/s]42531 examples [00:44, 953.60 examples/s]42627 examples [00:44, 951.51 examples/s]42734 examples [00:44, 982.87 examples/s]42836 examples [00:44, 990.60 examples/s]42936 examples [00:44, 977.44 examples/s]43036 examples [00:44, 982.04 examples/s]43135 examples [00:44, 967.57 examples/s]43232 examples [00:44, 966.14 examples/s]43332 examples [00:44, 973.43 examples/s]43431 examples [00:44, 975.89 examples/s]43529 examples [00:45, 963.16 examples/s]43626 examples [00:45, 951.99 examples/s]43722 examples [00:45, 918.38 examples/s]43817 examples [00:45, 927.16 examples/s]43911 examples [00:45, 930.93 examples/s]44010 examples [00:45, 946.08 examples/s]44105 examples [00:45, 926.94 examples/s]44199 examples [00:45, 930.79 examples/s]44296 examples [00:45, 939.91 examples/s]44391 examples [00:45, 942.05 examples/s]44486 examples [00:46, 943.60 examples/s]44581 examples [00:46, 940.48 examples/s]44676 examples [00:46, 924.03 examples/s]44779 examples [00:46, 952.66 examples/s]44879 examples [00:46, 964.75 examples/s]44979 examples [00:46, 972.49 examples/s]45077 examples [00:46, 957.01 examples/s]45173 examples [00:46, 951.73 examples/s]45269 examples [00:46, 923.29 examples/s]45362 examples [00:46, 898.72 examples/s]45453 examples [00:47, 880.08 examples/s]45542 examples [00:47, 882.41 examples/s]45631 examples [00:47, 880.04 examples/s]45727 examples [00:47, 899.39 examples/s]45829 examples [00:47, 930.72 examples/s]45923 examples [00:47, 926.59 examples/s]46017 examples [00:47, 928.13 examples/s]46112 examples [00:47, 934.00 examples/s]46206 examples [00:47, 921.78 examples/s]46299 examples [00:48, 919.87 examples/s]46392 examples [00:48, 908.17 examples/s]46483 examples [00:48, 905.78 examples/s]46577 examples [00:48, 915.62 examples/s]46669 examples [00:48, 915.00 examples/s]46761 examples [00:48, 915.14 examples/s]46855 examples [00:48, 920.55 examples/s]46953 examples [00:48, 935.14 examples/s]47050 examples [00:48, 942.41 examples/s]47150 examples [00:48, 956.12 examples/s]47246 examples [00:49, 954.40 examples/s]47342 examples [00:49, 938.51 examples/s]47436 examples [00:49, 929.51 examples/s]47530 examples [00:49, 911.21 examples/s]47631 examples [00:49, 937.26 examples/s]47727 examples [00:49, 943.00 examples/s]47822 examples [00:49, 922.27 examples/s]47919 examples [00:49, 935.27 examples/s]48013 examples [00:49, 933.93 examples/s]48107 examples [00:49, 918.22 examples/s]48199 examples [00:50, 900.05 examples/s]48291 examples [00:50, 905.07 examples/s]48383 examples [00:50, 906.47 examples/s]48474 examples [00:50, 898.57 examples/s]48576 examples [00:50, 930.66 examples/s]48677 examples [00:50, 952.10 examples/s]48773 examples [00:50, 950.92 examples/s]48873 examples [00:50, 962.66 examples/s]48970 examples [00:50, 958.34 examples/s]49066 examples [00:50, 946.69 examples/s]49161 examples [00:51, 947.04 examples/s]49256 examples [00:51, 945.97 examples/s]49356 examples [00:51, 960.32 examples/s]49457 examples [00:51, 973.33 examples/s]49556 examples [00:51, 977.26 examples/s]49654 examples [00:51, 973.89 examples/s]49752 examples [00:51, 952.01 examples/s]49855 examples [00:51, 972.84 examples/s]49953 examples [00:51, 959.90 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7275/50000 [00:00<00:00, 72747.29 examples/s] 38%|███▊      | 18956/50000 [00:00<00:00, 82029.86 examples/s] 62%|██████▏   | 30789/50000 [00:00<00:00, 90343.53 examples/s] 85%|████████▌ | 42506/50000 [00:00<00:00, 97005.28 examples/s]                                                               0 examples [00:00, ? examples/s]78 examples [00:00, 777.63 examples/s]181 examples [00:00, 838.50 examples/s]275 examples [00:00, 864.78 examples/s]378 examples [00:00, 907.75 examples/s]473 examples [00:00, 919.42 examples/s]567 examples [00:00, 924.78 examples/s]666 examples [00:00, 940.79 examples/s]763 examples [00:00, 948.47 examples/s]863 examples [00:00, 962.76 examples/s]958 examples [00:01, 958.68 examples/s]1053 examples [00:01, 955.78 examples/s]1148 examples [00:01, 952.00 examples/s]1243 examples [00:01, 934.30 examples/s]1346 examples [00:01, 960.23 examples/s]1444 examples [00:01, 963.83 examples/s]1542 examples [00:01, 968.61 examples/s]1639 examples [00:01, 963.12 examples/s]1736 examples [00:01, 951.50 examples/s]1832 examples [00:01, 948.25 examples/s]1936 examples [00:02, 972.61 examples/s]2034 examples [00:02, 972.44 examples/s]2136 examples [00:02, 983.86 examples/s]2235 examples [00:02, 974.64 examples/s]2334 examples [00:02, 978.33 examples/s]2432 examples [00:02, 962.77 examples/s]2530 examples [00:02, 967.18 examples/s]2631 examples [00:02, 978.11 examples/s]2729 examples [00:02, 952.54 examples/s]2825 examples [00:02, 951.69 examples/s]2921 examples [00:03, 950.18 examples/s]3021 examples [00:03, 963.52 examples/s]3118 examples [00:03, 959.74 examples/s]3215 examples [00:03, 960.37 examples/s]3314 examples [00:03, 968.04 examples/s]3411 examples [00:03, 950.77 examples/s]3511 examples [00:03, 963.38 examples/s]3612 examples [00:03, 975.62 examples/s]3710 examples [00:03, 965.67 examples/s]3815 examples [00:03, 988.16 examples/s]3917 examples [00:04, 996.79 examples/s]4017 examples [00:04, 994.61 examples/s]4118 examples [00:04, 998.19 examples/s]4218 examples [00:04, 989.67 examples/s]4319 examples [00:04, 993.69 examples/s]4419 examples [00:04, 963.21 examples/s]4518 examples [00:04, 969.96 examples/s]4616 examples [00:04, 959.57 examples/s]4713 examples [00:04, 956.82 examples/s]4809 examples [00:04, 937.05 examples/s]4903 examples [00:05, 922.61 examples/s]4996 examples [00:05, 903.52 examples/s]5097 examples [00:05, 930.95 examples/s]5193 examples [00:05, 938.92 examples/s]5289 examples [00:05, 942.24 examples/s]5385 examples [00:05, 945.69 examples/s]5484 examples [00:05, 956.90 examples/s]5583 examples [00:05, 962.69 examples/s]5680 examples [00:05, 931.76 examples/s]5774 examples [00:06, 934.01 examples/s]5873 examples [00:06, 949.91 examples/s]5969 examples [00:06, 936.59 examples/s]6063 examples [00:06, 919.68 examples/s]6160 examples [00:06, 933.81 examples/s]6254 examples [00:06, 933.37 examples/s]6348 examples [00:06, 932.71 examples/s]6442 examples [00:06, 924.63 examples/s]6541 examples [00:06, 940.35 examples/s]6636 examples [00:06, 924.53 examples/s]6735 examples [00:07, 940.25 examples/s]6830 examples [00:07, 941.40 examples/s]6925 examples [00:07, 938.79 examples/s]7027 examples [00:07, 959.73 examples/s]7124 examples [00:07, 950.67 examples/s]7220 examples [00:07, 947.27 examples/s]7317 examples [00:07, 953.37 examples/s]7413 examples [00:07, 954.48 examples/s]7509 examples [00:07, 950.87 examples/s]7605 examples [00:07, 942.16 examples/s]7703 examples [00:08, 952.94 examples/s]7805 examples [00:08, 970.30 examples/s]7906 examples [00:08, 979.03 examples/s]8009 examples [00:08, 993.33 examples/s]8109 examples [00:08, 965.78 examples/s]8206 examples [00:08, 940.67 examples/s]8301 examples [00:08, 935.23 examples/s]8395 examples [00:08, 926.76 examples/s]8494 examples [00:08, 944.30 examples/s]8590 examples [00:08, 948.95 examples/s]8686 examples [00:09, 939.08 examples/s]8783 examples [00:09, 945.15 examples/s]8878 examples [00:09, 936.54 examples/s]8972 examples [00:09, 935.92 examples/s]9066 examples [00:09, 934.31 examples/s]9161 examples [00:09, 936.59 examples/s]9262 examples [00:09, 957.24 examples/s]9364 examples [00:09, 973.99 examples/s]9462 examples [00:09, 964.25 examples/s]9561 examples [00:10, 971.82 examples/s]9659 examples [00:10, 959.29 examples/s]9756 examples [00:10, 957.21 examples/s]9858 examples [00:10, 975.05 examples/s]9956 examples [00:10, 970.90 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteZ5J3GZ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteZ5J3GZ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f60e3766268>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f60e3766268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f60e3766268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

 #####  get_Data DataLoader 
((<torch.utils.data.dataloader.DataLoader object at 0x7f60e28a7be0>, <torch.utils.data.dataloader.DataLoader object at 0x7f60c49a3208>), {})





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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f60e3766268>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f60e3766268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f60e3766268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

 #####  get_Data DataLoader 
((<torch.utils.data.dataloader.DataLoader object at 0x7f60c4a07828>, <torch.utils.data.dataloader.DataLoader object at 0x7f60c49e11d0>), {})





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

###### load_callable_from_uri LOADED <function split_train_valid at 0x7f60c1eefd90>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function split_train_valid at 0x7f60c1eefd90>

  function with postional parmater data_info <function split_train_valid at 0x7f60c1eefd90> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f60c1eefea0>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f60c1eefea0>

  function with postional parmater data_info <function create_tabular_dataset at 0x7f60c1eefea0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=0c53f9ce869ac205d0b0f1100475ce1e7df794a78ecf6c36ec9985cf3363a0c2
  Stored in directory: /tmp/pip-ephem-wheel-cache-a5t9na0y/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:49:20, 10.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:54:27, 14.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:53:18, 20.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:19:46, 28.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<5:48:54, 41.0kB/s].vector_cache/glove.6B.zip:   1%|          | 8.23M/862M [00:01<4:03:01, 58.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:01<2:49:23, 83.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.1M/862M [00:01<1:57:53, 119kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.8M/862M [00:01<1:22:03, 170kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.6M/862M [00:01<57:12, 243kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 32.1M/862M [00:02<40:00, 346kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.7M/862M [00:02<27:56, 493kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.4M/862M [00:02<19:34, 700kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.5M/862M [00:02<13:44, 992kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.0M/862M [00:02<09:39, 1.40MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<07:07, 1.90MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<06:53, 1.95MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<08:40, 1.55MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<06:50, 1.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.5M/862M [00:05<04:58, 2.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<08:05, 1.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<07:15, 1.84MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<05:27, 2.44MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<06:53, 1.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<05:22, 2.47MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<03:54, 3.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<32:28, 407kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<24:07, 548kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<17:12, 767kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<15:02, 875kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<13:05, 1.01MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<09:43, 1.35MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:12<06:59, 1.87MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<09:59, 1.31MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<08:21, 1.57MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:14<06:06, 2.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<07:20, 1.77MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<07:48, 1.67MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<06:01, 2.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.8M/862M [00:16<04:22, 2.96MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<10:17, 1.26MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<08:30, 1.52MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:18<06:14, 2.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<07:23, 1.74MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<07:48, 1.65MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<06:06, 2.11MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<06:20, 2.02MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<05:44, 2.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<04:20, 2.95MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.0M/862M [00:24<06:00, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<06:47, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<05:17, 2.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:24<03:54, 3.26MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:32, 1.68MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:35, 1.92MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:55, 2.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:24, 1.97MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:45, 2.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:17, 2.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:59, 2.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:38, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:17, 2.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<05:42, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:17, 2.36MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<04:01, 3.10MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:41, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:14, 2.37MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<03:58, 3.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:41, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:43, 1.84MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:23, 2.28MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<03:54, 3.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<14:24, 852kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<11:27, 1.07MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<08:17, 1.48MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<08:27, 1.44MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<08:37, 1.41MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:42, 1.82MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<04:50, 2.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<18:05, 670kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<14:00, 866kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<10:05, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<09:39, 1.25MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:05, 1.49MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<05:57, 2.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:46, 1.77MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:23, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:44, 2.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<04:10, 2.86MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:13, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:06, 1.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:18, 2.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<06:14, 1.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:59, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:33, 2.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<04:02, 2.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<17:05, 689kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<13:16, 887kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<09:33, 1.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<09:12, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<09:02, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:52, 1.70MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<04:57, 2.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:31, 1.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:14, 1.61MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:23, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:16, 1.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:59, 1.66MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:31, 2.09MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<04:00, 2.87MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<14:59, 768kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<11:46, 977kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<08:29, 1.35MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:24, 1.36MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<08:24, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:30, 1.75MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<04:41, 2.42MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<17:04, 666kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<13:11, 861kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<09:29, 1.20MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<09:05, 1.24MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:36, 1.48MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:35, 2.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:20, 1.77MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:56, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:29, 2.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:58, 2.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<07:57, 1.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:48, 1.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:01, 2.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:55, 1.87MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<06:37, 1.67MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:15, 2.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<03:48, 2.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<16:00, 688kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<12:25, 887kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<08:59, 1.22MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<08:39, 1.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<08:29, 1.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:27, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<04:38, 2.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<08:47, 1.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<07:21, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:23, 2.01MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:07, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:28, 1.97MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:08, 2.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:13, 2.06MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:50, 2.22MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<03:38, 2.94MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:53, 2.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:46, 1.85MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:33, 2.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<03:20, 3.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:31, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<05:44, 1.85MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:16, 2.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:16, 2.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:02, 1.74MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:43, 2.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<03:26, 3.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<07:23, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:20, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:40, 2.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:30, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:09, 1.69MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:53, 2.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<03:32, 2.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<14:59, 689kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<11:37, 887kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<08:22, 1.23MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<08:03, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<07:54, 1.30MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<06:06, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<04:23, 2.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<13:55, 732kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<10:52, 937kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<07:52, 1.29MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<07:41, 1.31MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<07:37, 1.32MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:47, 1.74MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:11, 2.40MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<06:47, 1.48MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:48, 1.73MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<04:17, 2.33MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:14, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:53, 1.69MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:34, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:20, 2.97MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:14, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:27, 1.81MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<04:03, 2.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:59, 1.97MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:40, 1.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:30, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:16, 2.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<14:07, 691kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<10:55, 894kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<07:52, 1.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<07:41, 1.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<07:32, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:48, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<04:09, 2.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<13:13, 728kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<10:19, 932kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<07:26, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:53<05:18, 1.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<2:27:40, 64.7kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<1:44:21, 91.5kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<1:13:08, 130kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<53:03, 179kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<39:14, 242kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<27:52, 340kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<19:33, 483kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<18:23, 512kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<13:52, 678kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<09:54, 948kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<09:01, 1.04MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<08:22, 1.12MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<06:18, 1.48MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<04:31, 2.06MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<07:11, 1.29MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<06:00, 1.54MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:24, 2.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:10, 1.78MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:27, 2.06MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:19, 2.76MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<02:28, 3.70MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<10:36, 862kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<09:26, 967kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<07:07, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<05:03, 1.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<12:11, 745kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<09:31, 952kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<06:52, 1.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:45, 1.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:44, 1.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:12, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<03:44, 2.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<12:11, 733kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<09:28, 942kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<06:51, 1.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<06:46, 1.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:43, 1.55MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<04:12, 2.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<04:50, 1.82MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:20, 1.65MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:13, 2.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<03:02, 2.87MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<11:30, 758kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<09:00, 967kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<06:30, 1.34MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<06:25, 1.35MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<06:24, 1.35MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:52, 1.77MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<03:31, 2.44MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:06, 1.40MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:13, 1.64MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:51, 2.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:31, 1.88MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:03, 1.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:00, 2.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<02:53, 2.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<11:05, 761kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<08:41, 971kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<06:16, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<06:10, 1.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:15, 1.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:52, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:31, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:00, 1.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:57, 2.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:52, 2.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<11:59, 687kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<09:15, 889kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<06:39, 1.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<06:29, 1.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<06:22, 1.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:54, 1.66MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<03:31, 2.30MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<11:08, 727kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<08:41, 931kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<06:16, 1.29MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<06:07, 1.31MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<06:05, 1.32MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:42, 1.70MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<03:22, 2.37MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<09:55, 801kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<07:47, 1.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<05:38, 1.40MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:43, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:55, 1.60MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:37, 2.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:12, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:36, 1.70MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:34, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:39, 2.93MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:58, 1.95MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:38, 2.13MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:43, 2.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:34, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:07, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:13, 2.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<02:25, 3.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:43, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:27, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:37, 2.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:28, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:06, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:17, 2.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<02:23, 3.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<08:38, 866kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<06:50, 1.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<04:58, 1.50MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<05:08, 1.44MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<05:09, 1.43MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:56, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:56, 2.51MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:49, 1.91MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:22, 2.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:35, 2.81MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:22, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:04, 1.79MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:15, 2.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:21, 3.06MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<05:47, 1.24MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:50, 1.48MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:33, 2.02MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<04:02, 1.76MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:20, 1.64MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:21, 2.12MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:28, 2.86MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:48, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:25, 2.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:35, 2.72MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:21, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:54, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:07, 2.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:15, 3.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<08:09, 849kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<06:28, 1.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:40, 1.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<04:46, 1.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<04:06, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:01, 2.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<02:11, 3.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<18:33, 366kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<14:28, 468kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<10:26, 648kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<07:25, 908kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<06:56, 967kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:36, 1.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:06, 1.63MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:19<04:18, 1.54MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:45, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:46, 2.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:21, 1.95MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:50, 1.71MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:03, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<02:12, 2.95MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<09:24, 691kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<07:10, 907kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<05:10, 1.25MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<05:02, 1.27MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:58, 1.29MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:50, 1.67MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<02:44, 2.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<07:52, 809kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<06:10, 1.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:27, 1.42MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<04:31, 1.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<04:30, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:25, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:29, 2.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:48, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:20, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:30, 2.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:05, 2.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:48, 2.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:06, 2.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:52, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:18, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:35, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<01:53, 3.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:04, 1.48MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:30, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:35, 2.31MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:05, 1.92MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:28, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:45, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<01:59, 2.95MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<08:31, 689kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<06:37, 887kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<04:46, 1.22MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<04:35, 1.27MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:52, 1.50MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:51, 2.02MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:14, 1.78MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:33, 1.61MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:48, 2.04MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<02:01, 2.81MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<06:47, 835kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<05:23, 1.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:54, 1.44MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:56, 1.42MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:00, 1.40MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:06, 1.79MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<02:13, 2.49MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<07:28, 740kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<05:51, 943kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:14, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:08, 1.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:04, 1.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:05, 1.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:13, 2.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:06, 1.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<03:28, 1.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:33, 2.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:56, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:14, 1.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:33, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<01:50, 2.86MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<06:17, 836kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<04:57, 1.06MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<03:34, 1.46MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:39, 1.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:42, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:52, 1.80MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<02:03, 2.49MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<06:54, 742kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<05:21, 953kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<03:52, 1.31MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:49, 1.32MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:48, 1.33MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:56, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:10, 2.29MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:54, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:15, 1.53MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:34, 1.93MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:51, 2.65MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<04:19, 1.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<04:14, 1.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:13, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:18, 2.11MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<04:03, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:57, 1.22MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<03:03, 1.58MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:11, 2.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:39, 1.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:19, 1.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:17, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:20, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<05:10, 909kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<04:43, 995kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:35, 1.31MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:32, 1.82MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:48, 963kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:26, 1.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:23, 1.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<02:24, 1.90MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<04:38, 985kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<04:13, 1.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<03:09, 1.44MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<02:16, 1.99MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<03:11, 1.40MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<03:20, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:36, 1.72MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<01:52, 2.38MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:57, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:40, 1.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:45, 1.60MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:58, 2.22MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<04:06, 1.06MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:46, 1.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:49, 1.54MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 604M/862M [04:24<02:02, 2.12MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:52, 1.49MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<03:03, 1.40MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:21, 1.81MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:26<01:41, 2.51MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:19, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:18, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:33, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<01:49, 2.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<04:16, 969kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:48, 1.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:51, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<02:02, 2.00MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<07:54, 516kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<06:21, 641kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<04:38, 876kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<03:15, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<13:44, 292kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<10:29, 382kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<07:33, 529kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:34<05:17, 747kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<06:47, 580kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<05:40, 693kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<04:11, 936kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<02:57, 1.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<04:52, 794kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<04:09, 929kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:05, 1.25MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:38<02:11, 1.74MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<12:27, 305kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<09:36, 396kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<06:53, 550kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<04:49, 778kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<05:05, 734kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<04:15, 876kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:08, 1.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:42<02:13, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<09:57, 368kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<07:51, 466kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<05:40, 644kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<03:58, 909kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<04:22, 821kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<03:54, 919kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:55, 1.22MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<02:04, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<04:10, 843kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<03:44, 941kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:49, 1.25MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<02:00, 1.73MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<04:02, 855kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<03:37, 952kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:42, 1.27MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<01:54, 1.78MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<03:11, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<03:04, 1.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:20, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<01:39, 2.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<03:25, 967kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<03:10, 1.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:22, 1.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:41, 1.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:26, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:25, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:52, 1.73MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<01:20, 2.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<03:52, 822kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<03:24, 931kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:33, 1.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:48, 1.73MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<03:42, 839kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<03:20, 928kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:29, 1.24MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:46, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:11, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:14, 1.36MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:43, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:14, 2.41MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<03:21, 883kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<03:04, 965kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:19, 1.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:38, 1.78MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<03:06, 934kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<02:43, 1.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:01, 1.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:26, 1.97MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<09:06, 311kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<07:03, 401kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<05:04, 556kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<03:32, 785kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<03:28, 796kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:02, 908kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:15, 1.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:35, 1.70MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:06, 1.28MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:59, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:31, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<01:04, 2.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<12:18, 213kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<09:05, 288kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<06:27, 404kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<04:28, 573kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<10:49, 236kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<08:03, 317kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<05:43, 443kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<03:57, 628kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<07:55, 314kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<06:00, 414kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<04:17, 575kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<02:58, 813kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<09:00, 268kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<06:45, 357kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<04:49, 498kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<03:20, 705kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<07:42, 305kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<05:50, 401kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<04:10, 558kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<02:53, 790kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<06:55, 330kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<05:22, 424kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<03:52, 584kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<02:41, 826kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<03:19, 665kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:48, 785kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:04, 1.06MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:27, 1.49MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:53, 1.13MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:49, 1.17MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:22, 1.54MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<00:58, 2.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:34, 1.31MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:28, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:07, 1.83MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:47, 2.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<10:12, 196kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<07:29, 267kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<05:17, 376kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<03:37, 534kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<15:15, 127kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<11:04, 175kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<07:48, 246kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<05:21, 350kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<06:00, 311kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<04:37, 403kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<03:19, 559kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<02:18, 788kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<02:11, 821kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<01:50, 973kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<01:20, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:56, 1.84MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:55, 898kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<02:11, 790kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:44, 990kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:15, 1.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:10, 1.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:12, 1.37MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:56, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:39, 2.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:46, 897kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:30, 1.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<01:06, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:01, 1.50MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:03, 1.43MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:48, 1.85MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<00:34, 2.55MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:56, 1.54MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:53, 1.62MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:40, 2.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<00:28, 2.94MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:15, 1.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:11, 1.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:54, 1.52MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:49<00:37, 2.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<02:30, 525kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<02:00, 651kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:27, 893kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<01:00, 1.25MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:16, 972kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:05, 1.14MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:47, 1.55MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<00:33, 2.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:35, 739kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:21, 861kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<01:00, 1.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:55<00:41, 1.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<03:48, 291kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<02:53, 382kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<02:03, 531kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:57<01:23, 751kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<07:34, 137kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<05:30, 188kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<03:51, 266kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<02:38, 377kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<02:09, 451kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:42, 566kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<01:13, 781kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:49, 1.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:12, 746kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:58, 925kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:41, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:36, 1.35MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:35, 1.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:26, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:25, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:26, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:20, 2.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:20, 2.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:21, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:17, 2.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:17, 2.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:19, 1.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:14, 2.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:10, 3.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:23, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:22, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:17, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:15, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:16, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:12, 2.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:13, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:10, 2.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:09, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:12, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:06, 2.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:10, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:09, 1.69MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:06, 2.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:04, 2.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 2.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 2.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:02, 2.02MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.60MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 3.56MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:01, 489kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 622kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<43:55:01,  2.53it/s]  0%|          | 897/400000 [00:00<30:40:36,  3.61it/s]  0%|          | 1688/400000 [00:00<21:26:07,  5.16it/s]  1%|          | 2562/400000 [00:00<14:58:32,  7.37it/s]  1%|          | 3437/400000 [00:00<10:27:49, 10.53it/s]  1%|          | 4289/400000 [00:00<7:18:45, 15.03it/s]   1%|▏         | 5116/400000 [00:00<5:06:43, 21.46it/s]  1%|▏         | 5865/400000 [00:01<3:34:34, 30.61it/s]  2%|▏         | 6685/400000 [00:01<2:30:07, 43.67it/s]  2%|▏         | 7526/400000 [00:01<1:45:05, 62.24it/s]  2%|▏         | 8394/400000 [00:01<1:13:37, 88.64it/s]  2%|▏         | 9234/400000 [00:01<51:39, 126.06it/s]   3%|▎         | 10056/400000 [00:01<36:19, 178.91it/s]  3%|▎         | 10936/400000 [00:01<25:35, 253.38it/s]  3%|▎         | 11853/400000 [00:01<18:05, 357.73it/s]  3%|▎         | 12735/400000 [00:01<12:50, 502.30it/s]  3%|▎         | 13673/400000 [00:01<09:10, 701.46it/s]  4%|▎         | 14562/400000 [00:02<06:38, 967.62it/s]  4%|▍         | 15461/400000 [00:02<04:51, 1321.35it/s]  4%|▍         | 16374/400000 [00:02<03:35, 1777.38it/s]  4%|▍         | 17266/400000 [00:02<02:44, 2323.42it/s]  5%|▍         | 18166/400000 [00:02<02:07, 2988.28it/s]  5%|▍         | 19045/400000 [00:02<01:42, 3702.99it/s]  5%|▍         | 19954/400000 [00:02<01:24, 4503.36it/s]  5%|▌         | 20834/400000 [00:02<01:12, 5236.49it/s]  5%|▌         | 21703/400000 [00:02<01:03, 5936.68it/s]  6%|▌         | 22570/400000 [00:03<00:58, 6466.28it/s]  6%|▌         | 23422/400000 [00:03<00:55, 6840.40it/s]  6%|▌         | 24256/400000 [00:03<00:52, 7162.73it/s]  6%|▋         | 25118/400000 [00:03<00:49, 7544.36it/s]  7%|▋         | 26041/400000 [00:03<00:46, 7981.39it/s]  7%|▋         | 26944/400000 [00:03<00:45, 8267.99it/s]  7%|▋         | 27819/400000 [00:03<00:44, 8319.82it/s]  7%|▋         | 28718/400000 [00:03<00:43, 8509.36it/s]  7%|▋         | 29636/400000 [00:03<00:42, 8699.75it/s]  8%|▊         | 30561/400000 [00:03<00:41, 8857.71it/s]  8%|▊         | 31493/400000 [00:04<00:40, 8990.39it/s]  8%|▊         | 32402/400000 [00:04<00:41, 8937.84it/s]  8%|▊         | 33334/400000 [00:04<00:40, 9048.70it/s]  9%|▊         | 34245/400000 [00:04<00:40, 8994.94it/s]  9%|▉         | 35178/400000 [00:04<00:40, 9091.13it/s]  9%|▉         | 36104/400000 [00:04<00:39, 9139.22it/s]  9%|▉         | 37020/400000 [00:04<00:40, 8968.66it/s]  9%|▉         | 37920/400000 [00:04<00:41, 8811.46it/s] 10%|▉         | 38804/400000 [00:04<00:41, 8676.21it/s] 10%|▉         | 39674/400000 [00:04<00:42, 8433.67it/s] 10%|█         | 40574/400000 [00:05<00:41, 8594.22it/s] 10%|█         | 41451/400000 [00:05<00:41, 8644.05it/s] 11%|█         | 42404/400000 [00:05<00:40, 8890.53it/s] 11%|█         | 43364/400000 [00:05<00:39, 9090.26it/s] 11%|█         | 44317/400000 [00:05<00:38, 9215.38it/s] 11%|█▏        | 45242/400000 [00:05<00:39, 9029.47it/s] 12%|█▏        | 46148/400000 [00:05<00:40, 8750.44it/s] 12%|█▏        | 47097/400000 [00:05<00:39, 8957.97it/s] 12%|█▏        | 48021/400000 [00:05<00:38, 9038.30it/s] 12%|█▏        | 48985/400000 [00:05<00:38, 9208.74it/s] 12%|█▏        | 49917/400000 [00:06<00:37, 9240.72it/s] 13%|█▎        | 50844/400000 [00:06<00:39, 8947.71it/s] 13%|█▎        | 51743/400000 [00:06<00:39, 8789.59it/s] 13%|█▎        | 52659/400000 [00:06<00:39, 8896.76it/s] 13%|█▎        | 53563/400000 [00:06<00:38, 8938.45it/s] 14%|█▎        | 54490/400000 [00:06<00:38, 9031.53it/s] 14%|█▍        | 55395/400000 [00:06<00:38, 8902.72it/s] 14%|█▍        | 56306/400000 [00:06<00:38, 8962.17it/s] 14%|█▍        | 57204/400000 [00:06<00:39, 8739.98it/s] 15%|█▍        | 58080/400000 [00:07<00:39, 8648.09it/s] 15%|█▍        | 58971/400000 [00:07<00:39, 8722.25it/s] 15%|█▍        | 59845/400000 [00:07<00:38, 8722.67it/s] 15%|█▌        | 60719/400000 [00:07<00:39, 8514.48it/s] 15%|█▌        | 61573/400000 [00:07<00:40, 8451.01it/s] 16%|█▌        | 62428/400000 [00:07<00:39, 8478.93it/s] 16%|█▌        | 63282/400000 [00:07<00:39, 8494.70it/s] 16%|█▌        | 64133/400000 [00:07<00:41, 8143.38it/s] 16%|█▋        | 65048/400000 [00:07<00:39, 8419.01it/s] 16%|█▋        | 65971/400000 [00:07<00:38, 8646.83it/s] 17%|█▋        | 66869/400000 [00:08<00:38, 8743.19it/s] 17%|█▋        | 67748/400000 [00:08<00:38, 8714.49it/s] 17%|█▋        | 68623/400000 [00:08<00:38, 8660.54it/s] 17%|█▋        | 69491/400000 [00:08<00:38, 8616.47it/s] 18%|█▊        | 70379/400000 [00:08<00:37, 8692.62it/s] 18%|█▊        | 71293/400000 [00:08<00:37, 8821.06it/s] 18%|█▊        | 72177/400000 [00:08<00:37, 8673.38it/s] 18%|█▊        | 73046/400000 [00:08<00:37, 8612.66it/s] 18%|█▊        | 73951/400000 [00:08<00:37, 8739.35it/s] 19%|█▊        | 74827/400000 [00:08<00:37, 8624.67it/s] 19%|█▉        | 75691/400000 [00:09<00:38, 8417.15it/s] 19%|█▉        | 76580/400000 [00:09<00:37, 8551.92it/s] 19%|█▉        | 77438/400000 [00:09<00:38, 8481.57it/s] 20%|█▉        | 78294/400000 [00:09<00:37, 8503.80it/s] 20%|█▉        | 79240/400000 [00:09<00:36, 8768.95it/s] 20%|██        | 80207/400000 [00:09<00:35, 9019.42it/s] 20%|██        | 81113/400000 [00:09<00:36, 8772.39it/s] 20%|██        | 81995/400000 [00:09<00:36, 8694.54it/s] 21%|██        | 82868/400000 [00:09<00:37, 8448.40it/s] 21%|██        | 83817/400000 [00:10<00:36, 8733.73it/s] 21%|██        | 84729/400000 [00:10<00:35, 8843.36it/s] 21%|██▏       | 85618/400000 [00:10<00:35, 8812.75it/s] 22%|██▏       | 86502/400000 [00:10<00:36, 8589.43it/s] 22%|██▏       | 87374/400000 [00:10<00:36, 8625.57it/s] 22%|██▏       | 88239/400000 [00:10<00:36, 8613.46it/s] 22%|██▏       | 89102/400000 [00:10<00:37, 8300.31it/s] 22%|██▏       | 89936/400000 [00:10<00:37, 8170.09it/s] 23%|██▎       | 90824/400000 [00:10<00:36, 8369.58it/s] 23%|██▎       | 91672/400000 [00:10<00:36, 8400.83it/s] 23%|██▎       | 92530/400000 [00:11<00:36, 8450.94it/s] 23%|██▎       | 93423/400000 [00:11<00:35, 8588.07it/s] 24%|██▎       | 94284/400000 [00:11<00:36, 8303.35it/s] 24%|██▍       | 95118/400000 [00:11<00:36, 8249.56it/s] 24%|██▍       | 95991/400000 [00:11<00:36, 8385.79it/s] 24%|██▍       | 96832/400000 [00:11<00:36, 8383.19it/s] 24%|██▍       | 97672/400000 [00:11<00:36, 8308.91it/s] 25%|██▍       | 98505/400000 [00:11<00:36, 8214.32it/s] 25%|██▍       | 99364/400000 [00:11<00:36, 8321.58it/s] 25%|██▌       | 100216/400000 [00:11<00:35, 8378.74it/s] 25%|██▌       | 101055/400000 [00:12<00:35, 8350.02it/s] 25%|██▌       | 101895/400000 [00:12<00:35, 8364.71it/s] 26%|██▌       | 102732/400000 [00:12<00:36, 8204.77it/s] 26%|██▌       | 103554/400000 [00:12<00:36, 8078.18it/s] 26%|██▌       | 104396/400000 [00:12<00:36, 8177.21it/s] 26%|██▋       | 105215/400000 [00:12<00:36, 8030.95it/s] 27%|██▋       | 106048/400000 [00:12<00:36, 8117.19it/s] 27%|██▋       | 106861/400000 [00:12<00:36, 8082.66it/s] 27%|██▋       | 107687/400000 [00:12<00:35, 8133.22it/s] 27%|██▋       | 108501/400000 [00:12<00:36, 8004.14it/s] 27%|██▋       | 109303/400000 [00:13<00:36, 7927.50it/s] 28%|██▊       | 110161/400000 [00:13<00:35, 8111.37it/s] 28%|██▊       | 110974/400000 [00:13<00:35, 8031.84it/s] 28%|██▊       | 111812/400000 [00:13<00:35, 8131.48it/s] 28%|██▊       | 112652/400000 [00:13<00:35, 8209.85it/s] 28%|██▊       | 113519/400000 [00:13<00:34, 8341.69it/s] 29%|██▊       | 114416/400000 [00:13<00:33, 8519.18it/s] 29%|██▉       | 115270/400000 [00:13<00:33, 8482.92it/s] 29%|██▉       | 116147/400000 [00:13<00:33, 8565.73it/s] 29%|██▉       | 117092/400000 [00:14<00:32, 8812.68it/s] 30%|██▉       | 118022/400000 [00:14<00:31, 8952.36it/s] 30%|██▉       | 118920/400000 [00:14<00:31, 8915.52it/s] 30%|██▉       | 119814/400000 [00:14<00:32, 8607.52it/s] 30%|███       | 120679/400000 [00:14<00:32, 8587.30it/s] 30%|███       | 121555/400000 [00:14<00:32, 8637.07it/s] 31%|███       | 122470/400000 [00:14<00:31, 8783.97it/s] 31%|███       | 123412/400000 [00:14<00:30, 8963.55it/s] 31%|███       | 124311/400000 [00:14<00:32, 8556.45it/s] 31%|███▏      | 125173/400000 [00:14<00:33, 8325.77it/s] 32%|███▏      | 126066/400000 [00:15<00:32, 8496.30it/s] 32%|███▏      | 126970/400000 [00:15<00:31, 8649.85it/s] 32%|███▏      | 127855/400000 [00:15<00:31, 8707.92it/s] 32%|███▏      | 128729/400000 [00:15<00:32, 8430.44it/s] 32%|███▏      | 129593/400000 [00:15<00:31, 8492.09it/s] 33%|███▎      | 130511/400000 [00:15<00:31, 8685.53it/s] 33%|███▎      | 131393/400000 [00:15<00:30, 8723.02it/s] 33%|███▎      | 132307/400000 [00:15<00:30, 8843.08it/s] 33%|███▎      | 133194/400000 [00:15<00:31, 8535.55it/s] 34%|███▎      | 134077/400000 [00:15<00:30, 8619.35it/s] 34%|███▍      | 135001/400000 [00:16<00:30, 8794.34it/s] 34%|███▍      | 135952/400000 [00:16<00:29, 8997.42it/s] 34%|███▍      | 136855/400000 [00:16<00:29, 8835.92it/s] 34%|███▍      | 137742/400000 [00:16<00:30, 8737.82it/s] 35%|███▍      | 138635/400000 [00:16<00:29, 8793.60it/s] 35%|███▍      | 139541/400000 [00:16<00:29, 8870.78it/s] 35%|███▌      | 140430/400000 [00:16<00:29, 8847.03it/s] 35%|███▌      | 141316/400000 [00:16<00:29, 8658.90it/s] 36%|███▌      | 142184/400000 [00:16<00:30, 8365.84it/s] 36%|███▌      | 143057/400000 [00:16<00:30, 8470.44it/s] 36%|███▌      | 143907/400000 [00:17<00:30, 8462.30it/s] 36%|███▌      | 144810/400000 [00:17<00:29, 8623.00it/s] 36%|███▋      | 145675/400000 [00:17<00:30, 8363.08it/s] 37%|███▋      | 146539/400000 [00:17<00:30, 8443.71it/s] 37%|███▋      | 147430/400000 [00:17<00:29, 8577.96it/s] 37%|███▋      | 148307/400000 [00:17<00:29, 8633.89it/s] 37%|███▋      | 149172/400000 [00:17<00:29, 8576.28it/s] 38%|███▊      | 150031/400000 [00:17<00:29, 8534.13it/s] 38%|███▊      | 150886/400000 [00:17<00:29, 8507.81it/s] 38%|███▊      | 151835/400000 [00:18<00:28, 8778.47it/s] 38%|███▊      | 152726/400000 [00:18<00:28, 8817.19it/s] 38%|███▊      | 153622/400000 [00:18<00:27, 8856.98it/s] 39%|███▊      | 154510/400000 [00:18<00:28, 8665.86it/s] 39%|███▉      | 155392/400000 [00:18<00:28, 8711.46it/s] 39%|███▉      | 156265/400000 [00:18<00:27, 8715.81it/s] 39%|███▉      | 157185/400000 [00:18<00:27, 8854.89it/s] 40%|███▉      | 158117/400000 [00:18<00:26, 8988.79it/s] 40%|███▉      | 159018/400000 [00:18<00:27, 8918.56it/s] 40%|███▉      | 159911/400000 [00:18<00:28, 8409.26it/s] 40%|████      | 160759/400000 [00:19<00:28, 8346.44it/s] 40%|████      | 161605/400000 [00:19<00:28, 8378.99it/s] 41%|████      | 162483/400000 [00:19<00:27, 8493.54it/s] 41%|████      | 163336/400000 [00:19<00:28, 8413.61it/s] 41%|████      | 164193/400000 [00:19<00:27, 8459.01it/s] 41%|████▏     | 165100/400000 [00:19<00:27, 8633.42it/s] 41%|████▏     | 165983/400000 [00:19<00:26, 8689.41it/s] 42%|████▏     | 166896/400000 [00:19<00:26, 8814.85it/s] 42%|████▏     | 167779/400000 [00:19<00:26, 8781.68it/s] 42%|████▏     | 168659/400000 [00:19<00:27, 8554.60it/s] 42%|████▏     | 169569/400000 [00:20<00:26, 8709.26it/s] 43%|████▎     | 170475/400000 [00:20<00:26, 8809.52it/s] 43%|████▎     | 171358/400000 [00:20<00:26, 8608.57it/s] 43%|████▎     | 172222/400000 [00:20<00:27, 8382.52it/s] 43%|████▎     | 173064/400000 [00:20<00:27, 8152.64it/s] 43%|████▎     | 173896/400000 [00:20<00:27, 8199.99it/s] 44%|████▎     | 174719/400000 [00:20<00:27, 8178.69it/s] 44%|████▍     | 175548/400000 [00:20<00:27, 8210.69it/s] 44%|████▍     | 176371/400000 [00:20<00:27, 8142.80it/s] 44%|████▍     | 177187/400000 [00:21<00:28, 7948.44it/s] 44%|████▍     | 177984/400000 [00:21<00:28, 7898.21it/s] 45%|████▍     | 178789/400000 [00:21<00:27, 7940.82it/s] 45%|████▍     | 179585/400000 [00:21<00:27, 7875.02it/s] 45%|████▌     | 180374/400000 [00:21<00:27, 7872.33it/s] 45%|████▌     | 181226/400000 [00:21<00:27, 8054.61it/s] 46%|████▌     | 182033/400000 [00:21<00:27, 7993.80it/s] 46%|████▌     | 182895/400000 [00:21<00:26, 8171.58it/s] 46%|████▌     | 183735/400000 [00:21<00:26, 8238.52it/s] 46%|████▌     | 184561/400000 [00:21<00:26, 8073.18it/s] 46%|████▋     | 185404/400000 [00:22<00:26, 8176.64it/s] 47%|████▋     | 186224/400000 [00:22<00:26, 8157.90it/s] 47%|████▋     | 187110/400000 [00:22<00:25, 8354.58it/s] 47%|████▋     | 187948/400000 [00:22<00:25, 8279.62it/s] 47%|████▋     | 188778/400000 [00:22<00:25, 8282.55it/s] 47%|████▋     | 189639/400000 [00:22<00:25, 8377.72it/s] 48%|████▊     | 190495/400000 [00:22<00:24, 8430.54it/s] 48%|████▊     | 191372/400000 [00:22<00:24, 8527.34it/s] 48%|████▊     | 192226/400000 [00:22<00:24, 8495.30it/s] 48%|████▊     | 193084/400000 [00:22<00:24, 8517.93it/s] 48%|████▊     | 193937/400000 [00:23<00:24, 8415.08it/s] 49%|████▊     | 194780/400000 [00:23<00:24, 8387.27it/s] 49%|████▉     | 195620/400000 [00:23<00:25, 7890.94it/s] 49%|████▉     | 196416/400000 [00:23<00:26, 7772.93it/s] 49%|████▉     | 197199/400000 [00:23<00:27, 7474.39it/s] 50%|████▉     | 198044/400000 [00:23<00:26, 7741.23it/s] 50%|████▉     | 198901/400000 [00:23<00:25, 7971.97it/s] 50%|████▉     | 199770/400000 [00:23<00:24, 8172.84it/s] 50%|█████     | 200596/400000 [00:23<00:24, 8197.55it/s] 50%|█████     | 201420/400000 [00:23<00:24, 8140.79it/s] 51%|█████     | 202255/400000 [00:24<00:24, 8199.55it/s] 51%|█████     | 203095/400000 [00:24<00:23, 8258.09it/s] 51%|█████     | 203923/400000 [00:24<00:23, 8221.59it/s] 51%|█████     | 204755/400000 [00:24<00:23, 8250.24it/s] 51%|█████▏    | 205581/400000 [00:24<00:23, 8144.59it/s] 52%|█████▏    | 206441/400000 [00:24<00:23, 8272.23it/s] 52%|█████▏    | 207356/400000 [00:24<00:22, 8517.27it/s] 52%|█████▏    | 208273/400000 [00:24<00:22, 8698.67it/s] 52%|█████▏    | 209146/400000 [00:24<00:23, 8245.07it/s] 52%|█████▏    | 209978/400000 [00:25<00:23, 8020.41it/s] 53%|█████▎    | 210792/400000 [00:25<00:23, 8053.63it/s] 53%|█████▎    | 211663/400000 [00:25<00:22, 8238.28it/s] 53%|█████▎    | 212491/400000 [00:25<00:22, 8203.46it/s] 53%|█████▎    | 213379/400000 [00:25<00:22, 8393.02it/s] 54%|█████▎    | 214222/400000 [00:25<00:22, 8230.35it/s] 54%|█████▍    | 215063/400000 [00:25<00:22, 8280.62it/s] 54%|█████▍    | 215902/400000 [00:25<00:22, 8310.33it/s] 54%|█████▍    | 216785/400000 [00:25<00:21, 8458.00it/s] 54%|█████▍    | 217642/400000 [00:25<00:21, 8491.15it/s] 55%|█████▍    | 218493/400000 [00:26<00:21, 8435.04it/s] 55%|█████▍    | 219409/400000 [00:26<00:20, 8639.16it/s] 55%|█████▌    | 220286/400000 [00:26<00:20, 8676.80it/s] 55%|█████▌    | 221156/400000 [00:26<00:21, 8453.69it/s] 56%|█████▌    | 222045/400000 [00:26<00:20, 8577.94it/s] 56%|█████▌    | 222905/400000 [00:26<00:21, 8346.08it/s] 56%|█████▌    | 223789/400000 [00:26<00:20, 8486.91it/s] 56%|█████▌    | 224641/400000 [00:26<00:20, 8377.82it/s] 56%|█████▋    | 225481/400000 [00:26<00:20, 8314.89it/s] 57%|█████▋    | 226315/400000 [00:26<00:20, 8272.13it/s] 57%|█████▋    | 227144/400000 [00:27<00:21, 8221.35it/s] 57%|█████▋    | 228025/400000 [00:27<00:20, 8387.39it/s] 57%|█████▋    | 228886/400000 [00:27<00:20, 8452.89it/s] 57%|█████▋    | 229733/400000 [00:27<00:20, 8374.39it/s] 58%|█████▊    | 230605/400000 [00:27<00:19, 8471.48it/s] 58%|█████▊    | 231454/400000 [00:27<00:19, 8457.47it/s] 58%|█████▊    | 232301/400000 [00:27<00:20, 8323.70it/s] 58%|█████▊    | 233151/400000 [00:27<00:19, 8374.13it/s] 58%|█████▊    | 233990/400000 [00:27<00:19, 8317.60it/s] 59%|█████▊    | 234845/400000 [00:27<00:19, 8385.98it/s] 59%|█████▉    | 235685/400000 [00:28<00:19, 8259.01it/s] 59%|█████▉    | 236512/400000 [00:28<00:19, 8193.48it/s] 59%|█████▉    | 237333/400000 [00:28<00:20, 8130.44it/s] 60%|█████▉    | 238147/400000 [00:28<00:20, 8018.21it/s] 60%|█████▉    | 239021/400000 [00:28<00:19, 8221.64it/s] 60%|█████▉    | 239867/400000 [00:28<00:19, 8287.53it/s] 60%|██████    | 240733/400000 [00:28<00:18, 8392.01it/s] 60%|██████    | 241632/400000 [00:28<00:18, 8561.88it/s] 61%|██████    | 242524/400000 [00:28<00:18, 8665.32it/s] 61%|██████    | 243393/400000 [00:28<00:18, 8528.50it/s] 61%|██████    | 244248/400000 [00:29<00:18, 8509.60it/s] 61%|██████▏   | 245111/400000 [00:29<00:18, 8544.27it/s] 61%|██████▏   | 245967/400000 [00:29<00:18, 8494.48it/s] 62%|██████▏   | 246831/400000 [00:29<00:17, 8533.96it/s] 62%|██████▏   | 247717/400000 [00:29<00:17, 8626.42it/s] 62%|██████▏   | 248581/400000 [00:29<00:17, 8488.18it/s] 62%|██████▏   | 249431/400000 [00:29<00:18, 8340.72it/s] 63%|██████▎   | 250292/400000 [00:29<00:17, 8417.68it/s] 63%|██████▎   | 251135/400000 [00:29<00:17, 8355.49it/s] 63%|██████▎   | 252019/400000 [00:30<00:17, 8493.37it/s] 63%|██████▎   | 252870/400000 [00:30<00:17, 8242.02it/s] 63%|██████▎   | 253752/400000 [00:30<00:17, 8405.61it/s] 64%|██████▎   | 254596/400000 [00:30<00:17, 8301.12it/s] 64%|██████▍   | 255475/400000 [00:30<00:17, 8440.48it/s] 64%|██████▍   | 256389/400000 [00:30<00:16, 8638.82it/s] 64%|██████▍   | 257256/400000 [00:30<00:16, 8522.80it/s] 65%|██████▍   | 258120/400000 [00:30<00:16, 8555.26it/s] 65%|██████▍   | 258978/400000 [00:30<00:16, 8494.71it/s] 65%|██████▍   | 259829/400000 [00:30<00:16, 8406.64it/s] 65%|██████▌   | 260714/400000 [00:31<00:16, 8533.27it/s] 65%|██████▌   | 261569/400000 [00:31<00:16, 8417.92it/s] 66%|██████▌   | 262443/400000 [00:31<00:16, 8510.75it/s] 66%|██████▌   | 263296/400000 [00:31<00:16, 8369.58it/s] 66%|██████▌   | 264135/400000 [00:31<00:16, 8367.41it/s] 66%|██████▌   | 264973/400000 [00:31<00:16, 7985.31it/s] 66%|██████▋   | 265776/400000 [00:31<00:16, 7976.02it/s] 67%|██████▋   | 266652/400000 [00:31<00:16, 8195.81it/s] 67%|██████▋   | 267548/400000 [00:31<00:15, 8408.98it/s] 67%|██████▋   | 268410/400000 [00:31<00:15, 8470.55it/s] 67%|██████▋   | 269311/400000 [00:32<00:15, 8624.31it/s] 68%|██████▊   | 270177/400000 [00:32<00:15, 8553.41it/s] 68%|██████▊   | 271035/400000 [00:32<00:15, 8546.72it/s] 68%|██████▊   | 271892/400000 [00:32<00:15, 8172.52it/s] 68%|██████▊   | 272780/400000 [00:32<00:15, 8372.09it/s] 68%|██████▊   | 273731/400000 [00:32<00:14, 8680.57it/s] 69%|██████▊   | 274605/400000 [00:32<00:14, 8605.02it/s] 69%|██████▉   | 275496/400000 [00:32<00:14, 8692.09it/s] 69%|██████▉   | 276411/400000 [00:32<00:14, 8823.52it/s] 69%|██████▉   | 277297/400000 [00:32<00:14, 8743.40it/s] 70%|██████▉   | 278180/400000 [00:33<00:13, 8766.76it/s] 70%|██████▉   | 279059/400000 [00:33<00:14, 8632.41it/s] 70%|██████▉   | 279924/400000 [00:33<00:14, 8448.86it/s] 70%|███████   | 280807/400000 [00:33<00:13, 8557.97it/s] 70%|███████   | 281767/400000 [00:33<00:13, 8844.51it/s] 71%|███████   | 282659/400000 [00:33<00:13, 8866.03it/s] 71%|███████   | 283549/400000 [00:33<00:13, 8684.71it/s] 71%|███████   | 284458/400000 [00:33<00:13, 8800.81it/s] 71%|███████▏  | 285341/400000 [00:33<00:13, 8741.36it/s] 72%|███████▏  | 286217/400000 [00:34<00:13, 8592.80it/s] 72%|███████▏  | 287095/400000 [00:34<00:13, 8647.37it/s] 72%|███████▏  | 287964/400000 [00:34<00:12, 8658.94it/s] 72%|███████▏  | 288831/400000 [00:34<00:13, 8487.60it/s] 72%|███████▏  | 289777/400000 [00:34<00:12, 8755.62it/s] 73%|███████▎  | 290687/400000 [00:34<00:12, 8854.29it/s] 73%|███████▎  | 291575/400000 [00:34<00:12, 8831.15it/s] 73%|███████▎  | 292507/400000 [00:34<00:11, 8972.12it/s] 73%|███████▎  | 293406/400000 [00:34<00:11, 8970.79it/s] 74%|███████▎  | 294305/400000 [00:34<00:12, 8655.00it/s] 74%|███████▍  | 295174/400000 [00:35<00:12, 8449.26it/s] 74%|███████▍  | 296036/400000 [00:35<00:12, 8497.81it/s] 74%|███████▍  | 296915/400000 [00:35<00:12, 8581.89it/s] 74%|███████▍  | 297776/400000 [00:35<00:12, 8453.48it/s] 75%|███████▍  | 298701/400000 [00:35<00:11, 8676.22it/s] 75%|███████▍  | 299656/400000 [00:35<00:11, 8921.02it/s] 75%|███████▌  | 300552/400000 [00:35<00:11, 8700.53it/s] 75%|███████▌  | 301429/400000 [00:35<00:11, 8719.75it/s] 76%|███████▌  | 302304/400000 [00:35<00:11, 8695.30it/s] 76%|███████▌  | 303240/400000 [00:35<00:10, 8883.81it/s] 76%|███████▌  | 304171/400000 [00:36<00:10, 9005.06it/s] 76%|███████▋  | 305083/400000 [00:36<00:10, 9039.07it/s] 76%|███████▋  | 305989/400000 [00:36<00:10, 8712.71it/s] 77%|███████▋  | 306878/400000 [00:36<00:10, 8764.01it/s] 77%|███████▋  | 307814/400000 [00:36<00:10, 8934.00it/s] 77%|███████▋  | 308710/400000 [00:36<00:10, 8718.56it/s] 77%|███████▋  | 309585/400000 [00:36<00:10, 8438.53it/s] 78%|███████▊  | 310464/400000 [00:36<00:10, 8538.36it/s] 78%|███████▊  | 311322/400000 [00:36<00:10, 8547.86it/s] 78%|███████▊  | 312219/400000 [00:36<00:10, 8668.76it/s] 78%|███████▊  | 313093/400000 [00:37<00:10, 8686.59it/s] 78%|███████▊  | 313964/400000 [00:37<00:10, 8381.87it/s] 79%|███████▊  | 314806/400000 [00:37<00:10, 8188.07it/s] 79%|███████▉  | 315629/400000 [00:37<00:10, 7920.38it/s] 79%|███████▉  | 316426/400000 [00:37<00:10, 7800.43it/s] 79%|███████▉  | 317210/400000 [00:37<00:10, 7809.55it/s] 80%|███████▉  | 318041/400000 [00:37<00:10, 7951.61it/s] 80%|███████▉  | 318910/400000 [00:37<00:09, 8157.15it/s] 80%|███████▉  | 319832/400000 [00:37<00:09, 8447.97it/s] 80%|████████  | 320784/400000 [00:38<00:09, 8740.97it/s] 80%|████████  | 321718/400000 [00:38<00:08, 8911.80it/s] 81%|████████  | 322615/400000 [00:38<00:08, 8617.13it/s] 81%|████████  | 323483/400000 [00:38<00:08, 8509.36it/s] 81%|████████  | 324393/400000 [00:38<00:08, 8677.70it/s] 81%|████████▏ | 325282/400000 [00:38<00:08, 8737.81it/s] 82%|████████▏ | 326212/400000 [00:38<00:08, 8896.85it/s] 82%|████████▏ | 327105/400000 [00:38<00:08, 8876.12it/s] 82%|████████▏ | 328016/400000 [00:38<00:08, 8942.80it/s] 82%|████████▏ | 328912/400000 [00:38<00:08, 8783.96it/s] 82%|████████▏ | 329793/400000 [00:39<00:08, 8735.59it/s] 83%|████████▎ | 330755/400000 [00:39<00:07, 8982.52it/s] 83%|████████▎ | 331656/400000 [00:39<00:07, 8631.79it/s] 83%|████████▎ | 332524/400000 [00:39<00:07, 8585.19it/s] 83%|████████▎ | 333465/400000 [00:39<00:07, 8814.79it/s] 84%|████████▎ | 334369/400000 [00:39<00:07, 8880.63it/s] 84%|████████▍ | 335260/400000 [00:39<00:07, 8783.59it/s] 84%|████████▍ | 336141/400000 [00:39<00:07, 8532.83it/s] 84%|████████▍ | 337005/400000 [00:39<00:07, 8564.57it/s] 84%|████████▍ | 337893/400000 [00:39<00:07, 8655.40it/s] 85%|████████▍ | 338806/400000 [00:40<00:06, 8790.15it/s] 85%|████████▍ | 339732/400000 [00:40<00:06, 8919.21it/s] 85%|████████▌ | 340626/400000 [00:40<00:06, 8491.59it/s] 85%|████████▌ | 341521/400000 [00:40<00:06, 8622.09it/s] 86%|████████▌ | 342456/400000 [00:40<00:06, 8826.60it/s] 86%|████████▌ | 343418/400000 [00:40<00:06, 9049.33it/s] 86%|████████▌ | 344331/400000 [00:40<00:06, 9071.79it/s] 86%|████████▋ | 345242/400000 [00:40<00:06, 9081.94it/s] 87%|████████▋ | 346153/400000 [00:40<00:06, 8933.86it/s] 87%|████████▋ | 347070/400000 [00:41<00:05, 9003.28it/s] 87%|████████▋ | 347972/400000 [00:41<00:05, 8943.20it/s] 87%|████████▋ | 348868/400000 [00:41<00:05, 8882.83it/s] 87%|████████▋ | 349758/400000 [00:41<00:05, 8624.18it/s] 88%|████████▊ | 350673/400000 [00:41<00:05, 8772.92it/s] 88%|████████▊ | 351588/400000 [00:41<00:05, 8880.13it/s] 88%|████████▊ | 352478/400000 [00:41<00:05, 8661.49it/s] 88%|████████▊ | 353347/400000 [00:41<00:05, 8516.96it/s] 89%|████████▊ | 354215/400000 [00:41<00:05, 8564.05it/s] 89%|████████▉ | 355122/400000 [00:41<00:05, 8707.48it/s] 89%|████████▉ | 356005/400000 [00:42<00:05, 8739.63it/s] 89%|████████▉ | 356881/400000 [00:42<00:05, 8609.51it/s] 89%|████████▉ | 357744/400000 [00:42<00:04, 8473.12it/s] 90%|████████▉ | 358593/400000 [00:42<00:04, 8415.38it/s] 90%|████████▉ | 359540/400000 [00:42<00:04, 8705.08it/s] 90%|█████████ | 360459/400000 [00:42<00:04, 8843.98it/s] 90%|█████████ | 361389/400000 [00:42<00:04, 8974.10it/s] 91%|█████████ | 362289/400000 [00:42<00:04, 8834.67it/s] 91%|█████████ | 363175/400000 [00:42<00:04, 8714.22it/s] 91%|█████████ | 364049/400000 [00:42<00:04, 8647.55it/s] 91%|█████████ | 364916/400000 [00:43<00:04, 8642.90it/s] 91%|█████████▏| 365800/400000 [00:43<00:03, 8697.37it/s] 92%|█████████▏| 366671/400000 [00:43<00:03, 8420.06it/s] 92%|█████████▏| 367519/400000 [00:43<00:03, 8436.61it/s] 92%|█████████▏| 368455/400000 [00:43<00:03, 8690.23it/s] 92%|█████████▏| 369355/400000 [00:43<00:03, 8777.89it/s] 93%|█████████▎| 370236/400000 [00:43<00:03, 8605.52it/s] 93%|█████████▎| 371099/400000 [00:43<00:03, 8398.73it/s] 93%|█████████▎| 371991/400000 [00:43<00:03, 8547.81it/s] 93%|█████████▎| 372852/400000 [00:44<00:03, 8564.72it/s] 93%|█████████▎| 373735/400000 [00:44<00:03, 8641.43it/s] 94%|█████████▎| 374666/400000 [00:44<00:02, 8831.37it/s] 94%|█████████▍| 375552/400000 [00:44<00:02, 8568.71it/s] 94%|█████████▍| 376413/400000 [00:44<00:02, 8562.99it/s] 94%|█████████▍| 377272/400000 [00:44<00:02, 8494.76it/s] 95%|█████████▍| 378124/400000 [00:44<00:02, 8395.85it/s] 95%|█████████▍| 378971/400000 [00:44<00:02, 8416.82it/s] 95%|█████████▍| 379814/400000 [00:44<00:02, 8292.03it/s] 95%|█████████▌| 380670/400000 [00:44<00:02, 8369.99it/s] 95%|█████████▌| 381508/400000 [00:45<00:02, 8266.39it/s] 96%|█████████▌| 382336/400000 [00:45<00:02, 8081.35it/s] 96%|█████████▌| 383146/400000 [00:45<00:02, 8037.58it/s] 96%|█████████▌| 383951/400000 [00:45<00:02, 7922.97it/s] 96%|█████████▌| 384843/400000 [00:45<00:01, 8195.95it/s] 96%|█████████▋| 385748/400000 [00:45<00:01, 8433.26it/s] 97%|█████████▋| 386596/400000 [00:45<00:01, 8414.87it/s] 97%|█████████▋| 387458/400000 [00:45<00:01, 8474.97it/s] 97%|█████████▋| 388316/400000 [00:45<00:01, 8503.36it/s] 97%|█████████▋| 389233/400000 [00:45<00:01, 8691.15it/s] 98%|█████████▊| 390105/400000 [00:46<00:01, 8692.24it/s] 98%|█████████▊| 390987/400000 [00:46<00:01, 8729.71it/s] 98%|█████████▊| 391862/400000 [00:46<00:00, 8718.55it/s] 98%|█████████▊| 392735/400000 [00:46<00:00, 8414.05it/s] 98%|█████████▊| 393620/400000 [00:46<00:00, 8539.73it/s] 99%|█████████▊| 394499/400000 [00:46<00:00, 8612.02it/s] 99%|█████████▉| 395375/400000 [00:46<00:00, 8654.14it/s] 99%|█████████▉| 396242/400000 [00:46<00:00, 8610.73it/s] 99%|█████████▉| 397105/400000 [00:46<00:00, 8405.20it/s] 99%|█████████▉| 397967/400000 [00:46<00:00, 8465.92it/s]100%|█████████▉| 398815/400000 [00:47<00:00, 8462.28it/s]100%|█████████▉| 399723/400000 [00:47<00:00, 8637.17it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8474.55it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

 #####  get_Data DataLoader 
((<torchtext.data.dataset.TabularDataset object at 0x7f614f776080>, <torchtext.data.dataset.TabularDataset object at 0x7f614f7760f0>, <torchtext.vocab.Vocab object at 0x7f614f776208>), {})



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

  <mlmodels.model_keras.charcnn.Model object at 0x7f6077d23080> 

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

2020-05-26 13:30:33.780676: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 66453504 exceeds 10% of system memory.
2020-05-26 13:30:33.821022: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 132644864 exceeds 10% of system memory.
2020-05-26 13:30:33.829254: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 132120576 exceeds 10% of system memory.
2020-05-26 13:30:34.502154: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 33161216 exceeds 10% of system memory.
2020-05-26 13:30:34.588158: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 132644864 exceeds 10% of system memory.

 Object Creation

 Object Compute

 Object get_data
Train on 354 samples, validate on 354 samples
Epoch 1/1

128/354 [=========>....................] - ETA: 16s - loss: 1.6815
256/354 [====================>.........] - ETA: 6s - loss: 1.2638 
354/354 [==============================] - 27s 77ms/step - loss: 1.1883 - val_loss: 0.6620

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
[[3.0134495e-05 3.4378132e-05 1.5815488e-01 8.4178060e-01]
 [2.6729755e-05 2.7237929e-05 1.3207301e-01 8.6787301e-01]
 [2.9390045e-05 3.2133954e-05 1.6103463e-01 8.3890384e-01]
 ...
 [5.3935870e-05 5.6254397e-05 1.5494391e-01 8.4494591e-01]
 [4.5704939e-05 5.1216255e-05 1.6853671e-01 8.3136636e-01]
 [4.2138279e-05 4.2360287e-05 1.5095575e-01 8.4895974e-01]]

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

  <mlmodels.model_keras.charcnn_zhang.Model object at 0x7f6077d2cfd0> 

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

128/354 [=========>....................] - ETA: 8s - loss: 1.3847
256/354 [====================>.........] - ETA: 3s - loss: 1.1880
354/354 [==============================] - 16s 46ms/step - loss: 1.5808 - val_loss: 0.6247

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
[[0.01587212 0.02135341 0.22289537 0.739879  ]
 [0.01585889 0.02134488 0.22291607 0.7398802 ]
 [0.01584144 0.02132432 0.22283413 0.7400001 ]
 ...
 [0.01606157 0.02158009 0.22354744 0.7388109 ]
 [0.0160911  0.02161224 0.22363898 0.73865765]
 [0.01608789 0.02160836 0.22362082 0.7386829 ]]

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f60c4148908> 

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

 1000/25000 [>.............................] - ETA: 13s - loss: 8.0653 - accuracy: 0.4740
 2000/25000 [=>............................] - ETA: 9s - loss: 8.0270 - accuracy: 0.4765 
 3000/25000 [==>...........................] - ETA: 8s - loss: 8.0040 - accuracy: 0.4780
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.9388 - accuracy: 0.4823
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8844 - accuracy: 0.4858
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7816 - accuracy: 0.4925
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7827 - accuracy: 0.4924
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7586 - accuracy: 0.4940
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7518 - accuracy: 0.4944
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7080 - accuracy: 0.4973
11000/25000 [============>.................] - ETA: 4s - loss: 7.7294 - accuracy: 0.4959
12000/25000 [=============>................] - ETA: 4s - loss: 7.7228 - accuracy: 0.4963
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7350 - accuracy: 0.4955
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7039 - accuracy: 0.4976
15000/25000 [=================>............] - ETA: 3s - loss: 7.6942 - accuracy: 0.4982
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7107 - accuracy: 0.4971
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7207 - accuracy: 0.4965
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7067 - accuracy: 0.4974
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7102 - accuracy: 0.4972
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7073 - accuracy: 0.4974
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7148 - accuracy: 0.4969
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6868 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6793 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 9s 376us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f60ccd8a8c8>

 ######### postional parameters :  ['out']

 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f60ccd8a8c8>

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

###### load_callable_from_uri LOADED <function train_test_split at 0x7f6135ee9bf8>

 ######### postional parameters :  []

 ######### Execute : preprocessor_func <function train_test_split at 0x7f6135ee9bf8>

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

###### load_callable_from_uri LOADED <function pickle_dump at 0x7f60e28b1598>

 ######### postional parameters :  ['t']

 ######### Execute : preprocessor_func <function pickle_dump at 0x7f60e28b1598>
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

  <mlmodels.model_tch.torchhub.Model object at 0x7f60774d40b8> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f60e3766268>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f60e3766268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f60e3766268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0036409833629926047 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.02054845881462097 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.0030930112799008687 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.023551243305206297 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 3 	 Loss: 0.0023913743247588473 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.020139781475067137 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.0015067803288499515 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.015423745185136796 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 5 	 Loss: 0.0014895528008540472 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.009071830421686173 	 Accuracy: 7
model saves at 7 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f60e3766268>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f60e3766268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f60e3766268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([9, 4, 4, 8, 7, 6, 6, 7, 2, 4]), array([9, 2, 4, 8, 7, 7, 0, 7, 2, 4]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f60c4148908> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f60e3766268>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f60e3766268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f60e3766268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0023039699792861937 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.014134277820587158 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.001683302124341329 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.05765317249298096 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f60e3766268>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f60e3766268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f60e3766268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8]), array([4, 9, 8, 8, 1, 9, 2, 3, 1, 8]))

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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 368, in test_run_model
    fittable = False if x in not_fittable_models else True)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 260, in test_module
    model = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.py", line 66, in __init__
    data_set, internal_states = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.py", line 183, in get_dataset
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 297, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 92, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth" to /home/runner/.cache/torch/checkpoints/resnet18-5c106cde.pth
  0%|          | 0.00/44.7M [00:00<?, ?B/s] 22%|██▏       | 10.0M/44.7M [00:00<00:00, 105MB/s] 49%|████▉     | 22.0M/44.7M [00:00<00:00, 111MB/s] 76%|███████▌  | 34.0M/44.7M [00:00<00:00, 115MB/s]100%|██████████| 44.7M/44.7M [00:00<00:00, 119MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7f6077352710> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f612f232048>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f612f232048>

  function with postional parmater data_info <function tf_dataset_download at 0x7f612f232048> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f60e3766268>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f60e3766268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f60e3766268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.47183469057083127 	 Accuracy: 60
Train Epoch: 1 	 Loss: 5.353988313674927 	 Accuracy: 260
model saves at 260 accuracy
Train Epoch: 2 	 Loss: 0.4059054517745972 	 Accuracy: 64
Train Epoch: 2 	 Loss: 7.7485198974609375 	 Accuracy: 200

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f612f232048>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f612f232048>

  function with postional parmater data_info <function tf_dataset_download at 0x7f612f232048> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f60e3766268>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f60e3766268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f60e3766268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([5, 8, 7, 7, 8, 2, 7, 5, 5, 5]), array([8, 8, 1, 2, 0, 7, 9, 5, 5, 3]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f6077d2ccc0> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f60772f76d8> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f60e3766268>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f60e3766268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f60e3766268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 

Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 49152/26421880 [00:00<01:35, 274830.27it/s]  1%|          | 188416/26421880 [00:00<01:13, 358390.87it/s]  2%|▏         | 466944/26421880 [00:00<00:54, 472762.05it/s]  5%|▍         | 1318912/26421880 [00:00<00:38, 656783.06it/s] 15%|█▍        | 3833856/26421880 [00:00<00:24, 922666.06it/s] 29%|██▊       | 7553024/26421880 [00:01<00:14, 1304213.35it/s] 43%|████▎     | 11272192/26421880 [00:01<00:08, 1833013.39it/s] 60%|██████    | 15982592/26421880 [00:01<00:04, 2554647.16it/s] 76%|███████▌  | 19955712/26421880 [00:01<00:01, 3551622.83it/s] 89%|████████▉ | 23601152/26421880 [00:01<00:00, 4852363.40it/s]26427392it [00:01, 16693451.20it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 102431.54it/s]           
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 40960/4422102 [00:00<00:12, 363772.96it/s]  2%|▏         | 106496/4422102 [00:00<00:11, 377187.55it/s]  9%|▉         | 401408/4422102 [00:00<00:07, 507382.47it/s] 21%|██▏       | 950272/4422102 [00:00<00:05, 681964.59it/s] 74%|███████▍  | 3276800/4422102 [00:00<00:01, 960550.96it/s]4423680it [00:00, 4479216.55it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 39360.89it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0033729889790217083 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.019670237183570864 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.0028426912824312846 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.014872707486152648 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 3 	 Loss: 0.002883961121241252 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.014913292229175567 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 4 	 Loss: 0.0027770491441090903 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.010567351222038269 	 Accuracy: 6
model saves at 6 accuracy
Train Epoch: 5 	 Loss: 0.0026041402717431384 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.011738826930522918 	 Accuracy: 5

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f60e3766268>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f60e3766268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f60e3766268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([3, 2, 1, 9, 4, 4, 2, 3, 5, 5]), array([3, 0, 1, 9, 4, 6, 3, 0, 5, 5]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f60778a79b0> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f60e3766268>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f60e3766268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f60e3766268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:44, 110270.86it/s]  0%|          | 40960/18165135 [00:00<02:28, 122396.65it/s]  1%|          | 98304/18165135 [00:00<01:57, 153899.91it/s]  1%|          | 204800/18165135 [00:01<01:29, 201252.81it/s]  2%|▏         | 425984/18165135 [00:01<01:05, 271737.14it/s]  5%|▍         | 860160/18165135 [00:01<00:46, 373278.20it/s]  9%|▉         | 1720320/18165135 [00:01<00:31, 518881.55it/s] 19%|█▉        | 3448832/18165135 [00:01<00:20, 727319.31it/s] 30%|███       | 5480448/18165135 [00:01<00:12, 1002248.84it/s] 34%|███▎      | 6111232/18165135 [00:02<00:13, 919709.67it/s]  36%|███▌      | 6578176/18165135 [00:02<00:09, 1165461.52it/s] 47%|████▋     | 8626176/18165135 [00:03<00:05, 1607259.74it/s] 51%|█████▏    | 9338880/18165135 [00:03<00:04, 1967464.03it/s] 55%|█████▍    | 9961472/18165135 [00:03<00:04, 1740755.02it/s] 68%|██████▊   | 12328960/18165135 [00:03<00:02, 2375888.78it/s] 72%|███████▏  | 13148160/18165135 [00:04<00:02, 2450293.80it/s] 76%|███████▌  | 13803520/18165135 [00:04<00:01, 2834704.11it/s] 80%|████████  | 14532608/18165135 [00:04<00:01, 3280315.25it/s] 83%|████████▎ | 15147008/18165135 [00:04<00:00, 3493431.60it/s] 87%|████████▋ | 15802368/18165135 [00:04<00:00, 3723048.46it/s] 91%|█████████ | 16498688/18165135 [00:04<00:00, 3963836.06it/s] 95%|█████████▍| 17219584/18165135 [00:04<00:00, 4185634.46it/s] 99%|█████████▉| 17981440/18165135 [00:05<00:00, 4425556.91it/s]18169856it [00:05, 3532301.84it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 110162.85it/s]32768it [00:00, 53195.02it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 110252.64it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 122417.95it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153868.21it/s]  7%|▋         | 204800/3041136 [00:01<00:14, 201242.66it/s] 14%|█▎        | 417792/3041136 [00:01<00:09, 271126.56it/s] 28%|██▊       | 851968/3041136 [00:01<00:05, 372395.58it/s] 48%|████▊     | 1474560/3041136 [00:01<00:03, 512415.67it/s] 70%|███████   | 2129920/3041136 [00:01<00:01, 696522.50it/s] 93%|█████████▎| 2818048/3041136 [00:01<00:00, 934513.25it/s]3047424it [00:01, 1621307.32it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 27127.17it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.004313418785730998 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.020691946148872375 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.003703306833902995 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.022111106157302857 	 Accuracy: 2
Train Epoch: 3 	 Loss: 0.002911732623974482 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.0203166743516922 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 4 	 Loss: 0.0025687621235847475 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.021557882606983186 	 Accuracy: 4
Train Epoch: 5 	 Loss: 0.0026443531066179275 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.015009343028068542 	 Accuracy: 4

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f60e3766268>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f60e3766268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f60e3766268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([7, 1, 1, 1, 0, 2, 3, 1, 6, 0]), array([9, 7, 1, 4, 0, 2, 3, 1, 2, 0]))

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

  <mlmodels.model_tch.textcnn.Model object at 0x7f60773fd518> 

  #### Fit   ######################################################## 

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

###### load_callable_from_uri LOADED <function split_train_valid at 0x7f60c1eefd90>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function split_train_valid at 0x7f60c1eefd90>

  function with postional parmater data_info <function split_train_valid at 0x7f60c1eefd90> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f60c1eefea0>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f60c1eefea0>

  function with postional parmater data_info <function create_tabular_dataset at 0x7f60c1eefea0> , (data_info, **args) 

  Download en 
Requirement already satisfied: en_core_web_sm==2.2.5 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (2.2.5)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
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
Train Epoch: 1 	 Loss: 0.011036490139208342 	 Accuracy: 57
Train Epoch: 1 	 Loss: 0.07579347491264343 	 Accuracy: 30

  model saves at 30% accuracy 

  #### Predict   #################################################### 
{'data_info': {'data_path': 'dataset/recommender/', 'dataset': 'IMDB_sample.txt', 'data_type': 'csv_dataset', 'batch_size': 64, 'train': True}, 'preprocessors': [{'uri': 'mlmodels.model_tch.textcnn:split_train_valid', 'args': {'frac': 0.99}}, {'uri': 'mlmodels.model_tch.textcnn:create_tabular_dataset', 'args': {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'}}], 'frac': 1}

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

###### load_callable_from_uri LOADED <function split_train_valid at 0x7f60c1eefd90>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function split_train_valid at 0x7f60c1eefd90>

  function with postional parmater data_info <function split_train_valid at 0x7f60c1eefd90> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f60c1eefea0>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f60c1eefea0>

  function with postional parmater data_info <function create_tabular_dataset at 0x7f60c1eefea0> , (data_info, **args) 

  Download en 
Requirement already satisfied: en_core_web_sm==2.2.5 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (2.2.5)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
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
(tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1]), tensor([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1,
        2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
        1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2]))

  #### Get  metrics   ################################################ 

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

###### load_callable_from_uri LOADED <function split_train_valid at 0x7f60c1eefd90>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function split_train_valid at 0x7f60c1eefd90>

  function with postional parmater data_info <function split_train_valid at 0x7f60c1eefd90> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f60c1eefea0>

 ######### postional parameters :  ['data_info']

 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f60c1eefea0>

  function with postional parmater data_info <function create_tabular_dataset at 0x7f60c1eefea0> , (data_info, **args) 

  Download en 
Requirement already satisfied: en_core_web_sm==2.2.5 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (2.2.5)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
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


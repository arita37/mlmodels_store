
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'bd7dfc233939710e05244f8e6a394a7ce12a3485', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

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
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'

  




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
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.json [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'

  




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
Dataset File path :  train/mlmodels/dataset/text/imdb.npz
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.json [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'

  




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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f66b3845e18> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f66b3845e18> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f671edfc048> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f671edfc048> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f6738b2ae18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f6738b2ae18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f66cc67c8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f66cc67c8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f66cc67c8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 502, in __init__
    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 502, in __init__
    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 607, in __init__
    data            = np.load( file_path,**args.get("numpy_loader_args", {}))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 428, in load
    fid = open(os_fspath(file), "rb")
FileNotFoundError: [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 301, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 93, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:24, 117715.08it/s] 78%|███████▊  | 7692288/9912422 [00:00<00:13, 168053.36it/s]9920512it [00:00, 37953021.37it/s]                           
0it [00:00, ?it/s]32768it [00:00, 537603.82it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 464615.88it/s]1654784it [00:00, 11532067.61it/s]                         
0it [00:00, ?it/s]8192it [00:00, 149967.87it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f66c9be9550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f66c9cb97b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f66cc67c510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f66cc67c510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f66cc67c510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:54,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:54,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:53,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:53,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:53,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:37,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:36,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:36,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:36,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:36,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:35,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:35,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:24,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:24,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 11.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 11.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 11.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:11, 11.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:10, 11.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:10, 11.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:10, 11.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:10, 11.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:10, 11.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:10, 11.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:10, 11.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 15.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:00<00:05, 19.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:05, 19.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:05, 19.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:00<00:05, 19.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 19.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 19.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 19.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 19.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 26.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 26.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 26.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 26.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 26.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 26.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 26.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 26.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 33.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 41.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 49.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 49.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 57.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 57.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 57.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 57.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 57.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 57.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 57.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 57.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 57.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 65.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 65.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 65.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 65.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 65.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 65.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 65.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 65.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 65.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 65.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 65.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 72.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 72.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 72.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 72.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 72.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 72.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 72.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 72.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 72.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 72.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 72.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 76.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 81.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 81.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 81.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 81.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 81.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 81.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 81.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 81.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 81.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 81.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 81.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 84.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 84.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 84.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 84.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 84.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 84.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.10s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.65 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.97s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  2.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 84.65 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.97s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.97s/ file]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 40.81 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.97s/ url]
0 examples [00:00, ? examples/s]2020-06-19 18:08:23.839882: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-19 18:08:23.851180: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-19 18:08:23.851941: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561791ceed20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-19 18:08:23.851958: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
28 examples [00:00, 278.61 examples/s]143 examples [00:00, 360.32 examples/s]259 examples [00:00, 453.82 examples/s]375 examples [00:00, 555.17 examples/s]486 examples [00:00, 652.90 examples/s]601 examples [00:00, 749.65 examples/s]715 examples [00:00, 834.09 examples/s]831 examples [00:00, 910.51 examples/s]948 examples [00:00, 974.48 examples/s]1067 examples [00:01, 1028.50 examples/s]1183 examples [00:01, 1064.60 examples/s]1298 examples [00:01, 1087.24 examples/s]1417 examples [00:01, 1113.73 examples/s]1532 examples [00:01, 1102.68 examples/s]1647 examples [00:01, 1113.94 examples/s]1767 examples [00:01, 1138.29 examples/s]1883 examples [00:01, 1133.55 examples/s]1998 examples [00:01, 1092.17 examples/s]2114 examples [00:01, 1110.44 examples/s]2229 examples [00:02, 1119.78 examples/s]2354 examples [00:02, 1154.17 examples/s]2471 examples [00:02, 1132.77 examples/s]2585 examples [00:02, 1099.96 examples/s]2711 examples [00:02, 1141.77 examples/s]2832 examples [00:02, 1160.23 examples/s]2955 examples [00:02, 1178.20 examples/s]3074 examples [00:02, 1180.63 examples/s]3194 examples [00:02, 1183.22 examples/s]3313 examples [00:02, 1178.89 examples/s]3432 examples [00:03, 1146.48 examples/s]3552 examples [00:03, 1161.65 examples/s]3669 examples [00:03, 1112.73 examples/s]3786 examples [00:03, 1129.24 examples/s]3902 examples [00:03, 1138.16 examples/s]4018 examples [00:03, 1142.63 examples/s]4133 examples [00:03, 1094.19 examples/s]4244 examples [00:03, 1085.93 examples/s]4356 examples [00:03, 1094.67 examples/s]4469 examples [00:04, 1101.61 examples/s]4580 examples [00:04, 1095.25 examples/s]4694 examples [00:04, 1107.28 examples/s]4809 examples [00:04, 1118.98 examples/s]4924 examples [00:04, 1127.49 examples/s]5037 examples [00:04, 1098.33 examples/s]5154 examples [00:04, 1117.22 examples/s]5276 examples [00:04, 1143.79 examples/s]5402 examples [00:04, 1174.24 examples/s]5523 examples [00:04, 1183.16 examples/s]5642 examples [00:05, 1165.46 examples/s]5774 examples [00:05, 1206.68 examples/s]5897 examples [00:05, 1211.44 examples/s]6019 examples [00:05, 1190.17 examples/s]6140 examples [00:05, 1195.84 examples/s]6260 examples [00:05, 1175.24 examples/s]6378 examples [00:05, 1159.20 examples/s]6495 examples [00:05, 1159.17 examples/s]6613 examples [00:05, 1164.01 examples/s]6741 examples [00:05, 1194.35 examples/s]6863 examples [00:06, 1201.72 examples/s]6992 examples [00:06, 1224.79 examples/s]7118 examples [00:06, 1234.67 examples/s]7242 examples [00:06, 1223.13 examples/s]7371 examples [00:06, 1242.38 examples/s]7506 examples [00:06, 1270.45 examples/s]7634 examples [00:06, 1265.41 examples/s]7768 examples [00:06, 1285.64 examples/s]7897 examples [00:06, 1283.89 examples/s]8026 examples [00:06, 1271.70 examples/s]8154 examples [00:07, 1251.58 examples/s]8280 examples [00:07, 1247.10 examples/s]8405 examples [00:07, 1210.81 examples/s]8527 examples [00:07, 1119.98 examples/s]8641 examples [00:07, 1064.16 examples/s]8753 examples [00:07, 1080.04 examples/s]8870 examples [00:07, 1102.82 examples/s]8991 examples [00:07, 1131.73 examples/s]9106 examples [00:07, 1134.84 examples/s]9221 examples [00:08, 1129.89 examples/s]9341 examples [00:08, 1147.55 examples/s]9458 examples [00:08, 1154.05 examples/s]9584 examples [00:08, 1183.23 examples/s]9703 examples [00:08, 1182.82 examples/s]9824 examples [00:08, 1189.41 examples/s]9949 examples [00:08, 1206.65 examples/s]10070 examples [00:08, 1148.13 examples/s]10186 examples [00:08, 1124.75 examples/s]10300 examples [00:08, 1110.21 examples/s]10412 examples [00:09, 1100.03 examples/s]10525 examples [00:09, 1106.13 examples/s]10636 examples [00:09, 1094.19 examples/s]10751 examples [00:09, 1109.43 examples/s]10870 examples [00:09, 1132.23 examples/s]10996 examples [00:09, 1166.65 examples/s]11116 examples [00:09, 1174.02 examples/s]11234 examples [00:09, 1143.66 examples/s]11352 examples [00:09, 1154.30 examples/s]11484 examples [00:09, 1198.97 examples/s]11605 examples [00:10, 1201.55 examples/s]11741 examples [00:10, 1244.15 examples/s]11876 examples [00:10, 1273.42 examples/s]12009 examples [00:10, 1287.14 examples/s]12139 examples [00:10, 1279.84 examples/s]12268 examples [00:10, 1264.77 examples/s]12395 examples [00:10, 1192.13 examples/s]12516 examples [00:10, 1171.19 examples/s]12640 examples [00:10, 1189.41 examples/s]12766 examples [00:11, 1207.62 examples/s]12898 examples [00:11, 1238.90 examples/s]13027 examples [00:11, 1253.16 examples/s]13153 examples [00:11, 1246.80 examples/s]13278 examples [00:11, 1215.71 examples/s]13400 examples [00:11, 1204.88 examples/s]13522 examples [00:11, 1207.31 examples/s]13643 examples [00:11, 1182.91 examples/s]13763 examples [00:11, 1184.81 examples/s]13882 examples [00:11, 1184.41 examples/s]14007 examples [00:12, 1200.44 examples/s]14130 examples [00:12, 1208.69 examples/s]14261 examples [00:12, 1234.82 examples/s]14388 examples [00:12, 1242.06 examples/s]14513 examples [00:12, 1238.39 examples/s]14637 examples [00:12, 1220.89 examples/s]14760 examples [00:12, 1216.85 examples/s]14882 examples [00:12, 1207.75 examples/s]15004 examples [00:12, 1209.98 examples/s]15126 examples [00:12, 1204.17 examples/s]15247 examples [00:13, 1202.47 examples/s]15368 examples [00:13, 1198.76 examples/s]15494 examples [00:13, 1215.25 examples/s]15616 examples [00:13, 1216.36 examples/s]15738 examples [00:13, 1199.91 examples/s]15860 examples [00:13, 1204.27 examples/s]15981 examples [00:13, 1168.52 examples/s]16103 examples [00:13, 1181.10 examples/s]16222 examples [00:13, 1170.29 examples/s]16340 examples [00:13, 1144.63 examples/s]16464 examples [00:14, 1169.73 examples/s]16582 examples [00:14, 1078.37 examples/s]16692 examples [00:14, 1084.08 examples/s]16802 examples [00:14, 1024.30 examples/s]16914 examples [00:14, 1050.03 examples/s]17032 examples [00:14, 1084.74 examples/s]17151 examples [00:14, 1113.82 examples/s]17270 examples [00:14, 1134.11 examples/s]17391 examples [00:14, 1154.39 examples/s]17508 examples [00:15, 1132.93 examples/s]17627 examples [00:15, 1148.94 examples/s]17746 examples [00:15, 1160.20 examples/s]17863 examples [00:15, 1146.15 examples/s]17981 examples [00:15, 1155.76 examples/s]18097 examples [00:15, 1149.16 examples/s]18214 examples [00:15, 1153.62 examples/s]18332 examples [00:15, 1159.09 examples/s]18448 examples [00:15, 1152.51 examples/s]18569 examples [00:15, 1167.22 examples/s]18686 examples [00:16, 1151.95 examples/s]18811 examples [00:16, 1178.52 examples/s]18933 examples [00:16, 1190.53 examples/s]19053 examples [00:16, 1189.49 examples/s]19176 examples [00:16, 1200.92 examples/s]19297 examples [00:16, 1201.73 examples/s]19422 examples [00:16, 1212.56 examples/s]19544 examples [00:16, 1210.62 examples/s]19671 examples [00:16, 1227.58 examples/s]19801 examples [00:16, 1247.57 examples/s]19926 examples [00:17, 1226.32 examples/s]20049 examples [00:17, 1183.88 examples/s]20181 examples [00:17, 1218.87 examples/s]20304 examples [00:17, 1210.68 examples/s]20426 examples [00:17, 1210.44 examples/s]20548 examples [00:17, 1199.69 examples/s]20669 examples [00:17, 1176.46 examples/s]20789 examples [00:17, 1181.40 examples/s]20908 examples [00:17, 1181.22 examples/s]21027 examples [00:18, 1165.01 examples/s]21146 examples [00:18, 1171.06 examples/s]21264 examples [00:18, 1164.77 examples/s]21385 examples [00:18, 1176.79 examples/s]21504 examples [00:18, 1179.73 examples/s]21627 examples [00:18, 1192.87 examples/s]21749 examples [00:18, 1198.76 examples/s]21874 examples [00:18, 1212.03 examples/s]22002 examples [00:18, 1231.30 examples/s]22126 examples [00:18, 1231.59 examples/s]22255 examples [00:19, 1247.70 examples/s]22381 examples [00:19, 1249.69 examples/s]22509 examples [00:19, 1255.75 examples/s]22635 examples [00:19, 1222.92 examples/s]22758 examples [00:19, 1174.15 examples/s]22884 examples [00:19, 1196.99 examples/s]23017 examples [00:19, 1233.80 examples/s]23142 examples [00:19, 1234.75 examples/s]23266 examples [00:19, 1216.43 examples/s]23399 examples [00:19, 1246.53 examples/s]23525 examples [00:20, 1240.61 examples/s]23650 examples [00:20, 1239.06 examples/s]23775 examples [00:20, 1221.34 examples/s]23898 examples [00:20, 1204.29 examples/s]24022 examples [00:20, 1213.30 examples/s]24153 examples [00:20, 1239.55 examples/s]24279 examples [00:20, 1245.37 examples/s]24415 examples [00:20, 1276.00 examples/s]24543 examples [00:20, 1276.26 examples/s]24671 examples [00:20, 1259.14 examples/s]24798 examples [00:21, 1231.88 examples/s]24922 examples [00:21, 1228.50 examples/s]25046 examples [00:21, 1204.18 examples/s]25167 examples [00:21, 1183.18 examples/s]25286 examples [00:21, 1183.59 examples/s]25405 examples [00:21, 1165.87 examples/s]25524 examples [00:21, 1171.78 examples/s]25642 examples [00:21, 1163.61 examples/s]25759 examples [00:21, 1161.70 examples/s]25876 examples [00:22, 1156.54 examples/s]25992 examples [00:22, 1138.17 examples/s]26106 examples [00:22, 1113.49 examples/s]26218 examples [00:22, 1115.40 examples/s]26330 examples [00:22, 1116.04 examples/s]26445 examples [00:22, 1123.87 examples/s]26558 examples [00:22, 1115.74 examples/s]26670 examples [00:22, 1110.76 examples/s]26782 examples [00:22, 1084.36 examples/s]26896 examples [00:22, 1098.06 examples/s]27009 examples [00:23, 1105.60 examples/s]27120 examples [00:23, 1073.44 examples/s]27228 examples [00:23, 1044.23 examples/s]27341 examples [00:23, 1066.21 examples/s]27452 examples [00:23, 1077.98 examples/s]27567 examples [00:23, 1098.62 examples/s]27679 examples [00:23, 1103.56 examples/s]27801 examples [00:23, 1135.48 examples/s]27918 examples [00:23, 1143.29 examples/s]28037 examples [00:23, 1156.27 examples/s]28155 examples [00:24, 1161.90 examples/s]28276 examples [00:24, 1175.14 examples/s]28394 examples [00:24, 1162.72 examples/s]28511 examples [00:24, 1128.73 examples/s]28625 examples [00:24, 1124.67 examples/s]28738 examples [00:24, 1118.92 examples/s]28851 examples [00:24, 1112.59 examples/s]28977 examples [00:24, 1151.22 examples/s]29099 examples [00:24, 1169.91 examples/s]29223 examples [00:24, 1187.62 examples/s]29343 examples [00:25, 1166.21 examples/s]29460 examples [00:25, 1163.25 examples/s]29583 examples [00:25, 1182.23 examples/s]29702 examples [00:25, 1172.54 examples/s]29825 examples [00:25, 1187.59 examples/s]29944 examples [00:25, 1166.62 examples/s]30061 examples [00:25, 1116.58 examples/s]30181 examples [00:25, 1138.18 examples/s]30296 examples [00:25, 1118.55 examples/s]30413 examples [00:26, 1130.46 examples/s]30528 examples [00:26, 1135.18 examples/s]30646 examples [00:26, 1146.98 examples/s]30769 examples [00:26, 1169.24 examples/s]30887 examples [00:26, 1123.27 examples/s]31000 examples [00:26, 1125.21 examples/s]31124 examples [00:26, 1154.39 examples/s]31248 examples [00:26, 1177.06 examples/s]31372 examples [00:26, 1194.81 examples/s]31492 examples [00:26, 1192.30 examples/s]31612 examples [00:27, 1186.80 examples/s]31731 examples [00:27, 1170.35 examples/s]31849 examples [00:27, 1137.89 examples/s]31965 examples [00:27, 1142.00 examples/s]32080 examples [00:27, 1127.41 examples/s]32195 examples [00:27, 1131.77 examples/s]32315 examples [00:27, 1148.53 examples/s]32431 examples [00:27, 1143.35 examples/s]32546 examples [00:27, 1138.46 examples/s]32660 examples [00:27, 1115.71 examples/s]32772 examples [00:28, 1104.58 examples/s]32886 examples [00:28, 1113.24 examples/s]32999 examples [00:28, 1116.39 examples/s]33114 examples [00:28, 1124.02 examples/s]33227 examples [00:28, 1103.81 examples/s]33343 examples [00:28, 1119.05 examples/s]33462 examples [00:28, 1137.04 examples/s]33581 examples [00:28, 1150.07 examples/s]33699 examples [00:28, 1157.06 examples/s]33815 examples [00:29, 1116.44 examples/s]33930 examples [00:29, 1124.02 examples/s]34052 examples [00:29, 1149.04 examples/s]34168 examples [00:29, 1135.47 examples/s]34282 examples [00:29, 1132.00 examples/s]34399 examples [00:29, 1140.47 examples/s]34520 examples [00:29, 1159.55 examples/s]34637 examples [00:29, 1128.83 examples/s]34751 examples [00:29, 1116.79 examples/s]34867 examples [00:29, 1128.98 examples/s]34983 examples [00:30, 1136.82 examples/s]35101 examples [00:30, 1148.07 examples/s]35216 examples [00:30, 1105.05 examples/s]35327 examples [00:30, 1102.27 examples/s]35444 examples [00:30, 1119.31 examples/s]35558 examples [00:30, 1125.36 examples/s]35673 examples [00:30, 1131.78 examples/s]35787 examples [00:30, 1124.89 examples/s]35906 examples [00:30, 1142.24 examples/s]36021 examples [00:30, 1139.53 examples/s]36139 examples [00:31, 1148.45 examples/s]36255 examples [00:31, 1151.01 examples/s]36375 examples [00:31, 1163.91 examples/s]36492 examples [00:31, 1163.38 examples/s]36609 examples [00:31, 1156.03 examples/s]36726 examples [00:31, 1159.59 examples/s]36844 examples [00:31, 1165.04 examples/s]36961 examples [00:31, 1151.78 examples/s]37081 examples [00:31, 1163.72 examples/s]37201 examples [00:31, 1170.46 examples/s]37321 examples [00:32, 1170.50 examples/s]37439 examples [00:32, 1119.68 examples/s]37552 examples [00:32, 1115.33 examples/s]37672 examples [00:32, 1136.54 examples/s]37794 examples [00:32, 1158.05 examples/s]37911 examples [00:32, 1147.75 examples/s]38032 examples [00:32, 1163.70 examples/s]38149 examples [00:32, 1107.65 examples/s]38267 examples [00:32, 1126.20 examples/s]38386 examples [00:33, 1142.35 examples/s]38501 examples [00:33, 1113.61 examples/s]38616 examples [00:33, 1121.94 examples/s]38729 examples [00:33, 1114.55 examples/s]38841 examples [00:33, 1112.01 examples/s]38961 examples [00:33, 1134.89 examples/s]39075 examples [00:33, 1132.38 examples/s]39194 examples [00:33, 1146.61 examples/s]39309 examples [00:33, 1123.46 examples/s]39425 examples [00:33, 1132.52 examples/s]39539 examples [00:34, 1115.80 examples/s]39655 examples [00:34, 1127.25 examples/s]39768 examples [00:34, 1099.31 examples/s]39879 examples [00:34, 1092.84 examples/s]39989 examples [00:34, 1086.71 examples/s]40098 examples [00:34, 1036.25 examples/s]40207 examples [00:34, 1050.48 examples/s]40313 examples [00:34, 1051.85 examples/s]40419 examples [00:34, 1024.92 examples/s]40527 examples [00:34, 1040.59 examples/s]40632 examples [00:35, 1033.52 examples/s]40744 examples [00:35, 1057.80 examples/s]40851 examples [00:35, 1037.87 examples/s]40962 examples [00:35, 1056.17 examples/s]41069 examples [00:35, 1059.05 examples/s]41176 examples [00:35, 1022.52 examples/s]41285 examples [00:35, 1041.22 examples/s]41394 examples [00:35, 1053.01 examples/s]41501 examples [00:35, 1056.07 examples/s]41615 examples [00:36, 1077.17 examples/s]41727 examples [00:36, 1087.81 examples/s]41842 examples [00:36, 1104.00 examples/s]41954 examples [00:36, 1108.40 examples/s]42065 examples [00:36, 1106.75 examples/s]42176 examples [00:36, 1063.26 examples/s]42283 examples [00:36, 1035.06 examples/s]42396 examples [00:36, 1059.84 examples/s]42508 examples [00:36, 1077.11 examples/s]42617 examples [00:36, 1080.04 examples/s]42734 examples [00:37, 1105.51 examples/s]42848 examples [00:37, 1114.48 examples/s]42960 examples [00:37, 1107.72 examples/s]43073 examples [00:37, 1113.75 examples/s]43190 examples [00:37, 1129.97 examples/s]43313 examples [00:37, 1157.42 examples/s]43438 examples [00:37, 1182.98 examples/s]43563 examples [00:37, 1199.77 examples/s]43684 examples [00:37, 1184.47 examples/s]43803 examples [00:37, 1143.69 examples/s]43918 examples [00:38, 1127.26 examples/s]44032 examples [00:38, 1081.74 examples/s]44141 examples [00:38, 1075.31 examples/s]44250 examples [00:38, 1065.32 examples/s]44360 examples [00:38, 1072.55 examples/s]44474 examples [00:38, 1090.23 examples/s]44594 examples [00:38, 1119.76 examples/s]44707 examples [00:38, 1083.29 examples/s]44826 examples [00:38, 1110.96 examples/s]44953 examples [00:39, 1153.91 examples/s]45082 examples [00:39, 1191.60 examples/s]45209 examples [00:39, 1213.92 examples/s]45332 examples [00:39, 1213.02 examples/s]45454 examples [00:39, 1182.88 examples/s]45581 examples [00:39, 1206.67 examples/s]45703 examples [00:39, 1207.31 examples/s]45828 examples [00:39, 1218.78 examples/s]45951 examples [00:39, 1210.29 examples/s]46073 examples [00:39, 1172.56 examples/s]46191 examples [00:40, 1171.42 examples/s]46315 examples [00:40, 1190.56 examples/s]46437 examples [00:40, 1197.92 examples/s]46557 examples [00:40, 1183.17 examples/s]46676 examples [00:40, 1143.54 examples/s]46791 examples [00:40, 1107.94 examples/s]46903 examples [00:40, 1095.59 examples/s]47020 examples [00:40, 1115.05 examples/s]47139 examples [00:40, 1134.68 examples/s]47254 examples [00:40, 1136.28 examples/s]47372 examples [00:41, 1148.19 examples/s]47492 examples [00:41, 1160.78 examples/s]47610 examples [00:41, 1165.17 examples/s]47727 examples [00:41, 1126.70 examples/s]47841 examples [00:41, 1127.88 examples/s]47955 examples [00:41, 1062.78 examples/s]48065 examples [00:41, 1071.92 examples/s]48173 examples [00:41, 982.95 examples/s] 48274 examples [00:41, 953.86 examples/s]48371 examples [00:42, 907.84 examples/s]48464 examples [00:42, 894.37 examples/s]48556 examples [00:42, 900.52 examples/s]48647 examples [00:42, 875.93 examples/s]48736 examples [00:42, 874.09 examples/s]48837 examples [00:42, 910.14 examples/s]48931 examples [00:42, 918.63 examples/s]49037 examples [00:42, 955.85 examples/s]49148 examples [00:42, 995.01 examples/s]49252 examples [00:42, 1007.76 examples/s]49354 examples [00:43, 1008.87 examples/s]49465 examples [00:43, 1034.67 examples/s]49573 examples [00:43, 1047.39 examples/s]49682 examples [00:43, 1058.44 examples/s]49791 examples [00:43, 1065.52 examples/s]49898 examples [00:43, 1060.54 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6631/50000 [00:00<00:00, 66306.43 examples/s] 39%|███▉      | 19598/50000 [00:00<00:00, 77694.90 examples/s] 65%|██████▍   | 32370/50000 [00:00<00:00, 88039.12 examples/s] 91%|█████████ | 45543/50000 [00:00<00:00, 97765.14 examples/s]                                                               0 examples [00:00, ? examples/s]89 examples [00:00, 884.58 examples/s]190 examples [00:00, 916.94 examples/s]290 examples [00:00, 939.91 examples/s]401 examples [00:00, 983.64 examples/s]508 examples [00:00, 1007.77 examples/s]606 examples [00:00, 998.50 examples/s] 716 examples [00:00, 1026.59 examples/s]826 examples [00:00, 1046.71 examples/s]927 examples [00:00, 1019.21 examples/s]1033 examples [00:01, 1029.35 examples/s]1134 examples [00:01, 976.30 examples/s] 1240 examples [00:01, 999.91 examples/s]1349 examples [00:01, 1023.31 examples/s]1459 examples [00:01, 1043.13 examples/s]1567 examples [00:01, 1051.95 examples/s]1677 examples [00:01, 1063.93 examples/s]1785 examples [00:01, 1068.43 examples/s]1892 examples [00:01, 1038.22 examples/s]1997 examples [00:01, 1016.23 examples/s]2110 examples [00:02, 1045.87 examples/s]2225 examples [00:02, 1074.02 examples/s]2337 examples [00:02, 1086.16 examples/s]2446 examples [00:02, 1072.81 examples/s]2561 examples [00:02, 1093.16 examples/s]2673 examples [00:02, 1098.64 examples/s]2790 examples [00:02, 1118.23 examples/s]2916 examples [00:02, 1156.80 examples/s]3038 examples [00:02, 1174.26 examples/s]3156 examples [00:02, 1175.42 examples/s]3279 examples [00:03, 1189.72 examples/s]3399 examples [00:03, 1174.27 examples/s]3517 examples [00:03, 1153.72 examples/s]3633 examples [00:03, 1125.78 examples/s]3752 examples [00:03, 1142.48 examples/s]3867 examples [00:03, 1139.59 examples/s]3988 examples [00:03, 1158.71 examples/s]4106 examples [00:03, 1163.50 examples/s]4223 examples [00:03, 1158.77 examples/s]4339 examples [00:03, 1138.16 examples/s]4453 examples [00:04, 1118.82 examples/s]4566 examples [00:04, 1032.34 examples/s]4671 examples [00:04, 1029.26 examples/s]4775 examples [00:04, 952.50 examples/s] 4881 examples [00:04, 981.05 examples/s]4996 examples [00:04, 1025.41 examples/s]5115 examples [00:04, 1069.16 examples/s]5235 examples [00:04, 1103.87 examples/s]5351 examples [00:04, 1117.00 examples/s]5465 examples [00:05, 1122.79 examples/s]5578 examples [00:05, 1093.39 examples/s]5689 examples [00:05, 1085.67 examples/s]5802 examples [00:05, 1097.40 examples/s]5914 examples [00:05, 1103.47 examples/s]6034 examples [00:05, 1130.12 examples/s]6151 examples [00:05, 1139.57 examples/s]6266 examples [00:05, 1139.26 examples/s]6381 examples [00:05, 1140.18 examples/s]6496 examples [00:05, 1120.78 examples/s]6609 examples [00:06, 1098.23 examples/s]6720 examples [00:06, 1100.34 examples/s]6831 examples [00:06, 1090.90 examples/s]6941 examples [00:06, 1038.42 examples/s]7046 examples [00:06, 1036.52 examples/s]7152 examples [00:06, 1042.73 examples/s]7257 examples [00:06, 1039.37 examples/s]7368 examples [00:06, 1057.53 examples/s]7480 examples [00:06, 1074.60 examples/s]7593 examples [00:07, 1088.89 examples/s]7705 examples [00:07, 1096.55 examples/s]7817 examples [00:07, 1103.20 examples/s]7930 examples [00:07, 1111.08 examples/s]8044 examples [00:07, 1116.57 examples/s]8156 examples [00:07, 1107.91 examples/s]8267 examples [00:07, 1097.97 examples/s]8379 examples [00:07, 1066.89 examples/s]8493 examples [00:07, 1085.94 examples/s]8602 examples [00:07, 1080.19 examples/s]8713 examples [00:08, 1088.67 examples/s]8823 examples [00:08, 1083.07 examples/s]8932 examples [00:08, 1026.22 examples/s]9036 examples [00:08, 908.30 examples/s] 9130 examples [00:08, 885.64 examples/s]9231 examples [00:08, 919.55 examples/s]9340 examples [00:08, 963.34 examples/s]9450 examples [00:08, 998.98 examples/s]9559 examples [00:08, 1022.91 examples/s]9663 examples [00:09, 1004.81 examples/s]9772 examples [00:09, 1028.87 examples/s]9876 examples [00:09, 1020.37 examples/s]9979 examples [00:09, 1012.40 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s] 99%|█████████▉| 9916/10000 [00:00<00:00, 99158.91 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSQU6QN/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSQU6QN/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f66cc67c8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f66cc67c8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f66cc67c8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f66cca3c978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6651af10b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f66cc67c8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f66cc67c8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f66cc67c8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f66c9cb9390>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f66c9cb9550>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f66441327b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f66441327b8> 

  function with postional parmater data_info <function split_train_valid at 0x7f66441327b8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f66441328c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f66441328c8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f66441328c8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=28dd316ff1af15f432b0da9ced7e8c5316ef045f3f96fc4aef6fbb415913282b
  Stored in directory: /tmp/pip-ephem-wheel-cache-0co2160r/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.0
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:09:18, 10.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:44:40, 15.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:04:28, 21.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:45:36, 30.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<5:25:05, 44.0kB/s].vector_cache/glove.6B.zip:   1%|          | 6.85M/862M [00:01<3:46:50, 62.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.2M/862M [00:01<2:38:05, 89.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.2M/862M [00:01<1:50:15, 128kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.5M/862M [00:01<1:16:52, 183kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.2M/862M [00:01<53:41, 260kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.0M/862M [00:01<37:31, 371kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.7M/862M [00:01<26:15, 528kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.0M/862M [00:02<18:23, 750kB/s].vector_cache/glove.6B.zip:   5%|▍         | 38.9M/862M [00:02<12:55, 1.06MB/s].vector_cache/glove.6B.zip:   5%|▍         | 43.0M/862M [00:02<09:05, 1.50MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.1M/862M [00:02<06:26, 2.11MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.9M/862M [00:02<04:35, 2.95MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:03<04:44, 2.84MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.2M/862M [00:03<03:27, 3.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<08:18, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<09:08, 1.47MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:05<07:09, 1.87MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.0M/862M [00:05<05:14, 2.56MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:07<07:33, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<06:51, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<05:16, 2.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:07<03:52, 3.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<57:34, 231kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<44:56, 296kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<32:40, 406kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.1M/862M [00:09<23:04, 574kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:09<16:21, 808kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<40:28, 327kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<32:59, 401kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<24:05, 548kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:11<17:10, 768kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:11<12:12, 1.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<19:21, 679kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:13<17:37, 746kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<13:15, 991kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.4M/862M [00:13<09:29, 1.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:13<06:54, 1.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<21:07, 619kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<19:06, 685kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:15<14:13, 919kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:15<10:19, 1.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:15<07:23, 1.76MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<24:50, 524kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:17<21:21, 609kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:17<15:54, 817kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.0M/862M [00:17<11:21, 1.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<11:47, 1.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:19<12:00, 1.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:19<09:13, 1.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.2M/862M [00:19<06:42, 1.92MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<08:39, 1.49MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:21<09:46, 1.32MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:21<07:38, 1.68MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:21<05:40, 2.26MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:21<04:08, 3.09MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<28:40, 446kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<23:34, 543kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:23<17:24, 735kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:23<12:25, 1.03MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<12:42, 1.00MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<12:37, 1.01MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:25<09:40, 1.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<07:02, 1.80MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<05:06, 2.48MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<33:03, 383kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<23:23, 541kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<18:50, 669kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<16:18, 772kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<12:11, 1.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<08:38, 1.45MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<07:42, 1.63MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<15:34, 804kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<13:01, 961kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<09:31, 1.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<07:09, 1.75MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:59, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:12, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<07:19, 1.70MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<05:17, 2.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:32, 1.45MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<09:24, 1.32MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<07:22, 1.68MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:21, 2.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:52, 1.39MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<09:35, 1.28MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<07:27, 1.65MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:22, 2.28MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:36, 1.42MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<09:21, 1.31MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<07:15, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<05:20, 2.29MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:48, 1.79MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<11:22, 1.07MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<09:18, 1.31MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<06:54, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<04:58, 2.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<20:18, 596kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<17:27, 694kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<12:50, 942kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<09:16, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<09:10, 1.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<09:32, 1.26MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:45<07:20, 1.64MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<05:22, 2.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:45, 1.77MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:49, 1.53MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<06:14, 1.92MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<04:33, 2.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:50, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:06, 1.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<07:06, 1.67MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:09, 2.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<09:34, 1.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<09:35, 1.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<07:27, 1.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:22, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:38, 1.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<09:39, 1.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<07:28, 1.57MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<05:21, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<09:31, 1.23MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<09:31, 1.23MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<07:18, 1.60MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:17, 2.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<09:31, 1.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<09:32, 1.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<07:22, 1.58MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:19, 2.17MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<09:55, 1.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<09:47, 1.18MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<07:33, 1.53MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:25, 2.12MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<09:43, 1.18MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<09:30, 1.21MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<07:16, 1.58MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:17, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:52, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:29, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:55, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:19, 2.63MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<09:00, 1.26MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<09:06, 1.24MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:57, 1.63MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<05:08, 2.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:11, 1.82MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<07:15, 1.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:44, 1.96MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<04:11, 2.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:27, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<07:17, 1.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:49, 1.92MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<04:13, 2.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<07:16, 1.53MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<07:51, 1.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<06:04, 1.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<04:24, 2.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:00, 1.58MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:28, 1.48MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<05:54, 1.87MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<04:16, 2.58MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:40, 1.27MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:39, 1.27MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<07:07, 1.54MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:08, 2.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:52, 2.82MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<1:38:17, 111kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<1:16:26, 143kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<55:28, 197kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<39:11, 278kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<27:44, 393kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<19:43, 551kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<14:10, 766kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<28:16, 384kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<23:23, 464kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<17:09, 632kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<12:19, 879kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<08:58, 1.21MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:38, 1.63MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<11:08, 969kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<15:21, 702kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<12:30, 862kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<09:09, 1.18MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:40, 1.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:02, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<08:28, 1.27MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<10:42, 1.00MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<08:32, 1.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<06:22, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:58, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<03:54, 2.74MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<03:10, 3.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<02:39, 4.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<2:00:54, 88.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<1:29:00, 120kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<1:03:25, 168kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<44:45, 238kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<31:41, 335kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<22:32, 470kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<23:17, 454kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<20:41, 512kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<15:32, 681kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<11:16, 937kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<08:15, 1.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<06:09, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<13:18, 791kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<13:36, 772kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<10:33, 995kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<07:44, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<05:49, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:26, 2.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<15:14, 686kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<16:22, 638kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<12:54, 809kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<09:27, 1.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<07:13, 1.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:33, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:18, 2.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<10:04, 1.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<09:45, 1.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<07:23, 1.40MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:34, 1.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<04:30, 2.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<03:34, 2.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<03:02, 3.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<02:32, 4.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<2:35:22, 66.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<1:53:51, 90.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<1:20:46, 128kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<56:51, 181kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<40:24, 254kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<28:36, 359kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<20:44, 495kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<14:52, 688kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<34:55, 293kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<29:30, 347kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<21:42, 471kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<15:38, 653kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<11:16, 905kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<08:29, 1.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<06:18, 1.61MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:55, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<3:44:10, 45.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<2:41:22, 63.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<1:54:00, 89.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<1:20:10, 127kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<56:23, 180kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<39:50, 254kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<32:17, 313kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<27:10, 372kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<20:08, 501kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<14:32, 693kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<10:31, 956kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<07:42, 1.30MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<05:55, 1.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<13:40, 734kB/s] .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<14:04, 713kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<10:57, 915kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<08:07, 1.23MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<06:04, 1.65MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<04:39, 2.14MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<07:41, 1.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<09:51, 1.01MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<07:50, 1.27MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:55, 1.68MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:43, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<03:38, 2.72MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<02:56, 3.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<07:53, 1.25MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<09:53, 1.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<07:50, 1.26MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [01:46<05:55, 1.66MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:35, 2.15MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<03:36, 2.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<02:57, 3.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<02:26, 4.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<29:19, 335kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<24:54, 394kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<18:32, 529kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<13:23, 732kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<09:41, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<07:05, 1.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<05:36, 1.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<15:12, 642kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<14:56, 653kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<11:32, 845kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<08:28, 1.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<06:16, 1.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:52, 1.99MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<03:46, 2.57MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<08:18, 1.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<10:11, 951kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<08:02, 1.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:59, 1.62MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:41, 2.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:37, 2.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<02:57, 3.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<02:26, 3.95MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<2:25:17, 66.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<1:46:00, 90.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<1:15:12, 128kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<53:00, 181kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<37:24, 256kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<26:25, 362kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:54<19:00, 503kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<33:47, 283kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<27:34, 346kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<20:15, 471kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<14:34, 654kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<10:29, 907kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<07:40, 1.24MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<10:12, 929kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<11:02, 858kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<08:31, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:18, 1.50MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:49, 1.96MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:45, 2.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<02:55, 3.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<11:09, 844kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [02:00<11:08, 845kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<08:33, 1.10MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:43, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:57, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:58, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:52, 2.41MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:50, 1.36MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:02<11:18, 826kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<09:36, 971kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<07:15, 1.29MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:35, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:39, 2.00MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:56, 2.36MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:17, 2.83MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<02:57, 3.14MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<02:36, 3.56MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<08:00, 1.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<08:38, 1.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<06:51, 1.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:21, 1.73MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:12, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:29, 2.64MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:00, 3.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<02:38, 3.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<02:22, 3.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<18:30, 498kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<15:46, 583kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<11:50, 777kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<08:48, 1.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<06:38, 1.38MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<05:03, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<04:15, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:21, 2.72MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<03:11, 2.86MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<5:46:31, 26.4kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<4:05:19, 37.2kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<2:52:14, 53.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<2:00:49, 75.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<1:25:14, 107kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<59:59, 152kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<42:42, 213kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<30:14, 300kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<21:52, 415kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<25:33, 355kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<20:39, 439kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<15:05, 601kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<11:08, 812kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<08:12, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<06:15, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:58, 1.81MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<03:54, 2.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<13:32, 665kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<12:16, 734kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<09:19, 964kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<06:58, 1.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:22, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:15, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:28, 2.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<07:21, 1.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<07:45, 1.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<06:09, 1.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:48, 1.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:50, 2.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:11, 2.78MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<02:42, 3.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<02:22, 3.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<02:08, 4.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<10:35, 835kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<14:42, 601kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<12:22, 714kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<09:20, 945kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<07:01, 1.25MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<05:23, 1.63MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:15, 2.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<03:22, 2.60MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<08:09, 1.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<08:13, 1.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<06:23, 1.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:51, 1.80MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:01, 2.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:11, 2.74MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<02:50, 3.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<02:22, 3.68MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<02:14, 3.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<21:16, 409kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<19:41, 442kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<14:48, 587kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<10:55, 795kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<08:00, 1.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<05:56, 1.46MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<04:46, 1.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:39, 2.36MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<13:14, 652kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<13:31, 638kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<10:26, 827kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<07:49, 1.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<05:52, 1.46MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<04:28, 1.92MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:31, 2.44MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<09:34, 895kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<10:54, 785kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<08:41, 985kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<06:26, 1.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:53, 1.75MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<03:46, 2.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<03:00, 2.83MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<23:43, 358kB/s] .vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<20:24, 416kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<15:07, 562kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<10:54, 777kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<08:05, 1.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<05:56, 1.42MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:26<04:33, 1.85MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<07:09, 1.18MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<08:29, 992kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<06:47, 1.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:05, 1.65MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:52, 2.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<02:59, 2.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<08:17, 1.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<08:59, 930kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<06:59, 1.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:11, 1.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<03:53, 2.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<02:55, 2.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<1:00:20, 137kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<45:09, 184kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<32:16, 257kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<22:47, 362kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<16:09, 510kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<15:07, 544kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<15:07, 543kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<11:47, 696kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<08:34, 956kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<06:12, 1.32MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<07:02, 1.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<07:02, 1.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<05:28, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<04:02, 2.01MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<02:59, 2.71MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<20:03, 403kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<18:02, 448kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<13:37, 593kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<09:48, 822kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<07:00, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<08:37, 930kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<09:37, 833kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<07:37, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<05:34, 1.43MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<04:01, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<07:53, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<08:48, 902kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<06:58, 1.14MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:04, 1.56MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<03:40, 2.14MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<09:28, 832kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<09:35, 821kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<07:27, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<05:23, 1.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<03:52, 2.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<1:03:37, 123kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<47:04, 166kB/s]  .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<33:34, 232kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<23:37, 329kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<18:21, 421kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<15:12, 509kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<11:09, 693kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<07:57, 968kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<07:41, 998kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<07:54, 970kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<06:08, 1.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:25, 1.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<05:10, 1.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<05:57, 1.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:41, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:24, 2.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:33, 2.95MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<08:27, 890kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<08:14, 914kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<06:15, 1.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:32, 1.65MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<03:18, 2.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<09:03, 824kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<08:37, 864kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<06:32, 1.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:43, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<03:26, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<08:58, 823kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<08:43, 847kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<06:40, 1.11MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:46, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:29, 1.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<06:05, 1.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<04:50, 1.51MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:31, 2.07MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:34, 1.59MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<05:25, 1.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<04:17, 1.69MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:08, 2.30MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<02:19, 3.09MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<08:22, 857kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<07:57, 903kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<06:02, 1.19MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:22, 1.63MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<03:11, 2.24MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<08:50, 805kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:06<08:23, 848kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<06:20, 1.12MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:36, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:19, 2.12MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<08:54, 791kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<08:31, 827kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<06:24, 1.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:46, 1.47MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:27, 2.02MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<05:07, 1.36MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<05:44, 1.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<04:27, 1.56MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:15, 2.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:16, 1.61MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<05:06, 1.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<04:05, 1.68MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<02:59, 2.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:00, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:09, 1.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:14<04:10, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:03, 2.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:59, 1.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:45, 1.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:44, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:44, 2.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:01, 3.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<26:30, 253kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<20:24, 328kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<14:39, 456kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<10:21, 643kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<08:55, 743kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<07:58, 831kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<05:56, 1.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:17, 1.53MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:32, 1.44MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:53, 1.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<03:47, 1.73MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:44, 2.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:08, 1.57MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:35, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<03:34, 1.81MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:36, 2.48MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:41, 1.74MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:09, 1.54MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:15, 1.97MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:23, 2.67MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:30, 1.81MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:01, 1.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:07, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:20, 2.71MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:06, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:38, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:54, 2.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:07, 2.93MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:23, 1.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:28, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:25, 1.81MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:31, 2.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:24, 1.80MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:09, 1.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:16, 1.87MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<02:25, 2.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:03, 1.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:49, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:02, 2.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:16, 2.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:53, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:14, 1.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:31, 2.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<01:53, 3.16MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:01, 1.97MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:31, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:45, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<02:01, 2.92MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:22, 1.74MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:30, 1.67MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:41, 2.17MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<01:56, 3.00MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<09:42, 597kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<07:53, 734kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<05:47, 999kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<04:58, 1.15MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:46, 1.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:36, 1.58MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<02:38, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:15, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:32, 1.60MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:44, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:01, 2.77MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:56, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:00, 1.86MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:18, 2.42MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<01:40, 3.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:31, 1.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:24, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:20, 1.64MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:27, 2.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:04, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:16, 1.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:32, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<01:52, 2.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:58, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:14, 1.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:33, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<01:50, 2.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<12:30, 425kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<09:31, 557kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<06:49, 775kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<04:48, 1.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<10:15, 511kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<08:10, 641kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<05:56, 881kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<04:11, 1.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<07:10, 721kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<06:02, 857kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:26, 1.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<03:10, 1.61MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:56, 1.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:46, 1.35MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:52, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:49, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:58, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:17, 2.19MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:42, 2.93MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:35, 1.92MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:47, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:11, 2.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:19, 2.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:35, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:01, 2.41MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<01:28, 3.30MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<04:09, 1.16MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<03:52, 1.25MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:56, 1.64MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:49, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:54, 1.64MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:15, 2.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:38, 2.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<03:42, 1.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:31, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:39, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:55, 2.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<03:07, 1.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:04, 1.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:20, 1.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:41, 2.71MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<04:20, 1.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<03:55, 1.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:57, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:06, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<04:23, 1.02MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<03:57, 1.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:57, 1.51MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:07, 2.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:25, 1.29MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<03:16, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:28, 1.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:47, 2.44MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:55, 1.49MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:54, 1.50MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:12, 1.97MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:37, 2.66MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:24, 1.78MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:32, 1.69MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<01:58, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:24, 2.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<2:04:05, 34.0kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<1:27:37, 48.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<1:01:18, 68.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<42:44, 97.7kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<31:00, 134kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<22:30, 184kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<15:54, 260kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<11:39, 350kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<08:57, 455kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<06:26, 630kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<04:31, 889kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<05:10, 775kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<04:23, 912kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<03:13, 1.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:21, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:35, 1.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:36, 1.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:59, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:26, 2.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:26, 1.58MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:28, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:54, 2.01MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:56, 1.95MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:06, 1.80MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:39, 2.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:11, 3.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<05:22, 694kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<04:30, 829kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:17, 1.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<02:20, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<03:53, 944kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<03:26, 1.06MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:34, 1.42MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:57, 1.85MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:07, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:40, 1.35MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:06, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:31, 2.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<01:07, 3.14MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<13:08, 268kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<10:17, 343kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<07:24, 475kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<05:11, 671kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<04:37, 749kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<04:14, 816kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<03:12, 1.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<02:17, 1.49MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:46, 1.22MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:56, 1.15MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:15, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:38, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:01, 1.64MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:19, 1.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:51, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:20, 2.43MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:09, 1.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:24, 1.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:53, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:21, 2.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:10, 1.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:20, 1.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:50, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:20, 2.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:12, 1.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:24, 1.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:52, 1.66MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:21, 2.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:46, 1.71MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:59, 1.52MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:33, 1.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:07, 2.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:01, 1.47MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:09, 1.38MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:41, 1.75MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:13, 2.40MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<02:07, 1.37MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<02:16, 1.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:45, 1.64MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:16, 2.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:35, 1.78MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:41, 1.67MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:19, 2.14MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:00, 2.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<00:46, 3.55MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<02:11, 1.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<02:40, 1.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:08, 1.28MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:33, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<01:06, 2.42MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<25:10, 107kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<18:43, 144kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<13:18, 202kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<09:19, 287kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<06:31, 406kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<05:40, 463kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<04:57, 529kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<03:41, 711kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:38, 987kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:52, 1.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:33, 1.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:40, 954kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:07, 1.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:31, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<01:05, 2.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<07:14, 344kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<06:00, 414kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<04:23, 566kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<03:07, 789kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<02:12, 1.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<02:49, 856kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<02:50, 854kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<02:11, 1.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:34, 1.52MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<01:07, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<22:09, 106kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<16:23, 143kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<11:38, 201kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<08:06, 286kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<05:38, 405kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<09:09, 249kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<07:12, 316kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<05:14, 434kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:40, 611kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<03:04, 720kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<02:55, 757kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:12, 998kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:34, 1.38MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<01:07, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<1:58:35, 18.1kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<1:23:45, 25.6kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<58:37, 36.4kB/s]  .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<40:33, 52.0kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<28:01, 74.2kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<25:00, 83.0kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<18:13, 114kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<12:53, 160kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<08:56, 228kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<06:39, 301kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<05:21, 374kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<03:55, 509kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:45, 715kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<02:22, 817kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<02:20, 826kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:46, 1.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:16, 1.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:56, 1.99MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<01:26, 1.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<01:54, 974kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:33, 1.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:07, 1.63MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:49, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<01:20, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<01:49, 985kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:30, 1.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:05, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:47, 2.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<01:47, 965kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<02:01, 855kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:37, 1.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:09, 1.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:49, 2.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:49, 907kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<02:00, 825kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<01:35, 1.04MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:08, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:48, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:39, 956kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<01:52, 849kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<01:29, 1.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:04, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:45, 2.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:58, 769kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<02:00, 756kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<01:32, 981kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:06, 1.35MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:48, 1.83MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:35, 2.48MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<25:40, 56.6kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<18:35, 78.1kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<13:06, 110kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<09:04, 157kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<06:13, 223kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<06:14, 222kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<04:59, 277kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<03:37, 380kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<02:31, 536kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<01:45, 752kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<02:57, 445kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<02:39, 495kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:59, 657kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:24, 917kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<00:59, 1.28MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:25, 875kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:30, 825kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:09, 1.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:50, 1.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:35, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:06, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:15, 933kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<01:00, 1.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:43, 1.60MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:30, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:38, 675kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:35, 692kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<01:13, 894kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:52, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:36, 1.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:30, 686kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:28, 700kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<01:08, 903kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:48, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:34, 1.72MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:32, 629kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:30, 639kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:08, 842kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:48, 1.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:34, 1.60MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:42, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:53, 1.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:42, 1.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:30, 1.69MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:21, 2.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:28, 566kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:23, 594kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:03, 778kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:44, 1.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:30, 1.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<01:27, 522kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<01:21, 561kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:01, 738kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:42, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:29, 1.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<01:24, 492kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<01:15, 547kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:56, 732kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:39, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:36, 1.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:30, 1.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:44, 841kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:36, 1.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:25, 1.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:18, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:22, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:36, 910kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:30, 1.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:21, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:15, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:11, 2.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:27, 1.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:27, 1.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:20, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:14, 1.86MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:10, 2.43MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:16, 1.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:26, 944kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:21, 1.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:15, 1.52MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:10, 2.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:15, 1.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:22, 947kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:18, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:12, 1.53MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:08, 2.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:12, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:17, 955kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:14, 1.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:09, 1.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:04, 2.78MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:18, 697kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:15, 784kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:11, 1.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:07, 1.43MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 1.95MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:14, 602kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:15, 562kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:11, 724kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:07, 995kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:04, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:03, 1.12MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:05, 852kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:03, 1.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.42MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.94MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.31MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 928kB/s] .vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 31/400000 [00:00<21:30, 309.82it/s]  0%|          | 996/400000 [00:00<15:13, 436.59it/s]  0%|          | 1930/400000 [00:00<10:51, 611.44it/s]  1%|          | 2749/400000 [00:00<07:49, 846.39it/s]  1%|          | 3668/400000 [00:00<05:40, 1163.18it/s]  1%|          | 4508/400000 [00:00<04:12, 1568.57it/s]  1%|▏         | 5413/400000 [00:00<03:09, 2085.87it/s]  2%|▏         | 6311/400000 [00:00<02:25, 2709.84it/s]  2%|▏         | 7215/400000 [00:00<01:54, 3430.45it/s]  2%|▏         | 8144/400000 [00:01<01:32, 4230.85it/s]  2%|▏         | 9066/400000 [00:01<01:17, 5050.14it/s]  3%|▎         | 10008/400000 [00:01<01:06, 5865.36it/s]  3%|▎         | 10910/400000 [00:01<01:00, 6459.28it/s]  3%|▎         | 11814/400000 [00:01<00:54, 7063.59it/s]  3%|▎         | 12781/400000 [00:01<00:50, 7684.10it/s]  3%|▎         | 13702/400000 [00:01<00:47, 8083.78it/s]  4%|▎         | 14645/400000 [00:01<00:45, 8444.50it/s]  4%|▍         | 15582/400000 [00:01<00:44, 8701.05it/s]  4%|▍         | 16537/400000 [00:01<00:42, 8939.20it/s]  4%|▍         | 17552/400000 [00:02<00:41, 9269.86it/s]  5%|▍         | 18529/400000 [00:02<00:40, 9413.99it/s]  5%|▍         | 19494/400000 [00:02<00:40, 9404.58it/s]  5%|▌         | 20451/400000 [00:02<00:40, 9297.90it/s]  5%|▌         | 21393/400000 [00:02<00:40, 9311.25it/s]  6%|▌         | 22333/400000 [00:02<00:40, 9281.50it/s]  6%|▌         | 23267/400000 [00:02<00:40, 9227.21it/s]  6%|▌         | 24194/400000 [00:02<00:40, 9186.91it/s]  6%|▋         | 25123/400000 [00:02<00:40, 9217.34it/s]  7%|▋         | 26095/400000 [00:02<00:39, 9360.25it/s]  7%|▋         | 27046/400000 [00:03<00:39, 9402.81it/s]  7%|▋         | 27988/400000 [00:03<00:39, 9347.57it/s]  7%|▋         | 28954/400000 [00:03<00:39, 9437.50it/s]  7%|▋         | 29900/400000 [00:03<00:39, 9443.45it/s]  8%|▊         | 30846/400000 [00:03<00:39, 9313.40it/s]  8%|▊         | 31779/400000 [00:03<00:40, 9197.77it/s]  8%|▊         | 32700/400000 [00:03<00:40, 9120.27it/s]  8%|▊         | 33668/400000 [00:03<00:39, 9279.03it/s]  9%|▊         | 34598/400000 [00:03<00:39, 9282.29it/s]  9%|▉         | 35528/400000 [00:03<00:39, 9286.18it/s]  9%|▉         | 36485/400000 [00:04<00:38, 9367.14it/s]  9%|▉         | 37423/400000 [00:04<00:39, 9171.61it/s] 10%|▉         | 38363/400000 [00:04<00:39, 9238.44it/s] 10%|▉         | 39330/400000 [00:04<00:38, 9363.57it/s] 10%|█         | 40303/400000 [00:04<00:37, 9469.73it/s] 10%|█         | 41303/400000 [00:04<00:37, 9622.18it/s] 11%|█         | 42267/400000 [00:04<00:37, 9469.13it/s] 11%|█         | 43216/400000 [00:04<00:38, 9298.20it/s] 11%|█         | 44148/400000 [00:04<00:38, 9229.67it/s] 11%|█▏        | 45087/400000 [00:04<00:38, 9275.52it/s] 12%|█▏        | 46047/400000 [00:05<00:37, 9369.29it/s] 12%|█▏        | 47030/400000 [00:05<00:37, 9501.11it/s] 12%|█▏        | 47998/400000 [00:05<00:36, 9552.14it/s] 12%|█▏        | 48955/400000 [00:05<00:36, 9512.32it/s] 12%|█▏        | 49907/400000 [00:05<00:37, 9414.80it/s] 13%|█▎        | 50850/400000 [00:05<00:37, 9313.34it/s] 13%|█▎        | 51783/400000 [00:05<00:37, 9317.24it/s] 13%|█▎        | 52770/400000 [00:05<00:36, 9475.58it/s] 13%|█▎        | 53772/400000 [00:05<00:35, 9629.62it/s] 14%|█▎        | 54737/400000 [00:05<00:36, 9405.09it/s] 14%|█▍        | 55680/400000 [00:06<00:36, 9403.27it/s] 14%|█▍        | 56627/400000 [00:06<00:36, 9420.16it/s] 14%|█▍        | 57575/400000 [00:06<00:36, 9436.95it/s] 15%|█▍        | 58527/400000 [00:06<00:36, 9459.84it/s] 15%|█▍        | 59474/400000 [00:06<00:36, 9400.29it/s] 15%|█▌        | 60427/400000 [00:06<00:35, 9436.57it/s] 15%|█▌        | 61372/400000 [00:06<00:36, 9361.02it/s] 16%|█▌        | 62343/400000 [00:06<00:35, 9461.15it/s] 16%|█▌        | 63305/400000 [00:06<00:35, 9507.34it/s] 16%|█▌        | 64257/400000 [00:06<00:35, 9453.76it/s] 16%|█▋        | 65203/400000 [00:07<00:35, 9365.34it/s] 17%|█▋        | 66140/400000 [00:07<00:36, 9259.69it/s] 17%|█▋        | 67083/400000 [00:07<00:35, 9310.04it/s] 17%|█▋        | 68103/400000 [00:07<00:34, 9558.11it/s] 17%|█▋        | 69144/400000 [00:07<00:33, 9797.36it/s] 18%|█▊        | 70151/400000 [00:07<00:33, 9875.95it/s] 18%|█▊        | 71141/400000 [00:07<00:33, 9767.96it/s] 18%|█▊        | 72120/400000 [00:07<00:33, 9710.93it/s] 18%|█▊        | 73093/400000 [00:07<00:34, 9515.19it/s] 19%|█▊        | 74047/400000 [00:08<00:34, 9435.34it/s] 19%|█▊        | 74999/400000 [00:08<00:34, 9457.58it/s] 19%|█▉        | 75946/400000 [00:08<00:34, 9443.95it/s] 19%|█▉        | 77022/400000 [00:08<00:32, 9801.21it/s] 20%|█▉        | 78093/400000 [00:08<00:32, 10057.13it/s] 20%|█▉        | 79161/400000 [00:08<00:31, 10236.11it/s] 20%|██        | 80189/400000 [00:08<00:32, 9789.97it/s]  20%|██        | 81216/400000 [00:08<00:32, 9928.11it/s] 21%|██        | 82258/400000 [00:08<00:31, 10067.62it/s] 21%|██        | 83269/400000 [00:08<00:31, 9985.87it/s]  21%|██        | 84275/400000 [00:09<00:31, 10007.06it/s] 21%|██▏       | 85298/400000 [00:09<00:31, 10072.91it/s] 22%|██▏       | 86307/400000 [00:09<00:32, 9802.24it/s]  22%|██▏       | 87290/400000 [00:09<00:32, 9733.71it/s] 22%|██▏       | 88266/400000 [00:09<00:32, 9723.39it/s] 22%|██▏       | 89240/400000 [00:09<00:32, 9556.65it/s] 23%|██▎       | 90212/400000 [00:09<00:32, 9604.01it/s] 23%|██▎       | 91174/400000 [00:09<00:32, 9434.57it/s] 23%|██▎       | 92165/400000 [00:09<00:32, 9569.79it/s] 23%|██▎       | 93132/400000 [00:09<00:31, 9597.09it/s] 24%|██▎       | 94134/400000 [00:10<00:31, 9720.03it/s] 24%|██▍       | 95154/400000 [00:10<00:30, 9857.59it/s] 24%|██▍       | 96142/400000 [00:10<00:30, 9854.74it/s] 24%|██▍       | 97129/400000 [00:10<00:30, 9837.98it/s] 25%|██▍       | 98143/400000 [00:10<00:30, 9923.89it/s] 25%|██▍       | 99137/400000 [00:10<00:31, 9687.88it/s] 25%|██▌       | 100139/400000 [00:10<00:30, 9782.26it/s] 25%|██▌       | 101119/400000 [00:10<00:30, 9665.89it/s] 26%|██▌       | 102087/400000 [00:10<00:31, 9461.63it/s] 26%|██▌       | 103036/400000 [00:10<00:32, 9268.97it/s] 26%|██▌       | 103966/400000 [00:11<00:32, 9210.20it/s] 26%|██▌       | 104889/400000 [00:11<00:32, 9203.94it/s] 26%|██▋       | 105811/400000 [00:11<00:32, 9101.43it/s] 27%|██▋       | 106756/400000 [00:11<00:31, 9202.14it/s] 27%|██▋       | 107743/400000 [00:11<00:31, 9390.20it/s] 27%|██▋       | 108745/400000 [00:11<00:30, 9569.88it/s] 27%|██▋       | 109708/400000 [00:11<00:30, 9587.66it/s] 28%|██▊       | 110669/400000 [00:11<00:30, 9581.68it/s] 28%|██▊       | 111629/400000 [00:11<00:30, 9514.29it/s] 28%|██▊       | 112593/400000 [00:11<00:30, 9551.38it/s] 28%|██▊       | 113549/400000 [00:12<00:30, 9291.93it/s] 29%|██▊       | 114525/400000 [00:12<00:30, 9425.29it/s] 29%|██▉       | 115476/400000 [00:12<00:30, 9447.21it/s] 29%|██▉       | 116423/400000 [00:12<00:30, 9295.49it/s] 29%|██▉       | 117404/400000 [00:12<00:29, 9441.91it/s] 30%|██▉       | 118367/400000 [00:12<00:29, 9497.32it/s] 30%|██▉       | 119318/400000 [00:12<00:30, 9250.42it/s] 30%|███       | 120246/400000 [00:12<00:30, 9161.65it/s] 30%|███       | 121180/400000 [00:12<00:30, 9213.10it/s] 31%|███       | 122125/400000 [00:13<00:29, 9281.06it/s] 31%|███       | 123055/400000 [00:13<00:29, 9262.71it/s] 31%|███       | 123987/400000 [00:13<00:29, 9278.11it/s] 31%|███       | 124934/400000 [00:13<00:29, 9333.23it/s] 31%|███▏      | 125899/400000 [00:13<00:29, 9424.76it/s] 32%|███▏      | 126907/400000 [00:13<00:28, 9611.96it/s] 32%|███▏      | 127870/400000 [00:13<00:28, 9561.22it/s] 32%|███▏      | 128839/400000 [00:13<00:28, 9599.16it/s] 32%|███▏      | 129840/400000 [00:13<00:27, 9716.69it/s] 33%|███▎      | 130910/400000 [00:13<00:26, 9988.43it/s] 33%|███▎      | 131990/400000 [00:14<00:26, 10218.42it/s] 33%|███▎      | 133015/400000 [00:14<00:26, 10214.71it/s] 34%|███▎      | 134046/400000 [00:14<00:25, 10241.35it/s] 34%|███▍      | 135072/400000 [00:14<00:26, 10147.44it/s] 34%|███▍      | 136088/400000 [00:14<00:26, 9969.00it/s]  34%|███▍      | 137087/400000 [00:14<00:26, 9928.44it/s] 35%|███▍      | 138110/400000 [00:14<00:26, 10016.88it/s] 35%|███▍      | 139113/400000 [00:14<00:26, 9786.93it/s]  35%|███▌      | 140164/400000 [00:14<00:26, 9991.07it/s] 35%|███▌      | 141188/400000 [00:14<00:25, 10062.50it/s] 36%|███▌      | 142284/400000 [00:15<00:24, 10315.02it/s] 36%|███▌      | 143319/400000 [00:15<00:25, 10058.75it/s] 36%|███▌      | 144329/400000 [00:15<00:25, 9962.09it/s]  36%|███▋      | 145348/400000 [00:15<00:25, 10028.45it/s] 37%|███▋      | 146418/400000 [00:15<00:24, 10220.14it/s] 37%|███▋      | 147443/400000 [00:15<00:24, 10216.54it/s] 37%|███▋      | 148467/400000 [00:15<00:24, 10209.68it/s] 37%|███▋      | 149490/400000 [00:15<00:24, 10042.13it/s] 38%|███▊      | 150496/400000 [00:15<00:24, 10026.01it/s] 38%|███▊      | 151500/400000 [00:15<00:25, 9876.92it/s]  38%|███▊      | 152525/400000 [00:16<00:24, 9984.03it/s] 38%|███▊      | 153589/400000 [00:16<00:24, 10170.28it/s] 39%|███▊      | 154608/400000 [00:16<00:24, 10152.15it/s] 39%|███▉      | 155625/400000 [00:16<00:24, 9906.01it/s]  39%|███▉      | 156618/400000 [00:16<00:24, 9749.22it/s] 39%|███▉      | 157595/400000 [00:16<00:25, 9671.67it/s] 40%|███▉      | 158564/400000 [00:16<00:25, 9553.47it/s] 40%|███▉      | 159521/400000 [00:16<00:25, 9472.66it/s] 40%|████      | 160470/400000 [00:16<00:25, 9470.27it/s] 40%|████      | 161418/400000 [00:16<00:25, 9455.11it/s] 41%|████      | 162379/400000 [00:17<00:25, 9499.70it/s] 41%|████      | 163330/400000 [00:17<00:24, 9483.03it/s] 41%|████      | 164279/400000 [00:17<00:25, 9424.43it/s] 41%|████▏     | 165261/400000 [00:17<00:24, 9537.71it/s] 42%|████▏     | 166282/400000 [00:17<00:24, 9729.55it/s] 42%|████▏     | 167257/400000 [00:17<00:24, 9591.45it/s] 42%|████▏     | 168218/400000 [00:17<00:24, 9498.07it/s] 42%|████▏     | 169169/400000 [00:17<00:24, 9360.96it/s] 43%|████▎     | 170107/400000 [00:17<00:24, 9244.27it/s] 43%|████▎     | 171033/400000 [00:18<00:25, 9154.50it/s] 43%|████▎     | 171950/400000 [00:18<00:25, 9119.41it/s] 43%|████▎     | 172863/400000 [00:18<00:24, 9116.47it/s] 43%|████▎     | 173776/400000 [00:18<00:25, 9018.27it/s] 44%|████▎     | 174679/400000 [00:18<00:25, 8986.16it/s] 44%|████▍     | 175612/400000 [00:18<00:24, 9085.50it/s] 44%|████▍     | 176563/400000 [00:18<00:24, 9207.52it/s] 44%|████▍     | 177485/400000 [00:18<00:24, 9085.68it/s] 45%|████▍     | 178395/400000 [00:18<00:24, 9082.83it/s] 45%|████▍     | 179316/400000 [00:18<00:24, 9118.96it/s] 45%|████▌     | 180261/400000 [00:19<00:23, 9214.78it/s] 45%|████▌     | 181188/400000 [00:19<00:23, 9229.02it/s] 46%|████▌     | 182112/400000 [00:19<00:23, 9232.11it/s] 46%|████▌     | 183036/400000 [00:19<00:23, 9175.04it/s] 46%|████▌     | 183960/400000 [00:19<00:23, 9193.18it/s] 46%|████▌     | 184901/400000 [00:19<00:23, 9257.10it/s] 46%|████▋     | 185831/400000 [00:19<00:23, 9267.86it/s] 47%|████▋     | 186758/400000 [00:19<00:23, 9241.67it/s] 47%|████▋     | 187683/400000 [00:19<00:23, 9223.12it/s] 47%|████▋     | 188606/400000 [00:19<00:23, 9171.16it/s] 47%|████▋     | 189555/400000 [00:20<00:22, 9262.94it/s] 48%|████▊     | 190497/400000 [00:20<00:22, 9308.97it/s] 48%|████▊     | 191438/400000 [00:20<00:22, 9338.48it/s] 48%|████▊     | 192373/400000 [00:20<00:22, 9300.68it/s] 48%|████▊     | 193304/400000 [00:20<00:22, 9296.84it/s] 49%|████▊     | 194234/400000 [00:20<00:22, 9233.22it/s] 49%|████▉     | 195158/400000 [00:20<00:22, 9199.16it/s] 49%|████▉     | 196079/400000 [00:20<00:22, 9193.44it/s] 49%|████▉     | 196999/400000 [00:20<00:22, 9172.26it/s] 49%|████▉     | 197917/400000 [00:20<00:22, 9101.09it/s] 50%|████▉     | 198828/400000 [00:21<00:22, 9066.28it/s] 50%|████▉     | 199736/400000 [00:21<00:22, 9070.07it/s] 50%|█████     | 200645/400000 [00:21<00:21, 9074.54it/s] 50%|█████     | 201553/400000 [00:21<00:22, 8974.15it/s] 51%|█████     | 202465/400000 [00:21<00:21, 9016.64it/s] 51%|█████     | 203449/400000 [00:21<00:21, 9247.37it/s] 51%|█████     | 204424/400000 [00:21<00:20, 9389.75it/s] 51%|█████▏    | 205407/400000 [00:21<00:20, 9516.58it/s] 52%|█████▏    | 206361/400000 [00:21<00:20, 9492.80it/s] 52%|█████▏    | 207312/400000 [00:21<00:20, 9485.23it/s] 52%|█████▏    | 208270/400000 [00:22<00:20, 9513.11it/s] 52%|█████▏    | 209222/400000 [00:22<00:20, 9436.39it/s] 53%|█████▎    | 210175/400000 [00:22<00:20, 9463.10it/s] 53%|█████▎    | 211122/400000 [00:22<00:20, 9292.77it/s] 53%|█████▎    | 212053/400000 [00:22<00:20, 9250.36it/s] 53%|█████▎    | 213025/400000 [00:22<00:19, 9385.70it/s] 53%|█████▎    | 213965/400000 [00:22<00:19, 9358.68it/s] 54%|█████▎    | 214924/400000 [00:22<00:19, 9426.15it/s] 54%|█████▍    | 215892/400000 [00:22<00:19, 9499.18it/s] 54%|█████▍    | 216865/400000 [00:22<00:19, 9565.68it/s] 54%|█████▍    | 217823/400000 [00:23<00:19, 9490.79it/s] 55%|█████▍    | 218773/400000 [00:23<00:19, 9396.70it/s] 55%|█████▍    | 219714/400000 [00:23<00:19, 9369.19it/s] 55%|█████▌    | 220674/400000 [00:23<00:19, 9437.04it/s] 55%|█████▌    | 221626/400000 [00:23<00:18, 9459.32it/s] 56%|█████▌    | 222585/400000 [00:23<00:18, 9497.31it/s] 56%|█████▌    | 223535/400000 [00:23<00:18, 9409.74it/s] 56%|█████▌    | 224477/400000 [00:23<00:18, 9353.64it/s] 56%|█████▋    | 225413/400000 [00:23<00:18, 9317.98it/s] 57%|█████▋    | 226346/400000 [00:23<00:18, 9256.29it/s] 57%|█████▋    | 227272/400000 [00:24<00:18, 9152.48it/s] 57%|█████▋    | 228230/400000 [00:24<00:18, 9274.93it/s] 57%|█████▋    | 229250/400000 [00:24<00:17, 9533.19it/s] 58%|█████▊    | 230206/400000 [00:24<00:18, 9362.74it/s] 58%|█████▊    | 231145/400000 [00:24<00:18, 9220.80it/s] 58%|█████▊    | 232070/400000 [00:24<00:18, 9118.11it/s] 58%|█████▊    | 232984/400000 [00:24<00:18, 8999.51it/s] 58%|█████▊    | 233886/400000 [00:24<00:19, 8649.81it/s] 59%|█████▊    | 234812/400000 [00:24<00:18, 8823.63it/s] 59%|█████▉    | 235864/400000 [00:25<00:17, 9270.45it/s] 59%|█████▉    | 236839/400000 [00:25<00:17, 9406.58it/s] 59%|█████▉    | 237787/400000 [00:25<00:17, 9258.56it/s] 60%|█████▉    | 238718/400000 [00:25<00:17, 9150.84it/s] 60%|█████▉    | 239637/400000 [00:25<00:17, 9054.55it/s] 60%|██████    | 240575/400000 [00:25<00:17, 9149.50it/s] 60%|██████    | 241509/400000 [00:25<00:17, 9205.35it/s] 61%|██████    | 242482/400000 [00:25<00:16, 9354.59it/s] 61%|██████    | 243450/400000 [00:25<00:16, 9448.98it/s] 61%|██████    | 244399/400000 [00:25<00:16, 9460.04it/s] 61%|██████▏   | 245347/400000 [00:26<00:16, 9383.83it/s] 62%|██████▏   | 246338/400000 [00:26<00:16, 9533.29it/s] 62%|██████▏   | 247297/400000 [00:26<00:15, 9549.29it/s] 62%|██████▏   | 248253/400000 [00:26<00:16, 9430.57it/s] 62%|██████▏   | 249197/400000 [00:26<00:16, 9382.18it/s] 63%|██████▎   | 250136/400000 [00:26<00:16, 9279.86it/s] 63%|██████▎   | 251065/400000 [00:26<00:16, 9254.50it/s] 63%|██████▎   | 251991/400000 [00:26<00:16, 9175.61it/s] 63%|██████▎   | 252914/400000 [00:26<00:16, 9189.45it/s] 63%|██████▎   | 253856/400000 [00:26<00:15, 9256.49it/s] 64%|██████▎   | 254783/400000 [00:27<00:15, 9213.85it/s] 64%|██████▍   | 255780/400000 [00:27<00:15, 9426.04it/s] 64%|██████▍   | 256731/400000 [00:27<00:15, 9449.34it/s] 64%|██████▍   | 257711/400000 [00:27<00:14, 9548.84it/s] 65%|██████▍   | 258718/400000 [00:27<00:14, 9698.67it/s] 65%|██████▍   | 259690/400000 [00:27<00:14, 9663.11it/s] 65%|██████▌   | 260669/400000 [00:27<00:14, 9698.65it/s] 65%|██████▌   | 261701/400000 [00:27<00:14, 9875.55it/s] 66%|██████▌   | 262690/400000 [00:27<00:13, 9817.43it/s] 66%|██████▌   | 263673/400000 [00:27<00:13, 9771.88it/s] 66%|██████▌   | 264651/400000 [00:28<00:14, 9626.95it/s] 66%|██████▋   | 265615/400000 [00:28<00:14, 9493.50it/s] 67%|██████▋   | 266566/400000 [00:28<00:14, 9332.90it/s] 67%|██████▋   | 267501/400000 [00:28<00:14, 9214.10it/s] 67%|██████▋   | 268424/400000 [00:28<00:14, 9109.48it/s] 67%|██████▋   | 269337/400000 [00:28<00:14, 8997.92it/s] 68%|██████▊   | 270238/400000 [00:28<00:14, 8941.96it/s] 68%|██████▊   | 271134/400000 [00:28<00:14, 8897.00it/s] 68%|██████▊   | 272031/400000 [00:28<00:14, 8917.99it/s] 68%|██████▊   | 272937/400000 [00:28<00:14, 8957.28it/s] 68%|██████▊   | 273834/400000 [00:29<00:14, 8949.74it/s] 69%|██████▊   | 274735/400000 [00:29<00:13, 8965.78it/s] 69%|██████▉   | 275632/400000 [00:29<00:13, 8901.45it/s] 69%|██████▉   | 276539/400000 [00:29<00:13, 8948.34it/s] 69%|██████▉   | 277436/400000 [00:29<00:13, 8953.06it/s] 70%|██████▉   | 278359/400000 [00:29<00:13, 9034.07it/s] 70%|██████▉   | 279382/400000 [00:29<00:12, 9361.15it/s] 70%|███████   | 280384/400000 [00:29<00:12, 9548.38it/s] 70%|███████   | 281343/400000 [00:29<00:12, 9283.57it/s] 71%|███████   | 282276/400000 [00:30<00:12, 9284.47it/s] 71%|███████   | 283208/400000 [00:30<00:12, 9231.08it/s] 71%|███████   | 284134/400000 [00:30<00:12, 9140.21it/s] 71%|███████▏  | 285050/400000 [00:30<00:12, 9135.03it/s] 71%|███████▏  | 285973/400000 [00:30<00:12, 9161.73it/s] 72%|███████▏  | 286901/400000 [00:30<00:12, 9194.79it/s] 72%|███████▏  | 287836/400000 [00:30<00:12, 9238.76it/s] 72%|███████▏  | 288802/400000 [00:30<00:11, 9358.99it/s] 72%|███████▏  | 289755/400000 [00:30<00:11, 9408.13it/s] 73%|███████▎  | 290705/400000 [00:30<00:11, 9434.61it/s] 73%|███████▎  | 291649/400000 [00:31<00:11, 9358.77it/s] 73%|███████▎  | 292586/400000 [00:31<00:11, 9345.33it/s] 73%|███████▎  | 293530/400000 [00:31<00:11, 9372.04it/s] 74%|███████▎  | 294468/400000 [00:31<00:11, 9334.22it/s] 74%|███████▍  | 295418/400000 [00:31<00:11, 9381.92it/s] 74%|███████▍  | 296357/400000 [00:31<00:11, 9339.03it/s] 74%|███████▍  | 297292/400000 [00:31<00:11, 9251.52it/s] 75%|███████▍  | 298226/400000 [00:31<00:10, 9277.03it/s] 75%|███████▍  | 299169/400000 [00:31<00:10, 9322.21it/s] 75%|███████▌  | 300152/400000 [00:31<00:10, 9468.47it/s] 75%|███████▌  | 301155/400000 [00:32<00:10, 9628.71it/s] 76%|███████▌  | 302120/400000 [00:32<00:10, 9560.10it/s] 76%|███████▌  | 303077/400000 [00:32<00:10, 9411.40it/s] 76%|███████▌  | 304038/400000 [00:32<00:10, 9468.72it/s] 76%|███████▌  | 304986/400000 [00:32<00:10, 9405.01it/s] 76%|███████▋  | 305949/400000 [00:32<00:09, 9471.31it/s] 77%|███████▋  | 306897/400000 [00:32<00:09, 9376.51it/s] 77%|███████▋  | 307836/400000 [00:32<00:09, 9352.70it/s] 77%|███████▋  | 308778/400000 [00:32<00:09, 9371.84it/s] 77%|███████▋  | 309747/400000 [00:32<00:09, 9464.06it/s] 78%|███████▊  | 310712/400000 [00:33<00:09, 9517.81it/s] 78%|███████▊  | 311665/400000 [00:33<00:09, 9356.73it/s] 78%|███████▊  | 312602/400000 [00:33<00:09, 9178.96it/s] 78%|███████▊  | 313539/400000 [00:33<00:09, 9234.20it/s] 79%|███████▊  | 314541/400000 [00:33<00:09, 9455.80it/s] 79%|███████▉  | 315489/400000 [00:33<00:09, 9380.19it/s] 79%|███████▉  | 316429/400000 [00:33<00:09, 9266.16it/s] 79%|███████▉  | 317358/400000 [00:33<00:08, 9229.59it/s] 80%|███████▉  | 318282/400000 [00:33<00:08, 9159.35it/s] 80%|███████▉  | 319199/400000 [00:33<00:08, 9090.10it/s] 80%|████████  | 320111/400000 [00:34<00:08, 9077.55it/s] 80%|████████  | 321020/400000 [00:34<00:08, 8896.18it/s] 80%|████████  | 321948/400000 [00:34<00:08, 9007.07it/s] 81%|████████  | 322883/400000 [00:34<00:08, 9107.23it/s] 81%|████████  | 323817/400000 [00:34<00:08, 9173.44it/s] 81%|████████  | 324737/400000 [00:34<00:08, 9179.96it/s] 81%|████████▏ | 325656/400000 [00:34<00:08, 9080.55it/s] 82%|████████▏ | 326603/400000 [00:34<00:07, 9192.13it/s] 82%|████████▏ | 327540/400000 [00:34<00:07, 9244.04it/s] 82%|████████▏ | 328466/400000 [00:34<00:07, 9179.18it/s] 82%|████████▏ | 329461/400000 [00:35<00:07, 9394.93it/s] 83%|████████▎ | 330424/400000 [00:35<00:07, 9463.13it/s] 83%|████████▎ | 331471/400000 [00:35<00:07, 9743.81it/s] 83%|████████▎ | 332449/400000 [00:35<00:07, 9648.82it/s] 83%|████████▎ | 333421/400000 [00:35<00:06, 9668.55it/s] 84%|████████▎ | 334452/400000 [00:35<00:06, 9850.76it/s] 84%|████████▍ | 335440/400000 [00:35<00:06, 9828.00it/s] 84%|████████▍ | 336477/400000 [00:35<00:06, 9984.44it/s] 84%|████████▍ | 337478/400000 [00:35<00:06, 9964.22it/s] 85%|████████▍ | 338476/400000 [00:35<00:06, 9775.79it/s] 85%|████████▍ | 339456/400000 [00:36<00:06, 9451.92it/s] 85%|████████▌ | 340405/400000 [00:36<00:06, 9409.66it/s] 85%|████████▌ | 341349/400000 [00:36<00:06, 9192.07it/s] 86%|████████▌ | 342277/400000 [00:36<00:06, 9217.57it/s] 86%|████████▌ | 343259/400000 [00:36<00:06, 9387.82it/s] 86%|████████▌ | 344200/400000 [00:36<00:05, 9318.37it/s] 86%|████████▋ | 345147/400000 [00:36<00:05, 9361.02it/s] 87%|████████▋ | 346154/400000 [00:36<00:05, 9561.14it/s] 87%|████████▋ | 347210/400000 [00:36<00:05, 9838.60it/s] 87%|████████▋ | 348213/400000 [00:37<00:05, 9894.88it/s] 87%|████████▋ | 349209/400000 [00:37<00:05, 9912.70it/s] 88%|████████▊ | 350202/400000 [00:37<00:05, 9762.12it/s] 88%|████████▊ | 351180/400000 [00:37<00:05, 9750.80it/s] 88%|████████▊ | 352157/400000 [00:37<00:04, 9649.73it/s] 88%|████████▊ | 353149/400000 [00:37<00:04, 9726.37it/s] 89%|████████▊ | 354123/400000 [00:37<00:04, 9640.01it/s] 89%|████████▉ | 355088/400000 [00:37<00:04, 9528.02it/s] 89%|████████▉ | 356042/400000 [00:37<00:04, 9475.87it/s] 89%|████████▉ | 356991/400000 [00:37<00:04, 9333.91it/s] 89%|████████▉ | 357926/400000 [00:38<00:04, 9272.97it/s] 90%|████████▉ | 358863/400000 [00:38<00:04, 9299.28it/s] 90%|████████▉ | 359794/400000 [00:38<00:04, 9122.78it/s] 90%|█████████ | 360711/400000 [00:38<00:04, 9136.84it/s] 90%|█████████ | 361637/400000 [00:38<00:04, 9172.82it/s] 91%|█████████ | 362581/400000 [00:38<00:04, 9250.35it/s] 91%|█████████ | 363533/400000 [00:38<00:03, 9329.21it/s] 91%|█████████ | 364467/400000 [00:38<00:03, 9292.78it/s] 91%|█████████▏| 365397/400000 [00:38<00:03, 9255.90it/s] 92%|█████████▏| 366323/400000 [00:38<00:03, 9243.19it/s] 92%|█████████▏| 367248/400000 [00:39<00:03, 9176.92it/s] 92%|█████████▏| 368178/400000 [00:39<00:03, 9212.53it/s] 92%|█████████▏| 369121/400000 [00:39<00:03, 9275.12it/s] 93%|█████████▎| 370067/400000 [00:39<00:03, 9329.39it/s] 93%|█████████▎| 371001/400000 [00:39<00:03, 9291.07it/s] 93%|█████████▎| 371963/400000 [00:39<00:02, 9386.29it/s] 93%|█████████▎| 372986/400000 [00:39<00:02, 9622.08it/s] 93%|█████████▎| 373952/400000 [00:39<00:02, 9631.84it/s] 94%|█████████▎| 374921/400000 [00:39<00:02, 9648.99it/s] 94%|█████████▍| 375887/400000 [00:39<00:02, 9427.76it/s] 94%|█████████▍| 376832/400000 [00:40<00:02, 9383.77it/s] 94%|█████████▍| 377772/400000 [00:40<00:02, 9382.53it/s] 95%|█████████▍| 378714/400000 [00:40<00:02, 9392.38it/s] 95%|█████████▍| 379654/400000 [00:40<00:02, 9355.32it/s] 95%|█████████▌| 380590/400000 [00:40<00:02, 9128.51it/s] 95%|█████████▌| 381549/400000 [00:40<00:01, 9261.36it/s] 96%|█████████▌| 382512/400000 [00:40<00:01, 9368.43it/s] 96%|█████████▌| 383451/400000 [00:40<00:01, 9323.92it/s] 96%|█████████▌| 384407/400000 [00:40<00:01, 9391.95it/s] 96%|█████████▋| 385348/400000 [00:40<00:01, 9352.72it/s] 97%|█████████▋| 386345/400000 [00:41<00:01, 9527.44it/s] 97%|█████████▋| 387299/400000 [00:41<00:01, 9449.31it/s] 97%|█████████▋| 388289/400000 [00:41<00:01, 9580.09it/s] 97%|█████████▋| 389298/400000 [00:41<00:01, 9724.27it/s] 98%|█████████▊| 390272/400000 [00:41<00:01, 9620.07it/s] 98%|█████████▊| 391236/400000 [00:41<00:00, 9586.17it/s] 98%|█████████▊| 392196/400000 [00:41<00:00, 9488.24it/s] 98%|█████████▊| 393146/400000 [00:41<00:00, 9360.87it/s] 99%|█████████▊| 394095/400000 [00:41<00:00, 9397.45it/s] 99%|█████████▉| 395046/400000 [00:41<00:00, 9430.13it/s] 99%|█████████▉| 396001/400000 [00:42<00:00, 9463.32it/s] 99%|█████████▉| 396962/400000 [00:42<00:00, 9505.36it/s] 99%|█████████▉| 397913/400000 [00:42<00:00, 9453.54it/s]100%|█████████▉| 398879/400000 [00:42<00:00, 9512.45it/s]100%|█████████▉| 399831/400000 [00:42<00:00, 9404.53it/s]100%|█████████▉| 399999/400000 [00:42<00:00, 9407.52it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f67388ad5c0>, <torchtext.data.dataset.TabularDataset object at 0x7f67388ad470>, <torchtext.vocab.Vocab object at 0x7f67388ad550>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{

  },
  "preprocessors":[
    {
      "name":"DataReader",
      "uri":"mlmodels.model_tch.util_transformer:TransformerDataReader",
      "args":{
        "train":{
          "uri":"sentence_transformers.readers:NLIDataReader",
          "dataset":"dataset\/text\/AllNLI\/train"
        },
        "test":{
          "uri":"sentence_transformers.readers:STSBenchmarkDataReader",
          "dataset":"dataset\/text\/stsbenchmark\/val"
        }
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.util_transformer:TransformerDataReader {'train': {'uri': 'sentence_transformers.readers:NLIDataReader', 'dataset': 'dataset/text/AllNLI/train'}, 'test': {'uri': 'sentence_transformers.readers:STSBenchmarkDataReader', 'dataset': 'dataset/text/stsbenchmark/val'}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.model_tch.util_transformer.TransformerDataReader'> 
cls_name : TransformerDataReader

  
 Object Creation 

  
 Object Compute 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.json Module ['sentence_transformers.readers', 'STSBenchmarkDataReader'] notfound, module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader', tuple index out of range

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 672, in load_function_uri
    return  getattr(importlib.import_module(package), name)
AttributeError: module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 275, in compute
    obj_preprocessor.compute(input_tmp)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/util_transformer.py", line 275, in compute
    test_func = load_function(self.test_reader)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['sentence_transformers.readers', 'STSBenchmarkDataReader'] notfound, module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader', tuple index out of range





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py --do test  

  #### Test unit Dataloader/Dataset   #################################### 

  


 #################### tf_dataset_download 

  tf_dataset_download mlmodels/preprocess/generic:tf_dataset_download {'train_samples': 500, 'test_samples': 500} 

  MNIST 

  Dataset Name is :  mnist 
WARNING:absl:Dataset mnist is hosted on GCS. It will automatically be downloaded to your
local data directory. If you'd instead prefer to read directly from our public
GCS bucket (recommended if you're running on GCP), you can instead set
data_dir=gs://tfds-data/datasets.

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 29.77 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 53.70 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 27.17 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 27.17 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.11 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.11 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.90 file/s]2020-06-19 18:17:22.713718: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-19 18:17:22.717623: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-19 18:17:22.717794: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562b1e375d70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-19 18:17:22.717807: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist_dataset_small.npy', 'cifar10', 'test', 'train', 'mnist2', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 42%|████▏     | 4194304/9912422 [00:00<00:00, 41099397.36it/s]9920512it [00:00, 32754806.92it/s]                             
0it [00:00, ?it/s]32768it [00:00, 588190.54it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 472762.37it/s]1654784it [00:00, 9052733.17it/s]                          
0it [00:00, ?it/s]8192it [00:00, 190944.72it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw
Processing...
Done!

  


 #################### PandasDataset 

  PandasDataset mlmodels/preprocess/generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': 'ISO-8859-1', 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX'], 'encoding': 'ISO-8859-1'}} 

  


 #################### NumpyDataset 

  NumpyDataset mlmodels/preprocess/generic:NumpyDataset {'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'numpy_loader_args': {}} 

Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/train/mnist.npz


  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6ca6da91408244e26c157e9e6467cc18ede43e71', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff4a6ea4488> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff4a6ea4488> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ff511b51400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ff511b51400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff530d6dea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff530d6dea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff4bee8e0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff4bee8e0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff4bee8e0d0> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 18%|█▊        | 1753088/9912422 [00:00<00:00, 17383434.07it/s] 65%|██████▍   | 6438912/9912422 [00:00<00:00, 21426759.54it/s]9920512it [00:00, 28616593.62it/s]                             
0it [00:00, ?it/s]32768it [00:00, 525571.13it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 155941.75it/s]1654784it [00:00, 11100041.34it/s]                         
0it [00:00, ?it/s]8192it [00:00, 176528.78it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4a6def550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4bd04f898>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ff4bee85d08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ff4bee85d08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ff4bee85d08> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:20,  1.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:20,  1.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 2/162 [00:00<01:05,  2.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:05,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 3/162 [00:00<00:51,  3.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:51,  3.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<00:41,  3.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:41,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<00:41,  3.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:01<00:33,  4.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<00:33,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<00:32,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:01<00:26,  5.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:26,  5.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:26,  5.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:01<00:20,  7.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:20,  7.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:20,  7.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:01<00:16,  9.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:16,  9.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:16,  9.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:16,  9.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:13, 11.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:13, 11.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:13, 11.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:13, 11.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:01<00:10, 13.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:10, 13.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:10, 13.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:10, 13.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:10, 13.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:01<00:08, 16.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:08, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:08, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:08, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:08, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:08, 16.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:01<00:06, 20.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:06, 20.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:06, 20.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:02<00:06, 20.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:02<00:06, 20.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:02<00:06, 20.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:02<00:05, 24.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:02<00:05, 24.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:02<00:05, 24.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:02<00:05, 24.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:02<00:05, 24.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:02<00:05, 24.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:02<00:05, 24.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:02<00:04, 29.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:02<00:04, 29.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:02<00:04, 29.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:02<00:04, 29.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:02<00:04, 29.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:02<00:04, 29.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:02<00:04, 29.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:02<00:03, 29.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:02<00:03, 35.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:02<00:03, 35.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:02<00:03, 35.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:02<00:03, 35.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:02<00:03, 35.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:02<00:03, 35.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:03, 35.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:03, 35.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:02<00:02, 40.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:02, 40.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:02, 40.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:02, 40.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:02, 40.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:02, 40.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:02, 40.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:02, 40.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:02<00:02, 45.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:02, 45.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:02, 45.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:02, 45.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:02, 45.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:02, 45.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:02, 45.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:02, 45.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:02<00:01, 49.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:01, 49.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:01, 49.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:01, 49.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:01, 49.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:01, 49.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:01, 49.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:02<00:01, 51.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:02<00:01, 54.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 54.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 57.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 57.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 57.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 57.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 57.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:03<00:01, 57.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:03<00:01, 57.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:03<00:01, 57.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:01, 57.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:01, 57.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:03<00:01, 57.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:03<00:01, 57.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:03<00:01, 57.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:03<00:01, 57.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:03<00:01, 57.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:03<00:01, 57.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:03<00:01, 57.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:03<00:01, 57.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:01, 57.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:03<00:01, 57.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 57.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:01, 57.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:00, 57.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:00, 57.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:00, 58.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:00, 58.82 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:00, 58.82 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:00, 58.82 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:00, 58.82 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:00, 58.82 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:00, 58.82 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:00, 58.82 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 59.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 59.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 59.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 59.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 59.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 59.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 59.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 59.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 60.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 60.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 60.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 60.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 60.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 60.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 60.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 60.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 61.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 61.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 61.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 61.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 61.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 61.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 61.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 61.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 60.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 60.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 60.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 60.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 60.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 60.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 60.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 60.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 61.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 61.49 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 61.49 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 61.49 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 61.49 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 61.49 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 61.49 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 61.49 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 61.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 61.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 61.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:04<00:00, 61.73 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:04<00:00, 61.73 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 61.73 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 61.73 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 61.73 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 62.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 62.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 62.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 62.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 62.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 62.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 62.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 62.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.18s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 62.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 62.25 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.28s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 62.25 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.28s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.28s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 25.80 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.28s/ url]
0 examples [00:00, ? examples/s]2020-08-14 18:17:47.405148: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-14 18:17:47.422261: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-08-14 18:17:47.422431: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563593a7baf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-14 18:17:47.422448: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
48 examples [00:00, 476.56 examples/s]155 examples [00:00, 571.42 examples/s]264 examples [00:00, 666.42 examples/s]375 examples [00:00, 756.67 examples/s]472 examples [00:00, 808.49 examples/s]577 examples [00:00, 867.59 examples/s]689 examples [00:00, 929.44 examples/s]799 examples [00:00, 974.37 examples/s]911 examples [00:00, 1013.37 examples/s]1023 examples [00:01, 1041.55 examples/s]1132 examples [00:01, 1055.07 examples/s]1244 examples [00:01, 1071.70 examples/s]1355 examples [00:01, 1082.34 examples/s]1465 examples [00:01, 1085.38 examples/s]1575 examples [00:01, 1085.95 examples/s]1684 examples [00:01, 1086.79 examples/s]1793 examples [00:01, 1073.58 examples/s]1901 examples [00:01, 1071.14 examples/s]2009 examples [00:01, 1073.49 examples/s]2117 examples [00:02, 1069.21 examples/s]2225 examples [00:02, 1071.83 examples/s]2336 examples [00:02, 1082.72 examples/s]2448 examples [00:02, 1093.31 examples/s]2558 examples [00:02, 1081.83 examples/s]2667 examples [00:02, 1031.96 examples/s]2774 examples [00:02, 1040.92 examples/s]2879 examples [00:02, 1042.75 examples/s]2990 examples [00:02, 1061.74 examples/s]3102 examples [00:02, 1077.33 examples/s]3210 examples [00:03, 1058.91 examples/s]3320 examples [00:03, 1069.90 examples/s]3432 examples [00:03, 1082.77 examples/s]3541 examples [00:03, 1078.39 examples/s]3649 examples [00:03, 1068.96 examples/s]3757 examples [00:03, 1049.33 examples/s]3866 examples [00:03, 1060.47 examples/s]3978 examples [00:03, 1075.96 examples/s]4088 examples [00:03, 1081.43 examples/s]4199 examples [00:03, 1089.66 examples/s]4309 examples [00:04, 1092.09 examples/s]4419 examples [00:04, 1094.29 examples/s]4530 examples [00:04, 1097.08 examples/s]4640 examples [00:04, 1054.58 examples/s]4746 examples [00:04, 1046.17 examples/s]4851 examples [00:04, 979.23 examples/s] 4961 examples [00:04, 1010.45 examples/s]5074 examples [00:04, 1041.41 examples/s]5187 examples [00:04, 1064.47 examples/s]5300 examples [00:05, 1080.99 examples/s]5414 examples [00:05, 1096.62 examples/s]5525 examples [00:05, 1092.23 examples/s]5635 examples [00:05, 1046.18 examples/s]5741 examples [00:05, 1045.05 examples/s]5846 examples [00:05, 1040.50 examples/s]5951 examples [00:05, 1012.67 examples/s]6059 examples [00:05, 1031.65 examples/s]6164 examples [00:05, 1036.61 examples/s]6276 examples [00:05, 1059.00 examples/s]6383 examples [00:06, 1047.82 examples/s]6489 examples [00:06, 1049.43 examples/s]6595 examples [00:06, 1047.67 examples/s]6706 examples [00:06, 1065.26 examples/s]6819 examples [00:06, 1082.52 examples/s]6931 examples [00:06, 1093.02 examples/s]7041 examples [00:06, 1072.63 examples/s]7149 examples [00:06, 1071.41 examples/s]7258 examples [00:06, 1076.41 examples/s]7369 examples [00:06, 1086.09 examples/s]7481 examples [00:07, 1094.27 examples/s]7591 examples [00:07, 1094.65 examples/s]7701 examples [00:07, 1096.10 examples/s]7813 examples [00:07, 1101.41 examples/s]7924 examples [00:07, 1103.11 examples/s]8035 examples [00:07, 1101.56 examples/s]8146 examples [00:07, 1102.27 examples/s]8257 examples [00:07, 1100.29 examples/s]8369 examples [00:07, 1104.93 examples/s]8480 examples [00:07, 1106.10 examples/s]8593 examples [00:08, 1110.71 examples/s]8706 examples [00:08, 1115.12 examples/s]8818 examples [00:08, 1110.18 examples/s]8932 examples [00:08, 1118.64 examples/s]9046 examples [00:08, 1124.14 examples/s]9159 examples [00:08, 1125.52 examples/s]9272 examples [00:08, 1051.15 examples/s]9381 examples [00:08, 1060.42 examples/s]9488 examples [00:08, 1039.17 examples/s]9602 examples [00:08, 1066.47 examples/s]9715 examples [00:09, 1083.31 examples/s]9827 examples [00:09, 1093.08 examples/s]9937 examples [00:09, 1084.40 examples/s]10046 examples [00:09, 1029.65 examples/s]10156 examples [00:09, 1048.97 examples/s]10262 examples [00:09, 1037.94 examples/s]10372 examples [00:09, 1053.35 examples/s]10478 examples [00:09, 1031.70 examples/s]10582 examples [00:09, 1005.17 examples/s]10694 examples [00:10, 1035.02 examples/s]10799 examples [00:10, 1035.05 examples/s]10909 examples [00:10, 1052.73 examples/s]11022 examples [00:10, 1073.56 examples/s]11135 examples [00:10, 1089.06 examples/s]11245 examples [00:10, 1061.98 examples/s]11352 examples [00:10, 1033.89 examples/s]11460 examples [00:10, 1045.33 examples/s]11571 examples [00:10, 1061.58 examples/s]11684 examples [00:10, 1078.89 examples/s]11798 examples [00:11, 1093.93 examples/s]11908 examples [00:11, 1094.62 examples/s]12018 examples [00:11, 1092.61 examples/s]12130 examples [00:11, 1098.09 examples/s]12243 examples [00:11, 1105.17 examples/s]12354 examples [00:11, 1099.78 examples/s]12465 examples [00:11, 1094.95 examples/s]12577 examples [00:11, 1100.52 examples/s]12690 examples [00:11, 1106.60 examples/s]12802 examples [00:11, 1108.41 examples/s]12913 examples [00:12, 1087.59 examples/s]13025 examples [00:12, 1096.71 examples/s]13135 examples [00:12, 1065.55 examples/s]13249 examples [00:12, 1084.55 examples/s]13363 examples [00:12, 1098.02 examples/s]13476 examples [00:12, 1105.58 examples/s]13587 examples [00:12, 1064.28 examples/s]13694 examples [00:12, 1063.40 examples/s]13805 examples [00:12, 1076.60 examples/s]13918 examples [00:13, 1090.44 examples/s]14031 examples [00:13, 1100.89 examples/s]14143 examples [00:13, 1104.60 examples/s]14254 examples [00:13, 1102.65 examples/s]14367 examples [00:13, 1109.71 examples/s]14479 examples [00:13, 1110.11 examples/s]14591 examples [00:13, 1111.07 examples/s]14703 examples [00:13, 1110.27 examples/s]14815 examples [00:13, 1098.58 examples/s]14926 examples [00:13, 1101.22 examples/s]15037 examples [00:14, 1093.94 examples/s]15147 examples [00:14, 1057.40 examples/s]15259 examples [00:14, 1075.22 examples/s]15367 examples [00:14, 1071.68 examples/s]15475 examples [00:14, 1051.39 examples/s]15584 examples [00:14, 1060.10 examples/s]15696 examples [00:14, 1075.01 examples/s]15804 examples [00:14, 1061.16 examples/s]15912 examples [00:14, 1065.15 examples/s]16025 examples [00:14, 1081.28 examples/s]16138 examples [00:15, 1094.77 examples/s]16250 examples [00:15, 1100.00 examples/s]16363 examples [00:15, 1106.57 examples/s]16474 examples [00:15, 1091.24 examples/s]16585 examples [00:15, 1096.72 examples/s]16698 examples [00:15, 1105.35 examples/s]16809 examples [00:15, 1105.35 examples/s]16920 examples [00:15, 1102.15 examples/s]17031 examples [00:15, 1065.33 examples/s]17138 examples [00:15, 1062.16 examples/s]17250 examples [00:16, 1077.63 examples/s]17362 examples [00:16, 1088.71 examples/s]17475 examples [00:16, 1099.74 examples/s]17586 examples [00:16, 1098.98 examples/s]17698 examples [00:16, 1103.87 examples/s]17809 examples [00:16, 1092.55 examples/s]17920 examples [00:16, 1096.51 examples/s]18031 examples [00:16, 1099.86 examples/s]18142 examples [00:16, 1094.39 examples/s]18252 examples [00:16, 1095.90 examples/s]18363 examples [00:17, 1099.47 examples/s]18474 examples [00:17, 1102.61 examples/s]18586 examples [00:17, 1105.92 examples/s]18697 examples [00:17, 1104.11 examples/s]18808 examples [00:17, 1102.18 examples/s]18920 examples [00:17, 1105.88 examples/s]19031 examples [00:17, 1103.67 examples/s]19142 examples [00:17, 1095.37 examples/s]19253 examples [00:17, 1098.70 examples/s]19363 examples [00:17, 1081.74 examples/s]19472 examples [00:18, 1072.00 examples/s]19585 examples [00:18, 1088.05 examples/s]19697 examples [00:18, 1094.90 examples/s]19808 examples [00:18, 1098.81 examples/s]19919 examples [00:18, 1100.29 examples/s]20030 examples [00:18, 1050.13 examples/s]20142 examples [00:18, 1069.33 examples/s]20250 examples [00:18, 1060.41 examples/s]20357 examples [00:18, 1050.50 examples/s]20463 examples [00:19, 1051.90 examples/s]20570 examples [00:19, 1054.68 examples/s]20676 examples [00:19, 1054.33 examples/s]20782 examples [00:19, 1043.49 examples/s]20893 examples [00:19, 1059.22 examples/s]21002 examples [00:19, 1067.59 examples/s]21113 examples [00:19, 1078.52 examples/s]21222 examples [00:19, 1079.34 examples/s]21333 examples [00:19, 1086.32 examples/s]21442 examples [00:19, 1085.88 examples/s]21553 examples [00:20, 1090.41 examples/s]21663 examples [00:20, 1091.67 examples/s]21773 examples [00:20, 1087.06 examples/s]21882 examples [00:20, 1085.12 examples/s]21991 examples [00:20, 1074.05 examples/s]22101 examples [00:20, 1079.78 examples/s]22212 examples [00:20, 1087.94 examples/s]22321 examples [00:20, 1055.93 examples/s]22431 examples [00:20, 1067.94 examples/s]22542 examples [00:20, 1078.86 examples/s]22653 examples [00:21, 1087.76 examples/s]22765 examples [00:21, 1094.42 examples/s]22877 examples [00:21, 1099.82 examples/s]22988 examples [00:21, 1099.19 examples/s]23098 examples [00:21, 1097.36 examples/s]23210 examples [00:21, 1102.37 examples/s]23321 examples [00:21, 1084.93 examples/s]23432 examples [00:21, 1091.72 examples/s]23542 examples [00:21, 1090.64 examples/s]23652 examples [00:21, 1093.15 examples/s]23765 examples [00:22, 1101.94 examples/s]23878 examples [00:22, 1107.95 examples/s]23991 examples [00:22, 1111.89 examples/s]24103 examples [00:22, 1103.76 examples/s]24214 examples [00:22, 1103.37 examples/s]24325 examples [00:22, 1086.33 examples/s]24434 examples [00:22, 1086.85 examples/s]24543 examples [00:22, 1087.40 examples/s]24654 examples [00:22, 1091.78 examples/s]24764 examples [00:22, 1092.05 examples/s]24875 examples [00:23, 1097.17 examples/s]24987 examples [00:23, 1103.69 examples/s]25098 examples [00:23, 1105.10 examples/s]25209 examples [00:23, 1099.85 examples/s]25320 examples [00:23, 1091.55 examples/s]25433 examples [00:23, 1100.74 examples/s]25544 examples [00:23, 1101.52 examples/s]25657 examples [00:23, 1108.92 examples/s]25768 examples [00:23, 1102.82 examples/s]25879 examples [00:23, 1096.49 examples/s]25989 examples [00:24, 1073.44 examples/s]26097 examples [00:24, 1073.75 examples/s]26205 examples [00:24, 1039.14 examples/s]26314 examples [00:24, 1053.44 examples/s]26420 examples [00:24, 1049.14 examples/s]26531 examples [00:24, 1065.41 examples/s]26640 examples [00:24, 1070.18 examples/s]26751 examples [00:24, 1080.70 examples/s]26862 examples [00:24, 1088.50 examples/s]26971 examples [00:25, 1084.83 examples/s]27080 examples [00:25, 1073.91 examples/s]27188 examples [00:25, 1056.46 examples/s]27300 examples [00:25, 1074.45 examples/s]27410 examples [00:25, 1081.14 examples/s]27519 examples [00:25, 1071.37 examples/s]27632 examples [00:25, 1088.14 examples/s]27741 examples [00:25, 1058.26 examples/s]27853 examples [00:25, 1074.16 examples/s]27966 examples [00:25, 1088.83 examples/s]28076 examples [00:26, 1087.73 examples/s]28185 examples [00:26, 1050.97 examples/s]28295 examples [00:26, 1062.63 examples/s]28403 examples [00:26, 1067.32 examples/s]28513 examples [00:26, 1075.37 examples/s]28621 examples [00:26, 1066.15 examples/s]28728 examples [00:26, 1052.81 examples/s]28837 examples [00:26, 1058.57 examples/s]28947 examples [00:26, 1069.97 examples/s]29058 examples [00:26, 1080.67 examples/s]29167 examples [00:27, 1078.89 examples/s]29281 examples [00:27, 1093.78 examples/s]29391 examples [00:27, 1084.81 examples/s]29500 examples [00:27, 1081.73 examples/s]29609 examples [00:27, 1071.08 examples/s]29720 examples [00:27, 1079.78 examples/s]29829 examples [00:27, 1076.09 examples/s]29941 examples [00:27, 1087.51 examples/s]30050 examples [00:27, 1011.43 examples/s]30159 examples [00:28, 1032.03 examples/s]30264 examples [00:28, 1024.10 examples/s]30369 examples [00:28, 1030.40 examples/s]30473 examples [00:28, 1014.88 examples/s]30585 examples [00:28, 1041.74 examples/s]30690 examples [00:28, 1041.57 examples/s]30800 examples [00:28, 1056.64 examples/s]30912 examples [00:28, 1073.45 examples/s]31024 examples [00:28, 1084.15 examples/s]31133 examples [00:28, 1085.10 examples/s]31243 examples [00:29, 1087.14 examples/s]31352 examples [00:29, 1075.34 examples/s]31463 examples [00:29, 1083.55 examples/s]31572 examples [00:29, 1076.79 examples/s]31683 examples [00:29, 1084.11 examples/s]31792 examples [00:29, 1060.57 examples/s]31899 examples [00:29, 1047.21 examples/s]32008 examples [00:29, 1059.65 examples/s]32115 examples [00:29, 1061.40 examples/s]32223 examples [00:29, 1066.50 examples/s]32332 examples [00:30, 1072.77 examples/s]32440 examples [00:30, 1069.23 examples/s]32547 examples [00:30, 1066.30 examples/s]32654 examples [00:30, 1060.82 examples/s]32766 examples [00:30, 1077.30 examples/s]32874 examples [00:30, 1074.37 examples/s]32985 examples [00:30, 1083.46 examples/s]33094 examples [00:30, 1067.23 examples/s]33201 examples [00:30, 1009.64 examples/s]33309 examples [00:30, 1028.66 examples/s]33420 examples [00:31, 1050.68 examples/s]33530 examples [00:31, 1063.15 examples/s]33637 examples [00:31, 1063.64 examples/s]33745 examples [00:31, 1066.08 examples/s]33856 examples [00:31, 1076.20 examples/s]33966 examples [00:31, 1080.50 examples/s]34076 examples [00:31, 1084.33 examples/s]34187 examples [00:31, 1091.48 examples/s]34298 examples [00:31, 1094.63 examples/s]34410 examples [00:31, 1101.89 examples/s]34521 examples [00:32, 1103.43 examples/s]34632 examples [00:32, 1099.81 examples/s]34745 examples [00:32, 1106.88 examples/s]34856 examples [00:32, 1101.55 examples/s]34968 examples [00:32, 1106.53 examples/s]35079 examples [00:32, 1107.37 examples/s]35190 examples [00:32, 1085.16 examples/s]35300 examples [00:32, 1088.17 examples/s]35413 examples [00:32, 1098.23 examples/s]35526 examples [00:32, 1106.02 examples/s]35637 examples [00:33, 1104.35 examples/s]35748 examples [00:33, 1096.69 examples/s]35859 examples [00:33, 1099.85 examples/s]35970 examples [00:33, 1097.82 examples/s]36080 examples [00:33, 1086.61 examples/s]36192 examples [00:33, 1093.56 examples/s]36302 examples [00:33, 1093.03 examples/s]36414 examples [00:33, 1100.38 examples/s]36528 examples [00:33, 1111.70 examples/s]36640 examples [00:34, 1113.08 examples/s]36752 examples [00:34, 1114.79 examples/s]36864 examples [00:34, 1106.56 examples/s]36977 examples [00:34, 1112.31 examples/s]37089 examples [00:34, 1107.82 examples/s]37200 examples [00:34, 1073.21 examples/s]37311 examples [00:34, 1083.44 examples/s]37420 examples [00:34, 1081.44 examples/s]37533 examples [00:34, 1092.57 examples/s]37647 examples [00:34, 1104.55 examples/s]37760 examples [00:35, 1109.02 examples/s]37873 examples [00:35, 1112.05 examples/s]37985 examples [00:35, 1102.91 examples/s]38096 examples [00:35, 1103.21 examples/s]38207 examples [00:35, 1104.87 examples/s]38318 examples [00:35, 1104.24 examples/s]38429 examples [00:35, 1078.22 examples/s]38537 examples [00:35, 1054.82 examples/s]38643 examples [00:35, 1056.20 examples/s]38749 examples [00:35, 1053.40 examples/s]38855 examples [00:36, 1040.01 examples/s]38960 examples [00:36, 1042.77 examples/s]39067 examples [00:36, 1048.75 examples/s]39172 examples [00:36, 1038.75 examples/s]39282 examples [00:36, 1055.28 examples/s]39389 examples [00:36, 1058.26 examples/s]39495 examples [00:36, 1039.58 examples/s]39604 examples [00:36, 1053.48 examples/s]39710 examples [00:36, 1049.80 examples/s]39816 examples [00:36, 1001.57 examples/s]39929 examples [00:37, 1035.50 examples/s]40034 examples [00:37, 1004.56 examples/s]40144 examples [00:37, 1030.75 examples/s]40254 examples [00:37, 1048.13 examples/s]40363 examples [00:37, 1057.82 examples/s]40473 examples [00:37, 1068.70 examples/s]40581 examples [00:37, 1067.97 examples/s]40692 examples [00:37, 1078.40 examples/s]40802 examples [00:37, 1083.96 examples/s]40913 examples [00:38, 1090.67 examples/s]41023 examples [00:38, 1092.35 examples/s]41134 examples [00:38, 1097.18 examples/s]41244 examples [00:38, 1094.58 examples/s]41356 examples [00:38, 1101.22 examples/s]41467 examples [00:38, 1101.63 examples/s]41579 examples [00:38, 1106.87 examples/s]41690 examples [00:38, 1102.46 examples/s]41801 examples [00:38, 1102.07 examples/s]41912 examples [00:38, 1103.00 examples/s]42023 examples [00:39, 1104.02 examples/s]42134 examples [00:39, 1090.10 examples/s]42245 examples [00:39, 1093.58 examples/s]42355 examples [00:39, 1068.26 examples/s]42462 examples [00:39, 1057.23 examples/s]42570 examples [00:39, 1063.07 examples/s]42677 examples [00:39, 1063.87 examples/s]42787 examples [00:39, 1072.16 examples/s]42897 examples [00:39, 1078.12 examples/s]43010 examples [00:39, 1090.78 examples/s]43121 examples [00:40, 1096.07 examples/s]43233 examples [00:40, 1102.28 examples/s]43344 examples [00:40, 1090.89 examples/s]43454 examples [00:40, 1093.46 examples/s]43567 examples [00:40, 1101.63 examples/s]43678 examples [00:40, 1099.52 examples/s]43790 examples [00:40, 1103.66 examples/s]43901 examples [00:40, 1103.60 examples/s]44012 examples [00:40, 1093.26 examples/s]44124 examples [00:40, 1099.68 examples/s]44236 examples [00:41, 1104.09 examples/s]44349 examples [00:41, 1108.97 examples/s]44461 examples [00:41, 1110.76 examples/s]44573 examples [00:41, 1100.09 examples/s]44685 examples [00:41, 1104.48 examples/s]44796 examples [00:41, 1103.30 examples/s]44908 examples [00:41, 1106.60 examples/s]45020 examples [00:41, 1108.40 examples/s]45131 examples [00:41, 1102.34 examples/s]45244 examples [00:41, 1108.09 examples/s]45355 examples [00:42, 1094.77 examples/s]45467 examples [00:42, 1099.72 examples/s]45580 examples [00:42, 1108.09 examples/s]45691 examples [00:42, 1098.97 examples/s]45801 examples [00:42, 1096.37 examples/s]45912 examples [00:42, 1099.39 examples/s]46022 examples [00:42, 1068.90 examples/s]46131 examples [00:42, 1072.44 examples/s]46241 examples [00:42, 1078.17 examples/s]46349 examples [00:42, 1076.98 examples/s]46459 examples [00:43, 1083.72 examples/s]46569 examples [00:43, 1086.23 examples/s]46679 examples [00:43, 1090.18 examples/s]46789 examples [00:43, 1081.00 examples/s]46901 examples [00:43, 1090.69 examples/s]47011 examples [00:43, 1076.99 examples/s]47119 examples [00:43, 1062.13 examples/s]47229 examples [00:43, 1072.82 examples/s]47337 examples [00:43, 1029.64 examples/s]47446 examples [00:44, 1044.59 examples/s]47557 examples [00:44, 1061.37 examples/s]47664 examples [00:44, 1052.27 examples/s]47771 examples [00:44, 1054.71 examples/s]47877 examples [00:44, 1053.68 examples/s]47983 examples [00:44, 1047.52 examples/s]48088 examples [00:44, 1017.60 examples/s]48198 examples [00:44, 1038.83 examples/s]48307 examples [00:44, 1052.02 examples/s]48418 examples [00:44, 1067.05 examples/s]48529 examples [00:45, 1078.63 examples/s]48640 examples [00:45, 1087.16 examples/s]48751 examples [00:45, 1092.51 examples/s]48861 examples [00:45, 1092.85 examples/s]48971 examples [00:45, 1085.43 examples/s]49082 examples [00:45, 1092.30 examples/s]49192 examples [00:45, 1080.50 examples/s]49301 examples [00:45, 1074.41 examples/s]49413 examples [00:45, 1087.35 examples/s]49522 examples [00:45, 1082.00 examples/s]49633 examples [00:46, 1087.56 examples/s]49744 examples [00:46, 1091.48 examples/s]49854 examples [00:46, 1087.60 examples/s]49963 examples [00:46, 1071.04 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7057/50000 [00:00<00:00, 70568.22 examples/s] 41%|████      | 20465/50000 [00:00<00:00, 82256.55 examples/s] 67%|██████▋   | 33507/50000 [00:00<00:00, 92504.27 examples/s] 94%|█████████▍| 46961/50000 [00:00<00:00, 102071.18 examples/s]                                                                0 examples [00:00, ? examples/s]87 examples [00:00, 868.07 examples/s]198 examples [00:00, 928.39 examples/s]311 examples [00:00, 980.27 examples/s]427 examples [00:00, 1026.42 examples/s]541 examples [00:00, 1057.71 examples/s]650 examples [00:00, 1065.62 examples/s]764 examples [00:00, 1084.63 examples/s]877 examples [00:00, 1096.95 examples/s]990 examples [00:00, 1106.19 examples/s]1104 examples [00:01, 1113.55 examples/s]1214 examples [00:01, 1101.78 examples/s]1325 examples [00:01, 1101.85 examples/s]1435 examples [00:01, 1089.00 examples/s]1548 examples [00:01, 1099.66 examples/s]1658 examples [00:01, 1098.19 examples/s]1769 examples [00:01, 1099.71 examples/s]1882 examples [00:01, 1107.32 examples/s]1993 examples [00:01, 1068.54 examples/s]2101 examples [00:01, 1059.30 examples/s]2214 examples [00:02, 1076.97 examples/s]2324 examples [00:02, 1082.60 examples/s]2433 examples [00:02, 971.19 examples/s] 2534 examples [00:02, 981.31 examples/s]2634 examples [00:02, 969.68 examples/s]2733 examples [00:02, 963.21 examples/s]2831 examples [00:02, 947.20 examples/s]2929 examples [00:02, 953.51 examples/s]3025 examples [00:02, 941.70 examples/s]3120 examples [00:02, 943.42 examples/s]3227 examples [00:03, 976.81 examples/s]3332 examples [00:03, 996.97 examples/s]3443 examples [00:03, 1028.07 examples/s]3555 examples [00:03, 1052.57 examples/s]3668 examples [00:03, 1073.14 examples/s]3776 examples [00:03, 1073.99 examples/s]3884 examples [00:03, 1063.05 examples/s]3996 examples [00:03, 1077.45 examples/s]4108 examples [00:03, 1089.72 examples/s]4221 examples [00:04, 1101.48 examples/s]4332 examples [00:04, 1103.30 examples/s]4445 examples [00:04, 1108.31 examples/s]4556 examples [00:04, 1071.92 examples/s]4669 examples [00:04, 1087.10 examples/s]4783 examples [00:04, 1100.92 examples/s]4894 examples [00:04, 1080.38 examples/s]5006 examples [00:04, 1091.89 examples/s]5120 examples [00:04, 1103.91 examples/s]5233 examples [00:04, 1108.81 examples/s]5347 examples [00:05, 1115.83 examples/s]5459 examples [00:05, 1110.78 examples/s]5574 examples [00:05, 1119.85 examples/s]5687 examples [00:05, 1110.83 examples/s]5801 examples [00:05, 1117.14 examples/s]5917 examples [00:05, 1126.92 examples/s]6030 examples [00:05, 1122.40 examples/s]6144 examples [00:05, 1125.98 examples/s]6257 examples [00:05, 1114.97 examples/s]6369 examples [00:05, 1112.97 examples/s]6484 examples [00:06, 1121.49 examples/s]6597 examples [00:06, 1121.39 examples/s]6710 examples [00:06, 1117.52 examples/s]6822 examples [00:06, 1039.43 examples/s]6931 examples [00:06, 1053.43 examples/s]7041 examples [00:06, 1066.92 examples/s]7152 examples [00:06, 1077.84 examples/s]7263 examples [00:06, 1084.22 examples/s]7372 examples [00:06, 1045.56 examples/s]7486 examples [00:06, 1070.74 examples/s]7598 examples [00:07, 1082.60 examples/s]7710 examples [00:07, 1093.16 examples/s]7824 examples [00:07, 1106.46 examples/s]7936 examples [00:07, 1109.48 examples/s]8048 examples [00:07, 1103.28 examples/s]8159 examples [00:07, 1104.66 examples/s]8270 examples [00:07, 1103.26 examples/s]8381 examples [00:07, 1081.04 examples/s]8490 examples [00:07, 1058.72 examples/s]8602 examples [00:08, 1074.97 examples/s]8710 examples [00:08, 1074.65 examples/s]8821 examples [00:08, 1082.94 examples/s]8934 examples [00:08, 1095.19 examples/s]9044 examples [00:08, 1081.32 examples/s]9157 examples [00:08, 1093.71 examples/s]9269 examples [00:08, 1100.75 examples/s]9381 examples [00:08, 1105.58 examples/s]9492 examples [00:08, 1101.78 examples/s]9603 examples [00:08, 1091.18 examples/s]9714 examples [00:09, 1095.36 examples/s]9826 examples [00:09, 1102.41 examples/s]9937 examples [00:09, 1100.96 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteJRH23L/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteJRH23L/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff4bee8e0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff4bee8e0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff4bee8e0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff50abdd5c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff449102748>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff4bee8e0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff4bee8e0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff4bee8e0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4bd04fcc0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4bd04fd68>), {}) 

  




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
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range

  




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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libtorch_cpu.so: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 248, in compute
    preprocessor_func = load_function(uri)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.10 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 20.80 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.27 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.27 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.76 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.76 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.20 file/s]2020-08-14 18:18:50.922244: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-14 18:18:50.926460: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-08-14 18:18:50.926613: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5607b0fefe60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-14 18:18:50.926627: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist2', 'train', 'test', 'cifar10', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 156888.77it/s] 74%|███████▍  | 7364608/9912422 [00:00<00:11, 223919.08it/s]9920512it [00:00, 41479480.55it/s]                           
0it [00:00, ?it/s]32768it [00:00, 545719.67it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:12, 133812.11it/s]1654784it [00:00, 10621947.29it/s]                         
0it [00:00, ?it/s]8192it [00:00, 141218.53it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

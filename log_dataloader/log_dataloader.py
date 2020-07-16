
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f1bf7d58158> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f1bf7d58158> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f1c638ff2f0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f1c638ff2f0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f1c81c44e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f1c81c44e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1c10c26730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1c10c26730> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1c10c26730> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 141639.70it/s] 75%|███████▌  | 7479296/9912422 [00:00<00:12, 202177.97it/s]9920512it [00:00, 42074056.60it/s]                           
0it [00:00, ?it/s]32768it [00:00, 651576.35it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1008142.30it/s]1654784it [00:00, 12577272.39it/s]                           
0it [00:00, ?it/s]8192it [00:00, 230859.47it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bee7d7ac8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bee7e8d68>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1c10c26378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1c10c26378> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f1c10c26378> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:31,  1.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:31,  1.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:31,  1.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:30,  1.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:30,  1.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:29,  1.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:29,  1.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:02,  2.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:02,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:02,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:01,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:01,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:01,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:00,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:00,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:59,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:59,  2.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:41,  3.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:41,  3.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:41,  3.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:41,  3.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:41,  3.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:40,  3.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:40,  3.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:40,  3.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:39,  3.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:39,  3.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:27,  4.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:27,  4.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:27,  4.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:27,  4.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:27,  4.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:27,  4.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:26,  4.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:26,  4.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:26,  4.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:26,  4.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:18,  6.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:18,  6.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:18,  6.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:18,  6.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:18,  6.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:18,  6.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:18,  6.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:17,  6.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:17,  6.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:17,  6.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:12,  9.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:12,  9.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:12,  9.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:12,  9.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:12,  9.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:12,  9.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  9.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:12,  9.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:11,  9.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11,  9.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:08, 12.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:08, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 17.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 17.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 22.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 29.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 29.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 29.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 29.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 29.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 29.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 29.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 29.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 29.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 29.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 36.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 43.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 43.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 43.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 43.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 43.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 43.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 43.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 43.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 43.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 43.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 51.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 51.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 57.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 57.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 57.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 57.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 57.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 57.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 57.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 57.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 57.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 57.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 69.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.51s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.10s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 77.36 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.10s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.10s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 39.47 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.11s/ url]
0 examples [00:00, ? examples/s]2020-07-16 00:07:54.621077: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-16 00:07:54.668845: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095110000 Hz
2020-07-16 00:07:54.669145: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d970b92e40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-16 00:07:54.669174: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.24 examples/s]148 examples [00:00,  3.19 examples/s]296 examples [00:00,  4.56 examples/s]432 examples [00:00,  6.50 examples/s]560 examples [00:00,  9.26 examples/s]692 examples [00:00, 13.19 examples/s]827 examples [00:01, 18.77 examples/s]957 examples [00:01, 26.65 examples/s]1082 examples [00:01, 37.72 examples/s]1209 examples [00:01, 53.21 examples/s]1345 examples [00:01, 74.76 examples/s]1487 examples [00:01, 104.44 examples/s]1632 examples [00:01, 144.72 examples/s]1768 examples [00:01, 197.58 examples/s]1904 examples [00:01, 263.84 examples/s]2035 examples [00:01, 345.53 examples/s]2181 examples [00:02, 447.90 examples/s]2326 examples [00:02, 564.70 examples/s]2463 examples [00:02, 679.32 examples/s]2607 examples [00:02, 807.03 examples/s]2752 examples [00:02, 930.70 examples/s]2893 examples [00:02, 1035.65 examples/s]3039 examples [00:02, 1134.10 examples/s]3181 examples [00:02, 1193.53 examples/s]3321 examples [00:02, 1248.71 examples/s]3465 examples [00:02, 1299.41 examples/s]3607 examples [00:03, 1288.06 examples/s]3744 examples [00:03, 1289.59 examples/s]3881 examples [00:03, 1311.52 examples/s]4028 examples [00:03, 1355.16 examples/s]4167 examples [00:03, 1359.13 examples/s]4306 examples [00:03, 1362.00 examples/s]4448 examples [00:03, 1377.61 examples/s]4587 examples [00:03, 1353.84 examples/s]4729 examples [00:03, 1372.54 examples/s]4868 examples [00:04, 1377.04 examples/s]5010 examples [00:04, 1389.10 examples/s]5152 examples [00:04, 1397.71 examples/s]5293 examples [00:04, 1387.68 examples/s]5440 examples [00:04, 1411.20 examples/s]5585 examples [00:04, 1421.49 examples/s]5728 examples [00:04, 1333.96 examples/s]5875 examples [00:04, 1371.71 examples/s]6014 examples [00:04, 1368.45 examples/s]6157 examples [00:04, 1385.55 examples/s]6302 examples [00:05, 1403.58 examples/s]6443 examples [00:05, 1391.82 examples/s]6587 examples [00:05, 1404.62 examples/s]6728 examples [00:05, 1359.14 examples/s]6870 examples [00:05, 1375.72 examples/s]7012 examples [00:05, 1386.55 examples/s]7154 examples [00:05, 1396.28 examples/s]7297 examples [00:05, 1404.34 examples/s]7438 examples [00:05, 1346.41 examples/s]7579 examples [00:05, 1364.60 examples/s]7723 examples [00:06, 1384.34 examples/s]7869 examples [00:06, 1405.79 examples/s]8011 examples [00:06, 1409.35 examples/s]8153 examples [00:06, 1396.34 examples/s]8298 examples [00:06, 1409.47 examples/s]8440 examples [00:06, 1401.79 examples/s]8581 examples [00:06, 1402.32 examples/s]8722 examples [00:06, 1388.32 examples/s]8861 examples [00:06, 1339.72 examples/s]8996 examples [00:07, 1328.35 examples/s]9138 examples [00:07, 1353.29 examples/s]9283 examples [00:07, 1380.31 examples/s]9424 examples [00:07, 1388.87 examples/s]9564 examples [00:07, 1370.16 examples/s]9703 examples [00:07, 1375.81 examples/s]9847 examples [00:07, 1393.10 examples/s]9990 examples [00:07, 1403.60 examples/s]10131 examples [00:07, 1331.99 examples/s]10266 examples [00:07, 1333.17 examples/s]10411 examples [00:08, 1365.84 examples/s]10554 examples [00:08, 1384.03 examples/s]10693 examples [00:08, 1350.45 examples/s]10829 examples [00:08, 1291.09 examples/s]10971 examples [00:08, 1326.48 examples/s]11109 examples [00:08, 1339.68 examples/s]11244 examples [00:08, 1337.59 examples/s]11384 examples [00:08, 1352.84 examples/s]11520 examples [00:08, 1350.80 examples/s]11656 examples [00:08, 1348.45 examples/s]11793 examples [00:09, 1351.95 examples/s]11935 examples [00:09, 1370.11 examples/s]12076 examples [00:09, 1380.41 examples/s]12215 examples [00:09, 1372.70 examples/s]12353 examples [00:09, 1349.98 examples/s]12489 examples [00:09, 1345.41 examples/s]12627 examples [00:09, 1354.19 examples/s]12769 examples [00:09, 1371.22 examples/s]12907 examples [00:09, 1302.42 examples/s]13039 examples [00:10, 1258.82 examples/s]13166 examples [00:10, 1249.78 examples/s]13301 examples [00:10, 1276.63 examples/s]13436 examples [00:10, 1295.61 examples/s]13570 examples [00:10, 1307.03 examples/s]13706 examples [00:10, 1320.05 examples/s]13839 examples [00:10, 1295.65 examples/s]13976 examples [00:10, 1315.91 examples/s]14113 examples [00:10, 1329.20 examples/s]14247 examples [00:10, 1298.40 examples/s]14383 examples [00:11, 1315.31 examples/s]14522 examples [00:11, 1336.74 examples/s]14661 examples [00:11, 1345.36 examples/s]14802 examples [00:11, 1363.16 examples/s]14939 examples [00:11, 1360.16 examples/s]15076 examples [00:11, 1332.22 examples/s]15213 examples [00:11, 1342.34 examples/s]15350 examples [00:11, 1348.61 examples/s]15488 examples [00:11, 1356.62 examples/s]15624 examples [00:11, 1350.21 examples/s]15760 examples [00:12, 1343.35 examples/s]15895 examples [00:12, 1339.28 examples/s]16030 examples [00:12, 1342.34 examples/s]16165 examples [00:12, 1320.58 examples/s]16298 examples [00:12, 1317.81 examples/s]16441 examples [00:12, 1348.70 examples/s]16583 examples [00:12, 1368.99 examples/s]16728 examples [00:12, 1390.36 examples/s]16871 examples [00:12, 1401.12 examples/s]17012 examples [00:12, 1368.13 examples/s]17151 examples [00:13, 1374.46 examples/s]17289 examples [00:13, 1366.45 examples/s]17426 examples [00:13, 1362.82 examples/s]17563 examples [00:13, 1360.34 examples/s]17700 examples [00:13, 1299.80 examples/s]17839 examples [00:13, 1324.52 examples/s]17977 examples [00:13, 1339.92 examples/s]18119 examples [00:13, 1360.95 examples/s]18260 examples [00:13, 1373.70 examples/s]18398 examples [00:13, 1355.03 examples/s]18540 examples [00:14, 1372.90 examples/s]18682 examples [00:14, 1386.39 examples/s]18822 examples [00:14, 1389.32 examples/s]18962 examples [00:14, 1391.28 examples/s]19102 examples [00:14, 1374.36 examples/s]19245 examples [00:14, 1388.22 examples/s]19384 examples [00:14, 1371.18 examples/s]19522 examples [00:14, 1369.26 examples/s]19660 examples [00:14, 1367.07 examples/s]19797 examples [00:15, 1322.42 examples/s]19938 examples [00:15, 1346.21 examples/s]20073 examples [00:15, 1271.20 examples/s]20202 examples [00:15, 1235.16 examples/s]20329 examples [00:15, 1243.70 examples/s]20459 examples [00:15, 1257.63 examples/s]20601 examples [00:15, 1301.57 examples/s]20743 examples [00:15, 1332.59 examples/s]20886 examples [00:15, 1359.47 examples/s]21025 examples [00:15, 1366.93 examples/s]21163 examples [00:16, 1351.83 examples/s]21299 examples [00:16, 1298.66 examples/s]21441 examples [00:16, 1332.37 examples/s]21575 examples [00:16, 1285.88 examples/s]21705 examples [00:16, 1256.67 examples/s]21836 examples [00:16, 1271.64 examples/s]21978 examples [00:16, 1310.96 examples/s]22119 examples [00:16, 1338.86 examples/s]22258 examples [00:16, 1352.18 examples/s]22402 examples [00:16, 1376.71 examples/s]22541 examples [00:17, 1361.51 examples/s]22683 examples [00:17, 1377.18 examples/s]22822 examples [00:17, 1313.21 examples/s]22955 examples [00:17, 1293.39 examples/s]23085 examples [00:17, 1248.51 examples/s]23211 examples [00:17, 1248.95 examples/s]23339 examples [00:17, 1256.80 examples/s]23466 examples [00:17, 1259.54 examples/s]23593 examples [00:17, 1257.44 examples/s]23724 examples [00:18, 1270.85 examples/s]23856 examples [00:18, 1283.06 examples/s]23985 examples [00:18, 1258.00 examples/s]24112 examples [00:18, 1235.85 examples/s]24239 examples [00:18, 1244.56 examples/s]24367 examples [00:18, 1254.89 examples/s]24493 examples [00:18, 1242.58 examples/s]24618 examples [00:18, 1239.26 examples/s]24743 examples [00:18, 1223.66 examples/s]24866 examples [00:18, 1215.05 examples/s]24988 examples [00:19, 1198.22 examples/s]25120 examples [00:19, 1231.55 examples/s]25262 examples [00:19, 1280.15 examples/s]25391 examples [00:19, 1278.40 examples/s]25520 examples [00:19, 1274.70 examples/s]25648 examples [00:19, 1249.40 examples/s]25774 examples [00:19, 1249.60 examples/s]25915 examples [00:19, 1291.94 examples/s]26058 examples [00:19, 1328.61 examples/s]26192 examples [00:19, 1325.19 examples/s]26325 examples [00:20, 1293.77 examples/s]26455 examples [00:20, 1236.06 examples/s]26581 examples [00:20, 1242.22 examples/s]26727 examples [00:20, 1298.52 examples/s]26872 examples [00:20, 1338.21 examples/s]27007 examples [00:20, 1303.91 examples/s]27139 examples [00:20, 1290.61 examples/s]27279 examples [00:20, 1319.01 examples/s]27418 examples [00:20, 1338.80 examples/s]27559 examples [00:21, 1358.46 examples/s]27696 examples [00:21, 1341.39 examples/s]27835 examples [00:21, 1354.09 examples/s]27974 examples [00:21, 1362.88 examples/s]28118 examples [00:21, 1383.03 examples/s]28259 examples [00:21, 1389.06 examples/s]28399 examples [00:21, 1358.21 examples/s]28536 examples [00:21, 1360.98 examples/s]28677 examples [00:21, 1375.25 examples/s]28815 examples [00:21, 1300.90 examples/s]28957 examples [00:22, 1333.84 examples/s]29092 examples [00:22, 1316.12 examples/s]29231 examples [00:22, 1335.98 examples/s]29371 examples [00:22, 1354.39 examples/s]29514 examples [00:22, 1374.80 examples/s]29655 examples [00:22, 1382.11 examples/s]29794 examples [00:22, 1383.61 examples/s]29939 examples [00:22, 1400.79 examples/s]30080 examples [00:22, 1308.67 examples/s]30220 examples [00:22, 1334.27 examples/s]30355 examples [00:23, 1338.92 examples/s]30495 examples [00:23, 1355.07 examples/s]30638 examples [00:23, 1375.86 examples/s]30780 examples [00:23, 1386.39 examples/s]30924 examples [00:23, 1399.42 examples/s]31065 examples [00:23, 1389.31 examples/s]31205 examples [00:23, 1355.04 examples/s]31341 examples [00:23, 1309.63 examples/s]31473 examples [00:23, 1293.35 examples/s]31614 examples [00:24, 1326.06 examples/s]31750 examples [00:24, 1332.97 examples/s]31890 examples [00:24, 1351.55 examples/s]32032 examples [00:24, 1369.37 examples/s]32170 examples [00:24, 1370.34 examples/s]32308 examples [00:24, 1354.46 examples/s]32444 examples [00:24, 1346.85 examples/s]32588 examples [00:24, 1370.85 examples/s]32728 examples [00:24, 1377.22 examples/s]32867 examples [00:24, 1378.60 examples/s]33005 examples [00:25, 1361.37 examples/s]33142 examples [00:25, 1331.72 examples/s]33276 examples [00:25, 1273.61 examples/s]33410 examples [00:25, 1292.10 examples/s]33548 examples [00:25, 1315.64 examples/s]33681 examples [00:25, 1236.10 examples/s]33812 examples [00:25, 1255.45 examples/s]33951 examples [00:25, 1292.01 examples/s]34087 examples [00:25, 1311.49 examples/s]34228 examples [00:25, 1338.92 examples/s]34363 examples [00:26, 1338.93 examples/s]34498 examples [00:26, 1331.68 examples/s]34632 examples [00:26, 1272.50 examples/s]34770 examples [00:26, 1301.24 examples/s]34909 examples [00:26, 1323.77 examples/s]35044 examples [00:26, 1330.07 examples/s]35178 examples [00:26, 1316.04 examples/s]35310 examples [00:26, 1294.56 examples/s]35450 examples [00:26, 1322.62 examples/s]35590 examples [00:27, 1343.46 examples/s]35732 examples [00:27, 1363.93 examples/s]35869 examples [00:27, 1332.55 examples/s]36012 examples [00:27, 1358.65 examples/s]36154 examples [00:27, 1374.21 examples/s]36296 examples [00:27, 1387.19 examples/s]36435 examples [00:27, 1381.56 examples/s]36574 examples [00:27, 1345.89 examples/s]36709 examples [00:27, 1288.70 examples/s]36851 examples [00:27, 1322.94 examples/s]36992 examples [00:28, 1346.45 examples/s]37128 examples [00:28, 1340.97 examples/s]37263 examples [00:28, 1311.38 examples/s]37402 examples [00:28, 1332.96 examples/s]37544 examples [00:28, 1357.15 examples/s]37685 examples [00:28, 1372.45 examples/s]37823 examples [00:28, 1359.00 examples/s]37960 examples [00:28, 1361.97 examples/s]38097 examples [00:28, 1353.16 examples/s]38233 examples [00:28, 1287.29 examples/s]38368 examples [00:29, 1303.87 examples/s]38502 examples [00:29, 1313.04 examples/s]38639 examples [00:29, 1326.44 examples/s]38772 examples [00:29, 1297.66 examples/s]38911 examples [00:29, 1321.53 examples/s]39050 examples [00:29, 1340.46 examples/s]39185 examples [00:29, 1340.10 examples/s]39326 examples [00:29, 1358.79 examples/s]39463 examples [00:29, 1354.66 examples/s]39605 examples [00:30, 1371.49 examples/s]39743 examples [00:30, 1364.86 examples/s]39880 examples [00:30, 1355.44 examples/s]40016 examples [00:30, 1314.46 examples/s]40158 examples [00:30, 1344.43 examples/s]40302 examples [00:30, 1371.46 examples/s]40444 examples [00:30, 1384.90 examples/s]40583 examples [00:30, 1371.05 examples/s]40727 examples [00:30, 1390.52 examples/s]40867 examples [00:30, 1369.07 examples/s]41005 examples [00:31, 1351.20 examples/s]41141 examples [00:31, 1349.74 examples/s]41277 examples [00:31, 1345.45 examples/s]41423 examples [00:31, 1376.42 examples/s]41564 examples [00:31, 1384.85 examples/s]41709 examples [00:31, 1403.06 examples/s]41850 examples [00:31, 1376.04 examples/s]41988 examples [00:31, 1363.12 examples/s]42130 examples [00:31, 1379.17 examples/s]42271 examples [00:31, 1386.76 examples/s]42415 examples [00:32, 1400.90 examples/s]42556 examples [00:32, 1337.58 examples/s]42691 examples [00:32, 1323.78 examples/s]42833 examples [00:32, 1349.33 examples/s]42974 examples [00:32, 1364.85 examples/s]43111 examples [00:32, 1359.52 examples/s]43249 examples [00:32, 1363.73 examples/s]43390 examples [00:32, 1375.20 examples/s]43528 examples [00:32, 1351.69 examples/s]43667 examples [00:32, 1360.41 examples/s]43806 examples [00:33, 1368.33 examples/s]43943 examples [00:33, 1356.27 examples/s]44081 examples [00:33, 1360.71 examples/s]44224 examples [00:33, 1378.34 examples/s]44368 examples [00:33, 1394.39 examples/s]44511 examples [00:33, 1404.21 examples/s]44652 examples [00:33, 1387.74 examples/s]44791 examples [00:33, 1362.92 examples/s]44932 examples [00:33, 1374.09 examples/s]45072 examples [00:33, 1379.70 examples/s]45211 examples [00:34, 1342.78 examples/s]45346 examples [00:34, 1296.63 examples/s]45477 examples [00:34, 1285.87 examples/s]45621 examples [00:34, 1327.70 examples/s]45764 examples [00:34, 1356.79 examples/s]45905 examples [00:34, 1369.68 examples/s]46043 examples [00:34, 1354.18 examples/s]46184 examples [00:34, 1370.21 examples/s]46322 examples [00:34, 1369.32 examples/s]46465 examples [00:35, 1385.01 examples/s]46608 examples [00:35, 1395.25 examples/s]46748 examples [00:35, 1366.93 examples/s]46892 examples [00:35, 1385.48 examples/s]47033 examples [00:35, 1392.69 examples/s]47176 examples [00:35, 1402.97 examples/s]47319 examples [00:35, 1410.13 examples/s]47461 examples [00:35, 1379.74 examples/s]47600 examples [00:35, 1346.95 examples/s]47736 examples [00:35, 1344.62 examples/s]47871 examples [00:36, 1339.02 examples/s]48014 examples [00:36, 1362.86 examples/s]48151 examples [00:36, 1336.20 examples/s]48294 examples [00:36, 1361.82 examples/s]48435 examples [00:36, 1374.84 examples/s]48573 examples [00:36, 1374.76 examples/s]48715 examples [00:36, 1386.44 examples/s]48854 examples [00:36, 1374.59 examples/s]48998 examples [00:36, 1392.84 examples/s]49139 examples [00:36, 1395.70 examples/s]49279 examples [00:37, 1378.70 examples/s]49420 examples [00:37, 1386.77 examples/s]49559 examples [00:37, 1370.76 examples/s]49700 examples [00:37, 1380.14 examples/s]49841 examples [00:37, 1387.33 examples/s]49980 examples [00:37, 1371.75 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 20%|██        | 10247/50000 [00:00<00:00, 102464.23 examples/s] 52%|█████▏    | 26168/50000 [00:00<00:00, 114731.42 examples/s] 86%|████████▌ | 43034/50000 [00:00<00:00, 126903.65 examples/s]                                                                0 examples [00:00, ? examples/s]113 examples [00:00, 1127.58 examples/s]257 examples [00:00, 1204.23 examples/s]401 examples [00:00, 1266.12 examples/s]540 examples [00:00, 1299.51 examples/s]676 examples [00:00, 1316.12 examples/s]814 examples [00:00, 1333.93 examples/s]951 examples [00:00, 1341.97 examples/s]1080 examples [00:00, 1323.15 examples/s]1207 examples [00:00, 1268.69 examples/s]1337 examples [00:01, 1277.78 examples/s]1466 examples [00:01, 1281.40 examples/s]1593 examples [00:01, 1261.88 examples/s]1718 examples [00:01, 1193.72 examples/s]1838 examples [00:01, 1182.10 examples/s]1963 examples [00:01, 1199.73 examples/s]2099 examples [00:01, 1241.75 examples/s]2231 examples [00:01, 1262.48 examples/s]2362 examples [00:01, 1275.82 examples/s]2490 examples [00:01, 1271.99 examples/s]2623 examples [00:02, 1287.90 examples/s]2757 examples [00:02, 1301.89 examples/s]2888 examples [00:02, 1298.18 examples/s]3018 examples [00:02, 1276.15 examples/s]3160 examples [00:02, 1314.15 examples/s]3303 examples [00:02, 1346.42 examples/s]3448 examples [00:02, 1373.35 examples/s]3592 examples [00:02, 1391.66 examples/s]3732 examples [00:02, 1373.08 examples/s]3870 examples [00:02, 1331.83 examples/s]4014 examples [00:03, 1360.93 examples/s]4158 examples [00:03, 1382.44 examples/s]4304 examples [00:03, 1402.52 examples/s]4445 examples [00:03, 1392.53 examples/s]4585 examples [00:03, 1387.88 examples/s]4724 examples [00:03, 1301.58 examples/s]4856 examples [00:03, 1271.31 examples/s]4985 examples [00:03, 1245.42 examples/s]5111 examples [00:03, 1209.15 examples/s]5240 examples [00:04, 1230.20 examples/s]5365 examples [00:04, 1233.15 examples/s]5493 examples [00:04, 1245.91 examples/s]5618 examples [00:04, 1243.28 examples/s]5743 examples [00:04, 1240.65 examples/s]5868 examples [00:04, 1209.62 examples/s]5993 examples [00:04, 1220.17 examples/s]6137 examples [00:04, 1276.93 examples/s]6283 examples [00:04, 1324.54 examples/s]6418 examples [00:04, 1329.61 examples/s]6564 examples [00:05, 1364.02 examples/s]6702 examples [00:05, 1365.88 examples/s]6840 examples [00:05, 1364.52 examples/s]6984 examples [00:05, 1384.66 examples/s]7123 examples [00:05, 1338.53 examples/s]7258 examples [00:05, 1340.96 examples/s]7393 examples [00:05, 1331.92 examples/s]7527 examples [00:05, 1325.87 examples/s]7660 examples [00:05, 1273.28 examples/s]7788 examples [00:05, 1258.03 examples/s]7925 examples [00:06, 1288.31 examples/s]8055 examples [00:06, 1289.12 examples/s]8193 examples [00:06, 1314.30 examples/s]8329 examples [00:06, 1325.34 examples/s]8462 examples [00:06, 1296.16 examples/s]8592 examples [00:06, 1246.27 examples/s]8721 examples [00:06, 1255.63 examples/s]8849 examples [00:06, 1260.63 examples/s]8976 examples [00:06, 1259.29 examples/s]9103 examples [00:07, 1249.07 examples/s]9247 examples [00:07, 1297.82 examples/s]9388 examples [00:07, 1327.30 examples/s]9531 examples [00:07, 1354.61 examples/s]9668 examples [00:07, 1351.99 examples/s]9810 examples [00:07, 1371.11 examples/s]9954 examples [00:07, 1389.12 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRINMPW/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRINMPW/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1c10c26730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1c10c26730> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1c10c26730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bee7d7ac8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1b9a06a208>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1c10c26730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1c10c26730> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1c10c26730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1b9a06a160>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bee8040f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f1b886a3400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f1b886a3400> 

  function with postional parmater data_info <function split_train_valid at 0x7f1b886a3400> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f1b886a3510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f1b886a3510> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f1b886a3510> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=898b5e4cc7f54f762ac567b7dee7c8c78f7bab63d7b254a364e0c9a2281e9ed3
  Stored in directory: /tmp/pip-ephem-wheel-cache-gma8byvg/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:15:30, 13.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:00:50, 18.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:09:47, 26.1kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:25:23, 37.2kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:29:07, 53.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.76M/862M [00:01<3:07:08, 75.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.4M/862M [00:01<2:10:13, 108kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:01<1:30:38, 155kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.8M/862M [00:01<1:03:05, 221kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:01<44:09, 314kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:01<30:47, 448kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.3M/862M [00:01<21:36, 636kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:02<15:05, 904kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:02<10:39, 1.27MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:02<07:29, 1.80MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:02<15:12, 887kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<12:30, 1.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:04<11:04, 1.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<08:19, 1.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:06<08:06, 1.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:06<07:11, 1.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<05:24, 2.47MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<06:36, 2.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<07:33, 1.76MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:09<05:56, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.9M/862M [00:09<04:30, 2.94MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<06:03, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:10<05:34, 2.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:11<04:11, 3.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<06:01, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:12<05:31, 2.38MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.8M/862M [00:12<04:12, 3.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:13<03:05, 4.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<2:38:11, 82.7kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<1:53:22, 115kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:14<1:19:50, 164kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:15<55:56, 233kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<44:33, 292kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:16<32:29, 400kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:16<23:02, 563kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<19:07, 677kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<16:03, 806kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:18<11:54, 1.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:19<08:28, 1.52MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<1:37:46, 132kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:20<1:09:44, 185kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:20<49:03, 262kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<37:13, 344kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<28:38, 447kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:22<20:36, 621kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<16:27, 774kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<14:05, 904kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:24<10:24, 1.22MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<07:30, 1.69MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:16, 1.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:48, 1.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<05:47, 2.18MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:57, 1.81MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:46, 1.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:07, 2.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<04:26, 2.82MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<18:20, 683kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<14:13, 881kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<10:17, 1.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<09:53, 1.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:40, 1.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:27, 1.67MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<05:22, 2.31MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<18:47, 660kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<14:31, 852kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<10:26, 1.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<09:58, 1.23MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<09:41, 1.27MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<07:21, 1.67MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<05:17, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<05:44, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<8:27:05, 24.2kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<5:55:17, 34.5kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<4:07:53, 49.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<3:05:48, 65.6kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<2:12:51, 91.7kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<1:33:28, 130kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<1:05:25, 186kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<50:41, 239kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<36:48, 329kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<26:00, 465kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<20:44, 581kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<15:50, 760kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<11:24, 1.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<10:33, 1.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<10:01, 1.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:34, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<05:27, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:34, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:19, 1.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:26, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:22, 1.86MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:46, 2.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:21, 2.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:37, 2.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:32, 1.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:13, 2.25MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<03:48, 3.08MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<16:51, 694kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<13:04, 894kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<09:27, 1.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<09:08, 1.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:57, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:54, 1.68MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<04:58, 2.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<17:31, 660kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<13:32, 854kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<09:43, 1.19MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<09:18, 1.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<09:03, 1.27MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:58, 1.65MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<05:00, 2.29MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<14:23, 794kB/s] .vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<11:19, 1.01MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<08:11, 1.39MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:12, 1.38MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:15, 1.38MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:23, 1.77MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<04:35, 2.46MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<15:18, 737kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<11:57, 943kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<08:39, 1.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<08:29, 1.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<08:25, 1.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:24, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<04:37, 2.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<08:04, 1.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<06:52, 1.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<05:06, 2.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:58, 1.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:37, 1.67MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:15, 2.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<03:47, 2.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<14:25, 763kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<11:18, 973kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<08:10, 1.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<08:04, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<08:03, 1.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:08, 1.78MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:26, 2.45MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<07:28, 1.45MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:26, 1.69MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<04:45, 2.28MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:39, 1.91MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:20, 1.70MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<05:02, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:20<03:38, 2.95MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<12:45, 841kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<10:07, 1.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<07:21, 1.45MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<07:27, 1.43MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<07:34, 1.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:48, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<04:09, 2.54MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<10:30, 1.01MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<08:32, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<06:13, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<06:38, 1.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:58, 1.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<05:27, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<03:56, 2.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<15:26, 677kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<11:57, 873kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<08:36, 1.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<08:14, 1.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<08:03, 1.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<06:06, 1.69MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<04:25, 2.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<07:00, 1.47MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<06:02, 1.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:27, 2.30MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<05:19, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:59, 1.71MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:44, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<03:25, 2.97MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<13:21, 762kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<10:27, 972kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<07:35, 1.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<07:29, 1.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<07:28, 1.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<05:47, 1.74MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<04:09, 2.42MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<12:28, 804kB/s] .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<09:50, 1.02MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<07:06, 1.41MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:08, 1.40MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:11, 1.39MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:31, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<03:59, 2.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<13:34, 729kB/s] .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<10:34, 935kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<07:39, 1.29MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<07:29, 1.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<07:24, 1.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:43, 1.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<04:06, 2.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<13:20, 731kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<10:24, 937kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<07:32, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:22, 1.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:17, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:33, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<04:00, 2.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<07:15, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<06:09, 1.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:33, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:15, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:47, 1.65MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:35, 2.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:19, 2.85MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<13:49, 685kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<10:43, 883kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<07:44, 1.22MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<07:25, 1.27MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<07:11, 1.31MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:31, 1.70MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:00<03:57, 2.36MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<22:01, 424kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<16:26, 567kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<11:44, 793kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<10:12, 909kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<09:06, 1.02MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<06:50, 1.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<04:53, 1.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<18:34, 495kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<14:01, 656kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<10:01, 915kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<08:58, 1.02MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<08:18, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<06:19, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<04:31, 2.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<12:19, 736kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<09:38, 939kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<06:59, 1.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<06:49, 1.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<06:45, 1.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<05:13, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:45, 2.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<12:08, 735kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<09:28, 941kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<06:49, 1.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:41, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:38, 1.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<05:08, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<03:41, 2.38MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<13:15, 663kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<10:14, 857kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<07:21, 1.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<07:01, 1.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<06:50, 1.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<05:15, 1.65MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:19<03:46, 2.29MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<11:53, 727kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<09:14, 936kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<06:40, 1.29MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<06:35, 1.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<06:30, 1.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:56, 1.73MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<03:36, 2.37MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:05, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:30, 1.89MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<03:21, 2.53MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:10, 2.02MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:48, 2.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<02:52, 2.92MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:56, 2.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:38, 2.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<02:43, 3.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:47, 2.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:30, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:36, 2.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<02:37, 3.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<11:35, 710kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<09:03, 909kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<06:30, 1.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:33<04:39, 1.75MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<18:34, 439kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<14:47, 551kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<10:43, 759kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<07:39, 1.06MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<07:28, 1.08MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<06:06, 1.32MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:28, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:52, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:11, 1.54MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<04:04, 1.96MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<02:57, 2.70MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<11:41, 680kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<09:01, 880kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<06:28, 1.22MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:17, 1.25MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:08, 1.28MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:43, 1.66MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<03:24, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<11:51, 659kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<09:07, 855kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<06:32, 1.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:18, 1.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:03, 1.28MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:38, 1.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:47<03:19, 2.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<26:42, 287kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<19:31, 393kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<13:49, 553kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<10:03, 756kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<5:46:05, 22.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<4:02:17, 31.4kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<2:48:38, 44.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<2:05:35, 60.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<1:29:37, 84.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<1:02:59, 119kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<43:59, 170kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<33:49, 221kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<24:20, 307kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<17:12, 433kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<13:34, 545kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<11:08, 664kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<08:11, 902kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<05:47, 1.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<12:58, 565kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<09:52, 741kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<07:04, 1.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<06:29, 1.12MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<05:18, 1.37MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:53, 1.86MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:22, 1.65MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:38, 1.55MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:38, 1.97MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<02:37, 2.71MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<10:29, 679kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<08:07, 877kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<05:51, 1.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:37, 1.26MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:29, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:09, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<03:00, 2.33MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:39, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:01, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<03:00, 2.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:35, 1.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:02, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:12, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<02:19, 2.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<09:56, 689kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<07:42, 888kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<05:32, 1.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<05:20, 1.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:28, 1.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:17, 2.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:45, 1.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:06, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:14, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:17<02:19, 2.85MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<08:53, 747kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<06:57, 954kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<05:00, 1.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:55, 1.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:53, 1.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:46, 1.74MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<02:42, 2.40MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<09:47, 664kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<07:31, 862kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<05:25, 1.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<05:13, 1.23MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<05:04, 1.26MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:54, 1.64MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<02:47, 2.28MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<08:44, 727kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<06:47, 936kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:52, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<04:49, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<04:45, 1.32MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:40, 1.71MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<02:38, 2.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<09:23, 662kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<07:15, 855kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<05:14, 1.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:59, 1.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:50, 1.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:39, 1.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<02:39, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:48, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:19, 1.82MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:29, 2.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:02, 1.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:27, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:41, 2.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<01:57, 3.04MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<03:28, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:05, 1.92MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:17, 2.58MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:52, 2.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:18, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:35, 2.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:41<01:52, 3.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<04:48, 1.21MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:59, 1.45MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:58, 1.95MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:10, 2.64MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:43, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:31, 1.62MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:41, 2.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:52, 1.97MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:15, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:35, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<01:51, 3.00MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<07:25, 753kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<05:48, 962kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<04:10, 1.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<04:07, 1.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:29, 1.58MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:35, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:59, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:41, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:01, 2.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:34, 2.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<03:00, 1.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:23, 2.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<01:43, 3.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<06:54, 769kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<05:25, 979kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:55, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:52, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:51, 1.36MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:59, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<02:08, 2.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<07:47, 665kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<06:01, 859kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:20, 1.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<04:07, 1.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<04:00, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:05, 1.65MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:12, 2.28MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<07:39, 658kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<05:54, 851kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:15, 1.18MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<04:02, 1.23MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:55, 1.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:00, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<02:09, 2.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<06:44, 726kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<05:15, 930kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<03:46, 1.29MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:40, 1.32MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:38, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:47, 1.73MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:59, 2.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:49, 986kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:54, 1.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<02:51, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:00, 1.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:05, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<02:21, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:42, 2.72MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:18, 1.39MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:50, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<02:04, 2.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:26, 1.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:45, 1.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<02:10, 2.08MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<01:33, 2.87MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<05:53, 760kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<04:37, 970kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<03:19, 1.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:15, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:15, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:28, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:49, 2.41MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:26, 1.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:10, 2.00MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:37, 2.65MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:05, 2.04MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:24, 1.78MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:55, 2.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<01:22, 3.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<05:28, 769kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:17, 980kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<03:06, 1.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<03:03, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<02:34, 1.61MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:53, 2.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:14, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:59, 2.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:29, 2.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:56, 2.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:14, 1.78MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:47, 2.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:34<01:17, 3.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:56, 1.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:28, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:48, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:08, 1.81MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:20, 1.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:51, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:38<01:19, 2.86MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<04:32, 836kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:35, 1.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<02:35, 1.45MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:36, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:38, 1.41MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<02:03, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:42<01:28, 2.49MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<05:21, 681kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<04:08, 881kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<02:58, 1.22MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:51, 1.25MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:47, 1.28MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:07, 1.68MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:46<01:30, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<03:12, 1.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:37, 1.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:55, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:05, 1.65MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:13, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:44, 1.97MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<01:14, 2.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<04:29, 752kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:30, 960kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<02:31, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:28, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:27, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:54, 1.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<01:21, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<04:51, 665kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<03:44, 863kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<02:40, 1.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:34, 1.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:29, 1.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:53, 1.67MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:21, 2.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:06, 1.46MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:49, 1.70MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:20, 2.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:35, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:45, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:22, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:04, 2.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<2:08:08, 23.1kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<1:29:30, 32.9kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<1:01:39, 47.0kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<45:59, 62.9kB/s]  .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<32:49, 88.0kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<23:02, 125kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<15:54, 178kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<14:07, 200kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<10:09, 277kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<07:07, 392kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<05:32, 497kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<04:27, 617kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:14, 843kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<02:15, 1.19MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<07:19, 366kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<05:24, 495kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<03:49, 694kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<03:12, 816kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:49, 927kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:06, 1.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<01:28, 1.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<03:27, 736kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:41, 942kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:56, 1.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:52, 1.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:51, 1.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:25, 1.72MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<01:00, 2.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<03:16, 734kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:33, 940kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<01:49, 1.30MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:46, 1.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:45, 1.33MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:19, 1.75MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:56, 2.42MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:46, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:28, 1.53MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:04, 2.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:14, 1.77MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:21, 1.62MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:03, 2.06MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:45, 2.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<02:58, 714kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:18, 921kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:38, 1.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:35, 1.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:34, 1.31MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:11, 1.72MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:50, 2.38MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:30, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:16, 1.56MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:55, 2.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:03, 1.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:08, 1.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:53, 2.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<00:37, 2.93MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<06:19, 293kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<04:36, 400kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<03:13, 564kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<02:36, 682kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:01, 880kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:26, 1.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:21, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:19, 1.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:01, 1.67MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<00:42, 2.31MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<02:15, 730kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:45, 935kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<01:14, 1.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<01:11, 1.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:11, 1.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:54, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:44<00:38, 2.36MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<02:16, 663kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:44, 857kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:14, 1.18MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<01:09, 1.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:07, 1.26MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:51, 1.64MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:36, 2.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<01:53, 724kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:27, 929kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:02, 1.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:59, 1.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:49, 1.55MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:36, 2.08MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:40, 1.81MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:44, 1.64MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:34, 2.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:24, 2.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<01:01, 1.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:50, 1.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:36, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:38, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:41, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:31, 2.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:22, 2.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:44, 1.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:37, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:27, 2.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:30, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:34, 1.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:26, 2.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:18, 2.87MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<01:16, 688kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:59, 886kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:41, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:38, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:37, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:28, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:19, 2.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<01:07, 660kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:51, 857kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:35, 1.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:32, 1.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:31, 1.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:23, 1.66MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:16, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:22, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:19, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:14, 2.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:16, 1.95MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:18, 1.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:14, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:09, 2.97MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:40, 690kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:30, 888kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:21, 1.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:18, 1.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:15, 1.52MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:10, 2.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:11, 1.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:11, 1.63MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:08, 2.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:05, 2.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 1.93MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:05, 2.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:04, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 2.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.33MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:00, 3.20MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:06, 425kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:04, 570kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:01, 797kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<42:45:16,  2.60it/s]  0%|          | 957/400000 [00:00<29:51:36,  3.71it/s]  0%|          | 1953/400000 [00:00<20:51:11,  5.30it/s]  1%|          | 2910/400000 [00:00<14:33:56,  7.57it/s]  1%|          | 3899/400000 [00:00<10:10:25, 10.81it/s]  1%|          | 4856/400000 [00:00<7:06:28, 15.44it/s]   1%|▏         | 5871/400000 [00:00<4:57:57, 22.05it/s]  2%|▏         | 6885/400000 [00:01<3:28:13, 31.46it/s]  2%|▏         | 7843/400000 [00:01<2:25:36, 44.89it/s]  2%|▏         | 8860/400000 [00:01<1:41:51, 64.00it/s]  2%|▏         | 9819/400000 [00:01<1:11:19, 91.17it/s]  3%|▎         | 10836/400000 [00:01<49:59, 129.75it/s]  3%|▎         | 11856/400000 [00:01<35:05, 184.35it/s]  3%|▎         | 12844/400000 [00:01<24:42, 261.19it/s]  3%|▎         | 13823/400000 [00:01<17:27, 368.75it/s]  4%|▎         | 14790/400000 [00:01<12:23, 518.17it/s]  4%|▍         | 15804/400000 [00:01<08:50, 724.37it/s]  4%|▍         | 16806/400000 [00:02<06:21, 1003.69it/s]  4%|▍         | 17791/400000 [00:02<04:38, 1372.12it/s]  5%|▍         | 18814/400000 [00:02<03:25, 1853.56it/s]  5%|▍         | 19819/400000 [00:02<02:34, 2453.92it/s]  5%|▌         | 20814/400000 [00:02<02:00, 3158.29it/s]  5%|▌         | 21797/400000 [00:02<01:36, 3938.57it/s]  6%|▌         | 22765/400000 [00:02<01:19, 4769.63it/s]  6%|▌         | 23746/400000 [00:02<01:06, 5638.47it/s]  6%|▌         | 24712/400000 [00:02<00:58, 6430.24it/s]  6%|▋         | 25709/400000 [00:03<00:52, 7196.47it/s]  7%|▋         | 26723/400000 [00:03<00:47, 7881.37it/s]  7%|▋         | 27708/400000 [00:03<00:45, 8174.79it/s]  7%|▋         | 28719/400000 [00:03<00:42, 8671.54it/s]  7%|▋         | 29715/400000 [00:03<00:41, 9020.86it/s]  8%|▊         | 30732/400000 [00:03<00:39, 9335.04it/s]  8%|▊         | 31738/400000 [00:03<00:38, 9540.57it/s]  8%|▊         | 32734/400000 [00:03<00:38, 9550.38it/s]  8%|▊         | 33740/400000 [00:03<00:37, 9696.44it/s]  9%|▊         | 34755/400000 [00:03<00:37, 9825.89it/s]  9%|▉         | 35753/400000 [00:04<00:38, 9560.85it/s]  9%|▉         | 36730/400000 [00:04<00:37, 9620.05it/s]  9%|▉         | 37701/400000 [00:04<00:38, 9484.47it/s] 10%|▉         | 38656/400000 [00:04<00:38, 9468.99it/s] 10%|▉         | 39664/400000 [00:04<00:37, 9644.07it/s] 10%|█         | 40665/400000 [00:04<00:36, 9750.50it/s] 10%|█         | 41684/400000 [00:04<00:36, 9876.19it/s] 11%|█         | 42675/400000 [00:04<00:36, 9787.78it/s] 11%|█         | 43680/400000 [00:04<00:36, 9862.91it/s] 11%|█         | 44675/400000 [00:04<00:35, 9888.30it/s] 11%|█▏        | 45687/400000 [00:05<00:35, 9954.86it/s] 12%|█▏        | 46710/400000 [00:05<00:35, 10035.21it/s] 12%|█▏        | 47715/400000 [00:05<00:35, 9903.14it/s]  12%|█▏        | 48736/400000 [00:05<00:35, 9991.49it/s] 12%|█▏        | 49765/400000 [00:05<00:34, 10076.83it/s] 13%|█▎        | 50784/400000 [00:05<00:34, 10109.87it/s] 13%|█▎        | 51796/400000 [00:05<00:34, 10097.25it/s] 13%|█▎        | 52807/400000 [00:05<00:34, 9973.50it/s]  13%|█▎        | 53805/400000 [00:05<00:35, 9771.59it/s] 14%|█▎        | 54784/400000 [00:05<00:35, 9756.65it/s] 14%|█▍        | 55803/400000 [00:06<00:34, 9880.99it/s] 14%|█▍        | 56827/400000 [00:06<00:34, 9984.43it/s] 14%|█▍        | 57827/400000 [00:06<00:34, 9851.69it/s] 15%|█▍        | 58835/400000 [00:06<00:34, 9917.75it/s] 15%|█▍        | 59843/400000 [00:06<00:34, 9963.57it/s] 15%|█▌        | 60847/400000 [00:06<00:33, 9984.80it/s] 15%|█▌        | 61864/400000 [00:06<00:33, 10036.92it/s] 16%|█▌        | 62869/400000 [00:06<00:34, 9763.48it/s]  16%|█▌        | 63885/400000 [00:06<00:34, 9878.49it/s] 16%|█▌        | 64875/400000 [00:06<00:34, 9752.07it/s] 16%|█▋        | 65852/400000 [00:07<00:34, 9729.56it/s] 17%|█▋        | 66876/400000 [00:07<00:33, 9876.31it/s] 17%|█▋        | 67865/400000 [00:07<00:34, 9645.26it/s] 17%|█▋        | 68860/400000 [00:07<00:34, 9732.94it/s] 17%|█▋        | 69858/400000 [00:07<00:33, 9804.66it/s] 18%|█▊        | 70840/400000 [00:07<00:33, 9786.41it/s] 18%|█▊        | 71858/400000 [00:07<00:33, 9899.83it/s] 18%|█▊        | 72849/400000 [00:07<00:33, 9800.56it/s] 18%|█▊        | 73862/400000 [00:07<00:32, 9894.66it/s] 19%|█▊        | 74884/400000 [00:07<00:32, 9987.80it/s] 19%|█▉        | 75911/400000 [00:08<00:32, 10068.92it/s] 19%|█▉        | 76933/400000 [00:08<00:31, 10112.36it/s] 19%|█▉        | 77945/400000 [00:08<00:32, 9955.38it/s]  20%|█▉        | 78967/400000 [00:08<00:31, 10033.18it/s] 20%|█▉        | 79995/400000 [00:08<00:31, 10105.96it/s] 20%|██        | 81015/400000 [00:08<00:31, 10131.81it/s] 21%|██        | 82029/400000 [00:08<00:31, 10129.51it/s] 21%|██        | 83043/400000 [00:08<00:31, 10055.97it/s] 21%|██        | 84052/400000 [00:08<00:31, 10064.20it/s] 21%|██▏       | 85072/400000 [00:09<00:31, 10102.37it/s] 22%|██▏       | 86083/400000 [00:09<00:31, 9974.11it/s]  22%|██▏       | 87081/400000 [00:09<00:31, 9916.86it/s] 22%|██▏       | 88074/400000 [00:09<00:31, 9858.37it/s] 22%|██▏       | 89079/400000 [00:09<00:31, 9913.88it/s] 23%|██▎       | 90089/400000 [00:09<00:31, 9966.23it/s] 23%|██▎       | 91086/400000 [00:09<00:31, 9771.08it/s] 23%|██▎       | 92107/400000 [00:09<00:31, 9898.42it/s] 23%|██▎       | 93098/400000 [00:09<00:30, 9900.64it/s] 24%|██▎       | 94095/400000 [00:09<00:30, 9920.74it/s] 24%|██▍       | 95106/400000 [00:10<00:30, 9976.15it/s] 24%|██▍       | 96113/400000 [00:10<00:30, 10001.38it/s] 24%|██▍       | 97121/400000 [00:10<00:30, 10023.35it/s] 25%|██▍       | 98124/400000 [00:10<00:30, 9949.13it/s]  25%|██▍       | 99120/400000 [00:10<00:30, 9845.88it/s] 25%|██▌       | 100128/400000 [00:10<00:30, 9913.47it/s] 25%|██▌       | 101141/400000 [00:10<00:29, 9975.50it/s] 26%|██▌       | 102139/400000 [00:10<00:30, 9883.66it/s] 26%|██▌       | 103128/400000 [00:10<00:30, 9615.74it/s] 26%|██▌       | 104133/400000 [00:10<00:30, 9740.13it/s] 26%|██▋       | 105143/400000 [00:11<00:29, 9845.13it/s] 27%|██▋       | 106133/400000 [00:11<00:29, 9861.09it/s] 27%|██▋       | 107145/400000 [00:11<00:29, 9936.98it/s] 27%|██▋       | 108140/400000 [00:11<00:29, 9783.01it/s] 27%|██▋       | 109120/400000 [00:11<00:30, 9597.03it/s] 28%|██▊       | 110120/400000 [00:11<00:29, 9713.51it/s] 28%|██▊       | 111139/400000 [00:11<00:29, 9850.21it/s] 28%|██▊       | 112138/400000 [00:11<00:29, 9891.61it/s] 28%|██▊       | 113129/400000 [00:11<00:29, 9862.17it/s] 29%|██▊       | 114125/400000 [00:11<00:28, 9890.02it/s] 29%|██▉       | 115141/400000 [00:12<00:28, 9969.01it/s] 29%|██▉       | 116139/400000 [00:12<00:28, 9843.72it/s] 29%|██▉       | 117125/400000 [00:12<00:28, 9757.52it/s] 30%|██▉       | 118102/400000 [00:12<00:29, 9665.17it/s] 30%|██▉       | 119070/400000 [00:12<00:29, 9513.53it/s] 30%|███       | 120023/400000 [00:12<00:29, 9463.98it/s] 30%|███       | 121037/400000 [00:12<00:28, 9655.03it/s] 31%|███       | 122040/400000 [00:12<00:28, 9762.52it/s] 31%|███       | 123018/400000 [00:12<00:28, 9730.84it/s] 31%|███       | 123993/400000 [00:12<00:28, 9659.68it/s] 31%|███       | 124978/400000 [00:13<00:28, 9714.18it/s] 31%|███▏      | 125967/400000 [00:13<00:28, 9763.24it/s] 32%|███▏      | 126977/400000 [00:13<00:27, 9861.14it/s] 32%|███▏      | 127965/400000 [00:13<00:27, 9864.68it/s] 32%|███▏      | 128952/400000 [00:13<00:27, 9757.89it/s] 32%|███▏      | 129957/400000 [00:13<00:27, 9842.81it/s] 33%|███▎      | 130975/400000 [00:13<00:27, 9937.40it/s] 33%|███▎      | 131975/400000 [00:13<00:26, 9954.79it/s] 33%|███▎      | 132971/400000 [00:13<00:27, 9778.84it/s] 33%|███▎      | 133950/400000 [00:13<00:27, 9688.54it/s] 34%|███▎      | 134943/400000 [00:14<00:27, 9757.76it/s] 34%|███▍      | 135942/400000 [00:14<00:26, 9824.74it/s] 34%|███▍      | 136966/400000 [00:14<00:26, 9940.73it/s] 34%|███▍      | 137961/400000 [00:14<00:26, 9933.50it/s] 35%|███▍      | 138955/400000 [00:14<00:26, 9917.68it/s] 35%|███▍      | 139966/400000 [00:14<00:26, 9973.13it/s] 35%|███▌      | 140981/400000 [00:14<00:25, 10024.59it/s] 35%|███▌      | 141985/400000 [00:14<00:25, 10026.59it/s] 36%|███▌      | 142994/400000 [00:14<00:25, 10044.04it/s] 36%|███▌      | 143999/400000 [00:14<00:25, 9878.18it/s]  36%|███▋      | 145009/400000 [00:15<00:25, 9942.06it/s] 37%|███▋      | 146004/400000 [00:15<00:25, 9937.46it/s] 37%|███▋      | 146999/400000 [00:15<00:26, 9713.47it/s] 37%|███▋      | 147972/400000 [00:15<00:26, 9686.60it/s] 37%|███▋      | 148950/400000 [00:15<00:25, 9712.39it/s] 37%|███▋      | 149934/400000 [00:15<00:25, 9748.00it/s] 38%|███▊      | 150913/400000 [00:15<00:25, 9759.18it/s] 38%|███▊      | 151911/400000 [00:15<00:25, 9821.51it/s] 38%|███▊      | 152917/400000 [00:15<00:24, 9891.31it/s] 38%|███▊      | 153907/400000 [00:16<00:25, 9840.21it/s] 39%|███▊      | 154892/400000 [00:16<00:25, 9654.70it/s] 39%|███▉      | 155894/400000 [00:16<00:25, 9759.74it/s] 39%|███▉      | 156924/400000 [00:16<00:24, 9915.01it/s] 39%|███▉      | 157918/400000 [00:16<00:24, 9914.01it/s] 40%|███▉      | 158911/400000 [00:16<00:24, 9742.34it/s] 40%|███▉      | 159929/400000 [00:16<00:24, 9868.11it/s] 40%|████      | 160953/400000 [00:16<00:23, 9975.78it/s] 40%|████      | 161977/400000 [00:16<00:23, 10050.91it/s] 41%|████      | 162987/400000 [00:16<00:23, 10062.45it/s] 41%|████      | 163994/400000 [00:17<00:23, 9955.62it/s]  41%|████▏     | 165023/400000 [00:17<00:23, 10051.93it/s] 42%|████▏     | 166029/400000 [00:17<00:23, 10001.20it/s] 42%|████▏     | 167030/400000 [00:17<00:23, 9818.87it/s]  42%|████▏     | 168014/400000 [00:17<00:24, 9655.10it/s] 42%|████▏     | 168992/400000 [00:17<00:23, 9690.91it/s] 42%|████▏     | 169999/400000 [00:17<00:23, 9801.47it/s] 43%|████▎     | 171023/400000 [00:17<00:23, 9925.93it/s] 43%|████▎     | 172028/400000 [00:17<00:22, 9962.09it/s] 43%|████▎     | 173027/400000 [00:17<00:22, 9968.91it/s] 44%|████▎     | 174041/400000 [00:18<00:22, 10017.55it/s] 44%|████▍     | 175058/400000 [00:18<00:22, 10060.09it/s] 44%|████▍     | 176085/400000 [00:18<00:22, 10120.47it/s] 44%|████▍     | 177098/400000 [00:18<00:22, 10097.86it/s] 45%|████▍     | 178109/400000 [00:18<00:22, 10057.65it/s] 45%|████▍     | 179115/400000 [00:18<00:21, 10056.34it/s] 45%|████▌     | 180122/400000 [00:18<00:21, 10059.64it/s] 45%|████▌     | 181132/400000 [00:18<00:21, 10070.54it/s] 46%|████▌     | 182140/400000 [00:18<00:21, 9954.24it/s]  46%|████▌     | 183136/400000 [00:18<00:21, 9922.09it/s] 46%|████▌     | 184152/400000 [00:19<00:21, 9992.27it/s] 46%|████▋     | 185152/400000 [00:19<00:21, 9967.71it/s] 47%|████▋     | 186160/400000 [00:19<00:21, 10000.88it/s] 47%|████▋     | 187190/400000 [00:19<00:21, 10087.62it/s] 47%|████▋     | 188200/400000 [00:19<00:21, 10043.97it/s] 47%|████▋     | 189205/400000 [00:19<00:21, 9973.96it/s]  48%|████▊     | 190211/400000 [00:19<00:20, 9998.61it/s] 48%|████▊     | 191229/400000 [00:19<00:20, 10051.91it/s] 48%|████▊     | 192246/400000 [00:19<00:20, 10086.21it/s] 48%|████▊     | 193255/400000 [00:19<00:20, 10013.22it/s] 49%|████▊     | 194265/400000 [00:20<00:20, 10037.41it/s] 49%|████▉     | 195275/400000 [00:20<00:20, 10054.50it/s] 49%|████▉     | 196299/400000 [00:20<00:20, 10106.65it/s] 49%|████▉     | 197313/400000 [00:20<00:20, 10115.80it/s] 50%|████▉     | 198325/400000 [00:20<00:20, 9946.67it/s]  50%|████▉     | 199321/400000 [00:20<00:20, 9923.93it/s] 50%|█████     | 200327/400000 [00:20<00:20, 9963.66it/s] 50%|█████     | 201335/400000 [00:20<00:19, 9996.03it/s] 51%|█████     | 202358/400000 [00:20<00:19, 10063.59it/s] 51%|█████     | 203365/400000 [00:20<00:19, 9951.90it/s]  51%|█████     | 204361/400000 [00:21<00:19, 9952.16it/s] 51%|█████▏    | 205373/400000 [00:21<00:19, 10000.60it/s] 52%|█████▏    | 206386/400000 [00:21<00:19, 10037.88it/s] 52%|█████▏    | 207391/400000 [00:21<00:19, 9965.87it/s]  52%|█████▏    | 208388/400000 [00:21<00:19, 9867.19it/s] 52%|█████▏    | 209376/400000 [00:21<00:19, 9855.82it/s] 53%|█████▎    | 210388/400000 [00:21<00:19, 9931.33it/s] 53%|█████▎    | 211403/400000 [00:21<00:18, 9994.07it/s] 53%|█████▎    | 212417/400000 [00:21<00:18, 10035.48it/s] 53%|█████▎    | 213424/400000 [00:21<00:18, 10044.99it/s] 54%|█████▎    | 214429/400000 [00:22<00:19, 9641.10it/s]  54%|█████▍    | 215439/400000 [00:22<00:18, 9773.42it/s] 54%|█████▍    | 216442/400000 [00:22<00:18, 9847.58it/s] 54%|█████▍    | 217463/400000 [00:22<00:18, 9952.32it/s] 55%|█████▍    | 218461/400000 [00:22<00:18, 9881.95it/s] 55%|█████▍    | 219451/400000 [00:22<00:19, 9463.96it/s] 55%|█████▌    | 220403/400000 [00:22<00:19, 9401.02it/s] 55%|█████▌    | 221414/400000 [00:22<00:18, 9601.92it/s] 56%|█████▌    | 222412/400000 [00:22<00:18, 9710.71it/s] 56%|█████▌    | 223386/400000 [00:23<00:18, 9703.55it/s] 56%|█████▌    | 224406/400000 [00:23<00:17, 9845.96it/s] 56%|█████▋    | 225393/400000 [00:23<00:17, 9828.80it/s] 57%|█████▋    | 226419/400000 [00:23<00:17, 9951.59it/s] 57%|█████▋    | 227418/400000 [00:23<00:17, 9961.80it/s] 57%|█████▋    | 228416/400000 [00:23<00:17, 9933.80it/s] 57%|█████▋    | 229411/400000 [00:23<00:17, 9916.86it/s] 58%|█████▊    | 230404/400000 [00:23<00:17, 9872.02it/s] 58%|█████▊    | 231422/400000 [00:23<00:16, 9960.50it/s] 58%|█████▊    | 232444/400000 [00:23<00:16, 10034.47it/s] 58%|█████▊    | 233448/400000 [00:24<00:16, 9860.73it/s]  59%|█████▊    | 234436/400000 [00:24<00:16, 9813.19it/s] 59%|█████▉    | 235419/400000 [00:24<00:16, 9745.15it/s] 59%|█████▉    | 236406/400000 [00:24<00:16, 9778.90it/s] 59%|█████▉    | 237429/400000 [00:24<00:16, 9907.74it/s] 60%|█████▉    | 238421/400000 [00:24<00:16, 9869.78it/s] 60%|█████▉    | 239424/400000 [00:24<00:16, 9916.65it/s] 60%|██████    | 240432/400000 [00:24<00:16, 9964.36it/s] 60%|██████    | 241446/400000 [00:24<00:15, 10014.34it/s] 61%|██████    | 242473/400000 [00:24<00:15, 10087.44it/s] 61%|██████    | 243483/400000 [00:25<00:16, 9690.62it/s]  61%|██████    | 244456/400000 [00:25<00:16, 9611.78it/s] 61%|██████▏   | 245420/400000 [00:25<00:16, 9287.34it/s] 62%|██████▏   | 246394/400000 [00:25<00:16, 9416.48it/s] 62%|██████▏   | 247419/400000 [00:25<00:15, 9651.03it/s] 62%|██████▏   | 248395/400000 [00:25<00:15, 9682.83it/s] 62%|██████▏   | 249393/400000 [00:25<00:15, 9768.95it/s] 63%|██████▎   | 250409/400000 [00:25<00:15, 9881.04it/s] 63%|██████▎   | 251414/400000 [00:25<00:14, 9929.95it/s] 63%|██████▎   | 252431/400000 [00:25<00:14, 9999.27it/s] 63%|██████▎   | 253432/400000 [00:26<00:14, 9945.13it/s] 64%|██████▎   | 254428/400000 [00:26<00:14, 9903.20it/s] 64%|██████▍   | 255419/400000 [00:26<00:14, 9891.00it/s] 64%|██████▍   | 256424/400000 [00:26<00:14, 9937.66it/s] 64%|██████▍   | 257449/400000 [00:26<00:14, 10029.09it/s] 65%|██████▍   | 258453/400000 [00:26<00:14, 9818.15it/s]  65%|██████▍   | 259469/400000 [00:26<00:14, 9918.14it/s] 65%|██████▌   | 260465/400000 [00:26<00:14, 9929.82it/s] 65%|██████▌   | 261459/400000 [00:26<00:14, 9882.19it/s] 66%|██████▌   | 262448/400000 [00:26<00:14, 9819.49it/s] 66%|██████▌   | 263431/400000 [00:27<00:13, 9802.38it/s] 66%|██████▌   | 264454/400000 [00:27<00:13, 9925.03it/s] 66%|██████▋   | 265448/400000 [00:27<00:13, 9873.32it/s] 67%|██████▋   | 266436/400000 [00:27<00:13, 9821.12it/s] 67%|██████▋   | 267455/400000 [00:27<00:13, 9927.77it/s] 67%|██████▋   | 268449/400000 [00:27<00:13, 9719.32it/s] 67%|██████▋   | 269444/400000 [00:27<00:13, 9785.13it/s] 68%|██████▊   | 270424/400000 [00:27<00:13, 9688.71it/s] 68%|██████▊   | 271412/400000 [00:27<00:13, 9738.12it/s] 68%|██████▊   | 272422/400000 [00:27<00:12, 9842.65it/s] 68%|██████▊   | 273408/400000 [00:28<00:12, 9800.75it/s] 69%|██████▊   | 274389/400000 [00:28<00:13, 9580.44it/s] 69%|██████▉   | 275393/400000 [00:28<00:12, 9713.63it/s] 69%|██████▉   | 276387/400000 [00:28<00:12, 9779.45it/s] 69%|██████▉   | 277374/400000 [00:28<00:12, 9803.26it/s] 70%|██████▉   | 278356/400000 [00:28<00:12, 9622.82it/s] 70%|██████▉   | 279366/400000 [00:28<00:12, 9758.53it/s] 70%|███████   | 280391/400000 [00:28<00:12, 9898.70it/s] 70%|███████   | 281414/400000 [00:28<00:11, 9995.23it/s] 71%|███████   | 282415/400000 [00:29<00:11, 9996.41it/s] 71%|███████   | 283416/400000 [00:29<00:12, 9649.81it/s] 71%|███████   | 284385/400000 [00:29<00:12, 9417.98it/s] 71%|███████▏  | 285340/400000 [00:29<00:12, 9456.12it/s] 72%|███████▏  | 286320/400000 [00:29<00:11, 9555.95it/s] 72%|███████▏  | 287333/400000 [00:29<00:11, 9718.74it/s] 72%|███████▏  | 288307/400000 [00:29<00:11, 9608.40it/s] 72%|███████▏  | 289327/400000 [00:29<00:11, 9777.62it/s] 73%|███████▎  | 290343/400000 [00:29<00:11, 9887.22it/s] 73%|███████▎  | 291334/400000 [00:29<00:11, 9841.16it/s] 73%|███████▎  | 292320/400000 [00:30<00:10, 9819.83it/s] 73%|███████▎  | 293303/400000 [00:30<00:10, 9753.24it/s] 74%|███████▎  | 294280/400000 [00:30<00:11, 9574.61it/s] 74%|███████▍  | 295270/400000 [00:30<00:10, 9668.14it/s] 74%|███████▍  | 296262/400000 [00:30<00:10, 9742.22it/s] 74%|███████▍  | 297290/400000 [00:30<00:10, 9897.04it/s] 75%|███████▍  | 298281/400000 [00:30<00:10, 9778.33it/s] 75%|███████▍  | 299293/400000 [00:30<00:10, 9876.54it/s] 75%|███████▌  | 300282/400000 [00:30<00:10, 9880.04it/s] 75%|███████▌  | 301300/400000 [00:30<00:09, 9966.85it/s] 76%|███████▌  | 302319/400000 [00:31<00:09, 10032.05it/s] 76%|███████▌  | 303323/400000 [00:31<00:09, 9685.42it/s]  76%|███████▌  | 304305/400000 [00:31<00:09, 9723.94it/s] 76%|███████▋  | 305289/400000 [00:31<00:09, 9755.41it/s] 77%|███████▋  | 306267/400000 [00:31<00:09, 9614.04it/s] 77%|███████▋  | 307230/400000 [00:31<00:09, 9427.04it/s] 77%|███████▋  | 308197/400000 [00:31<00:09, 9497.81it/s] 77%|███████▋  | 309214/400000 [00:31<00:09, 9689.45it/s] 78%|███████▊  | 310229/400000 [00:31<00:09, 9820.93it/s] 78%|███████▊  | 311213/400000 [00:31<00:09, 9437.19it/s] 78%|███████▊  | 312218/400000 [00:32<00:09, 9612.53it/s] 78%|███████▊  | 313184/400000 [00:32<00:09, 9329.68it/s] 79%|███████▊  | 314178/400000 [00:32<00:09, 9503.93it/s] 79%|███████▉  | 315187/400000 [00:32<00:08, 9670.20it/s] 79%|███████▉  | 316214/400000 [00:32<00:08, 9842.60it/s] 79%|███████▉  | 317209/400000 [00:32<00:08, 9873.78it/s] 80%|███████▉  | 318199/400000 [00:32<00:08, 9549.30it/s] 80%|███████▉  | 319206/400000 [00:32<00:08, 9699.36it/s] 80%|████████  | 320180/400000 [00:32<00:08, 9632.11it/s] 80%|████████  | 321146/400000 [00:33<00:08, 9496.95it/s] 81%|████████  | 322098/400000 [00:33<00:08, 9460.23it/s] 81%|████████  | 323046/400000 [00:33<00:08, 9396.29it/s] 81%|████████  | 324021/400000 [00:33<00:07, 9498.41it/s] 81%|████████▏ | 325000/400000 [00:33<00:07, 9582.36it/s] 81%|████████▏ | 325960/400000 [00:33<00:07, 9517.41it/s] 82%|████████▏ | 326913/400000 [00:33<00:07, 9330.92it/s] 82%|████████▏ | 327848/400000 [00:33<00:07, 9253.20it/s] 82%|████████▏ | 328845/400000 [00:33<00:07, 9455.73it/s] 82%|████████▏ | 329862/400000 [00:33<00:07, 9657.84it/s] 83%|████████▎ | 330868/400000 [00:34<00:07, 9773.65it/s] 83%|████████▎ | 331880/400000 [00:34<00:06, 9872.76it/s] 83%|████████▎ | 332869/400000 [00:34<00:06, 9779.86it/s] 83%|████████▎ | 333882/400000 [00:34<00:06, 9880.11it/s] 84%|████████▎ | 334892/400000 [00:34<00:06, 9943.04it/s] 84%|████████▍ | 335911/400000 [00:34<00:06, 10015.06it/s] 84%|████████▍ | 336914/400000 [00:34<00:06, 9922.97it/s]  84%|████████▍ | 337908/400000 [00:34<00:06, 9864.77it/s] 85%|████████▍ | 338907/400000 [00:34<00:06, 9900.73it/s] 85%|████████▍ | 339926/400000 [00:34<00:06, 9984.26it/s] 85%|████████▌ | 340929/400000 [00:35<00:05, 9995.07it/s] 85%|████████▌ | 341929/400000 [00:35<00:05, 9681.11it/s] 86%|████████▌ | 342900/400000 [00:35<00:06, 9208.68it/s] 86%|████████▌ | 343910/400000 [00:35<00:05, 9457.35it/s] 86%|████████▌ | 344897/400000 [00:35<00:05, 9576.05it/s] 86%|████████▋ | 345895/400000 [00:35<00:05, 9692.03it/s] 87%|████████▋ | 346926/400000 [00:35<00:05, 9868.51it/s] 87%|████████▋ | 347917/400000 [00:35<00:05, 9766.22it/s] 87%|████████▋ | 348938/400000 [00:35<00:05, 9893.59it/s] 87%|████████▋ | 349947/400000 [00:35<00:05, 9950.75it/s] 88%|████████▊ | 350960/400000 [00:36<00:04, 10001.32it/s] 88%|████████▊ | 351962/400000 [00:36<00:04, 10006.46it/s] 88%|████████▊ | 352964/400000 [00:36<00:04, 9938.29it/s]  88%|████████▊ | 353990/400000 [00:36<00:04, 10031.21it/s] 89%|████████▉ | 355012/400000 [00:36<00:04, 10084.47it/s] 89%|████████▉ | 356021/400000 [00:36<00:04, 10034.50it/s] 89%|████████▉ | 357031/400000 [00:36<00:04, 10053.99it/s] 90%|████████▉ | 358037/400000 [00:36<00:04, 9842.88it/s]  90%|████████▉ | 359042/400000 [00:36<00:04, 9903.49it/s] 90%|█████████ | 360039/400000 [00:36<00:04, 9922.79it/s] 90%|█████████ | 361051/400000 [00:37<00:03, 9978.38it/s] 91%|█████████ | 362050/400000 [00:37<00:03, 9810.51it/s] 91%|█████████ | 363033/400000 [00:37<00:03, 9599.64it/s] 91%|█████████ | 364053/400000 [00:37<00:03, 9771.07it/s] 91%|█████████▏| 365074/400000 [00:37<00:03, 9896.34it/s] 92%|█████████▏| 366094/400000 [00:37<00:03, 9983.97it/s] 92%|█████████▏| 367115/400000 [00:37<00:03, 10048.57it/s] 92%|█████████▏| 368121/400000 [00:37<00:03, 9948.98it/s]  92%|█████████▏| 369126/400000 [00:37<00:03, 9978.33it/s] 93%|█████████▎| 370152/400000 [00:37<00:02, 10059.61it/s] 93%|█████████▎| 371159/400000 [00:38<00:02, 9920.54it/s]  93%|█████████▎| 372152/400000 [00:38<00:03, 9272.35it/s] 93%|█████████▎| 373089/400000 [00:38<00:02, 9256.47it/s] 94%|█████████▎| 374068/400000 [00:38<00:02, 9410.09it/s] 94%|█████████▍| 375084/400000 [00:38<00:02, 9620.79it/s] 94%|█████████▍| 376102/400000 [00:38<00:02, 9779.63it/s] 94%|█████████▍| 377097/400000 [00:38<00:02, 9829.28it/s] 95%|█████████▍| 378083/400000 [00:38<00:02, 9791.72it/s] 95%|█████████▍| 379065/400000 [00:38<00:02, 9740.77it/s] 95%|█████████▌| 380067/400000 [00:39<00:02, 9821.78it/s] 95%|█████████▌| 381064/400000 [00:39<00:01, 9863.97it/s] 96%|█████████▌| 382052/400000 [00:39<00:01, 9863.38it/s] 96%|█████████▌| 383039/400000 [00:39<00:01, 9804.80it/s] 96%|█████████▌| 384032/400000 [00:39<00:01, 9839.22it/s] 96%|█████████▋| 385017/400000 [00:39<00:01, 9786.41it/s] 97%|█████████▋| 386026/400000 [00:39<00:01, 9875.55it/s] 97%|█████████▋| 387044/400000 [00:39<00:01, 9964.80it/s] 97%|█████████▋| 388041/400000 [00:39<00:01, 9873.33it/s] 97%|█████████▋| 389055/400000 [00:39<00:01, 9951.19it/s] 98%|█████████▊| 390051/400000 [00:40<00:01, 9822.50it/s] 98%|█████████▊| 391064/400000 [00:40<00:00, 9910.72it/s] 98%|█████████▊| 392070/400000 [00:40<00:00, 9952.92it/s] 98%|█████████▊| 393066/400000 [00:40<00:00, 9857.89it/s] 99%|█████████▊| 394053/400000 [00:40<00:00, 9783.86it/s] 99%|█████████▉| 395032/400000 [00:40<00:00, 9560.96it/s] 99%|█████████▉| 396036/400000 [00:40<00:00, 9698.12it/s] 99%|█████████▉| 397058/400000 [00:40<00:00, 9848.98it/s]100%|█████████▉| 398045/400000 [00:40<00:00, 9785.79it/s]100%|█████████▉| 399055/400000 [00:40<00:00, 9877.28it/s]100%|█████████▉| 399999/400000 [00:41<00:00, 9743.56it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f1b886c14a8>, <torchtext.data.dataset.TabularDataset object at 0x7f1b886c15f8>, <torchtext.vocab.Vocab object at 0x7f1b886c1518>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.52 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.52 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  5.52 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.52 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.46 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.46 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.36 file/s]2020-07-16 00:16:41.806616: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-16 00:16:41.810883: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095110000 Hz
2020-07-16 00:16:41.811048: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556f47d3f2a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-16 00:16:41.811060: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:17, 127386.15it/s] 63%|██████▎   | 6217728/9912422 [00:00<00:20, 181820.12it/s]9920512it [00:00, 38021653.76it/s]                           
0it [00:00, ?it/s]32768it [00:00, 739746.35it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1000890.92it/s]1654784it [00:00, 11287876.86it/s]                           
0it [00:00, ?it/s]8192it [00:00, 243587.62it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

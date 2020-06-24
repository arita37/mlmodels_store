
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fd63c96cea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fd63c96cea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fd6a7c2c1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fd6a7c2c1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fd6c5f73e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fd6c5f73e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd654f53488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd654f53488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd654f53488> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 10%|▉         | 974848/9912422 [00:00<00:00, 9742380.42it/s] 40%|███▉      | 3940352/9912422 [00:00<00:00, 12151524.51it/s] 93%|█████████▎| 9232384/9912422 [00:00<00:00, 15803959.78it/s]9920512it [00:00, 11481794.17it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 200277.38it/s]           
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 485695.78it/s]1654784it [00:00, 11535939.39it/s]                         
0it [00:00, ?it/s]8192it [00:00, 197012.33it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd63c07c358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd63c0735f8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fd654f530d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fd654f530d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fd654f530d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:18,  2.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:18,  2.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:17,  2.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<00:55,  2.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:55,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:54,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:54,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:53,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:00<00:38,  3.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:38,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:38,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:37,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:37,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:37,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:27,  5.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:27,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:26,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:26,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:26,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:19,  7.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:19,  7.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:19,  7.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:18,  7.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:18,  7.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:18,  7.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:01<00:13,  9.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:13,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:13,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:13,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:13,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:13,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:01<00:10, 13.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:10, 13.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:10, 13.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:09, 13.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:09, 13.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:09, 13.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:07, 16.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:07, 16.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:07, 16.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:07, 16.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:07, 16.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:07, 16.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:07, 16.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:05, 20.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:05, 20.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:05, 20.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:05, 20.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:05, 20.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:05, 20.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:05, 20.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 25.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 25.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 25.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 25.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 25.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 25.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 25.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:03, 29.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:03, 29.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 29.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 29.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 29.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 29.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 29.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 34.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 34.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:02, 34.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:02, 34.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 34.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 34.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 34.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 37.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 37.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 37.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 37.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 37.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 37.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 39.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 39.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 39.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 39.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 39.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 39.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 39.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 42.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 42.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 42.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:01, 42.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:01, 42.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:01, 42.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 42.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 44.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 45.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 45.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 45.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 45.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 45.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 45.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 45.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 47.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 47.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 47.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 47.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 47.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 47.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 47.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 48.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 49.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 49.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 49.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 49.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 49.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 49.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 49.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:01, 49.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 49.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 49.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 49.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 49.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 49.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 50.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 50.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 50.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 50.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 50.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 50.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 50.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 50.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 50.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 50.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 50.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 50.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 50.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 50.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 51.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 51.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 51.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 51.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 51.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 51.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 51.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 51.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 51.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 51.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 51.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 51.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 51.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 51.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 51.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 51.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 51.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 51.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 51.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 51.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 51.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 52.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 52.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 52.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 52.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 52.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 52.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 52.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 52.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 52.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 52.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 52.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 52.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 52.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 52.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 52.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.73s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.73s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.73s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 52.40 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.74s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.73s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 52.40 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.74s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.74s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 28.23 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.74s/ url]
0 examples [00:00, ? examples/s]2020-06-24 00:08:45.127142: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-24 00:08:45.138903: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-06-24 00:08:45.139219: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c9acdfefe0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-24 00:08:45.139238: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
49 examples [00:00, 486.55 examples/s]166 examples [00:00, 589.27 examples/s]283 examples [00:00, 691.61 examples/s]401 examples [00:00, 789.35 examples/s]519 examples [00:00, 874.90 examples/s]635 examples [00:00, 942.88 examples/s]751 examples [00:00, 998.78 examples/s]868 examples [00:00, 1044.39 examples/s]987 examples [00:00, 1082.08 examples/s]1105 examples [00:01, 1105.02 examples/s]1218 examples [00:01, 1087.91 examples/s]1336 examples [00:01, 1113.73 examples/s]1454 examples [00:01, 1130.50 examples/s]1570 examples [00:01, 1138.99 examples/s]1687 examples [00:01, 1146.61 examples/s]1804 examples [00:01, 1151.07 examples/s]1920 examples [00:01, 1151.95 examples/s]2037 examples [00:01, 1154.99 examples/s]2154 examples [00:01, 1157.33 examples/s]2270 examples [00:02, 1150.31 examples/s]2386 examples [00:02, 1140.75 examples/s]2504 examples [00:02, 1149.61 examples/s]2622 examples [00:02, 1156.68 examples/s]2738 examples [00:02, 1148.61 examples/s]2854 examples [00:02, 1150.97 examples/s]2970 examples [00:02, 1145.79 examples/s]3085 examples [00:02, 1146.31 examples/s]3200 examples [00:02, 1147.20 examples/s]3315 examples [00:02, 1143.51 examples/s]3430 examples [00:03, 1138.83 examples/s]3544 examples [00:03, 1115.42 examples/s]3656 examples [00:03, 1112.83 examples/s]3768 examples [00:03, 1097.11 examples/s]3884 examples [00:03, 1114.65 examples/s]3996 examples [00:03, 1098.16 examples/s]4111 examples [00:03, 1111.03 examples/s]4229 examples [00:03, 1130.16 examples/s]4344 examples [00:03, 1135.93 examples/s]4462 examples [00:03, 1147.48 examples/s]4577 examples [00:04, 1130.30 examples/s]4691 examples [00:04, 1081.69 examples/s]4808 examples [00:04, 1106.55 examples/s]4926 examples [00:04, 1125.79 examples/s]5043 examples [00:04, 1135.89 examples/s]5157 examples [00:04, 1136.99 examples/s]5271 examples [00:04, 1101.10 examples/s]5385 examples [00:04, 1111.34 examples/s]5497 examples [00:04, 1106.22 examples/s]5614 examples [00:04, 1124.35 examples/s]5727 examples [00:05, 1124.85 examples/s]5844 examples [00:05, 1137.03 examples/s]5960 examples [00:05, 1142.09 examples/s]6078 examples [00:05, 1151.53 examples/s]6196 examples [00:05, 1158.79 examples/s]6312 examples [00:05, 1158.90 examples/s]6430 examples [00:05, 1162.00 examples/s]6547 examples [00:05, 1152.79 examples/s]6664 examples [00:05, 1157.13 examples/s]6780 examples [00:06, 1149.46 examples/s]6895 examples [00:06, 1149.49 examples/s]7013 examples [00:06, 1155.89 examples/s]7129 examples [00:06, 1153.95 examples/s]7246 examples [00:06, 1157.84 examples/s]7362 examples [00:06, 1157.65 examples/s]7478 examples [00:06, 1119.09 examples/s]7591 examples [00:06, 1096.70 examples/s]7706 examples [00:06, 1111.86 examples/s]7820 examples [00:06, 1119.89 examples/s]7933 examples [00:07, 1084.95 examples/s]8042 examples [00:07, 1027.38 examples/s]8156 examples [00:07, 1058.51 examples/s]8274 examples [00:07, 1091.12 examples/s]8389 examples [00:07, 1106.02 examples/s]8501 examples [00:07, 1105.24 examples/s]8614 examples [00:07, 1110.03 examples/s]8729 examples [00:07, 1120.89 examples/s]8847 examples [00:07, 1135.61 examples/s]8964 examples [00:07, 1143.77 examples/s]9079 examples [00:08, 1121.95 examples/s]9193 examples [00:08, 1124.98 examples/s]9306 examples [00:08, 1112.76 examples/s]9420 examples [00:08, 1118.79 examples/s]9532 examples [00:08, 1101.54 examples/s]9643 examples [00:08, 1094.63 examples/s]9757 examples [00:08, 1106.40 examples/s]9872 examples [00:08, 1118.35 examples/s]9988 examples [00:08, 1128.04 examples/s]10101 examples [00:09, 1055.82 examples/s]10216 examples [00:09, 1080.38 examples/s]10330 examples [00:09, 1095.72 examples/s]10441 examples [00:09, 1091.01 examples/s]10551 examples [00:09, 1087.73 examples/s]10661 examples [00:09, 1068.50 examples/s]10776 examples [00:09, 1090.50 examples/s]10891 examples [00:09, 1106.64 examples/s]11003 examples [00:09, 1110.55 examples/s]11116 examples [00:09, 1116.26 examples/s]11228 examples [00:10, 1111.62 examples/s]11340 examples [00:10, 1106.36 examples/s]11451 examples [00:10, 1070.63 examples/s]11567 examples [00:10, 1094.23 examples/s]11678 examples [00:10, 1098.88 examples/s]11794 examples [00:10, 1115.71 examples/s]11909 examples [00:10, 1124.23 examples/s]12022 examples [00:10, 1090.55 examples/s]12132 examples [00:10, 1083.17 examples/s]12244 examples [00:10, 1091.47 examples/s]12360 examples [00:11, 1109.68 examples/s]12474 examples [00:11, 1118.43 examples/s]12587 examples [00:11, 1119.54 examples/s]12705 examples [00:11, 1135.95 examples/s]12819 examples [00:11, 1129.94 examples/s]12936 examples [00:11, 1141.19 examples/s]13051 examples [00:11, 1143.61 examples/s]13166 examples [00:11, 1137.39 examples/s]13280 examples [00:11, 1135.19 examples/s]13394 examples [00:11, 1110.81 examples/s]13508 examples [00:12, 1117.12 examples/s]13623 examples [00:12, 1124.63 examples/s]13736 examples [00:12, 1123.56 examples/s]13849 examples [00:12, 1078.79 examples/s]13964 examples [00:12, 1097.24 examples/s]14079 examples [00:12, 1111.75 examples/s]14195 examples [00:12, 1123.51 examples/s]14308 examples [00:12, 1121.77 examples/s]14423 examples [00:12, 1129.92 examples/s]14537 examples [00:12, 1127.98 examples/s]14654 examples [00:13, 1137.57 examples/s]14770 examples [00:13, 1143.67 examples/s]14885 examples [00:13, 1106.79 examples/s]15003 examples [00:13, 1126.78 examples/s]15116 examples [00:13, 1117.75 examples/s]15229 examples [00:13, 1113.31 examples/s]15341 examples [00:13, 1104.15 examples/s]15456 examples [00:13, 1114.96 examples/s]15572 examples [00:13, 1125.45 examples/s]15685 examples [00:14, 1112.90 examples/s]15799 examples [00:14, 1119.96 examples/s]15912 examples [00:14, 1117.10 examples/s]16027 examples [00:14, 1124.95 examples/s]16142 examples [00:14, 1132.05 examples/s]16256 examples [00:14, 1129.88 examples/s]16370 examples [00:14, 1125.56 examples/s]16483 examples [00:14, 1126.28 examples/s]16598 examples [00:14, 1132.62 examples/s]16712 examples [00:14, 1131.41 examples/s]16827 examples [00:15, 1135.83 examples/s]16941 examples [00:15, 1130.54 examples/s]17055 examples [00:15, 1133.16 examples/s]17170 examples [00:15, 1136.30 examples/s]17284 examples [00:15, 1112.07 examples/s]17396 examples [00:15, 1108.17 examples/s]17507 examples [00:15, 1104.50 examples/s]17620 examples [00:15, 1110.37 examples/s]17737 examples [00:15, 1125.02 examples/s]17850 examples [00:15, 1120.40 examples/s]17963 examples [00:16, 1110.70 examples/s]18077 examples [00:16, 1117.19 examples/s]18189 examples [00:16, 1111.03 examples/s]18301 examples [00:16, 1073.03 examples/s]18409 examples [00:16, 1062.09 examples/s]18527 examples [00:16, 1092.91 examples/s]18643 examples [00:16, 1110.58 examples/s]18755 examples [00:16, 1101.11 examples/s]18869 examples [00:16, 1110.89 examples/s]18985 examples [00:16, 1124.77 examples/s]19102 examples [00:17, 1135.98 examples/s]19218 examples [00:17, 1142.49 examples/s]19333 examples [00:17, 1117.66 examples/s]19445 examples [00:17, 1046.42 examples/s]19551 examples [00:17, 1037.07 examples/s]19667 examples [00:17, 1070.18 examples/s]19785 examples [00:17, 1099.29 examples/s]19899 examples [00:17, 1108.77 examples/s]20011 examples [00:17, 1074.99 examples/s]20129 examples [00:18, 1102.11 examples/s]20245 examples [00:18, 1117.14 examples/s]20363 examples [00:18, 1135.04 examples/s]20477 examples [00:18, 1132.00 examples/s]20593 examples [00:18, 1139.64 examples/s]20708 examples [00:18, 1125.22 examples/s]20821 examples [00:18, 1073.54 examples/s]20929 examples [00:18, 996.97 examples/s] 21031 examples [00:18, 937.68 examples/s]21127 examples [00:18, 931.24 examples/s]21224 examples [00:19, 939.75 examples/s]21319 examples [00:19, 938.61 examples/s]21414 examples [00:19, 940.56 examples/s]21509 examples [00:19, 917.00 examples/s]21603 examples [00:19, 922.82 examples/s]21715 examples [00:19, 972.28 examples/s]21824 examples [00:19, 1003.36 examples/s]21936 examples [00:19, 1034.85 examples/s]22041 examples [00:19, 1036.11 examples/s]22149 examples [00:20, 1047.92 examples/s]22264 examples [00:20, 1074.90 examples/s]22375 examples [00:20, 1082.47 examples/s]22484 examples [00:20, 1070.66 examples/s]22593 examples [00:20, 1076.03 examples/s]22704 examples [00:20, 1085.45 examples/s]22821 examples [00:20, 1107.03 examples/s]22934 examples [00:20, 1111.11 examples/s]23046 examples [00:20, 1102.59 examples/s]23157 examples [00:20, 1061.05 examples/s]23270 examples [00:21, 1078.48 examples/s]23387 examples [00:21, 1102.44 examples/s]23498 examples [00:21, 1085.45 examples/s]23610 examples [00:21, 1093.54 examples/s]23724 examples [00:21, 1106.98 examples/s]23841 examples [00:21, 1123.35 examples/s]23956 examples [00:21, 1130.50 examples/s]24070 examples [00:21, 1131.96 examples/s]24184 examples [00:21, 1131.18 examples/s]24299 examples [00:21, 1135.78 examples/s]24416 examples [00:22, 1143.14 examples/s]24532 examples [00:22, 1147.53 examples/s]24647 examples [00:22, 1116.72 examples/s]24759 examples [00:22, 1113.02 examples/s]24871 examples [00:22, 1088.59 examples/s]24985 examples [00:22, 1100.80 examples/s]25096 examples [00:22, 1096.54 examples/s]25206 examples [00:22, 1087.26 examples/s]25320 examples [00:22, 1100.28 examples/s]25436 examples [00:22, 1115.69 examples/s]25550 examples [00:23, 1120.92 examples/s]25663 examples [00:23, 1119.01 examples/s]25777 examples [00:23, 1123.58 examples/s]25890 examples [00:23, 1122.25 examples/s]26005 examples [00:23, 1130.23 examples/s]26119 examples [00:23, 1123.09 examples/s]26233 examples [00:23, 1125.85 examples/s]26347 examples [00:23, 1128.44 examples/s]26460 examples [00:23, 1108.72 examples/s]26571 examples [00:23, 1107.74 examples/s]26684 examples [00:24, 1113.06 examples/s]26799 examples [00:24, 1121.28 examples/s]26912 examples [00:24, 1122.60 examples/s]27025 examples [00:24, 1100.44 examples/s]27138 examples [00:24, 1107.53 examples/s]27254 examples [00:24, 1121.84 examples/s]27370 examples [00:24, 1132.40 examples/s]27484 examples [00:24, 1116.18 examples/s]27596 examples [00:24, 1094.77 examples/s]27707 examples [00:25, 1097.02 examples/s]27817 examples [00:25, 1096.87 examples/s]27929 examples [00:25, 1102.90 examples/s]28042 examples [00:25, 1110.10 examples/s]28154 examples [00:25, 1110.40 examples/s]28269 examples [00:25, 1119.46 examples/s]28381 examples [00:25, 1075.47 examples/s]28495 examples [00:25, 1092.21 examples/s]28612 examples [00:25, 1114.27 examples/s]28724 examples [00:25, 1109.98 examples/s]28841 examples [00:26, 1124.95 examples/s]28954 examples [00:26, 1126.19 examples/s]29067 examples [00:26, 1116.56 examples/s]29183 examples [00:26, 1126.25 examples/s]29299 examples [00:26, 1133.48 examples/s]29415 examples [00:26, 1140.06 examples/s]29531 examples [00:26, 1145.29 examples/s]29646 examples [00:26, 1145.41 examples/s]29761 examples [00:26, 1121.47 examples/s]29874 examples [00:26, 1101.00 examples/s]29987 examples [00:27, 1107.17 examples/s]30098 examples [00:27, 1051.04 examples/s]30212 examples [00:27, 1074.62 examples/s]30324 examples [00:27, 1085.93 examples/s]30434 examples [00:27, 1088.91 examples/s]30546 examples [00:27, 1097.54 examples/s]30660 examples [00:27, 1107.74 examples/s]30772 examples [00:27, 1111.35 examples/s]30884 examples [00:27, 1105.08 examples/s]30995 examples [00:27, 1062.38 examples/s]31109 examples [00:28, 1082.55 examples/s]31227 examples [00:28, 1108.89 examples/s]31345 examples [00:28, 1126.73 examples/s]31461 examples [00:28, 1134.71 examples/s]31575 examples [00:28, 1134.35 examples/s]31691 examples [00:28, 1140.15 examples/s]31806 examples [00:28, 1102.82 examples/s]31923 examples [00:28, 1119.93 examples/s]32039 examples [00:28, 1129.99 examples/s]32153 examples [00:29, 1129.54 examples/s]32268 examples [00:29, 1134.02 examples/s]32383 examples [00:29, 1138.41 examples/s]32497 examples [00:29, 1138.35 examples/s]32611 examples [00:29, 1135.40 examples/s]32725 examples [00:29, 1102.51 examples/s]32841 examples [00:29, 1118.89 examples/s]32955 examples [00:29, 1124.74 examples/s]33072 examples [00:29, 1136.47 examples/s]33186 examples [00:29, 1128.52 examples/s]33300 examples [00:30, 1129.96 examples/s]33414 examples [00:30, 1124.64 examples/s]33532 examples [00:30, 1139.77 examples/s]33649 examples [00:30, 1147.50 examples/s]33764 examples [00:30, 1147.34 examples/s]33879 examples [00:30, 1139.80 examples/s]33996 examples [00:30, 1146.45 examples/s]34111 examples [00:30, 1144.39 examples/s]34229 examples [00:30, 1153.78 examples/s]34347 examples [00:30, 1159.94 examples/s]34464 examples [00:31, 1149.15 examples/s]34580 examples [00:31, 1150.47 examples/s]34698 examples [00:31, 1157.93 examples/s]34814 examples [00:31, 1140.88 examples/s]34929 examples [00:31, 1140.20 examples/s]35044 examples [00:31, 1140.93 examples/s]35161 examples [00:31, 1148.71 examples/s]35276 examples [00:31, 1100.71 examples/s]35387 examples [00:31, 1079.12 examples/s]35500 examples [00:31, 1091.16 examples/s]35613 examples [00:32, 1099.83 examples/s]35724 examples [00:32, 1064.01 examples/s]35840 examples [00:32, 1090.74 examples/s]35956 examples [00:32, 1110.21 examples/s]36068 examples [00:32, 1109.31 examples/s]36180 examples [00:32, 1103.84 examples/s]36291 examples [00:32, 1084.19 examples/s]36400 examples [00:32, 1072.80 examples/s]36508 examples [00:32, 1068.18 examples/s]36618 examples [00:32, 1076.43 examples/s]36726 examples [00:33, 1067.52 examples/s]36833 examples [00:33, 1065.13 examples/s]36940 examples [00:33, 1054.60 examples/s]37046 examples [00:33, 1045.02 examples/s]37157 examples [00:33, 1062.36 examples/s]37265 examples [00:33, 1065.48 examples/s]37372 examples [00:33, 1063.55 examples/s]37483 examples [00:33, 1076.40 examples/s]37594 examples [00:33, 1086.21 examples/s]37705 examples [00:34, 1091.20 examples/s]37821 examples [00:34, 1108.60 examples/s]37934 examples [00:34, 1113.79 examples/s]38046 examples [00:34, 1100.37 examples/s]38158 examples [00:34, 1104.75 examples/s]38271 examples [00:34, 1111.82 examples/s]38386 examples [00:34, 1121.27 examples/s]38499 examples [00:34, 1123.20 examples/s]38612 examples [00:34, 1096.89 examples/s]38726 examples [00:34, 1106.66 examples/s]38841 examples [00:35, 1117.87 examples/s]38956 examples [00:35, 1125.72 examples/s]39073 examples [00:35, 1136.01 examples/s]39187 examples [00:35, 1128.99 examples/s]39302 examples [00:35, 1135.03 examples/s]39416 examples [00:35, 1133.26 examples/s]39533 examples [00:35, 1142.44 examples/s]39648 examples [00:35, 1144.03 examples/s]39764 examples [00:35, 1147.08 examples/s]39880 examples [00:35, 1149.45 examples/s]39995 examples [00:36, 1147.47 examples/s]40110 examples [00:36, 1095.09 examples/s]40226 examples [00:36, 1111.90 examples/s]40341 examples [00:36, 1121.79 examples/s]40455 examples [00:36, 1126.87 examples/s]40568 examples [00:36, 1122.46 examples/s]40684 examples [00:36, 1132.07 examples/s]40799 examples [00:36, 1134.83 examples/s]40913 examples [00:36, 1127.92 examples/s]41028 examples [00:36, 1132.89 examples/s]41143 examples [00:37, 1135.82 examples/s]41259 examples [00:37, 1141.63 examples/s]41376 examples [00:37, 1148.91 examples/s]41492 examples [00:37, 1149.90 examples/s]41608 examples [00:37, 1150.48 examples/s]41724 examples [00:37, 1141.79 examples/s]41841 examples [00:37, 1147.56 examples/s]41956 examples [00:37, 1145.22 examples/s]42071 examples [00:37, 1103.57 examples/s]42187 examples [00:37, 1119.80 examples/s]42300 examples [00:38, 1116.77 examples/s]42416 examples [00:38, 1127.18 examples/s]42533 examples [00:38, 1137.85 examples/s]42647 examples [00:38, 1133.03 examples/s]42762 examples [00:38, 1136.30 examples/s]42876 examples [00:38, 1133.12 examples/s]42990 examples [00:38, 1122.72 examples/s]43105 examples [00:38, 1128.66 examples/s]43221 examples [00:38, 1136.11 examples/s]43335 examples [00:38, 1134.59 examples/s]43449 examples [00:39, 1130.26 examples/s]43565 examples [00:39, 1137.67 examples/s]43680 examples [00:39, 1140.46 examples/s]43796 examples [00:39, 1145.71 examples/s]43911 examples [00:39, 1133.07 examples/s]44025 examples [00:39, 1117.36 examples/s]44137 examples [00:39, 1117.40 examples/s]44251 examples [00:39, 1122.42 examples/s]44365 examples [00:39, 1126.41 examples/s]44478 examples [00:40, 1123.95 examples/s]44594 examples [00:40, 1132.71 examples/s]44712 examples [00:40, 1143.48 examples/s]44829 examples [00:40, 1148.55 examples/s]44944 examples [00:40, 1125.34 examples/s]45059 examples [00:40, 1130.93 examples/s]45173 examples [00:40, 1127.43 examples/s]45289 examples [00:40, 1134.40 examples/s]45403 examples [00:40, 1130.59 examples/s]45517 examples [00:40, 1098.41 examples/s]45631 examples [00:41, 1109.19 examples/s]45745 examples [00:41, 1116.28 examples/s]45861 examples [00:41, 1127.16 examples/s]45974 examples [00:41, 1124.15 examples/s]46089 examples [00:41, 1130.35 examples/s]46203 examples [00:41, 1130.76 examples/s]46317 examples [00:41, 1123.04 examples/s]46433 examples [00:41, 1133.48 examples/s]46547 examples [00:41, 1098.96 examples/s]46662 examples [00:41, 1111.85 examples/s]46778 examples [00:42, 1123.52 examples/s]46891 examples [00:42, 1122.69 examples/s]47004 examples [00:42, 1118.04 examples/s]47116 examples [00:42, 1115.29 examples/s]47228 examples [00:42, 1109.99 examples/s]47340 examples [00:42, 1112.71 examples/s]47452 examples [00:42, 1086.77 examples/s]47566 examples [00:42, 1099.85 examples/s]47682 examples [00:42, 1115.36 examples/s]47796 examples [00:42, 1119.78 examples/s]47913 examples [00:43, 1132.96 examples/s]48028 examples [00:43, 1136.17 examples/s]48144 examples [00:43, 1141.89 examples/s]48259 examples [00:43, 1132.63 examples/s]48374 examples [00:43, 1135.49 examples/s]48490 examples [00:43, 1139.60 examples/s]48604 examples [00:43, 1133.55 examples/s]48718 examples [00:43, 1080.65 examples/s]48835 examples [00:43, 1104.27 examples/s]48946 examples [00:43, 1084.48 examples/s]49063 examples [00:44, 1107.93 examples/s]49176 examples [00:44, 1112.26 examples/s]49292 examples [00:44, 1126.16 examples/s]49409 examples [00:44, 1136.41 examples/s]49525 examples [00:44, 1142.59 examples/s]49642 examples [00:44, 1147.89 examples/s]49757 examples [00:44, 1135.51 examples/s]49871 examples [00:44, 1132.86 examples/s]49987 examples [00:44, 1138.65 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 8041/50000 [00:00<00:00, 80409.50 examples/s] 44%|████▍     | 22043/50000 [00:00<00:00, 92181.65 examples/s] 72%|███████▏  | 35959/50000 [00:00<00:00, 102568.36 examples/s]                                                                0 examples [00:00, ? examples/s]92 examples [00:00, 913.60 examples/s]208 examples [00:00, 974.70 examples/s]326 examples [00:00, 1026.42 examples/s]440 examples [00:00, 1056.85 examples/s]555 examples [00:00, 1081.48 examples/s]672 examples [00:00, 1106.36 examples/s]787 examples [00:00, 1117.60 examples/s]905 examples [00:00, 1134.38 examples/s]1018 examples [00:00, 1131.99 examples/s]1136 examples [00:01, 1144.11 examples/s]1252 examples [00:01, 1148.51 examples/s]1369 examples [00:01, 1154.37 examples/s]1488 examples [00:01, 1164.71 examples/s]1604 examples [00:01, 1156.99 examples/s]1723 examples [00:01, 1165.00 examples/s]1840 examples [00:01, 1162.10 examples/s]1956 examples [00:01, 1092.70 examples/s]2066 examples [00:01, 1075.54 examples/s]2182 examples [00:01, 1097.54 examples/s]2299 examples [00:02, 1118.17 examples/s]2417 examples [00:02, 1134.55 examples/s]2531 examples [00:02, 1093.67 examples/s]2647 examples [00:02, 1110.98 examples/s]2762 examples [00:02, 1121.23 examples/s]2880 examples [00:02, 1137.16 examples/s]2996 examples [00:02, 1141.33 examples/s]3111 examples [00:02, 1143.32 examples/s]3227 examples [00:02, 1145.67 examples/s]3342 examples [00:02, 1138.90 examples/s]3459 examples [00:03, 1147.51 examples/s]3574 examples [00:03, 1142.25 examples/s]3689 examples [00:03, 1139.52 examples/s]3806 examples [00:03, 1145.24 examples/s]3921 examples [00:03, 1138.68 examples/s]4039 examples [00:03, 1148.86 examples/s]4154 examples [00:03, 1148.98 examples/s]4269 examples [00:03, 1149.10 examples/s]4384 examples [00:03, 1146.62 examples/s]4499 examples [00:03, 1142.35 examples/s]4615 examples [00:04, 1145.57 examples/s]4732 examples [00:04, 1150.19 examples/s]4848 examples [00:04, 1136.85 examples/s]4965 examples [00:04, 1144.69 examples/s]5080 examples [00:04, 1138.98 examples/s]5194 examples [00:04, 1139.18 examples/s]5308 examples [00:04, 1118.74 examples/s]5420 examples [00:04, 1088.67 examples/s]5535 examples [00:04, 1105.14 examples/s]5648 examples [00:04, 1110.11 examples/s]5763 examples [00:05, 1119.45 examples/s]5879 examples [00:05, 1129.64 examples/s]5996 examples [00:05, 1140.96 examples/s]6111 examples [00:05, 1131.25 examples/s]6227 examples [00:05, 1137.46 examples/s]6341 examples [00:05, 1125.63 examples/s]6454 examples [00:05, 1112.77 examples/s]6571 examples [00:05, 1128.13 examples/s]6684 examples [00:05, 1121.07 examples/s]6802 examples [00:06, 1136.91 examples/s]6917 examples [00:06, 1140.25 examples/s]7033 examples [00:06, 1143.93 examples/s]7151 examples [00:06, 1152.37 examples/s]7269 examples [00:06, 1157.43 examples/s]7387 examples [00:06, 1161.82 examples/s]7505 examples [00:06, 1165.86 examples/s]7624 examples [00:06, 1171.82 examples/s]7742 examples [00:06, 1170.03 examples/s]7860 examples [00:06, 1170.39 examples/s]7978 examples [00:07, 1172.31 examples/s]8096 examples [00:07, 1146.56 examples/s]8213 examples [00:07, 1153.09 examples/s]8330 examples [00:07, 1155.68 examples/s]8446 examples [00:07, 1114.13 examples/s]8564 examples [00:07, 1132.81 examples/s]8681 examples [00:07, 1141.65 examples/s]8796 examples [00:07, 1142.64 examples/s]8911 examples [00:07, 1123.20 examples/s]9026 examples [00:07, 1128.41 examples/s]9143 examples [00:08, 1140.29 examples/s]9258 examples [00:08, 1141.61 examples/s]9373 examples [00:08, 1143.95 examples/s]9490 examples [00:08, 1149.37 examples/s]9605 examples [00:08, 1145.55 examples/s]9720 examples [00:08, 1144.52 examples/s]9836 examples [00:08, 1146.52 examples/s]9953 examples [00:08, 1151.97 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteFZBAR2/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteFZBAR2/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd654f53488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd654f53488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd654f53488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd65217d4e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd5ea4886a0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd654f53488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd654f53488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd654f53488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd63c100588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd63c0730f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fd5e8122158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fd5e8122158> 

  function with postional parmater data_info <function split_train_valid at 0x7fd5e8122158> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fd5e8122268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fd5e8122268> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fd5e8122268> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=98653837958d7df21b046a503bd71d1258253d8da478880792177458810c23df
  Stored in directory: /tmp/pip-ephem-wheel-cache-yr9h8mk3/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:06:08, 11.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:18:07, 16.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:03:52, 23.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:03:13, 33.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:55:31, 48.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.40M/862M [00:01<3:25:34, 69.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:01<2:23:02, 98.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:01<1:39:33, 141kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:01<1:09:19, 201kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.8M/862M [00:01<48:17, 287kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.0M/862M [00:01<33:48, 408kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.7M/862M [00:02<23:34, 581kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.6M/862M [00:02<16:35, 822kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.3M/862M [00:02<11:37, 1.17MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:02<08:46, 1.54MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<08:01, 1.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<07:26, 1.81MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:04<05:35, 2.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:05<04:04, 3.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<21:50, 612kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<17:15, 774kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:06<12:33, 1.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<11:12, 1.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<11:37, 1.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:08<09:00, 1.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.0M/862M [00:09<06:30, 2.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<08:18, 1.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<07:45, 1.70MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:10<05:54, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<06:32, 2.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<08:33, 1.54MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:12<06:49, 1.93MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.4M/862M [00:12<04:57, 2.64MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<07:38, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<07:16, 1.80MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:14<05:29, 2.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:14<03:58, 3.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<32:41, 398kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<26:49, 485kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<19:44, 659kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.9M/862M [00:16<13:59, 927kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<14:17, 906kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<11:54, 1.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:18<08:47, 1.47MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<08:29, 1.52MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<09:50, 1.31MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:20<07:50, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.3M/862M [00:20<05:43, 2.24MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<08:26, 1.52MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<07:47, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<05:51, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:22<04:15, 3.00MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<14:29, 879kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<13:46, 925kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<10:27, 1.22MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<07:30, 1.69MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:06, 1.39MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:15, 1.54MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:09, 2.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<04:29, 2.81MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<11:08, 1.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:38, 1.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<07:12, 1.75MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:19, 1.71MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:58, 1.80MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<05:20, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:00, 2.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:02, 2.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:36, 2.70MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<03:21, 3.70MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<30:49, 402kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<29:40, 418kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<21:04, 587kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<16:53, 729kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<13:38, 904kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<09:59, 1.23MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<09:12, 1.33MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:14, 1.49MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:13, 1.97MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:34, 1.85MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:23, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:54, 2.48MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:39, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:44, 2.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:23, 2.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<03:12, 3.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<26:13, 460kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<22:00, 548kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<16:17, 739kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<11:35, 1.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<12:15, 977kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<10:05, 1.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<07:30, 1.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<05:23, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<14:14, 836kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<11:43, 1.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<08:38, 1.38MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:11, 1.44MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:29, 1.58MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:40, 2.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<04:06, 2.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<19:45, 596kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<15:34, 756kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<11:19, 1.04MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<10:02, 1.17MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:46, 1.33MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:34, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<04:42, 2.47MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<3:51:04, 50.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<2:43:27, 71.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<1:54:36, 101kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<1:20:09, 144kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<1:03:53, 181kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<48:13, 240kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<34:27, 335kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<24:15, 475kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<20:12, 569kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<15:50, 726kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<11:29, 999kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<10:06, 1.13MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<08:45, 1.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:28, 1.76MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<04:39, 2.44MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<19:23, 586kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<16:48, 676kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<12:36, 900kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<08:58, 1.26MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<10:13, 1.10MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:48, 1.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:30, 1.73MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:37, 1.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:16, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:47, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:23, 2.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:25, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<04:08, 2.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:03, 3.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<09:56, 1.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<07:03, 1.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<10:07, 1.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:41, 1.27MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<06:25, 1.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<04:36, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<37:34, 291kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<29:36, 370kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<21:31, 508kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<15:13, 716kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<14:24, 755kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<11:41, 930kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<08:30, 1.28MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<06:03, 1.78MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<22:10, 487kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<18:37, 580kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<13:43, 786kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<09:47, 1.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<10:22, 1.03MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:50, 1.21MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:34, 1.63MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:32, 1.63MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<07:48, 1.37MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:08, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:29, 2.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:45, 1.57MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<06:18, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:47, 2.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:16, 2.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:13, 2.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<04:02, 2.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:44, 2.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:29, 1.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:18, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<03:54, 2.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:20, 1.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:57, 1.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<04:29, 2.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:02, 2.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:29, 1.59MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:19, 1.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:34<03:54, 2.63MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:17, 1.63MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:54, 1.74MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<04:30, 2.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:00, 2.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:35, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:22, 1.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<03:56, 2.58MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:15, 1.62MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:51, 1.72MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<04:24, 2.29MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<03:11, 3.15MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<18:27, 545kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<15:58, 629kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<11:48, 850kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<08:23, 1.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<08:50, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:40, 1.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<05:43, 1.74MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:48, 1.71MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:31, 1.79MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<04:09, 2.38MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<03:01, 3.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<14:07, 697kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<12:41, 776kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<09:37, 1.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<06:51, 1.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<08:12, 1.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<07:11, 1.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<05:23, 1.81MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<05:31, 1.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<06:49, 1.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:30, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<04:01, 2.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:09, 1.57MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:43, 1.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:20, 2.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:47, 2.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<04:48, 1.99MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:39, 2.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<02:39, 3.58MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<31:40, 300kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<25:03, 379kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<18:07, 523kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<12:48, 738kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<11:37, 810kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<09:31, 990kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<06:57, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:34, 1.42MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<07:17, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:48, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<04:13, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<06:10, 1.50MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:41, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:18, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:41, 1.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:38, 1.98MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<03:35, 2.57MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:10, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:15, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<03:15, 2.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<02:22, 3.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<46:49, 194kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<35:32, 255kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<25:31, 355kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<17:57, 503kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<15:35, 578kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<12:13, 736kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<08:50, 1.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<07:48, 1.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<06:46, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<05:00, 1.78MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<03:35, 2.47MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<21:27, 414kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<17:42, 501kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<13:02, 680kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<09:15, 953kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<09:31, 925kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<07:57, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:49, 1.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<04:10, 2.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<10:10, 859kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<09:45, 895kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<07:28, 1.17MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<05:22, 1.62MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<06:45, 1.28MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<06:00, 1.44MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:30, 1.91MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<04:43, 1.82MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:54, 1.45MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:40, 1.84MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:23, 2.52MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:58, 1.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:44, 1.80MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:37, 2.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:04, 2.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:24, 1.56MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:19, 1.95MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:08, 2.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<05:02, 1.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:45, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:38, 2.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:03, 2.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:04, 2.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:06, 2.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:15, 3.65MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<16:04, 513kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<13:44, 600kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<10:13, 806kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<07:16, 1.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<07:55, 1.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<06:44, 1.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:58, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:35<03:32, 2.29MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<8:07:09, 16.7kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<5:43:22, 23.6kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<4:00:36, 33.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<2:47:52, 48.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<1:59:43, 67.2kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<1:24:56, 94.6kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<59:37, 135kB/s]   .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<42:57, 186kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<32:28, 245kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<23:18, 342kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<16:21, 484kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<14:07, 559kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<11:02, 715kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<08:00, 985kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<07:00, 1.12MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<07:08, 1.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:29, 1.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:56, 1.97MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:23, 1.44MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:54, 1.58MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:40, 2.11MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:58, 1.94MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:06, 1.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:03, 1.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<02:58, 2.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:42, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:24, 1.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:19, 2.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:42, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:42, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:51, 2.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:22, 2.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:37, 1.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:47, 1.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:46, 2.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:31, 1.64MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:15, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<03:14, 2.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:36, 2.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:36, 2.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<02:47, 2.63MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:16, 2.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:21, 2.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<02:36, 2.78MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:08, 2.30MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:23, 1.65MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:31, 2.05MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<02:34, 2.78MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:46, 1.89MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:41, 1.93MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<02:48, 2.53MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:15, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:19, 2.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<02:32, 2.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:06<01:50, 3.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<17:23, 403kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<13:13, 529kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<09:29, 736kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<07:52, 881kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<07:39, 906kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<05:47, 1.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<04:08, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:06, 1.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:34, 1.50MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:26, 1.99MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:38, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:36, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:39, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:40, 2.53MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:41, 1.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:34, 1.88MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:42, 2.48MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<01:57, 3.41MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<23:29, 284kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<17:24, 382kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<12:21, 537kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<09:48, 672kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<08:52, 743kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<06:41, 983kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<04:47, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<05:36, 1.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:53, 1.33MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:37, 1.79MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<02:37, 2.47MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<05:40, 1.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:45, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<03:29, 1.84MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:43, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:35, 1.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:41, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:40, 2.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:30, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<03:12, 1.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<04:43, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:14, 1.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:11, 1.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<02:17, 2.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<21:45, 284kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<16:08, 382kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<11:28, 537kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<08:03, 759kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<10:57, 557kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<08:34, 712kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<06:12, 981kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<05:25, 1.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:41, 1.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:30, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:31, 1.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:23, 1.76MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:33, 2.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<01:51, 3.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<05:44, 1.03MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<05:26, 1.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:10, 1.41MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:58, 1.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<05:43, 1.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:51, 1.20MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:36, 1.61MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:33, 1.62MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:21, 1.71MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:33, 2.25MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<02:49, 2.02MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:41, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:57, 1.92MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:09, 2.61MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:25, 1.64MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:13, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:27, 2.28MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:43, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:29, 1.59MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:49, 1.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:03, 2.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:16, 1.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:05, 1.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:21, 2.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:37, 2.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:28, 1.56MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:50, 1.91MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:04, 2.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:18, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:05, 1.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:20, 2.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:35, 2.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:24, 1.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:46, 1.90MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:01, 2.59MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<03:13, 1.61MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:01, 1.72MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:18, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:32, 2.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:19, 1.54MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:42, 1.89MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<01:58, 2.57MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<03:08, 1.61MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:56, 1.72MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:14, 2.25MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:28, 2.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:14, 1.54MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:35, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:53, 2.63MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:35, 1.90MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:32, 1.94MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<01:55, 2.56MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<01:23, 3.51MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<19:09, 254kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<14:06, 345kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<09:59, 485kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<07:48, 614kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<06:10, 776kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:27, 1.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<03:09, 1.50MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<05:27, 867kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<05:15, 898kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:59, 1.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:51, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:34, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:11, 1.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:23, 1.94MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:30, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:25, 1.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:51, 2.46MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:07, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:08, 2.10MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:38, 2.74MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:57, 2.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:43, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:10, 2.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:35, 2.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<02:20, 1.88MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:18, 1.90MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:44, 2.50MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:59, 2.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:42, 1.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:09, 1.99MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:35, 2.69MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:33, 1.66MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:24, 1.76MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:50, 2.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:02, 2.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:02, 2.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:32, 2.68MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:49, 2.24MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:31, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:04, 1.98MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<01:30, 2.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:27, 1.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:18, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:45, 2.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:56, 2.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:29, 1.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:02, 1.94MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<01:29, 2.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:23, 1.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:14, 1.74MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:42, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:53, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:52, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:26, 2.63MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:41, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:44, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:19, 2.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:41<00:57, 3.85MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<20:40, 179kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<15:30, 238kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<11:06, 331kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<07:46, 469kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<06:39, 543kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<05:11, 697kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<03:43, 966kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:45<02:37, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<08:54, 399kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<06:45, 526kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<04:49, 731kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:58, 876kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:17, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:25, 1.43MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:17, 1.48MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:06, 1.61MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:35, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:43, 1.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:42, 1.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:17, 2.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<00:56, 3.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:55, 1.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:31, 1.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:51, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<01:19, 2.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<08:53, 360kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<06:42, 478kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<04:46, 667kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<03:20, 942kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<09:11, 341kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<06:49, 459kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<04:52, 640kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<03:24, 904kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<07:55, 387kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<05:59, 511kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<04:17, 711kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:30, 856kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:53, 1.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:07, 1.41MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<01:29, 1.97MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<08:36, 341kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<06:51, 427kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<05:00, 583kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<03:30, 822kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<03:26, 833kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:49, 1.01MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:03, 1.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:55, 1.45MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:11, 1.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:42, 1.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:13, 2.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:46, 1.53MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:39, 1.64MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:14, 2.19MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:11<00:53, 3.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<04:47, 554kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<03:45, 706kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:41, 978kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<01:52, 1.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<11:35, 223kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<08:25, 307kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<05:57, 431kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<04:07, 611kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<16:33, 152kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<11:53, 211kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<08:23, 298kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<05:50, 423kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<04:54, 498kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<04:11, 585kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<03:04, 793kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<02:12, 1.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:00, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:45, 1.35MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:17, 1.81MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:55, 2.50MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<02:05, 1.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:47, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:20, 1.71MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:19, 1.69MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:15, 1.78MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:57, 2.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:03, 2.06MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:59, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<00:44, 2.87MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:56, 2.22MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:17, 1.62MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:03, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:45, 2.69MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:13, 1.67MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:09, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:52, 2.31MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:57, 2.05MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:15, 1.55MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:00, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:43, 2.66MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:03, 1.79MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:01, 1.86MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:46, 2.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:51, 2.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:09, 1.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:56, 1.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:40, 2.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:04, 1.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:00, 1.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:46, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:49, 2.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:03, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:52, 1.93MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:37, 2.65MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:59, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:56, 1.73MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:42, 2.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:45, 2.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:45, 2.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:34, 2.64MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:39, 2.23MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:54, 1.62MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:43, 2.02MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:31, 2.73MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:42, 1.99MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:42, 2.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:31, 2.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:36, 2.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:37, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:28, 2.81MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:50<00:19, 3.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<03:32, 360kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<02:39, 478kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<01:52, 666kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:29, 810kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:24, 857kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<01:03, 1.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:44, 1.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:53, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:47, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:35, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:35, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:34, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:25, 2.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:28, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:37, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:30, 1.92MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:21, 2.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:34, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:32, 1.73MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:23, 2.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:16, 3.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:32, 557kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:12, 712kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:51, 985kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:42, 1.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:36, 1.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:26, 1.75MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:06<00:17, 2.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<07:23, 97.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<05:22, 134kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<03:45, 189kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<02:31, 268kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:55, 339kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:26, 452kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:59, 633kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:10<00:39, 893kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:53, 657kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:41, 845kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:28, 1.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:24, 1.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:24, 1.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:18, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:12, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:18, 1.44MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:16, 1.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:12, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:11, 1.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:14, 1.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:11, 1.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:07, 2.58MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:09, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:09, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:06, 2.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:04, 2.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:02, 3.78MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<05:03, 33.3kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<03:28, 47.3kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<02:10, 67.3kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<01:03, 94.5kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:43, 132kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:24, 188kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:07, 256kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:04, 346kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:01, 486kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 61/400000 [00:00<10:55, 609.90it/s]  0%|          | 966/400000 [00:00<07:51, 846.82it/s]  0%|          | 1851/400000 [00:00<05:42, 1162.08it/s]  1%|          | 2746/400000 [00:00<04:12, 1572.53it/s]  1%|          | 3646/400000 [00:00<03:09, 2089.87it/s]  1%|          | 4550/400000 [00:00<02:25, 2716.25it/s]  1%|▏         | 5455/400000 [00:00<01:54, 3438.05it/s]  2%|▏         | 6349/400000 [00:00<01:33, 4216.30it/s]  2%|▏         | 7267/400000 [00:00<01:18, 5032.50it/s]  2%|▏         | 8166/400000 [00:01<01:07, 5797.89it/s]  2%|▏         | 9055/400000 [00:01<01:00, 6472.11it/s]  2%|▏         | 9974/400000 [00:01<00:54, 7100.83it/s]  3%|▎         | 10891/400000 [00:01<00:51, 7614.95it/s]  3%|▎         | 11787/400000 [00:01<00:48, 7973.36it/s]  3%|▎         | 12683/400000 [00:01<00:47, 8225.12it/s]  3%|▎         | 13577/400000 [00:01<00:47, 8216.59it/s]  4%|▎         | 14498/400000 [00:01<00:45, 8488.92it/s]  4%|▍         | 15384/400000 [00:01<00:44, 8562.29it/s]  4%|▍         | 16306/400000 [00:01<00:43, 8748.76it/s]  4%|▍         | 17201/400000 [00:02<00:43, 8785.51it/s]  5%|▍         | 18094/400000 [00:02<00:43, 8812.37it/s]  5%|▍         | 18985/400000 [00:02<00:43, 8837.66it/s]  5%|▍         | 19905/400000 [00:02<00:42, 8941.93it/s]  5%|▌         | 20805/400000 [00:02<00:42, 8956.61it/s]  5%|▌         | 21705/400000 [00:02<00:42, 8890.28it/s]  6%|▌         | 22608/400000 [00:02<00:42, 8930.88it/s]  6%|▌         | 23503/400000 [00:02<00:43, 8673.37it/s]  6%|▌         | 24426/400000 [00:02<00:42, 8833.03it/s]  6%|▋         | 25353/400000 [00:02<00:41, 8957.11it/s]  7%|▋         | 26260/400000 [00:03<00:41, 8989.01it/s]  7%|▋         | 27178/400000 [00:03<00:41, 9044.88it/s]  7%|▋         | 28084/400000 [00:03<00:41, 9005.53it/s]  7%|▋         | 29009/400000 [00:03<00:40, 9075.34it/s]  7%|▋         | 29919/400000 [00:03<00:40, 9081.96it/s]  8%|▊         | 30828/400000 [00:03<00:40, 9036.05it/s]  8%|▊         | 31733/400000 [00:03<00:40, 9022.15it/s]  8%|▊         | 32636/400000 [00:03<00:40, 8962.69it/s]  8%|▊         | 33562/400000 [00:03<00:40, 9049.35it/s]  9%|▊         | 34484/400000 [00:03<00:40, 9097.33it/s]  9%|▉         | 35395/400000 [00:04<00:40, 8998.25it/s]  9%|▉         | 36302/400000 [00:04<00:40, 9012.01it/s]  9%|▉         | 37223/400000 [00:04<00:39, 9070.21it/s] 10%|▉         | 38131/400000 [00:04<00:40, 9040.63it/s] 10%|▉         | 39036/400000 [00:04<00:39, 9041.32it/s] 10%|▉         | 39941/400000 [00:04<00:40, 8855.74it/s] 10%|█         | 40828/400000 [00:04<00:40, 8852.64it/s] 10%|█         | 41714/400000 [00:04<00:40, 8850.59it/s] 11%|█         | 42616/400000 [00:04<00:40, 8898.61it/s] 11%|█         | 43542/400000 [00:04<00:39, 9004.02it/s] 11%|█         | 44443/400000 [00:05<00:39, 8953.07it/s] 11%|█▏        | 45356/400000 [00:05<00:39, 9002.63it/s] 12%|█▏        | 46284/400000 [00:05<00:38, 9082.96it/s] 12%|█▏        | 47214/400000 [00:05<00:38, 9146.68it/s] 12%|█▏        | 48140/400000 [00:05<00:38, 9178.21it/s] 12%|█▏        | 49059/400000 [00:05<00:38, 9095.54it/s] 12%|█▏        | 49969/400000 [00:05<00:39, 8789.47it/s] 13%|█▎        | 50851/400000 [00:05<00:39, 8767.41it/s] 13%|█▎        | 51782/400000 [00:05<00:39, 8922.11it/s] 13%|█▎        | 52701/400000 [00:05<00:38, 8998.09it/s] 13%|█▎        | 53603/400000 [00:06<00:38, 8917.33it/s] 14%|█▎        | 54519/400000 [00:06<00:38, 8988.49it/s] 14%|█▍        | 55443/400000 [00:06<00:38, 9061.76it/s] 14%|█▍        | 56363/400000 [00:06<00:37, 9093.73it/s] 14%|█▍        | 57273/400000 [00:06<00:37, 9058.27it/s] 15%|█▍        | 58180/400000 [00:06<00:38, 8971.56it/s] 15%|█▍        | 59107/400000 [00:06<00:37, 9058.25it/s] 15%|█▌        | 60029/400000 [00:06<00:37, 9104.44it/s] 15%|█▌        | 60940/400000 [00:06<00:37, 9104.72it/s] 15%|█▌        | 61873/400000 [00:06<00:36, 9170.12it/s] 16%|█▌        | 62791/400000 [00:07<00:37, 9104.65it/s] 16%|█▌        | 63725/400000 [00:07<00:36, 9171.31it/s] 16%|█▌        | 64643/400000 [00:07<00:36, 9144.49it/s] 16%|█▋        | 65558/400000 [00:07<00:37, 8985.49it/s] 17%|█▋        | 66484/400000 [00:07<00:36, 9065.44it/s] 17%|█▋        | 67392/400000 [00:07<00:36, 8998.88it/s] 17%|█▋        | 68318/400000 [00:07<00:36, 9074.63it/s] 17%|█▋        | 69230/400000 [00:07<00:36, 9087.78it/s] 18%|█▊        | 70148/400000 [00:07<00:36, 9114.53it/s] 18%|█▊        | 71060/400000 [00:07<00:36, 9088.96it/s] 18%|█▊        | 71970/400000 [00:08<00:36, 9057.01it/s] 18%|█▊        | 72878/400000 [00:08<00:36, 9063.10it/s] 18%|█▊        | 73798/400000 [00:08<00:35, 9103.12it/s] 19%|█▊        | 74712/400000 [00:08<00:35, 9111.85it/s] 19%|█▉        | 75624/400000 [00:08<00:35, 9104.91it/s] 19%|█▉        | 76535/400000 [00:08<00:35, 9096.96it/s] 19%|█▉        | 77445/400000 [00:08<00:36, 8921.34it/s] 20%|█▉        | 78372/400000 [00:08<00:35, 9022.63it/s] 20%|█▉        | 79287/400000 [00:08<00:35, 9058.34it/s] 20%|██        | 80198/400000 [00:08<00:35, 9071.00it/s] 20%|██        | 81106/400000 [00:09<00:35, 8960.01it/s] 21%|██        | 82022/400000 [00:09<00:35, 9017.43it/s] 21%|██        | 82940/400000 [00:09<00:34, 9065.30it/s] 21%|██        | 83847/400000 [00:09<00:34, 9064.00it/s] 21%|██        | 84754/400000 [00:09<00:34, 9035.77it/s] 21%|██▏       | 85658/400000 [00:09<00:35, 8916.73it/s] 22%|██▏       | 86551/400000 [00:09<00:35, 8905.47it/s] 22%|██▏       | 87442/400000 [00:09<00:35, 8891.34it/s] 22%|██▏       | 88351/400000 [00:09<00:34, 8949.56it/s] 22%|██▏       | 89247/400000 [00:10<00:34, 8942.30it/s] 23%|██▎       | 90142/400000 [00:10<00:34, 8902.05it/s] 23%|██▎       | 91061/400000 [00:10<00:34, 8985.80it/s] 23%|██▎       | 91982/400000 [00:10<00:34, 9050.07it/s] 23%|██▎       | 92892/400000 [00:10<00:33, 9064.91it/s] 23%|██▎       | 93814/400000 [00:10<00:33, 9109.89it/s] 24%|██▎       | 94726/400000 [00:10<00:33, 8983.89it/s] 24%|██▍       | 95634/400000 [00:10<00:33, 9010.75it/s] 24%|██▍       | 96536/400000 [00:10<00:34, 8890.87it/s] 24%|██▍       | 97426/400000 [00:10<00:34, 8760.08it/s] 25%|██▍       | 98337/400000 [00:11<00:34, 8860.75it/s] 25%|██▍       | 99235/400000 [00:11<00:33, 8894.53it/s] 25%|██▌       | 100161/400000 [00:11<00:33, 9000.07it/s] 25%|██▌       | 101081/400000 [00:11<00:32, 9059.08it/s] 25%|██▌       | 101988/400000 [00:11<00:33, 9016.07it/s] 26%|██▌       | 102900/400000 [00:11<00:32, 9046.88it/s] 26%|██▌       | 103806/400000 [00:11<00:33, 8865.37it/s] 26%|██▌       | 104714/400000 [00:11<00:33, 8926.55it/s] 26%|██▋       | 105626/400000 [00:11<00:32, 8981.10it/s] 27%|██▋       | 106542/400000 [00:11<00:32, 9031.93it/s] 27%|██▋       | 107446/400000 [00:12<00:32, 9032.96it/s] 27%|██▋       | 108350/400000 [00:12<00:32, 9005.19it/s] 27%|██▋       | 109274/400000 [00:12<00:32, 9074.09it/s] 28%|██▊       | 110182/400000 [00:12<00:32, 9042.03it/s] 28%|██▊       | 111105/400000 [00:12<00:31, 9095.14it/s] 28%|██▊       | 112028/400000 [00:12<00:31, 9132.43it/s] 28%|██▊       | 112942/400000 [00:12<00:32, 8906.28it/s] 28%|██▊       | 113870/400000 [00:12<00:31, 9014.07it/s] 29%|██▊       | 114773/400000 [00:12<00:32, 8844.30it/s] 29%|██▉       | 115682/400000 [00:12<00:31, 8915.09it/s] 29%|██▉       | 116575/400000 [00:13<00:31, 8863.77it/s] 29%|██▉       | 117463/400000 [00:13<00:31, 8843.79it/s] 30%|██▉       | 118394/400000 [00:13<00:31, 8976.88it/s] 30%|██▉       | 119293/400000 [00:13<00:31, 8886.36it/s] 30%|███       | 120216/400000 [00:13<00:31, 8984.13it/s] 30%|███       | 121116/400000 [00:13<00:31, 8906.28it/s] 31%|███       | 122008/400000 [00:13<00:31, 8771.63it/s] 31%|███       | 122887/400000 [00:13<00:31, 8766.31it/s] 31%|███       | 123767/400000 [00:13<00:31, 8774.93it/s] 31%|███       | 124688/400000 [00:13<00:30, 8898.65it/s] 31%|███▏      | 125604/400000 [00:14<00:30, 8974.44it/s] 32%|███▏      | 126507/400000 [00:14<00:30, 8988.76it/s] 32%|███▏      | 127416/400000 [00:14<00:30, 9017.13it/s] 32%|███▏      | 128319/400000 [00:14<00:30, 8999.61it/s] 32%|███▏      | 129241/400000 [00:14<00:29, 9063.05it/s] 33%|███▎      | 130173/400000 [00:14<00:29, 9138.28it/s] 33%|███▎      | 131088/400000 [00:14<00:29, 9089.63it/s] 33%|███▎      | 132004/400000 [00:14<00:29, 9107.62it/s] 33%|███▎      | 132915/400000 [00:14<00:29, 9038.23it/s] 33%|███▎      | 133820/400000 [00:14<00:29, 9039.11it/s] 34%|███▎      | 134747/400000 [00:15<00:29, 9106.91it/s] 34%|███▍      | 135658/400000 [00:15<00:29, 9079.31it/s] 34%|███▍      | 136567/400000 [00:15<00:29, 9046.19it/s] 34%|███▍      | 137489/400000 [00:15<00:28, 9094.94it/s] 35%|███▍      | 138416/400000 [00:15<00:28, 9146.31it/s] 35%|███▍      | 139331/400000 [00:15<00:28, 9085.02it/s] 35%|███▌      | 140240/400000 [00:15<00:29, 8799.07it/s] 35%|███▌      | 141167/400000 [00:15<00:28, 8933.68it/s] 36%|███▌      | 142081/400000 [00:15<00:28, 8993.70it/s] 36%|███▌      | 143008/400000 [00:15<00:28, 9073.38it/s] 36%|███▌      | 143936/400000 [00:16<00:28, 9131.99it/s] 36%|███▌      | 144851/400000 [00:16<00:28, 9089.92it/s] 36%|███▋      | 145776/400000 [00:16<00:27, 9136.03it/s] 37%|███▋      | 146691/400000 [00:16<00:27, 9120.98it/s] 37%|███▋      | 147617/400000 [00:16<00:27, 9160.17it/s] 37%|███▋      | 148546/400000 [00:16<00:27, 9195.56it/s] 37%|███▋      | 149466/400000 [00:16<00:27, 8987.28it/s] 38%|███▊      | 150393/400000 [00:16<00:27, 9069.01it/s] 38%|███▊      | 151318/400000 [00:16<00:27, 9122.32it/s] 38%|███▊      | 152243/400000 [00:16<00:27, 9157.54it/s] 38%|███▊      | 153160/400000 [00:17<00:27, 9081.52it/s] 39%|███▊      | 154069/400000 [00:17<00:27, 9005.42it/s] 39%|███▊      | 154971/400000 [00:17<00:27, 8977.49it/s] 39%|███▉      | 155870/400000 [00:17<00:27, 8972.79it/s] 39%|███▉      | 156800/400000 [00:17<00:26, 9066.75it/s] 39%|███▉      | 157708/400000 [00:17<00:26, 9056.13it/s] 40%|███▉      | 158614/400000 [00:17<00:27, 8847.79it/s] 40%|███▉      | 159501/400000 [00:17<00:27, 8851.19it/s] 40%|████      | 160430/400000 [00:17<00:26, 8977.16it/s] 40%|████      | 161361/400000 [00:18<00:26, 9071.68it/s] 41%|████      | 162277/400000 [00:18<00:26, 9095.51it/s] 41%|████      | 163188/400000 [00:18<00:26, 8963.15it/s] 41%|████      | 164114/400000 [00:18<00:26, 9049.97it/s] 41%|████▏     | 165020/400000 [00:18<00:26, 9021.79it/s] 41%|████▏     | 165928/400000 [00:18<00:25, 9037.66it/s] 42%|████▏     | 166858/400000 [00:18<00:25, 9114.56it/s] 42%|████▏     | 167770/400000 [00:18<00:26, 8793.84it/s] 42%|████▏     | 168690/400000 [00:18<00:25, 8910.80it/s] 42%|████▏     | 169608/400000 [00:18<00:25, 8988.21it/s] 43%|████▎     | 170511/400000 [00:19<00:25, 8999.70it/s] 43%|████▎     | 171435/400000 [00:19<00:25, 9068.19it/s] 43%|████▎     | 172343/400000 [00:19<00:25, 9070.44it/s] 43%|████▎     | 173251/400000 [00:19<00:25, 9049.48it/s] 44%|████▎     | 174165/400000 [00:19<00:24, 9075.23it/s] 44%|████▍     | 175079/400000 [00:19<00:24, 9094.04it/s] 44%|████▍     | 176004/400000 [00:19<00:24, 9139.76it/s] 44%|████▍     | 176919/400000 [00:19<00:24, 8988.47it/s] 44%|████▍     | 177843/400000 [00:19<00:24, 9061.24it/s] 45%|████▍     | 178750/400000 [00:19<00:24, 9062.14it/s] 45%|████▍     | 179662/400000 [00:20<00:24, 9076.68it/s] 45%|████▌     | 180570/400000 [00:20<00:24, 8903.95it/s] 45%|████▌     | 181474/400000 [00:20<00:24, 8940.74it/s] 46%|████▌     | 182374/400000 [00:20<00:24, 8957.53it/s] 46%|████▌     | 183271/400000 [00:20<00:24, 8904.45it/s] 46%|████▌     | 184197/400000 [00:20<00:23, 9006.71it/s] 46%|████▋     | 185126/400000 [00:20<00:23, 9087.04it/s] 47%|████▋     | 186036/400000 [00:20<00:23, 9084.66it/s] 47%|████▋     | 186956/400000 [00:20<00:23, 9118.94it/s] 47%|████▋     | 187869/400000 [00:20<00:23, 9105.69it/s] 47%|████▋     | 188794/400000 [00:21<00:23, 9145.82it/s] 47%|████▋     | 189719/400000 [00:21<00:22, 9175.85it/s] 48%|████▊     | 190637/400000 [00:21<00:23, 9056.01it/s] 48%|████▊     | 191554/400000 [00:21<00:22, 9089.50it/s] 48%|████▊     | 192468/400000 [00:21<00:22, 9103.17it/s] 48%|████▊     | 193391/400000 [00:21<00:22, 9139.34it/s] 49%|████▊     | 194306/400000 [00:21<00:22, 9035.08it/s] 49%|████▉     | 195211/400000 [00:21<00:22, 9037.85it/s] 49%|████▉     | 196116/400000 [00:21<00:22, 9033.57it/s] 49%|████▉     | 197020/400000 [00:21<00:22, 8915.10it/s] 49%|████▉     | 197923/400000 [00:22<00:22, 8948.08it/s] 50%|████▉     | 198857/400000 [00:22<00:22, 9060.50it/s] 50%|████▉     | 199764/400000 [00:22<00:22, 9040.05it/s] 50%|█████     | 200669/400000 [00:22<00:22, 8967.65it/s] 50%|█████     | 201592/400000 [00:22<00:21, 9043.53it/s] 51%|█████     | 202516/400000 [00:22<00:21, 9099.57it/s] 51%|█████     | 203437/400000 [00:22<00:21, 9130.07it/s] 51%|█████     | 204351/400000 [00:22<00:22, 8761.80it/s] 51%|█████▏    | 205279/400000 [00:22<00:21, 8910.94it/s] 52%|█████▏    | 206207/400000 [00:22<00:21, 9016.87it/s] 52%|█████▏    | 207137/400000 [00:23<00:21, 9098.71it/s] 52%|█████▏    | 208059/400000 [00:23<00:21, 9134.33it/s] 52%|█████▏    | 208974/400000 [00:23<00:20, 9102.66it/s] 52%|█████▏    | 209897/400000 [00:23<00:20, 9138.02it/s] 53%|█████▎    | 210824/400000 [00:23<00:20, 9175.88it/s] 53%|█████▎    | 211753/400000 [00:23<00:20, 9207.77it/s] 53%|█████▎    | 212676/400000 [00:23<00:20, 9213.36it/s] 53%|█████▎    | 213598/400000 [00:23<00:20, 9179.58it/s] 54%|█████▎    | 214517/400000 [00:23<00:20, 9084.11it/s] 54%|█████▍    | 215428/400000 [00:23<00:20, 9089.96it/s] 54%|█████▍    | 216355/400000 [00:24<00:20, 9141.89it/s] 54%|█████▍    | 217272/400000 [00:24<00:19, 9149.34it/s] 55%|█████▍    | 218188/400000 [00:24<00:19, 9136.88it/s] 55%|█████▍    | 219102/400000 [00:24<00:19, 9065.83it/s] 55%|█████▌    | 220009/400000 [00:24<00:19, 9036.70it/s] 55%|█████▌    | 220919/400000 [00:24<00:19, 9053.64it/s] 55%|█████▌    | 221833/400000 [00:24<00:19, 9077.05it/s] 56%|█████▌    | 222741/400000 [00:24<00:19, 8952.91it/s] 56%|█████▌    | 223662/400000 [00:24<00:19, 9027.78it/s] 56%|█████▌    | 224583/400000 [00:24<00:19, 9080.84it/s] 56%|█████▋    | 225507/400000 [00:25<00:19, 9126.32it/s] 57%|█████▋    | 226420/400000 [00:25<00:19, 9069.29it/s] 57%|█████▋    | 227328/400000 [00:25<00:19, 9038.05it/s] 57%|█████▋    | 228233/400000 [00:25<00:19, 8946.18it/s] 57%|█████▋    | 229158/400000 [00:25<00:18, 9033.82it/s] 58%|█████▊    | 230067/400000 [00:25<00:18, 9048.85it/s] 58%|█████▊    | 230985/400000 [00:25<00:18, 9086.51it/s] 58%|█████▊    | 231894/400000 [00:25<00:18, 9056.80it/s] 58%|█████▊    | 232810/400000 [00:25<00:18, 9084.67it/s] 58%|█████▊    | 233729/400000 [00:26<00:18, 9114.42it/s] 59%|█████▊    | 234641/400000 [00:26<00:18, 9100.09it/s] 59%|█████▉    | 235552/400000 [00:26<00:18, 9036.49it/s] 59%|█████▉    | 236459/400000 [00:26<00:18, 9045.20it/s] 59%|█████▉    | 237380/400000 [00:26<00:17, 9091.33it/s] 60%|█████▉    | 238301/400000 [00:26<00:17, 9124.38it/s] 60%|█████▉    | 239214/400000 [00:26<00:17, 9028.24it/s] 60%|██████    | 240141/400000 [00:26<00:17, 9097.56it/s] 60%|██████    | 241052/400000 [00:26<00:17, 9030.09it/s] 60%|██████    | 241974/400000 [00:26<00:17, 9084.65it/s] 61%|██████    | 242883/400000 [00:27<00:17, 9037.34it/s] 61%|██████    | 243799/400000 [00:27<00:17, 9072.36it/s] 61%|██████    | 244719/400000 [00:27<00:17, 9109.71it/s] 61%|██████▏   | 245631/400000 [00:27<00:16, 9086.91it/s] 62%|██████▏   | 246540/400000 [00:27<00:16, 9080.50it/s] 62%|██████▏   | 247449/400000 [00:27<00:16, 9037.37it/s] 62%|██████▏   | 248353/400000 [00:27<00:16, 8990.45it/s] 62%|██████▏   | 249253/400000 [00:27<00:16, 8889.15it/s] 63%|██████▎   | 250204/400000 [00:27<00:16, 9066.62it/s] 63%|██████▎   | 251112/400000 [00:27<00:17, 8742.51it/s] 63%|██████▎   | 252098/400000 [00:28<00:16, 9049.33it/s] 63%|██████▎   | 253110/400000 [00:28<00:15, 9345.09it/s] 64%|██████▎   | 254174/400000 [00:28<00:15, 9697.00it/s] 64%|██████▍   | 255237/400000 [00:28<00:14, 9959.01it/s] 64%|██████▍   | 256240/400000 [00:28<00:14, 9806.66it/s] 64%|██████▍   | 257350/400000 [00:28<00:14, 10160.91it/s] 65%|██████▍   | 258397/400000 [00:28<00:13, 10250.25it/s] 65%|██████▍   | 259448/400000 [00:28<00:13, 10324.61it/s] 65%|██████▌   | 260507/400000 [00:28<00:13, 10402.46it/s] 65%|██████▌   | 261550/400000 [00:28<00:13, 10381.28it/s] 66%|██████▌   | 262670/400000 [00:29<00:12, 10613.86it/s] 66%|██████▌   | 263765/400000 [00:29<00:12, 10709.77it/s] 66%|██████▌   | 264867/400000 [00:29<00:12, 10800.58it/s] 66%|██████▋   | 265949/400000 [00:29<00:12, 10656.36it/s] 67%|██████▋   | 267017/400000 [00:29<00:12, 10608.19it/s] 67%|██████▋   | 268148/400000 [00:29<00:12, 10806.95it/s] 67%|██████▋   | 269244/400000 [00:29<00:12, 10850.52it/s] 68%|██████▊   | 270331/400000 [00:29<00:12, 10537.27it/s] 68%|██████▊   | 271398/400000 [00:29<00:12, 10574.88it/s] 68%|██████▊   | 272458/400000 [00:29<00:12, 10528.85it/s] 68%|██████▊   | 273513/400000 [00:30<00:12, 10178.46it/s] 69%|██████▊   | 274616/400000 [00:30<00:12, 10417.90it/s] 69%|██████▉   | 275697/400000 [00:30<00:11, 10530.25it/s] 69%|██████▉   | 276754/400000 [00:30<00:11, 10423.80it/s] 69%|██████▉   | 277799/400000 [00:30<00:12, 10117.55it/s] 70%|██████▉   | 278898/400000 [00:30<00:11, 10364.24it/s] 70%|██████▉   | 279939/400000 [00:30<00:11, 10322.54it/s] 70%|███████   | 280975/400000 [00:30<00:11, 10128.84it/s] 71%|███████   | 282010/400000 [00:30<00:11, 10194.07it/s] 71%|███████   | 283048/400000 [00:31<00:11, 10246.06it/s] 71%|███████   | 284075/400000 [00:31<00:11, 10220.98it/s] 71%|███████▏  | 285147/400000 [00:31<00:11, 10364.89it/s] 72%|███████▏  | 286291/400000 [00:31<00:10, 10665.54it/s] 72%|███████▏  | 287361/400000 [00:31<00:10, 10562.41it/s] 72%|███████▏  | 288483/400000 [00:31<00:10, 10749.09it/s] 72%|███████▏  | 289561/400000 [00:31<00:10, 10226.15it/s] 73%|███████▎  | 290591/400000 [00:31<00:10, 10121.74it/s] 73%|███████▎  | 291676/400000 [00:31<00:10, 10323.44it/s] 73%|███████▎  | 292737/400000 [00:31<00:10, 10405.52it/s] 73%|███████▎  | 293781/400000 [00:32<00:10, 10360.23it/s] 74%|███████▎  | 294839/400000 [00:32<00:10, 10424.21it/s] 74%|███████▍  | 295935/400000 [00:32<00:09, 10576.38it/s] 74%|███████▍  | 297046/400000 [00:32<00:09, 10730.36it/s] 75%|███████▍  | 298121/400000 [00:32<00:09, 10291.78it/s] 75%|███████▍  | 299156/400000 [00:32<00:09, 10287.24it/s] 75%|███████▌  | 300243/400000 [00:32<00:09, 10454.78it/s] 75%|███████▌  | 301292/400000 [00:32<00:09, 10275.74it/s] 76%|███████▌  | 302418/400000 [00:32<00:09, 10551.44it/s] 76%|███████▌  | 303478/400000 [00:32<00:09, 10349.29it/s] 76%|███████▌  | 304572/400000 [00:33<00:09, 10517.75it/s] 76%|███████▋  | 305633/400000 [00:33<00:08, 10543.99it/s] 77%|███████▋  | 306690/400000 [00:33<00:08, 10497.41it/s] 77%|███████▋  | 307789/400000 [00:33<00:08, 10638.61it/s] 77%|███████▋  | 308855/400000 [00:33<00:08, 10478.78it/s] 77%|███████▋  | 309905/400000 [00:33<00:08, 10433.09it/s] 78%|███████▊  | 310991/400000 [00:33<00:08, 10554.99it/s] 78%|███████▊  | 312062/400000 [00:33<00:08, 10598.36it/s] 78%|███████▊  | 313145/400000 [00:33<00:08, 10666.38it/s] 79%|███████▊  | 314213/400000 [00:33<00:08, 10217.93it/s] 79%|███████▉  | 315260/400000 [00:34<00:08, 10291.29it/s] 79%|███████▉  | 316293/400000 [00:34<00:08, 10158.94it/s] 79%|███████▉  | 317371/400000 [00:34<00:07, 10337.41it/s] 80%|███████▉  | 318427/400000 [00:34<00:07, 10402.28it/s] 80%|███████▉  | 319470/400000 [00:34<00:07, 10405.39it/s] 80%|████████  | 320560/400000 [00:34<00:07, 10546.97it/s] 80%|████████  | 321654/400000 [00:34<00:07, 10661.45it/s] 81%|████████  | 322722/400000 [00:34<00:07, 10658.58it/s] 81%|████████  | 323789/400000 [00:34<00:07, 10638.47it/s] 81%|████████  | 324854/400000 [00:34<00:07, 10352.01it/s] 81%|████████▏ | 325943/400000 [00:35<00:07, 10504.43it/s] 82%|████████▏ | 326996/400000 [00:35<00:07, 10068.87it/s] 82%|████████▏ | 328083/400000 [00:35<00:06, 10294.91it/s] 82%|████████▏ | 329139/400000 [00:35<00:06, 10371.72it/s] 83%|████████▎ | 330180/400000 [00:35<00:06, 10103.43it/s] 83%|████████▎ | 331195/400000 [00:35<00:06, 10004.63it/s] 83%|████████▎ | 332230/400000 [00:35<00:06, 10105.58it/s] 83%|████████▎ | 333316/400000 [00:35<00:06, 10319.98it/s] 84%|████████▎ | 334403/400000 [00:35<00:06, 10477.31it/s] 84%|████████▍ | 335454/400000 [00:36<00:06, 10386.30it/s] 84%|████████▍ | 336504/400000 [00:36<00:06, 10417.80it/s] 84%|████████▍ | 337548/400000 [00:36<00:05, 10418.91it/s] 85%|████████▍ | 338591/400000 [00:36<00:05, 10309.76it/s] 85%|████████▍ | 339659/400000 [00:36<00:05, 10415.98it/s] 85%|████████▌ | 340702/400000 [00:36<00:05, 10313.67it/s] 85%|████████▌ | 341735/400000 [00:36<00:05, 10170.92it/s] 86%|████████▌ | 342804/400000 [00:36<00:05, 10319.89it/s] 86%|████████▌ | 343838/400000 [00:36<00:05, 10214.78it/s] 86%|████████▌ | 344915/400000 [00:36<00:05, 10373.47it/s] 86%|████████▋ | 345961/400000 [00:37<00:05, 10396.94it/s] 87%|████████▋ | 347016/400000 [00:37<00:05, 10439.75it/s] 87%|████████▋ | 348085/400000 [00:37<00:04, 10513.51it/s] 87%|████████▋ | 349137/400000 [00:37<00:04, 10186.04it/s] 88%|████████▊ | 350159/400000 [00:37<00:05, 9554.04it/s]  88%|████████▊ | 351125/400000 [00:37<00:05, 9551.94it/s] 88%|████████▊ | 352182/400000 [00:37<00:04, 9833.71it/s] 88%|████████▊ | 353173/400000 [00:37<00:04, 9548.24it/s] 89%|████████▊ | 354300/400000 [00:37<00:04, 10004.28it/s] 89%|████████▉ | 355311/400000 [00:37<00:04, 9812.60it/s]  89%|████████▉ | 356346/400000 [00:38<00:04, 9965.84it/s] 89%|████████▉ | 357429/400000 [00:38<00:04, 10208.41it/s] 90%|████████▉ | 358521/400000 [00:38<00:03, 10411.92it/s] 90%|████████▉ | 359568/400000 [00:38<00:03, 10298.52it/s] 90%|█████████ | 360602/400000 [00:38<00:03, 10307.38it/s] 90%|█████████ | 361671/400000 [00:38<00:03, 10417.61it/s] 91%|█████████ | 362783/400000 [00:38<00:03, 10617.50it/s] 91%|█████████ | 363908/400000 [00:38<00:03, 10797.87it/s] 91%|█████████▏| 365050/400000 [00:38<00:03, 10974.12it/s] 92%|█████████▏| 366150/400000 [00:39<00:03, 10975.40it/s] 92%|█████████▏| 367250/400000 [00:39<00:03, 10718.94it/s] 92%|█████████▏| 368366/400000 [00:39<00:02, 10844.67it/s] 92%|█████████▏| 369453/400000 [00:39<00:02, 10697.70it/s] 93%|█████████▎| 370525/400000 [00:39<00:02, 10600.69it/s] 93%|█████████▎| 371589/400000 [00:39<00:02, 10611.34it/s] 93%|█████████▎| 372654/400000 [00:39<00:02, 10621.88it/s] 93%|█████████▎| 373717/400000 [00:39<00:02, 10580.43it/s] 94%|█████████▎| 374776/400000 [00:39<00:02, 10493.50it/s] 94%|█████████▍| 375826/400000 [00:39<00:02, 10320.78it/s] 94%|█████████▍| 376860/400000 [00:40<00:02, 10199.63it/s] 94%|█████████▍| 377882/400000 [00:40<00:02, 10145.26it/s] 95%|█████████▍| 378898/400000 [00:40<00:02, 10062.34it/s] 95%|█████████▍| 379923/400000 [00:40<00:01, 10116.32it/s] 95%|█████████▌| 380936/400000 [00:40<00:01, 10035.58it/s] 95%|█████████▌| 381941/400000 [00:40<00:01, 10000.06it/s] 96%|█████████▌| 383036/400000 [00:40<00:01, 10265.45it/s] 96%|█████████▌| 384065/400000 [00:40<00:01, 10153.33it/s] 96%|█████████▋| 385083/400000 [00:40<00:01, 10085.17it/s] 97%|█████████▋| 386175/400000 [00:40<00:01, 10318.86it/s] 97%|█████████▋| 387239/400000 [00:41<00:01, 10410.82it/s] 97%|█████████▋| 388282/400000 [00:41<00:01, 10379.37it/s] 97%|█████████▋| 389364/400000 [00:41<00:01, 10507.19it/s] 98%|█████████▊| 390438/400000 [00:41<00:00, 10574.58it/s] 98%|█████████▊| 391540/400000 [00:41<00:00, 10701.70it/s] 98%|█████████▊| 392612/400000 [00:41<00:00, 10354.34it/s] 98%|█████████▊| 393651/400000 [00:41<00:00, 10360.87it/s] 99%|█████████▊| 394775/400000 [00:41<00:00, 10608.00it/s] 99%|█████████▉| 395908/400000 [00:41<00:00, 10813.09it/s] 99%|█████████▉| 396993/400000 [00:41<00:00, 10734.50it/s]100%|█████████▉| 398069/400000 [00:42<00:00, 10658.60it/s]100%|█████████▉| 399170/400000 [00:42<00:00, 10761.38it/s]100%|█████████▉| 399999/400000 [00:42<00:00, 9468.85it/s] Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fd5e81376a0>, <torchtext.data.dataset.TabularDataset object at 0x7fd5e81377f0>, <torchtext.vocab.Vocab object at 0x7fd5e8137710>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.19 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.19 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.19 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.10 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.10 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.65 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.65 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.57 file/s]2020-06-24 00:17:43.254351: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-24 00:17:43.258357: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-06-24 00:17:43.258856: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56402f629a50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-24 00:17:43.258928: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:20, 123184.97it/s] 80%|████████  | 7979008/9912422 [00:00<00:10, 175861.28it/s]9920512it [00:00, 39983321.56it/s]                           
0it [00:00, ?it/s]32768it [00:00, 588623.82it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162144.98it/s]1654784it [00:00, 11677508.24it/s]                         
0it [00:00, ?it/s]8192it [00:00, 202890.67it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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


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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff4abf80ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff4abf80ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ff51723f1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ff51723f1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff535587e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff535587e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff4c4567488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff4c4567488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff4c4567488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 450772.24it/s] 90%|█████████ | 8962048/9912422 [00:00<00:01, 642567.25it/s]9920512it [00:00, 45536332.16it/s]                           
0it [00:00, ?it/s]32768it [00:00, 686638.59it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1017528.28it/s]1654784it [00:00, 12599899.34it/s]                           
0it [00:00, ?it/s]8192it [00:00, 224015.61it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4c1a9ae80>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4c1abc668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ff4c45670d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ff4c45670d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ff4c45670d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<02:07,  1.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<02:07,  1.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 2/162 [00:00<01:36,  1.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:36,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 3/162 [00:01<01:13,  2.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:01<01:13,  2.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:01<01:13,  2.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:01<00:55,  2.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<00:55,  2.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<00:55,  2.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:01<00:42,  3.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<00:42,  3.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:41,  3.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:01<00:32,  4.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:32,  4.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:32,  4.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:01<00:24,  6.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:24,  6.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:24,  6.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:01<00:19,  7.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:19,  7.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:19,  7.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:15,  9.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:15,  9.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:15,  9.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:02<00:13, 10.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:02<00:13, 10.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:02<00:13, 10.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:02<00:11, 12.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:02<00:11, 12.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:02<00:11, 12.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:02<00:10, 13.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:02<00:10, 13.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:02<00:10, 13.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:02<00:09, 14.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:02<00:09, 14.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:02<00:09, 14.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:02<00:08, 15.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:02<00:08, 15.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:02<00:08, 15.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:02<00:08, 16.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:02<00:08, 16.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:02<00:08, 16.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:02<00:08, 16.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:02<00:08, 16.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:02<00:07, 16.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:02<00:07, 16.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:02<00:07, 16.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:02<00:07, 16.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:02<00:07, 17.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:02<00:07, 17.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:02<00:07, 17.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:03<00:07, 17.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:03<00:07, 17.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:03<00:07, 17.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:03<00:06, 17.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:03<00:06, 17.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:03<00:06, 17.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:03<00:06, 17.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:03<00:06, 18.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:03<00:06, 18.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:03<00:06, 18.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:03<00:06, 18.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:03<00:05, 20.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:03<00:05, 20.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:03<00:05, 20.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:03<00:05, 20.02 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:03<00:05, 21.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:03<00:05, 21.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:03<00:05, 21.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:03<00:05, 21.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:03<00:05, 22.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:03<00:05, 22.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:03<00:05, 22.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:03<00:04, 22.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:03<00:04, 22.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:03<00:04, 22.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:03<00:04, 22.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:03<00:04, 22.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:03<00:04, 23.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:03<00:04, 23.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:03<00:04, 23.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:03<00:04, 23.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:04<00:04, 23.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:04<00:04, 23.62 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:04<00:04, 23.62 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:04<00:04, 23.62 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:04<00:04, 23.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:04<00:04, 23.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:04<00:04, 23.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:04<00:04, 23.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:04<00:04, 24.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:04<00:04, 24.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:04<00:03, 24.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:04<00:03, 24.25 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:04<00:03, 24.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:04<00:03, 24.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:04<00:03, 24.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:04<00:03, 24.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:04<00:03, 24.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:04<00:03, 24.49 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:04<00:03, 24.49 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:04<00:03, 24.49 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:04<00:03, 24.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:04<00:03, 24.28 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:04<00:03, 24.28 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:04<00:03, 24.28 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:04<00:03, 24.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:04<00:03, 24.43 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:04<00:03, 24.43 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:04<00:03, 24.43 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:04<00:03, 24.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:04<00:03, 24.59 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:04<00:03, 24.59 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:04<00:03, 24.59 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:04<00:03, 23.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:04<00:03, 23.46 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:05<00:03, 23.46 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:05<00:03, 23.46 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:05<00:03, 22.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:05<00:03, 22.63 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:05<00:03, 22.63 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:05<00:03, 22.63 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:05<00:03, 21.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:05<00:03, 21.60 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:05<00:03, 21.60 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:05<00:03, 21.60 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:05<00:03, 22.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:05<00:03, 22.74 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:05<00:03, 22.74 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:05<00:03, 22.74 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:05<00:03, 21.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:05<00:03, 21.45 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:05<00:03, 21.45 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:05<00:03, 21.45 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:05<00:02, 22.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:05<00:02, 22.64 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:05<00:02, 22.64 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:05<00:02, 22.64 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:05<00:02, 23.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:05<00:02, 23.25 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:05<00:02, 23.25 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:05<00:02, 23.25 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:05<00:02, 23.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:05<00:02, 23.76 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:05<00:02, 23.76 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:06<00:02, 23.76 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:06<00:02, 23.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:06<00:02, 23.36 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:06<00:02, 23.36 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:06<00:02, 23.36 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:06<00:02, 21.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:06<00:02, 21.65 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:06<00:02, 21.65 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:06<00:02, 21.65 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:06<00:02, 20.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:06<00:02, 20.03 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:06<00:02, 20.03 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:06<00:02, 20.03 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:06<00:02, 19.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:06<00:02, 19.02 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:06<00:02, 19.02 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:06<00:02, 18.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:06<00:02, 18.46 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:06<00:02, 18.46 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:06<00:02, 18.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:06<00:02, 18.11 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:06<00:02, 18.11 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:06<00:02, 17.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:06<00:02, 17.88 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:06<00:02, 17.88 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:07<00:02, 18.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:07<00:02, 18.26 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:07<00:02, 18.26 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:07<00:02, 18.26 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:07<00:01, 19.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:07<00:01, 19.64 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:07<00:01, 19.64 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:07<00:01, 19.64 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:07<00:01, 21.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:07<00:01, 21.63 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:07<00:01, 21.63 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:07<00:01, 21.63 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:07<00:01, 23.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:07<00:01, 23.34 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:07<00:01, 23.34 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:07<00:01, 23.34 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:07<00:01, 24.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:07<00:01, 24.76 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:07<00:01, 24.76 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:07<00:01, 24.76 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:07<00:00, 25.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:07<00:00, 25.77 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:07<00:00, 25.77 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:07<00:00, 25.77 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:07<00:00, 26.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:07<00:00, 26.60 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:07<00:00, 26.60 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:07<00:00, 26.60 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:07<00:00, 26.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:07<00:00, 26.67 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:07<00:00, 26.67 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:07<00:00, 26.67 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:07<00:00, 26.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:07<00:00, 26.99 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:07<00:00, 26.99 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:07<00:00, 26.99 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:08<00:00, 26.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:08<00:00, 26.94 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:08<00:00, 26.94 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:08<00:00, 26.94 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:08<00:00, 26.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:08<00:00, 26.99 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:08<00:00, 26.99 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:08<00:00, 26.99 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:08<00:00, 26.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:08<00:00, 26.81 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:08<00:00, 26.81 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:08<00:00, 26.81 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:08<00:00, 26.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:08<00:00, 26.85 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:08<00:00, 26.85 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:08<00:00, 26.85 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:08<00:00, 27.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00, 27.01 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:08<00:00,  8.46s/ url]Dl Completed...: 100%|██████████| 1/1 [00:08<00:00,  8.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00, 27.01 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:08<00:00,  8.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00, 27.01 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:08<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:10<00:00, 10.86s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:10<00:00,  8.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:10<00:00, 27.01 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:10<00:00, 10.86s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:10<00:00, 10.86s/ file]
Dl Size...: 100%|██████████| 162/162 [00:10<00:00, 14.92 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:10<00:00, 10.86s/ url]
0 examples [00:00, ? examples/s]2020-06-30 18:09:35.709863: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-30 18:09:35.721806: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-30 18:09:35.722505: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561e95769230 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-30 18:09:35.722526: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
21 examples [00:00, 208.75 examples/s]118 examples [00:00, 272.91 examples/s]208 examples [00:00, 344.63 examples/s]296 examples [00:00, 421.46 examples/s]385 examples [00:00, 500.50 examples/s]479 examples [00:00, 581.91 examples/s]575 examples [00:00, 659.54 examples/s]659 examples [00:00, 704.44 examples/s]755 examples [00:00, 763.96 examples/s]852 examples [00:01, 815.50 examples/s]951 examples [00:01, 860.49 examples/s]1050 examples [00:01, 894.93 examples/s]1148 examples [00:01, 918.09 examples/s]1249 examples [00:01, 942.06 examples/s]1347 examples [00:01, 933.65 examples/s]1444 examples [00:01, 943.86 examples/s]1540 examples [00:01, 948.57 examples/s]1636 examples [00:01, 935.19 examples/s]1731 examples [00:01, 931.34 examples/s]1825 examples [00:02, 912.00 examples/s]1917 examples [00:02, 907.74 examples/s]2009 examples [00:02, 875.09 examples/s]2097 examples [00:02, 865.76 examples/s]2187 examples [00:02, 873.55 examples/s]2278 examples [00:02, 882.71 examples/s]2375 examples [00:02, 905.06 examples/s]2472 examples [00:02, 920.39 examples/s]2571 examples [00:02, 938.77 examples/s]2670 examples [00:02, 952.88 examples/s]2766 examples [00:03, 953.24 examples/s]2862 examples [00:03, 946.15 examples/s]2957 examples [00:03, 939.60 examples/s]3052 examples [00:03, 930.79 examples/s]3153 examples [00:03, 950.64 examples/s]3250 examples [00:03, 956.29 examples/s]3351 examples [00:03, 970.31 examples/s]3452 examples [00:03, 979.34 examples/s]3551 examples [00:03, 958.07 examples/s]3647 examples [00:03, 926.66 examples/s]3741 examples [00:04, 902.53 examples/s]3834 examples [00:04, 908.90 examples/s]3936 examples [00:04, 937.56 examples/s]4036 examples [00:04, 954.05 examples/s]4136 examples [00:04, 967.33 examples/s]4236 examples [00:04, 976.24 examples/s]4334 examples [00:04, 956.70 examples/s]4435 examples [00:04, 971.25 examples/s]4535 examples [00:04, 974.14 examples/s]4634 examples [00:05, 977.04 examples/s]4735 examples [00:05, 985.48 examples/s]4834 examples [00:05, 985.96 examples/s]4937 examples [00:05, 996.25 examples/s]5037 examples [00:05, 996.86 examples/s]5137 examples [00:05, 987.96 examples/s]5236 examples [00:05, 958.57 examples/s]5333 examples [00:05, 946.80 examples/s]5428 examples [00:05, 941.64 examples/s]5523 examples [00:05, 941.17 examples/s]5618 examples [00:06, 941.65 examples/s]5717 examples [00:06, 953.89 examples/s]5818 examples [00:06, 969.71 examples/s]5916 examples [00:06, 963.80 examples/s]6013 examples [00:06, 912.47 examples/s]6105 examples [00:06, 905.23 examples/s]6204 examples [00:06, 928.81 examples/s]6301 examples [00:06, 939.48 examples/s]6398 examples [00:06, 948.10 examples/s]6494 examples [00:06, 934.10 examples/s]6592 examples [00:07, 945.47 examples/s]6687 examples [00:07, 915.95 examples/s]6779 examples [00:07, 900.43 examples/s]6878 examples [00:07, 925.12 examples/s]6977 examples [00:07, 943.29 examples/s]7072 examples [00:07, 944.77 examples/s]7168 examples [00:07, 949.04 examples/s]7264 examples [00:07, 950.81 examples/s]7360 examples [00:07, 952.82 examples/s]7460 examples [00:07, 965.19 examples/s]7557 examples [00:08, 941.88 examples/s]7657 examples [00:08, 958.17 examples/s]7754 examples [00:08, 944.36 examples/s]7852 examples [00:08, 953.60 examples/s]7948 examples [00:08, 914.17 examples/s]8040 examples [00:08, 899.56 examples/s]8135 examples [00:08, 912.19 examples/s]8235 examples [00:08, 934.59 examples/s]8329 examples [00:08, 934.75 examples/s]8423 examples [00:09, 934.62 examples/s]8517 examples [00:09, 929.19 examples/s]8611 examples [00:09, 931.68 examples/s]8705 examples [00:09, 933.50 examples/s]8805 examples [00:09, 950.44 examples/s]8906 examples [00:09, 963.90 examples/s]9004 examples [00:09, 966.84 examples/s]9102 examples [00:09, 970.62 examples/s]9204 examples [00:09, 983.56 examples/s]9303 examples [00:09, 973.48 examples/s]9401 examples [00:10, 948.45 examples/s]9499 examples [00:10, 957.26 examples/s]9595 examples [00:10, 898.41 examples/s]9694 examples [00:10, 922.02 examples/s]9796 examples [00:10, 947.06 examples/s]9892 examples [00:10, 947.42 examples/s]9992 examples [00:10, 961.73 examples/s]10089 examples [00:10, 908.14 examples/s]10188 examples [00:10, 929.49 examples/s]10290 examples [00:10, 952.47 examples/s]10392 examples [00:11, 969.46 examples/s]10491 examples [00:11, 974.48 examples/s]10589 examples [00:11, 929.79 examples/s]10683 examples [00:11, 916.27 examples/s]10776 examples [00:11, 919.95 examples/s]10874 examples [00:11, 936.99 examples/s]10971 examples [00:11, 942.49 examples/s]11066 examples [00:11, 942.47 examples/s]11161 examples [00:11, 941.69 examples/s]11256 examples [00:12, 873.68 examples/s]11355 examples [00:12, 903.91 examples/s]11456 examples [00:12, 930.80 examples/s]11558 examples [00:12, 953.36 examples/s]11660 examples [00:12, 971.93 examples/s]11762 examples [00:12, 983.70 examples/s]11862 examples [00:12, 987.93 examples/s]11962 examples [00:12, 983.65 examples/s]12064 examples [00:12, 993.44 examples/s]12164 examples [00:12, 981.39 examples/s]12263 examples [00:13, 943.61 examples/s]12364 examples [00:13, 960.51 examples/s]12463 examples [00:13, 968.72 examples/s]12564 examples [00:13, 978.23 examples/s]12666 examples [00:13, 989.28 examples/s]12767 examples [00:13, 993.11 examples/s]12867 examples [00:13, 988.56 examples/s]12966 examples [00:13, 971.68 examples/s]13064 examples [00:13, 964.94 examples/s]13161 examples [00:13, 965.05 examples/s]13258 examples [00:14, 960.65 examples/s]13355 examples [00:14, 955.43 examples/s]13451 examples [00:14, 942.14 examples/s]13546 examples [00:14, 922.52 examples/s]13645 examples [00:14, 940.72 examples/s]13741 examples [00:14, 945.42 examples/s]13836 examples [00:14, 926.11 examples/s]13929 examples [00:14, 919.52 examples/s]14022 examples [00:14, 918.70 examples/s]14115 examples [00:15, 919.32 examples/s]14208 examples [00:15, 906.29 examples/s]14300 examples [00:15, 908.97 examples/s]14391 examples [00:15, 890.38 examples/s]14481 examples [00:15, 858.26 examples/s]14578 examples [00:15, 888.48 examples/s]14678 examples [00:15, 917.95 examples/s]14777 examples [00:15, 937.86 examples/s]14875 examples [00:15, 949.42 examples/s]14975 examples [00:15, 961.71 examples/s]15072 examples [00:16, 942.87 examples/s]15169 examples [00:16, 948.68 examples/s]15271 examples [00:16, 968.27 examples/s]15369 examples [00:16, 953.17 examples/s]15467 examples [00:16, 958.45 examples/s]15568 examples [00:16, 970.94 examples/s]15666 examples [00:16, 972.73 examples/s]15764 examples [00:16, 972.81 examples/s]15862 examples [00:16, 936.45 examples/s]15956 examples [00:16, 930.60 examples/s]16050 examples [00:17, 920.37 examples/s]16143 examples [00:17, 912.10 examples/s]16238 examples [00:17, 921.59 examples/s]16331 examples [00:17, 914.65 examples/s]16433 examples [00:17, 941.67 examples/s]16533 examples [00:17, 957.23 examples/s]16629 examples [00:17, 957.64 examples/s]16725 examples [00:17, 906.42 examples/s]16817 examples [00:17, 909.64 examples/s]16914 examples [00:18, 924.63 examples/s]17011 examples [00:18, 935.26 examples/s]17108 examples [00:18, 945.05 examples/s]17206 examples [00:18, 954.51 examples/s]17305 examples [00:18, 962.48 examples/s]17406 examples [00:18, 973.41 examples/s]17504 examples [00:18, 952.31 examples/s]17600 examples [00:18, 940.09 examples/s]17698 examples [00:18, 950.74 examples/s]17797 examples [00:18, 961.78 examples/s]17894 examples [00:19, 963.58 examples/s]17991 examples [00:19, 938.55 examples/s]18087 examples [00:19, 944.50 examples/s]18182 examples [00:19, 918.91 examples/s]18278 examples [00:19, 929.32 examples/s]18376 examples [00:19, 943.26 examples/s]18474 examples [00:19, 952.29 examples/s]18574 examples [00:19, 964.23 examples/s]18675 examples [00:19, 975.90 examples/s]18773 examples [00:19, 972.95 examples/s]18871 examples [00:20, 963.64 examples/s]18971 examples [00:20, 974.00 examples/s]19069 examples [00:20, 957.69 examples/s]19165 examples [00:20, 951.42 examples/s]19265 examples [00:20, 964.14 examples/s]19363 examples [00:20, 967.69 examples/s]19463 examples [00:20, 974.64 examples/s]19561 examples [00:20, 971.85 examples/s]19660 examples [00:20, 975.54 examples/s]19762 examples [00:20, 986.82 examples/s]19864 examples [00:21, 996.25 examples/s]19966 examples [00:21, 1002.74 examples/s]20067 examples [00:21, 962.57 examples/s] 20165 examples [00:21, 967.60 examples/s]20265 examples [00:21, 975.04 examples/s]20363 examples [00:21, 968.26 examples/s]20460 examples [00:21, 919.80 examples/s]20553 examples [00:21, 917.89 examples/s]20646 examples [00:21, 921.48 examples/s]20746 examples [00:22, 941.57 examples/s]20843 examples [00:22, 947.92 examples/s]20939 examples [00:22, 933.34 examples/s]21038 examples [00:22, 949.30 examples/s]21135 examples [00:22, 954.14 examples/s]21232 examples [00:22, 958.60 examples/s]21328 examples [00:22, 935.94 examples/s]21424 examples [00:22, 940.58 examples/s]21519 examples [00:22, 941.94 examples/s]21614 examples [00:22, 936.00 examples/s]21708 examples [00:23, 929.70 examples/s]21803 examples [00:23, 934.49 examples/s]21904 examples [00:23, 953.12 examples/s]22000 examples [00:23, 945.29 examples/s]22095 examples [00:23, 936.99 examples/s]22192 examples [00:23, 945.63 examples/s]22292 examples [00:23, 958.87 examples/s]22390 examples [00:23, 964.41 examples/s]22487 examples [00:23, 942.92 examples/s]22582 examples [00:23, 931.87 examples/s]22683 examples [00:24, 953.09 examples/s]22779 examples [00:24, 904.96 examples/s]22880 examples [00:24, 932.79 examples/s]22978 examples [00:24, 945.07 examples/s]23077 examples [00:24, 957.00 examples/s]23174 examples [00:24, 953.53 examples/s]23273 examples [00:24, 963.06 examples/s]23370 examples [00:24, 946.96 examples/s]23468 examples [00:24, 956.01 examples/s]23566 examples [00:24, 961.41 examples/s]23663 examples [00:25, 951.04 examples/s]23761 examples [00:25, 957.25 examples/s]23857 examples [00:25, 912.96 examples/s]23953 examples [00:25, 926.36 examples/s]24047 examples [00:25, 920.08 examples/s]24144 examples [00:25, 933.15 examples/s]24245 examples [00:25, 953.45 examples/s]24343 examples [00:25, 960.81 examples/s]24440 examples [00:25, 962.56 examples/s]24537 examples [00:26, 936.39 examples/s]24636 examples [00:26, 950.46 examples/s]24735 examples [00:26, 961.83 examples/s]24837 examples [00:26, 976.31 examples/s]24938 examples [00:26, 984.19 examples/s]25037 examples [00:26, 962.22 examples/s]25134 examples [00:26, 963.03 examples/s]25231 examples [00:26, 937.90 examples/s]25326 examples [00:26, 938.38 examples/s]25423 examples [00:26, 943.92 examples/s]25519 examples [00:27, 946.33 examples/s]25617 examples [00:27, 953.94 examples/s]25713 examples [00:27, 939.27 examples/s]25809 examples [00:27, 944.25 examples/s]25908 examples [00:27, 957.37 examples/s]26004 examples [00:27, 955.82 examples/s]26100 examples [00:27, 946.60 examples/s]26196 examples [00:27, 947.60 examples/s]26291 examples [00:27, 931.60 examples/s]26385 examples [00:27, 926.44 examples/s]26482 examples [00:28, 937.69 examples/s]26578 examples [00:28, 942.75 examples/s]26675 examples [00:28, 948.87 examples/s]26770 examples [00:28, 900.54 examples/s]26861 examples [00:28, 895.75 examples/s]26953 examples [00:28, 902.52 examples/s]27049 examples [00:28, 918.87 examples/s]27149 examples [00:28, 939.32 examples/s]27250 examples [00:28, 958.29 examples/s]27347 examples [00:29, 922.87 examples/s]27440 examples [00:29, 916.78 examples/s]27537 examples [00:29, 931.88 examples/s]27639 examples [00:29, 955.18 examples/s]27735 examples [00:29, 938.32 examples/s]27830 examples [00:29, 931.96 examples/s]27924 examples [00:29, 921.68 examples/s]28021 examples [00:29, 933.74 examples/s]28123 examples [00:29, 955.99 examples/s]28225 examples [00:29, 972.71 examples/s]28323 examples [00:30, 953.11 examples/s]28419 examples [00:30, 937.87 examples/s]28514 examples [00:30, 921.20 examples/s]28609 examples [00:30, 927.73 examples/s]28703 examples [00:30, 929.89 examples/s]28800 examples [00:30, 939.27 examples/s]28896 examples [00:30, 944.55 examples/s]28994 examples [00:30, 954.76 examples/s]29096 examples [00:30, 971.44 examples/s]29194 examples [00:30, 943.40 examples/s]29293 examples [00:31, 956.28 examples/s]29394 examples [00:31, 970.10 examples/s]29492 examples [00:31, 953.89 examples/s]29594 examples [00:31, 972.55 examples/s]29692 examples [00:31, 957.89 examples/s]29789 examples [00:31, 917.41 examples/s]29883 examples [00:31, 921.69 examples/s]29976 examples [00:31, 923.04 examples/s]30069 examples [00:31, 856.94 examples/s]30170 examples [00:32, 895.80 examples/s]30262 examples [00:32, 902.71 examples/s]30361 examples [00:32, 927.22 examples/s]30460 examples [00:32, 943.55 examples/s]30555 examples [00:32, 941.34 examples/s]30650 examples [00:32, 935.10 examples/s]30748 examples [00:32, 946.99 examples/s]30844 examples [00:32, 950.31 examples/s]30944 examples [00:32, 963.02 examples/s]31046 examples [00:32, 976.89 examples/s]31144 examples [00:33, 963.11 examples/s]31241 examples [00:33, 945.22 examples/s]31339 examples [00:33, 954.17 examples/s]31436 examples [00:33, 957.14 examples/s]31534 examples [00:33, 963.70 examples/s]31631 examples [00:33, 954.28 examples/s]31727 examples [00:33, 952.93 examples/s]31825 examples [00:33, 959.89 examples/s]31925 examples [00:33, 968.67 examples/s]32022 examples [00:33, 933.24 examples/s]32122 examples [00:34, 952.02 examples/s]32218 examples [00:34, 948.12 examples/s]32314 examples [00:34, 932.94 examples/s]32412 examples [00:34, 945.16 examples/s]32507 examples [00:34, 946.48 examples/s]32602 examples [00:34, 933.98 examples/s]32696 examples [00:34, 902.85 examples/s]32793 examples [00:34, 920.35 examples/s]32890 examples [00:34, 932.66 examples/s]32986 examples [00:35, 938.46 examples/s]33084 examples [00:35, 950.02 examples/s]33180 examples [00:35, 946.93 examples/s]33281 examples [00:35, 964.29 examples/s]33382 examples [00:35, 975.21 examples/s]33480 examples [00:35, 968.61 examples/s]33579 examples [00:35, 972.92 examples/s]33677 examples [00:35, 947.51 examples/s]33772 examples [00:35, 931.15 examples/s]33870 examples [00:35, 943.24 examples/s]33970 examples [00:36, 958.57 examples/s]34067 examples [00:36, 950.45 examples/s]34165 examples [00:36, 956.87 examples/s]34261 examples [00:36, 923.24 examples/s]34356 examples [00:36, 930.97 examples/s]34451 examples [00:36, 933.83 examples/s]34546 examples [00:36, 935.86 examples/s]34640 examples [00:36, 913.45 examples/s]34732 examples [00:36, 912.97 examples/s]34830 examples [00:36, 931.78 examples/s]34929 examples [00:37, 947.50 examples/s]35030 examples [00:37, 963.90 examples/s]35130 examples [00:37, 969.21 examples/s]35231 examples [00:37, 978.92 examples/s]35330 examples [00:37, 961.79 examples/s]35427 examples [00:37, 948.84 examples/s]35523 examples [00:37, 911.29 examples/s]35615 examples [00:37, 894.06 examples/s]35709 examples [00:37, 906.21 examples/s]35803 examples [00:37, 914.87 examples/s]35898 examples [00:38, 924.34 examples/s]35991 examples [00:38, 882.10 examples/s]36090 examples [00:38, 911.63 examples/s]36192 examples [00:38, 938.88 examples/s]36293 examples [00:38, 956.99 examples/s]36393 examples [00:38, 968.51 examples/s]36491 examples [00:38, 964.54 examples/s]36588 examples [00:38, 943.42 examples/s]36691 examples [00:38, 967.58 examples/s]36792 examples [00:39, 979.56 examples/s]36892 examples [00:39, 983.72 examples/s]36991 examples [00:39, 962.89 examples/s]37088 examples [00:39, 940.58 examples/s]37183 examples [00:39, 928.92 examples/s]37277 examples [00:39, 905.90 examples/s]37374 examples [00:39, 922.22 examples/s]37467 examples [00:39, 923.44 examples/s]37560 examples [00:39, 921.17 examples/s]37658 examples [00:39, 936.81 examples/s]37761 examples [00:40, 960.85 examples/s]37858 examples [00:40, 940.97 examples/s]37958 examples [00:40, 955.43 examples/s]38057 examples [00:40, 963.55 examples/s]38156 examples [00:40, 969.92 examples/s]38257 examples [00:40, 978.38 examples/s]38355 examples [00:40, 973.75 examples/s]38453 examples [00:40, 934.04 examples/s]38552 examples [00:40, 950.00 examples/s]38653 examples [00:40, 964.11 examples/s]38755 examples [00:41, 977.84 examples/s]38854 examples [00:41, 966.91 examples/s]38951 examples [00:41, 918.50 examples/s]39044 examples [00:41, 910.24 examples/s]39136 examples [00:41, 896.20 examples/s]39236 examples [00:41, 924.82 examples/s]39338 examples [00:41, 949.18 examples/s]39435 examples [00:41, 954.90 examples/s]39534 examples [00:41, 964.54 examples/s]39632 examples [00:42, 966.80 examples/s]39731 examples [00:42, 972.41 examples/s]39829 examples [00:42, 971.81 examples/s]39927 examples [00:42, 952.49 examples/s]40023 examples [00:42, 865.66 examples/s]40114 examples [00:42, 878.12 examples/s]40203 examples [00:42, 858.75 examples/s]40290 examples [00:42, 846.47 examples/s]40382 examples [00:42, 866.59 examples/s]40473 examples [00:42, 877.91 examples/s]40562 examples [00:43, 867.01 examples/s]40651 examples [00:43, 865.75 examples/s]40738 examples [00:43, 862.17 examples/s]40834 examples [00:43, 887.60 examples/s]40926 examples [00:43, 895.29 examples/s]41016 examples [00:43, 890.79 examples/s]41106 examples [00:43, 871.49 examples/s]41194 examples [00:43, 844.23 examples/s]41291 examples [00:43, 877.51 examples/s]41390 examples [00:44, 908.21 examples/s]41486 examples [00:44, 920.93 examples/s]41586 examples [00:44, 941.00 examples/s]41681 examples [00:44, 933.00 examples/s]41775 examples [00:44, 932.23 examples/s]41869 examples [00:44, 929.04 examples/s]41963 examples [00:44, 917.01 examples/s]42055 examples [00:44, 908.35 examples/s]42146 examples [00:44, 877.04 examples/s]42241 examples [00:44, 896.15 examples/s]42340 examples [00:45, 920.02 examples/s]42441 examples [00:45, 943.01 examples/s]42536 examples [00:45, 921.83 examples/s]42629 examples [00:45, 919.58 examples/s]42722 examples [00:45, 909.64 examples/s]42814 examples [00:45, 910.46 examples/s]42913 examples [00:45, 932.83 examples/s]43013 examples [00:45, 950.95 examples/s]43112 examples [00:45, 960.90 examples/s]43211 examples [00:45, 968.58 examples/s]43313 examples [00:46, 982.55 examples/s]43412 examples [00:46, 968.27 examples/s]43509 examples [00:46, 904.55 examples/s]43601 examples [00:46, 906.66 examples/s]43695 examples [00:46, 914.39 examples/s]43787 examples [00:46, 906.96 examples/s]43884 examples [00:46, 924.71 examples/s]43978 examples [00:46, 927.43 examples/s]44073 examples [00:46, 932.34 examples/s]44170 examples [00:47, 942.43 examples/s]44266 examples [00:47, 946.91 examples/s]44366 examples [00:47, 961.94 examples/s]44464 examples [00:47, 966.00 examples/s]44561 examples [00:47, 960.81 examples/s]44658 examples [00:47, 959.40 examples/s]44755 examples [00:47, 961.48 examples/s]44852 examples [00:47, 952.52 examples/s]44948 examples [00:47, 938.15 examples/s]45043 examples [00:47, 940.48 examples/s]45138 examples [00:48, 918.06 examples/s]45230 examples [00:48, 863.78 examples/s]45320 examples [00:48, 872.92 examples/s]45421 examples [00:48, 908.36 examples/s]45513 examples [00:48, 905.07 examples/s]45612 examples [00:48, 928.66 examples/s]45713 examples [00:48, 949.91 examples/s]45810 examples [00:48, 954.81 examples/s]45912 examples [00:48, 971.16 examples/s]46010 examples [00:48, 970.03 examples/s]46112 examples [00:49, 982.62 examples/s]46213 examples [00:49, 990.13 examples/s]46314 examples [00:49, 995.74 examples/s]46415 examples [00:49, 998.34 examples/s]46515 examples [00:49, 942.51 examples/s]46610 examples [00:49, 925.58 examples/s]46708 examples [00:49, 940.88 examples/s]46809 examples [00:49, 958.59 examples/s]46906 examples [00:49, 930.60 examples/s]47007 examples [00:50, 951.32 examples/s]47108 examples [00:50, 967.62 examples/s]47206 examples [00:50, 953.07 examples/s]47306 examples [00:50, 965.37 examples/s]47403 examples [00:50, 951.78 examples/s]47499 examples [00:50, 926.90 examples/s]47592 examples [00:50, 918.90 examples/s]47685 examples [00:50, 914.87 examples/s]47780 examples [00:50, 923.77 examples/s]47874 examples [00:50, 928.20 examples/s]47967 examples [00:51, 928.25 examples/s]48061 examples [00:51, 931.01 examples/s]48155 examples [00:51, 898.31 examples/s]48248 examples [00:51, 903.01 examples/s]48351 examples [00:51, 936.36 examples/s]48449 examples [00:51, 947.87 examples/s]48545 examples [00:51, 924.32 examples/s]48638 examples [00:51, 911.58 examples/s]48732 examples [00:51, 919.68 examples/s]48825 examples [00:51, 910.51 examples/s]48922 examples [00:52, 925.09 examples/s]49020 examples [00:52, 939.86 examples/s]49115 examples [00:52, 942.20 examples/s]49216 examples [00:52, 959.73 examples/s]49318 examples [00:52, 975.52 examples/s]49419 examples [00:52, 983.55 examples/s]49520 examples [00:52, 991.28 examples/s]49620 examples [00:52, 982.05 examples/s]49719 examples [00:52, 979.17 examples/s]49817 examples [00:52, 950.71 examples/s]49916 examples [00:53, 960.09 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6594/50000 [00:00<00:00, 65938.18 examples/s] 34%|███▍      | 17009/50000 [00:00<00:00, 74093.34 examples/s] 57%|█████▋    | 28275/50000 [00:00<00:00, 82573.17 examples/s] 79%|███████▊  | 39326/50000 [00:00<00:00, 89348.76 examples/s]                                                               0 examples [00:00, ? examples/s]79 examples [00:00, 789.33 examples/s]169 examples [00:00, 818.21 examples/s]265 examples [00:00, 854.80 examples/s]367 examples [00:00, 897.12 examples/s]468 examples [00:00, 927.78 examples/s]564 examples [00:00, 936.90 examples/s]663 examples [00:00, 949.45 examples/s]752 examples [00:00, 898.86 examples/s]846 examples [00:00, 910.38 examples/s]943 examples [00:01, 927.20 examples/s]1040 examples [00:01, 934.71 examples/s]1139 examples [00:01, 950.42 examples/s]1234 examples [00:01, 921.35 examples/s]1330 examples [00:01, 932.58 examples/s]1429 examples [00:01, 948.65 examples/s]1524 examples [00:01, 931.07 examples/s]1625 examples [00:01, 951.34 examples/s]1727 examples [00:01, 969.76 examples/s]1826 examples [00:01, 974.41 examples/s]1928 examples [00:02, 984.97 examples/s]2030 examples [00:02, 994.98 examples/s]2130 examples [00:02, 928.54 examples/s]2224 examples [00:02, 919.68 examples/s]2317 examples [00:02, 917.68 examples/s]2411 examples [00:02, 923.34 examples/s]2512 examples [00:02, 947.55 examples/s]2608 examples [00:02, 945.12 examples/s]2704 examples [00:02, 949.15 examples/s]2800 examples [00:02, 903.17 examples/s]2891 examples [00:03, 882.62 examples/s]2980 examples [00:03, 873.56 examples/s]3074 examples [00:03, 891.49 examples/s]3168 examples [00:03, 904.35 examples/s]3264 examples [00:03, 918.88 examples/s]3365 examples [00:03, 942.88 examples/s]3465 examples [00:03, 957.64 examples/s]3567 examples [00:03, 972.96 examples/s]3667 examples [00:03, 980.18 examples/s]3766 examples [00:04, 896.84 examples/s]3858 examples [00:04, 891.27 examples/s]3950 examples [00:04, 897.67 examples/s]4044 examples [00:04, 907.83 examples/s]4136 examples [00:04, 910.80 examples/s]4228 examples [00:04, 888.06 examples/s]4318 examples [00:04, 857.36 examples/s]4411 examples [00:04, 877.72 examples/s]4512 examples [00:04, 911.58 examples/s]4604 examples [00:04, 911.09 examples/s]4703 examples [00:05, 931.35 examples/s]4804 examples [00:05, 953.08 examples/s]4900 examples [00:05, 937.46 examples/s]4995 examples [00:05, 910.51 examples/s]5094 examples [00:05, 932.54 examples/s]5188 examples [00:05, 934.14 examples/s]5282 examples [00:05, 924.16 examples/s]5382 examples [00:05, 944.65 examples/s]5484 examples [00:05, 963.69 examples/s]5583 examples [00:05, 968.96 examples/s]5686 examples [00:06, 984.39 examples/s]5786 examples [00:06, 987.30 examples/s]5886 examples [00:06, 989.96 examples/s]5988 examples [00:06, 996.84 examples/s]6088 examples [00:06, 983.96 examples/s]6187 examples [00:06, 904.58 examples/s]6289 examples [00:06, 935.46 examples/s]6391 examples [00:06, 957.22 examples/s]6493 examples [00:06, 972.83 examples/s]6592 examples [00:07, 929.67 examples/s]6694 examples [00:07, 953.90 examples/s]6793 examples [00:07, 962.31 examples/s]6891 examples [00:07, 966.69 examples/s]6991 examples [00:07, 976.09 examples/s]7091 examples [00:07, 982.74 examples/s]7190 examples [00:07, 984.68 examples/s]7292 examples [00:07, 994.93 examples/s]7393 examples [00:07, 997.12 examples/s]7494 examples [00:07, 1000.27 examples/s]7595 examples [00:08, 998.21 examples/s] 7696 examples [00:08, 1000.56 examples/s]7797 examples [00:08, 1003.15 examples/s]7898 examples [00:08, 1002.14 examples/s]7999 examples [00:08, 967.04 examples/s] 8096 examples [00:08, 939.31 examples/s]8191 examples [00:08, 915.80 examples/s]8289 examples [00:08, 926.11 examples/s]8382 examples [00:08, 924.34 examples/s]8475 examples [00:08, 919.72 examples/s]8568 examples [00:09, 920.63 examples/s]8661 examples [00:09, 914.23 examples/s]8764 examples [00:09, 944.04 examples/s]8859 examples [00:09, 925.15 examples/s]8952 examples [00:09, 904.18 examples/s]9052 examples [00:09, 928.44 examples/s]9155 examples [00:09, 955.16 examples/s]9255 examples [00:09, 966.55 examples/s]9357 examples [00:09, 981.64 examples/s]9460 examples [00:10, 993.73 examples/s]9560 examples [00:10, 994.87 examples/s]9663 examples [00:10, 1004.39 examples/s]9764 examples [00:10, 1003.02 examples/s]9865 examples [00:10, 978.35 examples/s] 9964 examples [00:10, 959.86 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete0ZNHZQ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete0ZNHZQ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff4c4567488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff4c4567488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff4c4567488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4c1791588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff45155da90>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff4c4567488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff4c4567488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff4c4567488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4c1abc438>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4c1abc0f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7ff44b8341e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7ff44b8341e0> 

  function with postional parmater data_info <function split_train_valid at 0x7ff44b8341e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7ff44b8342f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7ff44b8342f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7ff44b8342f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=6e5584bf23cd28e01b81f64dec79766097eebf4d4b322a47db6d97df5339d83d
  Stored in directory: /tmp/pip-ephem-wheel-cache-w4skh1lc/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<19:56:00, 12.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:11:02, 16.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:58:57, 24.0kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:59:46, 34.2kB/s].vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:01<4:53:06, 48.8kB/s].vector_cache/glove.6B.zip:   1%|          | 7.64M/862M [00:01<3:24:18, 69.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:22:13, 99.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.8M/862M [00:01<1:39:12, 142kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.8M/862M [00:01<1:09:07, 203kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.6M/862M [00:01<48:16, 289kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.4M/862M [00:01<33:41, 412kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.0M/862M [00:01<23:35, 585kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.7M/862M [00:02<16:30, 831kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.5M/862M [00:02<11:36, 1.18MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.3M/862M [00:02<08:09, 1.66MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.0M/862M [00:02<05:48, 2.33MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:02<05:35, 2.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<05:48, 2.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<07:54, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:04<06:21, 2.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:05<04:43, 2.84MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<06:29, 2.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<06:07, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:06<04:40, 2.85MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<05:59, 2.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<06:59, 1.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:08<05:29, 2.42MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<04:00, 3.31MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<10:06, 1.31MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<08:24, 1.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:10<06:12, 2.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<07:26, 1.77MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<06:31, 2.01MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:12<04:50, 2.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<06:29, 2.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<05:51, 2.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:14<04:22, 2.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<06:10, 2.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<06:56, 1.88MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:16<05:24, 2.40MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:16<03:56, 3.29MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<11:22, 1.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<09:15, 1.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:18<06:48, 1.90MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<07:47, 1.65MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<06:44, 1.91MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:20<04:59, 2.58MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:21<04:18, 2.98MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<9:31:57, 22.4kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<6:40:40, 32.0kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:22<4:39:33, 45.6kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<3:30:06, 60.7kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<2:29:35, 85.2kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<1:45:14, 121kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<1:15:26, 168kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<54:08, 234kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<38:09, 332kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<29:32, 427kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<23:12, 543kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<16:51, 747kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<11:53, 1.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<12:41:59, 16.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<8:55:47, 23.4kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<6:15:13, 33.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<4:22:13, 47.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<3:07:47, 66.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<2:12:38, 94.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<1:32:58, 134kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<1:07:48, 183kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<49:58, 248kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<35:34, 348kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<26:48, 460kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<19:59, 617kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<14:16, 862kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<12:50, 955kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<11:35, 1.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:39, 1.41MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<06:12, 1.97MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<10:30, 1.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:36, 1.42MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:16, 1.94MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<07:14, 1.68MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<07:30, 1.61MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:47, 2.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:13, 2.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<08:11, 1.47MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:46, 1.78MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<05:02, 2.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<06:21, 1.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:51, 1.75MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:19, 2.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<03:51, 3.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<11:44, 1.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<09:25, 1.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<06:50, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<07:34, 1.56MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<06:30, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:50, 2.44MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:10, 1.91MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:43, 1.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:17, 2.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:35, 2.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:05, 2.30MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:51, 3.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:25, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<04:56, 2.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<03:42, 3.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:21, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:05, 1.90MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<04:46, 2.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:31, 3.27MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<06:36, 1.74MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:47, 1.99MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<04:20, 2.65MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:43, 2.00MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:18, 1.81MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<04:59, 2.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:37, 3.13MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<1:32:52, 123kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<1:05:05, 174kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<47:40, 237kB/s]  .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<34:27, 328kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<24:21, 463kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<19:39, 572kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<16:01, 701kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<11:43, 957kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<08:17, 1.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<32:10, 347kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<24:46, 451kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<17:48, 626kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<12:35, 883kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<09:59, 1.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<8:18:32, 22.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<5:49:09, 31.8kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<4:03:25, 45.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<3:08:51, 58.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<2:14:23, 82.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<1:34:30, 117kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<1:07:38, 162kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<49:38, 221kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<35:17, 310kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<24:44, 441kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<50:04, 218kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<36:09, 301kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<25:31, 426kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<20:19, 533kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<16:24, 660kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<11:58, 903kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<08:26, 1.27MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<48:13, 223kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<34:38, 310kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<24:28, 438kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<19:36, 545kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<14:48, 722kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<10:34, 1.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<09:54, 1.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<09:05, 1.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:50, 1.55MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<04:51, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<1:00:02, 176kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<43:05, 245kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<30:19, 347kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<23:37, 444kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<17:34, 596kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<12:32, 834kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<11:13, 928kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<08:55, 1.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:29, 1.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:58, 1.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<06:58, 1.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:23, 1.92MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:52, 2.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<9:56:55, 17.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<6:58:38, 24.5kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<4:52:35, 35.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<3:26:27, 49.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<2:26:32, 69.7kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<1:42:57, 99.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<1:13:21, 138kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<52:20, 194kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<36:48, 275kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<28:02, 359kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<20:37, 488kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<14:36, 687kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<12:35, 795kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<10:49, 924kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<08:00, 1.25MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<05:42, 1.74MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<11:49, 841kB/s] .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<09:07, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<06:39, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<06:57, 1.42MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<06:55, 1.42MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<05:20, 1.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:18, 1.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:42, 2.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:32, 2.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:45, 2.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:17, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:07, 2.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<03:02, 3.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:27, 1.77MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:49, 2.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:36, 2.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:46, 2.01MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:16, 1.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:05, 2.34MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<02:59, 3.19MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<06:41, 1.43MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<05:38, 1.69MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:09, 2.28MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:07, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:31, 2.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:23, 2.78MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<02:51, 3.29MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<7:29:16, 20.9kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<5:14:33, 29.8kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<3:39:08, 42.6kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<2:45:30, 56.3kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<1:57:41, 79.2kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<1:22:43, 112kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<59:06, 157kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<42:19, 219kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<29:47, 310kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<22:53, 401kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<17:52, 514kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<12:54, 711kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:06<09:05, 1.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<16:08, 565kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<12:04, 755kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<08:39, 1.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<08:11, 1.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<07:33, 1.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:44, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<05:27, 1.65MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:44, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:29, 2.56MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<02:36, 3.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<07:50, 1.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<06:43, 1.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:59, 1.78MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<05:09, 1.71MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<06:08, 1.44MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:49, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<03:33, 2.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:40, 1.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:09, 2.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:07, 2.80MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:14, 2.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<03:50, 2.26MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<02:54, 2.99MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:04, 2.12MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:35, 1.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:35, 2.40MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<02:35, 3.31MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<22:02, 389kB/s] .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<16:18, 525kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<11:35, 736kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<10:04, 844kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<08:45, 970kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:33, 1.29MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<05:55, 1.42MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<05:00, 1.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:42, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:32, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:52, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:47, 2.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<02:43, 3.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<10:57, 757kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<08:30, 973kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<06:06, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:12, 1.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:02, 1.63MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:43, 2.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<02:41, 3.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<32:41, 249kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<23:41, 344kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<16:44, 485kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<13:35, 595kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<10:19, 783kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<07:24, 1.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<07:03, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<06:33, 1.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<04:56, 1.62MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<03:32, 2.25MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<07:52, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<06:19, 1.26MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<04:35, 1.72MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:02, 1.56MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:19, 1.82MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<03:12, 2.44MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:05, 1.91MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:39, 2.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:44, 2.83MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:44, 2.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:24, 2.27MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:34, 3.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:36, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:18, 2.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:29, 3.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<02:11, 3.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<5:47:07, 21.9kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<4:02:58, 31.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<2:49:02, 44.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<2:10:04, 58.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<1:32:34, 81.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<1:05:03, 116kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<45:23, 165kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<37:13, 201kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<26:49, 278kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<18:52, 394kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<14:50, 499kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<11:08, 664kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<07:57, 926kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<07:16, 1.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<06:34, 1.11MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:57, 1.47MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:37, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<03:59, 1.82MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:58, 2.44MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:44, 1.92MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:04, 1.77MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:12, 2.23MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<02:19, 3.06MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<59:58, 119kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<42:41, 167kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<29:57, 237kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<22:30, 314kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<16:19, 432kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<11:35, 607kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<08:09, 856kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<57:54, 121kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<41:12, 170kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<28:54, 241kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<21:46, 318kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<16:37, 416kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<11:57, 578kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<08:23, 817kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<1:02:26, 110kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<44:21, 154kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<31:05, 219kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<23:14, 292kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<17:38, 385kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<12:37, 536kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<08:51, 759kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<11:25, 588kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<08:41, 773kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<06:13, 1.07MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<05:54, 1.13MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<05:29, 1.21MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:07, 1.61MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:58, 2.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:46, 1.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<04:00, 1.64MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:57, 2.21MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:35, 1.81MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:50, 1.70MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:59, 2.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:08, 3.00MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<20:23, 316kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<14:55, 431kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<10:33, 608kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<08:50, 721kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<07:29, 851kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<05:31, 1.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<03:53, 1.62MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<25:57, 243kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<18:48, 335kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<13:16, 473kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<10:41, 584kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<08:44, 713kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<06:25, 968kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<04:34, 1.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<05:36, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:27, 1.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<03:16, 1.88MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:32<02:21, 2.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<13:10, 463kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<09:50, 619kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<06:59, 868kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<06:18, 956kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<05:38, 1.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:11, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<03:01, 1.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:11, 1.42MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:33, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:37, 2.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:08, 2.76MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<4:41:11, 21.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<3:16:42, 29.9kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<2:16:33, 42.7kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<1:47:00, 54.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<1:16:02, 76.6kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<53:24, 109kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<37:59, 152kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<27:09, 212kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<19:04, 300kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<14:35, 390kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<10:46, 528kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<07:39, 739kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<06:39, 845kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<05:47, 971kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:17, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:48<03:03, 1.82MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:41, 975kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:33, 1.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:17, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:34, 1.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:36, 1.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:46, 1.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<01:59, 2.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<1:03:59, 84.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<45:17, 119kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<31:41, 170kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<23:17, 230kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<17:22, 308kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<12:23, 430kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<08:40, 610kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<09:55, 532kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<07:28, 705kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<05:20, 981kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<04:56, 1.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:03, 1.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:58, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<03:09, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:27, 1.49MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:39, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<01:55, 2.64MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<03:09, 1.61MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:44, 1.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:01, 2.49MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:33, 1.95MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:48, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:10, 2.29MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:34, 3.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<04:20, 1.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:32, 1.39MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:34, 1.90MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:56, 1.66MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:32, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:52, 2.58MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:26, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:12, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:39, 2.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:16, 2.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:33, 1.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<01:59, 2.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:26, 3.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:20, 1.39MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:49, 1.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<02:04, 2.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:30, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<03:55, 1.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<03:02, 1.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<02:19, 1.97MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:23, 1.90MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:07, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<01:34, 2.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:08, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:24, 1.85MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:54, 2.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:01, 2.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<01:52, 2.34MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:23, 3.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:58, 2.18MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:15, 1.91MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:45, 2.44MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<01:16, 3.33MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:22, 1.26MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:48, 1.51MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<02:03, 2.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:24, 1.74MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:31, 1.65MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:56, 2.14MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:25, 2.89MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:20, 3.07MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<2:55:44, 23.4kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<2:02:51, 33.4kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<1:25:00, 47.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<1:04:41, 62.5kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<46:03, 87.8kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<32:20, 125kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:33<22:23, 178kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<4:05:44, 16.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<2:52:09, 23.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<1:59:49, 32.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<1:24:01, 46.5kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<59:06, 66.0kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<41:12, 94.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<29:29, 130kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<20:59, 183kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<14:40, 259kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<11:03, 341kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<08:30, 443kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<06:07, 613kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<04:16, 868kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<3:37:55, 17.0kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<2:32:39, 24.2kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<1:46:12, 34.6kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<1:14:28, 48.8kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<52:49, 68.7kB/s]  .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<37:00, 97.8kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:45<25:47, 139kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<18:58, 188kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<13:37, 261kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<09:33, 370kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<07:26, 470kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<05:54, 591kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<04:16, 815kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<03:00, 1.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<04:02, 848kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:07, 1.10MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:14, 1.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:36, 2.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<07:29, 449kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<05:48, 577kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<04:13, 791kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<02:58, 1.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<03:33, 924kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:49, 1.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:02, 1.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:10, 1.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:09, 1.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:40, 1.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<01:11, 2.66MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<3:03:00, 17.2kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<2:08:09, 24.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<1:29:03, 35.1kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<1:02:20, 49.5kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<44:12, 69.7kB/s]  .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<30:57, 99.1kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<21:21, 141kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<24:17, 124kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<17:16, 174kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<12:03, 248kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<09:01, 326kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:04<06:55, 425kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<04:57, 592kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<03:27, 837kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<04:11, 686kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<03:11, 903kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<02:15, 1.26MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<01:36, 1.75MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<12:25, 226kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<08:58, 313kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<06:17, 442kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<04:59, 549kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<04:02, 677kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<02:57, 922kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:10<02:03, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<2:36:10, 17.1kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<1:49:20, 24.4kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<1:15:51, 34.8kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<52:59, 49.2kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<37:34, 69.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<26:17, 98.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:14<18:06, 140kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<15:19, 166kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<10:56, 231kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<07:38, 328kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<05:51, 422kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<04:35, 537kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<03:19, 739kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<02:22, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<1:49:25, 21.9kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<1:16:19, 31.3kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:20<52:20, 44.7kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<39:45, 58.7kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<28:16, 82.4kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<19:47, 117kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<13:54, 163kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<10:12, 222kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<07:12, 312kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<04:59, 443kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<04:36, 476kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:27, 635kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<02:26, 886kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:10, 975kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:57, 1.09MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:27, 1.45MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<01:01, 2.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:38, 779kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:01, 1.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<01:27, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:28, 1.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:13, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:53, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:04, 1.79MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:08, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:52, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:37, 2.99MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:17, 1.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:05, 1.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:48, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:58, 1.84MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:51, 2.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:38, 2.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:50, 2.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:56, 1.82MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:43, 2.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:31, 3.20MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<01:07, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:57, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:41, 2.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:51, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:43, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:31, 2.90MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:23, 3.92MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<09:23, 161kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<06:41, 224kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<04:38, 318kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<03:30, 411kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<02:44, 525kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:57, 726kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<01:21, 1.02MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<01:56, 707kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<01:29, 914kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:03, 1.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:01, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:59, 1.32MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:44, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:31, 2.40MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:54, 1.37MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:45, 1.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:32, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:38, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:41, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:31, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:22, 3.01MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:36, 684kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<01:13, 886kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:51, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:49, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:40, 1.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:29, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:34, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:36, 1.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:27, 2.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:27, 1.98MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:24, 2.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:17, 2.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:23, 2.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:26, 1.86MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:20, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:20, 2.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:19, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:13, 3.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:18, 2.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:21, 1.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:16, 2.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:11, 3.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:25, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:21, 1.71MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:15, 2.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:17, 1.86MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:19, 1.72MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:14, 2.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:13, 2.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:09, 2.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:11, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:12, 1.91MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:09, 2.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:06, 3.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:31, 657kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:23, 856kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:15, 1.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:13, 1.21MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:12, 1.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:09, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:05, 2.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:10, 1.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:08, 1.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:05, 1.92MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.60MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:25<00:01, 2.84MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:31, 133kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:20, 187kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:08, 265kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 348kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<15:37:18,  7.11it/s]  0%|          | 653/400000 [00:00<10:55:21, 10.16it/s]  0%|          | 1402/400000 [00:00<7:38:09, 14.50it/s]  1%|          | 2055/400000 [00:00<5:20:29, 20.69it/s]  1%|          | 2753/400000 [00:00<3:44:13, 29.53it/s]  1%|          | 3440/400000 [00:00<2:36:58, 42.10it/s]  1%|          | 4118/400000 [00:00<1:49:59, 59.99it/s]  1%|          | 4876/400000 [00:00<1:17:06, 85.41it/s]  1%|▏         | 5651/400000 [00:00<54:07, 121.44it/s]   2%|▏         | 6414/400000 [00:01<38:04, 172.30it/s]  2%|▏         | 7167/400000 [00:01<26:51, 243.76it/s]  2%|▏         | 7906/400000 [00:01<19:01, 343.37it/s]  2%|▏         | 8672/400000 [00:01<13:33, 481.28it/s]  2%|▏         | 9447/400000 [00:01<09:43, 669.71it/s]  3%|▎         | 10209/400000 [00:01<07:02, 922.00it/s]  3%|▎         | 10963/400000 [00:01<05:12, 1245.27it/s]  3%|▎         | 11738/400000 [00:01<03:53, 1664.28it/s]  3%|▎         | 12484/400000 [00:01<02:59, 2159.32it/s]  3%|▎         | 13261/400000 [00:01<02:20, 2756.11it/s]  4%|▎         | 14017/400000 [00:02<01:53, 3405.03it/s]  4%|▎         | 14767/400000 [00:02<01:35, 4041.38it/s]  4%|▍         | 15507/400000 [00:02<01:23, 4622.96it/s]  4%|▍         | 16259/400000 [00:02<01:13, 5226.84it/s]  4%|▍         | 17024/400000 [00:02<01:06, 5774.81it/s]  4%|▍         | 17768/400000 [00:02<01:02, 6155.11it/s]  5%|▍         | 18507/400000 [00:02<00:58, 6474.35it/s]  5%|▍         | 19245/400000 [00:02<00:56, 6703.44it/s]  5%|▍         | 19981/400000 [00:02<00:55, 6865.39it/s]  5%|▌         | 20743/400000 [00:02<00:53, 7075.54it/s]  5%|▌         | 21504/400000 [00:03<00:52, 7226.30it/s]  6%|▌         | 22252/400000 [00:03<00:51, 7267.30it/s]  6%|▌         | 22997/400000 [00:03<00:51, 7284.86it/s]  6%|▌         | 23738/400000 [00:03<00:51, 7263.10it/s]  6%|▌         | 24497/400000 [00:03<00:51, 7357.33it/s]  6%|▋         | 25258/400000 [00:03<00:50, 7428.97it/s]  7%|▋         | 26006/400000 [00:03<00:50, 7345.94it/s]  7%|▋         | 26767/400000 [00:03<00:50, 7415.60it/s]  7%|▋         | 27525/400000 [00:03<00:50, 7407.97it/s]  7%|▋         | 28268/400000 [00:04<00:52, 7053.02it/s]  7%|▋         | 28979/400000 [00:04<00:53, 6959.10it/s]  7%|▋         | 29696/400000 [00:04<00:52, 7020.12it/s]  8%|▊         | 30422/400000 [00:04<00:52, 7090.31it/s]  8%|▊         | 31188/400000 [00:04<00:50, 7251.11it/s]  8%|▊         | 31964/400000 [00:04<00:49, 7395.51it/s]  8%|▊         | 32717/400000 [00:04<00:49, 7435.19it/s]  8%|▊         | 33487/400000 [00:04<00:48, 7510.02it/s]  9%|▊         | 34255/400000 [00:04<00:48, 7558.63it/s]  9%|▉         | 35025/400000 [00:04<00:48, 7599.10it/s]  9%|▉         | 35786/400000 [00:05<00:48, 7444.51it/s]  9%|▉         | 36532/400000 [00:05<00:51, 7013.56it/s]  9%|▉         | 37240/400000 [00:05<00:52, 6945.07it/s]  9%|▉         | 37940/400000 [00:05<00:52, 6960.08it/s] 10%|▉         | 38670/400000 [00:05<00:51, 7057.17it/s] 10%|▉         | 39395/400000 [00:05<00:50, 7112.79it/s] 10%|█         | 40109/400000 [00:05<00:51, 6982.66it/s] 10%|█         | 40864/400000 [00:05<00:50, 7143.67it/s] 10%|█         | 41613/400000 [00:05<00:49, 7243.26it/s] 11%|█         | 42396/400000 [00:05<00:48, 7408.95it/s] 11%|█         | 43140/400000 [00:06<00:48, 7404.02it/s] 11%|█         | 43904/400000 [00:06<00:47, 7471.40it/s] 11%|█         | 44678/400000 [00:06<00:47, 7538.40it/s] 11%|█▏        | 45433/400000 [00:06<00:48, 7373.62it/s] 12%|█▏        | 46195/400000 [00:06<00:47, 7445.24it/s] 12%|█▏        | 46981/400000 [00:06<00:46, 7564.40it/s] 12%|█▏        | 47739/400000 [00:06<00:47, 7395.39it/s] 12%|█▏        | 48518/400000 [00:06<00:46, 7507.62it/s] 12%|█▏        | 49287/400000 [00:06<00:46, 7560.63it/s] 13%|█▎        | 50045/400000 [00:06<00:46, 7515.92it/s] 13%|█▎        | 50829/400000 [00:07<00:45, 7609.40it/s] 13%|█▎        | 51591/400000 [00:07<00:47, 7330.12it/s] 13%|█▎        | 52327/400000 [00:07<00:47, 7333.35it/s] 13%|█▎        | 53102/400000 [00:07<00:46, 7451.87it/s] 13%|█▎        | 53850/400000 [00:07<00:46, 7395.76it/s] 14%|█▎        | 54612/400000 [00:07<00:46, 7460.60it/s] 14%|█▍        | 55360/400000 [00:07<00:46, 7351.23it/s] 14%|█▍        | 56133/400000 [00:07<00:46, 7459.37it/s] 14%|█▍        | 56906/400000 [00:07<00:45, 7531.11it/s] 14%|█▍        | 57682/400000 [00:07<00:45, 7595.96it/s] 15%|█▍        | 58443/400000 [00:08<00:44, 7590.16it/s] 15%|█▍        | 59203/400000 [00:08<00:45, 7420.88it/s] 15%|█▍        | 59975/400000 [00:08<00:45, 7507.17it/s] 15%|█▌        | 60727/400000 [00:08<00:45, 7446.05it/s] 15%|█▌        | 61494/400000 [00:08<00:45, 7511.82it/s] 16%|█▌        | 62273/400000 [00:08<00:44, 7592.59it/s] 16%|█▌        | 63034/400000 [00:08<00:44, 7554.32it/s] 16%|█▌        | 63790/400000 [00:08<00:45, 7372.40it/s] 16%|█▌        | 64541/400000 [00:08<00:45, 7413.11it/s] 16%|█▋        | 65318/400000 [00:09<00:44, 7515.08it/s] 17%|█▋        | 66090/400000 [00:09<00:44, 7574.40it/s] 17%|█▋        | 66849/400000 [00:09<00:44, 7540.37it/s] 17%|█▋        | 67604/400000 [00:09<00:44, 7522.16it/s] 17%|█▋        | 68357/400000 [00:09<00:44, 7509.72it/s] 17%|█▋        | 69109/400000 [00:09<00:44, 7454.50it/s] 17%|█▋        | 69868/400000 [00:09<00:44, 7493.01it/s] 18%|█▊        | 70618/400000 [00:09<00:44, 7448.92it/s] 18%|█▊        | 71364/400000 [00:09<00:45, 7222.16it/s] 18%|█▊        | 72147/400000 [00:09<00:44, 7394.15it/s] 18%|█▊        | 72908/400000 [00:10<00:43, 7452.23it/s] 18%|█▊        | 73689/400000 [00:10<00:43, 7553.44it/s] 19%|█▊        | 74446/400000 [00:10<00:43, 7426.18it/s] 19%|█▉        | 75218/400000 [00:10<00:43, 7511.72it/s] 19%|█▉        | 75971/400000 [00:10<00:43, 7495.13it/s] 19%|█▉        | 76722/400000 [00:10<00:44, 7310.80it/s] 19%|█▉        | 77493/400000 [00:10<00:43, 7425.82it/s] 20%|█▉        | 78238/400000 [00:10<00:44, 7297.62it/s] 20%|█▉        | 78970/400000 [00:10<00:45, 7124.28it/s] 20%|█▉        | 79685/400000 [00:10<00:45, 7021.23it/s] 20%|██        | 80389/400000 [00:11<00:45, 6990.47it/s] 20%|██        | 81155/400000 [00:11<00:44, 7177.19it/s] 20%|██        | 81924/400000 [00:11<00:43, 7323.64it/s] 21%|██        | 82702/400000 [00:11<00:42, 7453.45it/s] 21%|██        | 83486/400000 [00:11<00:41, 7564.01it/s] 21%|██        | 84245/400000 [00:11<00:42, 7432.59it/s] 21%|██        | 84991/400000 [00:11<00:43, 7304.22it/s] 21%|██▏       | 85746/400000 [00:11<00:42, 7365.01it/s] 22%|██▏       | 86484/400000 [00:11<00:43, 7208.68it/s] 22%|██▏       | 87247/400000 [00:11<00:42, 7327.45it/s] 22%|██▏       | 87998/400000 [00:12<00:42, 7379.57it/s] 22%|██▏       | 88738/400000 [00:12<00:42, 7305.12it/s] 22%|██▏       | 89470/400000 [00:12<00:43, 7102.80it/s] 23%|██▎       | 90183/400000 [00:12<00:44, 7041.02it/s] 23%|██▎       | 90889/400000 [00:12<00:44, 6978.82it/s] 23%|██▎       | 91589/400000 [00:12<00:44, 6936.39it/s] 23%|██▎       | 92346/400000 [00:12<00:43, 7112.92it/s] 23%|██▎       | 93069/400000 [00:12<00:42, 7147.38it/s] 23%|██▎       | 93838/400000 [00:12<00:41, 7300.55it/s] 24%|██▎       | 94572/400000 [00:13<00:41, 7309.75it/s] 24%|██▍       | 95327/400000 [00:13<00:41, 7378.07it/s] 24%|██▍       | 96066/400000 [00:13<00:41, 7250.63it/s] 24%|██▍       | 96814/400000 [00:13<00:41, 7315.61it/s] 24%|██▍       | 97556/400000 [00:13<00:41, 7345.59it/s] 25%|██▍       | 98333/400000 [00:13<00:40, 7467.80it/s] 25%|██▍       | 99094/400000 [00:13<00:40, 7507.60it/s] 25%|██▍       | 99846/400000 [00:13<00:40, 7460.42it/s] 25%|██▌       | 100593/400000 [00:13<00:42, 7063.44it/s] 25%|██▌       | 101305/400000 [00:13<00:42, 6952.58it/s] 26%|██▌       | 102052/400000 [00:14<00:41, 7098.19it/s] 26%|██▌       | 102828/400000 [00:14<00:40, 7283.89it/s] 26%|██▌       | 103593/400000 [00:14<00:40, 7389.50it/s] 26%|██▌       | 104335/400000 [00:14<00:40, 7344.14it/s] 26%|██▋       | 105072/400000 [00:14<00:41, 7158.94it/s] 26%|██▋       | 105791/400000 [00:14<00:41, 7058.05it/s] 27%|██▋       | 106499/400000 [00:14<00:41, 7033.16it/s] 27%|██▋       | 107204/400000 [00:14<00:43, 6795.04it/s] 27%|██▋       | 107897/400000 [00:14<00:42, 6829.37it/s] 27%|██▋       | 108633/400000 [00:14<00:41, 6979.06it/s] 27%|██▋       | 109401/400000 [00:15<00:40, 7173.42it/s] 28%|██▊       | 110156/400000 [00:15<00:39, 7281.34it/s] 28%|██▊       | 110931/400000 [00:15<00:38, 7415.20it/s] 28%|██▊       | 111675/400000 [00:15<00:38, 7402.50it/s] 28%|██▊       | 112445/400000 [00:15<00:38, 7488.19it/s] 28%|██▊       | 113219/400000 [00:15<00:38, 7536.90it/s] 28%|██▊       | 113974/400000 [00:15<00:38, 7467.09it/s] 29%|██▊       | 114722/400000 [00:15<00:39, 7202.43it/s] 29%|██▉       | 115445/400000 [00:15<00:40, 7077.40it/s] 29%|██▉       | 116156/400000 [00:16<00:40, 7070.25it/s] 29%|██▉       | 116916/400000 [00:16<00:39, 7220.81it/s] 29%|██▉       | 117684/400000 [00:16<00:38, 7350.94it/s] 30%|██▉       | 118463/400000 [00:16<00:37, 7476.03it/s] 30%|██▉       | 119213/400000 [00:16<00:37, 7468.08it/s] 30%|██▉       | 119989/400000 [00:16<00:37, 7552.01it/s] 30%|███       | 120746/400000 [00:16<00:37, 7546.24it/s] 30%|███       | 121513/400000 [00:16<00:36, 7571.92it/s] 31%|███       | 122291/400000 [00:16<00:36, 7632.83it/s] 31%|███       | 123055/400000 [00:16<00:37, 7401.92it/s] 31%|███       | 123798/400000 [00:17<00:39, 7074.36it/s] 31%|███       | 124564/400000 [00:17<00:38, 7238.43it/s] 31%|███▏      | 125340/400000 [00:17<00:37, 7385.99it/s] 32%|███▏      | 126121/400000 [00:17<00:36, 7507.48it/s] 32%|███▏      | 126891/400000 [00:17<00:36, 7563.48it/s] 32%|███▏      | 127657/400000 [00:17<00:35, 7591.09it/s] 32%|███▏      | 128438/400000 [00:17<00:35, 7653.07it/s] 32%|███▏      | 129206/400000 [00:17<00:35, 7659.67it/s] 32%|███▏      | 129981/400000 [00:17<00:35, 7685.25it/s] 33%|███▎      | 130751/400000 [00:17<00:35, 7490.19it/s] 33%|███▎      | 131529/400000 [00:18<00:35, 7572.82it/s] 33%|███▎      | 132296/400000 [00:18<00:35, 7600.66it/s] 33%|███▎      | 133057/400000 [00:18<00:35, 7417.33it/s] 33%|███▎      | 133801/400000 [00:18<00:35, 7419.39it/s] 34%|███▎      | 134570/400000 [00:18<00:35, 7498.34it/s] 34%|███▍      | 135342/400000 [00:18<00:34, 7562.32it/s] 34%|███▍      | 136100/400000 [00:18<00:35, 7493.10it/s] 34%|███▍      | 136869/400000 [00:18<00:34, 7551.10it/s] 34%|███▍      | 137625/400000 [00:18<00:34, 7503.19it/s] 35%|███▍      | 138376/400000 [00:18<00:34, 7503.85it/s] 35%|███▍      | 139158/400000 [00:19<00:34, 7595.91it/s] 35%|███▍      | 139933/400000 [00:19<00:34, 7639.10it/s] 35%|███▌      | 140698/400000 [00:19<00:34, 7532.80it/s] 35%|███▌      | 141473/400000 [00:19<00:34, 7595.11it/s] 36%|███▌      | 142242/400000 [00:19<00:33, 7622.94it/s] 36%|███▌      | 143010/400000 [00:19<00:33, 7639.20it/s] 36%|███▌      | 143775/400000 [00:19<00:33, 7632.06it/s] 36%|███▌      | 144539/400000 [00:19<00:33, 7622.44it/s] 36%|███▋      | 145302/400000 [00:19<00:33, 7606.40it/s] 37%|███▋      | 146063/400000 [00:19<00:33, 7604.29it/s] 37%|███▋      | 146824/400000 [00:20<00:33, 7514.22it/s] 37%|███▋      | 147576/400000 [00:20<00:33, 7432.49it/s] 37%|███▋      | 148339/400000 [00:20<00:33, 7490.52it/s] 37%|███▋      | 149092/400000 [00:20<00:33, 7502.12it/s] 37%|███▋      | 149864/400000 [00:20<00:33, 7565.63it/s] 38%|███▊      | 150621/400000 [00:20<00:33, 7542.67it/s] 38%|███▊      | 151376/400000 [00:20<00:33, 7457.91it/s] 38%|███▊      | 152142/400000 [00:20<00:32, 7512.77it/s] 38%|███▊      | 152926/400000 [00:20<00:32, 7605.10it/s] 38%|███▊      | 153688/400000 [00:20<00:32, 7561.54it/s] 39%|███▊      | 154474/400000 [00:21<00:32, 7648.66it/s] 39%|███▉      | 155245/400000 [00:21<00:31, 7664.22it/s] 39%|███▉      | 156018/400000 [00:21<00:31, 7683.15it/s] 39%|███▉      | 156787/400000 [00:21<00:31, 7635.44it/s] 39%|███▉      | 157551/400000 [00:21<00:32, 7389.44it/s] 40%|███▉      | 158331/400000 [00:21<00:32, 7505.84it/s] 40%|███▉      | 159106/400000 [00:21<00:31, 7576.80it/s] 40%|███▉      | 159874/400000 [00:21<00:31, 7604.59it/s] 40%|████      | 160636/400000 [00:21<00:31, 7598.99it/s] 40%|████      | 161397/400000 [00:21<00:31, 7531.37it/s] 41%|████      | 162151/400000 [00:22<00:31, 7509.00it/s] 41%|████      | 162903/400000 [00:22<00:31, 7417.07it/s] 41%|████      | 163646/400000 [00:22<00:32, 7240.63it/s] 41%|████      | 164372/400000 [00:22<00:33, 7119.47it/s] 41%|████▏     | 165086/400000 [00:22<00:33, 7037.90it/s] 41%|████▏     | 165791/400000 [00:22<00:33, 7007.80it/s] 42%|████▏     | 166493/400000 [00:22<00:33, 6929.44it/s] 42%|████▏     | 167187/400000 [00:22<00:33, 6898.08it/s] 42%|████▏     | 167928/400000 [00:22<00:32, 7044.03it/s] 42%|████▏     | 168676/400000 [00:23<00:32, 7166.97it/s] 42%|████▏     | 169464/400000 [00:23<00:31, 7366.74it/s] 43%|████▎     | 170233/400000 [00:23<00:30, 7459.28it/s] 43%|████▎     | 171008/400000 [00:23<00:30, 7541.79it/s] 43%|████▎     | 171764/400000 [00:23<00:30, 7494.12it/s] 43%|████▎     | 172515/400000 [00:23<00:30, 7370.77it/s] 43%|████▎     | 173254/400000 [00:23<00:31, 7257.32it/s] 43%|████▎     | 173982/400000 [00:23<00:31, 7102.62it/s] 44%|████▎     | 174694/400000 [00:23<00:31, 7041.12it/s] 44%|████▍     | 175474/400000 [00:23<00:30, 7252.44it/s] 44%|████▍     | 176237/400000 [00:24<00:30, 7359.42it/s] 44%|████▍     | 177014/400000 [00:24<00:29, 7475.92it/s] 44%|████▍     | 177764/400000 [00:24<00:30, 7365.69it/s] 45%|████▍     | 178503/400000 [00:24<00:30, 7368.90it/s] 45%|████▍     | 179249/400000 [00:24<00:29, 7394.94it/s] 45%|████▍     | 179990/400000 [00:24<00:30, 7241.15it/s] 45%|████▌     | 180716/400000 [00:24<00:30, 7140.30it/s] 45%|████▌     | 181482/400000 [00:24<00:29, 7288.53it/s] 46%|████▌     | 182254/400000 [00:24<00:29, 7411.53it/s] 46%|████▌     | 183022/400000 [00:24<00:28, 7487.48it/s] 46%|████▌     | 183773/400000 [00:25<00:29, 7209.50it/s] 46%|████▌     | 184539/400000 [00:25<00:29, 7337.21it/s] 46%|████▋     | 185276/400000 [00:25<00:29, 7188.79it/s] 46%|████▋     | 185998/400000 [00:25<00:30, 7102.95it/s] 47%|████▋     | 186725/400000 [00:25<00:29, 7151.99it/s] 47%|████▋     | 187478/400000 [00:25<00:29, 7258.47it/s] 47%|████▋     | 188245/400000 [00:25<00:28, 7375.77it/s] 47%|████▋     | 188985/400000 [00:25<00:28, 7348.08it/s] 47%|████▋     | 189721/400000 [00:25<00:28, 7258.12it/s] 48%|████▊     | 190472/400000 [00:25<00:28, 7330.39it/s] 48%|████▊     | 191206/400000 [00:26<00:29, 7188.18it/s] 48%|████▊     | 191927/400000 [00:26<00:29, 7017.43it/s] 48%|████▊     | 192673/400000 [00:26<00:29, 7142.70it/s] 48%|████▊     | 193435/400000 [00:26<00:28, 7278.11it/s] 49%|████▊     | 194193/400000 [00:26<00:27, 7364.59it/s] 49%|████▊     | 194932/400000 [00:26<00:27, 7352.79it/s] 49%|████▉     | 195669/400000 [00:26<00:27, 7347.76it/s] 49%|████▉     | 196405/400000 [00:26<00:28, 7065.21it/s] 49%|████▉     | 197115/400000 [00:26<00:29, 6986.74it/s] 49%|████▉     | 197864/400000 [00:27<00:28, 7129.92it/s] 50%|████▉     | 198637/400000 [00:27<00:27, 7299.91it/s] 50%|████▉     | 199408/400000 [00:27<00:27, 7418.13it/s] 50%|█████     | 200182/400000 [00:27<00:26, 7510.56it/s] 50%|█████     | 200935/400000 [00:27<00:26, 7387.16it/s] 50%|█████     | 201697/400000 [00:27<00:26, 7454.17it/s] 51%|█████     | 202470/400000 [00:27<00:26, 7534.37it/s] 51%|█████     | 203225/400000 [00:27<00:27, 7286.93it/s] 51%|█████     | 203957/400000 [00:27<00:27, 7214.77it/s] 51%|█████     | 204713/400000 [00:27<00:26, 7314.49it/s] 51%|█████▏    | 205492/400000 [00:28<00:26, 7450.02it/s] 52%|█████▏    | 206239/400000 [00:28<00:26, 7368.97it/s] 52%|█████▏    | 206978/400000 [00:28<00:26, 7238.22it/s] 52%|█████▏    | 207704/400000 [00:28<00:27, 7002.75it/s] 52%|█████▏    | 208408/400000 [00:28<00:28, 6829.83it/s] 52%|█████▏    | 209114/400000 [00:28<00:27, 6895.84it/s] 52%|█████▏    | 209884/400000 [00:28<00:26, 7118.64it/s] 53%|█████▎    | 210622/400000 [00:28<00:26, 7193.36it/s] 53%|█████▎    | 211393/400000 [00:28<00:25, 7339.59it/s] 53%|█████▎    | 212158/400000 [00:28<00:25, 7430.02it/s] 53%|█████▎    | 212912/400000 [00:29<00:25, 7462.41it/s] 53%|█████▎    | 213660/400000 [00:29<00:25, 7272.81it/s] 54%|█████▎    | 214409/400000 [00:29<00:25, 7336.16it/s] 54%|█████▍    | 215181/400000 [00:29<00:24, 7446.93it/s] 54%|█████▍    | 215949/400000 [00:29<00:24, 7515.35it/s] 54%|█████▍    | 216702/400000 [00:29<00:24, 7467.29it/s] 54%|█████▍    | 217450/400000 [00:29<00:24, 7448.56it/s] 55%|█████▍    | 218227/400000 [00:29<00:24, 7542.08it/s] 55%|█████▍    | 218993/400000 [00:29<00:23, 7577.04it/s] 55%|█████▍    | 219752/400000 [00:29<00:23, 7540.03it/s] 55%|█████▌    | 220507/400000 [00:30<00:24, 7470.99it/s] 55%|█████▌    | 221280/400000 [00:30<00:23, 7538.60it/s] 56%|█████▌    | 222056/400000 [00:30<00:23, 7601.30it/s] 56%|█████▌    | 222817/400000 [00:30<00:23, 7594.32it/s] 56%|█████▌    | 223598/400000 [00:30<00:23, 7657.50it/s] 56%|█████▌    | 224376/400000 [00:30<00:22, 7692.63it/s] 56%|█████▋    | 225146/400000 [00:30<00:22, 7684.45it/s] 56%|█████▋    | 225915/400000 [00:30<00:22, 7654.09it/s] 57%|█████▋    | 226681/400000 [00:30<00:22, 7653.78it/s] 57%|█████▋    | 227449/400000 [00:31<00:22, 7660.61it/s] 57%|█████▋    | 228216/400000 [00:31<00:22, 7498.47it/s] 57%|█████▋    | 228986/400000 [00:31<00:22, 7556.37it/s] 57%|█████▋    | 229770/400000 [00:31<00:22, 7637.68it/s] 58%|█████▊    | 230535/400000 [00:31<00:22, 7379.10it/s] 58%|█████▊    | 231276/400000 [00:31<00:23, 7105.49it/s] 58%|█████▊    | 231998/400000 [00:31<00:23, 7138.74it/s] 58%|█████▊    | 232752/400000 [00:31<00:23, 7253.18it/s] 58%|█████▊    | 233528/400000 [00:31<00:22, 7396.28it/s] 59%|█████▊    | 234307/400000 [00:31<00:22, 7507.27it/s] 59%|█████▉    | 235060/400000 [00:32<00:22, 7480.56it/s] 59%|█████▉    | 235858/400000 [00:32<00:21, 7607.34it/s] 59%|█████▉    | 236621/400000 [00:32<00:21, 7511.60it/s] 59%|█████▉    | 237375/400000 [00:32<00:21, 7518.62it/s] 60%|█████▉    | 238158/400000 [00:32<00:21, 7607.66it/s] 60%|█████▉    | 238926/400000 [00:32<00:21, 7627.60it/s] 60%|█████▉    | 239698/400000 [00:32<00:20, 7652.58it/s] 60%|██████    | 240469/400000 [00:32<00:20, 7669.25it/s] 60%|██████    | 241237/400000 [00:32<00:20, 7659.88it/s] 61%|██████    | 242004/400000 [00:32<00:20, 7636.41it/s] 61%|██████    | 242772/400000 [00:33<00:20, 7648.71it/s] 61%|██████    | 243538/400000 [00:33<00:20, 7592.90it/s] 61%|██████    | 244298/400000 [00:33<00:20, 7427.12it/s] 61%|██████▏   | 245042/400000 [00:33<00:20, 7415.00it/s] 61%|██████▏   | 245785/400000 [00:33<00:20, 7352.45it/s] 62%|██████▏   | 246552/400000 [00:33<00:20, 7442.64it/s] 62%|██████▏   | 247297/400000 [00:33<00:20, 7390.68it/s] 62%|██████▏   | 248066/400000 [00:33<00:20, 7476.20it/s] 62%|██████▏   | 248815/400000 [00:33<00:20, 7465.40it/s] 62%|██████▏   | 249595/400000 [00:33<00:19, 7560.31it/s] 63%|██████▎   | 250352/400000 [00:34<00:19, 7561.25it/s] 63%|██████▎   | 251132/400000 [00:34<00:19, 7628.33it/s] 63%|██████▎   | 251896/400000 [00:34<00:19, 7505.78it/s] 63%|██████▎   | 252665/400000 [00:34<00:19, 7558.52it/s] 63%|██████▎   | 253435/400000 [00:34<00:19, 7599.05it/s] 64%|██████▎   | 254215/400000 [00:34<00:19, 7656.65it/s] 64%|██████▎   | 254982/400000 [00:34<00:19, 7401.59it/s] 64%|██████▍   | 255736/400000 [00:34<00:19, 7442.15it/s] 64%|██████▍   | 256482/400000 [00:34<00:19, 7329.51it/s] 64%|██████▍   | 257217/400000 [00:34<00:19, 7282.05it/s] 64%|██████▍   | 257999/400000 [00:35<00:19, 7435.38it/s] 65%|██████▍   | 258781/400000 [00:35<00:18, 7546.41it/s] 65%|██████▍   | 259545/400000 [00:35<00:18, 7572.03it/s] 65%|██████▌   | 260306/400000 [00:35<00:18, 7573.66it/s] 65%|██████▌   | 261082/400000 [00:35<00:18, 7627.04it/s] 65%|██████▌   | 261861/400000 [00:35<00:18, 7672.80it/s] 66%|██████▌   | 262629/400000 [00:35<00:18, 7579.96it/s] 66%|██████▌   | 263402/400000 [00:35<00:17, 7623.95it/s] 66%|██████▌   | 264184/400000 [00:35<00:17, 7680.61it/s] 66%|██████▌   | 264953/400000 [00:35<00:17, 7660.55it/s] 66%|██████▋   | 265732/400000 [00:36<00:17, 7697.08it/s] 67%|██████▋   | 266502/400000 [00:36<00:17, 7554.48it/s] 67%|██████▋   | 267259/400000 [00:36<00:17, 7513.88it/s] 67%|██████▋   | 268026/400000 [00:36<00:17, 7557.74it/s] 67%|██████▋   | 268795/400000 [00:36<00:17, 7595.88it/s] 67%|██████▋   | 269555/400000 [00:36<00:17, 7441.22it/s] 68%|██████▊   | 270301/400000 [00:36<00:17, 7237.91it/s] 68%|██████▊   | 271027/400000 [00:36<00:18, 7067.48it/s] 68%|██████▊   | 271788/400000 [00:36<00:17, 7220.43it/s] 68%|██████▊   | 272566/400000 [00:37<00:17, 7378.55it/s] 68%|██████▊   | 273310/400000 [00:37<00:17, 7394.17it/s] 69%|██████▊   | 274089/400000 [00:37<00:16, 7507.01it/s] 69%|██████▊   | 274842/400000 [00:37<00:16, 7381.63it/s] 69%|██████▉   | 275582/400000 [00:37<00:17, 7237.10it/s] 69%|██████▉   | 276346/400000 [00:37<00:16, 7352.14it/s] 69%|██████▉   | 277083/400000 [00:37<00:16, 7231.98it/s] 69%|██████▉   | 277850/400000 [00:37<00:16, 7357.00it/s] 70%|██████▉   | 278623/400000 [00:37<00:16, 7464.72it/s] 70%|██████▉   | 279385/400000 [00:37<00:16, 7510.13it/s] 70%|███████   | 280160/400000 [00:38<00:15, 7579.34it/s] 70%|███████   | 280919/400000 [00:38<00:15, 7513.41it/s] 70%|███████   | 281672/400000 [00:38<00:15, 7415.47it/s] 71%|███████   | 282431/400000 [00:38<00:15, 7466.26it/s] 71%|███████   | 283209/400000 [00:38<00:15, 7555.38it/s] 71%|███████   | 283995/400000 [00:38<00:15, 7641.60it/s] 71%|███████   | 284760/400000 [00:38<00:15, 7606.53it/s] 71%|███████▏  | 285535/400000 [00:38<00:14, 7647.68it/s] 72%|███████▏  | 286301/400000 [00:38<00:15, 7397.66it/s] 72%|███████▏  | 287043/400000 [00:38<00:15, 7215.08it/s] 72%|███████▏  | 287767/400000 [00:39<00:15, 7091.39it/s] 72%|███████▏  | 288512/400000 [00:39<00:15, 7194.66it/s] 72%|███████▏  | 289260/400000 [00:39<00:15, 7276.78it/s] 72%|███████▏  | 289990/400000 [00:39<00:15, 7115.62it/s] 73%|███████▎  | 290757/400000 [00:39<00:15, 7271.21it/s] 73%|███████▎  | 291487/400000 [00:39<00:15, 7081.39it/s] 73%|███████▎  | 292198/400000 [00:39<00:15, 6931.54it/s] 73%|███████▎  | 292894/400000 [00:39<00:15, 6880.83it/s] 73%|███████▎  | 293662/400000 [00:39<00:14, 7101.19it/s] 74%|███████▎  | 294407/400000 [00:40<00:14, 7200.18it/s] 74%|███████▍  | 295160/400000 [00:40<00:14, 7295.56it/s] 74%|███████▍  | 295913/400000 [00:40<00:14, 7364.14it/s] 74%|███████▍  | 296675/400000 [00:40<00:13, 7437.87it/s] 74%|███████▍  | 297421/400000 [00:40<00:14, 7256.55it/s] 75%|███████▍  | 298207/400000 [00:40<00:13, 7425.44it/s] 75%|███████▍  | 298952/400000 [00:40<00:13, 7413.11it/s] 75%|███████▍  | 299695/400000 [00:40<00:14, 7114.05it/s] 75%|███████▌  | 300411/400000 [00:40<00:14, 7085.79it/s] 75%|███████▌  | 301184/400000 [00:40<00:13, 7264.98it/s] 75%|███████▌  | 301968/400000 [00:41<00:13, 7427.07it/s] 76%|███████▌  | 302732/400000 [00:41<00:12, 7487.38it/s] 76%|███████▌  | 303483/400000 [00:41<00:13, 7318.31it/s] 76%|███████▌  | 304218/400000 [00:41<00:13, 7101.14it/s] 76%|███████▌  | 304988/400000 [00:41<00:13, 7270.17it/s] 76%|███████▋  | 305759/400000 [00:41<00:12, 7396.59it/s] 77%|███████▋  | 306526/400000 [00:41<00:12, 7475.74it/s] 77%|███████▋  | 307291/400000 [00:41<00:12, 7524.41it/s] 77%|███████▋  | 308060/400000 [00:41<00:12, 7572.00it/s] 77%|███████▋  | 308842/400000 [00:41<00:11, 7642.73it/s] 77%|███████▋  | 309633/400000 [00:42<00:11, 7720.18it/s] 78%|███████▊  | 310413/400000 [00:42<00:11, 7743.41it/s] 78%|███████▊  | 311188/400000 [00:42<00:11, 7632.89it/s] 78%|███████▊  | 311953/400000 [00:42<00:11, 7420.85it/s] 78%|███████▊  | 312697/400000 [00:42<00:12, 7252.13it/s] 78%|███████▊  | 313425/400000 [00:42<00:12, 7154.14it/s] 79%|███████▊  | 314143/400000 [00:42<00:12, 7085.84it/s] 79%|███████▊  | 314910/400000 [00:42<00:11, 7248.90it/s] 79%|███████▉  | 315673/400000 [00:42<00:11, 7357.20it/s] 79%|███████▉  | 316411/400000 [00:42<00:11, 7285.87it/s] 79%|███████▉  | 317180/400000 [00:43<00:11, 7400.99it/s] 79%|███████▉  | 317939/400000 [00:43<00:11, 7454.17it/s] 80%|███████▉  | 318710/400000 [00:43<00:10, 7528.56it/s] 80%|███████▉  | 319464/400000 [00:43<00:10, 7450.61it/s] 80%|████████  | 320237/400000 [00:43<00:10, 7531.07it/s] 80%|████████  | 320999/400000 [00:43<00:10, 7555.46it/s] 80%|████████  | 321769/400000 [00:43<00:10, 7596.78it/s] 81%|████████  | 322543/400000 [00:43<00:10, 7638.89it/s] 81%|████████  | 323310/400000 [00:43<00:10, 7644.98it/s] 81%|████████  | 324078/400000 [00:43<00:09, 7653.27it/s] 81%|████████  | 324864/400000 [00:44<00:09, 7712.62it/s] 81%|████████▏ | 325648/400000 [00:44<00:09, 7749.02it/s] 82%|████████▏ | 326424/400000 [00:44<00:09, 7695.71it/s] 82%|████████▏ | 327194/400000 [00:44<00:09, 7553.82it/s] 82%|████████▏ | 327951/400000 [00:44<00:09, 7554.24it/s] 82%|████████▏ | 328707/400000 [00:44<00:09, 7525.37it/s] 82%|████████▏ | 329460/400000 [00:44<00:09, 7480.84it/s] 83%|████████▎ | 330209/400000 [00:44<00:09, 7448.66it/s] 83%|████████▎ | 330955/400000 [00:44<00:09, 7270.18it/s] 83%|████████▎ | 331684/400000 [00:45<00:09, 7110.62it/s] 83%|████████▎ | 332453/400000 [00:45<00:09, 7273.09it/s] 83%|████████▎ | 333202/400000 [00:45<00:09, 7336.53it/s] 83%|████████▎ | 333947/400000 [00:45<00:08, 7366.06it/s] 84%|████████▎ | 334685/400000 [00:45<00:09, 7143.89it/s] 84%|████████▍ | 335402/400000 [00:45<00:09, 7118.15it/s] 84%|████████▍ | 336181/400000 [00:45<00:08, 7300.82it/s] 84%|████████▍ | 336940/400000 [00:45<00:08, 7383.87it/s] 84%|████████▍ | 337710/400000 [00:45<00:08, 7475.41it/s] 85%|████████▍ | 338483/400000 [00:45<00:08, 7549.05it/s] 85%|████████▍ | 339268/400000 [00:46<00:07, 7636.40it/s] 85%|████████▌ | 340054/400000 [00:46<00:07, 7692.01it/s] 85%|████████▌ | 340825/400000 [00:46<00:07, 7552.95it/s] 85%|████████▌ | 341582/400000 [00:46<00:08, 7171.18it/s] 86%|████████▌ | 342304/400000 [00:46<00:08, 7077.61it/s] 86%|████████▌ | 343018/400000 [00:46<00:08, 7093.92it/s] 86%|████████▌ | 343784/400000 [00:46<00:07, 7252.68it/s] 86%|████████▌ | 344522/400000 [00:46<00:07, 7288.62it/s] 86%|████████▋ | 345292/400000 [00:46<00:07, 7406.18it/s] 87%|████████▋ | 346035/400000 [00:46<00:07, 7398.30it/s] 87%|████████▋ | 346811/400000 [00:47<00:07, 7503.07it/s] 87%|████████▋ | 347582/400000 [00:47<00:06, 7563.36it/s] 87%|████████▋ | 348366/400000 [00:47<00:06, 7643.28it/s] 87%|████████▋ | 349144/400000 [00:47<00:06, 7681.78it/s] 87%|████████▋ | 349913/400000 [00:47<00:06, 7458.53it/s] 88%|████████▊ | 350661/400000 [00:47<00:06, 7437.75it/s] 88%|████████▊ | 351451/400000 [00:47<00:06, 7569.43it/s] 88%|████████▊ | 352217/400000 [00:47<00:06, 7594.24it/s] 88%|████████▊ | 352997/400000 [00:47<00:06, 7652.89it/s] 88%|████████▊ | 353764/400000 [00:47<00:06, 7622.92it/s] 89%|████████▊ | 354527/400000 [00:48<00:05, 7581.86it/s] 89%|████████▉ | 355318/400000 [00:48<00:05, 7650.48it/s] 89%|████████▉ | 356102/400000 [00:48<00:05, 7705.59it/s] 89%|████████▉ | 356880/400000 [00:48<00:05, 7725.17it/s] 89%|████████▉ | 357653/400000 [00:48<00:05, 7629.36it/s] 90%|████████▉ | 358417/400000 [00:48<00:05, 7611.56it/s] 90%|████████▉ | 359194/400000 [00:48<00:05, 7657.80it/s] 90%|████████▉ | 359961/400000 [00:48<00:05, 7531.97it/s] 90%|█████████ | 360737/400000 [00:48<00:05, 7596.39it/s] 90%|█████████ | 361498/400000 [00:49<00:05, 7350.17it/s] 91%|█████████ | 362236/400000 [00:49<00:05, 7279.45it/s] 91%|█████████ | 362992/400000 [00:49<00:05, 7360.42it/s] 91%|█████████ | 363763/400000 [00:49<00:04, 7459.57it/s] 91%|█████████ | 364543/400000 [00:49<00:04, 7558.16it/s] 91%|█████████▏| 365301/400000 [00:49<00:04, 7539.07it/s] 92%|█████████▏| 366066/400000 [00:49<00:04, 7571.15it/s] 92%|█████████▏| 366824/400000 [00:49<00:04, 7451.92it/s] 92%|█████████▏| 367594/400000 [00:49<00:04, 7522.50it/s] 92%|█████████▏| 368369/400000 [00:49<00:04, 7588.52it/s] 92%|█████████▏| 369132/400000 [00:50<00:04, 7599.13it/s] 92%|█████████▏| 369893/400000 [00:50<00:04, 7355.77it/s] 93%|█████████▎| 370660/400000 [00:50<00:03, 7447.14it/s] 93%|█████████▎| 371408/400000 [00:50<00:03, 7456.40it/s] 93%|█████████▎| 372155/400000 [00:50<00:03, 7458.58it/s] 93%|█████████▎| 372907/400000 [00:50<00:03, 7474.18it/s] 93%|█████████▎| 373688/400000 [00:50<00:03, 7570.93it/s] 94%|█████████▎| 374464/400000 [00:50<00:03, 7626.60it/s] 94%|█████████▍| 375228/400000 [00:50<00:03, 7565.31it/s] 94%|█████████▍| 376004/400000 [00:50<00:03, 7619.07it/s] 94%|█████████▍| 376767/400000 [00:51<00:03, 7606.31it/s] 94%|█████████▍| 377556/400000 [00:51<00:02, 7686.95it/s] 95%|█████████▍| 378326/400000 [00:51<00:02, 7675.74it/s] 95%|█████████▍| 379102/400000 [00:51<00:02, 7699.18it/s] 95%|█████████▍| 379873/400000 [00:51<00:02, 7541.68it/s] 95%|█████████▌| 380629/400000 [00:51<00:02, 7536.70it/s] 95%|█████████▌| 381401/400000 [00:51<00:02, 7588.55it/s] 96%|█████████▌| 382180/400000 [00:51<00:02, 7647.32it/s] 96%|█████████▌| 382946/400000 [00:51<00:02, 7509.23it/s] 96%|█████████▌| 383704/400000 [00:51<00:02, 7528.93it/s] 96%|█████████▌| 384468/400000 [00:52<00:02, 7559.36it/s] 96%|█████████▋| 385254/400000 [00:52<00:01, 7644.42it/s] 97%|█████████▋| 386020/400000 [00:52<00:01, 7526.44it/s] 97%|█████████▋| 386801/400000 [00:52<00:01, 7607.86it/s] 97%|█████████▋| 387563/400000 [00:52<00:01, 7572.45it/s] 97%|█████████▋| 388321/400000 [00:52<00:01, 7540.30it/s] 97%|█████████▋| 389100/400000 [00:52<00:01, 7611.61it/s] 97%|█████████▋| 389885/400000 [00:52<00:01, 7681.53it/s] 98%|█████████▊| 390654/400000 [00:52<00:01, 7552.39it/s] 98%|█████████▊| 391421/400000 [00:52<00:01, 7586.52it/s] 98%|█████████▊| 392181/400000 [00:53<00:01, 7497.45it/s] 98%|█████████▊| 392952/400000 [00:53<00:00, 7557.74it/s] 98%|█████████▊| 393725/400000 [00:53<00:00, 7606.94it/s] 99%|█████████▊| 394494/400000 [00:53<00:00, 7630.36it/s] 99%|█████████▉| 395258/400000 [00:53<00:00, 7431.02it/s] 99%|█████████▉| 396003/400000 [00:53<00:00, 7353.46it/s] 99%|█████████▉| 396751/400000 [00:53<00:00, 7389.24it/s] 99%|█████████▉| 397526/400000 [00:53<00:00, 7492.17it/s]100%|█████████▉| 398320/400000 [00:53<00:00, 7618.43it/s]100%|█████████▉| 399104/400000 [00:53<00:00, 7659.39it/s]100%|█████████▉| 399871/400000 [00:54<00:00, 7543.13it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7392.77it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7ff44b854780>, <torchtext.data.dataset.TabularDataset object at 0x7ff44b8548d0>, <torchtext.vocab.Vocab object at 0x7ff44b8547f0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.18 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.18 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.18 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.81 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.81 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.88 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.88 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.43 file/s]2020-06-30 18:19:00.248642: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-30 18:19:00.254264: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-30 18:19:00.254534: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562ad8ff6190 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-30 18:19:00.254595: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 27%|██▋       | 2637824/9912422 [00:00<00:00, 25244971.02it/s]9920512it [00:00, 17005482.25it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 93746.58it/s]            
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 3018823.27it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 21533.26it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

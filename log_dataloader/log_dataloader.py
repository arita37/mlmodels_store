
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f0bc88f2ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f0bc88f2ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f0c33bb21e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f0c33bb21e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f0c51efae18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f0c51efae18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0be0eda488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0be0eda488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0be0eda488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 23%|██▎       | 2244608/9912422 [00:00<00:00, 21994959.06it/s] 76%|███████▌  | 7503872/9912422 [00:00<00:00, 26623005.59it/s]9920512it [00:00, 26414930.10it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1699925.21it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1049931.60it/s]1654784it [00:00, 12147774.06it/s]                           
0it [00:00, ?it/s]8192it [00:00, 217414.41it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0bc7f1c3c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0bc7f26668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f0be0eda0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f0be0eda0d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f0be0eda0d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:04,  2.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:04,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:03,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:03,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:02,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:02,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:02,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:01,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:01,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:43,  3.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:43,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:43,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:42,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:42,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:42,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:41,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:29,  4.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:29,  4.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:29,  4.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:29,  4.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:29,  4.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:29,  4.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:21,  6.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:21,  6.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:21,  6.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:21,  6.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:20,  6.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:20,  6.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:20,  6.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:15,  9.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:15,  9.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:14,  9.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:14,  9.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:14,  9.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:14,  9.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:14,  9.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:14,  9.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:00<00:10, 12.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:10, 12.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:10, 12.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:10, 12.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 12.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:10, 12.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:10, 12.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 12.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:07, 16.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:07, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:07, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 21.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 26.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 32.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 38.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 38.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 38.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 38.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 38.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 38.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 38.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 38.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:01, 44.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 48.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 52.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 55.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 58.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 58.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 58.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 58.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 58.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 58.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 58.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 58.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 61.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 61.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 63.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 65.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 65.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 65.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 65.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 65.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 65.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 65.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 66.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 66.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 66.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 66.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 66.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 66.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 67.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.86s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.86s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.86s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.94 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.35s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.86s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 68.94 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.35s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.35s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.26 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.35s/ url]
0 examples [00:00, ? examples/s]2020-06-25 18:09:54.964529: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-25 18:09:55.025239: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-25 18:09:55.025450: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55df17ac8720 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-25 18:09:55.025472: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.80 examples/s]82 examples [00:00,  2.57 examples/s]176 examples [00:00,  3.67 examples/s]263 examples [00:00,  5.23 examples/s]354 examples [00:00,  7.45 examples/s]443 examples [00:01, 10.61 examples/s]532 examples [00:01, 15.08 examples/s]623 examples [00:01, 21.39 examples/s]705 examples [00:01, 30.19 examples/s]789 examples [00:01, 42.47 examples/s]879 examples [00:01, 59.46 examples/s]968 examples [00:01, 82.58 examples/s]1058 examples [00:01, 113.50 examples/s]1148 examples [00:01, 153.80 examples/s]1240 examples [00:01, 204.95 examples/s]1333 examples [00:02, 267.38 examples/s]1424 examples [00:02, 338.81 examples/s]1514 examples [00:02, 415.43 examples/s]1607 examples [00:02, 497.46 examples/s]1698 examples [00:02, 574.45 examples/s]1789 examples [00:02, 634.91 examples/s]1878 examples [00:02, 694.18 examples/s]1967 examples [00:02, 732.27 examples/s]2057 examples [00:02, 775.17 examples/s]2145 examples [00:02, 792.46 examples/s]2232 examples [00:03, 804.04 examples/s]2321 examples [00:03, 827.75 examples/s]2412 examples [00:03, 850.17 examples/s]2503 examples [00:03, 865.09 examples/s]2598 examples [00:03, 886.35 examples/s]2689 examples [00:03, 887.20 examples/s]2779 examples [00:03, 885.79 examples/s]2870 examples [00:03, 891.18 examples/s]2960 examples [00:03, 890.20 examples/s]3050 examples [00:04, 859.86 examples/s]3137 examples [00:04, 852.06 examples/s]3226 examples [00:04, 861.13 examples/s]3313 examples [00:04, 862.02 examples/s]3400 examples [00:04, 864.20 examples/s]3489 examples [00:04, 870.16 examples/s]3577 examples [00:04, 860.51 examples/s]3664 examples [00:04, 847.26 examples/s]3752 examples [00:04, 856.76 examples/s]3841 examples [00:04, 864.63 examples/s]3930 examples [00:05, 870.66 examples/s]4018 examples [00:05, 852.34 examples/s]4106 examples [00:05, 858.33 examples/s]4198 examples [00:05, 873.52 examples/s]4288 examples [00:05, 881.02 examples/s]4380 examples [00:05, 890.29 examples/s]4470 examples [00:05, 892.80 examples/s]4560 examples [00:05, 883.56 examples/s]4652 examples [00:05, 893.37 examples/s]4742 examples [00:05, 888.95 examples/s]4831 examples [00:06, 879.51 examples/s]4920 examples [00:06, 877.45 examples/s]5008 examples [00:06, 868.46 examples/s]5095 examples [00:06, 857.53 examples/s]5183 examples [00:06, 862.74 examples/s]5275 examples [00:06, 877.88 examples/s]5369 examples [00:06, 893.88 examples/s]5461 examples [00:06, 898.95 examples/s]5552 examples [00:06, 893.80 examples/s]5642 examples [00:06, 880.61 examples/s]5731 examples [00:07, 882.42 examples/s]5820 examples [00:07, 872.67 examples/s]5908 examples [00:07, 862.91 examples/s]5995 examples [00:07, 863.80 examples/s]6084 examples [00:07, 869.39 examples/s]6173 examples [00:07, 872.30 examples/s]6261 examples [00:07, 866.95 examples/s]6348 examples [00:07, 848.48 examples/s]6437 examples [00:07, 857.93 examples/s]6529 examples [00:08, 873.71 examples/s]6618 examples [00:08, 877.35 examples/s]6706 examples [00:08, 863.97 examples/s]6794 examples [00:08, 867.18 examples/s]6882 examples [00:08, 867.78 examples/s]6973 examples [00:08, 878.29 examples/s]7064 examples [00:08, 887.48 examples/s]7153 examples [00:08, 877.36 examples/s]7241 examples [00:08, 858.28 examples/s]7334 examples [00:08, 877.27 examples/s]7424 examples [00:09, 882.52 examples/s]7516 examples [00:09, 891.55 examples/s]7606 examples [00:09, 893.54 examples/s]7697 examples [00:09, 896.60 examples/s]7787 examples [00:09, 893.46 examples/s]7877 examples [00:09, 862.18 examples/s]7964 examples [00:09, 835.53 examples/s]8048 examples [00:09, 803.30 examples/s]8136 examples [00:09, 823.60 examples/s]8228 examples [00:09, 849.54 examples/s]8314 examples [00:10, 839.38 examples/s]8405 examples [00:10, 857.20 examples/s]8496 examples [00:10, 871.88 examples/s]8588 examples [00:10, 883.58 examples/s]8680 examples [00:10, 893.43 examples/s]8772 examples [00:10, 898.72 examples/s]8865 examples [00:10, 905.48 examples/s]8956 examples [00:10, 905.58 examples/s]9051 examples [00:10, 916.10 examples/s]9143 examples [00:10, 911.84 examples/s]9235 examples [00:11, 912.65 examples/s]9327 examples [00:11, 908.13 examples/s]9418 examples [00:11, 900.79 examples/s]9509 examples [00:11, 890.48 examples/s]9599 examples [00:11, 884.00 examples/s]9688 examples [00:11, 885.40 examples/s]9777 examples [00:11, 881.20 examples/s]9866 examples [00:11, 866.28 examples/s]9956 examples [00:11, 876.10 examples/s]10044 examples [00:12, 828.43 examples/s]10136 examples [00:12, 852.79 examples/s]10228 examples [00:12, 869.56 examples/s]10317 examples [00:12, 875.47 examples/s]10409 examples [00:12, 886.09 examples/s]10500 examples [00:12, 890.02 examples/s]10590 examples [00:12, 889.46 examples/s]10680 examples [00:12, 878.78 examples/s]10769 examples [00:12, 873.28 examples/s]10860 examples [00:12, 883.07 examples/s]10952 examples [00:13, 891.54 examples/s]11042 examples [00:13, 880.55 examples/s]11131 examples [00:13, 869.94 examples/s]11220 examples [00:13, 873.50 examples/s]11311 examples [00:13, 883.02 examples/s]11404 examples [00:13, 894.24 examples/s]11494 examples [00:13, 880.41 examples/s]11583 examples [00:13, 853.60 examples/s]11669 examples [00:13, 838.89 examples/s]11755 examples [00:13, 844.12 examples/s]11844 examples [00:14, 854.88 examples/s]11935 examples [00:14, 870.69 examples/s]12025 examples [00:14, 878.95 examples/s]12114 examples [00:14, 867.40 examples/s]12201 examples [00:14, 867.05 examples/s]12291 examples [00:14, 874.73 examples/s]12383 examples [00:14, 886.04 examples/s]12472 examples [00:14, 855.31 examples/s]12558 examples [00:14, 853.98 examples/s]12648 examples [00:15, 865.26 examples/s]12738 examples [00:15, 874.33 examples/s]12826 examples [00:15, 840.49 examples/s]12911 examples [00:15, 830.18 examples/s]12995 examples [00:15, 811.12 examples/s]13077 examples [00:15, 791.96 examples/s]13160 examples [00:15, 802.37 examples/s]13247 examples [00:15, 819.25 examples/s]13335 examples [00:15, 836.09 examples/s]13420 examples [00:15, 839.94 examples/s]13505 examples [00:16, 829.12 examples/s]13589 examples [00:16, 818.60 examples/s]13678 examples [00:16, 837.74 examples/s]13767 examples [00:16, 849.85 examples/s]13853 examples [00:16, 852.08 examples/s]13939 examples [00:16, 842.20 examples/s]14025 examples [00:16, 845.04 examples/s]14114 examples [00:16, 856.29 examples/s]14202 examples [00:16, 861.66 examples/s]14289 examples [00:16, 858.65 examples/s]14377 examples [00:17, 863.40 examples/s]14468 examples [00:17, 874.26 examples/s]14559 examples [00:17, 882.24 examples/s]14653 examples [00:17, 896.34 examples/s]14743 examples [00:17, 889.62 examples/s]14833 examples [00:17, 863.80 examples/s]14920 examples [00:17, 865.28 examples/s]15012 examples [00:17, 878.67 examples/s]15104 examples [00:17, 890.43 examples/s]15194 examples [00:17, 887.41 examples/s]15286 examples [00:18, 894.28 examples/s]15376 examples [00:18, 879.34 examples/s]15466 examples [00:18, 885.07 examples/s]15561 examples [00:18, 903.45 examples/s]15652 examples [00:18, 896.50 examples/s]15742 examples [00:18, 856.75 examples/s]15829 examples [00:18, 849.76 examples/s]15915 examples [00:18, 851.08 examples/s]16003 examples [00:18, 858.11 examples/s]16089 examples [00:19, 857.78 examples/s]16176 examples [00:19, 861.31 examples/s]16263 examples [00:19, 840.41 examples/s]16352 examples [00:19, 852.66 examples/s]16440 examples [00:19, 860.22 examples/s]16527 examples [00:19, 854.43 examples/s]16613 examples [00:19, 834.10 examples/s]16698 examples [00:19, 835.76 examples/s]16782 examples [00:19, 824.13 examples/s]16865 examples [00:19, 822.21 examples/s]16952 examples [00:20, 835.43 examples/s]17040 examples [00:20, 848.15 examples/s]17129 examples [00:20, 859.32 examples/s]17219 examples [00:20, 868.50 examples/s]17308 examples [00:20, 872.15 examples/s]17396 examples [00:20, 870.38 examples/s]17484 examples [00:20, 840.89 examples/s]17575 examples [00:20, 859.84 examples/s]17663 examples [00:20, 864.42 examples/s]17750 examples [00:20, 864.09 examples/s]17837 examples [00:21, 860.13 examples/s]17925 examples [00:21, 865.09 examples/s]18012 examples [00:21, 858.38 examples/s]18103 examples [00:21, 870.51 examples/s]18191 examples [00:21, 850.46 examples/s]18278 examples [00:21, 854.38 examples/s]18367 examples [00:21, 863.35 examples/s]18454 examples [00:21, 864.09 examples/s]18544 examples [00:21, 874.15 examples/s]18633 examples [00:21, 878.20 examples/s]18721 examples [00:22, 850.81 examples/s]18811 examples [00:22, 864.24 examples/s]18898 examples [00:22, 864.93 examples/s]18985 examples [00:22, 861.70 examples/s]19078 examples [00:22, 878.41 examples/s]19167 examples [00:22, 880.86 examples/s]19260 examples [00:22, 893.72 examples/s]19351 examples [00:22, 897.88 examples/s]19442 examples [00:22, 900.60 examples/s]19533 examples [00:23, 895.86 examples/s]19623 examples [00:23, 862.51 examples/s]19710 examples [00:23, 831.15 examples/s]19794 examples [00:23, 833.36 examples/s]19879 examples [00:23, 835.36 examples/s]19972 examples [00:23, 860.19 examples/s]20059 examples [00:23, 776.96 examples/s]20139 examples [00:23, 781.71 examples/s]20229 examples [00:23, 811.82 examples/s]20315 examples [00:23, 823.25 examples/s]20402 examples [00:24, 836.29 examples/s]20490 examples [00:24, 846.48 examples/s]20581 examples [00:24, 861.99 examples/s]20674 examples [00:24, 879.45 examples/s]20764 examples [00:24, 884.26 examples/s]20853 examples [00:24, 875.14 examples/s]20941 examples [00:24, 839.74 examples/s]21027 examples [00:24, 844.18 examples/s]21116 examples [00:24, 857.26 examples/s]21205 examples [00:25, 864.82 examples/s]21292 examples [00:25, 860.27 examples/s]21379 examples [00:25, 861.62 examples/s]21466 examples [00:25, 842.21 examples/s]21551 examples [00:25, 832.87 examples/s]21635 examples [00:25, 819.85 examples/s]21719 examples [00:25, 823.19 examples/s]21806 examples [00:25, 835.44 examples/s]21893 examples [00:25, 844.84 examples/s]21981 examples [00:25, 853.28 examples/s]22067 examples [00:26, 850.95 examples/s]22153 examples [00:26, 829.57 examples/s]22237 examples [00:26, 826.67 examples/s]22323 examples [00:26, 832.59 examples/s]22407 examples [00:26, 807.13 examples/s]22488 examples [00:26, 773.06 examples/s]22572 examples [00:26, 789.68 examples/s]22658 examples [00:26, 807.41 examples/s]22740 examples [00:26, 792.19 examples/s]22824 examples [00:26, 802.98 examples/s]22909 examples [00:27, 814.25 examples/s]22991 examples [00:27, 804.24 examples/s]23077 examples [00:27, 818.88 examples/s]23167 examples [00:27, 841.40 examples/s]23258 examples [00:27, 858.49 examples/s]23345 examples [00:27, 822.80 examples/s]23429 examples [00:27, 827.08 examples/s]23520 examples [00:27, 848.46 examples/s]23607 examples [00:27, 852.36 examples/s]23699 examples [00:28, 870.63 examples/s]23787 examples [00:28, 860.02 examples/s]23874 examples [00:28, 837.27 examples/s]23959 examples [00:28, 829.86 examples/s]24045 examples [00:28, 838.39 examples/s]24136 examples [00:28, 857.66 examples/s]24222 examples [00:28, 839.29 examples/s]24313 examples [00:28, 858.41 examples/s]24400 examples [00:28, 860.87 examples/s]24489 examples [00:28, 867.16 examples/s]24576 examples [00:29, 863.60 examples/s]24663 examples [00:29, 864.60 examples/s]24751 examples [00:29, 867.80 examples/s]24838 examples [00:29, 864.91 examples/s]24925 examples [00:29, 865.04 examples/s]25012 examples [00:29, 826.09 examples/s]25098 examples [00:29, 835.69 examples/s]25185 examples [00:29, 844.92 examples/s]25270 examples [00:29, 844.67 examples/s]25355 examples [00:29, 844.59 examples/s]25442 examples [00:30, 851.42 examples/s]25532 examples [00:30, 863.00 examples/s]25619 examples [00:30, 825.10 examples/s]25706 examples [00:30, 837.29 examples/s]25791 examples [00:30, 839.92 examples/s]25876 examples [00:30, 825.01 examples/s]25959 examples [00:30, 811.70 examples/s]26041 examples [00:30, 805.42 examples/s]26122 examples [00:30, 800.10 examples/s]26211 examples [00:30, 824.09 examples/s]26298 examples [00:31, 835.73 examples/s]26387 examples [00:31, 849.33 examples/s]26473 examples [00:31, 845.08 examples/s]26562 examples [00:31, 855.53 examples/s]26649 examples [00:31, 857.47 examples/s]26735 examples [00:31, 832.05 examples/s]26819 examples [00:31, 816.57 examples/s]26908 examples [00:31, 835.43 examples/s]26996 examples [00:31, 846.08 examples/s]27084 examples [00:32, 853.36 examples/s]27175 examples [00:32, 868.31 examples/s]27263 examples [00:32, 862.03 examples/s]27350 examples [00:32, 860.47 examples/s]27437 examples [00:32, 862.00 examples/s]27526 examples [00:32, 868.64 examples/s]27613 examples [00:32, 822.50 examples/s]27696 examples [00:32, 820.95 examples/s]27783 examples [00:32, 834.53 examples/s]27872 examples [00:32, 848.30 examples/s]27967 examples [00:33, 874.34 examples/s]28057 examples [00:33, 881.85 examples/s]28146 examples [00:33, 874.03 examples/s]28234 examples [00:33, 861.38 examples/s]28321 examples [00:33, 863.33 examples/s]28411 examples [00:33, 872.54 examples/s]28502 examples [00:33, 881.55 examples/s]28591 examples [00:33, 874.86 examples/s]28679 examples [00:33, 868.95 examples/s]28766 examples [00:33, 865.02 examples/s]28853 examples [00:34, 865.27 examples/s]28940 examples [00:34, 861.56 examples/s]29027 examples [00:34, 855.30 examples/s]29114 examples [00:34, 858.50 examples/s]29203 examples [00:34, 866.23 examples/s]29290 examples [00:34, 862.50 examples/s]29377 examples [00:34, 844.48 examples/s]29465 examples [00:34, 853.28 examples/s]29561 examples [00:34, 881.91 examples/s]29650 examples [00:34, 871.43 examples/s]29738 examples [00:35, 861.18 examples/s]29826 examples [00:35, 865.71 examples/s]29913 examples [00:35, 863.05 examples/s]30000 examples [00:35, 864.89 examples/s]30087 examples [00:35, 786.57 examples/s]30168 examples [00:35, 755.56 examples/s]30245 examples [00:35, 752.11 examples/s]30331 examples [00:35, 780.53 examples/s]30415 examples [00:35, 795.22 examples/s]30503 examples [00:36, 817.79 examples/s]30587 examples [00:36, 823.24 examples/s]30670 examples [00:36, 824.06 examples/s]30753 examples [00:36, 812.12 examples/s]30842 examples [00:36, 832.31 examples/s]30926 examples [00:36, 828.62 examples/s]31010 examples [00:36, 825.97 examples/s]31098 examples [00:36, 838.19 examples/s]31188 examples [00:36, 853.71 examples/s]31277 examples [00:36, 861.71 examples/s]31368 examples [00:37, 875.13 examples/s]31457 examples [00:37, 878.32 examples/s]31545 examples [00:37, 859.20 examples/s]31632 examples [00:37, 857.28 examples/s]31719 examples [00:37, 859.01 examples/s]31808 examples [00:37, 866.51 examples/s]31896 examples [00:37, 869.11 examples/s]31983 examples [00:37, 853.98 examples/s]32074 examples [00:37, 869.31 examples/s]32164 examples [00:37, 877.90 examples/s]32254 examples [00:38, 882.26 examples/s]32346 examples [00:38, 891.23 examples/s]32436 examples [00:38, 859.37 examples/s]32523 examples [00:38, 850.59 examples/s]32609 examples [00:38, 842.41 examples/s]32699 examples [00:38, 856.67 examples/s]32785 examples [00:38, 857.07 examples/s]32874 examples [00:38, 865.69 examples/s]32966 examples [00:38, 879.34 examples/s]33056 examples [00:39, 883.61 examples/s]33147 examples [00:39, 888.75 examples/s]33237 examples [00:39, 890.19 examples/s]33327 examples [00:39, 881.88 examples/s]33416 examples [00:39, 878.29 examples/s]33504 examples [00:39, 873.81 examples/s]33592 examples [00:39, 865.83 examples/s]33679 examples [00:39, 862.58 examples/s]33766 examples [00:39, 850.98 examples/s]33852 examples [00:39, 849.92 examples/s]33938 examples [00:40, 843.59 examples/s]34025 examples [00:40, 846.21 examples/s]34111 examples [00:40, 849.67 examples/s]34198 examples [00:40, 853.82 examples/s]34288 examples [00:40, 864.72 examples/s]34377 examples [00:40, 872.06 examples/s]34465 examples [00:40, 865.49 examples/s]34552 examples [00:40, 842.35 examples/s]34637 examples [00:40, 837.72 examples/s]34721 examples [00:40, 818.71 examples/s]34804 examples [00:41, 814.47 examples/s]34892 examples [00:41, 831.04 examples/s]34983 examples [00:41, 851.19 examples/s]35074 examples [00:41, 865.49 examples/s]35165 examples [00:41, 871.04 examples/s]35253 examples [00:41, 865.18 examples/s]35341 examples [00:41, 868.87 examples/s]35428 examples [00:41, 856.81 examples/s]35514 examples [00:41, 826.26 examples/s]35600 examples [00:42, 834.23 examples/s]35684 examples [00:42, 817.21 examples/s]35766 examples [00:42, 817.84 examples/s]35851 examples [00:42, 825.44 examples/s]35938 examples [00:42, 836.87 examples/s]36028 examples [00:42, 852.27 examples/s]36119 examples [00:42, 868.05 examples/s]36211 examples [00:42, 880.39 examples/s]36300 examples [00:42, 879.67 examples/s]36389 examples [00:42, 876.89 examples/s]36477 examples [00:43, 870.24 examples/s]36566 examples [00:43, 875.36 examples/s]36654 examples [00:43, 875.19 examples/s]36745 examples [00:43, 883.88 examples/s]36836 examples [00:43, 890.27 examples/s]36926 examples [00:43, 887.10 examples/s]37018 examples [00:43, 896.59 examples/s]37108 examples [00:43, 893.06 examples/s]37198 examples [00:43, 882.50 examples/s]37290 examples [00:43, 890.97 examples/s]37380 examples [00:44, 888.45 examples/s]37469 examples [00:44, 881.95 examples/s]37558 examples [00:44, 875.24 examples/s]37646 examples [00:44, 871.77 examples/s]37734 examples [00:44, 869.80 examples/s]37824 examples [00:44, 878.53 examples/s]37912 examples [00:44, 875.14 examples/s]38003 examples [00:44, 883.62 examples/s]38092 examples [00:44, 876.99 examples/s]38180 examples [00:44, 850.90 examples/s]38266 examples [00:45, 853.12 examples/s]38354 examples [00:45, 858.96 examples/s]38441 examples [00:45, 851.07 examples/s]38527 examples [00:45, 845.89 examples/s]38612 examples [00:45, 839.00 examples/s]38696 examples [00:45, 813.01 examples/s]38785 examples [00:45, 834.15 examples/s]38870 examples [00:45, 838.55 examples/s]38958 examples [00:45, 850.26 examples/s]39048 examples [00:45, 862.28 examples/s]39139 examples [00:46, 873.36 examples/s]39227 examples [00:46, 868.08 examples/s]39317 examples [00:46, 876.96 examples/s]39406 examples [00:46, 880.59 examples/s]39495 examples [00:46, 881.51 examples/s]39584 examples [00:46, 874.25 examples/s]39672 examples [00:46, 873.30 examples/s]39761 examples [00:46, 876.90 examples/s]39849 examples [00:46, 873.79 examples/s]39938 examples [00:46, 875.95 examples/s]40026 examples [00:47, 806.84 examples/s]40115 examples [00:47, 829.84 examples/s]40204 examples [00:47, 845.47 examples/s]40292 examples [00:47, 853.30 examples/s]40382 examples [00:47, 864.09 examples/s]40469 examples [00:47, 863.77 examples/s]40557 examples [00:47, 866.44 examples/s]40644 examples [00:47, 865.80 examples/s]40731 examples [00:47, 865.95 examples/s]40821 examples [00:48, 874.31 examples/s]40909 examples [00:48, 871.13 examples/s]40999 examples [00:48, 877.19 examples/s]41088 examples [00:48, 878.95 examples/s]41178 examples [00:48, 883.51 examples/s]41267 examples [00:48, 874.76 examples/s]41358 examples [00:48, 884.59 examples/s]41447 examples [00:48, 884.85 examples/s]41536 examples [00:48, 881.82 examples/s]41625 examples [00:48, 875.85 examples/s]41713 examples [00:49, 857.25 examples/s]41800 examples [00:49, 859.55 examples/s]41889 examples [00:49, 865.93 examples/s]41976 examples [00:49, 850.31 examples/s]42062 examples [00:49, 850.56 examples/s]42148 examples [00:49, 831.28 examples/s]42235 examples [00:49, 842.48 examples/s]42325 examples [00:49, 858.79 examples/s]42415 examples [00:49, 868.78 examples/s]42504 examples [00:49, 872.76 examples/s]42592 examples [00:50, 806.07 examples/s]42679 examples [00:50, 821.80 examples/s]42771 examples [00:50, 847.92 examples/s]42862 examples [00:50, 863.20 examples/s]42952 examples [00:50, 871.70 examples/s]43040 examples [00:50, 871.19 examples/s]43128 examples [00:50, 862.01 examples/s]43216 examples [00:50, 865.54 examples/s]43303 examples [00:50, 856.52 examples/s]43389 examples [00:51, 856.90 examples/s]43475 examples [00:51, 839.42 examples/s]43560 examples [00:51, 833.84 examples/s]43648 examples [00:51, 846.71 examples/s]43734 examples [00:51, 850.06 examples/s]43825 examples [00:51, 866.07 examples/s]43913 examples [00:51, 868.74 examples/s]44000 examples [00:51, 854.38 examples/s]44087 examples [00:51, 856.39 examples/s]44173 examples [00:51, 843.79 examples/s]44258 examples [00:52, 844.85 examples/s]44347 examples [00:52, 856.48 examples/s]44436 examples [00:52, 864.65 examples/s]44527 examples [00:52, 875.29 examples/s]44615 examples [00:52, 874.04 examples/s]44703 examples [00:52, 870.95 examples/s]44793 examples [00:52, 877.35 examples/s]44883 examples [00:52, 881.92 examples/s]44975 examples [00:52, 892.01 examples/s]45066 examples [00:52, 896.53 examples/s]45156 examples [00:53, 896.07 examples/s]45246 examples [00:53, 885.10 examples/s]45338 examples [00:53, 892.92 examples/s]45428 examples [00:53, 894.22 examples/s]45519 examples [00:53, 897.55 examples/s]45609 examples [00:53, 896.64 examples/s]45699 examples [00:53, 883.62 examples/s]45788 examples [00:53, 878.27 examples/s]45876 examples [00:53, 876.65 examples/s]45964 examples [00:53, 851.80 examples/s]46055 examples [00:54, 868.39 examples/s]46149 examples [00:54, 888.24 examples/s]46239 examples [00:54, 890.19 examples/s]46329 examples [00:54, 890.16 examples/s]46419 examples [00:54, 884.75 examples/s]46508 examples [00:54, 870.86 examples/s]46596 examples [00:54, 864.17 examples/s]46683 examples [00:54, 860.98 examples/s]46770 examples [00:54, 860.25 examples/s]46858 examples [00:54, 864.95 examples/s]46945 examples [00:55, 865.24 examples/s]47032 examples [00:55, 860.11 examples/s]47119 examples [00:55, 853.01 examples/s]47209 examples [00:55, 865.73 examples/s]47296 examples [00:55, 865.36 examples/s]47383 examples [00:55, 858.91 examples/s]47469 examples [00:55, 852.06 examples/s]47559 examples [00:55, 863.25 examples/s]47647 examples [00:55, 866.69 examples/s]47736 examples [00:56, 871.91 examples/s]47824 examples [00:56, 870.62 examples/s]47912 examples [00:56, 871.07 examples/s]48000 examples [00:56, 873.25 examples/s]48090 examples [00:56, 879.14 examples/s]48179 examples [00:56, 879.83 examples/s]48267 examples [00:56, 849.22 examples/s]48353 examples [00:56, 823.25 examples/s]48437 examples [00:56, 826.26 examples/s]48527 examples [00:56, 844.67 examples/s]48612 examples [00:57, 842.54 examples/s]48699 examples [00:57, 846.48 examples/s]48785 examples [00:57, 849.94 examples/s]48872 examples [00:57, 854.18 examples/s]48958 examples [00:57, 852.10 examples/s]49046 examples [00:57, 858.58 examples/s]49132 examples [00:57, 853.66 examples/s]49218 examples [00:57, 833.93 examples/s]49307 examples [00:57, 847.31 examples/s]49399 examples [00:57, 866.63 examples/s]49491 examples [00:58, 879.36 examples/s]49580 examples [00:58, 879.63 examples/s]49672 examples [00:58, 888.82 examples/s]49761 examples [00:58, 843.26 examples/s]49850 examples [00:58, 855.57 examples/s]49938 examples [00:58, 861.83 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█▏        | 5646/50000 [00:00<00:00, 56456.96 examples/s] 31%|███       | 15578/50000 [00:00<00:00, 64852.83 examples/s] 51%|█████▏    | 25740/50000 [00:00<00:00, 72748.68 examples/s] 73%|███████▎  | 36452/50000 [00:00<00:00, 80495.53 examples/s] 94%|█████████▍| 47145/50000 [00:00<00:00, 86942.71 examples/s]                                                               0 examples [00:00, ? examples/s]63 examples [00:00, 628.08 examples/s]150 examples [00:00, 683.53 examples/s]237 examples [00:00, 729.71 examples/s]323 examples [00:00, 763.98 examples/s]416 examples [00:00, 805.41 examples/s]508 examples [00:00, 835.06 examples/s]603 examples [00:00, 865.05 examples/s]693 examples [00:00, 874.11 examples/s]779 examples [00:00, 867.06 examples/s]868 examples [00:01, 860.58 examples/s]953 examples [00:01, 854.93 examples/s]1038 examples [00:01, 825.20 examples/s]1124 examples [00:01, 833.20 examples/s]1208 examples [00:01, 832.74 examples/s]1297 examples [00:01, 847.90 examples/s]1389 examples [00:01, 867.10 examples/s]1482 examples [00:01, 882.63 examples/s]1573 examples [00:01, 886.86 examples/s]1665 examples [00:01, 893.78 examples/s]1755 examples [00:02, 895.49 examples/s]1845 examples [00:02, 895.56 examples/s]1936 examples [00:02, 898.72 examples/s]2026 examples [00:02, 898.41 examples/s]2117 examples [00:02, 900.00 examples/s]2208 examples [00:02, 898.48 examples/s]2298 examples [00:02, 883.60 examples/s]2389 examples [00:02, 890.63 examples/s]2480 examples [00:02, 895.30 examples/s]2571 examples [00:02, 898.38 examples/s]2661 examples [00:03, 893.23 examples/s]2753 examples [00:03, 899.02 examples/s]2846 examples [00:03, 905.74 examples/s]2937 examples [00:03, 905.16 examples/s]3030 examples [00:03, 910.03 examples/s]3122 examples [00:03, 904.14 examples/s]3213 examples [00:03, 905.65 examples/s]3306 examples [00:03, 911.57 examples/s]3398 examples [00:03, 911.09 examples/s]3490 examples [00:03, 898.71 examples/s]3580 examples [00:04, 841.86 examples/s]3669 examples [00:04, 853.91 examples/s]3760 examples [00:04, 869.23 examples/s]3852 examples [00:04, 882.79 examples/s]3944 examples [00:04, 891.62 examples/s]4035 examples [00:04, 896.80 examples/s]4125 examples [00:04, 887.96 examples/s]4217 examples [00:04, 896.15 examples/s]4307 examples [00:04, 862.88 examples/s]4396 examples [00:04, 869.47 examples/s]4484 examples [00:05, 872.24 examples/s]4576 examples [00:05, 883.99 examples/s]4670 examples [00:05, 897.69 examples/s]4761 examples [00:05, 900.48 examples/s]4855 examples [00:05, 909.96 examples/s]4947 examples [00:05, 905.86 examples/s]5041 examples [00:05, 912.99 examples/s]5134 examples [00:05, 916.74 examples/s]5227 examples [00:05, 918.96 examples/s]5319 examples [00:06, 919.06 examples/s]5412 examples [00:06, 920.91 examples/s]5505 examples [00:06, 919.25 examples/s]5597 examples [00:06, 915.34 examples/s]5689 examples [00:06, 906.01 examples/s]5780 examples [00:06, 902.75 examples/s]5871 examples [00:06, 895.32 examples/s]5961 examples [00:06, 893.45 examples/s]6051 examples [00:06, 877.51 examples/s]6139 examples [00:06, 867.32 examples/s]6226 examples [00:07, 858.24 examples/s]6312 examples [00:07, 841.60 examples/s]6398 examples [00:07, 846.08 examples/s]6486 examples [00:07, 855.66 examples/s]6572 examples [00:07, 853.06 examples/s]6661 examples [00:07, 862.66 examples/s]6748 examples [00:07, 863.05 examples/s]6844 examples [00:07, 889.61 examples/s]6937 examples [00:07, 899.37 examples/s]7028 examples [00:07, 885.25 examples/s]7117 examples [00:08, 882.57 examples/s]7210 examples [00:08, 893.92 examples/s]7301 examples [00:08, 898.23 examples/s]7391 examples [00:08, 898.32 examples/s]7482 examples [00:08, 901.46 examples/s]7574 examples [00:08, 906.38 examples/s]7668 examples [00:08, 913.90 examples/s]7760 examples [00:08, 915.16 examples/s]7853 examples [00:08, 918.59 examples/s]7945 examples [00:08, 918.71 examples/s]8037 examples [00:09, 914.00 examples/s]8129 examples [00:09, 905.07 examples/s]8220 examples [00:09, 905.76 examples/s]8311 examples [00:09, 899.09 examples/s]8406 examples [00:09, 911.02 examples/s]8498 examples [00:09, 909.07 examples/s]8589 examples [00:09, 901.51 examples/s]8680 examples [00:09, 894.00 examples/s]8771 examples [00:09, 898.72 examples/s]8861 examples [00:09, 871.40 examples/s]8949 examples [00:10, 837.02 examples/s]9040 examples [00:10, 857.60 examples/s]9132 examples [00:10, 873.09 examples/s]9222 examples [00:10, 878.13 examples/s]9311 examples [00:10, 881.54 examples/s]9400 examples [00:10, 871.86 examples/s]9488 examples [00:10, 855.75 examples/s]9574 examples [00:10, 848.30 examples/s]9659 examples [00:10, 826.05 examples/s]9743 examples [00:11, 828.81 examples/s]9827 examples [00:11, 826.89 examples/s]9911 examples [00:11, 828.73 examples/s]9999 examples [00:11, 843.00 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 98%|█████████▊| 9842/10000 [00:00<00:00, 98410.24 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete2Z5RGG/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete2Z5RGG/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0be0eda488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0be0eda488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0be0eda488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0b7fce23c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0b7fce2b00>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0be0eda488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0be0eda488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0be0eda488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0bc8087588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0bc7f26160>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f0b6085ee18> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f0b6085ee18> 

  function with postional parmater data_info <function split_train_valid at 0x7f0b6085ee18> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f0b6085ef28> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f0b6085ef28> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f0b6085ef28> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=247e90a5a4350e8bb0704d31a998f80c7cea2b11c2f923d7ed236d62e49108e5
  Stored in directory: /tmp/pip-ephem-wheel-cache-xvtoieh3/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:17:43, 11.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:09:10, 15.8kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:39:49, 22.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:28:25, 32.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<5:13:05, 45.7kB/s].vector_cache/glove.6B.zip:   1%|          | 7.77M/862M [00:01<3:38:12, 65.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:31:54, 93.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.4M/862M [00:01<1:46:02, 133kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.3M/862M [00:01<1:13:52, 190kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.4M/862M [00:01<51:43, 270kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 30.1M/862M [00:01<36:03, 385kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.3M/862M [00:02<25:10, 548kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:02<17:38, 778kB/s].vector_cache/glove.6B.zip:   5%|▍         | 43.0M/862M [00:02<12:24, 1.10MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.2M/862M [00:02<08:42, 1.56MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<06:21, 2.12MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<06:21, 2.12MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<07:13, 1.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<05:44, 2.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<06:11, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<05:46, 2.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:06<04:23, 3.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:08<06:09, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<05:44, 2.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:08<04:19, 3.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<06:05, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<05:41, 2.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:10<04:18, 3.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:12<06:06, 2.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<05:27, 2.41MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<04:18, 3.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:12<03:09, 4.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<54:28, 240kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<39:31, 331kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:14<27:57, 467kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<22:34, 577kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<16:57, 768kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<12:31, 1.04MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:16<08:52, 1.46MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<54:47, 237kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<39:43, 326kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:18<28:03, 461kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<22:38, 569kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<16:58, 759kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<12:15, 1.05MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:20<08:43, 1.47MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:22<54:06, 237kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<39:15, 327kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<27:46, 461kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:24<22:22, 570kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<17:01, 749kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:24<12:13, 1.04MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<11:31, 1.10MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<09:25, 1.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:52, 1.84MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<07:47, 1.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:48, 1.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:05, 2.47MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:30, 1.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:54, 2.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:24, 2.84MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<06:00, 2.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:32, 2.25MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<04:12, 2.96MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<05:51, 2.12MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:28, 2.27MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<04:05, 3.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<03:01, 4.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<1:08:07, 181kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<48:57, 252kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<34:28, 357kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<26:58, 455kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<19:58, 614kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<14:25, 850kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<10:12, 1.20MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<55:53, 218kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<40:27, 302kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<28:34, 426kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<22:47, 533kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<17:15, 703kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<12:19, 983kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<11:27, 1.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<09:05, 1.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:54, 1.75MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<04:56, 2.43MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<51:24, 233kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<37:13, 322kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<26:16, 455kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<21:10, 564kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<16:05, 741kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<11:33, 1.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<10:51, 1.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<08:52, 1.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:30, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<07:18, 1.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:23, 1.85MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<04:47, 2.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:05, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:31, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<04:10, 2.80MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:38, 2.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:12, 2.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<03:54, 2.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:28, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:04, 2.28MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<03:51, 3.00MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:23, 2.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:00, 2.30MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<03:48, 3.02MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:22, 2.13MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<04:47, 2.39MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<03:37, 3.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<02:41, 4.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<44:25, 256kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<32:18, 352kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<22:52, 496kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<18:36, 608kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<14:14, 794kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<10:14, 1.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<09:47, 1.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:03, 1.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:56, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<06:46, 1.65MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:56, 1.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:26, 2.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:43, 1.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:11, 2.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:52, 2.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:18, 2.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<04:55, 2.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:44, 2.95MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:10, 2.12MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<04:48, 2.28MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:36, 3.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:06, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<04:43, 2.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:35, 3.03MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:02, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<04:29, 2.41MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<03:29, 3.09MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<02:34, 4.19MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<44:48, 240kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<32:29, 331kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<22:58, 467kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<18:32, 577kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<14:08, 756kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<10:07, 1.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<09:34, 1.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<07:50, 1.35MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<05:45, 1.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:30, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:41, 1.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:12, 2.50MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<05:25, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:55, 2.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:43, 2.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<05:02, 2.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:38, 2.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:31, 2.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:59, 2.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:43, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:29, 2.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:17, 3.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<06:10, 1.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<05:26, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:04, 2.51MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:13, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:45, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:35, 2.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:53, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:31, 2.24MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<03:26, 2.95MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:45, 2.12MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:14, 2.38MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<03:16, 3.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<02:24, 4.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<39:11, 255kB/s] .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<28:31, 351kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<20:08, 496kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<16:23, 606kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<12:31, 793kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<09:00, 1.10MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<08:36, 1.15MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<06:54, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<05:15, 1.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:49, 2.56MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<31:11, 314kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<23:21, 420kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<16:39, 587kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<11:47, 827kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<12:37, 771kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<10:21, 940kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<07:33, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:29, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<07:06, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<06:33, 1.47MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:54, 1.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<03:33, 2.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<09:35, 1.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<08:16, 1.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:05, 1.57MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:54<04:24, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<08:01, 1.19MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<07:09, 1.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<05:22, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:27, 1.74MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:21, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:06, 2.29MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:33, 2.06MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:43, 1.99MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:40, 2.55MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:14, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:28, 2.09MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:27, 2.69MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:04, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:21, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:24, 2.71MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:02, 2.28MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:18, 2.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:22, 2.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:01, 2.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:09, 1.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:07, 2.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<03:00, 3.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:10, 1.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:29, 1.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:52, 1.85MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:38, 2.47MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<02:41, 3.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<09:19, 963kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<08:45, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:40, 1.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<04:47, 1.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<10:32, 846kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<09:35, 928kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<07:14, 1.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<05:10, 1.71MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<11:24, 775kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<10:03, 878kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<07:28, 1.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<05:20, 1.64MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<07:13, 1.21MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<07:03, 1.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<05:22, 1.63MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:55, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<05:09, 1.68MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<05:58, 1.46MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:02, 1.72MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:05, 2.12MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:14, 2.68MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<02:27, 3.53MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:53, 1.77MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<06:13, 1.38MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:56, 1.74MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:43, 2.31MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<02:44, 3.13MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<09:40, 884kB/s] .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<09:35, 893kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<07:23, 1.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<05:20, 1.60MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:13, 1.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:58, 1.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:30, 1.54MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<04:00, 2.11MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:31, 1.53MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<06:16, 1.34MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<04:52, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:43, 2.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:09, 2.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:49, 1.73MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:45, 2.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<02:46, 3.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:42, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<05:43, 1.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:36, 1.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<03:21, 2.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:45, 1.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<06:03, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:45, 1.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<03:25, 2.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<07:34, 1.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<07:07, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<05:22, 1.51MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<04:00, 2.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<05:08, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<06:25, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<05:12, 1.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<03:46, 2.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<02:48, 2.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<15:57, 502kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<13:55, 575kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<10:24, 768kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<07:25, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<07:30, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<07:59, 993kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<06:16, 1.26MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<04:30, 1.75MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<05:26, 1.45MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<06:23, 1.23MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<05:07, 1.53MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<03:44, 2.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:58, 1.57MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<06:00, 1.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:43, 1.65MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:36, 2.16MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:36, 2.97MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<28:35, 270kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<22:30, 343kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<16:21, 472kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<11:34, 664kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<10:26, 734kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<09:38, 795kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<07:14, 1.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<05:13, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<03:47, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<1:11:36, 106kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<52:36, 144kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<37:15, 204kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<26:20, 287kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<18:25, 408kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<54:39, 138kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<40:32, 185kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<28:49, 261kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<20:18, 369kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<14:15, 523kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<1:01:33, 121kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<45:30, 164kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<32:17, 230kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<22:45, 326kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<15:56, 463kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<1:19:19, 93.1kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<57:46, 128kB/s]   .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<40:51, 180kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<28:44, 256kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<20:07, 364kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<29:11, 251kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<22:32, 324kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<16:18, 448kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<11:30, 631kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<10:25, 695kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<09:32, 759kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<07:12, 1.00MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<05:08, 1.40MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<05:57, 1.21MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<06:20, 1.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:53, 1.47MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:39, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:38, 2.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<16:27, 432kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<13:48, 514kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<10:06, 702kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<07:20, 964kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<06:21, 1.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<06:36, 1.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:09, 1.36MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:43, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<04:49, 1.44MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<05:38, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:24, 1.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:14, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:21, 2.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<3:12:12, 35.9kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<2:16:38, 50.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<1:36:03, 71.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<1:07:03, 102kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<49:00, 139kB/s]  .vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:14<36:21, 188kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<25:50, 264kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<18:11, 374kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<14:10, 477kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<11:40, 579kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<08:51, 762kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<06:24, 1.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:37, 1.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<05:30, 1.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<06:00, 1.11MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<04:38, 1.44MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:26, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:42, 1.78MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:37, 1.43MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:44, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:44, 2.40MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:59, 1.64MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:54, 1.33MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<03:56, 1.66MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:52, 2.26MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:05, 1.58MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:50, 1.34MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<03:48, 1.70MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:53, 2.23MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:05, 3.07MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<2:59:05, 35.8kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<2:07:16, 50.4kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<1:29:21, 71.7kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<1:02:31, 102kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<44:51, 141kB/s]  .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<33:12, 191kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<23:36, 268kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<16:41, 379kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<12:49, 489kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<10:52, 577kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<08:03, 778kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<05:44, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<06:05, 1.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<06:09, 1.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<04:46, 1.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:26, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:31, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<05:08, 1.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:07, 1.48MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:59, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:51, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<04:26, 1.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:32, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:34, 2.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:53, 1.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:26, 1.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:32, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:34, 2.31MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:59, 1.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:25, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:26, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:49, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:03, 2.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<04:30, 1.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<06:11, 946kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:05, 1.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:44, 1.56MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:43, 2.13MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<05:46, 1.00MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<06:46, 855kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<05:20, 1.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:53, 1.48MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<02:52, 2.00MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:07, 2.69MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<28:16, 202kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<22:44, 252kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<16:36, 344kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<11:45, 484kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<08:18, 681kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<10:51, 520kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<10:16, 550kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<07:45, 728kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<05:46, 975kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:07, 1.36MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:24, 1.65MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:36, 1.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<06:50, 817kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<05:42, 977kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<04:10, 1.33MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:06, 1.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:19, 2.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:47, 1.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<05:08, 1.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:57, 1.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:54, 1.89MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<02:12, 2.48MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<01:43, 3.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<06:29, 838kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<07:42, 706kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<06:14, 871kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<04:34, 1.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:19, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:10, 1.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:28, 1.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:31, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<02:35, 2.06MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<01:55, 2.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:49, 1.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<06:25, 826kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<05:15, 1.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:51, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<02:47, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:08, 2.46MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<08:30, 616kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<07:28, 700kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<05:35, 935kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<04:02, 1.29MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<02:57, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<09:29, 545kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<09:18, 555kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<07:14, 714kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<05:14, 982kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<03:46, 1.35MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<04:46, 1.07MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<04:43, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:39, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:41, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<01:59, 2.53MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<11:06, 453kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<10:24, 483kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<07:58, 630kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<05:45, 870kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<04:08, 1.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<04:57, 1.00MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<04:50, 1.03MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:43, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:43, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<02:01, 2.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<10:28, 467kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<10:10, 481kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<07:46, 628kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<05:36, 868kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<04:01, 1.20MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<04:49, 1.00MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<04:41, 1.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:36, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:37, 1.82MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:57, 2.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<04:08, 1.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<05:37, 844kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<04:35, 1.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:20, 1.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:30, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<01:50, 2.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<07:19, 640kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<06:24, 730kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<04:44, 984kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:28, 1.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<02:31, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:54, 2.42MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<18:22, 251kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<15:01, 307kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<11:56, 386kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<08:41, 530kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<06:13, 737kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<04:27, 1.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<03:14, 1.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<13:17, 342kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<10:33, 430kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<07:38, 594kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<05:31, 819kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<03:57, 1.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<02:52, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<10:52, 412kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<10:14, 437kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<07:41, 581kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<05:38, 792kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<04:01, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<02:58, 1.49MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<03:58, 1.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:00, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<03:05, 1.42MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:16, 1.92MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<01:41, 2.57MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<38:28, 113kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<29:11, 149kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<20:59, 207kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<14:47, 292kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<10:23, 413kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<09:16, 460kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<08:42, 490kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<06:39, 640kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<04:46, 891kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<03:27, 1.23MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<03:35, 1.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<04:29, 934kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<03:39, 1.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:41, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:57, 2.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<04:08, 998kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<04:50, 852kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:47, 1.09MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:52, 1.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:04, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:34, 2.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<04:40, 868kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:14, 958kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:11, 1.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:18, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:56, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:57, 1.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:12, 1.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:20, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:42, 2.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<03:16, 1.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:59, 982kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:13, 1.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:20, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:42, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:24, 1.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<03:56, 976kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<03:04, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:20, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:41, 2.25MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:52, 1.32MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<03:26, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:43, 1.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:01, 1.85MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:29, 2.52MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:44, 1.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<03:19, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:35, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:59, 1.86MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:26, 2.53MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:29, 1.04MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<03:42, 983kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:51, 1.28MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:05, 1.73MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:31, 2.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<03:37, 986kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<03:41, 968kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:52, 1.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:04, 1.71MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:24, 1.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:46, 1.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:09, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:35, 2.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:09, 2.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<12:43, 271kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<09:52, 348kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<07:08, 480kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<05:01, 676kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<04:36, 731kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<04:18, 781kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:13, 1.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:21, 1.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:15, 1.47MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:26, 1.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:55, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:22, 2.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:41, 1.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<02:38, 1.22MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:01, 1.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:26, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<03:12, 984kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:57, 1.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:14, 1.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:35, 1.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<04:02, 767kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<03:29, 886kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:34, 1.20MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:52, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:00, 1.51MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:56, 1.56MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:36, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:11, 2.52MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<00:52, 3.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<06:32, 452kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<05:09, 572kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<03:43, 791kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:38, 1.11MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:43, 1.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:27, 1.17MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:51, 1.55MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:21, 2.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:58, 1.43MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:25, 1.17MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:57, 1.44MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:25, 1.96MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:41, 1.62MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:17, 1.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:51, 1.47MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:20, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<00:59, 2.70MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:44, 978kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:50, 942kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:11, 1.22MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<01:37, 1.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:09, 2.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<03:41, 707kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:29, 749kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:37, 995kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:54, 1.36MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<01:21, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<04:22, 581kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<03:52, 655kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:53, 877kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:16<02:04, 1.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<01:28, 1.69MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<05:18, 467kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<04:30, 548kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<03:20, 737kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<02:21, 1.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<01:41, 1.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<08:03, 298kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<06:22, 377kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<04:37, 517kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<03:14, 727kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:59, 780kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<02:51, 818kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<02:08, 1.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:35, 1.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<01:07, 2.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<07:47, 291kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<06:06, 371kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<04:25, 510kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<03:05, 718kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:53, 760kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:39, 826kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:00, 1.09MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<01:25, 1.50MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:40, 1.27MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:47, 1.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:22, 1.54MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:00, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:07, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:20, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:03, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:46, 2.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:59, 2.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:16, 1.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<01:00, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:45, 2.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:54, 2.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:09, 1.66MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:55, 2.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:40, 2.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:07, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:20, 1.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:03, 1.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:46, 2.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:55, 1.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:09, 1.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:55, 1.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:40, 2.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:04, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:13, 1.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:56, 1.80MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:43, 2.35MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:48, 2.05MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:00, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:48, 2.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:35, 2.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:02, 1.52MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:09, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:53, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:40, 2.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:44, 2.02MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:58, 1.54MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:46, 1.93MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:34, 2.53MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:25, 3.44MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<03:36, 398kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<02:55, 488kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<02:08, 665kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<01:29, 934kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:30, 909kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:25, 958kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:04, 1.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:45, 1.72MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:59, 1.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:02, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:48, 1.57MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:34, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:50, 1.46MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:56, 1.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:43, 1.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:31, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:37, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:45, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:35, 1.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:26, 2.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:31, 2.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:45, 1.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:43, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:41, 1.55MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:39, 1.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:38, 1.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:36, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:33, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:35, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:32, 1.91MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:35, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:32, 1.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:34, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:32, 1.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:34, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:31, 1.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:32, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:30, 2.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:31, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:29, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:31, 1.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:29, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:30, 1.97MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:28, 2.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:31, 1.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:29, 2.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:30, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:28, 2.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:29, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:27, 2.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:29, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:27, 2.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:28, 2.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:26, 2.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:27, 2.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:25, 2.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:26, 2.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:25, 2.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:26, 2.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:24, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:25, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:24, 2.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:25, 2.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:23, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:24, 2.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:23, 2.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:23, 2.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:22, 2.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:23, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:21, 2.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:22, 2.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:22, 2.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:21, 2.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:21, 2.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:20, 2.52MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:21, 2.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:20, 2.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:20, 2.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:20, 2.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:20, 2.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:19, 2.59MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:19, 2.59MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:18, 2.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<03:15, 258kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<02:24, 348kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:46, 470kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:19, 625kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:59, 822kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:45, 1.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:40, 1.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:32, 1.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:27, 1.76MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:24, 1.95MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:21, 2.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:20, 2.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:18, 2.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:55, 830kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:45, 1.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:36, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:29, 1.51MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:25, 1.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:20, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:19, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:17, 2.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:15, 2.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:14, 3.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:14, 2.99MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<01:19, 528kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<01:04, 649kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:48, 853kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:36, 1.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:28, 1.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:21, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:18, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:15, 2.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:12, 2.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<01:20, 472kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<01:09, 547kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:51, 734kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:37, 987kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:28, 1.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:21, 1.66MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:16, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:13, 2.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:10, 3.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<01:21, 418kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<01:05, 513kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:47, 700kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:34, 950kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:24, 1.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:18, 1.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:13, 2.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:27, 1.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:32, 903kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:25, 1.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:18, 1.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:13, 2.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:10, 2.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:19, 1.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:23, 1.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:18, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:09, 2.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:07, 3.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:22, 961kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:22, 968kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:16, 1.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:11, 1.69MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:07, 2.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:13, 1.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:19, 874kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:15, 1.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:10, 1.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 1.92MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:05, 2.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<01:03, 209kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:46, 278kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:31, 389kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:23<00:19, 550kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:12, 767kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:15, 577kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:14, 607kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:10, 792kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:06, 1.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:04, 1.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:04, 1.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:03, 1.29MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.77MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.40MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 948kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.23MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<25:08:13,  4.42it/s]  0%|          | 730/400000 [00:00<17:34:06,  6.31it/s]  0%|          | 1480/400000 [00:00<12:16:45,  9.02it/s]  1%|          | 2217/400000 [00:00<8:35:02, 12.87it/s]   1%|          | 2953/400000 [00:00<6:00:08, 18.37it/s]  1%|          | 3656/400000 [00:00<4:11:55, 26.22it/s]  1%|          | 4406/400000 [00:00<2:56:16, 37.40it/s]  1%|▏         | 5146/400000 [00:00<2:03:25, 53.32it/s]  1%|▏         | 5874/400000 [00:01<1:26:30, 75.93it/s]  2%|▏         | 6576/400000 [00:01<1:00:43, 107.97it/s]  2%|▏         | 7277/400000 [00:01<42:43, 153.22it/s]    2%|▏         | 7973/400000 [00:01<30:07, 216.84it/s]  2%|▏         | 8697/400000 [00:01<21:19, 305.85it/s]  2%|▏         | 9411/400000 [00:01<15:10, 429.05it/s]  3%|▎         | 10145/400000 [00:01<10:52, 597.93it/s]  3%|▎         | 10864/400000 [00:01<07:51, 824.78it/s]  3%|▎         | 11580/400000 [00:01<05:45, 1122.61it/s]  3%|▎         | 12299/400000 [00:01<04:17, 1503.02it/s]  3%|▎         | 13016/400000 [00:02<03:16, 1964.67it/s]  3%|▎         | 13746/400000 [00:02<02:33, 2516.42it/s]  4%|▎         | 14467/400000 [00:02<02:03, 3126.89it/s]  4%|▍         | 15184/400000 [00:02<01:43, 3734.62it/s]  4%|▍         | 15902/400000 [00:02<01:28, 4361.91it/s]  4%|▍         | 16630/400000 [00:02<01:17, 4957.70it/s]  4%|▍         | 17351/400000 [00:02<01:09, 5470.11it/s]  5%|▍         | 18068/400000 [00:02<01:04, 5880.48it/s]  5%|▍         | 18793/400000 [00:02<01:01, 6232.46it/s]  5%|▍         | 19532/400000 [00:02<00:58, 6537.66it/s]  5%|▌         | 20269/400000 [00:03<00:56, 6766.32it/s]  5%|▌         | 21000/400000 [00:03<00:54, 6919.28it/s]  5%|▌         | 21729/400000 [00:03<00:53, 7022.68it/s]  6%|▌         | 22458/400000 [00:03<00:54, 6933.71it/s]  6%|▌         | 23190/400000 [00:03<00:53, 7044.08it/s]  6%|▌         | 23908/400000 [00:03<00:53, 7039.38it/s]  6%|▌         | 24627/400000 [00:03<00:53, 7082.34it/s]  6%|▋         | 25362/400000 [00:03<00:52, 7160.44it/s]  7%|▋         | 26097/400000 [00:03<00:51, 7215.19it/s]  7%|▋         | 26844/400000 [00:03<00:51, 7288.52it/s]  7%|▋         | 27602/400000 [00:04<00:50, 7373.14it/s]  7%|▋         | 28342/400000 [00:04<00:50, 7368.82it/s]  7%|▋         | 29081/400000 [00:04<00:52, 7053.69it/s]  7%|▋         | 29791/400000 [00:04<00:53, 6896.85it/s]  8%|▊         | 30492/400000 [00:04<00:53, 6929.27it/s]  8%|▊         | 31188/400000 [00:04<00:54, 6775.27it/s]  8%|▊         | 31869/400000 [00:04<00:54, 6715.14it/s]  8%|▊         | 32543/400000 [00:04<00:54, 6688.00it/s]  8%|▊         | 33253/400000 [00:04<00:53, 6804.41it/s]  8%|▊         | 33980/400000 [00:04<00:52, 6935.46it/s]  9%|▊         | 34721/400000 [00:05<00:51, 7069.93it/s]  9%|▉         | 35477/400000 [00:05<00:50, 7208.94it/s]  9%|▉         | 36221/400000 [00:05<00:50, 7275.15it/s]  9%|▉         | 36950/400000 [00:05<00:50, 7194.35it/s]  9%|▉         | 37671/400000 [00:05<00:50, 7192.72it/s] 10%|▉         | 38392/400000 [00:05<00:50, 7091.49it/s] 10%|▉         | 39104/400000 [00:05<00:50, 7098.82it/s] 10%|▉         | 39856/400000 [00:05<00:49, 7219.23it/s] 10%|█         | 40581/400000 [00:05<00:49, 7227.51it/s] 10%|█         | 41305/400000 [00:05<00:49, 7182.49it/s] 11%|█         | 42055/400000 [00:06<00:49, 7274.47it/s] 11%|█         | 42784/400000 [00:06<00:49, 7256.54it/s] 11%|█         | 43511/400000 [00:06<00:49, 7258.01it/s] 11%|█         | 44238/400000 [00:06<00:49, 7207.91it/s] 11%|█         | 44960/400000 [00:06<00:49, 7176.66it/s] 11%|█▏        | 45678/400000 [00:06<00:50, 6982.79it/s] 12%|█▏        | 46378/400000 [00:06<00:50, 6949.68it/s] 12%|█▏        | 47086/400000 [00:06<00:50, 6985.91it/s] 12%|█▏        | 47789/400000 [00:06<00:50, 6996.69it/s] 12%|█▏        | 48526/400000 [00:07<00:49, 7101.98it/s] 12%|█▏        | 49260/400000 [00:07<00:48, 7169.62it/s] 12%|█▏        | 49978/400000 [00:07<00:48, 7171.15it/s] 13%|█▎        | 50698/400000 [00:07<00:48, 7179.28it/s] 13%|█▎        | 51417/400000 [00:07<00:49, 7083.55it/s] 13%|█▎        | 52144/400000 [00:07<00:48, 7137.42it/s] 13%|█▎        | 52879/400000 [00:07<00:48, 7197.03it/s] 13%|█▎        | 53600/400000 [00:07<00:48, 7182.79it/s] 14%|█▎        | 54319/400000 [00:07<00:49, 6996.52it/s] 14%|█▍        | 55020/400000 [00:07<00:49, 6996.77it/s] 14%|█▍        | 55752/400000 [00:08<00:48, 7089.98it/s] 14%|█▍        | 56473/400000 [00:08<00:48, 7123.57it/s] 14%|█▍        | 57187/400000 [00:08<00:49, 6935.47it/s] 14%|█▍        | 57890/400000 [00:08<00:49, 6962.97it/s] 15%|█▍        | 58588/400000 [00:08<00:49, 6844.92it/s] 15%|█▍        | 59339/400000 [00:08<00:48, 7029.17it/s] 15%|█▌        | 60072/400000 [00:08<00:47, 7116.50it/s] 15%|█▌        | 60821/400000 [00:08<00:46, 7222.30it/s] 15%|█▌        | 61561/400000 [00:08<00:46, 7274.64it/s] 16%|█▌        | 62290/400000 [00:08<00:46, 7236.46it/s] 16%|█▌        | 63015/400000 [00:09<00:46, 7202.26it/s] 16%|█▌        | 63769/400000 [00:09<00:46, 7299.78it/s] 16%|█▌        | 64500/400000 [00:09<00:46, 7230.37it/s] 16%|█▋        | 65236/400000 [00:09<00:46, 7266.26it/s] 16%|█▋        | 65964/400000 [00:09<00:45, 7263.79it/s] 17%|█▋        | 66691/400000 [00:09<00:46, 7245.82it/s] 17%|█▋        | 67425/400000 [00:09<00:45, 7271.60it/s] 17%|█▋        | 68175/400000 [00:09<00:45, 7337.97it/s] 17%|█▋        | 68910/400000 [00:09<00:45, 7329.54it/s] 17%|█▋        | 69644/400000 [00:09<00:45, 7324.28it/s] 18%|█▊        | 70388/400000 [00:10<00:44, 7356.68it/s] 18%|█▊        | 71124/400000 [00:10<00:44, 7325.56it/s] 18%|█▊        | 71857/400000 [00:10<00:45, 7212.22it/s] 18%|█▊        | 72608/400000 [00:10<00:44, 7298.37it/s] 18%|█▊        | 73339/400000 [00:10<00:45, 7145.56it/s] 19%|█▊        | 74095/400000 [00:10<00:44, 7264.85it/s] 19%|█▊        | 74836/400000 [00:10<00:44, 7305.67it/s] 19%|█▉        | 75568/400000 [00:10<00:44, 7300.33it/s] 19%|█▉        | 76320/400000 [00:10<00:43, 7362.22it/s] 19%|█▉        | 77057/400000 [00:10<00:44, 7329.20it/s] 19%|█▉        | 77791/400000 [00:11<00:43, 7330.21it/s] 20%|█▉        | 78525/400000 [00:11<00:43, 7318.71it/s] 20%|█▉        | 79258/400000 [00:11<00:44, 7246.28it/s] 20%|█▉        | 79983/400000 [00:11<00:45, 7079.34it/s] 20%|██        | 80693/400000 [00:11<00:45, 7048.61it/s] 20%|██        | 81453/400000 [00:11<00:44, 7205.08it/s] 21%|██        | 82200/400000 [00:11<00:43, 7279.84it/s] 21%|██        | 82930/400000 [00:11<00:43, 7267.00it/s] 21%|██        | 83670/400000 [00:11<00:43, 7304.20it/s] 21%|██        | 84402/400000 [00:11<00:43, 7181.10it/s] 21%|██▏       | 85141/400000 [00:12<00:43, 7241.89it/s] 21%|██▏       | 85866/400000 [00:12<00:43, 7228.88it/s] 22%|██▏       | 86590/400000 [00:12<00:43, 7137.69it/s] 22%|██▏       | 87332/400000 [00:12<00:43, 7216.07it/s] 22%|██▏       | 88055/400000 [00:12<00:43, 7204.66it/s] 22%|██▏       | 88776/400000 [00:12<00:43, 7133.86it/s] 22%|██▏       | 89490/400000 [00:12<00:43, 7124.78it/s] 23%|██▎       | 90203/400000 [00:12<00:43, 7060.97it/s] 23%|██▎       | 90910/400000 [00:12<00:43, 7026.83it/s] 23%|██▎       | 91613/400000 [00:13<00:44, 6973.29it/s] 23%|██▎       | 92340/400000 [00:13<00:43, 7057.79it/s] 23%|██▎       | 93047/400000 [00:13<00:43, 6977.41it/s] 23%|██▎       | 93786/400000 [00:13<00:43, 7095.78it/s] 24%|██▎       | 94527/400000 [00:13<00:42, 7185.66it/s] 24%|██▍       | 95247/400000 [00:13<00:42, 7154.70it/s] 24%|██▍       | 95964/400000 [00:13<00:42, 7105.91it/s] 24%|██▍       | 96694/400000 [00:13<00:42, 7160.93it/s] 24%|██▍       | 97433/400000 [00:13<00:41, 7225.61it/s] 25%|██▍       | 98184/400000 [00:13<00:41, 7306.50it/s] 25%|██▍       | 98916/400000 [00:14<00:41, 7280.97it/s] 25%|██▍       | 99656/400000 [00:14<00:41, 7315.78it/s] 25%|██▌       | 100388/400000 [00:14<00:41, 7154.60it/s] 25%|██▌       | 101105/400000 [00:14<00:41, 7150.03it/s] 25%|██▌       | 101848/400000 [00:14<00:41, 7231.45it/s] 26%|██▌       | 102590/400000 [00:14<00:40, 7285.95it/s] 26%|██▌       | 103340/400000 [00:14<00:40, 7346.31it/s] 26%|██▌       | 104076/400000 [00:14<00:41, 7130.53it/s] 26%|██▌       | 104834/400000 [00:14<00:40, 7257.66it/s] 26%|██▋       | 105576/400000 [00:14<00:40, 7304.33it/s] 27%|██▋       | 106308/400000 [00:15<00:40, 7218.93it/s] 27%|██▋       | 107041/400000 [00:15<00:40, 7251.46it/s] 27%|██▋       | 107774/400000 [00:15<00:40, 7272.30it/s] 27%|██▋       | 108530/400000 [00:15<00:39, 7355.63it/s] 27%|██▋       | 109281/400000 [00:15<00:39, 7400.80it/s] 28%|██▊       | 110022/400000 [00:15<00:40, 7229.24it/s] 28%|██▊       | 110747/400000 [00:15<00:41, 7013.94it/s] 28%|██▊       | 111492/400000 [00:15<00:40, 7137.39it/s] 28%|██▊       | 112230/400000 [00:15<00:39, 7206.08it/s] 28%|██▊       | 112981/400000 [00:15<00:39, 7292.17it/s] 28%|██▊       | 113719/400000 [00:16<00:39, 7316.91it/s] 29%|██▊       | 114452/400000 [00:16<00:39, 7267.89it/s] 29%|██▉       | 115190/400000 [00:16<00:39, 7299.94it/s] 29%|██▉       | 115921/400000 [00:16<00:39, 7201.11it/s] 29%|██▉       | 116642/400000 [00:16<00:39, 7134.88it/s] 29%|██▉       | 117359/400000 [00:16<00:39, 7142.65it/s] 30%|██▉       | 118079/400000 [00:16<00:39, 7157.71it/s] 30%|██▉       | 118796/400000 [00:16<00:39, 7079.95it/s] 30%|██▉       | 119511/400000 [00:16<00:39, 7097.62it/s] 30%|███       | 120232/400000 [00:16<00:39, 7130.40it/s] 30%|███       | 120954/400000 [00:17<00:38, 7156.09it/s] 30%|███       | 121706/400000 [00:17<00:38, 7260.90it/s] 31%|███       | 122434/400000 [00:17<00:38, 7264.71it/s] 31%|███       | 123161/400000 [00:17<00:38, 7186.96it/s] 31%|███       | 123884/400000 [00:17<00:38, 7198.98it/s] 31%|███       | 124605/400000 [00:17<00:39, 6992.74it/s] 31%|███▏      | 125306/400000 [00:17<00:39, 6960.62it/s] 32%|███▏      | 126049/400000 [00:17<00:38, 7093.47it/s] 32%|███▏      | 126799/400000 [00:17<00:37, 7207.77it/s] 32%|███▏      | 127522/400000 [00:17<00:38, 7156.70it/s] 32%|███▏      | 128239/400000 [00:18<00:37, 7158.06it/s] 32%|███▏      | 128995/400000 [00:18<00:37, 7272.09it/s] 32%|███▏      | 129740/400000 [00:18<00:36, 7323.87it/s] 33%|███▎      | 130485/400000 [00:18<00:36, 7360.87it/s] 33%|███▎      | 131243/400000 [00:18<00:36, 7422.27it/s] 33%|███▎      | 131986/400000 [00:18<00:36, 7327.04it/s] 33%|███▎      | 132720/400000 [00:18<00:36, 7306.99it/s] 33%|███▎      | 133455/400000 [00:18<00:36, 7317.26it/s] 34%|███▎      | 134195/400000 [00:18<00:36, 7339.09it/s] 34%|███▎      | 134949/400000 [00:19<00:35, 7396.61it/s] 34%|███▍      | 135689/400000 [00:19<00:35, 7381.60it/s] 34%|███▍      | 136428/400000 [00:19<00:35, 7338.46it/s] 34%|███▍      | 137163/400000 [00:19<00:36, 7273.52it/s] 34%|███▍      | 137900/400000 [00:19<00:35, 7301.84it/s] 35%|███▍      | 138631/400000 [00:19<00:35, 7301.58it/s] 35%|███▍      | 139362/400000 [00:19<00:36, 7207.52it/s] 35%|███▌      | 140086/400000 [00:19<00:36, 7215.01it/s] 35%|███▌      | 140808/400000 [00:19<00:36, 7149.51it/s] 35%|███▌      | 141526/400000 [00:19<00:36, 7157.70it/s] 36%|███▌      | 142242/400000 [00:20<00:36, 7148.14it/s] 36%|███▌      | 142957/400000 [00:20<00:35, 7147.24it/s] 36%|███▌      | 143689/400000 [00:20<00:35, 7198.05it/s] 36%|███▌      | 144409/400000 [00:20<00:36, 7094.16it/s] 36%|███▋      | 145119/400000 [00:20<00:36, 6991.04it/s] 36%|███▋      | 145824/400000 [00:20<00:36, 7003.55it/s] 37%|███▋      | 146547/400000 [00:20<00:35, 7069.31it/s] 37%|███▋      | 147273/400000 [00:20<00:35, 7121.36it/s] 37%|███▋      | 147986/400000 [00:20<00:35, 7041.51it/s] 37%|███▋      | 148691/400000 [00:20<00:36, 6962.72it/s] 37%|███▋      | 149391/400000 [00:21<00:35, 6972.32it/s] 38%|███▊      | 150095/400000 [00:21<00:35, 6991.17it/s] 38%|███▊      | 150808/400000 [00:21<00:35, 7027.66it/s] 38%|███▊      | 151527/400000 [00:21<00:35, 7074.31it/s] 38%|███▊      | 152235/400000 [00:21<00:35, 7058.15it/s] 38%|███▊      | 152941/400000 [00:21<00:35, 7057.90it/s] 38%|███▊      | 153647/400000 [00:21<00:35, 6973.68it/s] 39%|███▊      | 154360/400000 [00:21<00:35, 7018.06it/s] 39%|███▉      | 155118/400000 [00:21<00:34, 7175.09it/s] 39%|███▉      | 155866/400000 [00:21<00:33, 7261.80it/s] 39%|███▉      | 156598/400000 [00:22<00:33, 7276.98it/s] 39%|███▉      | 157327/400000 [00:22<00:33, 7216.56it/s] 40%|███▉      | 158050/400000 [00:22<00:34, 7093.26it/s] 40%|███▉      | 158771/400000 [00:22<00:33, 7127.08it/s] 40%|███▉      | 159490/400000 [00:22<00:33, 7143.27it/s] 40%|████      | 160205/400000 [00:22<00:33, 7126.92it/s] 40%|████      | 160919/400000 [00:22<00:33, 7098.72it/s] 40%|████      | 161630/400000 [00:22<00:34, 6953.75it/s] 41%|████      | 162327/400000 [00:22<00:34, 6802.87it/s] 41%|████      | 163034/400000 [00:22<00:34, 6878.86it/s] 41%|████      | 163751/400000 [00:23<00:33, 6961.86it/s] 41%|████      | 164459/400000 [00:23<00:33, 6994.68it/s] 41%|████▏     | 165215/400000 [00:23<00:32, 7154.67it/s] 41%|████▏     | 165932/400000 [00:23<00:32, 7125.32it/s] 42%|████▏     | 166646/400000 [00:23<00:33, 6885.52it/s] 42%|████▏     | 167401/400000 [00:23<00:32, 7071.73it/s] 42%|████▏     | 168112/400000 [00:23<00:33, 7014.53it/s] 42%|████▏     | 168816/400000 [00:23<00:32, 7021.06it/s] 42%|████▏     | 169520/400000 [00:23<00:32, 7014.96it/s] 43%|████▎     | 170248/400000 [00:23<00:32, 7090.06it/s] 43%|████▎     | 170971/400000 [00:24<00:32, 7130.88it/s] 43%|████▎     | 171685/400000 [00:24<00:32, 7044.25it/s] 43%|████▎     | 172419/400000 [00:24<00:31, 7129.11it/s] 43%|████▎     | 173142/400000 [00:24<00:31, 7157.60it/s] 43%|████▎     | 173872/400000 [00:24<00:31, 7199.68it/s] 44%|████▎     | 174604/400000 [00:24<00:31, 7233.37it/s] 44%|████▍     | 175328/400000 [00:24<00:31, 7131.57it/s] 44%|████▍     | 176042/400000 [00:24<00:32, 6988.70it/s] 44%|████▍     | 176775/400000 [00:24<00:31, 7085.97it/s] 44%|████▍     | 177533/400000 [00:25<00:30, 7225.92it/s] 45%|████▍     | 178258/400000 [00:25<00:30, 7169.86it/s] 45%|████▍     | 178977/400000 [00:25<00:31, 6961.79it/s] 45%|████▍     | 179676/400000 [00:25<00:31, 6888.14it/s] 45%|████▌     | 180407/400000 [00:25<00:31, 7007.79it/s] 45%|████▌     | 181169/400000 [00:25<00:30, 7179.68it/s] 45%|████▌     | 181895/400000 [00:25<00:30, 7201.06it/s] 46%|████▌     | 182617/400000 [00:25<00:30, 7115.35it/s] 46%|████▌     | 183343/400000 [00:25<00:30, 7156.77it/s] 46%|████▌     | 184084/400000 [00:25<00:29, 7230.05it/s] 46%|████▌     | 184810/400000 [00:26<00:29, 7237.87it/s] 46%|████▋     | 185535/400000 [00:26<00:30, 7090.04it/s] 47%|████▋     | 186246/400000 [00:26<00:31, 6847.64it/s] 47%|████▋     | 186969/400000 [00:26<00:30, 6957.32it/s] 47%|████▋     | 187667/400000 [00:26<00:30, 6951.24it/s] 47%|████▋     | 188394/400000 [00:26<00:30, 7042.75it/s] 47%|████▋     | 189109/400000 [00:26<00:29, 7074.58it/s] 47%|████▋     | 189822/400000 [00:26<00:29, 7089.99it/s] 48%|████▊     | 190566/400000 [00:26<00:29, 7189.15it/s] 48%|████▊     | 191286/400000 [00:26<00:29, 7092.31it/s] 48%|████▊     | 191997/400000 [00:27<00:29, 6979.87it/s] 48%|████▊     | 192707/400000 [00:27<00:29, 7013.83it/s] 48%|████▊     | 193410/400000 [00:27<00:29, 6893.97it/s] 49%|████▊     | 194160/400000 [00:27<00:29, 7064.07it/s] 49%|████▊     | 194910/400000 [00:27<00:28, 7187.99it/s] 49%|████▉     | 195650/400000 [00:27<00:28, 7248.61it/s] 49%|████▉     | 196377/400000 [00:27<00:28, 7194.54it/s] 49%|████▉     | 197105/400000 [00:27<00:28, 7218.09it/s] 49%|████▉     | 197840/400000 [00:27<00:27, 7254.51it/s] 50%|████▉     | 198578/400000 [00:27<00:27, 7291.13it/s] 50%|████▉     | 199308/400000 [00:28<00:27, 7237.99it/s] 50%|█████     | 200033/400000 [00:28<00:27, 7220.21it/s] 50%|█████     | 200756/400000 [00:28<00:28, 7078.62it/s] 50%|█████     | 201512/400000 [00:28<00:27, 7214.89it/s] 51%|█████     | 202252/400000 [00:28<00:27, 7268.55it/s] 51%|█████     | 202980/400000 [00:28<00:27, 7270.78it/s] 51%|█████     | 203708/400000 [00:28<00:27, 7212.32it/s] 51%|█████     | 204430/400000 [00:28<00:27, 7135.61it/s] 51%|█████▏    | 205167/400000 [00:28<00:27, 7203.38it/s] 51%|█████▏    | 205913/400000 [00:28<00:26, 7278.40it/s] 52%|█████▏    | 206642/400000 [00:29<00:26, 7249.64it/s] 52%|█████▏    | 207368/400000 [00:29<00:26, 7169.86it/s] 52%|█████▏    | 208111/400000 [00:29<00:26, 7245.36it/s] 52%|█████▏    | 208856/400000 [00:29<00:26, 7303.40it/s] 52%|█████▏    | 209593/400000 [00:29<00:26, 7321.35it/s] 53%|█████▎    | 210326/400000 [00:29<00:25, 7301.64it/s] 53%|█████▎    | 211057/400000 [00:29<00:27, 6804.60it/s] 53%|█████▎    | 211745/400000 [00:29<00:28, 6594.04it/s] 53%|█████▎    | 212496/400000 [00:29<00:27, 6842.03it/s] 53%|█████▎    | 213210/400000 [00:30<00:26, 6923.98it/s] 53%|█████▎    | 213908/400000 [00:30<00:26, 6904.01it/s] 54%|█████▎    | 214602/400000 [00:30<00:26, 6888.60it/s] 54%|█████▍    | 215294/400000 [00:30<00:27, 6837.03it/s] 54%|█████▍    | 215998/400000 [00:30<00:26, 6896.08it/s] 54%|█████▍    | 216748/400000 [00:30<00:25, 7066.62it/s] 54%|█████▍    | 217481/400000 [00:30<00:25, 7141.65it/s] 55%|█████▍    | 218220/400000 [00:30<00:25, 7212.33it/s] 55%|█████▍    | 218943/400000 [00:30<00:25, 7101.40it/s] 55%|█████▍    | 219655/400000 [00:30<00:26, 6871.84it/s] 55%|█████▌    | 220345/400000 [00:31<00:26, 6830.87it/s] 55%|█████▌    | 221068/400000 [00:31<00:25, 6943.32it/s] 55%|█████▌    | 221780/400000 [00:31<00:25, 6995.25it/s] 56%|█████▌    | 222484/400000 [00:31<00:25, 7007.82it/s] 56%|█████▌    | 223189/400000 [00:31<00:25, 7019.09it/s] 56%|█████▌    | 223892/400000 [00:31<00:25, 6984.58it/s] 56%|█████▌    | 224632/400000 [00:31<00:24, 7102.24it/s] 56%|█████▋    | 225367/400000 [00:31<00:24, 7174.22it/s] 57%|█████▋    | 226089/400000 [00:31<00:24, 7185.16it/s] 57%|█████▋    | 226840/400000 [00:31<00:23, 7277.84it/s] 57%|█████▋    | 227591/400000 [00:32<00:23, 7343.71it/s] 57%|█████▋    | 228346/400000 [00:32<00:23, 7404.00it/s] 57%|█████▋    | 229089/400000 [00:32<00:23, 7409.08it/s] 57%|█████▋    | 229831/400000 [00:32<00:23, 7267.68it/s] 58%|█████▊    | 230559/400000 [00:32<00:23, 7064.37it/s] 58%|█████▊    | 231288/400000 [00:32<00:23, 7128.75it/s] 58%|█████▊    | 232021/400000 [00:32<00:23, 7187.94it/s] 58%|█████▊    | 232748/400000 [00:32<00:23, 7210.58it/s] 58%|█████▊    | 233486/400000 [00:32<00:22, 7259.12it/s] 59%|█████▊    | 234213/400000 [00:32<00:23, 7091.21it/s] 59%|█████▊    | 234924/400000 [00:33<00:23, 6915.78it/s] 59%|█████▉    | 235632/400000 [00:33<00:23, 6964.23it/s] 59%|█████▉    | 236330/400000 [00:33<00:23, 6919.46it/s] 59%|█████▉    | 237023/400000 [00:33<00:23, 6872.47it/s] 59%|█████▉    | 237712/400000 [00:33<00:23, 6830.35it/s] 60%|█████▉    | 238405/400000 [00:33<00:23, 6859.19it/s] 60%|█████▉    | 239153/400000 [00:33<00:22, 7032.87it/s] 60%|█████▉    | 239858/400000 [00:33<00:23, 6875.79it/s] 60%|██████    | 240548/400000 [00:33<00:23, 6831.12it/s] 60%|██████    | 241252/400000 [00:34<00:23, 6891.46it/s] 61%|██████    | 242012/400000 [00:34<00:22, 7088.83it/s] 61%|██████    | 242724/400000 [00:34<00:22, 7030.73it/s] 61%|██████    | 243429/400000 [00:34<00:22, 6931.40it/s] 61%|██████    | 244124/400000 [00:34<00:22, 6891.13it/s] 61%|██████    | 244815/400000 [00:34<00:22, 6850.13it/s] 61%|██████▏   | 245526/400000 [00:34<00:22, 6924.32it/s] 62%|██████▏   | 246253/400000 [00:34<00:21, 7023.09it/s] 62%|██████▏   | 246989/400000 [00:34<00:21, 7120.14it/s] 62%|██████▏   | 247702/400000 [00:34<00:21, 6963.58it/s] 62%|██████▏   | 248458/400000 [00:35<00:21, 7130.25it/s] 62%|██████▏   | 249187/400000 [00:35<00:21, 7177.21it/s] 62%|██████▏   | 249909/400000 [00:35<00:20, 7187.92it/s] 63%|██████▎   | 250629/400000 [00:35<00:21, 7068.90it/s] 63%|██████▎   | 251338/400000 [00:35<00:21, 7006.92it/s] 63%|██████▎   | 252040/400000 [00:35<00:21, 6868.07it/s] 63%|██████▎   | 252729/400000 [00:35<00:21, 6873.22it/s] 63%|██████▎   | 253476/400000 [00:35<00:20, 7040.73it/s] 64%|██████▎   | 254234/400000 [00:35<00:20, 7192.39it/s] 64%|██████▎   | 254956/400000 [00:35<00:20, 7199.24it/s] 64%|██████▍   | 255688/400000 [00:36<00:19, 7232.59it/s] 64%|██████▍   | 256413/400000 [00:36<00:19, 7195.37it/s] 64%|██████▍   | 257134/400000 [00:36<00:20, 6994.42it/s] 64%|██████▍   | 257836/400000 [00:36<00:20, 6879.35it/s] 65%|██████▍   | 258526/400000 [00:36<00:20, 6874.45it/s] 65%|██████▍   | 259268/400000 [00:36<00:20, 7027.72it/s] 65%|██████▌   | 260001/400000 [00:36<00:19, 7114.05it/s] 65%|██████▌   | 260716/400000 [00:36<00:19, 7122.88it/s] 65%|██████▌   | 261456/400000 [00:36<00:19, 7202.02it/s] 66%|██████▌   | 262178/400000 [00:36<00:19, 7111.72it/s] 66%|██████▌   | 262891/400000 [00:37<00:19, 7059.24it/s] 66%|██████▌   | 263598/400000 [00:37<00:19, 6988.30it/s] 66%|██████▌   | 264327/400000 [00:37<00:19, 7074.68it/s] 66%|██████▋   | 265052/400000 [00:37<00:18, 7124.74it/s] 66%|██████▋   | 265807/400000 [00:37<00:18, 7244.84it/s] 67%|██████▋   | 266555/400000 [00:37<00:18, 7312.59it/s] 67%|██████▋   | 267296/400000 [00:37<00:18, 7340.14it/s] 67%|██████▋   | 268031/400000 [00:37<00:18, 7266.61it/s] 67%|██████▋   | 268759/400000 [00:37<00:18, 7087.52it/s] 67%|██████▋   | 269473/400000 [00:37<00:18, 7101.32it/s] 68%|██████▊   | 270228/400000 [00:38<00:17, 7227.62it/s] 68%|██████▊   | 270952/400000 [00:38<00:18, 7108.83it/s] 68%|██████▊   | 271667/400000 [00:38<00:18, 7121.01it/s] 68%|██████▊   | 272396/400000 [00:38<00:17, 7169.13it/s] 68%|██████▊   | 273126/400000 [00:38<00:17, 7205.15it/s] 68%|██████▊   | 273874/400000 [00:38<00:17, 7285.20it/s] 69%|██████▊   | 274604/400000 [00:38<00:17, 7057.47it/s] 69%|██████▉   | 275312/400000 [00:38<00:17, 7004.81it/s] 69%|██████▉   | 276014/400000 [00:38<00:17, 6891.71it/s] 69%|██████▉   | 276705/400000 [00:39<00:17, 6854.45it/s] 69%|██████▉   | 277439/400000 [00:39<00:17, 6992.25it/s] 70%|██████▉   | 278140/400000 [00:39<00:17, 6974.49it/s] 70%|██████▉   | 278894/400000 [00:39<00:16, 7134.65it/s] 70%|██████▉   | 279627/400000 [00:39<00:16, 7190.08it/s] 70%|███████   | 280348/400000 [00:39<00:16, 7183.86it/s] 70%|███████   | 281068/400000 [00:39<00:16, 7092.70it/s] 70%|███████   | 281779/400000 [00:39<00:16, 7021.78it/s] 71%|███████   | 282539/400000 [00:39<00:16, 7183.27it/s] 71%|███████   | 283265/400000 [00:39<00:16, 7205.73it/s] 71%|███████   | 283998/400000 [00:40<00:16, 7240.68it/s] 71%|███████   | 284723/400000 [00:40<00:16, 7161.32it/s] 71%|███████▏  | 285440/400000 [00:40<00:16, 6856.76it/s] 72%|███████▏  | 286177/400000 [00:40<00:16, 7002.16it/s] 72%|███████▏  | 286912/400000 [00:40<00:15, 7099.11it/s] 72%|███████▏  | 287634/400000 [00:40<00:15, 7133.29it/s] 72%|███████▏  | 288350/400000 [00:40<00:15, 7136.84it/s] 72%|███████▏  | 289072/400000 [00:40<00:15, 7160.24it/s] 72%|███████▏  | 289805/400000 [00:40<00:15, 7208.80it/s] 73%|███████▎  | 290527/400000 [00:40<00:15, 7001.04it/s] 73%|███████▎  | 291275/400000 [00:41<00:15, 7136.22it/s] 73%|███████▎  | 291991/400000 [00:41<00:15, 7015.94it/s] 73%|███████▎  | 292705/400000 [00:41<00:15, 7052.25it/s] 73%|███████▎  | 293466/400000 [00:41<00:14, 7210.30it/s] 74%|███████▎  | 294206/400000 [00:41<00:14, 7265.44it/s] 74%|███████▎  | 294963/400000 [00:41<00:14, 7351.42it/s] 74%|███████▍  | 295706/400000 [00:41<00:14, 7373.75it/s] 74%|███████▍  | 296462/400000 [00:41<00:13, 7428.19it/s] 74%|███████▍  | 297206/400000 [00:41<00:13, 7417.59it/s] 74%|███████▍  | 297949/400000 [00:41<00:13, 7411.96it/s] 75%|███████▍  | 298691/400000 [00:42<00:13, 7359.60it/s] 75%|███████▍  | 299445/400000 [00:42<00:13, 7411.68it/s] 75%|███████▌  | 300200/400000 [00:42<00:13, 7449.92it/s] 75%|███████▌  | 300946/400000 [00:42<00:13, 7373.12it/s] 75%|███████▌  | 301684/400000 [00:42<00:13, 7295.36it/s] 76%|███████▌  | 302415/400000 [00:42<00:13, 7298.17it/s] 76%|███████▌  | 303146/400000 [00:42<00:13, 7298.25it/s] 76%|███████▌  | 303898/400000 [00:42<00:13, 7362.91it/s] 76%|███████▌  | 304649/400000 [00:42<00:12, 7404.92it/s] 76%|███████▋  | 305390/400000 [00:42<00:13, 7201.10it/s] 77%|███████▋  | 306139/400000 [00:43<00:12, 7285.24it/s] 77%|███████▋  | 306869/400000 [00:43<00:12, 7177.98it/s] 77%|███████▋  | 307589/400000 [00:43<00:12, 7165.38it/s] 77%|███████▋  | 308326/400000 [00:43<00:12, 7223.46it/s] 77%|███████▋  | 309070/400000 [00:43<00:12, 7286.82it/s] 77%|███████▋  | 309800/400000 [00:43<00:12, 7231.27it/s] 78%|███████▊  | 310524/400000 [00:43<00:12, 7071.64it/s] 78%|███████▊  | 311233/400000 [00:43<00:12, 6975.89it/s] 78%|███████▊  | 311932/400000 [00:43<00:12, 6877.10it/s] 78%|███████▊  | 312656/400000 [00:44<00:12, 6980.73it/s] 78%|███████▊  | 313414/400000 [00:44<00:12, 7149.98it/s] 79%|███████▊  | 314165/400000 [00:44<00:11, 7253.43it/s] 79%|███████▊  | 314892/400000 [00:44<00:11, 7184.71it/s] 79%|███████▉  | 315612/400000 [00:44<00:11, 7037.12it/s] 79%|███████▉  | 316318/400000 [00:44<00:12, 6860.33it/s] 79%|███████▉  | 317014/400000 [00:44<00:12, 6888.12it/s] 79%|███████▉  | 317705/400000 [00:44<00:12, 6847.31it/s] 80%|███████▉  | 318397/400000 [00:44<00:11, 6868.79it/s] 80%|███████▉  | 319085/400000 [00:44<00:11, 6855.41it/s] 80%|███████▉  | 319772/400000 [00:45<00:11, 6755.00it/s] 80%|████████  | 320462/400000 [00:45<00:11, 6796.90it/s] 80%|████████  | 321176/400000 [00:45<00:11, 6896.31it/s] 80%|████████  | 321867/400000 [00:45<00:11, 6879.65it/s] 81%|████████  | 322579/400000 [00:45<00:11, 6947.69it/s] 81%|████████  | 323275/400000 [00:45<00:11, 6856.80it/s] 81%|████████  | 323998/400000 [00:45<00:10, 6963.35it/s] 81%|████████  | 324697/400000 [00:45<00:10, 6968.76it/s] 81%|████████▏ | 325416/400000 [00:45<00:10, 7032.20it/s] 82%|████████▏ | 326147/400000 [00:45<00:10, 7111.67it/s] 82%|████████▏ | 326878/400000 [00:46<00:10, 7167.73it/s] 82%|████████▏ | 327610/400000 [00:46<00:10, 7208.83it/s] 82%|████████▏ | 328332/400000 [00:46<00:10, 7159.42it/s] 82%|████████▏ | 329049/400000 [00:46<00:10, 7055.22it/s] 82%|████████▏ | 329813/400000 [00:46<00:09, 7219.06it/s] 83%|████████▎ | 330541/400000 [00:46<00:09, 7236.50it/s] 83%|████████▎ | 331274/400000 [00:46<00:09, 7263.40it/s] 83%|████████▎ | 332002/400000 [00:46<00:09, 7239.34it/s] 83%|████████▎ | 332727/400000 [00:46<00:09, 7146.08it/s] 83%|████████▎ | 333457/400000 [00:46<00:09, 7190.43it/s] 84%|████████▎ | 334200/400000 [00:47<00:09, 7258.67it/s] 84%|████████▎ | 334958/400000 [00:47<00:08, 7350.95it/s] 84%|████████▍ | 335711/400000 [00:47<00:08, 7402.27it/s] 84%|████████▍ | 336471/400000 [00:47<00:08, 7459.86it/s] 84%|████████▍ | 337218/400000 [00:47<00:08, 7368.95it/s] 84%|████████▍ | 337956/400000 [00:47<00:08, 7353.73it/s] 85%|████████▍ | 338692/400000 [00:47<00:08, 7337.99it/s] 85%|████████▍ | 339427/400000 [00:47<00:08, 6885.11it/s] 85%|████████▌ | 340143/400000 [00:47<00:08, 6962.69it/s] 85%|████████▌ | 340879/400000 [00:47<00:08, 7075.09it/s] 85%|████████▌ | 341590/400000 [00:48<00:08, 6831.52it/s] 86%|████████▌ | 342278/400000 [00:48<00:08, 6799.38it/s] 86%|████████▌ | 342961/400000 [00:48<00:08, 6731.00it/s] 86%|████████▌ | 343666/400000 [00:48<00:08, 6822.50it/s] 86%|████████▌ | 344373/400000 [00:48<00:08, 6892.28it/s] 86%|████████▋ | 345105/400000 [00:48<00:07, 7012.96it/s] 86%|████████▋ | 345821/400000 [00:48<00:07, 7047.53it/s] 87%|████████▋ | 346527/400000 [00:48<00:07, 7047.96it/s] 87%|████████▋ | 347240/400000 [00:48<00:07, 7070.63it/s] 87%|████████▋ | 347996/400000 [00:48<00:07, 7209.73it/s] 87%|████████▋ | 348725/400000 [00:49<00:07, 7232.97it/s] 87%|████████▋ | 349450/400000 [00:49<00:07, 7186.28it/s] 88%|████████▊ | 350170/400000 [00:49<00:07, 7048.79it/s] 88%|████████▊ | 350878/400000 [00:49<00:06, 7057.03it/s] 88%|████████▊ | 351593/400000 [00:49<00:06, 7083.55it/s] 88%|████████▊ | 352302/400000 [00:49<00:06, 6938.41it/s] 88%|████████▊ | 353024/400000 [00:49<00:06, 7019.17it/s] 88%|████████▊ | 353783/400000 [00:49<00:06, 7179.89it/s] 89%|████████▊ | 354546/400000 [00:49<00:06, 7307.00it/s] 89%|████████▉ | 355304/400000 [00:50<00:06, 7386.08it/s] 89%|████████▉ | 356044/400000 [00:50<00:05, 7388.77it/s] 89%|████████▉ | 356784/400000 [00:50<00:05, 7379.65it/s] 89%|████████▉ | 357525/400000 [00:50<00:05, 7387.45it/s] 90%|████████▉ | 358265/400000 [00:50<00:05, 7308.53it/s] 90%|████████▉ | 358997/400000 [00:50<00:05, 7162.28it/s] 90%|████████▉ | 359715/400000 [00:50<00:05, 7155.25it/s] 90%|█████████ | 360445/400000 [00:50<00:05, 7196.53it/s] 90%|█████████ | 361184/400000 [00:50<00:05, 7252.70it/s] 90%|█████████ | 361944/400000 [00:50<00:05, 7351.00it/s] 91%|█████████ | 362680/400000 [00:51<00:05, 7331.00it/s] 91%|█████████ | 363425/400000 [00:51<00:04, 7364.39it/s] 91%|█████████ | 364162/400000 [00:51<00:04, 7173.20it/s] 91%|█████████ | 364881/400000 [00:51<00:04, 7140.78it/s] 91%|█████████▏| 365619/400000 [00:51<00:04, 7208.21it/s] 92%|█████████▏| 366352/400000 [00:51<00:04, 7243.91it/s] 92%|█████████▏| 367093/400000 [00:51<00:04, 7291.58it/s] 92%|█████████▏| 367847/400000 [00:51<00:04, 7362.71it/s] 92%|█████████▏| 368599/400000 [00:51<00:04, 7408.43it/s] 92%|█████████▏| 369356/400000 [00:51<00:04, 7453.95it/s] 93%|█████████▎| 370102/400000 [00:52<00:04, 7401.40it/s] 93%|█████████▎| 370844/400000 [00:52<00:03, 7406.06it/s] 93%|█████████▎| 371585/400000 [00:52<00:03, 7348.51it/s] 93%|█████████▎| 372338/400000 [00:52<00:03, 7399.75it/s] 93%|█████████▎| 373092/400000 [00:52<00:03, 7440.53it/s] 93%|█████████▎| 373845/400000 [00:52<00:03, 7465.73it/s] 94%|█████████▎| 374592/400000 [00:52<00:03, 7450.39it/s] 94%|█████████▍| 375338/400000 [00:52<00:03, 7408.91it/s] 94%|█████████▍| 376080/400000 [00:52<00:03, 7396.74it/s] 94%|█████████▍| 376820/400000 [00:52<00:03, 7223.81it/s] 94%|█████████▍| 377549/400000 [00:53<00:03, 7240.53it/s] 95%|█████████▍| 378278/400000 [00:53<00:02, 7254.23it/s] 95%|█████████▍| 379004/400000 [00:53<00:02, 7032.64it/s] 95%|█████████▍| 379732/400000 [00:53<00:02, 7103.84it/s] 95%|█████████▌| 380475/400000 [00:53<00:02, 7198.59it/s] 95%|█████████▌| 381215/400000 [00:53<00:02, 7256.02it/s] 95%|█████████▌| 381942/400000 [00:53<00:02, 7217.80it/s] 96%|█████████▌| 382681/400000 [00:53<00:02, 7267.87it/s] 96%|█████████▌| 383436/400000 [00:53<00:02, 7347.79it/s] 96%|█████████▌| 384172/400000 [00:53<00:02, 7270.18it/s] 96%|█████████▌| 384917/400000 [00:54<00:02, 7322.13it/s] 96%|█████████▋| 385675/400000 [00:54<00:01, 7395.65it/s] 97%|█████████▋| 386433/400000 [00:54<00:01, 7447.98it/s] 97%|█████████▋| 387194/400000 [00:54<00:01, 7493.32it/s] 97%|█████████▋| 387944/400000 [00:54<00:01, 7429.53it/s] 97%|█████████▋| 388688/400000 [00:54<00:01, 7297.74it/s] 97%|█████████▋| 389419/400000 [00:54<00:01, 7164.32it/s] 98%|█████████▊| 390172/400000 [00:54<00:01, 7263.55it/s] 98%|█████████▊| 390900/400000 [00:54<00:01, 7240.77it/s] 98%|█████████▊| 391651/400000 [00:54<00:01, 7317.08it/s] 98%|█████████▊| 392386/400000 [00:55<00:01, 7326.32it/s] 98%|█████████▊| 393120/400000 [00:55<00:00, 7238.85it/s] 98%|█████████▊| 393845/400000 [00:55<00:00, 7129.99it/s] 99%|█████████▊| 394579/400000 [00:55<00:00, 7191.03it/s] 99%|█████████▉| 395299/400000 [00:55<00:00, 7114.53it/s] 99%|█████████▉| 396012/400000 [00:55<00:00, 7118.63it/s] 99%|█████████▉| 396733/400000 [00:55<00:00, 7143.15it/s] 99%|█████████▉| 397484/400000 [00:55<00:00, 7248.34it/s]100%|█████████▉| 398210/400000 [00:55<00:00, 7181.36it/s]100%|█████████▉| 398929/400000 [00:56<00:00, 7168.65it/s]100%|█████████▉| 399647/400000 [00:56<00:00, 7099.10it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7121.96it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f0b6087e160>, <torchtext.data.dataset.TabularDataset object at 0x7f0b6087e2b0>, <torchtext.vocab.Vocab object at 0x7f0b6087e1d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.54 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.54 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.54 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.54 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.86 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.86 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.39 file/s]2020-06-25 18:19:27.753850: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-25 18:19:27.759110: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-25 18:19:27.759387: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a0df3571d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-25 18:19:27.759407: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 34%|███▍      | 3366912/9912422 [00:00<00:00, 33662169.76it/s]9920512it [00:00, 32924699.96it/s]                             
0it [00:00, ?it/s]32768it [00:00, 681136.06it/s]
0it [00:00, ?it/s]  5%|▌         | 90112/1648877 [00:00<00:01, 899069.48it/s]1654784it [00:00, 12414488.05it/s]                         
0it [00:00, ?it/s]8192it [00:00, 260818.73it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

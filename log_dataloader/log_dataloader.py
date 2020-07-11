
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f860c305048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f860c305048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f8677eac1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f8677eac1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f86961f4ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f86961f4ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f86251da620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f86251da620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f86251da620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 49%|████▉     | 4833280/9912422 [00:00<00:00, 48171482.02it/s]9920512it [00:00, 30377634.37it/s]                             
0it [00:00, ?it/s]32768it [00:00, 621428.94it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 464607.51it/s]1654784it [00:00, 10786830.07it/s]                         
0it [00:00, ?it/s]8192it [00:00, 191645.50it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8622733978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f860c20ebe0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f86251da268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f86251da268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f86251da268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:29,  1.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:29,  1.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:28,  1.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:28,  1.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:27,  1.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:27,  1.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<01:01,  2.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:01,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:01,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:00,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:00,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:00,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:59,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:42,  3.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:42,  3.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:41,  3.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:41,  3.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:41,  3.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:41,  3.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:40,  3.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:29,  4.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:29,  4.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:28,  4.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:28,  4.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:28,  4.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:28,  4.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:28,  4.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:27,  4.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:20,  6.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:20,  6.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:19,  6.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:19,  6.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:19,  6.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:19,  6.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:19,  6.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:14,  9.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:14,  9.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:13,  9.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:13,  9.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:13,  9.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:13,  9.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:13,  9.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:10, 12.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:10, 12.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:09, 12.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:09, 12.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:09, 12.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 12.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:09, 12.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 16.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 16.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 16.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 16.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 16.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 16.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 16.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 20.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 20.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 20.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 20.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 20.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 20.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 20.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 20.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 25.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 30.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 30.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 30.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 30.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 30.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 30.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 30.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 35.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 35.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 35.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 35.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 35.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 35.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 35.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 40.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 40.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 40.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 40.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 40.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 40.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 40.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 44.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 44.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 44.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 44.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 44.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 44.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 44.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 44.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 44.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 50.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 56.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 56.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 56.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 56.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 56.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 56.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 56.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 56.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 56.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 60.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 60.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 60.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 60.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 60.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 60.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 60.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 60.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 60.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 64.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 66.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 69.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 69.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 69.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 69.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 69.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 69.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 69.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 69.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 69.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 71.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 71.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 71.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 71.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 74.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 74.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 74.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 75.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 75.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 75.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 75.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 75.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 75.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 75.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.99s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.99s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.99s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.92 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.18s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.99s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 75.92 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.18s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.18s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.29 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.18s/ url]
0 examples [00:00, ? examples/s]2020-07-11 06:54:55.006688: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-11 06:54:55.019334: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-11 06:54:55.019514: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56202b5bdcb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-11 06:54:55.019532: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.92 examples/s]116 examples [00:00,  5.60 examples/s]232 examples [00:00,  7.98 examples/s]346 examples [00:00, 11.36 examples/s]453 examples [00:00, 16.16 examples/s]564 examples [00:00, 22.94 examples/s]677 examples [00:00, 32.49 examples/s]791 examples [00:00, 45.85 examples/s]900 examples [00:01, 64.34 examples/s]1010 examples [00:01, 89.65 examples/s]1116 examples [00:01, 123.03 examples/s]1220 examples [00:01, 167.24 examples/s]1337 examples [00:01, 225.05 examples/s]1449 examples [00:01, 295.83 examples/s]1558 examples [00:01, 378.02 examples/s]1666 examples [00:01, 462.01 examples/s]1777 examples [00:01, 559.45 examples/s]1892 examples [00:01, 660.50 examples/s]2001 examples [00:02, 740.90 examples/s]2113 examples [00:02, 823.17 examples/s]2222 examples [00:02, 884.41 examples/s]2340 examples [00:02, 955.04 examples/s]2451 examples [00:02, 996.00 examples/s]2562 examples [00:02, 1016.93 examples/s]2672 examples [00:02, 1039.11 examples/s]2782 examples [00:02, 1039.79 examples/s]2893 examples [00:02, 1057.45 examples/s]3005 examples [00:03, 1073.30 examples/s]3115 examples [00:03, 1078.66 examples/s]3225 examples [00:03, 1074.65 examples/s]3334 examples [00:03, 1066.50 examples/s]3442 examples [00:03, 1056.12 examples/s]3551 examples [00:03, 1065.78 examples/s]3658 examples [00:03, 1049.19 examples/s]3764 examples [00:03, 1033.76 examples/s]3868 examples [00:03, 999.53 examples/s] 3972 examples [00:03, 1009.26 examples/s]4085 examples [00:04, 1042.60 examples/s]4190 examples [00:04, 1019.64 examples/s]4293 examples [00:04, 1003.69 examples/s]4394 examples [00:04, 1002.52 examples/s]4502 examples [00:04, 1022.56 examples/s]4617 examples [00:04, 1056.73 examples/s]4724 examples [00:04, 1050.68 examples/s]4830 examples [00:04, 1005.22 examples/s]4942 examples [00:04, 1035.53 examples/s]5056 examples [00:04, 1062.60 examples/s]5166 examples [00:05, 1071.39 examples/s]5276 examples [00:05, 1077.97 examples/s]5385 examples [00:05, 1075.42 examples/s]5493 examples [00:05, 1068.65 examples/s]5608 examples [00:05, 1091.42 examples/s]5718 examples [00:05, 1088.67 examples/s]5829 examples [00:05, 1093.75 examples/s]5939 examples [00:05, 1084.54 examples/s]6048 examples [00:05, 1086.15 examples/s]6157 examples [00:06, 1068.64 examples/s]6264 examples [00:06, 1051.99 examples/s]6370 examples [00:06, 1036.69 examples/s]6474 examples [00:06, 930.11 examples/s] 6570 examples [00:06, 906.15 examples/s]6663 examples [00:06, 905.53 examples/s]6763 examples [00:06, 930.27 examples/s]6858 examples [00:06, 910.20 examples/s]6950 examples [00:06, 845.62 examples/s]7049 examples [00:07, 883.12 examples/s]7150 examples [00:07, 917.54 examples/s]7254 examples [00:07, 950.42 examples/s]7363 examples [00:07, 988.02 examples/s]7464 examples [00:07, 985.63 examples/s]7566 examples [00:07, 995.01 examples/s]7669 examples [00:07, 1002.76 examples/s]7773 examples [00:07, 1011.98 examples/s]7875 examples [00:07, 995.52 examples/s] 7975 examples [00:07, 988.15 examples/s]8077 examples [00:08, 996.48 examples/s]8185 examples [00:08, 1018.61 examples/s]8288 examples [00:08, 1015.54 examples/s]8394 examples [00:08, 1026.47 examples/s]8500 examples [00:08, 1033.89 examples/s]8604 examples [00:08, 978.19 examples/s] 8716 examples [00:08, 1015.94 examples/s]8825 examples [00:08, 1034.27 examples/s]8930 examples [00:08, 1030.67 examples/s]9039 examples [00:08, 1046.99 examples/s]9155 examples [00:09, 1078.01 examples/s]9270 examples [00:09, 1096.61 examples/s]9381 examples [00:09, 1070.55 examples/s]9489 examples [00:09, 1066.28 examples/s]9602 examples [00:09, 1082.60 examples/s]9718 examples [00:09, 1103.82 examples/s]9829 examples [00:09, 1087.54 examples/s]9939 examples [00:09, 1068.75 examples/s]10047 examples [00:09, 989.51 examples/s]10150 examples [00:10, 1000.09 examples/s]10260 examples [00:10, 1026.53 examples/s]10365 examples [00:10, 1032.99 examples/s]10476 examples [00:10, 1052.77 examples/s]10582 examples [00:10, 1038.15 examples/s]10687 examples [00:10, 1029.17 examples/s]10794 examples [00:10, 1038.49 examples/s]10899 examples [00:10, 1025.70 examples/s]11010 examples [00:10, 1047.23 examples/s]11115 examples [00:10, 1027.55 examples/s]11219 examples [00:11, 1026.01 examples/s]11333 examples [00:11, 1056.90 examples/s]11443 examples [00:11, 1068.06 examples/s]11551 examples [00:11, 1058.75 examples/s]11658 examples [00:11, 1057.34 examples/s]11772 examples [00:11, 1079.68 examples/s]11886 examples [00:11, 1096.30 examples/s]11996 examples [00:11, 1075.11 examples/s]12104 examples [00:11, 1056.32 examples/s]12210 examples [00:11, 1004.77 examples/s]12315 examples [00:12, 1016.86 examples/s]12418 examples [00:12, 1015.83 examples/s]12520 examples [00:12, 1000.80 examples/s]12627 examples [00:12, 1020.01 examples/s]12735 examples [00:12, 1035.37 examples/s]12851 examples [00:12, 1067.85 examples/s]12963 examples [00:12, 1081.41 examples/s]13072 examples [00:12, 1082.78 examples/s]13181 examples [00:12, 1071.08 examples/s]13289 examples [00:12, 1041.50 examples/s]13399 examples [00:13, 1056.59 examples/s]13509 examples [00:13, 1066.29 examples/s]13616 examples [00:13, 1058.76 examples/s]13731 examples [00:13, 1083.27 examples/s]13841 examples [00:13, 1086.87 examples/s]13952 examples [00:13, 1091.99 examples/s]14066 examples [00:13, 1104.80 examples/s]14182 examples [00:13, 1118.53 examples/s]14298 examples [00:13, 1129.00 examples/s]14412 examples [00:14, 1112.00 examples/s]14527 examples [00:14, 1121.50 examples/s]14640 examples [00:14, 1098.53 examples/s]14751 examples [00:14, 1029.86 examples/s]14858 examples [00:14, 1040.77 examples/s]14968 examples [00:14, 1057.15 examples/s]15082 examples [00:14, 1080.33 examples/s]15198 examples [00:14, 1100.80 examples/s]15311 examples [00:14, 1108.00 examples/s]15423 examples [00:14, 1111.43 examples/s]15535 examples [00:15, 1102.23 examples/s]15646 examples [00:15, 1101.83 examples/s]15761 examples [00:15, 1115.16 examples/s]15876 examples [00:15, 1121.82 examples/s]15990 examples [00:15, 1126.48 examples/s]16103 examples [00:15, 1115.08 examples/s]16215 examples [00:15, 1113.04 examples/s]16327 examples [00:15, 1050.59 examples/s]16433 examples [00:15, 1035.12 examples/s]16538 examples [00:15, 1021.08 examples/s]16648 examples [00:16, 1041.21 examples/s]16760 examples [00:16, 1061.45 examples/s]16870 examples [00:16, 1071.13 examples/s]16978 examples [00:16, 1050.22 examples/s]17089 examples [00:16, 1066.64 examples/s]17196 examples [00:16, 1067.08 examples/s]17303 examples [00:16, 1065.80 examples/s]17410 examples [00:16, 1066.68 examples/s]17521 examples [00:16, 1078.00 examples/s]17631 examples [00:16, 1081.55 examples/s]17740 examples [00:17, 1068.91 examples/s]17851 examples [00:17, 1080.33 examples/s]17960 examples [00:17, 1076.01 examples/s]18068 examples [00:17, 1053.98 examples/s]18174 examples [00:17, 1009.20 examples/s]18282 examples [00:17, 1027.68 examples/s]18386 examples [00:17, 1030.96 examples/s]18490 examples [00:17, 1028.37 examples/s]18594 examples [00:17, 988.67 examples/s] 18694 examples [00:18, 959.03 examples/s]18791 examples [00:18, 921.10 examples/s]18884 examples [00:18, 885.05 examples/s]18981 examples [00:18, 908.46 examples/s]19085 examples [00:18, 942.19 examples/s]19183 examples [00:18, 953.14 examples/s]19284 examples [00:18, 968.32 examples/s]19387 examples [00:18, 984.64 examples/s]19496 examples [00:18, 1011.82 examples/s]19607 examples [00:18, 1039.15 examples/s]19712 examples [00:19, 1030.68 examples/s]19819 examples [00:19, 1041.25 examples/s]19926 examples [00:19, 1047.02 examples/s]20031 examples [00:19, 975.33 examples/s] 20139 examples [00:19, 1003.39 examples/s]20248 examples [00:19, 1026.83 examples/s]20357 examples [00:19, 1044.54 examples/s]20463 examples [00:19, 1043.44 examples/s]20575 examples [00:19, 1064.94 examples/s]20686 examples [00:20, 1076.57 examples/s]20794 examples [00:20, 1072.20 examples/s]20905 examples [00:20, 1081.75 examples/s]21014 examples [00:20, 1075.32 examples/s]21125 examples [00:20, 1083.73 examples/s]21234 examples [00:20, 1022.68 examples/s]21338 examples [00:20, 1016.01 examples/s]21445 examples [00:20, 1031.39 examples/s]21557 examples [00:20, 1054.94 examples/s]21665 examples [00:20, 1060.41 examples/s]21779 examples [00:21, 1081.74 examples/s]21888 examples [00:21, 1059.77 examples/s]22000 examples [00:21, 1076.22 examples/s]22108 examples [00:21, 1076.27 examples/s]22223 examples [00:21, 1095.51 examples/s]22337 examples [00:21, 1108.46 examples/s]22449 examples [00:21, 1092.36 examples/s]22559 examples [00:21, 1090.05 examples/s]22669 examples [00:21, 1085.45 examples/s]22780 examples [00:21, 1090.05 examples/s]22890 examples [00:22, 1080.96 examples/s]22999 examples [00:22, 1069.84 examples/s]23112 examples [00:22, 1087.00 examples/s]23221 examples [00:22, 1080.65 examples/s]23330 examples [00:22, 1078.62 examples/s]23438 examples [00:22, 1038.50 examples/s]23543 examples [00:22, 1029.28 examples/s]23647 examples [00:22, 1006.23 examples/s]23758 examples [00:22, 1035.22 examples/s]23868 examples [00:23, 1051.88 examples/s]23974 examples [00:23, 1020.05 examples/s]24077 examples [00:23, 1019.52 examples/s]24180 examples [00:23, 1005.93 examples/s]24293 examples [00:23, 1039.78 examples/s]24399 examples [00:23, 1045.26 examples/s]24504 examples [00:23, 1014.99 examples/s]24608 examples [00:23, 1021.97 examples/s]24721 examples [00:23, 1050.10 examples/s]24831 examples [00:23, 1063.06 examples/s]24947 examples [00:24, 1088.42 examples/s]25057 examples [00:24, 1074.57 examples/s]25165 examples [00:24, 1071.91 examples/s]25278 examples [00:24, 1086.30 examples/s]25394 examples [00:24, 1104.98 examples/s]25509 examples [00:24, 1117.56 examples/s]25622 examples [00:24, 1121.04 examples/s]25735 examples [00:24, 1106.51 examples/s]25848 examples [00:24, 1111.69 examples/s]25962 examples [00:24, 1117.49 examples/s]26076 examples [00:25, 1123.21 examples/s]26189 examples [00:25, 1119.55 examples/s]26302 examples [00:25, 1097.38 examples/s]26417 examples [00:25, 1110.70 examples/s]26529 examples [00:25, 1104.13 examples/s]26640 examples [00:25, 1104.59 examples/s]26751 examples [00:25, 1085.86 examples/s]26860 examples [00:25, 1079.53 examples/s]26973 examples [00:25, 1093.39 examples/s]27088 examples [00:25, 1109.62 examples/s]27200 examples [00:26, 1104.77 examples/s]27311 examples [00:26, 1050.04 examples/s]27417 examples [00:26, 1028.83 examples/s]27521 examples [00:26, 1025.47 examples/s]27624 examples [00:26, 1025.82 examples/s]27727 examples [00:26, 1013.32 examples/s]27829 examples [00:26, 946.70 examples/s] 27925 examples [00:26, 944.90 examples/s]28029 examples [00:26, 971.20 examples/s]28141 examples [00:27, 1009.55 examples/s]28246 examples [00:27, 1019.22 examples/s]28349 examples [00:27, 1017.80 examples/s]28452 examples [00:27, 1016.71 examples/s]28562 examples [00:27, 1038.84 examples/s]28674 examples [00:27, 1059.33 examples/s]28789 examples [00:27, 1082.60 examples/s]28898 examples [00:27, 1055.76 examples/s]29005 examples [00:27, 1059.20 examples/s]29112 examples [00:27, 1053.55 examples/s]29223 examples [00:28, 1069.25 examples/s]29331 examples [00:28, 1033.28 examples/s]29435 examples [00:28, 998.95 examples/s] 29536 examples [00:28, 953.26 examples/s]29633 examples [00:28, 908.55 examples/s]29725 examples [00:28, 903.51 examples/s]29817 examples [00:28, 907.88 examples/s]29912 examples [00:28, 919.65 examples/s]30005 examples [00:28, 883.74 examples/s]30109 examples [00:29, 924.56 examples/s]30218 examples [00:29, 968.60 examples/s]30329 examples [00:29, 1005.98 examples/s]30431 examples [00:29, 990.78 examples/s] 30539 examples [00:29, 1015.67 examples/s]30648 examples [00:29, 1035.28 examples/s]30758 examples [00:29, 1051.44 examples/s]30864 examples [00:29, 1020.87 examples/s]30967 examples [00:29, 1022.40 examples/s]31080 examples [00:29, 1051.67 examples/s]31196 examples [00:30, 1081.61 examples/s]31309 examples [00:30, 1095.64 examples/s]31419 examples [00:30, 1085.20 examples/s]31528 examples [00:30, 1071.90 examples/s]31641 examples [00:30, 1088.63 examples/s]31757 examples [00:30, 1108.74 examples/s]31870 examples [00:30, 1112.53 examples/s]31982 examples [00:30, 1097.37 examples/s]32092 examples [00:30, 1057.68 examples/s]32203 examples [00:30, 1071.12 examples/s]32319 examples [00:31, 1094.34 examples/s]32434 examples [00:31, 1109.51 examples/s]32546 examples [00:31, 1103.01 examples/s]32657 examples [00:31, 1075.00 examples/s]32770 examples [00:31, 1090.31 examples/s]32886 examples [00:31, 1108.25 examples/s]32998 examples [00:31, 1091.21 examples/s]33108 examples [00:31, 1039.89 examples/s]33213 examples [00:31, 1034.73 examples/s]33317 examples [00:32, 1023.59 examples/s]33432 examples [00:32, 1056.24 examples/s]33544 examples [00:32, 1072.30 examples/s]33652 examples [00:32, 1064.77 examples/s]33760 examples [00:32, 1067.94 examples/s]33874 examples [00:32, 1086.22 examples/s]33990 examples [00:32, 1104.98 examples/s]34101 examples [00:32, 1087.66 examples/s]34210 examples [00:32, 1050.74 examples/s]34316 examples [00:32, 1041.18 examples/s]34421 examples [00:33, 1032.63 examples/s]34525 examples [00:33, 1004.35 examples/s]34626 examples [00:33, 1003.58 examples/s]34727 examples [00:33, 1000.44 examples/s]34836 examples [00:33, 1023.82 examples/s]34939 examples [00:33, 1022.19 examples/s]35044 examples [00:33, 1029.52 examples/s]35148 examples [00:33, 1013.73 examples/s]35250 examples [00:33, 1009.06 examples/s]35353 examples [00:33, 1013.96 examples/s]35457 examples [00:34, 1020.20 examples/s]35565 examples [00:34, 1036.47 examples/s]35669 examples [00:34, 1004.53 examples/s]35771 examples [00:34, 1008.15 examples/s]35873 examples [00:34, 1009.21 examples/s]35979 examples [00:34, 1021.42 examples/s]36087 examples [00:34, 1036.68 examples/s]36196 examples [00:34, 1049.51 examples/s]36302 examples [00:34, 1040.74 examples/s]36410 examples [00:35, 1051.96 examples/s]36520 examples [00:35, 1063.17 examples/s]36627 examples [00:35, 1051.81 examples/s]36733 examples [00:35, 1033.03 examples/s]36837 examples [00:35, 1017.73 examples/s]36943 examples [00:35, 1028.89 examples/s]37048 examples [00:35, 1032.78 examples/s]37152 examples [00:35, 1023.13 examples/s]37255 examples [00:35, 985.80 examples/s] 37357 examples [00:35, 994.31 examples/s]37463 examples [00:36, 1012.55 examples/s]37567 examples [00:36, 1017.27 examples/s]37669 examples [00:36, 991.86 examples/s] 37769 examples [00:36, 970.58 examples/s]37867 examples [00:36, 956.29 examples/s]37975 examples [00:36, 989.97 examples/s]38081 examples [00:36, 1008.70 examples/s]38189 examples [00:36, 1027.84 examples/s]38297 examples [00:36, 1042.72 examples/s]38402 examples [00:36, 1007.83 examples/s]38515 examples [00:37, 1041.18 examples/s]38628 examples [00:37, 1064.19 examples/s]38735 examples [00:37, 1050.91 examples/s]38849 examples [00:37, 1074.08 examples/s]38957 examples [00:37, 1068.39 examples/s]39069 examples [00:37, 1082.99 examples/s]39184 examples [00:37, 1099.90 examples/s]39295 examples [00:37, 1085.55 examples/s]39404 examples [00:37, 1086.55 examples/s]39513 examples [00:37, 1077.06 examples/s]39621 examples [00:38, 1064.34 examples/s]39730 examples [00:38, 1068.49 examples/s]39837 examples [00:38, 1063.17 examples/s]39946 examples [00:38, 1070.01 examples/s]40054 examples [00:38, 1018.65 examples/s]40160 examples [00:38, 1028.69 examples/s]40266 examples [00:38, 1035.23 examples/s]40372 examples [00:38, 1037.54 examples/s]40476 examples [00:38, 993.16 examples/s] 40587 examples [00:39, 1023.58 examples/s]40701 examples [00:39, 1054.38 examples/s]40808 examples [00:39, 1058.56 examples/s]40915 examples [00:39, 1055.70 examples/s]41022 examples [00:39, 1057.65 examples/s]41135 examples [00:39, 1076.01 examples/s]41250 examples [00:39, 1095.63 examples/s]41364 examples [00:39, 1107.65 examples/s]41475 examples [00:39, 1086.67 examples/s]41584 examples [00:39, 1071.30 examples/s]41692 examples [00:40, 1073.28 examples/s]41805 examples [00:40, 1087.11 examples/s]41914 examples [00:40, 1087.67 examples/s]42023 examples [00:40, 1087.75 examples/s]42133 examples [00:40, 1090.44 examples/s]42243 examples [00:40, 1080.33 examples/s]42352 examples [00:40, 1060.81 examples/s]42459 examples [00:40, 1023.83 examples/s]42562 examples [00:40, 1022.23 examples/s]42665 examples [00:40, 1016.59 examples/s]42778 examples [00:41, 1046.76 examples/s]42891 examples [00:41, 1068.51 examples/s]42999 examples [00:41, 1069.37 examples/s]43109 examples [00:41, 1077.23 examples/s]43217 examples [00:41, 1070.25 examples/s]43325 examples [00:41, 1065.59 examples/s]43437 examples [00:41, 1080.10 examples/s]43550 examples [00:41, 1093.80 examples/s]43663 examples [00:41, 1102.69 examples/s]43774 examples [00:42, 1012.46 examples/s]43886 examples [00:42, 1040.45 examples/s]43994 examples [00:42, 1051.60 examples/s]44107 examples [00:42, 1071.27 examples/s]44220 examples [00:42, 1087.35 examples/s]44330 examples [00:42, 1080.84 examples/s]44441 examples [00:42, 1086.36 examples/s]44550 examples [00:42, 1055.44 examples/s]44656 examples [00:42, 1031.58 examples/s]44760 examples [00:42, 1027.95 examples/s]44864 examples [00:43, 1019.29 examples/s]44967 examples [00:43, 1021.24 examples/s]45070 examples [00:43, 1014.41 examples/s]45172 examples [00:43, 1007.19 examples/s]45279 examples [00:43, 1024.07 examples/s]45382 examples [00:43, 1014.60 examples/s]45490 examples [00:43, 1032.13 examples/s]45594 examples [00:43, 1032.30 examples/s]45698 examples [00:43, 1004.95 examples/s]45799 examples [00:43, 993.65 examples/s] 45900 examples [00:44, 997.30 examples/s]46004 examples [00:44, 1007.15 examples/s]46110 examples [00:44, 1020.44 examples/s]46215 examples [00:44, 1026.74 examples/s]46318 examples [00:44, 1022.25 examples/s]46421 examples [00:44, 1007.98 examples/s]46524 examples [00:44, 1013.95 examples/s]46631 examples [00:44, 1028.94 examples/s]46735 examples [00:44, 1028.97 examples/s]46845 examples [00:44, 1047.45 examples/s]46950 examples [00:45, 992.26 examples/s] 47058 examples [00:45, 1014.11 examples/s]47161 examples [00:45, 995.64 examples/s] 47263 examples [00:45, 1001.61 examples/s]47364 examples [00:45, 998.20 examples/s] 47468 examples [00:45, 1009.87 examples/s]47575 examples [00:45, 1025.06 examples/s]47682 examples [00:45, 1037.46 examples/s]47788 examples [00:45, 1040.84 examples/s]47893 examples [00:46, 1007.22 examples/s]47995 examples [00:46, 963.93 examples/s] 48093 examples [00:46, 944.20 examples/s]48197 examples [00:46, 970.73 examples/s]48299 examples [00:46, 984.07 examples/s]48403 examples [00:46, 997.87 examples/s]48504 examples [00:46, 1000.47 examples/s]48617 examples [00:46, 1033.86 examples/s]48729 examples [00:46, 1056.64 examples/s]48836 examples [00:46, 1052.37 examples/s]48946 examples [00:47, 1061.40 examples/s]49053 examples [00:47, 1026.26 examples/s]49160 examples [00:47, 1036.67 examples/s]49264 examples [00:47, 1016.61 examples/s]49376 examples [00:47, 1045.10 examples/s]49481 examples [00:47, 1035.42 examples/s]49585 examples [00:47, 1031.05 examples/s]49689 examples [00:47, 1016.40 examples/s]49798 examples [00:47, 1035.68 examples/s]49905 examples [00:48, 1044.41 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 7924/50000 [00:00<00:00, 79238.94 examples/s] 44%|████▍     | 22142/50000 [00:00<00:00, 91373.39 examples/s] 71%|███████   | 35410/50000 [00:00<00:00, 100784.99 examples/s] 98%|█████████▊| 48966/50000 [00:00<00:00, 109187.21 examples/s]                                                                0 examples [00:00, ? examples/s]82 examples [00:00, 817.05 examples/s]189 examples [00:00, 878.20 examples/s]289 examples [00:00, 910.41 examples/s]393 examples [00:00, 944.12 examples/s]497 examples [00:00, 969.42 examples/s]598 examples [00:00, 979.23 examples/s]705 examples [00:00, 1003.31 examples/s]815 examples [00:00, 1030.19 examples/s]918 examples [00:00, 1028.73 examples/s]1023 examples [00:01, 1033.11 examples/s]1125 examples [00:01, 1026.72 examples/s]1236 examples [00:01, 1050.01 examples/s]1341 examples [00:01, 1043.60 examples/s]1445 examples [00:01, 1029.67 examples/s]1550 examples [00:01, 1033.42 examples/s]1657 examples [00:01, 1041.85 examples/s]1764 examples [00:01, 1045.98 examples/s]1871 examples [00:01, 1050.16 examples/s]1976 examples [00:01, 1047.41 examples/s]2081 examples [00:02, 1040.83 examples/s]2186 examples [00:02, 1030.24 examples/s]2290 examples [00:02, 998.03 examples/s] 2395 examples [00:02, 1010.98 examples/s]2500 examples [00:02, 1020.10 examples/s]2611 examples [00:02, 1044.69 examples/s]2716 examples [00:02, 984.38 examples/s] 2824 examples [00:02, 1008.87 examples/s]2939 examples [00:02, 1045.82 examples/s]3051 examples [00:02, 1065.28 examples/s]3166 examples [00:03, 1088.13 examples/s]3276 examples [00:03, 1077.30 examples/s]3385 examples [00:03, 1076.89 examples/s]3497 examples [00:03, 1087.36 examples/s]3606 examples [00:03, 1087.32 examples/s]3715 examples [00:03, 1085.20 examples/s]3824 examples [00:03, 1084.49 examples/s]3933 examples [00:03, 1084.95 examples/s]4042 examples [00:03, 1086.09 examples/s]4153 examples [00:03, 1093.13 examples/s]4263 examples [00:04, 1089.80 examples/s]4373 examples [00:04, 1080.50 examples/s]4482 examples [00:04, 1071.32 examples/s]4590 examples [00:04, 1044.81 examples/s]4695 examples [00:04, 1017.28 examples/s]4799 examples [00:04, 1023.32 examples/s]4906 examples [00:04, 1035.03 examples/s]5016 examples [00:04, 1052.43 examples/s]5129 examples [00:04, 1073.29 examples/s]5242 examples [00:04, 1087.69 examples/s]5355 examples [00:05, 1097.14 examples/s]5465 examples [00:05, 1082.54 examples/s]5577 examples [00:05, 1091.42 examples/s]5687 examples [00:05, 1085.16 examples/s]5796 examples [00:05, 1079.38 examples/s]5905 examples [00:05, 1075.36 examples/s]6013 examples [00:05, 1002.12 examples/s]6115 examples [00:05, 1003.63 examples/s]6227 examples [00:05, 1033.55 examples/s]6337 examples [00:06, 1051.77 examples/s]6443 examples [00:06, 1042.31 examples/s]6548 examples [00:06, 997.55 examples/s] 6659 examples [00:06, 1027.01 examples/s]6775 examples [00:06, 1060.88 examples/s]6884 examples [00:06, 1068.64 examples/s]6996 examples [00:06, 1081.60 examples/s]7105 examples [00:06, 1065.36 examples/s]7217 examples [00:06, 1080.04 examples/s]7331 examples [00:06, 1095.65 examples/s]7442 examples [00:07, 1099.15 examples/s]7554 examples [00:07, 1104.56 examples/s]7665 examples [00:07, 1086.74 examples/s]7776 examples [00:07, 1092.85 examples/s]7886 examples [00:07, 1071.05 examples/s]7999 examples [00:07, 1086.11 examples/s]8108 examples [00:07, 1064.13 examples/s]8219 examples [00:07, 1076.95 examples/s]8333 examples [00:07, 1093.72 examples/s]8446 examples [00:07, 1103.81 examples/s]8563 examples [00:08, 1119.75 examples/s]8677 examples [00:08, 1125.00 examples/s]8790 examples [00:08, 1117.27 examples/s]8902 examples [00:08, 1098.47 examples/s]9012 examples [00:08, 1061.17 examples/s]9123 examples [00:08, 1074.14 examples/s]9234 examples [00:08, 1084.35 examples/s]9343 examples [00:08, 1028.19 examples/s]9460 examples [00:08, 1064.38 examples/s]9573 examples [00:09, 1081.45 examples/s]9686 examples [00:09, 1092.70 examples/s]9796 examples [00:09, 1092.99 examples/s]9906 examples [00:09, 1083.58 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteFQ3EQT/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteFQ3EQT/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f86251da620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f86251da620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f86251da620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8622733940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f85ae6465c0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f86251da620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f86251da620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f86251da620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f860c20e5c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f860c20e358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f85ac15f268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f85ac15f268> 

  function with postional parmater data_info <function split_train_valid at 0x7f85ac15f268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f85ac15f378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f85ac15f378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f85ac15f378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=68f865d1b1753b3a576a33aa3ecaada31b70bd2536eda6227e1269153e3afe21
  Stored in directory: /tmp/pip-ephem-wheel-cache-8gzacry7/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:22:58, 10.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:36:03, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:40:26, 20.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<8:10:38, 29.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.19M/862M [00:01<5:42:43, 41.8kB/s].vector_cache/glove.6B.zip:   1%|          | 6.76M/862M [00:01<3:59:03, 59.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<2:46:20, 85.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.5M/862M [00:01<1:56:07, 122kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:01<1:20:50, 173kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.7M/862M [00:01<56:17, 247kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:01<39:24, 352kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.2M/862M [00:02<27:31, 501kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.7M/862M [00:02<19:15, 713kB/s].vector_cache/glove.6B.zip:   5%|▍         | 43.0M/862M [00:02<13:30, 1.01MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.2M/862M [00:02<09:30, 1.43MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<06:49, 1.98MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:04<06:40, 2.01MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<08:29, 1.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<06:50, 1.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.7M/862M [00:04<05:00, 2.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<09:15, 1.44MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<08:16, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:06<06:09, 2.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:08<06:53, 1.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<07:53, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<06:13, 2.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.6M/862M [00:08<04:31, 2.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<15:17, 866kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<12:07, 1.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:10<08:47, 1.50MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<09:01, 1.46MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<07:38, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:12<05:40, 2.31MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<07:02, 1.86MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<07:33, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<05:50, 2.24MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.8M/862M [00:14<04:14, 3.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<11:44, 1.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<09:33, 1.36MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:16<07:00, 1.85MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<07:56, 1.63MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<06:51, 1.89MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:18<05:04, 2.55MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<06:37, 1.95MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<07:13, 1.78MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<05:41, 2.26MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:20<04:06, 3.11MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<1:49:11, 117kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<1:17:41, 165kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:22<54:33, 234kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:24<41:05, 310kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<31:18, 407kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<22:30, 566kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<15:51, 800kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<2:12:55, 95.4kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<1:34:18, 134kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<1:06:09, 191kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<49:09, 257kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<36:56, 341kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<26:27, 476kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<20:29, 612kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<15:36, 804kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<11:10, 1.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<10:44, 1.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<10:02, 1.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:34, 1.65MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<05:24, 2.29MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<22:34, 550kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<17:04, 726kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<12:14, 1.01MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<11:27, 1.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<09:15, 1.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<06:43, 1.83MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<07:37, 1.61MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:49, 1.57MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:05, 2.01MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<06:13, 1.96MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:36, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:11, 2.91MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:46, 2.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<06:23, 1.90MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:04, 2.38MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<03:40, 3.28MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<11:40:58, 17.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<8:11:40, 24.5kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<5:43:44, 35.0kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<4:02:43, 49.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<2:52:15, 69.6kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<2:00:58, 99.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<1:24:29, 141kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<1:14:28, 160kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<53:17, 224kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<37:31, 317kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<28:59, 409kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<22:40, 523kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<16:21, 724kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<11:35, 1.02MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<13:36, 867kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<10:42, 1.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<07:48, 1.51MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<05:35, 2.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<47:35, 246kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<35:39, 329kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<25:32, 459kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<17:58, 649kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<18:46, 621kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<14:42, 793kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<10:36, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<09:39, 1.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<09:43, 1.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<07:27, 1.55MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:57<05:22, 2.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<07:52, 1.46MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<07:01, 1.64MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<05:13, 2.20MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<03:47, 3.02MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<24:03, 476kB/s] .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<19:54, 575kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<14:40, 779kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<10:25, 1.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<12:37, 901kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<10:19, 1.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<07:31, 1.51MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:03<05:22, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<5:42:17, 33.0kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<4:02:26, 46.6kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<2:50:11, 66.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<1:58:59, 94.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<1:26:36, 130kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<1:02:01, 181kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<43:42, 257kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<32:39, 342kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<25:51, 432kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<18:42, 596kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<13:14, 840kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<12:37, 880kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<10:16, 1.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<07:29, 1.48MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:22, 1.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<08:09, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:19, 1.74MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<04:34, 2.40MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:57, 1.58MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<06:20, 1.73MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:47, 2.28MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:27, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:46, 1.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:27, 1.99MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<03:57, 2.74MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:51, 1.38MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:55, 1.56MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:08, 2.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:40, 1.90MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:57, 1.55MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:28, 1.96MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:21<03:57, 2.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<07:01, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<06:20, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:47, 2.23MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:24, 1.96MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:37, 1.60MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:19, 1.99MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:52, 2.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:39, 1.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:44, 1.56MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:03, 2.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:33, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:17, 1.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:59, 2.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:49, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:10, 1.69MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:00, 2.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<03:39, 2.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<07:24, 1.40MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:32, 1.58MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:51, 2.12MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:23, 1.91MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<06:32, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:14, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<03:49, 2.67MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:27, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:35, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:56, 2.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:25, 1.87MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<06:33, 1.55MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:08, 1.97MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<03:44, 2.70MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:13, 1.61MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:41, 1.77MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:18, 2.33MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:56, 2.02MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<06:09, 1.62MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<04:57, 2.01MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<03:37, 2.74MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<07:11, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:21, 1.56MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:43, 2.09MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:13, 1.89MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:18, 1.56MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:47<05:03, 1.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:41, 2.66MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<07:09, 1.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:17, 1.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:43, 2.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:10, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<06:13, 1.56MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:53, 1.98MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:34, 2.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:25, 1.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:05, 1.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:52, 2.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:33, 2.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:45, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:32, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:20, 2.85MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:02, 1.89MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:46, 1.99MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:38, 2.60MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:21, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<05:36, 1.68MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:27, 2.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<03:15, 2.88MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:15, 1.78MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:56, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<03:43, 2.51MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<04:25, 2.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:37, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:27, 2.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<03:14, 2.86MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<06:08, 1.51MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:30, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:08, 2.22MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:39, 1.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 313M/862M [02:07<05:44, 1.59MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:37, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<03:22, 2.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<06:37, 1.37MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:40, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:11, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:49, 1.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<05:47, 1.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:34, 1.97MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<03:18, 2.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<06:05, 1.47MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<05:26, 1.65MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:05, 2.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:34, 1.94MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<05:29, 1.62MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:25, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<03:13, 2.73MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<06:23, 1.38MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<05:39, 1.56MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:11, 2.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<03:02, 2.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<11:19, 772kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<09:04, 963kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<06:37, 1.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<06:18, 1.38MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<06:37, 1.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:12, 1.67MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<03:44, 2.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<06:24, 1.34MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<05:38, 1.53MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:10, 2.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:34, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:19, 1.97MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<03:18, 2.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:56, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:54, 1.72MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<03:54, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<02:55, 2.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:59, 2.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:54, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:00, 2.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:42, 2.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:42, 2.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<02:52, 2.90MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:36, 2.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:37, 1.79MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:47, 2.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:45, 2.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:46, 1.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:07, 1.60MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<03:51, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<04:15, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<04:50, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:51, 2.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<02:47, 2.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<08:17, 971kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<06:53, 1.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<05:02, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<05:03, 1.58MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<05:41, 1.40MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<04:25, 1.80MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:11, 2.48MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:26, 1.46MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:50, 1.63MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<03:38, 2.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:03, 1.93MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:56, 1.59MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:58, 1.97MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<02:53, 2.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:40, 1.37MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:00, 1.55MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<03:42, 2.09MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:05, 1.88MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:43, 2.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<02:51, 2.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:28, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:29, 1.70MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:35, 2.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:38, 2.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<05:15, 1.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:40, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:28, 2.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:52, 1.93MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:43, 1.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:48, 1.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:46, 2.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:24, 1.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:47, 1.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<03:35, 2.07MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<03:55, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<08:40, 848kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<07:30, 981kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:32, 1.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<03:58, 1.84MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:42, 1.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<06:42, 1.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:15, 1.39MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<03:50, 1.89MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:28, 1.62MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<05:48, 1.24MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:36, 1.57MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<03:22, 2.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:10, 1.71MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:23, 1.33MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:22, 1.63MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<03:12, 2.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:05, 1.74MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:08, 1.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:06, 1.73MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:01, 2.34MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:58, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:02, 1.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<04:05, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<02:58, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:59, 1.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:53, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:52, 1.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:50, 2.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:55, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:49, 1.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:53, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:50, 2.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:56, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:48, 1.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:48, 1.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:48, 2.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:53, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:26, 1.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:31, 1.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:19, 2.02MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<02:26, 2.73MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<31:11, 214kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<24:45, 269kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<18:01, 370kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<12:44, 521kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<08:58, 735kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<1:05:06, 101kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<48:12, 137kB/s]  .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<34:22, 192kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<24:07, 272kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<18:15, 358kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<15:24, 424kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<11:24, 572kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<08:05, 803kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<07:07, 908kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<07:24, 873kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<05:48, 1.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:11, 1.53MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<04:23, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<05:26, 1.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:17, 1.49MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:09, 2.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:40, 1.72MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:56, 1.28MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:57, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<02:54, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:29, 1.79MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<04:46, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<03:55, 1.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:53, 2.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:27, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:43, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:46, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:47, 2.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<02:02, 2.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<11:57, 512kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<10:38, 575kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<07:54, 772kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<05:40, 1.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<05:22, 1.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<05:51, 1.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:33, 1.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:19, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<03:44, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<04:50, 1.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:55, 1.52MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:50, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:26, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:26, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:33, 1.66MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:37, 2.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:12, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:25, 1.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:31, 1.65MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:36, 2.23MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:12, 1.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<04:14, 1.36MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:29, 1.65MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:34, 2.23MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:08, 1.81MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<04:19, 1.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:28, 1.64MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:33, 2.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:06, 1.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<04:07, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:23, 1.66MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:28, 2.26MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:04, 1.81MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<04:03, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:20, 1.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<02:27, 2.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:01, 1.82MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:09, 1.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:18, 1.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:26, 2.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:59, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:58, 1.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:15, 1.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<02:23, 2.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:56, 1.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:03, 1.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:13, 1.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:22, 2.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<01:44, 3.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<17:24, 304kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<14:00, 378kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<10:15, 514kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<07:16, 722kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<06:18, 828kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<06:22, 818kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:00<04:55, 1.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:32, 1.47MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<02:32, 2.03MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<22:05, 233kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<17:14, 299kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<12:31, 411kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<08:50, 578kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<07:22, 689kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<07:03, 720kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:23, 942kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:52, 1.30MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:53, 1.29MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:27, 1.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:33, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:34, 1.93MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:58, 1.66MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:47, 1.30MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:04, 1.61MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:15, 2.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:45, 1.77MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:37, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:55, 1.66MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:08, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:42, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:33, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:52, 1.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:05, 2.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:39, 1.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:22, 1.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:44, 1.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:59, 2.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:39, 1.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<03:21, 1.39MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:39, 1.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:57, 2.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:33, 1.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:17, 1.40MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:35, 1.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:55, 2.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:25, 3.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:19<07:57, 569kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<06:57, 651kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<05:08, 879kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<03:40, 1.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:38, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<11:23, 392kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<09:14, 482kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<06:44, 660kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<04:46, 924kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<04:36, 951kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<04:28, 981kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<03:26, 1.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<02:28, 1.75MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:12, 1.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:24, 1.27MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:40, 1.61MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<01:56, 2.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:45, 1.54MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:07, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:29, 1.71MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:48, 2.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:35, 1.62MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:59, 1.40MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:20, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:42, 2.43MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:30, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:51, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:15, 1.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:39, 2.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:32, 1.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:10, 1.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:32, 1.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:50, 2.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:23, 1.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:57, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:18, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:41, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:19, 1.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:29, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:55, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:23, 2.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:05, 1.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:05, 1.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:21, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<01:41, 2.25MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:41, 1.40MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:50, 1.32MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:13, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:35, 2.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:45, 1.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:51, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:12, 1.67MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:34, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:35, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:39, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:04, 1.74MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:29, 2.39MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:44, 1.30MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:33, 1.39MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:56, 1.83MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<01:23, 2.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<19:11, 182kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<14:01, 248kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<09:54, 350kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<06:53, 497kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<07:09, 478kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<06:11, 553kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<04:34, 747kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<03:15, 1.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:58, 1.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:51, 1.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:09, 1.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<01:31, 2.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<03:06, 1.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:51, 1.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:08, 1.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<01:31, 2.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:51, 1.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:25, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:49, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:49, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:32, 1.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:02, 1.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:30, 2.07MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:44, 1.76MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:50, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:25, 2.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:28, 2.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:14, 1.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:49, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:20, 2.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:35, 1.85MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:42, 1.72MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:18, 2.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<00:56, 3.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:36, 1.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:12, 1.30MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:37, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:39, 1.68MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<02:07, 1.31MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:44, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:15, 2.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:33, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:34, 1.73MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:13, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<01:17, 2.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:22, 1.93MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:04, 2.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:10, 2.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:47, 1.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:26, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:04, 2.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:21, 1.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:25, 1.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:06, 2.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:10, 2.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:16, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:58, 2.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:41, 3.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<07:55, 301kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<05:58, 398kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<04:16, 555kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<03:18, 699kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<03:10, 728kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:26, 948kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:43, 1.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:45, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:37, 1.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:14, 1.81MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:12, 1.80MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:39, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:21, 1.60MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:59, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:11, 1.76MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:14, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:56, 2.23MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:39, 3.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<15:35, 131kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<11:40, 175kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<08:18, 245kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<05:46, 347kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<04:28, 441kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<03:28, 566kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<02:29, 784kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:44, 1.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:26, 781kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:02, 927kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:29, 1.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:19, 1.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:15, 1.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:56, 1.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:39, 2.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:00, 880kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:03, 857kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:35, 1.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:08, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:12, 1.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:08, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:51, 1.95MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<00:36, 2.70MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:05, 779kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<02:04, 787kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:33, 1.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<01:06, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:08, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:05, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:49, 1.87MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:48, 1.84MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:50, 1.77MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:38, 2.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:40, 2.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:00, 1.42MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:49, 1.70MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:36, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:44, 1.81MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:45, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:35, 2.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:36, 2.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:54, 1.42MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:44, 1.71MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:32, 2.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:40, 1.81MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:41, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:31, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:32, 2.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:48, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:39, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:28, 2.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:35, 1.83MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:36, 1.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:27, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:28, 2.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:40, 1.47MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:34, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:24, 2.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:30, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:31, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:23, 2.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:24, 2.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:35, 1.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:29, 1.76MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:21, 2.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:25, 1.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:26, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:20, 2.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:20, 2.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:30, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:24, 1.74MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:17, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:21, 1.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:22, 1.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<00:16, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:16, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:25, 1.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:20, 1.71MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:14, 2.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:17, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:17, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:13, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:13, 2.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:14, 1.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:10, 2.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:10, 2.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:11, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:08, 2.54MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:09, 2.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:06, 2.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:04, 3.58MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:21, 698kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:20, 728kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:15, 962kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:09, 1.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:08, 1.29MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 2.51MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:06, 990kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:05, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.49MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.57MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.60MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.10MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<80:16:00,  1.38it/s]  0%|          | 982/400000 [00:00<56:03:08,  1.98it/s]  0%|          | 1945/400000 [00:00<39:08:43,  2.82it/s]  1%|          | 2798/400000 [00:01<27:20:49,  4.03it/s]  1%|          | 3682/400000 [00:01<19:06:14,  5.76it/s]  1%|          | 4659/400000 [00:01<13:20:35,  8.23it/s]  1%|▏         | 5617/400000 [00:01<9:19:15, 11.75it/s]   2%|▏         | 6467/400000 [00:01<6:30:52, 16.78it/s]  2%|▏         | 7367/400000 [00:01<4:33:12, 23.95it/s]  2%|▏         | 8336/400000 [00:01<3:10:58, 34.18it/s]  2%|▏         | 9356/400000 [00:01<2:13:31, 48.76it/s]  3%|▎         | 10332/400000 [00:01<1:33:25, 69.51it/s]  3%|▎         | 11301/400000 [00:01<1:05:26, 98.99it/s]  3%|▎         | 12251/400000 [00:02<45:54, 140.75it/s]   3%|▎         | 13216/400000 [00:02<32:15, 199.82it/s]  4%|▎         | 14159/400000 [00:02<22:44, 282.83it/s]  4%|▍         | 15095/400000 [00:02<16:05, 398.71it/s]  4%|▍         | 16022/400000 [00:02<11:30, 556.39it/s]  4%|▍         | 16955/400000 [00:02<08:14, 775.02it/s]  4%|▍         | 17906/400000 [00:02<05:57, 1069.77it/s]  5%|▍         | 18881/400000 [00:02<04:21, 1459.56it/s]  5%|▍         | 19814/400000 [00:02<03:14, 1954.07it/s]  5%|▌         | 20754/400000 [00:02<02:27, 2563.13it/s]  5%|▌         | 21686/400000 [00:03<01:56, 3246.89it/s]  6%|▌         | 22662/400000 [00:03<01:32, 4058.98it/s]  6%|▌         | 23635/400000 [00:03<01:16, 4919.02it/s]  6%|▌         | 24578/400000 [00:03<01:06, 5627.78it/s]  6%|▋         | 25493/400000 [00:03<01:03, 5863.85it/s]  7%|▋         | 26429/400000 [00:03<00:56, 6602.58it/s]  7%|▋         | 27307/400000 [00:03<00:52, 7131.89it/s]  7%|▋         | 28244/400000 [00:03<00:48, 7682.17it/s]  7%|▋         | 29154/400000 [00:03<00:46, 8055.49it/s]  8%|▊         | 30049/400000 [00:04<00:44, 8291.70it/s]  8%|▊         | 30997/400000 [00:04<00:42, 8613.72it/s]  8%|▊         | 31917/400000 [00:04<00:41, 8780.56it/s]  8%|▊         | 32830/400000 [00:04<00:42, 8559.65it/s]  8%|▊         | 33712/400000 [00:04<00:43, 8335.20it/s]  9%|▊         | 34565/400000 [00:04<00:44, 8246.60it/s]  9%|▉         | 35473/400000 [00:04<00:43, 8476.96it/s]  9%|▉         | 36448/400000 [00:04<00:41, 8821.66it/s]  9%|▉         | 37431/400000 [00:04<00:39, 9099.85it/s] 10%|▉         | 38365/400000 [00:04<00:39, 9169.59it/s] 10%|▉         | 39296/400000 [00:05<00:39, 9209.54it/s] 10%|█         | 40223/400000 [00:05<00:39, 9184.60it/s] 10%|█         | 41146/400000 [00:05<00:39, 9169.21it/s] 11%|█         | 42066/400000 [00:05<00:39, 9099.91it/s] 11%|█         | 42984/400000 [00:05<00:39, 9121.42it/s] 11%|█         | 43969/400000 [00:05<00:38, 9327.96it/s] 11%|█         | 44904/400000 [00:05<00:38, 9253.82it/s] 11%|█▏        | 45831/400000 [00:05<00:38, 9174.20it/s] 12%|█▏        | 46750/400000 [00:05<00:39, 9008.80it/s] 12%|█▏        | 47684/400000 [00:05<00:38, 9103.04it/s] 12%|█▏        | 48605/400000 [00:06<00:38, 9134.04it/s] 12%|█▏        | 49577/400000 [00:06<00:37, 9301.49it/s] 13%|█▎        | 50582/400000 [00:06<00:36, 9513.56it/s] 13%|█▎        | 51536/400000 [00:06<00:37, 9368.71it/s] 13%|█▎        | 52500/400000 [00:06<00:36, 9447.66it/s] 13%|█▎        | 53447/400000 [00:06<00:37, 9339.64it/s] 14%|█▎        | 54404/400000 [00:06<00:36, 9406.38it/s] 14%|█▍        | 55346/400000 [00:06<00:36, 9357.58it/s] 14%|█▍        | 56286/400000 [00:06<00:36, 9367.92it/s] 14%|█▍        | 57224/400000 [00:06<00:36, 9334.29it/s] 15%|█▍        | 58158/400000 [00:07<00:36, 9304.47it/s] 15%|█▍        | 59089/400000 [00:07<00:37, 9202.04it/s] 15%|█▌        | 60010/400000 [00:07<00:36, 9201.59it/s] 15%|█▌        | 60931/400000 [00:07<00:36, 9201.01it/s] 15%|█▌        | 61852/400000 [00:07<00:36, 9142.49it/s] 16%|█▌        | 62767/400000 [00:07<00:37, 9023.82it/s] 16%|█▌        | 63728/400000 [00:07<00:36, 9191.18it/s] 16%|█▌        | 64674/400000 [00:07<00:36, 9268.95it/s] 16%|█▋        | 65632/400000 [00:07<00:35, 9358.49it/s] 17%|█▋        | 66571/400000 [00:08<00:35, 9366.57it/s] 17%|█▋        | 67531/400000 [00:08<00:35, 9433.14it/s] 17%|█▋        | 68475/400000 [00:08<00:35, 9426.24it/s] 17%|█▋        | 69419/400000 [00:08<00:35, 9425.75it/s] 18%|█▊        | 70362/400000 [00:08<00:35, 9378.89it/s] 18%|█▊        | 71340/400000 [00:08<00:34, 9494.87it/s] 18%|█▊        | 72291/400000 [00:08<00:34, 9496.59it/s] 18%|█▊        | 73242/400000 [00:08<00:34, 9474.04it/s] 19%|█▊        | 74190/400000 [00:08<00:34, 9449.60it/s] 19%|█▉        | 75136/400000 [00:08<00:34, 9381.15it/s] 19%|█▉        | 76075/400000 [00:09<00:35, 9247.85it/s] 19%|█▉        | 77001/400000 [00:09<00:35, 9089.31it/s] 19%|█▉        | 77911/400000 [00:09<00:35, 8976.34it/s] 20%|█▉        | 78810/400000 [00:09<00:35, 8970.16it/s] 20%|█▉        | 79710/400000 [00:09<00:35, 8975.86it/s] 20%|██        | 80609/400000 [00:09<00:35, 8893.02it/s] 20%|██        | 81499/400000 [00:09<00:36, 8785.90it/s] 21%|██        | 82379/400000 [00:09<00:38, 8317.85it/s] 21%|██        | 83277/400000 [00:09<00:37, 8505.45it/s] 21%|██        | 84262/400000 [00:09<00:35, 8868.28it/s] 21%|██▏       | 85220/400000 [00:10<00:34, 9069.23it/s] 22%|██▏       | 86157/400000 [00:10<00:34, 9156.66it/s] 22%|██▏       | 87133/400000 [00:10<00:33, 9328.18it/s] 22%|██▏       | 88084/400000 [00:10<00:33, 9380.49it/s] 22%|██▏       | 89055/400000 [00:10<00:32, 9476.89it/s] 23%|██▎       | 90005/400000 [00:10<00:32, 9445.71it/s] 23%|██▎       | 90952/400000 [00:10<00:33, 9099.75it/s] 23%|██▎       | 91892/400000 [00:10<00:33, 9187.34it/s] 23%|██▎       | 92842/400000 [00:10<00:33, 9278.90it/s] 23%|██▎       | 93773/400000 [00:10<00:33, 9271.19it/s] 24%|██▎       | 94702/400000 [00:11<00:33, 9137.99it/s] 24%|██▍       | 95618/400000 [00:11<00:33, 9076.03it/s] 24%|██▍       | 96527/400000 [00:11<00:33, 9056.93it/s] 24%|██▍       | 97434/400000 [00:11<00:34, 8849.44it/s] 25%|██▍       | 98321/400000 [00:11<00:34, 8751.78it/s] 25%|██▍       | 99222/400000 [00:11<00:34, 8825.96it/s] 25%|██▌       | 100154/400000 [00:11<00:33, 8966.25it/s] 25%|██▌       | 101052/400000 [00:11<00:33, 8901.50it/s] 25%|██▌       | 101968/400000 [00:11<00:33, 8977.44it/s] 26%|██▌       | 102878/400000 [00:11<00:32, 9011.90it/s] 26%|██▌       | 103835/400000 [00:12<00:32, 9170.06it/s] 26%|██▌       | 104754/400000 [00:12<00:32, 8993.41it/s] 26%|██▋       | 105655/400000 [00:12<00:32, 8954.38it/s] 27%|██▋       | 106620/400000 [00:12<00:32, 9149.99it/s] 27%|██▋       | 107609/400000 [00:12<00:31, 9359.21it/s] 27%|██▋       | 108548/400000 [00:12<00:31, 9351.28it/s] 27%|██▋       | 109485/400000 [00:12<00:31, 9295.55it/s] 28%|██▊       | 110440/400000 [00:12<00:30, 9370.43it/s] 28%|██▊       | 111379/400000 [00:12<00:31, 9298.99it/s] 28%|██▊       | 112310/400000 [00:13<00:31, 9277.24it/s] 28%|██▊       | 113239/400000 [00:13<00:31, 9215.54it/s] 29%|██▊       | 114172/400000 [00:13<00:30, 9248.87it/s] 29%|██▉       | 115132/400000 [00:13<00:30, 9349.13it/s] 29%|██▉       | 116068/400000 [00:13<00:30, 9333.52it/s] 29%|██▉       | 117002/400000 [00:13<00:30, 9301.97it/s] 29%|██▉       | 117977/400000 [00:13<00:29, 9430.40it/s] 30%|██▉       | 118921/400000 [00:13<00:30, 9217.75it/s] 30%|██▉       | 119854/400000 [00:13<00:30, 9248.97it/s] 30%|███       | 120780/400000 [00:13<00:30, 9180.44it/s] 30%|███       | 121699/400000 [00:14<00:30, 9110.47it/s] 31%|███       | 122670/400000 [00:14<00:29, 9281.87it/s] 31%|███       | 123604/400000 [00:14<00:29, 9295.09it/s] 31%|███       | 124552/400000 [00:14<00:29, 9348.78it/s] 31%|███▏      | 125488/400000 [00:14<00:29, 9345.00it/s] 32%|███▏      | 126446/400000 [00:14<00:29, 9413.68it/s] 32%|███▏      | 127399/400000 [00:14<00:28, 9446.57it/s] 32%|███▏      | 128345/400000 [00:14<00:28, 9405.18it/s] 32%|███▏      | 129286/400000 [00:14<00:29, 9130.46it/s] 33%|███▎      | 130220/400000 [00:14<00:29, 9191.32it/s] 33%|███▎      | 131141/400000 [00:15<00:29, 9071.29it/s] 33%|███▎      | 132050/400000 [00:15<00:29, 9055.14it/s] 33%|███▎      | 132957/400000 [00:15<00:29, 8905.34it/s] 33%|███▎      | 133849/400000 [00:15<00:30, 8822.51it/s] 34%|███▎      | 134799/400000 [00:15<00:29, 9014.09it/s] 34%|███▍      | 135744/400000 [00:15<00:28, 9140.42it/s] 34%|███▍      | 136660/400000 [00:15<00:28, 9129.52it/s] 34%|███▍      | 137580/400000 [00:15<00:28, 9150.48it/s] 35%|███▍      | 138496/400000 [00:15<00:28, 9128.99it/s] 35%|███▍      | 139410/400000 [00:15<00:28, 9039.29it/s] 35%|███▌      | 140317/400000 [00:16<00:28, 9047.53it/s] 35%|███▌      | 141225/400000 [00:16<00:28, 9056.13it/s] 36%|███▌      | 142131/400000 [00:16<00:28, 8906.04it/s] 36%|███▌      | 143097/400000 [00:16<00:28, 9118.07it/s] 36%|███▌      | 144012/400000 [00:16<00:28, 9126.33it/s] 36%|███▌      | 144971/400000 [00:16<00:27, 9258.60it/s] 36%|███▋      | 145899/400000 [00:16<00:27, 9129.91it/s] 37%|███▋      | 146814/400000 [00:16<00:29, 8703.87it/s] 37%|███▋      | 147690/400000 [00:16<00:29, 8617.09it/s] 37%|███▋      | 148556/400000 [00:16<00:29, 8568.03it/s] 37%|███▋      | 149423/400000 [00:17<00:29, 8597.90it/s] 38%|███▊      | 150304/400000 [00:17<00:28, 8659.09it/s] 38%|███▊      | 151172/400000 [00:17<00:29, 8523.63it/s] 38%|███▊      | 152105/400000 [00:17<00:28, 8749.25it/s] 38%|███▊      | 153095/400000 [00:17<00:27, 9064.27it/s] 39%|███▊      | 154101/400000 [00:17<00:26, 9339.93it/s] 39%|███▉      | 155071/400000 [00:17<00:25, 9444.38it/s] 39%|███▉      | 156041/400000 [00:17<00:25, 9518.30it/s] 39%|███▉      | 156996/400000 [00:17<00:25, 9395.11it/s] 39%|███▉      | 157959/400000 [00:17<00:25, 9463.34it/s] 40%|███▉      | 158915/400000 [00:18<00:25, 9491.05it/s] 40%|███▉      | 159866/400000 [00:18<00:25, 9369.55it/s] 40%|████      | 160811/400000 [00:18<00:25, 9341.09it/s] 40%|████      | 161794/400000 [00:18<00:25, 9481.14it/s] 41%|████      | 162759/400000 [00:18<00:24, 9530.86it/s] 41%|████      | 163713/400000 [00:18<00:24, 9486.97it/s] 41%|████      | 164695/400000 [00:18<00:24, 9582.82it/s] 41%|████▏     | 165654/400000 [00:18<00:24, 9583.29it/s] 42%|████▏     | 166613/400000 [00:18<00:24, 9584.68it/s] 42%|████▏     | 167582/400000 [00:19<00:24, 9614.70it/s] 42%|████▏     | 168544/400000 [00:19<00:24, 9539.32it/s] 42%|████▏     | 169528/400000 [00:19<00:23, 9625.65it/s] 43%|████▎     | 170491/400000 [00:19<00:23, 9618.99it/s] 43%|████▎     | 171454/400000 [00:19<00:23, 9608.46it/s] 43%|████▎     | 172416/400000 [00:19<00:23, 9554.44it/s] 43%|████▎     | 173391/400000 [00:19<00:23, 9611.83it/s] 44%|████▎     | 174373/400000 [00:19<00:23, 9670.81it/s] 44%|████▍     | 175341/400000 [00:19<00:23, 9520.39it/s] 44%|████▍     | 176311/400000 [00:19<00:23, 9572.44it/s] 44%|████▍     | 177269/400000 [00:20<00:23, 9517.19it/s] 45%|████▍     | 178238/400000 [00:20<00:23, 9568.23it/s] 45%|████▍     | 179221/400000 [00:20<00:22, 9644.39it/s] 45%|████▌     | 180218/400000 [00:20<00:22, 9739.79it/s] 45%|████▌     | 181193/400000 [00:20<00:23, 9284.66it/s] 46%|████▌     | 182127/400000 [00:20<00:23, 9223.06it/s] 46%|████▌     | 183053/400000 [00:20<00:24, 9036.96it/s] 46%|████▌     | 184003/400000 [00:20<00:23, 9170.30it/s] 46%|████▌     | 184925/400000 [00:20<00:23, 9180.62it/s] 46%|████▋     | 185866/400000 [00:20<00:23, 9247.06it/s] 47%|████▋     | 186826/400000 [00:21<00:22, 9348.00it/s] 47%|████▋     | 187772/400000 [00:21<00:22, 9380.42it/s] 47%|████▋     | 188711/400000 [00:21<00:22, 9380.27it/s] 47%|████▋     | 189650/400000 [00:21<00:23, 9131.69it/s] 48%|████▊     | 190603/400000 [00:21<00:22, 9245.94it/s] 48%|████▊     | 191588/400000 [00:21<00:22, 9418.19it/s] 48%|████▊     | 192558/400000 [00:21<00:21, 9499.29it/s] 48%|████▊     | 193538/400000 [00:21<00:21, 9586.94it/s] 49%|████▊     | 194498/400000 [00:21<00:21, 9486.08it/s] 49%|████▉     | 195448/400000 [00:21<00:22, 9194.93it/s] 49%|████▉     | 196371/400000 [00:22<00:23, 8835.48it/s] 49%|████▉     | 197286/400000 [00:22<00:22, 8926.58it/s] 50%|████▉     | 198243/400000 [00:22<00:22, 9109.89it/s] 50%|████▉     | 199179/400000 [00:22<00:21, 9182.87it/s] 50%|█████     | 200150/400000 [00:22<00:21, 9334.80it/s] 50%|█████     | 201118/400000 [00:22<00:21, 9434.87it/s] 51%|█████     | 202077/400000 [00:22<00:20, 9479.57it/s] 51%|█████     | 203027/400000 [00:22<00:20, 9453.02it/s] 51%|█████     | 203974/400000 [00:22<00:21, 9286.16it/s] 51%|█████     | 204916/400000 [00:22<00:20, 9324.35it/s] 51%|█████▏    | 205852/400000 [00:23<00:20, 9334.04it/s] 52%|█████▏    | 206792/400000 [00:23<00:20, 9350.34it/s] 52%|█████▏    | 207737/400000 [00:23<00:20, 9379.00it/s] 52%|█████▏    | 208676/400000 [00:23<00:20, 9281.14it/s] 52%|█████▏    | 209645/400000 [00:23<00:20, 9399.83it/s] 53%|█████▎    | 210594/400000 [00:23<00:20, 9425.50it/s] 53%|█████▎    | 211548/400000 [00:23<00:19, 9457.01it/s] 53%|█████▎    | 212525/400000 [00:23<00:19, 9547.03it/s] 53%|█████▎    | 213481/400000 [00:23<00:19, 9468.63it/s] 54%|█████▎    | 214443/400000 [00:23<00:19, 9512.43it/s] 54%|█████▍    | 215395/400000 [00:24<00:19, 9497.49it/s] 54%|█████▍    | 216360/400000 [00:24<00:19, 9540.28it/s] 54%|█████▍    | 217315/400000 [00:24<00:19, 9453.74it/s] 55%|█████▍    | 218261/400000 [00:24<00:19, 9137.44it/s] 55%|█████▍    | 219222/400000 [00:24<00:19, 9273.49it/s] 55%|█████▌    | 220176/400000 [00:24<00:19, 9350.85it/s] 55%|█████▌    | 221161/400000 [00:24<00:18, 9492.75it/s] 56%|█████▌    | 222148/400000 [00:24<00:18, 9602.74it/s] 56%|█████▌    | 223110/400000 [00:24<00:18, 9498.36it/s] 56%|█████▌    | 224062/400000 [00:25<00:18, 9413.20it/s] 56%|█████▋    | 225011/400000 [00:25<00:18, 9435.39it/s] 56%|█████▋    | 225993/400000 [00:25<00:18, 9546.84it/s] 57%|█████▋    | 226951/400000 [00:25<00:18, 9555.03it/s] 57%|█████▋    | 227908/400000 [00:25<00:18, 9238.64it/s] 57%|█████▋    | 228908/400000 [00:25<00:18, 9451.99it/s] 57%|█████▋    | 229874/400000 [00:25<00:17, 9513.16it/s] 58%|█████▊    | 230843/400000 [00:25<00:17, 9565.36it/s] 58%|█████▊    | 231802/400000 [00:25<00:17, 9530.95it/s] 58%|█████▊    | 232757/400000 [00:25<00:17, 9320.48it/s] 58%|█████▊    | 233691/400000 [00:26<00:18, 9198.80it/s] 59%|█████▊    | 234646/400000 [00:26<00:17, 9300.73it/s] 59%|█████▉    | 235616/400000 [00:26<00:17, 9415.09it/s] 59%|█████▉    | 236559/400000 [00:26<00:17, 9413.40it/s] 59%|█████▉    | 237502/400000 [00:26<00:18, 9009.53it/s] 60%|█████▉    | 238408/400000 [00:26<00:18, 8941.34it/s] 60%|█████▉    | 239339/400000 [00:26<00:17, 9048.80it/s] 60%|██████    | 240286/400000 [00:26<00:17, 9169.89it/s] 60%|██████    | 241273/400000 [00:26<00:16, 9367.27it/s] 61%|██████    | 242213/400000 [00:26<00:16, 9358.00it/s] 61%|██████    | 243151/400000 [00:27<00:16, 9258.05it/s] 61%|██████    | 244079/400000 [00:27<00:16, 9251.58it/s] 61%|██████▏   | 245006/400000 [00:27<00:17, 9095.53it/s] 61%|██████▏   | 245917/400000 [00:27<00:17, 8780.33it/s] 62%|██████▏   | 246799/400000 [00:27<00:17, 8639.02it/s] 62%|██████▏   | 247688/400000 [00:27<00:17, 8709.22it/s] 62%|██████▏   | 248562/400000 [00:27<00:17, 8651.10it/s] 62%|██████▏   | 249436/400000 [00:27<00:17, 8676.61it/s] 63%|██████▎   | 250326/400000 [00:27<00:17, 8740.76it/s] 63%|██████▎   | 251249/400000 [00:27<00:16, 8879.99it/s] 63%|██████▎   | 252139/400000 [00:28<00:16, 8850.12it/s] 63%|██████▎   | 253113/400000 [00:28<00:16, 9097.75it/s] 64%|██████▎   | 254060/400000 [00:28<00:15, 9204.86it/s] 64%|██████▍   | 255014/400000 [00:28<00:15, 9302.16it/s] 64%|██████▍   | 255946/400000 [00:28<00:15, 9275.79it/s] 64%|██████▍   | 256926/400000 [00:28<00:15, 9426.73it/s] 64%|██████▍   | 257887/400000 [00:28<00:14, 9479.25it/s] 65%|██████▍   | 258870/400000 [00:28<00:14, 9579.26it/s] 65%|██████▍   | 259831/400000 [00:28<00:14, 9585.59it/s] 65%|██████▌   | 260791/400000 [00:29<00:15, 9138.96it/s] 65%|██████▌   | 261710/400000 [00:29<00:15, 9094.23it/s] 66%|██████▌   | 262623/400000 [00:29<00:15, 9038.66it/s] 66%|██████▌   | 263568/400000 [00:29<00:14, 9155.70it/s] 66%|██████▌   | 264486/400000 [00:29<00:14, 9039.48it/s] 66%|██████▋   | 265446/400000 [00:29<00:14, 9199.40it/s] 67%|██████▋   | 266368/400000 [00:29<00:14, 8957.14it/s] 67%|██████▋   | 267340/400000 [00:29<00:14, 9171.35it/s] 67%|██████▋   | 268276/400000 [00:29<00:14, 9226.23it/s] 67%|██████▋   | 269209/400000 [00:29<00:14, 9256.95it/s] 68%|██████▊   | 270137/400000 [00:30<00:14, 9123.69it/s] 68%|██████▊   | 271051/400000 [00:30<00:14, 9126.71it/s] 68%|██████▊   | 272010/400000 [00:30<00:13, 9257.83it/s] 68%|██████▊   | 272938/400000 [00:30<00:13, 9126.83it/s] 68%|██████▊   | 273852/400000 [00:30<00:13, 9052.78it/s] 69%|██████▊   | 274759/400000 [00:30<00:13, 9011.23it/s] 69%|██████▉   | 275663/400000 [00:30<00:13, 9019.58it/s] 69%|██████▉   | 276634/400000 [00:30<00:13, 9216.16it/s] 69%|██████▉   | 277617/400000 [00:30<00:13, 9389.82it/s] 70%|██████▉   | 278577/400000 [00:30<00:12, 9450.93it/s] 70%|██████▉   | 279524/400000 [00:31<00:13, 9235.33it/s] 70%|███████   | 280450/400000 [00:31<00:13, 9107.72it/s] 70%|███████   | 281381/400000 [00:31<00:12, 9166.45it/s] 71%|███████   | 282367/400000 [00:31<00:12, 9363.74it/s] 71%|███████   | 283306/400000 [00:31<00:12, 9057.13it/s] 71%|███████   | 284216/400000 [00:31<00:12, 9052.56it/s] 71%|███████▏  | 285124/400000 [00:31<00:12, 9029.24it/s] 72%|███████▏  | 286072/400000 [00:31<00:12, 9157.57it/s] 72%|███████▏  | 287067/400000 [00:31<00:12, 9379.05it/s] 72%|███████▏  | 288021/400000 [00:31<00:11, 9424.36it/s] 72%|███████▏  | 288966/400000 [00:32<00:12, 9128.46it/s] 72%|███████▏  | 289925/400000 [00:32<00:11, 9260.98it/s] 73%|███████▎  | 290854/400000 [00:32<00:11, 9156.76it/s] 73%|███████▎  | 291773/400000 [00:32<00:11, 9163.75it/s] 73%|███████▎  | 292717/400000 [00:32<00:11, 9242.41it/s] 73%|███████▎  | 293656/400000 [00:32<00:11, 9285.55it/s] 74%|███████▎  | 294644/400000 [00:32<00:11, 9453.76it/s] 74%|███████▍  | 295615/400000 [00:32<00:10, 9526.99it/s] 74%|███████▍  | 296569/400000 [00:32<00:11, 9354.77it/s] 74%|███████▍  | 297525/400000 [00:32<00:10, 9414.29it/s] 75%|███████▍  | 298474/400000 [00:33<00:10, 9434.27it/s] 75%|███████▍  | 299472/400000 [00:33<00:10, 9589.56it/s] 75%|███████▌  | 300433/400000 [00:33<00:10, 9577.68it/s] 75%|███████▌  | 301392/400000 [00:33<00:10, 9364.36it/s] 76%|███████▌  | 302331/400000 [00:33<00:10, 9238.10it/s] 76%|███████▌  | 303262/400000 [00:33<00:10, 9258.37it/s] 76%|███████▌  | 304260/400000 [00:33<00:10, 9463.00it/s] 76%|███████▋  | 305264/400000 [00:33<00:09, 9627.68it/s] 77%|███████▋  | 306258/400000 [00:33<00:09, 9718.93it/s] 77%|███████▋  | 307235/400000 [00:34<00:09, 9731.39it/s] 77%|███████▋  | 308210/400000 [00:34<00:09, 9472.14it/s] 77%|███████▋  | 309160/400000 [00:34<00:09, 9295.82it/s] 78%|███████▊  | 310092/400000 [00:34<00:09, 9185.06it/s] 78%|███████▊  | 311013/400000 [00:34<00:09, 9191.71it/s] 78%|███████▊  | 311934/400000 [00:34<00:09, 9164.02it/s] 78%|███████▊  | 312881/400000 [00:34<00:09, 9253.35it/s] 78%|███████▊  | 313808/400000 [00:34<00:09, 9015.55it/s] 79%|███████▊  | 314791/400000 [00:34<00:09, 9245.02it/s] 79%|███████▉  | 315722/400000 [00:34<00:09, 9263.85it/s] 79%|███████▉  | 316651/400000 [00:35<00:09, 9242.58it/s] 79%|███████▉  | 317577/400000 [00:35<00:09, 9039.75it/s] 80%|███████▉  | 318500/400000 [00:35<00:08, 9092.73it/s] 80%|███████▉  | 319462/400000 [00:35<00:08, 9243.75it/s] 80%|████████  | 320416/400000 [00:35<00:08, 9328.89it/s] 80%|████████  | 321409/400000 [00:35<00:08, 9499.92it/s] 81%|████████  | 322361/400000 [00:35<00:08, 9242.93it/s] 81%|████████  | 323326/400000 [00:35<00:08, 9360.06it/s] 81%|████████  | 324293/400000 [00:35<00:08, 9449.26it/s] 81%|████████▏ | 325240/400000 [00:35<00:07, 9352.79it/s] 82%|████████▏ | 326177/400000 [00:36<00:08, 9080.32it/s] 82%|████████▏ | 327149/400000 [00:36<00:07, 9261.36it/s] 82%|████████▏ | 328159/400000 [00:36<00:07, 9496.11it/s] 82%|████████▏ | 329113/400000 [00:36<00:07, 9279.91it/s] 83%|████████▎ | 330045/400000 [00:36<00:07, 8986.74it/s] 83%|████████▎ | 330949/400000 [00:36<00:07, 8787.16it/s] 83%|████████▎ | 331849/400000 [00:36<00:07, 8849.05it/s] 83%|████████▎ | 332737/400000 [00:36<00:07, 8825.10it/s] 83%|████████▎ | 333622/400000 [00:36<00:07, 8812.46it/s] 84%|████████▎ | 334505/400000 [00:37<00:07, 8681.92it/s] 84%|████████▍ | 335391/400000 [00:37<00:07, 8732.44it/s] 84%|████████▍ | 336313/400000 [00:37<00:07, 8871.15it/s] 84%|████████▍ | 337250/400000 [00:37<00:06, 9014.26it/s] 85%|████████▍ | 338153/400000 [00:37<00:06, 8854.49it/s] 85%|████████▍ | 339041/400000 [00:37<00:06, 8819.21it/s] 85%|████████▍ | 339925/400000 [00:37<00:06, 8810.67it/s] 85%|████████▌ | 340909/400000 [00:37<00:06, 9093.89it/s] 85%|████████▌ | 341911/400000 [00:37<00:06, 9351.84it/s] 86%|████████▌ | 342917/400000 [00:37<00:05, 9552.65it/s] 86%|████████▌ | 343886/400000 [00:38<00:05, 9592.48it/s] 86%|████████▌ | 344848/400000 [00:38<00:05, 9456.56it/s] 86%|████████▋ | 345796/400000 [00:38<00:05, 9360.78it/s] 87%|████████▋ | 346734/400000 [00:38<00:05, 9149.24it/s] 87%|████████▋ | 347652/400000 [00:38<00:05, 8986.66it/s] 87%|████████▋ | 348629/400000 [00:38<00:05, 9206.12it/s] 87%|████████▋ | 349574/400000 [00:38<00:05, 9275.28it/s] 88%|████████▊ | 350516/400000 [00:38<00:05, 9315.66it/s] 88%|████████▊ | 351493/400000 [00:38<00:05, 9446.36it/s] 88%|████████▊ | 352440/400000 [00:38<00:05, 9442.76it/s] 88%|████████▊ | 353386/400000 [00:39<00:04, 9432.40it/s] 89%|████████▊ | 354330/400000 [00:39<00:04, 9408.79it/s] 89%|████████▉ | 355297/400000 [00:39<00:04, 9485.63it/s] 89%|████████▉ | 356247/400000 [00:39<00:04, 9405.10it/s] 89%|████████▉ | 357189/400000 [00:39<00:04, 9391.10it/s] 90%|████████▉ | 358129/400000 [00:39<00:04, 9372.90it/s] 90%|████████▉ | 359067/400000 [00:39<00:04, 9244.69it/s] 90%|█████████ | 360040/400000 [00:39<00:04, 9382.85it/s] 90%|█████████ | 361016/400000 [00:39<00:04, 9492.78it/s] 90%|█████████ | 362000/400000 [00:39<00:03, 9593.71it/s] 91%|█████████ | 362961/400000 [00:40<00:03, 9581.29it/s] 91%|█████████ | 363920/400000 [00:40<00:03, 9532.95it/s] 91%|█████████ | 364874/400000 [00:40<00:03, 9480.44it/s] 91%|█████████▏| 365826/400000 [00:40<00:03, 9483.91it/s] 92%|█████████▏| 366775/400000 [00:40<00:03, 9377.00it/s] 92%|█████████▏| 367747/400000 [00:40<00:03, 9475.98it/s] 92%|█████████▏| 368696/400000 [00:40<00:03, 9192.53it/s] 92%|█████████▏| 369655/400000 [00:40<00:03, 9307.25it/s] 93%|█████████▎| 370659/400000 [00:40<00:03, 9513.43it/s] 93%|█████████▎| 371613/400000 [00:40<00:02, 9484.49it/s] 93%|█████████▎| 372570/400000 [00:41<00:02, 9509.88it/s] 93%|█████████▎| 373523/400000 [00:41<00:02, 9251.80it/s] 94%|█████████▎| 374451/400000 [00:41<00:02, 9177.84it/s] 94%|█████████▍| 375371/400000 [00:41<00:02, 9133.62it/s] 94%|█████████▍| 376324/400000 [00:41<00:02, 9248.17it/s] 94%|█████████▍| 377251/400000 [00:41<00:02, 8862.92it/s] 95%|█████████▍| 378142/400000 [00:41<00:02, 8594.73it/s] 95%|█████████▍| 379108/400000 [00:41<00:02, 8886.52it/s] 95%|█████████▌| 380051/400000 [00:41<00:02, 9041.52it/s] 95%|█████████▌| 380971/400000 [00:42<00:02, 9086.26it/s] 95%|█████████▌| 381884/400000 [00:42<00:02, 8868.67it/s] 96%|█████████▌| 382775/400000 [00:42<00:01, 8650.87it/s] 96%|█████████▌| 383644/400000 [00:42<00:01, 8472.99it/s] 96%|█████████▌| 384562/400000 [00:42<00:01, 8671.24it/s] 96%|█████████▋| 385501/400000 [00:42<00:01, 8873.41it/s] 97%|█████████▋| 386467/400000 [00:42<00:01, 9095.43it/s] 97%|█████████▋| 387381/400000 [00:42<00:01, 9006.99it/s] 97%|█████████▋| 388343/400000 [00:42<00:01, 9182.51it/s] 97%|█████████▋| 389288/400000 [00:42<00:01, 9259.83it/s] 98%|█████████▊| 390263/400000 [00:43<00:01, 9399.70it/s] 98%|█████████▊| 391272/400000 [00:43<00:00, 9596.20it/s] 98%|█████████▊| 392235/400000 [00:43<00:00, 9442.77it/s] 98%|█████████▊| 393210/400000 [00:43<00:00, 9530.37it/s] 99%|█████████▊| 394165/400000 [00:43<00:00, 9356.66it/s] 99%|█████████▉| 395149/400000 [00:43<00:00, 9496.37it/s] 99%|█████████▉| 396118/400000 [00:43<00:00, 9553.41it/s] 99%|█████████▉| 397075/400000 [00:43<00:00, 9112.37it/s]100%|█████████▉| 398062/400000 [00:43<00:00, 9326.24it/s]100%|█████████▉| 399000/400000 [00:43<00:00, 9340.22it/s]100%|█████████▉| 399968/400000 [00:44<00:00, 9439.05it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 9075.26it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f85ac1780f0>, <torchtext.data.dataset.TabularDataset object at 0x7f85ac178240>, <torchtext.vocab.Vocab object at 0x7f85ac178160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.36 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.36 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.36 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.96 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.96 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.12 file/s]2020-07-11 07:03:55.661566: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-11 07:03:55.665360: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-11 07:03:55.666199: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c01dd0aeb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-11 07:03:55.666216: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:10, 140242.69it/s] 80%|████████  | 7962624/9912422 [00:00<00:09, 200194.46it/s]9920512it [00:00, 42176067.75it/s]                           
0it [00:00, ?it/s]32768it [00:00, 580151.09it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 487395.22it/s]1654784it [00:00, 12238056.96it/s]                         
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 75288.72it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

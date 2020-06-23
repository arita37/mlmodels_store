
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff91877eea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff91877eea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ff983a3d1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ff983a3d1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff9a1d85e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff9a1d85e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff930d65488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff930d65488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff930d65488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:18, 126704.31it/s] 83%|████████▎ | 8232960/9912422 [00:00<00:09, 180885.43it/s]9920512it [00:00, 40980389.29it/s]                           
0it [00:00, ?it/s]32768it [00:00, 650672.52it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 476661.38it/s]1654784it [00:00, 12380209.18it/s]                         
0it [00:00, ?it/s]8192it [00:00, 265802.35it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff92e298f28>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff92e294668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ff930d650d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ff930d650d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ff930d650d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:15,  2.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:15,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:15,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:14,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:14,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:13,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:13,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:12,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  3.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:51,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:50,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:50,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:50,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:49,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:49,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:49,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:34,  4.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:34,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:33,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:33,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:33,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:33,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:32,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:15,  8.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:15,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:00<00:10, 11.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 14.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 19.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 25.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 25.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 25.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 25.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 25.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 25.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 25.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 25.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 31.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 38.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 45.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 57.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 62.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 62.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 62.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 62.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 62.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 62.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 62.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 62.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 62.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 65.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 65.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 65.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 65.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 65.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 65.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 65.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 65.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 68.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 71.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 71.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 71.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 71.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 72.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 72.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 72.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 72.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 73.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.59s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.04 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.60s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 74.04 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.60s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.60s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.23 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.60s/ url]
0 examples [00:00, ? examples/s]2020-06-23 00:08:41.004754: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-23 00:08:41.023787: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-23 00:08:41.024002: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5592f3d9bcb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-23 00:08:41.024037: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.16 examples/s]99 examples [00:00,  3.08 examples/s]207 examples [00:00,  4.39 examples/s]304 examples [00:00,  6.26 examples/s]405 examples [00:00,  8.92 examples/s]511 examples [00:00, 12.70 examples/s]614 examples [00:01, 18.04 examples/s]715 examples [00:01, 25.58 examples/s]824 examples [00:01, 36.17 examples/s]931 examples [00:01, 50.93 examples/s]1032 examples [00:01, 71.22 examples/s]1135 examples [00:01, 98.81 examples/s]1242 examples [00:01, 135.75 examples/s]1353 examples [00:01, 184.25 examples/s]1461 examples [00:01, 245.27 examples/s]1568 examples [00:01, 318.95 examples/s]1675 examples [00:02, 403.17 examples/s]1781 examples [00:02, 492.93 examples/s]1886 examples [00:02, 580.21 examples/s]1996 examples [00:02, 675.53 examples/s]2102 examples [00:02, 757.17 examples/s]2210 examples [00:02, 830.38 examples/s]2320 examples [00:02, 896.28 examples/s]2429 examples [00:02, 946.48 examples/s]2537 examples [00:02, 963.83 examples/s]2643 examples [00:02, 986.91 examples/s]2749 examples [00:03, 994.75 examples/s]2854 examples [00:03, 967.52 examples/s]2955 examples [00:03, 967.69 examples/s]3057 examples [00:03, 980.54 examples/s]3157 examples [00:03, 980.48 examples/s]3263 examples [00:03, 1003.01 examples/s]3365 examples [00:03, 1004.59 examples/s]3470 examples [00:03, 1017.24 examples/s]3573 examples [00:03, 1009.97 examples/s]3675 examples [00:04, 1008.20 examples/s]3778 examples [00:04, 1012.40 examples/s]3880 examples [00:04, 1011.09 examples/s]3985 examples [00:04, 1020.17 examples/s]4097 examples [00:04, 1048.03 examples/s]4203 examples [00:04, 1021.68 examples/s]4306 examples [00:04, 1020.47 examples/s]4409 examples [00:04, 1012.14 examples/s]4511 examples [00:04, 993.81 examples/s] 4611 examples [00:04, 976.31 examples/s]4711 examples [00:05, 981.74 examples/s]4817 examples [00:05, 1003.48 examples/s]4927 examples [00:05, 1029.69 examples/s]5031 examples [00:05, 1027.99 examples/s]5139 examples [00:05, 1041.19 examples/s]5244 examples [00:05, 1033.91 examples/s]5348 examples [00:05, 1034.13 examples/s]5458 examples [00:05, 1047.62 examples/s]5564 examples [00:05, 1049.81 examples/s]5670 examples [00:05, 1040.84 examples/s]5775 examples [00:06, 1037.26 examples/s]5879 examples [00:06, 1028.67 examples/s]5982 examples [00:06, 1007.05 examples/s]6088 examples [00:06, 1020.95 examples/s]6191 examples [00:06, 1009.11 examples/s]6293 examples [00:06, 947.70 examples/s] 6389 examples [00:06, 948.25 examples/s]6487 examples [00:06, 957.39 examples/s]6586 examples [00:06, 964.06 examples/s]6690 examples [00:07, 984.70 examples/s]6793 examples [00:07, 996.04 examples/s]6900 examples [00:07, 1013.45 examples/s]7002 examples [00:07, 997.08 examples/s] 7112 examples [00:07, 1023.62 examples/s]7220 examples [00:07, 1038.78 examples/s]7325 examples [00:07, 1026.87 examples/s]7431 examples [00:07, 1035.23 examples/s]7539 examples [00:07, 1046.25 examples/s]7645 examples [00:07, 1048.83 examples/s]7750 examples [00:08, 1047.59 examples/s]7855 examples [00:08, 1035.41 examples/s]7959 examples [00:08, 1024.50 examples/s]8062 examples [00:08, 958.64 examples/s] 8167 examples [00:08, 982.58 examples/s]8267 examples [00:08, 977.36 examples/s]8371 examples [00:08, 991.47 examples/s]8474 examples [00:08, 1001.04 examples/s]8575 examples [00:08, 1002.76 examples/s]8676 examples [00:08, 973.34 examples/s] 8774 examples [00:09, 965.89 examples/s]8871 examples [00:09, 956.18 examples/s]8971 examples [00:09, 968.27 examples/s]9070 examples [00:09, 973.49 examples/s]9171 examples [00:09, 983.57 examples/s]9275 examples [00:09, 998.42 examples/s]9384 examples [00:09, 1019.30 examples/s]9492 examples [00:09, 1034.30 examples/s]9596 examples [00:09, 989.74 examples/s] 9696 examples [00:10, 980.27 examples/s]9796 examples [00:10, 985.43 examples/s]9902 examples [00:10, 1005.28 examples/s]10003 examples [00:10, 940.85 examples/s]10111 examples [00:10, 977.65 examples/s]10218 examples [00:10, 1002.03 examples/s]10320 examples [00:10, 1003.79 examples/s]10428 examples [00:10, 1025.40 examples/s]10537 examples [00:10, 1043.92 examples/s]10642 examples [00:10, 1037.67 examples/s]10747 examples [00:11, 1030.45 examples/s]10851 examples [00:11, 1014.36 examples/s]10954 examples [00:11, 1016.88 examples/s]11056 examples [00:11, 998.98 examples/s] 11157 examples [00:11, 966.35 examples/s]11254 examples [00:11, 962.38 examples/s]11351 examples [00:11, 932.94 examples/s]11457 examples [00:11, 966.53 examples/s]11564 examples [00:11, 993.57 examples/s]11669 examples [00:11, 1008.00 examples/s]11775 examples [00:12, 1021.38 examples/s]11880 examples [00:12, 1028.15 examples/s]11984 examples [00:12, 1026.57 examples/s]12088 examples [00:12, 1028.96 examples/s]12192 examples [00:12, 1022.41 examples/s]12295 examples [00:12, 1005.93 examples/s]12405 examples [00:12, 1030.25 examples/s]12513 examples [00:12, 1042.47 examples/s]12622 examples [00:12, 1056.20 examples/s]12728 examples [00:13, 1039.21 examples/s]12833 examples [00:13, 1002.75 examples/s]12935 examples [00:13, 1005.83 examples/s]13037 examples [00:13, 1008.35 examples/s]13142 examples [00:13, 1020.02 examples/s]13246 examples [00:13, 1023.74 examples/s]13349 examples [00:13, 1009.44 examples/s]13457 examples [00:13, 1027.22 examples/s]13566 examples [00:13, 1043.70 examples/s]13673 examples [00:13, 1051.05 examples/s]13782 examples [00:14, 1060.50 examples/s]13889 examples [00:14, 1042.99 examples/s]13998 examples [00:14, 1053.96 examples/s]14107 examples [00:14, 1062.07 examples/s]14217 examples [00:14, 1070.33 examples/s]14325 examples [00:14, 1069.79 examples/s]14433 examples [00:14, 1022.59 examples/s]14539 examples [00:14, 1033.01 examples/s]14647 examples [00:14, 1043.54 examples/s]14752 examples [00:14, 1040.19 examples/s]14859 examples [00:15, 1047.37 examples/s]14965 examples [00:15, 1049.44 examples/s]15072 examples [00:15, 1053.43 examples/s]15182 examples [00:15, 1065.83 examples/s]15289 examples [00:15, 1057.92 examples/s]15398 examples [00:15, 1067.27 examples/s]15505 examples [00:15, 1044.76 examples/s]15611 examples [00:15, 1047.95 examples/s]15720 examples [00:15, 1058.86 examples/s]15832 examples [00:15, 1073.83 examples/s]15940 examples [00:16, 1032.75 examples/s]16044 examples [00:16, 1022.66 examples/s]16149 examples [00:16, 1030.46 examples/s]16253 examples [00:16, 1028.44 examples/s]16357 examples [00:16, 1021.49 examples/s]16460 examples [00:16, 996.83 examples/s] 16560 examples [00:16, 977.16 examples/s]16667 examples [00:16, 1003.27 examples/s]16778 examples [00:16, 1031.48 examples/s]16888 examples [00:17, 1048.92 examples/s]16997 examples [00:17, 1060.83 examples/s]17104 examples [00:17, 1045.75 examples/s]17209 examples [00:17, 1028.38 examples/s]17318 examples [00:17, 1044.29 examples/s]17426 examples [00:17, 1052.24 examples/s]17534 examples [00:17, 1059.27 examples/s]17641 examples [00:17, 1053.35 examples/s]17751 examples [00:17, 1064.01 examples/s]17858 examples [00:17, 1045.17 examples/s]17966 examples [00:18, 1052.48 examples/s]18076 examples [00:18, 1063.87 examples/s]18183 examples [00:18, 1046.20 examples/s]18291 examples [00:18, 1055.12 examples/s]18402 examples [00:18, 1068.42 examples/s]18509 examples [00:18, 1065.70 examples/s]18616 examples [00:18, 1049.20 examples/s]18722 examples [00:18, 1044.96 examples/s]18833 examples [00:18, 1062.40 examples/s]18940 examples [00:18, 1063.74 examples/s]19051 examples [00:19, 1075.09 examples/s]19159 examples [00:19, 1051.94 examples/s]19265 examples [00:19, 1048.85 examples/s]19372 examples [00:19, 1052.59 examples/s]19481 examples [00:19, 1060.70 examples/s]19588 examples [00:19, 1050.15 examples/s]19694 examples [00:19, 1045.42 examples/s]19799 examples [00:19, 1033.26 examples/s]19903 examples [00:19, 1014.20 examples/s]20005 examples [00:19, 968.61 examples/s] 20116 examples [00:20, 1005.43 examples/s]20223 examples [00:20, 1023.76 examples/s]20326 examples [00:20, 1011.22 examples/s]20437 examples [00:20, 1037.72 examples/s]20546 examples [00:20, 1050.76 examples/s]20655 examples [00:20, 1061.42 examples/s]20762 examples [00:20, 1056.11 examples/s]20868 examples [00:20, 1030.70 examples/s]20972 examples [00:20, 1023.81 examples/s]21078 examples [00:21, 1034.21 examples/s]21182 examples [00:21, 1033.73 examples/s]21287 examples [00:21, 1036.53 examples/s]21391 examples [00:21, 1026.00 examples/s]21496 examples [00:21, 1032.24 examples/s]21600 examples [00:21, 1028.65 examples/s]21703 examples [00:21, 1022.32 examples/s]21806 examples [00:21, 997.36 examples/s] 21906 examples [00:21, 971.69 examples/s]22006 examples [00:21, 978.46 examples/s]22112 examples [00:22, 999.31 examples/s]22213 examples [00:22, 985.82 examples/s]22316 examples [00:22, 997.66 examples/s]22418 examples [00:22, 1002.10 examples/s]22522 examples [00:22, 1011.78 examples/s]22626 examples [00:22, 1018.95 examples/s]22728 examples [00:22, 961.14 examples/s] 22826 examples [00:22, 964.89 examples/s]22928 examples [00:22, 978.23 examples/s]23033 examples [00:22, 996.84 examples/s]23134 examples [00:23, 993.78 examples/s]23234 examples [00:23, 930.34 examples/s]23331 examples [00:23, 940.75 examples/s]23432 examples [00:23, 959.85 examples/s]23538 examples [00:23, 986.07 examples/s]23644 examples [00:23, 1006.59 examples/s]23750 examples [00:23, 1020.53 examples/s]23853 examples [00:23, 995.78 examples/s] 23955 examples [00:23, 1002.13 examples/s]24060 examples [00:24, 1013.67 examples/s]24162 examples [00:24, 1015.19 examples/s]24264 examples [00:24, 1002.57 examples/s]24365 examples [00:24, 1003.84 examples/s]24466 examples [00:24, 994.23 examples/s] 24572 examples [00:24, 1012.63 examples/s]24675 examples [00:24, 1016.58 examples/s]24777 examples [00:24, 1010.00 examples/s]24879 examples [00:24, 981.60 examples/s] 24978 examples [00:24, 957.01 examples/s]25081 examples [00:25, 977.72 examples/s]25180 examples [00:25, 977.03 examples/s]25278 examples [00:25, 950.90 examples/s]25374 examples [00:25, 942.15 examples/s]25469 examples [00:25, 929.95 examples/s]25574 examples [00:25, 961.60 examples/s]25675 examples [00:25, 972.69 examples/s]25781 examples [00:25, 994.61 examples/s]25881 examples [00:25, 912.83 examples/s]25974 examples [00:26, 897.34 examples/s]26065 examples [00:26, 871.66 examples/s]26158 examples [00:26, 886.28 examples/s]26258 examples [00:26, 917.54 examples/s]26351 examples [00:26, 907.84 examples/s]26444 examples [00:26, 913.55 examples/s]26536 examples [00:26, 903.86 examples/s]26640 examples [00:26, 938.82 examples/s]26742 examples [00:26, 960.70 examples/s]26843 examples [00:26, 972.41 examples/s]26947 examples [00:27, 989.48 examples/s]27052 examples [00:27, 1005.27 examples/s]27157 examples [00:27, 1016.11 examples/s]27259 examples [00:27, 1009.83 examples/s]27361 examples [00:27, 961.70 examples/s] 27464 examples [00:27, 978.91 examples/s]27570 examples [00:27, 1001.44 examples/s]27679 examples [00:27, 1024.89 examples/s]27784 examples [00:27, 1029.31 examples/s]27891 examples [00:27, 1040.92 examples/s]27997 examples [00:28, 1044.84 examples/s]28102 examples [00:28, 1037.36 examples/s]28206 examples [00:28, 997.81 examples/s] 28309 examples [00:28, 1004.59 examples/s]28413 examples [00:28, 1013.71 examples/s]28515 examples [00:28, 978.47 examples/s] 28614 examples [00:28, 954.30 examples/s]28710 examples [00:28, 953.65 examples/s]28812 examples [00:28, 970.97 examples/s]28910 examples [00:29, 966.64 examples/s]29017 examples [00:29, 994.74 examples/s]29123 examples [00:29, 1012.80 examples/s]29229 examples [00:29, 1025.66 examples/s]29332 examples [00:29, 1018.24 examples/s]29439 examples [00:29, 1031.65 examples/s]29543 examples [00:29, 1020.88 examples/s]29648 examples [00:29, 1028.41 examples/s]29751 examples [00:29, 1008.68 examples/s]29853 examples [00:29, 974.50 examples/s] 29951 examples [00:30, 973.23 examples/s]30049 examples [00:30, 934.30 examples/s]30154 examples [00:30, 965.61 examples/s]30256 examples [00:30, 980.71 examples/s]30355 examples [00:30, 983.31 examples/s]30456 examples [00:30, 989.02 examples/s]30566 examples [00:30, 1018.32 examples/s]30671 examples [00:30, 1025.04 examples/s]30777 examples [00:30, 1032.54 examples/s]30881 examples [00:30, 1028.67 examples/s]30989 examples [00:31, 1042.45 examples/s]31095 examples [00:31, 1045.75 examples/s]31202 examples [00:31, 1052.77 examples/s]31308 examples [00:31, 1033.53 examples/s]31412 examples [00:31, 1023.94 examples/s]31518 examples [00:31, 1031.21 examples/s]31622 examples [00:31, 1028.73 examples/s]31725 examples [00:31, 1026.55 examples/s]31831 examples [00:31, 1035.92 examples/s]31935 examples [00:31, 1026.74 examples/s]32038 examples [00:32, 1016.66 examples/s]32141 examples [00:32, 1019.23 examples/s]32247 examples [00:32, 1030.57 examples/s]32353 examples [00:32, 1038.74 examples/s]32457 examples [00:32, 1024.42 examples/s]32560 examples [00:32, 1022.97 examples/s]32663 examples [00:32, 1022.93 examples/s]32771 examples [00:32, 1038.67 examples/s]32875 examples [00:32, 1032.17 examples/s]32980 examples [00:32, 1036.37 examples/s]33088 examples [00:33, 1046.58 examples/s]33193 examples [00:33, 1022.96 examples/s]33296 examples [00:33, 1012.34 examples/s]33400 examples [00:33, 1019.81 examples/s]33503 examples [00:33, 1021.86 examples/s]33612 examples [00:33, 1040.08 examples/s]33723 examples [00:33, 1057.89 examples/s]33833 examples [00:33, 1069.84 examples/s]33941 examples [00:33, 1021.66 examples/s]34044 examples [00:34, 1021.68 examples/s]34147 examples [00:34, 1011.33 examples/s]34249 examples [00:34, 977.80 examples/s] 34349 examples [00:34, 981.59 examples/s]34448 examples [00:34, 931.12 examples/s]34545 examples [00:34, 941.66 examples/s]34648 examples [00:34, 966.05 examples/s]34747 examples [00:34, 972.71 examples/s]34845 examples [00:34, 885.85 examples/s]34936 examples [00:34, 890.32 examples/s]35036 examples [00:35, 918.72 examples/s]35132 examples [00:35, 928.45 examples/s]35226 examples [00:35, 905.67 examples/s]35328 examples [00:35, 935.27 examples/s]35423 examples [00:35, 937.60 examples/s]35518 examples [00:35, 939.60 examples/s]35624 examples [00:35, 970.93 examples/s]35730 examples [00:35, 994.31 examples/s]35833 examples [00:35, 1003.62 examples/s]35939 examples [00:36, 1017.37 examples/s]36047 examples [00:36, 1033.66 examples/s]36152 examples [00:36, 1035.52 examples/s]36256 examples [00:36, 1034.34 examples/s]36363 examples [00:36, 1042.82 examples/s]36468 examples [00:36, 993.58 examples/s] 36568 examples [00:36, 981.29 examples/s]36667 examples [00:36, 929.41 examples/s]36770 examples [00:36, 957.19 examples/s]36875 examples [00:36, 982.38 examples/s]36980 examples [00:37, 1001.46 examples/s]37093 examples [00:37, 1036.69 examples/s]37198 examples [00:37, 1039.95 examples/s]37303 examples [00:37, 1031.33 examples/s]37407 examples [00:37, 971.94 examples/s] 37506 examples [00:37, 970.44 examples/s]37607 examples [00:37, 981.40 examples/s]37713 examples [00:37, 1001.88 examples/s]37814 examples [00:37, 988.97 examples/s] 37916 examples [00:37, 996.35 examples/s]38016 examples [00:38, 992.84 examples/s]38116 examples [00:38, 974.12 examples/s]38214 examples [00:38, 970.29 examples/s]38312 examples [00:38, 967.15 examples/s]38412 examples [00:38, 974.98 examples/s]38511 examples [00:38, 977.43 examples/s]38611 examples [00:38, 980.73 examples/s]38711 examples [00:38, 985.69 examples/s]38812 examples [00:38, 991.78 examples/s]38912 examples [00:39, 978.32 examples/s]39011 examples [00:39, 981.15 examples/s]39115 examples [00:39, 996.67 examples/s]39215 examples [00:39, 987.20 examples/s]39314 examples [00:39, 985.66 examples/s]39413 examples [00:39, 947.79 examples/s]39509 examples [00:39, 945.87 examples/s]39617 examples [00:39, 980.35 examples/s]39716 examples [00:39, 925.93 examples/s]39810 examples [00:39, 876.33 examples/s]39899 examples [00:40, 857.49 examples/s]39986 examples [00:40, 836.94 examples/s]40071 examples [00:40, 803.27 examples/s]40159 examples [00:40, 823.06 examples/s]40243 examples [00:40, 814.77 examples/s]40330 examples [00:40, 829.37 examples/s]40417 examples [00:40, 840.72 examples/s]40502 examples [00:40, 840.90 examples/s]40590 examples [00:40, 850.27 examples/s]40680 examples [00:41, 864.34 examples/s]40767 examples [00:41, 861.03 examples/s]40857 examples [00:41, 870.95 examples/s]40947 examples [00:41, 879.38 examples/s]41046 examples [00:41, 907.06 examples/s]41137 examples [00:41, 893.40 examples/s]41227 examples [00:41, 887.98 examples/s]41323 examples [00:41, 907.72 examples/s]41426 examples [00:41, 939.52 examples/s]41521 examples [00:41, 932.04 examples/s]41616 examples [00:42, 934.87 examples/s]41717 examples [00:42, 954.20 examples/s]41818 examples [00:42, 969.66 examples/s]41916 examples [00:42, 969.77 examples/s]42014 examples [00:42, 971.03 examples/s]42112 examples [00:42, 962.33 examples/s]42213 examples [00:42, 974.10 examples/s]42311 examples [00:42, 927.69 examples/s]42411 examples [00:42, 947.90 examples/s]42510 examples [00:42, 958.42 examples/s]42614 examples [00:43, 979.34 examples/s]42713 examples [00:43, 973.14 examples/s]42811 examples [00:43, 943.52 examples/s]42913 examples [00:43, 965.21 examples/s]43010 examples [00:43, 963.51 examples/s]43107 examples [00:43, 931.70 examples/s]43207 examples [00:43, 950.11 examples/s]43308 examples [00:43, 966.23 examples/s]43405 examples [00:43, 953.31 examples/s]43503 examples [00:43, 959.29 examples/s]43600 examples [00:44, 957.11 examples/s]43702 examples [00:44, 974.14 examples/s]43805 examples [00:44, 988.99 examples/s]43905 examples [00:44, 991.08 examples/s]44005 examples [00:44, 989.53 examples/s]44105 examples [00:44, 989.60 examples/s]44205 examples [00:44, 989.75 examples/s]44305 examples [00:44, 983.66 examples/s]44404 examples [00:44, 972.62 examples/s]44502 examples [00:45, 968.42 examples/s]44601 examples [00:45, 972.57 examples/s]44700 examples [00:45, 975.23 examples/s]44799 examples [00:45, 978.53 examples/s]44901 examples [00:45, 988.17 examples/s]45000 examples [00:45, 971.78 examples/s]45098 examples [00:45, 973.54 examples/s]45196 examples [00:45, 973.14 examples/s]45298 examples [00:45, 984.58 examples/s]45401 examples [00:45, 995.76 examples/s]45501 examples [00:46, 958.45 examples/s]45598 examples [00:46, 922.83 examples/s]45691 examples [00:46, 922.76 examples/s]45786 examples [00:46, 930.24 examples/s]45880 examples [00:46, 922.28 examples/s]45973 examples [00:46, 900.27 examples/s]46064 examples [00:46, 896.00 examples/s]46154 examples [00:46, 876.52 examples/s]46248 examples [00:46, 894.49 examples/s]46340 examples [00:46, 900.94 examples/s]46436 examples [00:47, 917.65 examples/s]46528 examples [00:47, 901.81 examples/s]46628 examples [00:47, 927.50 examples/s]46728 examples [00:47, 945.69 examples/s]46823 examples [00:47, 931.74 examples/s]46917 examples [00:47, 932.14 examples/s]47011 examples [00:47, 924.16 examples/s]47104 examples [00:47, 902.01 examples/s]47200 examples [00:47, 916.63 examples/s]47292 examples [00:48, 888.42 examples/s]47382 examples [00:48, 885.72 examples/s]47472 examples [00:48, 889.10 examples/s]47569 examples [00:48, 910.43 examples/s]47661 examples [00:48, 906.11 examples/s]47752 examples [00:48, 887.31 examples/s]47847 examples [00:48, 904.96 examples/s]47938 examples [00:48, 904.48 examples/s]48031 examples [00:48, 910.42 examples/s]48125 examples [00:48, 917.33 examples/s]48220 examples [00:49, 926.06 examples/s]48319 examples [00:49, 941.18 examples/s]48417 examples [00:49, 951.38 examples/s]48513 examples [00:49, 948.36 examples/s]48608 examples [00:49, 940.76 examples/s]48703 examples [00:49, 925.51 examples/s]48796 examples [00:49, 897.52 examples/s]48887 examples [00:49, 900.03 examples/s]48978 examples [00:49, 894.70 examples/s]49073 examples [00:49, 908.85 examples/s]49165 examples [00:50, 902.23 examples/s]49256 examples [00:50, 887.03 examples/s]49345 examples [00:50, 867.50 examples/s]49437 examples [00:50, 879.76 examples/s]49526 examples [00:50, 881.59 examples/s]49620 examples [00:50, 897.41 examples/s]49710 examples [00:50, 888.02 examples/s]49808 examples [00:50, 910.77 examples/s]49902 examples [00:50, 917.73 examples/s]49998 examples [00:50, 927.97 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7377/50000 [00:00<00:00, 73764.27 examples/s] 36%|███▌      | 17980/50000 [00:00<00:00, 81174.26 examples/s] 59%|█████▉    | 29493/50000 [00:00<00:00, 89052.88 examples/s] 82%|████████▏ | 40960/50000 [00:00<00:00, 95447.59 examples/s]                                                               0 examples [00:00, ? examples/s]66 examples [00:00, 653.12 examples/s]160 examples [00:00, 717.84 examples/s]263 examples [00:00, 788.49 examples/s]367 examples [00:00, 848.65 examples/s]469 examples [00:00, 893.67 examples/s]568 examples [00:00, 920.43 examples/s]664 examples [00:00, 930.61 examples/s]757 examples [00:00, 928.56 examples/s]853 examples [00:00, 936.61 examples/s]949 examples [00:01, 940.71 examples/s]1050 examples [00:01, 959.67 examples/s]1146 examples [00:01, 933.47 examples/s]1240 examples [00:01, 933.84 examples/s]1334 examples [00:01, 914.73 examples/s]1426 examples [00:01, 880.28 examples/s]1515 examples [00:01, 875.49 examples/s]1609 examples [00:01, 890.60 examples/s]1709 examples [00:01, 919.63 examples/s]1807 examples [00:01, 936.79 examples/s]1906 examples [00:02, 951.09 examples/s]2006 examples [00:02, 963.84 examples/s]2103 examples [00:02, 965.39 examples/s]2202 examples [00:02, 970.88 examples/s]2300 examples [00:02, 972.27 examples/s]2398 examples [00:02, 961.08 examples/s]2495 examples [00:02, 952.08 examples/s]2595 examples [00:02, 965.42 examples/s]2694 examples [00:02, 970.18 examples/s]2792 examples [00:02, 968.40 examples/s]2889 examples [00:03, 934.73 examples/s]2986 examples [00:03, 942.89 examples/s]3086 examples [00:03, 958.98 examples/s]3183 examples [00:03, 943.28 examples/s]3280 examples [00:03, 948.72 examples/s]3376 examples [00:03, 932.49 examples/s]3476 examples [00:03, 949.79 examples/s]3575 examples [00:03, 958.71 examples/s]3677 examples [00:03, 974.51 examples/s]3776 examples [00:03, 976.43 examples/s]3876 examples [00:04, 982.59 examples/s]3978 examples [00:04, 989.51 examples/s]4078 examples [00:04, 960.12 examples/s]4175 examples [00:04, 950.00 examples/s]4273 examples [00:04, 958.15 examples/s]4373 examples [00:04, 970.24 examples/s]4471 examples [00:04, 970.91 examples/s]4569 examples [00:04, 943.75 examples/s]4664 examples [00:04, 937.65 examples/s]4758 examples [00:05, 919.79 examples/s]4851 examples [00:05, 918.03 examples/s]4943 examples [00:05, 908.98 examples/s]5035 examples [00:05, 909.25 examples/s]5130 examples [00:05, 918.97 examples/s]5223 examples [00:05, 921.68 examples/s]5316 examples [00:05, 917.81 examples/s]5408 examples [00:05, 916.47 examples/s]5500 examples [00:05, 910.25 examples/s]5592 examples [00:05, 911.38 examples/s]5684 examples [00:06, 906.83 examples/s]5776 examples [00:06, 908.77 examples/s]5867 examples [00:06, 903.38 examples/s]5958 examples [00:06, 902.67 examples/s]6049 examples [00:06, 893.52 examples/s]6142 examples [00:06, 902.52 examples/s]6235 examples [00:06, 910.19 examples/s]6338 examples [00:06, 942.65 examples/s]6437 examples [00:06, 953.89 examples/s]6540 examples [00:06, 973.64 examples/s]6639 examples [00:07, 977.22 examples/s]6737 examples [00:07, 965.81 examples/s]6834 examples [00:07, 918.10 examples/s]6933 examples [00:07, 936.89 examples/s]7029 examples [00:07, 943.36 examples/s]7124 examples [00:07, 943.74 examples/s]7219 examples [00:07, 942.09 examples/s]7317 examples [00:07, 952.45 examples/s]7413 examples [00:07, 938.36 examples/s]7507 examples [00:07, 920.58 examples/s]7611 examples [00:08, 951.63 examples/s]7709 examples [00:08, 957.35 examples/s]7806 examples [00:08, 956.44 examples/s]7902 examples [00:08, 935.62 examples/s]8006 examples [00:08, 962.50 examples/s]8108 examples [00:08, 977.24 examples/s]8213 examples [00:08, 996.59 examples/s]8313 examples [00:08, 953.35 examples/s]8409 examples [00:08, 922.89 examples/s]8502 examples [00:09, 921.40 examples/s]8600 examples [00:09, 938.23 examples/s]8699 examples [00:09, 952.89 examples/s]8799 examples [00:09, 965.94 examples/s]8896 examples [00:09, 966.28 examples/s]9001 examples [00:09, 987.72 examples/s]9107 examples [00:09, 1005.96 examples/s]9211 examples [00:09, 1013.26 examples/s]9313 examples [00:09, 1011.22 examples/s]9415 examples [00:09, 986.80 examples/s] 9515 examples [00:10, 989.83 examples/s]9615 examples [00:10, 948.94 examples/s]9711 examples [00:10, 934.03 examples/s]9805 examples [00:10, 902.49 examples/s]9899 examples [00:10, 911.19 examples/s]9994 examples [00:10, 920.48 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteI6O9QA/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteI6O9QA/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff930d65488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff930d65488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff930d65488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff917f125f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff8ba1a4b70>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff930d65488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff930d65488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff930d65488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff8ba1a4a20>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff92e294160>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7ff8ae936158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7ff8ae936158> 

  function with postional parmater data_info <function split_train_valid at 0x7ff8ae936158> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7ff8ae936268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7ff8ae936268> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7ff8ae936268> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=4eef763279db24ad95c954bd2697c0a4515440a4373d526ec712519c216bc9e2
  Stored in directory: /tmp/pip-ephem-wheel-cache-jo5az4a1/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:31:36, 10.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:42:02, 14.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:44:37, 20.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:13:43, 29.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.51M/862M [00:01<5:44:43, 41.5kB/s].vector_cache/glove.6B.zip:   1%|          | 6.64M/862M [00:01<4:00:33, 59.3kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.6M/862M [00:01<2:47:30, 84.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:01<1:56:55, 121kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.9M/862M [00:01<1:21:27, 172kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.4M/862M [00:01<56:53, 246kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 28.6M/862M [00:01<39:39, 350kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.9M/862M [00:02<27:46, 498kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.2M/862M [00:02<19:27, 708kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.5M/862M [00:02<13:38, 1.00MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.8M/862M [00:02<09:35, 1.42MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.0M/862M [00:02<06:46, 2.00MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:02<05:14, 2.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<05:34, 2.41MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<07:30, 1.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<06:03, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.7M/862M [00:05<04:26, 3.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<07:20, 1.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<06:36, 2.02MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:07<04:55, 2.71MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<06:22, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<07:08, 1.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:08<05:33, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:09<04:03, 3.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<10:45, 1.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<08:52, 1.49MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:10<06:32, 2.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<07:39, 1.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<08:01, 1.64MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:12<06:17, 2.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:13<04:32, 2.89MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<15:19, 855kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<11:58, 1.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:14<08:40, 1.51MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:14<06:18, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<14:09, 920kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<12:55, 1.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<09:39, 1.35MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.8M/862M [00:16<06:56, 1.87MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<09:51, 1.31MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<08:23, 1.54MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:18<06:11, 2.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<07:02, 1.83MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<06:24, 2.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:20<04:47, 2.68MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<06:02, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<07:11, 1.78MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<05:39, 2.26MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.4M/862M [00:22<04:07, 3.09MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<08:17, 1.54MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<07:15, 1.75MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<05:27, 2.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:27, 1.96MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:25, 1.71MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:53, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<04:16, 2.95MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:16, 1.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:58, 1.58MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<05:56, 2.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:46, 1.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:47, 1.61MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:04, 2.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<04:25, 2.82MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:40, 1.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:50, 1.82MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<05:05, 2.45MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:08, 2.02MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:17, 1.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:46, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<04:12, 2.93MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<11:10, 1.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<09:16, 1.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<06:50, 1.80MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<05:22, 2.28MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<9:50:50, 20.7kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<6:53:56, 29.6kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<4:48:51, 42.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<3:30:05, 58.0kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<2:29:49, 81.3kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<1:45:30, 115kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<1:13:45, 164kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<1:00:25, 201kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<43:42, 277kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<30:53, 391kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<24:04, 501kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<19:36, 614kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<14:19, 840kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:44<10:09, 1.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<12:44, 941kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<10:19, 1.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<07:33, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:45, 1.53MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:18, 1.43MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:25, 1.85MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<04:37, 2.56MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<09:39, 1.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:08, 1.45MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:02, 1.96MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:40, 1.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:50, 2.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:25, 2.65MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:32, 2.11MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:41, 1.75MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:22, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<03:53, 2.99MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<10:42, 1.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<08:49, 1.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<06:26, 1.80MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<06:57, 1.66MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:14, 1.85MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<04:41, 2.46MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:40, 2.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:46, 1.70MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:25, 2.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<03:55, 2.91MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<10:24, 1.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<08:36, 1.33MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<06:20, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:48, 1.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:23, 1.53MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:51, 1.94MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<04:14, 2.66MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<10:34, 1.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:42, 1.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<06:24, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:49, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<07:22, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:45, 1.95MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<04:10, 2.67MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<10:22, 1.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<08:34, 1.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<06:16, 1.77MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:41, 1.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:59, 1.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:30, 2.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:26, 2.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:27, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:09, 2.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<03:45, 2.91MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<10:29, 1.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<08:37, 1.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<06:20, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:41, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<07:18, 1.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:39, 1.92MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<04:05, 2.65MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<08:03, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:54, 1.56MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<05:08, 2.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:49, 1.84MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:26, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:02, 2.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<03:37, 2.94MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<10:39, 1.00MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<08:42, 1.22MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<06:20, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:39, 1.59MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:59, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:23, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<03:54, 2.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<07:01, 1.50MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:08, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:36, 2.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:23, 1.94MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:00, 2.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<03:48, 2.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:48, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<05:52, 1.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:37, 2.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<03:22, 3.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<06:16, 1.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:37, 1.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:14, 2.42MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<05:03, 2.02MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:53, 1.74MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:44, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<03:25, 2.97MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<09:14, 1.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<07:33, 1.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:30, 1.84MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<06:05, 1.66MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<06:34, 1.53MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<05:07, 1.97MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<03:42, 2.71MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<07:32, 1.33MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<06:22, 1.57MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:43, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:29, 1.81MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<06:14, 1.60MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:54, 2.03MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<03:34, 2.78MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:06, 1.62MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:26, 1.82MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:05, 2.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:53, 2.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:35, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:20, 2.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:09, 3.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<07:00, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:56, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:24, 2.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:13, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:53, 1.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:41, 2.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:24, 2.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<09:23, 1.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<07:40, 1.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:38, 1.70MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:56, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<06:09, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:49, 1.98MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:29, 2.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<16:59, 558kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<12:54, 734kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<09:14, 1.02MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<08:31, 1.10MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<08:14, 1.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<06:14, 1.51MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<04:28, 2.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<07:12, 1.30MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:08, 1.52MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:33, 2.04MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:06, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:40, 1.98MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:32, 2.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:22, 2.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:17, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:09, 2.21MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:03, 3.00MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:00, 1.83MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:34, 2.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:25, 2.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:17, 2.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:54, 1.85MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<03:50, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<02:50, 3.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:57, 1.81MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:26, 2.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:20, 2.68MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:18, 2.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:53, 1.82MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:54, 2.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<02:50, 3.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<15:34, 568kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<11:50, 747kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<08:30, 1.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<07:52, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<07:38, 1.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:46, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:09, 2.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:50, 1.49MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<05:02, 1.73MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:43, 2.34MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:30, 1.92MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:03, 1.71MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:55, 2.20MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:51, 3.00MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:15, 1.63MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:35, 1.86MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<03:26, 2.48MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:17, 1.98MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:47, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:44, 2.27MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<02:42, 3.12MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<06:21, 1.33MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<05:21, 1.57MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:27<03:56, 2.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:36, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:07, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:05, 2.69MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:00, 2.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:42, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:47, 2.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<02:44, 3.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<07:26, 1.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:10, 1.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<04:31, 1.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:51, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:16, 1.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:10, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:01, 2.68MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<07:55, 1.02MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<06:29, 1.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:44, 1.70MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:59, 1.61MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:26, 1.48MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<04:11, 1.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:39<03:03, 2.61MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:45, 1.67MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:15, 1.87MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<03:10, 2.50MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:51, 2.04MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:36, 1.71MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:41, 2.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<02:40, 2.92MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<07:29, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<06:01, 1.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<04:26, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:42, 1.65MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:04, 1.53MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:00, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<02:53, 2.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<07:16, 1.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:54, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<04:19, 1.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:41, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:53, 1.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:45, 2.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<02:43, 2.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:02, 1.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:20, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:12, 2.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:53, 1.92MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:18, 1.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:21, 2.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<02:25, 3.05MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<05:40, 1.30MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:46, 1.55MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:31, 2.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:04, 1.80MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:37, 1.59MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:37, 2.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<02:37, 2.76MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<06:41, 1.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<05:31, 1.31MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:02, 1.79MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:18, 1.67MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:37, 1.55MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:33, 2.02MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<02:35, 2.75MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<02:24, 2.96MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<5:17:56, 22.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<3:42:37, 31.9kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<2:34:57, 45.6kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<1:52:57, 62.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<1:20:39, 87.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<56:45, 124kB/s]   .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<39:35, 177kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<31:56, 219kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<23:10, 301kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<16:22, 425kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<12:50, 538kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<09:47, 705kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<07:00, 983kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<06:18, 1.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<06:04, 1.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:34, 1.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<03:17, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:39, 1.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:03, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:01, 2.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:30, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:15, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:27, 2.71MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:05, 2.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:56, 2.25MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:14, 2.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:56, 2.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:38, 1.80MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:56, 2.23MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<02:08, 3.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<06:09, 1.06MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<05:04, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:43, 1.73MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:56, 1.63MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:18, 1.49MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:22, 1.90MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<02:26, 2.61MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<06:03, 1.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:58, 1.28MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:38, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:51, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<04:00, 1.57MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:08, 1.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<02:16, 2.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<11:22, 546kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<08:41, 715kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<06:14, 992kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<05:37, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<05:25, 1.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:05, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:56, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:25, 1.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:48, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:50, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:13, 1.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:51, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:10, 2.75MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:45, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:14, 1.84MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:31, 2.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<01:50, 3.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:07, 1.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:34, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<02:39, 2.20MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:03, 1.89MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:32, 1.64MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:46, 2.09MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:00, 2.87MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:48, 1.50MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:20, 1.71MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:29, 2.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:55, 1.94MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:24, 1.66MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:39, 2.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<01:55, 2.91MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:45, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:17, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<02:27, 2.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:51, 1.93MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:20, 1.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:39, 2.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<01:55, 2.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<05:18, 1.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<04:21, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<03:12, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:20, 1.61MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:34, 1.51MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:48, 1.91MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<02:02, 2.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<05:17, 1.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:10, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:03, 1.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:17, 1.60MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:24, 1.54MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:36, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<01:54, 2.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:08, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:48, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:06, 2.44MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:31, 2.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:00, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:23, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:44, 2.90MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<04:52, 1.03MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:59, 1.26MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:55, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:04, 1.61MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:17, 1.51MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:35, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<01:52, 2.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<04:52, 1.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:58, 1.23MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:53, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:00, 1.60MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:07, 1.55MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:24, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:44, 2.75MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:17, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:51, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:07, 2.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:27, 1.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:44, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:07, 2.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:32, 3.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:17, 1.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:50, 1.62MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:06, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:25, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:13, 2.04MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:40, 2.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:06, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<02:33, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:00, 2.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:27, 3.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:38, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:22, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<01:46, 2.46MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:08, 2.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:32, 1.71MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:01, 2.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:27, 2.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<04:01, 1.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<03:18, 1.29MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:24, 1.76MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:33, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:42, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:07, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:30, 2.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<07:18, 566kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<05:33, 744kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:58, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<03:39, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<03:29, 1.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:41, 1.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:55, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<04:14, 942kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<03:25, 1.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:30, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:33, 1.54MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:41, 1.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:04, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:29, 2.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:28, 1.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:11, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:37, 2.36MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:55, 1.97MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:15, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<01:45, 2.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:16, 2.93MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:27, 1.51MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:09, 1.72MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:35, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:52, 1.95MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:44, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:18, 2.78MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:39, 2.17MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:56, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:31, 2.35MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:07, 3.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:48, 1.93MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:40, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:15, 2.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:35, 2.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:31, 2.26MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:50<01:09, 2.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:30, 2.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:49, 1.84MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:26, 2.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:02, 3.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:02, 1.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:49, 1.81MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:20, 2.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:36, 2.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:54, 1.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:29, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:04, 2.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:58, 1.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:45, 1.80MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:18, 2.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<01:32, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<01:45, 1.76MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:22, 2.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<00:59, 3.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:45, 1.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:16, 1.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:39, 1.80MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:46, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:51, 1.59MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:25, 2.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:01, 2.82MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:56, 1.49MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:41, 1.70MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:15, 2.26MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:27, 1.93MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:41, 1.66MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:19, 2.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<00:57, 2.90MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:12, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:51, 1.48MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:22, 1.98MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:30, 1.78MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:40, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:19, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<00:56, 2.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:24, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:59, 1.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:26, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:31, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:41, 1.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:18, 1.94MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:56, 2.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:38, 1.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:26, 1.72MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:03, 2.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:14, 1.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:26, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:07, 2.12MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:48, 2.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:22, 1.69MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:14, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:55, 2.49MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:06, 2.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:19, 1.71MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:01, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:44, 2.96MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:43, 3.05MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<1:35:13, 23.0kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<1:06:25, 32.8kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<45:39, 46.9kB/s]  .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<32:47, 64.7kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<23:25, 90.5kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<16:24, 129kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<11:21, 183kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<08:32, 240kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<06:10, 331kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<04:19, 467kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<03:24, 582kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:50, 698kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:05, 941kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<01:27, 1.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:24, 794kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:53, 1.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:21, 1.39MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:20, 1.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:21, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:02, 1.75MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:44, 2.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:44, 1.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:24, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:00, 1.72MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:04, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:09, 1.47MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:54, 1.87MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<00:38, 2.56MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:37, 1.01MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:18, 1.24MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:56, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:59, 1.58MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:04, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:49, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:34, 2.60MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:03, 1.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:54, 1.63MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:40, 2.18MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:45, 1.89MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:50, 1.70MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:39, 2.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:48<00:27, 2.95MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<02:22, 573kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:48, 746kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:17, 1.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:08, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:06, 1.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:50, 1.53MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:34, 2.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:06, 1.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:54, 1.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:39, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:41, 1.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:45, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:34, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:24, 2.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:38, 1.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:34, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:25, 2.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:29, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:35, 1.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:27, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:19, 2.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:31, 1.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:28, 1.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:21, 2.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:25, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:23, 2.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:17, 2.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:21, 2.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:26, 1.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:20, 2.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:14, 3.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:25, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:20, 2.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:13, 2.92MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:38, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:36, 1.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:27, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:18, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:31, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:25, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:18, 1.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:18, 1.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:19, 1.60MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:14, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:10, 2.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:17, 1.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:10, 2.39MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:11, 1.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.68MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:10, 2.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:06, 2.94MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:14, 1.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:11, 1.57MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:08, 2.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:08, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:09, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:04, 2.84MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:08, 1.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:06, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 2.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.04MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 2.79MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.59MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.83MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 2.44MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<18:38:22,  5.96it/s]  0%|          | 937/400000 [00:00<13:01:14,  8.51it/s]  0%|          | 1799/400000 [00:00<9:05:55, 12.16it/s]  1%|          | 2774/400000 [00:00<6:21:24, 17.36it/s]  1%|          | 3706/400000 [00:00<4:26:34, 24.78it/s]  1%|          | 4596/400000 [00:00<3:06:24, 35.35it/s]  1%|▏         | 5481/400000 [00:00<2:10:24, 50.42it/s]  2%|▏         | 6385/400000 [00:00<1:31:17, 71.85it/s]  2%|▏         | 7379/400000 [00:00<1:03:56, 102.33it/s]  2%|▏         | 8376/400000 [00:01<44:50, 145.55it/s]    2%|▏         | 9352/400000 [00:01<31:30, 206.60it/s]  3%|▎         | 10292/400000 [00:01<22:12, 292.39it/s]  3%|▎         | 11227/400000 [00:01<15:44, 411.79it/s]  3%|▎         | 12143/400000 [00:01<11:12, 576.71it/s]  3%|▎         | 13108/400000 [00:01<08:01, 803.29it/s]  4%|▎         | 14087/400000 [00:01<05:48, 1108.54it/s]  4%|▍         | 15027/400000 [00:01<04:15, 1505.47it/s]  4%|▍         | 15960/400000 [00:01<03:11, 2005.89it/s]  4%|▍         | 16932/400000 [00:01<02:25, 2632.63it/s]  4%|▍         | 17907/400000 [00:02<01:53, 3370.60it/s]  5%|▍         | 18864/400000 [00:02<01:31, 4183.38it/s]  5%|▍         | 19815/400000 [00:02<01:16, 4992.66it/s]  5%|▌         | 20754/400000 [00:02<01:06, 5668.53it/s]  5%|▌         | 21722/400000 [00:02<00:58, 6472.38it/s]  6%|▌         | 22695/400000 [00:02<00:52, 7194.69it/s]  6%|▌         | 23662/400000 [00:02<00:48, 7791.48it/s]  6%|▌         | 24612/400000 [00:02<00:45, 8235.24it/s]  6%|▋         | 25560/400000 [00:02<00:44, 8460.42it/s]  7%|▋         | 26571/400000 [00:03<00:41, 8892.39it/s]  7%|▋         | 27560/400000 [00:03<00:40, 9168.68it/s]  7%|▋         | 28550/400000 [00:03<00:39, 9374.84it/s]  7%|▋         | 29554/400000 [00:03<00:38, 9562.94it/s]  8%|▊         | 30537/400000 [00:03<00:39, 9412.77it/s]  8%|▊         | 31498/400000 [00:03<00:38, 9448.97it/s]  8%|▊         | 32457/400000 [00:03<00:39, 9417.74it/s]  8%|▊         | 33409/400000 [00:03<00:38, 9443.71it/s]  9%|▊         | 34360/400000 [00:03<00:38, 9453.84it/s]  9%|▉         | 35310/400000 [00:03<00:39, 9276.87it/s]  9%|▉         | 36242/400000 [00:04<00:40, 8914.72it/s]  9%|▉         | 37140/400000 [00:04<00:43, 8341.04it/s]  9%|▉         | 37986/400000 [00:04<00:46, 7839.35it/s] 10%|▉         | 38785/400000 [00:04<00:47, 7611.88it/s] 10%|▉         | 39590/400000 [00:04<00:46, 7738.07it/s] 10%|█         | 40386/400000 [00:04<00:46, 7801.27it/s] 10%|█         | 41173/400000 [00:04<00:46, 7789.97it/s] 10%|█         | 41957/400000 [00:04<00:46, 7728.97it/s] 11%|█         | 42733/400000 [00:04<00:46, 7703.74it/s] 11%|█         | 43531/400000 [00:05<00:45, 7783.69it/s] 11%|█         | 44377/400000 [00:05<00:44, 7973.70it/s] 11%|█▏        | 45238/400000 [00:05<00:43, 8153.25it/s] 12%|█▏        | 46174/400000 [00:05<00:41, 8480.32it/s] 12%|█▏        | 47061/400000 [00:05<00:41, 8592.72it/s] 12%|█▏        | 48048/400000 [00:05<00:39, 8939.02it/s] 12%|█▏        | 48949/400000 [00:05<00:39, 8920.76it/s] 12%|█▏        | 49846/400000 [00:05<00:40, 8732.78it/s] 13%|█▎        | 50766/400000 [00:05<00:39, 8866.69it/s] 13%|█▎        | 51692/400000 [00:05<00:38, 8979.26it/s] 13%|█▎        | 52593/400000 [00:06<00:39, 8905.47it/s] 13%|█▎        | 53503/400000 [00:06<00:38, 8961.62it/s] 14%|█▎        | 54433/400000 [00:06<00:38, 9056.59it/s] 14%|█▍        | 55353/400000 [00:06<00:37, 9097.42it/s] 14%|█▍        | 56264/400000 [00:06<00:38, 8988.00it/s] 14%|█▍        | 57186/400000 [00:06<00:37, 9055.96it/s] 15%|█▍        | 58121/400000 [00:06<00:37, 9141.62it/s] 15%|█▍        | 59036/400000 [00:06<00:37, 9139.28it/s] 15%|█▌        | 60018/400000 [00:06<00:36, 9332.90it/s] 15%|█▌        | 60953/400000 [00:06<00:36, 9250.41it/s] 15%|█▌        | 61925/400000 [00:07<00:36, 9384.99it/s] 16%|█▌        | 62866/400000 [00:07<00:35, 9389.79it/s] 16%|█▌        | 63825/400000 [00:07<00:35, 9448.93it/s] 16%|█▌        | 64787/400000 [00:07<00:35, 9498.04it/s] 16%|█▋        | 65738/400000 [00:07<00:36, 9040.92it/s] 17%|█▋        | 66648/400000 [00:07<00:37, 8985.36it/s] 17%|█▋        | 67622/400000 [00:07<00:36, 9198.81it/s] 17%|█▋        | 68570/400000 [00:07<00:35, 9281.33it/s] 17%|█▋        | 69530/400000 [00:07<00:35, 9370.08it/s] 18%|█▊        | 70470/400000 [00:07<00:35, 9314.44it/s] 18%|█▊        | 71404/400000 [00:08<00:35, 9246.04it/s] 18%|█▊        | 72354/400000 [00:08<00:35, 9318.18it/s] 18%|█▊        | 73287/400000 [00:08<00:35, 9274.63it/s] 19%|█▊        | 74217/400000 [00:08<00:35, 9279.14it/s] 19%|█▉        | 75146/400000 [00:08<00:35, 9159.48it/s] 19%|█▉        | 76075/400000 [00:08<00:35, 9196.49it/s] 19%|█▉        | 76996/400000 [00:08<00:35, 9171.62it/s] 19%|█▉        | 77927/400000 [00:08<00:34, 9212.44it/s] 20%|█▉        | 78852/400000 [00:08<00:34, 9222.01it/s] 20%|█▉        | 79775/400000 [00:08<00:35, 9074.21it/s] 20%|██        | 80685/400000 [00:09<00:35, 9080.16it/s] 20%|██        | 81613/400000 [00:09<00:34, 9138.48it/s] 21%|██        | 82575/400000 [00:09<00:34, 9277.18it/s] 21%|██        | 83547/400000 [00:09<00:33, 9404.74it/s] 21%|██        | 84489/400000 [00:09<00:33, 9299.40it/s] 21%|██▏       | 85425/400000 [00:09<00:33, 9317.38it/s] 22%|██▏       | 86381/400000 [00:09<00:33, 9386.49it/s] 22%|██▏       | 87321/400000 [00:09<00:33, 9247.45it/s] 22%|██▏       | 88247/400000 [00:09<00:35, 8812.56it/s] 22%|██▏       | 89134/400000 [00:09<00:35, 8704.41it/s] 23%|██▎       | 90009/400000 [00:10<00:36, 8582.29it/s] 23%|██▎       | 90919/400000 [00:10<00:35, 8731.18it/s] 23%|██▎       | 91896/400000 [00:10<00:34, 9016.15it/s] 23%|██▎       | 92843/400000 [00:10<00:33, 9146.59it/s] 23%|██▎       | 93762/400000 [00:10<00:34, 8904.18it/s] 24%|██▎       | 94657/400000 [00:10<00:34, 8753.27it/s] 24%|██▍       | 95548/400000 [00:10<00:34, 8798.47it/s] 24%|██▍       | 96489/400000 [00:10<00:33, 8970.01it/s] 24%|██▍       | 97423/400000 [00:10<00:33, 9076.34it/s] 25%|██▍       | 98333/400000 [00:11<00:33, 9059.89it/s] 25%|██▍       | 99241/400000 [00:11<00:34, 8778.29it/s] 25%|██▌       | 100122/400000 [00:11<00:35, 8440.94it/s] 25%|██▌       | 100971/400000 [00:11<00:36, 8188.56it/s] 25%|██▌       | 101795/400000 [00:11<00:37, 8031.88it/s] 26%|██▌       | 102675/400000 [00:11<00:36, 8245.87it/s] 26%|██▌       | 103592/400000 [00:11<00:34, 8502.35it/s] 26%|██▌       | 104523/400000 [00:11<00:33, 8728.04it/s] 26%|██▋       | 105457/400000 [00:11<00:33, 8902.94it/s] 27%|██▋       | 106355/400000 [00:11<00:32, 8925.10it/s] 27%|██▋       | 107272/400000 [00:12<00:32, 8995.30it/s] 27%|██▋       | 108254/400000 [00:12<00:31, 9226.82it/s] 27%|██▋       | 109210/400000 [00:12<00:31, 9324.07it/s] 28%|██▊       | 110145/400000 [00:12<00:31, 9268.54it/s] 28%|██▊       | 111074/400000 [00:12<00:31, 9256.60it/s] 28%|██▊       | 112001/400000 [00:12<00:31, 9068.54it/s] 28%|██▊       | 112910/400000 [00:12<00:31, 9016.36it/s] 28%|██▊       | 113813/400000 [00:12<00:31, 8972.51it/s] 29%|██▊       | 114758/400000 [00:12<00:31, 9109.97it/s] 29%|██▉       | 115679/400000 [00:12<00:31, 9138.05it/s] 29%|██▉       | 116594/400000 [00:13<00:31, 9138.68it/s] 29%|██▉       | 117561/400000 [00:13<00:30, 9291.29it/s] 30%|██▉       | 118516/400000 [00:13<00:30, 9365.31it/s] 30%|██▉       | 119454/400000 [00:13<00:30, 9301.32it/s] 30%|███       | 120385/400000 [00:13<00:30, 9096.53it/s] 30%|███       | 121297/400000 [00:13<00:30, 9022.18it/s] 31%|███       | 122264/400000 [00:13<00:30, 9205.12it/s] 31%|███       | 123187/400000 [00:13<00:30, 9052.84it/s] 31%|███       | 124127/400000 [00:13<00:30, 9152.83it/s] 31%|███▏      | 125044/400000 [00:14<00:30, 9058.43it/s] 31%|███▏      | 125952/400000 [00:14<00:31, 8817.52it/s] 32%|███▏      | 126927/400000 [00:14<00:30, 9053.56it/s] 32%|███▏      | 127858/400000 [00:14<00:29, 9127.86it/s] 32%|███▏      | 128774/400000 [00:14<00:30, 9035.77it/s] 32%|███▏      | 129739/400000 [00:14<00:29, 9210.78it/s] 33%|███▎      | 130663/400000 [00:14<00:29, 9075.19it/s] 33%|███▎      | 131588/400000 [00:14<00:29, 9125.92it/s] 33%|███▎      | 132541/400000 [00:14<00:28, 9240.90it/s] 33%|███▎      | 133492/400000 [00:14<00:28, 9319.24it/s] 34%|███▎      | 134466/400000 [00:15<00:28, 9439.90it/s] 34%|███▍      | 135412/400000 [00:15<00:28, 9301.33it/s] 34%|███▍      | 136344/400000 [00:15<00:28, 9291.55it/s] 34%|███▍      | 137299/400000 [00:15<00:28, 9366.35it/s] 35%|███▍      | 138259/400000 [00:15<00:27, 9435.04it/s] 35%|███▍      | 139220/400000 [00:15<00:27, 9484.38it/s] 35%|███▌      | 140169/400000 [00:15<00:29, 8933.52it/s] 35%|███▌      | 141070/400000 [00:15<00:29, 8849.96it/s] 35%|███▌      | 141961/400000 [00:15<00:29, 8827.28it/s] 36%|███▌      | 142910/400000 [00:15<00:28, 9013.74it/s] 36%|███▌      | 143883/400000 [00:16<00:27, 9215.05it/s] 36%|███▌      | 144809/400000 [00:16<00:28, 9090.18it/s] 36%|███▋      | 145754/400000 [00:16<00:27, 9194.12it/s] 37%|███▋      | 146714/400000 [00:16<00:27, 9309.71it/s] 37%|███▋      | 147647/400000 [00:16<00:27, 9151.90it/s] 37%|███▋      | 148565/400000 [00:16<00:27, 9106.93it/s] 37%|███▋      | 149478/400000 [00:16<00:27, 9018.17it/s] 38%|███▊      | 150397/400000 [00:16<00:27, 9067.63it/s] 38%|███▊      | 151355/400000 [00:16<00:26, 9213.33it/s] 38%|███▊      | 152305/400000 [00:16<00:26, 9295.92it/s] 38%|███▊      | 153249/400000 [00:17<00:26, 9335.44it/s] 39%|███▊      | 154188/400000 [00:17<00:26, 9351.45it/s] 39%|███▉      | 155149/400000 [00:17<00:25, 9426.11it/s] 39%|███▉      | 156093/400000 [00:17<00:26, 9222.45it/s] 39%|███▉      | 157017/400000 [00:17<00:26, 9028.70it/s] 39%|███▉      | 157985/400000 [00:17<00:26, 9212.71it/s] 40%|███▉      | 158909/400000 [00:17<00:26, 9113.20it/s] 40%|███▉      | 159823/400000 [00:17<00:26, 8948.13it/s] 40%|████      | 160751/400000 [00:17<00:26, 9042.81it/s] 40%|████      | 161657/400000 [00:17<00:26, 8977.75it/s] 41%|████      | 162605/400000 [00:18<00:26, 9121.42it/s] 41%|████      | 163519/400000 [00:18<00:26, 9056.53it/s] 41%|████      | 164497/400000 [00:18<00:25, 9259.91it/s] 41%|████▏     | 165468/400000 [00:18<00:24, 9389.56it/s] 42%|████▏     | 166421/400000 [00:18<00:24, 9428.72it/s] 42%|████▏     | 167366/400000 [00:18<00:24, 9307.11it/s] 42%|████▏     | 168298/400000 [00:18<00:25, 9088.60it/s] 42%|████▏     | 169224/400000 [00:18<00:25, 9137.27it/s] 43%|████▎     | 170140/400000 [00:18<00:25, 8968.23it/s] 43%|████▎     | 171039/400000 [00:19<00:25, 8942.54it/s] 43%|████▎     | 171940/400000 [00:19<00:25, 8961.19it/s] 43%|████▎     | 172837/400000 [00:19<00:25, 8926.79it/s] 43%|████▎     | 173737/400000 [00:19<00:25, 8948.18it/s] 44%|████▎     | 174674/400000 [00:19<00:24, 9068.12it/s] 44%|████▍     | 175599/400000 [00:19<00:24, 9116.70it/s] 44%|████▍     | 176514/400000 [00:19<00:24, 9126.00it/s] 44%|████▍     | 177427/400000 [00:19<00:24, 9061.85it/s] 45%|████▍     | 178365/400000 [00:19<00:24, 9153.75it/s] 45%|████▍     | 179281/400000 [00:19<00:24, 9049.39it/s] 45%|████▌     | 180195/400000 [00:20<00:24, 9074.56it/s] 45%|████▌     | 181103/400000 [00:20<00:25, 8734.29it/s] 45%|████▌     | 181980/400000 [00:20<00:24, 8738.71it/s] 46%|████▌     | 182926/400000 [00:20<00:24, 8941.65it/s] 46%|████▌     | 183904/400000 [00:20<00:23, 9176.87it/s] 46%|████▌     | 184826/400000 [00:20<00:23, 9086.47it/s] 46%|████▋     | 185738/400000 [00:20<00:24, 8919.89it/s] 47%|████▋     | 186633/400000 [00:20<00:24, 8887.35it/s] 47%|████▋     | 187574/400000 [00:20<00:23, 9037.79it/s] 47%|████▋     | 188524/400000 [00:20<00:23, 9169.49it/s] 47%|████▋     | 189443/400000 [00:21<00:23, 9065.14it/s] 48%|████▊     | 190377/400000 [00:21<00:22, 9144.65it/s] 48%|████▊     | 191293/400000 [00:21<00:22, 9144.15it/s] 48%|████▊     | 192209/400000 [00:21<00:22, 9131.38it/s] 48%|████▊     | 193174/400000 [00:21<00:22, 9274.22it/s] 49%|████▊     | 194145/400000 [00:21<00:21, 9399.56it/s] 49%|████▉     | 195087/400000 [00:21<00:22, 9273.68it/s] 49%|████▉     | 196016/400000 [00:21<00:22, 9072.30it/s] 49%|████▉     | 196969/400000 [00:21<00:22, 9203.54it/s] 49%|████▉     | 197927/400000 [00:21<00:21, 9311.66it/s] 50%|████▉     | 198873/400000 [00:22<00:21, 9354.39it/s] 50%|████▉     | 199810/400000 [00:22<00:21, 9313.62it/s] 50%|█████     | 200743/400000 [00:22<00:22, 8918.10it/s] 50%|█████     | 201694/400000 [00:22<00:21, 9086.32it/s] 51%|█████     | 202620/400000 [00:22<00:21, 9137.64it/s] 51%|█████     | 203537/400000 [00:22<00:21, 9099.00it/s] 51%|█████     | 204454/400000 [00:22<00:21, 9118.47it/s] 51%|█████▏    | 205368/400000 [00:22<00:21, 9099.43it/s] 52%|█████▏    | 206316/400000 [00:22<00:21, 9207.08it/s] 52%|█████▏    | 207239/400000 [00:22<00:20, 9213.64it/s] 52%|█████▏    | 208182/400000 [00:23<00:20, 9276.19it/s] 52%|█████▏    | 209117/400000 [00:23<00:20, 9297.89it/s] 53%|█████▎    | 210048/400000 [00:23<00:20, 9231.92it/s] 53%|█████▎    | 211014/400000 [00:23<00:20, 9356.14it/s] 53%|█████▎    | 211951/400000 [00:23<00:20, 9326.38it/s] 53%|█████▎    | 212885/400000 [00:23<00:20, 9158.82it/s] 53%|█████▎    | 213820/400000 [00:23<00:20, 9213.53it/s] 54%|█████▎    | 214743/400000 [00:23<00:20, 9097.12it/s] 54%|█████▍    | 215654/400000 [00:23<00:20, 9003.94it/s] 54%|█████▍    | 216556/400000 [00:24<00:20, 8843.54it/s] 54%|█████▍    | 217487/400000 [00:24<00:20, 8978.27it/s] 55%|█████▍    | 218397/400000 [00:24<00:20, 9012.77it/s] 55%|█████▍    | 219300/400000 [00:24<00:20, 8932.98it/s] 55%|█████▌    | 220259/400000 [00:24<00:19, 9119.28it/s] 55%|█████▌    | 221212/400000 [00:24<00:19, 9236.41it/s] 56%|█████▌    | 222169/400000 [00:24<00:19, 9331.84it/s] 56%|█████▌    | 223104/400000 [00:24<00:19, 9145.79it/s] 56%|█████▌    | 224021/400000 [00:24<00:19, 8878.51it/s] 56%|█████▌    | 224912/400000 [00:24<00:19, 8849.55it/s] 56%|█████▋    | 225800/400000 [00:25<00:19, 8835.68it/s] 57%|█████▋    | 226686/400000 [00:25<00:20, 8494.84it/s] 57%|█████▋    | 227579/400000 [00:25<00:20, 8619.86it/s] 57%|█████▋    | 228445/400000 [00:25<00:20, 8442.47it/s] 57%|█████▋    | 229325/400000 [00:25<00:19, 8544.48it/s] 58%|█████▊    | 230182/400000 [00:25<00:20, 8263.09it/s] 58%|█████▊    | 231012/400000 [00:25<00:20, 8243.92it/s] 58%|█████▊    | 231877/400000 [00:25<00:20, 8359.89it/s] 58%|█████▊    | 232766/400000 [00:25<00:19, 8510.48it/s] 58%|█████▊    | 233712/400000 [00:25<00:18, 8773.59it/s] 59%|█████▊    | 234593/400000 [00:26<00:18, 8710.80it/s] 59%|█████▉    | 235515/400000 [00:26<00:18, 8857.41it/s] 59%|█████▉    | 236404/400000 [00:26<00:18, 8738.19it/s] 59%|█████▉    | 237300/400000 [00:26<00:18, 8800.11it/s] 60%|█████▉    | 238238/400000 [00:26<00:18, 8964.40it/s] 60%|█████▉    | 239137/400000 [00:26<00:18, 8816.18it/s] 60%|██████    | 240093/400000 [00:26<00:17, 9025.87it/s] 60%|██████    | 240999/400000 [00:26<00:18, 8589.22it/s] 60%|██████    | 241865/400000 [00:26<00:19, 8245.56it/s] 61%|██████    | 242697/400000 [00:27<00:19, 7942.51it/s] 61%|██████    | 243532/400000 [00:27<00:19, 8058.93it/s] 61%|██████    | 244432/400000 [00:27<00:18, 8319.49it/s] 61%|██████▏   | 245313/400000 [00:27<00:18, 8456.89it/s] 62%|██████▏   | 246277/400000 [00:27<00:17, 8778.92it/s] 62%|██████▏   | 247224/400000 [00:27<00:17, 8973.86it/s] 62%|██████▏   | 248136/400000 [00:27<00:16, 9013.22it/s] 62%|██████▏   | 249042/400000 [00:27<00:16, 8932.72it/s] 62%|██████▏   | 249939/400000 [00:27<00:17, 8683.52it/s] 63%|██████▎   | 250811/400000 [00:27<00:17, 8640.79it/s] 63%|██████▎   | 251678/400000 [00:28<00:17, 8632.80it/s] 63%|██████▎   | 252544/400000 [00:28<00:17, 8621.26it/s] 63%|██████▎   | 253408/400000 [00:28<00:17, 8556.88it/s] 64%|██████▎   | 254272/400000 [00:28<00:16, 8579.52it/s] 64%|██████▍   | 255178/400000 [00:28<00:16, 8717.38it/s] 64%|██████▍   | 256079/400000 [00:28<00:16, 8801.32it/s] 64%|██████▍   | 256961/400000 [00:28<00:16, 8655.43it/s] 64%|██████▍   | 257842/400000 [00:28<00:16, 8699.36it/s] 65%|██████▍   | 258713/400000 [00:28<00:16, 8669.82it/s] 65%|██████▍   | 259581/400000 [00:28<00:16, 8466.73it/s] 65%|██████▌   | 260505/400000 [00:29<00:16, 8683.02it/s] 65%|██████▌   | 261438/400000 [00:29<00:15, 8865.23it/s] 66%|██████▌   | 262328/400000 [00:29<00:15, 8821.66it/s] 66%|██████▌   | 263213/400000 [00:29<00:16, 8395.63it/s] 66%|██████▌   | 264059/400000 [00:29<00:16, 8376.35it/s] 66%|██████▌   | 264956/400000 [00:29<00:15, 8545.23it/s] 66%|██████▋   | 265892/400000 [00:29<00:15, 8771.24it/s] 67%|██████▋   | 266813/400000 [00:29<00:14, 8897.39it/s] 67%|██████▋   | 267753/400000 [00:29<00:14, 9040.75it/s] 67%|██████▋   | 268679/400000 [00:29<00:14, 9104.00it/s] 67%|██████▋   | 269592/400000 [00:30<00:14, 9017.54it/s] 68%|██████▊   | 270568/400000 [00:30<00:14, 9224.29it/s] 68%|██████▊   | 271516/400000 [00:30<00:13, 9296.22it/s] 68%|██████▊   | 272448/400000 [00:30<00:13, 9179.99it/s] 68%|██████▊   | 273368/400000 [00:30<00:13, 9051.47it/s] 69%|██████▊   | 274275/400000 [00:30<00:14, 8930.33it/s] 69%|██████▉   | 275171/400000 [00:30<00:13, 8937.98it/s] 69%|██████▉   | 276066/400000 [00:30<00:14, 8741.50it/s] 69%|██████▉   | 276942/400000 [00:30<00:14, 8584.46it/s] 69%|██████▉   | 277831/400000 [00:31<00:14, 8671.50it/s] 70%|██████▉   | 278735/400000 [00:31<00:13, 8778.23it/s] 70%|██████▉   | 279690/400000 [00:31<00:13, 8994.89it/s] 70%|███████   | 280618/400000 [00:31<00:13, 9077.29it/s] 70%|███████   | 281528/400000 [00:31<00:13, 8948.72it/s] 71%|███████   | 282480/400000 [00:31<00:12, 9110.87it/s] 71%|███████   | 283437/400000 [00:31<00:12, 9242.19it/s] 71%|███████   | 284363/400000 [00:31<00:12, 8991.88it/s] 71%|███████▏  | 285265/400000 [00:31<00:13, 8717.01it/s] 72%|███████▏  | 286141/400000 [00:31<00:13, 8711.00it/s] 72%|███████▏  | 287063/400000 [00:32<00:12, 8854.81it/s] 72%|███████▏  | 287973/400000 [00:32<00:12, 8925.18it/s] 72%|███████▏  | 288911/400000 [00:32<00:12, 9055.11it/s] 72%|███████▏  | 289831/400000 [00:32<00:12, 9096.43it/s] 73%|███████▎  | 290750/400000 [00:32<00:11, 9122.01it/s] 73%|███████▎  | 291714/400000 [00:32<00:11, 9270.12it/s] 73%|███████▎  | 292671/400000 [00:32<00:11, 9358.04it/s] 73%|███████▎  | 293633/400000 [00:32<00:11, 9433.20it/s] 74%|███████▎  | 294578/400000 [00:32<00:11, 9359.04it/s] 74%|███████▍  | 295553/400000 [00:32<00:11, 9472.30it/s] 74%|███████▍  | 296531/400000 [00:33<00:10, 9561.88it/s] 74%|███████▍  | 297491/400000 [00:33<00:10, 9571.13it/s] 75%|███████▍  | 298449/400000 [00:33<00:10, 9428.46it/s] 75%|███████▍  | 299393/400000 [00:33<00:10, 9323.52it/s] 75%|███████▌  | 300345/400000 [00:33<00:10, 9380.64it/s] 75%|███████▌  | 301284/400000 [00:33<00:10, 9383.02it/s] 76%|███████▌  | 302251/400000 [00:33<00:10, 9466.57it/s] 76%|███████▌  | 303199/400000 [00:33<00:10, 8985.66it/s] 76%|███████▌  | 304104/400000 [00:33<00:10, 8768.65it/s] 76%|███████▌  | 304986/400000 [00:33<00:10, 8780.02it/s] 76%|███████▋  | 305887/400000 [00:34<00:10, 8847.40it/s] 77%|███████▋  | 306784/400000 [00:34<00:10, 8880.25it/s] 77%|███████▋  | 307689/400000 [00:34<00:10, 8928.61it/s] 77%|███████▋  | 308605/400000 [00:34<00:10, 8994.66it/s] 77%|███████▋  | 309536/400000 [00:34<00:09, 9086.09it/s] 78%|███████▊  | 310455/400000 [00:34<00:09, 9113.73it/s] 78%|███████▊  | 311368/400000 [00:34<00:09, 9077.00it/s] 78%|███████▊  | 312277/400000 [00:34<00:10, 8766.00it/s] 78%|███████▊  | 313157/400000 [00:34<00:10, 8633.59it/s] 79%|███████▊  | 314121/400000 [00:35<00:09, 8910.61it/s] 79%|███████▉  | 315082/400000 [00:35<00:09, 9106.89it/s] 79%|███████▉  | 315999/400000 [00:35<00:09, 9125.43it/s] 79%|███████▉  | 316986/400000 [00:35<00:08, 9336.39it/s] 79%|███████▉  | 317923/400000 [00:35<00:08, 9261.48it/s] 80%|███████▉  | 318852/400000 [00:35<00:08, 9137.94it/s] 80%|███████▉  | 319768/400000 [00:35<00:08, 9044.37it/s] 80%|████████  | 320675/400000 [00:35<00:08, 9026.12it/s] 80%|████████  | 321598/400000 [00:35<00:08, 9085.52it/s] 81%|████████  | 322530/400000 [00:35<00:08, 9152.47it/s] 81%|████████  | 323451/400000 [00:36<00:08, 9167.92it/s] 81%|████████  | 324393/400000 [00:36<00:08, 9241.81it/s] 81%|████████▏ | 325353/400000 [00:36<00:07, 9344.98it/s] 82%|████████▏ | 326289/400000 [00:36<00:08, 9166.04it/s] 82%|████████▏ | 327207/400000 [00:36<00:07, 9148.76it/s] 82%|████████▏ | 328123/400000 [00:36<00:07, 9146.81it/s] 82%|████████▏ | 329065/400000 [00:36<00:07, 9224.05it/s] 82%|████████▏ | 329988/400000 [00:36<00:07, 9204.92it/s] 83%|████████▎ | 330978/400000 [00:36<00:07, 9400.26it/s] 83%|████████▎ | 331920/400000 [00:36<00:07, 9339.69it/s] 83%|████████▎ | 332855/400000 [00:37<00:07, 9211.35it/s] 83%|████████▎ | 333778/400000 [00:37<00:07, 9057.92it/s] 84%|████████▎ | 334711/400000 [00:37<00:07, 9135.52it/s] 84%|████████▍ | 335661/400000 [00:37<00:06, 9241.22it/s] 84%|████████▍ | 336602/400000 [00:37<00:06, 9289.12it/s] 84%|████████▍ | 337557/400000 [00:37<00:06, 9363.45it/s] 85%|████████▍ | 338495/400000 [00:37<00:06, 9319.69it/s] 85%|████████▍ | 339428/400000 [00:37<00:06, 9231.26it/s] 85%|████████▌ | 340385/400000 [00:37<00:06, 9330.30it/s] 85%|████████▌ | 341319/400000 [00:37<00:06, 9213.23it/s] 86%|████████▌ | 342292/400000 [00:38<00:06, 9360.84it/s] 86%|████████▌ | 343265/400000 [00:38<00:05, 9467.68it/s] 86%|████████▌ | 344213/400000 [00:38<00:05, 9386.24it/s] 86%|████████▋ | 345160/400000 [00:38<00:05, 9409.12it/s] 87%|████████▋ | 346102/400000 [00:38<00:05, 9194.68it/s] 87%|████████▋ | 347024/400000 [00:38<00:05, 9170.76it/s] 87%|████████▋ | 347967/400000 [00:38<00:05, 9244.76it/s] 87%|████████▋ | 348893/400000 [00:38<00:05, 8911.39it/s] 87%|████████▋ | 349807/400000 [00:38<00:05, 8977.32it/s] 88%|████████▊ | 350749/400000 [00:38<00:05, 9103.14it/s] 88%|████████▊ | 351662/400000 [00:39<00:05, 8757.19it/s] 88%|████████▊ | 352542/400000 [00:39<00:05, 8403.24it/s] 88%|████████▊ | 353389/400000 [00:39<00:05, 8212.11it/s] 89%|████████▊ | 354220/400000 [00:39<00:05, 8240.89it/s] 89%|████████▉ | 355120/400000 [00:39<00:05, 8452.63it/s] 89%|████████▉ | 356097/400000 [00:39<00:04, 8806.74it/s] 89%|████████▉ | 357046/400000 [00:39<00:04, 8998.18it/s] 90%|████████▉ | 358001/400000 [00:39<00:04, 9155.34it/s] 90%|████████▉ | 358994/400000 [00:39<00:04, 9372.97it/s] 90%|████████▉ | 359936/400000 [00:40<00:04, 9161.75it/s] 90%|█████████ | 360857/400000 [00:40<00:04, 9062.95it/s] 90%|█████████ | 361828/400000 [00:40<00:04, 9246.41it/s] 91%|█████████ | 362764/400000 [00:40<00:04, 9278.26it/s] 91%|█████████ | 363698/400000 [00:40<00:03, 9295.19it/s] 91%|█████████ | 364630/400000 [00:40<00:03, 8935.76it/s] 91%|█████████▏| 365563/400000 [00:40<00:03, 9048.38it/s] 92%|█████████▏| 366472/400000 [00:40<00:03, 8943.34it/s] 92%|█████████▏| 367369/400000 [00:40<00:03, 8930.03it/s] 92%|█████████▏| 368264/400000 [00:40<00:03, 8914.80it/s] 92%|█████████▏| 369183/400000 [00:41<00:03, 8993.14it/s] 93%|█████████▎| 370084/400000 [00:41<00:03, 8898.97it/s] 93%|█████████▎| 370975/400000 [00:41<00:03, 8893.76it/s] 93%|█████████▎| 371866/400000 [00:41<00:03, 8822.80it/s] 93%|█████████▎| 372820/400000 [00:41<00:03, 9025.97it/s] 93%|█████████▎| 373725/400000 [00:41<00:02, 9012.29it/s] 94%|█████████▎| 374656/400000 [00:41<00:02, 9097.95it/s] 94%|█████████▍| 375567/400000 [00:41<00:02, 9037.20it/s] 94%|█████████▍| 376506/400000 [00:41<00:02, 9139.82it/s] 94%|█████████▍| 377454/400000 [00:41<00:02, 9238.77it/s] 95%|█████████▍| 378379/400000 [00:42<00:02, 9090.02it/s] 95%|█████████▍| 379290/400000 [00:42<00:02, 9058.64it/s] 95%|█████████▌| 380226/400000 [00:42<00:02, 9145.72it/s] 95%|█████████▌| 381155/400000 [00:42<00:02, 9186.28it/s] 96%|█████████▌| 382149/400000 [00:42<00:01, 9400.06it/s] 96%|█████████▌| 383091/400000 [00:42<00:01, 9252.71it/s] 96%|█████████▌| 384058/400000 [00:42<00:01, 9372.21it/s] 96%|█████████▋| 385028/400000 [00:42<00:01, 9467.32it/s] 97%|█████████▋| 386010/400000 [00:42<00:01, 9567.55it/s] 97%|█████████▋| 386987/400000 [00:42<00:01, 9627.03it/s] 97%|█████████▋| 387951/400000 [00:43<00:01, 9335.28it/s] 97%|█████████▋| 388888/400000 [00:43<00:01, 9327.05it/s] 97%|█████████▋| 389863/400000 [00:43<00:01, 9447.93it/s] 98%|█████████▊| 390835/400000 [00:43<00:00, 9525.89it/s] 98%|█████████▊| 391789/400000 [00:43<00:00, 9510.95it/s] 98%|█████████▊| 392742/400000 [00:43<00:00, 9206.20it/s] 98%|█████████▊| 393666/400000 [00:43<00:00, 8470.70it/s] 99%|█████████▊| 394527/400000 [00:43<00:00, 8064.94it/s] 99%|█████████▉| 395368/400000 [00:43<00:00, 8163.67it/s] 99%|█████████▉| 396195/400000 [00:44<00:00, 8072.32it/s] 99%|█████████▉| 397171/400000 [00:44<00:00, 8512.73it/s]100%|█████████▉| 398101/400000 [00:44<00:00, 8734.05it/s]100%|█████████▉| 399058/400000 [00:44<00:00, 8967.52it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8999.69it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7ff8ae94a748>, <torchtext.data.dataset.TabularDataset object at 0x7ff8ae94a898>, <torchtext.vocab.Vocab object at 0x7ff8ae94a7b8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 14.21 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 26.64 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.17 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.17 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.03 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.03 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.28 file/s]2020-06-23 00:17:48.060637: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-23 00:17:48.065217: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-23 00:17:48.065429: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559f0c005e00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-23 00:17:48.065447: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▌      | 3530752/9912422 [00:00<00:00, 35046271.24it/s]9920512it [00:00, 35817458.65it/s]                             
0it [00:00, ?it/s]32768it [00:00, 675591.48it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 472864.30it/s]1654784it [00:00, 12722423.32it/s]                         
0it [00:00, ?it/s]8192it [00:00, 168251.12it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

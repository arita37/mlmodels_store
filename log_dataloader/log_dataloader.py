
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f709be18ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f709be18ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f71070d81e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f71070d81e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f712541fe18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f712541fe18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f70b43ff488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f70b43ff488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f70b43ff488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:17, 127644.06it/s] 79%|███████▉  | 7823360/9912422 [00:00<00:11, 182220.52it/s]9920512it [00:00, 39110777.38it/s]                           
0it [00:00, ?it/s]32768it [00:00, 680158.53it/s]
0it [00:00, ?it/s]  6%|▌         | 98304/1648877 [00:00<00:01, 982804.28it/s]1654784it [00:00, 12707236.48it/s]                         
0it [00:00, ?it/s]8192it [00:00, 136615.44it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f709b4413c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f709b44b668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f70b43ff0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f70b43ff0d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f70b43ff0d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:18,  2.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:18,  2.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:17,  2.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:17,  2.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:16,  2.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:16,  2.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:15,  2.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:15,  2.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:14,  2.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:14,  2.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:00<00:52,  2.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:52,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:51,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:51,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:51,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:50,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:50,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:50,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:49,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:49,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:49,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:48,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:34,  4.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:34,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:34,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:33,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:33,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:33,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:33,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:32,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:32,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:32,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:32,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:31,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:22,  5.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:22,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:22,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:22,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:21,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:21,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:21,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:21,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:21,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:21,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:20,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:20,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:00<00:14,  8.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:14,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:14,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:14,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:14,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:14,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:14,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:13,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:13,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:13,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:13,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:13,  8.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 11.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:09, 11.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:09, 11.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:09, 11.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:09, 11.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:08, 11.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:08, 11.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:08, 11.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:06, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:06, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:06, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 20.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 27.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 34.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 34.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 34.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 34.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 34.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 34.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 34.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 34.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 34.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 34.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 34.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 34.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 43.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 43.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 43.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 43.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 43.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 43.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 43.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 43.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:01, 43.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:01, 43.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:01, 43.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 43.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 53.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 53.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 53.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 53.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 53.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 53.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 53.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 53.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 53.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 53.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 53.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 53.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 62.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 62.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 62.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 62.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 62.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 62.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 62.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 62.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 62.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 62.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 62.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 62.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 71.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 71.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 71.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 71.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 71.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 71.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 71.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 71.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 71.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 71.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 71.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 71.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 78.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 78.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:01<00:00, 78.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:01<00:00, 78.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:01<00:00, 78.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:01<00:00, 78.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.04s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.04s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.04s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.90 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.09s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.04s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 78.90 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.09s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.09s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 39.57 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.10s/ url]
0 examples [00:00, ? examples/s]2020-06-28 18:09:39.742147: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-28 18:09:39.759630: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-28 18:09:39.760683: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c57f0745a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-28 18:09:39.760708: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.09 examples/s]110 examples [00:00,  2.99 examples/s]219 examples [00:00,  4.27 examples/s]330 examples [00:00,  6.08 examples/s]445 examples [00:00,  8.67 examples/s]541 examples [00:00, 12.34 examples/s]638 examples [00:01, 17.53 examples/s]748 examples [00:01, 24.87 examples/s]859 examples [00:01, 35.19 examples/s]971 examples [00:01, 49.60 examples/s]1075 examples [00:01, 69.42 examples/s]1183 examples [00:01, 96.50 examples/s]1288 examples [00:01, 132.51 examples/s]1392 examples [00:01, 178.79 examples/s]1494 examples [00:01, 236.82 examples/s]1601 examples [00:01, 308.96 examples/s]1707 examples [00:02, 392.13 examples/s]1813 examples [00:02, 483.52 examples/s]1926 examples [00:02, 582.91 examples/s]2033 examples [00:02, 670.87 examples/s]2142 examples [00:02, 757.25 examples/s]2250 examples [00:02, 831.70 examples/s]2358 examples [00:02, 892.52 examples/s]2466 examples [00:02, 928.10 examples/s]2573 examples [00:02, 966.11 examples/s]2680 examples [00:03, 992.80 examples/s]2789 examples [00:03, 1018.02 examples/s]2896 examples [00:03, 1024.92 examples/s]3006 examples [00:03, 1044.90 examples/s]3113 examples [00:03, 1033.39 examples/s]3219 examples [00:03, 1019.24 examples/s]3325 examples [00:03, 1030.51 examples/s]3429 examples [00:03, 1022.01 examples/s]3538 examples [00:03, 1041.18 examples/s]3648 examples [00:03, 1056.44 examples/s]3757 examples [00:04, 1065.72 examples/s]3864 examples [00:04, 1036.76 examples/s]3975 examples [00:04, 1055.86 examples/s]4084 examples [00:04, 1064.41 examples/s]4197 examples [00:04, 1081.60 examples/s]4306 examples [00:04, 1074.57 examples/s]4417 examples [00:04, 1082.62 examples/s]4526 examples [00:04, 1077.67 examples/s]4637 examples [00:04, 1086.39 examples/s]4749 examples [00:04, 1094.28 examples/s]4859 examples [00:05, 1070.29 examples/s]4967 examples [00:05, 1068.88 examples/s]5075 examples [00:05, 1005.74 examples/s]5180 examples [00:05, 1017.92 examples/s]5283 examples [00:05, 1020.41 examples/s]5390 examples [00:05, 1034.69 examples/s]5495 examples [00:05, 1036.88 examples/s]5601 examples [00:05, 1030.37 examples/s]5705 examples [00:05, 1025.74 examples/s]5810 examples [00:05, 1032.70 examples/s]5914 examples [00:06, 1033.75 examples/s]6023 examples [00:06, 1049.04 examples/s]6130 examples [00:06, 1053.49 examples/s]6241 examples [00:06, 1068.82 examples/s]6354 examples [00:06, 1083.78 examples/s]6463 examples [00:06, 1072.09 examples/s]6576 examples [00:06, 1086.53 examples/s]6685 examples [00:06, 1078.71 examples/s]6793 examples [00:06, 1037.86 examples/s]6898 examples [00:07, 1036.47 examples/s]7004 examples [00:07, 1042.36 examples/s]7109 examples [00:07, 1038.34 examples/s]7213 examples [00:07, 1037.07 examples/s]7320 examples [00:07, 1044.24 examples/s]7425 examples [00:07, 1011.29 examples/s]7527 examples [00:07, 1000.67 examples/s]7632 examples [00:07, 1013.52 examples/s]7734 examples [00:07, 999.48 examples/s] 7837 examples [00:07, 1005.98 examples/s]7944 examples [00:08, 1022.01 examples/s]8051 examples [00:08, 1033.58 examples/s]8160 examples [00:08, 1048.41 examples/s]8267 examples [00:08, 1053.53 examples/s]8378 examples [00:08, 1068.23 examples/s]8485 examples [00:08, 1053.20 examples/s]8591 examples [00:08, 1018.11 examples/s]8694 examples [00:08, 1013.22 examples/s]8796 examples [00:08, 1014.14 examples/s]8903 examples [00:08, 1028.28 examples/s]9007 examples [00:09, 1031.76 examples/s]9112 examples [00:09, 1032.54 examples/s]9218 examples [00:09, 1039.17 examples/s]9327 examples [00:09, 1051.19 examples/s]9439 examples [00:09, 1070.10 examples/s]9547 examples [00:09, 1058.54 examples/s]9654 examples [00:09, 1061.73 examples/s]9765 examples [00:09, 1073.27 examples/s]9873 examples [00:09, 1074.28 examples/s]9982 examples [00:09, 1078.66 examples/s]10090 examples [00:10, 1016.82 examples/s]10193 examples [00:10, 1019.68 examples/s]10301 examples [00:10, 1035.64 examples/s]10406 examples [00:10, 1018.48 examples/s]10509 examples [00:10, 995.55 examples/s] 10616 examples [00:10, 1014.90 examples/s]10725 examples [00:10, 1034.05 examples/s]10832 examples [00:10, 1044.03 examples/s]10939 examples [00:10, 1049.68 examples/s]11046 examples [00:11, 1054.26 examples/s]11152 examples [00:11, 1035.25 examples/s]11258 examples [00:11, 1041.87 examples/s]11369 examples [00:11, 1060.66 examples/s]11482 examples [00:11, 1078.38 examples/s]11595 examples [00:11, 1090.95 examples/s]11705 examples [00:11, 1074.43 examples/s]11815 examples [00:11, 1081.96 examples/s]11927 examples [00:11, 1091.21 examples/s]12037 examples [00:11, 1082.09 examples/s]12146 examples [00:12, 1068.98 examples/s]12254 examples [00:12, 1063.19 examples/s]12363 examples [00:12, 1070.39 examples/s]12471 examples [00:12, 1067.73 examples/s]12580 examples [00:12, 1072.23 examples/s]12688 examples [00:12, 1065.81 examples/s]12795 examples [00:12, 1059.76 examples/s]12902 examples [00:12, 1053.78 examples/s]13013 examples [00:12, 1068.33 examples/s]13120 examples [00:12, 1067.02 examples/s]13228 examples [00:13, 1070.68 examples/s]13336 examples [00:13, 1069.10 examples/s]13444 examples [00:13, 1071.23 examples/s]13553 examples [00:13, 1074.91 examples/s]13665 examples [00:13, 1087.76 examples/s]13774 examples [00:13, 1040.13 examples/s]13879 examples [00:13, 1041.60 examples/s]13984 examples [00:13, 1024.52 examples/s]14094 examples [00:13, 1044.58 examples/s]14203 examples [00:13, 1056.99 examples/s]14312 examples [00:14, 1064.65 examples/s]14420 examples [00:14, 1068.43 examples/s]14527 examples [00:14, 1050.69 examples/s]14633 examples [00:14, 1047.36 examples/s]14738 examples [00:14, 1046.88 examples/s]14846 examples [00:14, 1056.27 examples/s]14954 examples [00:14, 1061.94 examples/s]15061 examples [00:14, 1058.67 examples/s]15167 examples [00:14, 1046.58 examples/s]15272 examples [00:15, 1019.70 examples/s]15379 examples [00:15, 1032.20 examples/s]15487 examples [00:15, 1042.81 examples/s]15596 examples [00:15, 1054.71 examples/s]15703 examples [00:15, 1058.79 examples/s]15809 examples [00:15, 1021.55 examples/s]15912 examples [00:15, 1015.07 examples/s]16014 examples [00:15, 1011.29 examples/s]16117 examples [00:15, 1015.91 examples/s]16221 examples [00:15, 1021.67 examples/s]16331 examples [00:16, 1043.56 examples/s]16436 examples [00:16, 1042.78 examples/s]16545 examples [00:16, 1053.41 examples/s]16655 examples [00:16, 1064.17 examples/s]16766 examples [00:16, 1077.02 examples/s]16877 examples [00:16, 1084.35 examples/s]16986 examples [00:16, 1052.95 examples/s]17092 examples [00:16, 1052.11 examples/s]17198 examples [00:16, 1053.74 examples/s]17308 examples [00:16, 1065.69 examples/s]17416 examples [00:17, 1066.92 examples/s]17524 examples [00:17, 1068.34 examples/s]17631 examples [00:17, 1055.29 examples/s]17737 examples [00:17, 1041.84 examples/s]17842 examples [00:17, 1044.00 examples/s]17949 examples [00:17, 1049.89 examples/s]18058 examples [00:17, 1060.91 examples/s]18165 examples [00:17, 1059.87 examples/s]18274 examples [00:17, 1066.32 examples/s]18385 examples [00:17, 1078.86 examples/s]18495 examples [00:18, 1082.90 examples/s]18605 examples [00:18, 1087.43 examples/s]18714 examples [00:18, 1075.37 examples/s]18822 examples [00:18, 1074.08 examples/s]18930 examples [00:18, 1057.95 examples/s]19036 examples [00:18, 1058.33 examples/s]19142 examples [00:18, 1051.33 examples/s]19248 examples [00:18, 1032.85 examples/s]19352 examples [00:18, 1034.48 examples/s]19457 examples [00:18, 1039.03 examples/s]19566 examples [00:19, 1051.53 examples/s]19674 examples [00:19, 1057.24 examples/s]19780 examples [00:19, 1050.97 examples/s]19887 examples [00:19, 1054.97 examples/s]19994 examples [00:19, 1058.33 examples/s]20100 examples [00:19, 990.48 examples/s] 20200 examples [00:19, 986.43 examples/s]20308 examples [00:19, 1010.58 examples/s]20415 examples [00:19, 1026.43 examples/s]20525 examples [00:20, 1046.13 examples/s]20635 examples [00:20, 1059.35 examples/s]20742 examples [00:20, 1054.21 examples/s]20848 examples [00:20, 1050.36 examples/s]20954 examples [00:20, 1032.29 examples/s]21059 examples [00:20, 1034.89 examples/s]21165 examples [00:20, 1039.51 examples/s]21270 examples [00:20, 1016.29 examples/s]21372 examples [00:20, 1013.72 examples/s]21478 examples [00:20, 1025.88 examples/s]21584 examples [00:21, 1034.33 examples/s]21691 examples [00:21, 1042.68 examples/s]21797 examples [00:21, 1045.18 examples/s]21905 examples [00:21, 1054.39 examples/s]22011 examples [00:21, 1044.36 examples/s]22121 examples [00:21, 1057.92 examples/s]22232 examples [00:21, 1071.50 examples/s]22341 examples [00:21, 1076.28 examples/s]22451 examples [00:21, 1082.41 examples/s]22560 examples [00:21, 1076.27 examples/s]22668 examples [00:22, 1031.48 examples/s]22772 examples [00:22, 1023.54 examples/s]22884 examples [00:22, 1048.15 examples/s]22990 examples [00:22, 1027.13 examples/s]23094 examples [00:22, 1019.11 examples/s]23206 examples [00:22, 1046.32 examples/s]23317 examples [00:22, 1062.15 examples/s]23424 examples [00:22, 1027.75 examples/s]23528 examples [00:22, 1007.90 examples/s]23636 examples [00:22, 1026.92 examples/s]23747 examples [00:23, 1048.43 examples/s]23855 examples [00:23, 1057.09 examples/s]23966 examples [00:23, 1070.65 examples/s]24074 examples [00:23, 1070.05 examples/s]24182 examples [00:23, 1061.53 examples/s]24289 examples [00:23, 1054.65 examples/s]24398 examples [00:23, 1063.15 examples/s]24505 examples [00:23, 1054.76 examples/s]24614 examples [00:23, 1063.88 examples/s]24721 examples [00:24, 1063.08 examples/s]24828 examples [00:24, 1040.31 examples/s]24941 examples [00:24, 1065.24 examples/s]25052 examples [00:24, 1077.57 examples/s]25160 examples [00:24, 1073.02 examples/s]25269 examples [00:24, 1076.05 examples/s]25381 examples [00:24, 1087.73 examples/s]25490 examples [00:24, 1086.49 examples/s]25604 examples [00:24, 1101.07 examples/s]25715 examples [00:24, 1082.14 examples/s]25824 examples [00:25, 1072.11 examples/s]25932 examples [00:25, 1067.83 examples/s]26042 examples [00:25, 1075.54 examples/s]26150 examples [00:25, 1073.52 examples/s]26258 examples [00:25, 1068.27 examples/s]26365 examples [00:25, 1029.97 examples/s]26469 examples [00:25, 1006.86 examples/s]26571 examples [00:25, 1001.41 examples/s]26672 examples [00:25, 988.56 examples/s] 26772 examples [00:25, 966.40 examples/s]26869 examples [00:26, 964.37 examples/s]26976 examples [00:26, 992.41 examples/s]27086 examples [00:26, 1021.02 examples/s]27190 examples [00:26, 1024.50 examples/s]27293 examples [00:26, 1026.10 examples/s]27400 examples [00:26, 1036.34 examples/s]27510 examples [00:26, 1054.49 examples/s]27616 examples [00:26, 1056.12 examples/s]27722 examples [00:26, 1033.47 examples/s]27826 examples [00:26, 1015.28 examples/s]27935 examples [00:27, 1035.02 examples/s]28039 examples [00:27, 1028.11 examples/s]28151 examples [00:27, 1053.32 examples/s]28257 examples [00:27, 1024.44 examples/s]28360 examples [00:27, 984.37 examples/s] 28466 examples [00:27, 1003.33 examples/s]28576 examples [00:27, 1029.56 examples/s]28686 examples [00:27, 1048.15 examples/s]28796 examples [00:27, 1061.00 examples/s]28903 examples [00:28, 1051.22 examples/s]29009 examples [00:28, 1048.69 examples/s]29117 examples [00:28, 1057.22 examples/s]29228 examples [00:28, 1069.77 examples/s]29337 examples [00:28, 1075.02 examples/s]29445 examples [00:28, 1070.17 examples/s]29558 examples [00:28, 1086.53 examples/s]29667 examples [00:28, 1087.00 examples/s]29776 examples [00:28, 1056.50 examples/s]29882 examples [00:28, 1038.48 examples/s]29987 examples [00:29, 1037.90 examples/s]30091 examples [00:29, 975.77 examples/s] 30200 examples [00:29, 1005.87 examples/s]30309 examples [00:29, 1029.01 examples/s]30416 examples [00:29, 1038.98 examples/s]30521 examples [00:29, 1034.62 examples/s]30628 examples [00:29, 1043.62 examples/s]30733 examples [00:29, 1011.69 examples/s]30844 examples [00:29, 1038.76 examples/s]30954 examples [00:29, 1054.09 examples/s]31060 examples [00:30, 1021.21 examples/s]31170 examples [00:30, 1042.71 examples/s]31275 examples [00:30, 1033.94 examples/s]31384 examples [00:30, 1049.99 examples/s]31494 examples [00:30, 1063.67 examples/s]31601 examples [00:30, 1062.46 examples/s]31708 examples [00:30, 1061.20 examples/s]31815 examples [00:30, 1042.12 examples/s]31922 examples [00:30, 1050.27 examples/s]32028 examples [00:31, 1040.16 examples/s]32133 examples [00:31, 1029.37 examples/s]32244 examples [00:31, 1051.84 examples/s]32354 examples [00:31, 1065.44 examples/s]32464 examples [00:31, 1075.57 examples/s]32572 examples [00:31, 1076.37 examples/s]32680 examples [00:31, 1075.18 examples/s]32794 examples [00:31, 1091.16 examples/s]32904 examples [00:31, 1061.47 examples/s]33011 examples [00:31, 1051.47 examples/s]33117 examples [00:32, 1053.22 examples/s]33223 examples [00:32, 1048.06 examples/s]33336 examples [00:32, 1068.76 examples/s]33448 examples [00:32, 1082.25 examples/s]33557 examples [00:32, 1069.22 examples/s]33665 examples [00:32, 1047.00 examples/s]33770 examples [00:32, 1010.34 examples/s]33876 examples [00:32, 1023.14 examples/s]33983 examples [00:32, 1035.29 examples/s]34091 examples [00:32, 1047.20 examples/s]34199 examples [00:33, 1056.77 examples/s]34305 examples [00:33, 1048.39 examples/s]34418 examples [00:33, 1069.24 examples/s]34526 examples [00:33, 1062.01 examples/s]34635 examples [00:33, 1068.56 examples/s]34746 examples [00:33, 1079.88 examples/s]34855 examples [00:33, 1060.55 examples/s]34963 examples [00:33, 1063.97 examples/s]35070 examples [00:33, 1049.86 examples/s]35176 examples [00:33, 1047.11 examples/s]35281 examples [00:34, 1013.50 examples/s]35383 examples [00:34, 956.89 examples/s] 35482 examples [00:34, 965.16 examples/s]35591 examples [00:34, 999.09 examples/s]35703 examples [00:34, 1030.33 examples/s]35813 examples [00:34, 1048.32 examples/s]35919 examples [00:34, 1049.21 examples/s]36027 examples [00:34, 1058.16 examples/s]36135 examples [00:34, 1064.44 examples/s]36242 examples [00:35, 1025.46 examples/s]36346 examples [00:35, 1028.77 examples/s]36451 examples [00:35, 1032.97 examples/s]36557 examples [00:35, 1040.83 examples/s]36662 examples [00:35, 1039.45 examples/s]36767 examples [00:35, 1042.08 examples/s]36872 examples [00:35, 1031.55 examples/s]36978 examples [00:35, 1037.31 examples/s]37087 examples [00:35, 1049.84 examples/s]37193 examples [00:35, 1039.99 examples/s]37304 examples [00:36, 1058.50 examples/s]37411 examples [00:36, 1060.19 examples/s]37518 examples [00:36, 1061.88 examples/s]37628 examples [00:36, 1070.99 examples/s]37739 examples [00:36, 1080.32 examples/s]37849 examples [00:36, 1084.48 examples/s]37958 examples [00:36, 1066.24 examples/s]38065 examples [00:36, 1060.16 examples/s]38176 examples [00:36, 1074.26 examples/s]38286 examples [00:36, 1081.80 examples/s]38395 examples [00:37, 1080.03 examples/s]38504 examples [00:37, 1071.04 examples/s]38618 examples [00:37, 1088.05 examples/s]38728 examples [00:37, 1088.04 examples/s]38840 examples [00:37, 1094.95 examples/s]38950 examples [00:37, 1084.00 examples/s]39059 examples [00:37, 1063.17 examples/s]39167 examples [00:37, 1064.30 examples/s]39277 examples [00:37, 1072.43 examples/s]39386 examples [00:37, 1074.76 examples/s]39494 examples [00:38, 1041.49 examples/s]39601 examples [00:38, 1049.17 examples/s]39710 examples [00:38, 1060.44 examples/s]39821 examples [00:38, 1072.70 examples/s]39930 examples [00:38, 1076.65 examples/s]40038 examples [00:38, 1031.16 examples/s]40150 examples [00:38, 1053.76 examples/s]40264 examples [00:38, 1076.00 examples/s]40373 examples [00:38, 1047.41 examples/s]40479 examples [00:39, 1048.63 examples/s]40585 examples [00:39, 1042.01 examples/s]40690 examples [00:39, 1037.35 examples/s]40798 examples [00:39, 1048.70 examples/s]40906 examples [00:39, 1056.96 examples/s]41015 examples [00:39, 1065.60 examples/s]41122 examples [00:39, 1042.66 examples/s]41227 examples [00:39, 1042.44 examples/s]41338 examples [00:39, 1060.37 examples/s]41448 examples [00:39, 1070.04 examples/s]41556 examples [00:40, 1071.19 examples/s]41664 examples [00:40, 1062.38 examples/s]41771 examples [00:40, 1041.39 examples/s]41882 examples [00:40, 1060.42 examples/s]41990 examples [00:40, 1065.49 examples/s]42099 examples [00:40, 1069.99 examples/s]42207 examples [00:40, 1070.22 examples/s]42315 examples [00:40, 1059.75 examples/s]42425 examples [00:40, 1069.72 examples/s]42533 examples [00:40, 1056.58 examples/s]42639 examples [00:41, 1042.20 examples/s]42744 examples [00:41, 1007.55 examples/s]42851 examples [00:41, 1025.20 examples/s]42963 examples [00:41, 1051.50 examples/s]43074 examples [00:41, 1066.47 examples/s]43181 examples [00:41, 1054.86 examples/s]43293 examples [00:41, 1072.76 examples/s]43401 examples [00:41, 1063.85 examples/s]43510 examples [00:41, 1070.16 examples/s]43618 examples [00:42, 1037.95 examples/s]43724 examples [00:42, 1042.80 examples/s]43835 examples [00:42, 1061.33 examples/s]43945 examples [00:42, 1071.43 examples/s]44055 examples [00:42, 1077.95 examples/s]44164 examples [00:42, 1079.52 examples/s]44275 examples [00:42, 1086.72 examples/s]44384 examples [00:42, 1065.91 examples/s]44491 examples [00:42, 1007.73 examples/s]44600 examples [00:42, 1028.93 examples/s]44706 examples [00:43, 1036.90 examples/s]44814 examples [00:43, 1048.52 examples/s]44926 examples [00:43, 1067.77 examples/s]45036 examples [00:43, 1077.05 examples/s]45149 examples [00:43, 1090.01 examples/s]45262 examples [00:43, 1101.68 examples/s]45375 examples [00:43, 1107.97 examples/s]45486 examples [00:43, 1095.72 examples/s]45596 examples [00:43, 1071.17 examples/s]45705 examples [00:43, 1074.87 examples/s]45813 examples [00:44, 1071.03 examples/s]45924 examples [00:44, 1081.37 examples/s]46033 examples [00:44, 1045.91 examples/s]46138 examples [00:44, 1046.14 examples/s]46243 examples [00:44, 1041.82 examples/s]46348 examples [00:44, 1034.36 examples/s]46456 examples [00:44, 1045.62 examples/s]46562 examples [00:44, 1047.74 examples/s]46667 examples [00:44, 981.69 examples/s] 46773 examples [00:44, 1001.97 examples/s]46879 examples [00:45, 1017.79 examples/s]46982 examples [00:45, 965.17 examples/s] 47086 examples [00:45, 984.32 examples/s]47192 examples [00:45, 1004.59 examples/s]47303 examples [00:45, 1033.04 examples/s]47411 examples [00:45, 1046.61 examples/s]47518 examples [00:45, 1051.31 examples/s]47624 examples [00:45, 1012.36 examples/s]47726 examples [00:45, 1013.32 examples/s]47834 examples [00:46, 1029.79 examples/s]47938 examples [00:46, 988.62 examples/s] 48039 examples [00:46, 994.38 examples/s]48146 examples [00:46, 1014.16 examples/s]48258 examples [00:46, 1042.74 examples/s]48371 examples [00:46, 1064.61 examples/s]48479 examples [00:46, 1066.88 examples/s]48587 examples [00:46, 1063.78 examples/s]48694 examples [00:46, 1061.32 examples/s]48805 examples [00:46, 1075.01 examples/s]48913 examples [00:47, 1070.34 examples/s]49022 examples [00:47, 1073.84 examples/s]49130 examples [00:47, 1067.82 examples/s]49237 examples [00:47, 1046.20 examples/s]49347 examples [00:47, 1060.49 examples/s]49459 examples [00:47, 1075.71 examples/s]49567 examples [00:47, 1075.03 examples/s]49675 examples [00:47, 1028.42 examples/s]49779 examples [00:47, 1016.24 examples/s]49882 examples [00:47, 1009.83 examples/s]49989 examples [00:48, 1025.70 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7224/50000 [00:00<00:00, 72236.80 examples/s] 40%|████      | 20141/50000 [00:00<00:00, 83243.51 examples/s] 65%|██████▌   | 32618/50000 [00:00<00:00, 92476.92 examples/s] 92%|█████████▏| 46069/50000 [00:00<00:00, 102041.98 examples/s]                                                                0 examples [00:00, ? examples/s]89 examples [00:00, 886.96 examples/s]198 examples [00:00, 938.65 examples/s]308 examples [00:00, 974.96 examples/s]413 examples [00:00, 995.44 examples/s]519 examples [00:00, 1012.45 examples/s]621 examples [00:00, 1013.65 examples/s]729 examples [00:00, 1031.98 examples/s]839 examples [00:00, 1051.12 examples/s]947 examples [00:00, 1058.60 examples/s]1060 examples [00:01, 1077.92 examples/s]1169 examples [00:01, 1080.96 examples/s]1276 examples [00:01, 1052.65 examples/s]1383 examples [00:01, 1055.10 examples/s]1492 examples [00:01, 1063.03 examples/s]1598 examples [00:01, 1058.12 examples/s]1705 examples [00:01, 1060.99 examples/s]1814 examples [00:01, 1068.57 examples/s]1921 examples [00:01, 1032.78 examples/s]2029 examples [00:01, 1045.03 examples/s]2140 examples [00:02, 1063.67 examples/s]2252 examples [00:02, 1078.11 examples/s]2360 examples [00:02, 1068.64 examples/s]2469 examples [00:02, 1074.11 examples/s]2577 examples [00:02, 1073.93 examples/s]2685 examples [00:02, 1067.04 examples/s]2792 examples [00:02, 1066.50 examples/s]2899 examples [00:02, 1040.91 examples/s]3008 examples [00:02, 1053.46 examples/s]3114 examples [00:02, 1053.75 examples/s]3226 examples [00:03, 1071.67 examples/s]3336 examples [00:03, 1079.21 examples/s]3445 examples [00:03, 1067.69 examples/s]3555 examples [00:03, 1075.23 examples/s]3663 examples [00:03, 1067.34 examples/s]3773 examples [00:03, 1074.20 examples/s]3884 examples [00:03, 1084.55 examples/s]3993 examples [00:03, 1069.24 examples/s]4101 examples [00:03, 1063.28 examples/s]4210 examples [00:03, 1071.02 examples/s]4323 examples [00:04, 1086.35 examples/s]4434 examples [00:04, 1091.78 examples/s]4544 examples [00:04, 1087.55 examples/s]4653 examples [00:04, 1071.05 examples/s]4761 examples [00:04, 1056.02 examples/s]4872 examples [00:04, 1070.30 examples/s]4986 examples [00:04, 1087.81 examples/s]5099 examples [00:04, 1097.39 examples/s]5209 examples [00:04, 1080.76 examples/s]5318 examples [00:05, 1017.06 examples/s]5427 examples [00:05, 1037.77 examples/s]5532 examples [00:05, 1029.52 examples/s]5641 examples [00:05, 1041.74 examples/s]5746 examples [00:05, 1028.11 examples/s]5850 examples [00:05, 1030.85 examples/s]5957 examples [00:05, 1041.16 examples/s]6070 examples [00:05, 1064.77 examples/s]6178 examples [00:05, 1068.45 examples/s]6290 examples [00:05, 1083.33 examples/s]6399 examples [00:06, 1058.15 examples/s]6506 examples [00:06, 1038.01 examples/s]6619 examples [00:06, 1062.10 examples/s]6726 examples [00:06, 1064.19 examples/s]6836 examples [00:06, 1073.23 examples/s]6948 examples [00:06, 1085.26 examples/s]7058 examples [00:06, 1086.86 examples/s]7167 examples [00:06, 1080.08 examples/s]7276 examples [00:06, 1082.95 examples/s]7387 examples [00:06, 1090.36 examples/s]7499 examples [00:07, 1097.49 examples/s]7610 examples [00:07, 1100.58 examples/s]7721 examples [00:07, 1057.54 examples/s]7828 examples [00:07, 1054.64 examples/s]7934 examples [00:07, 1055.49 examples/s]8042 examples [00:07, 1061.66 examples/s]8149 examples [00:07, 1055.05 examples/s]8257 examples [00:07, 1060.02 examples/s]8369 examples [00:07, 1075.46 examples/s]8477 examples [00:07, 1034.47 examples/s]8588 examples [00:08, 1055.19 examples/s]8700 examples [00:08, 1072.18 examples/s]8808 examples [00:08, 1063.83 examples/s]8915 examples [00:08, 1042.70 examples/s]9022 examples [00:08, 1049.52 examples/s]9134 examples [00:08, 1067.25 examples/s]9246 examples [00:08, 1082.20 examples/s]9355 examples [00:08, 1035.77 examples/s]9462 examples [00:08, 1043.89 examples/s]9567 examples [00:09, 1043.85 examples/s]9676 examples [00:09, 1056.28 examples/s]9785 examples [00:09, 1065.83 examples/s]9892 examples [00:09, 1065.54 examples/s]9999 examples [00:09, 1039.23 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteTZ4R2V/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteTZ4R2V/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f70b43ff488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f70b43ff488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f70b43ff488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f709b5ac5f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f70b16292e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f70b43ff488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f70b43ff488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f70b43ff488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f709b44b438>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f709b44b0f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f70338f1158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f70338f1158> 

  function with postional parmater data_info <function split_train_valid at 0x7f70338f1158> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f70338f1268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f70338f1268> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f70338f1268> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=562c9e3febe48600e63022b0fe70c4db1c5f2faf3a1529f2d5902d9d216970d7
  Stored in directory: /tmp/pip-ephem-wheel-cache-2891d_7c/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:21:11, 13.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:04:58, 18.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:12:43, 26.0kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:27:26, 37.0kB/s].vector_cache/glove.6B.zip:   0%|          | 3.57M/862M [00:01<4:30:35, 52.9kB/s].vector_cache/glove.6B.zip:   1%|          | 9.02M/862M [00:01<3:08:17, 75.5kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:01<2:11:28, 108kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:01<1:31:29, 154kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.5M/862M [00:01<1:03:58, 219kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.0M/862M [00:01<44:34, 313kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.1M/862M [00:01<31:13, 445kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.2M/862M [00:01<21:51, 632kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.7M/862M [00:01<15:16, 899kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.4M/862M [00:02<10:45, 1.27MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.9M/862M [00:02<07:40, 1.77MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:02<05:27, 2.48MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.3M/862M [00:04<05:43, 2.35MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.5M/862M [00:04<06:45, 1.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<05:23, 2.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.4M/862M [00:06<05:56, 2.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<05:34, 2.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:06<04:11, 3.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.5M/862M [00:08<06:00, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:08<05:35, 2.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:08<04:11, 3.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:10<05:58, 2.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<06:52, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<05:27, 2.42MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:12<05:58, 2.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<05:32, 2.37MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:12<04:09, 3.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:14<05:56, 2.21MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:14<06:50, 1.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<05:27, 2.40MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:16<05:55, 2.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<05:28, 2.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:16<04:09, 3.13MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:18<05:56, 2.18MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:18<06:49, 1.90MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<05:26, 2.38MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.3M/862M [00:20<05:52, 2.19MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<05:27, 2.36MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:20<04:06, 3.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:21<05:52, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:22<06:44, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<05:16, 2.43MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:22<03:50, 3.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:23<10:00, 1.27MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<08:19, 1.53MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<06:08, 2.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<04:30, 2.82MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<23:38, 537kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<18:32, 684kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<13:27, 941kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<11:38, 1.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<10:08, 1.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:34, 1.66MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<07:32, 1.66MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<07:11, 1.74MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:31, 2.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<06:05, 2.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<06:14, 2.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<04:51, 2.57MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<05:36, 2.21MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<05:53, 2.10MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<04:32, 2.73MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<03:17, 3.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<1:47:09, 115kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<1:16:53, 161kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<54:13, 227kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<40:00, 307kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<29:56, 410kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<21:24, 573kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<17:06, 713kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<13:54, 878kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<10:08, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<09:15, 1.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<08:20, 1.46MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<06:14, 1.94MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<04:28, 2.69MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<37:19, 323kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<28:01, 431kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<20:03, 601kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<16:08, 744kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<13:11, 910kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<09:41, 1.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<08:53, 1.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<08:03, 1.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<06:02, 1.97MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<06:21, 1.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:19, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<04:52, 2.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:01, 2.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<7:46:17, 25.3kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<5:26:22, 36.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<3:49:33, 51.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<2:44:14, 71.4kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<1:55:41, 101kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<1:20:58, 144kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<1:00:19, 193kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<43:59, 265kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<31:08, 374kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:55<21:50, 531kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<46:36, 249kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<34:27, 336kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<24:32, 472kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<19:10, 601kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<15:15, 755kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<11:06, 1.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<09:48, 1.17MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<08:36, 1.33MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:28, 1.77MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:33, 1.74MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:23, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:54, 2.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<05:27, 2.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<05:36, 2.02MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<04:21, 2.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:03, 2.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:16, 2.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:03, 2.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<02:57, 3.78MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<35:03, 319kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<26:17, 425kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<18:48, 593kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<15:06, 736kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<12:19, 901kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<09:00, 1.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<08:16, 1.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:29, 1.47MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:36, 1.97MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<04:01, 2.73MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<1:37:20, 113kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<1:11:30, 153kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<50:53, 216kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<35:41, 306kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<28:28, 383kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<21:35, 505kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<15:31, 701kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<12:45, 850kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<10:37, 1.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:50, 1.38MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<07:24, 1.45MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<08:28, 1.27MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:41, 1.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:57, 2.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<05:41, 1.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<05:39, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:22, 2.44MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<04:57, 2.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:05, 2.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<03:58, 2.67MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:39, 2.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<04:56, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<03:51, 2.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:34, 2.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:51, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<03:47, 2.76MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<04:31, 2.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<04:48, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<03:46, 2.76MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:29, 2.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:43, 2.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<03:38, 2.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<02:44, 3.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:58, 1.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:48, 1.77MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<04:27, 2.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:56, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:01, 2.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:51, 2.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<02:48, 3.62MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<22:45, 446kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<17:31, 579kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<12:34, 805kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<08:54, 1.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<14:04, 716kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<11:27, 880kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:40<08:23, 1.20MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<07:38, 1.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<08:38, 1.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<06:49, 1.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<04:55, 2.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:39, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:14, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<04:42, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:44<03:23, 2.91MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<20:41, 477kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<16:03, 614kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<11:36, 848kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<08:35, 1.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<6:39:31, 24.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<4:40:11, 35.0kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<3:15:38, 49.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<2:19:46, 69.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<1:40:48, 96.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<1:11:09, 137kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<49:50, 195kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<37:18, 259kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<27:34, 351kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<19:39, 491kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<15:24, 623kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<13:42, 700kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<10:13, 938kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<07:18, 1.31MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<07:35, 1.26MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:46, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<05:03, 1.88MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:13, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:08, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<03:58, 2.38MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:26, 2.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:49, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:37, 2.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:23, 2.75MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<04:55, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:45, 1.96MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:39, 2.55MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:17, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:25, 2.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:24, 2.72MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<02:28, 3.71MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<13:37, 675kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<10:58, 837kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<08:01, 1.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<07:19, 1.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:33, 1.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<04:04, 2.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:02, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:54, 1.84MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:47, 2.39MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:15, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<04:12, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:14, 2.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:57, 2.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:10, 2.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:15, 2.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:51, 2.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<05:34, 1.59MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:29, 1.97MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<03:15, 2.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<05:02, 1.74MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:54, 1.79MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:46, 2.32MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:11, 2.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<05:36, 1.55MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:36, 1.89MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:22, 2.56MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:09, 1.67MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:59, 1.73MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:46, 2.28MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:10, 2.05MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:14, 2.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<03:18, 2.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:49, 2.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:01, 2.11MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:08, 2.69MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:41, 2.28MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:56, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<03:02, 2.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:37, 2.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:49, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:56, 2.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<02:10, 3.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<06:37, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<05:54, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:24, 1.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<03:09, 2.60MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<32:46, 251kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<24:13, 339kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<17:11, 477kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<12:03, 676kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<1:45:19, 77.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<1:14:56, 109kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<52:37, 155kB/s]  .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<36:46, 220kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<31:36, 256kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<24:37, 328kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<17:51, 452kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<12:35, 638kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<11:23, 703kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<09:12, 869kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<06:45, 1.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<06:07, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<05:15, 1.51MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<03:52, 2.04MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:19, 1.82MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:14, 1.86MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<03:16, 2.40MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<02:20, 3.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<5:12:20, 24.9kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<3:46:15, 34.4kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<2:40:08, 48.6kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<1:52:19, 69.2kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<1:18:26, 98.7kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<57:14, 135kB/s]   .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<41:15, 187kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<29:07, 264kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<21:38, 354kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<16:18, 469kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<11:38, 656kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<08:11, 926kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<16:52, 449kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<14:07, 536kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<10:22, 730kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 410M/862M [02:52<07:22, 1.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<07:02, 1.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<06:04, 1.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:32, 1.65MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:29, 1.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:19, 1.72MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:15, 2.28MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:21, 3.12MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<08:44, 844kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<07:16, 1.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:22, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<05:02, 1.45MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:39, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:31, 2.06MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:45, 1.93MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:46, 1.92MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:54, 2.48MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:18, 2.16MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:27, 2.07MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<02:38, 2.70MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<01:55, 3.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<13:41, 518kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<10:42, 663kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<07:42, 919kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<05:26, 1.29MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<12:55, 544kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<10:07, 694kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<07:20, 954kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<06:21, 1.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:33, 1.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:08, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:07, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:56, 1.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<02:59, 2.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:09, 3.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<08:10, 835kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<06:45, 1.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:59, 1.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<04:40, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<04:20, 1.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:17, 2.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:29, 1.92MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<03:28, 1.93MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<02:40, 2.48MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:02, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:10, 2.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:26, 2.71MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<01:46, 3.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<08:12, 798kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<06:46, 966kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<04:59, 1.31MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:37, 1.40MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<04:13, 1.53MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<03:12, 2.02MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:22, 1.90MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:20, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:34, 2.48MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:56, 2.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:02, 2.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:21, 2.69MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<01:43, 3.66MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<07:28, 839kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<06:13, 1.01MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<04:35, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:18, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:59, 1.55MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:59, 2.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:11, 1.92MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:10, 1.93MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:25, 2.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:00, 3.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<4:10:56, 24.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<2:55:48, 34.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<2:02:42, 49.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<1:26:49, 69.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<1:02:35, 95.9kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<44:13, 136kB/s]   .vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<30:52, 193kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<23:27, 253kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<17:18, 342kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<12:16, 482kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<08:38, 680kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<08:40, 676kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<07:52, 744kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:58, 981kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:41<04:15, 1.37MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:50, 1.20MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:13, 1.37MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:07, 1.85MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:14, 1.76MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:10, 1.80MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:24, 2.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<01:44, 3.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<20:58, 270kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<15:32, 364kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<11:41, 483kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<08:53, 628kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<07:04, 789kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<05:07, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<03:37, 1.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<13:28, 410kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<10:18, 535kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<07:24, 742kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<06:07, 890kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<05:07, 1.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:47, 1.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:35, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:21, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:33, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:43, 1.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:36, 2.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<01:59, 2.64MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:24, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:17, 1.59MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:38, 1.98MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<01:55, 2.69MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:45, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:45, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:05, 2.46MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:01<01:30, 3.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<51:17, 99.6kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<36:41, 139kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<25:47, 197kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<18:47, 268kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<13:55, 362kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<09:54, 506kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<07:45, 640kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<06:12, 799kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<04:29, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<03:10, 1.54MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<31:21, 156kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<22:42, 216kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<16:01, 304kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<11:58, 403kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<09:06, 529kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<06:32, 735kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<05:23, 883kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<04:32, 1.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:19, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<02:21, 1.99MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<05:52, 798kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<04:49, 971kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:33, 1.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:17, 1.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:43, 1.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:58, 1.55MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<02:08, 2.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:58, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:46, 1.64MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:06, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:16, 1.98MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:17, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<01:46, 2.51MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:01, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:45, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:12, 1.99MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:37, 2.70MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:14, 1.94MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:15, 1.93MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:44, 2.48MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:58, 2.17MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:02, 2.10MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:35, 2.69MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:51, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:57, 2.14MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:31, 2.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:48, 2.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:55, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:29, 2.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:46, 2.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:51, 2.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:25, 2.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:32<01:02, 3.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<04:27, 899kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<03:44, 1.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:44, 1.45MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<01:56, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<09:25, 417kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<07:12, 544kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<05:10, 755kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<04:16, 904kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:35, 1.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<02:39, 1.45MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<02:01, 1.87MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<2:38:37, 24.0kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<1:51:05, 34.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<1:17:07, 48.7kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<54:51, 68.0kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<39:35, 94.2kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<27:54, 133kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<19:26, 190kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<14:40, 249kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<10:51, 337kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<07:42, 473kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<05:58, 602kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<04:43, 760kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<03:25, 1.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<03:00, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:38, 1.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:58, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:58, 1.74MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:55, 1.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:28, 2.32MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:37, 2.08MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:39, 2.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:16, 2.65MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<00:55, 3.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<04:35, 723kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<03:44, 887kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:43, 1.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:27, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:14, 1.45MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:41, 1.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:43, 1.83MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:41, 1.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:18, 2.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:27, 2.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:30, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:10, 2.63MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:21, 2.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:25, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:06, 2.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:17, 2.29MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:21, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:04, 2.76MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:15, 2.31MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:20, 2.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:05, 2.64MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<00:49, 3.49MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:35, 1.79MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:51, 991kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:19, 1.22MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:41, 1.66MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:16, 2.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<00:56, 2.95MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<05:10, 535kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<05:07, 539kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:56, 701kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:49, 968kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<02:01, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:45, 977kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<03:15, 827kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:37, 1.03MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:53, 1.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:22, 1.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<02:03, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:36, 1.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:08, 1.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:33, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:07, 2.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<02:28, 1.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<02:53, 885kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:18, 1.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:40, 1.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<01:12, 2.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<03:08, 792kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<03:18, 753kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<02:35, 961kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:51, 1.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:20, 1.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<02:07, 1.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<02:28, 975kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:59, 1.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:27, 1.64MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<01:03, 2.24MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<03:09, 744kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<03:15, 720kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:31, 926kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:49, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:19, 1.74MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:47, 1.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<02:11, 1.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:45, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:17, 1.75MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:56, 2.38MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<03:11, 694kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<03:08, 706kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:24, 914kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:43, 1.27MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<01:13, 1.76MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<03:51, 555kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<03:34, 599kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<02:43, 786kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:56, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<01:22, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<04:01, 515kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<03:39, 566kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:46, 746kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:58, 1.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<01:24, 1.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<04:23, 457kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<03:53, 516kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<02:54, 685kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:03, 956kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<01:27, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<03:25, 566kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<03:11, 606kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<02:25, 796kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:43, 1.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<01:13, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<09:04, 206kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<07:06, 263kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<05:08, 361kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<03:36, 509kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<02:30, 717kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<05:04, 355kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<04:17, 419kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<03:10, 564kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<02:14, 788kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:34, 1.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<03:52, 446kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<03:25, 504kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<02:33, 671kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:49, 935kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:16, 1.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<02:49, 589kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<02:40, 622kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<02:01, 815kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:26, 1.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:05, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<58:46, 27.2kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<41:06, 38.7kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<28:26, 55.2kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<19:43, 78.7kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<14:07, 108kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<10:30, 145kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<07:27, 204kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<05:10, 289kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<03:37, 409kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<03:02, 479kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<02:43, 535kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<02:01, 716kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<01:25, 999kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<01:01, 1.38MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:13, 1.13MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:23, 1.00MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:05, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:46, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:34, 2.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:55, 1.42MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:11, 1.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:57, 1.37MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:41, 1.87MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:29, 2.54MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:16, 977kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:24, 887kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:06, 1.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:47, 1.55MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<00:33, 2.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:27, 815kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:27, 808kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:07, 1.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:48, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:47, 1.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:58, 1.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:45, 1.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:33, 1.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:24, 2.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:44, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:54, 1.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:43, 1.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:31, 1.92MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:34, 1.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:44, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:35, 1.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:25, 2.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:18, 2.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:41, 1.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:49, 1.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:39, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:27, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:30, 1.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:38, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:31, 1.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:22, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:25, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:33, 1.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:27, 1.65MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:19, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:23, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:29, 1.39MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:24, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:17, 2.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:20, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:26, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:20, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:14, 2.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:10, 3.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:40, 833kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:38, 873kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:28, 1.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:19, 1.59MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:21, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:23, 1.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:18, 1.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:08, 2.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:40, 619kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:35, 712kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:25, 948kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:17, 1.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:18, 1.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:18, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:13, 1.49MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:09, 2.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:10, 1.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:12, 1.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:09, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:05, 2.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:08, 1.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:09, 1.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:04, 1.78MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 1.88MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.63MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.79MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.33MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.32MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 826/400000 [00:00<00:48, 8257.65it/s]  0%|          | 1696/400000 [00:00<00:47, 8385.55it/s]  1%|          | 2605/400000 [00:00<00:46, 8583.85it/s]  1%|          | 3510/400000 [00:00<00:45, 8717.69it/s]  1%|          | 4389/400000 [00:00<00:45, 8737.48it/s]  1%|▏         | 5227/400000 [00:00<00:45, 8624.22it/s]  2%|▏         | 6001/400000 [00:00<00:47, 8336.22it/s]  2%|▏         | 6815/400000 [00:00<00:47, 8275.26it/s]  2%|▏         | 7598/400000 [00:00<00:48, 8118.28it/s]  2%|▏         | 8446/400000 [00:01<00:47, 8221.53it/s]  2%|▏         | 9304/400000 [00:01<00:46, 8323.23it/s]  3%|▎         | 10141/400000 [00:01<00:46, 8335.70it/s]  3%|▎         | 10989/400000 [00:01<00:46, 8378.12it/s]  3%|▎         | 11887/400000 [00:01<00:45, 8549.01it/s]  3%|▎         | 12743/400000 [00:01<00:45, 8552.12it/s]  3%|▎         | 13596/400000 [00:01<00:45, 8475.41it/s]  4%|▎         | 14506/400000 [00:01<00:44, 8650.37it/s]  4%|▍         | 15382/400000 [00:01<00:44, 8681.05it/s]  4%|▍         | 16267/400000 [00:01<00:43, 8730.38it/s]  4%|▍         | 17153/400000 [00:02<00:43, 8766.41it/s]  5%|▍         | 18030/400000 [00:02<00:43, 8716.51it/s]  5%|▍         | 18934/400000 [00:02<00:43, 8810.53it/s]  5%|▍         | 19816/400000 [00:02<00:43, 8791.21it/s]  5%|▌         | 20696/400000 [00:02<00:43, 8782.75it/s]  5%|▌         | 21575/400000 [00:02<00:44, 8560.76it/s]  6%|▌         | 22433/400000 [00:02<00:44, 8467.10it/s]  6%|▌         | 23318/400000 [00:02<00:43, 8577.01it/s]  6%|▌         | 24211/400000 [00:02<00:43, 8679.34it/s]  6%|▋         | 25109/400000 [00:02<00:42, 8766.82it/s]  6%|▋         | 25987/400000 [00:03<00:42, 8743.33it/s]  7%|▋         | 26863/400000 [00:03<00:43, 8614.13it/s]  7%|▋         | 27746/400000 [00:03<00:42, 8677.71it/s]  7%|▋         | 28641/400000 [00:03<00:42, 8757.48it/s]  7%|▋         | 29532/400000 [00:03<00:42, 8802.07it/s]  8%|▊         | 30457/400000 [00:03<00:41, 8930.24it/s]  8%|▊         | 31372/400000 [00:03<00:40, 8993.77it/s]  8%|▊         | 32273/400000 [00:03<00:40, 8973.87it/s]  8%|▊         | 33171/400000 [00:03<00:40, 8974.66it/s]  9%|▊         | 34069/400000 [00:03<00:41, 8903.03it/s]  9%|▊         | 34972/400000 [00:04<00:40, 8938.14it/s]  9%|▉         | 35867/400000 [00:04<00:40, 8894.66it/s]  9%|▉         | 36757/400000 [00:04<00:42, 8592.97it/s]  9%|▉         | 37679/400000 [00:04<00:41, 8770.96it/s] 10%|▉         | 38559/400000 [00:04<00:41, 8762.67it/s] 10%|▉         | 39444/400000 [00:04<00:41, 8787.98it/s] 10%|█         | 40325/400000 [00:04<00:41, 8760.19it/s] 10%|█         | 41211/400000 [00:04<00:40, 8786.98it/s] 11%|█         | 42106/400000 [00:04<00:40, 8832.62it/s] 11%|█         | 43013/400000 [00:04<00:40, 8901.20it/s] 11%|█         | 43904/400000 [00:05<00:40, 8837.25it/s] 11%|█         | 44789/400000 [00:05<00:41, 8628.95it/s] 11%|█▏        | 45687/400000 [00:05<00:40, 8730.84it/s] 12%|█▏        | 46634/400000 [00:05<00:39, 8939.67it/s] 12%|█▏        | 47531/400000 [00:05<00:40, 8802.86it/s] 12%|█▏        | 48414/400000 [00:05<00:40, 8716.70it/s] 12%|█▏        | 49288/400000 [00:05<00:40, 8613.78it/s] 13%|█▎        | 50169/400000 [00:05<00:40, 8670.81it/s] 13%|█▎        | 51038/400000 [00:05<00:40, 8550.77it/s] 13%|█▎        | 51931/400000 [00:05<00:40, 8661.04it/s] 13%|█▎        | 52799/400000 [00:06<00:40, 8599.57it/s] 13%|█▎        | 53660/400000 [00:06<00:40, 8512.02it/s] 14%|█▎        | 54530/400000 [00:06<00:40, 8567.24it/s] 14%|█▍        | 55388/400000 [00:06<00:40, 8567.03it/s] 14%|█▍        | 56261/400000 [00:06<00:39, 8614.84it/s] 14%|█▍        | 57153/400000 [00:06<00:39, 8703.14it/s] 15%|█▍        | 58041/400000 [00:06<00:39, 8754.85it/s] 15%|█▍        | 58960/400000 [00:06<00:38, 8879.70it/s] 15%|█▍        | 59849/400000 [00:06<00:38, 8758.08it/s] 15%|█▌        | 60726/400000 [00:06<00:38, 8740.42it/s] 15%|█▌        | 61633/400000 [00:07<00:38, 8834.33it/s] 16%|█▌        | 62518/400000 [00:07<00:38, 8724.37it/s] 16%|█▌        | 63412/400000 [00:07<00:38, 8787.27it/s] 16%|█▌        | 64342/400000 [00:07<00:37, 8932.29it/s] 16%|█▋        | 65239/400000 [00:07<00:37, 8943.47it/s] 17%|█▋        | 66135/400000 [00:07<00:37, 8905.17it/s] 17%|█▋        | 67030/400000 [00:07<00:37, 8917.75it/s] 17%|█▋        | 67923/400000 [00:07<00:37, 8772.22it/s] 17%|█▋        | 68846/400000 [00:07<00:37, 8904.11it/s] 17%|█▋        | 69738/400000 [00:08<00:38, 8678.38it/s] 18%|█▊        | 70608/400000 [00:08<00:37, 8677.01it/s] 18%|█▊        | 71478/400000 [00:08<00:37, 8659.98it/s] 18%|█▊        | 72371/400000 [00:08<00:37, 8738.17it/s] 18%|█▊        | 73281/400000 [00:08<00:36, 8841.86it/s] 19%|█▊        | 74193/400000 [00:08<00:36, 8922.27it/s] 19%|█▉        | 75097/400000 [00:08<00:36, 8953.26it/s] 19%|█▉        | 75993/400000 [00:08<00:36, 8905.74it/s] 19%|█▉        | 76885/400000 [00:08<00:36, 8883.31it/s] 19%|█▉        | 77774/400000 [00:08<00:36, 8729.93it/s] 20%|█▉        | 78648/400000 [00:09<00:36, 8707.19it/s] 20%|█▉        | 79529/400000 [00:09<00:36, 8737.65it/s] 20%|██        | 80410/400000 [00:09<00:36, 8756.43it/s] 20%|██        | 81357/400000 [00:09<00:35, 8957.76it/s] 21%|██        | 82302/400000 [00:09<00:34, 9098.23it/s] 21%|██        | 83256/400000 [00:09<00:34, 9226.36it/s] 21%|██        | 84181/400000 [00:09<00:34, 9229.56it/s] 21%|██▏       | 85105/400000 [00:09<00:35, 8898.78it/s] 21%|██▏       | 85999/400000 [00:09<00:36, 8630.52it/s] 22%|██▏       | 86894/400000 [00:09<00:35, 8722.60it/s] 22%|██▏       | 87770/400000 [00:10<00:35, 8684.94it/s] 22%|██▏       | 88641/400000 [00:10<00:35, 8670.34it/s] 22%|██▏       | 89522/400000 [00:10<00:35, 8711.52it/s] 23%|██▎       | 90434/400000 [00:10<00:35, 8829.53it/s] 23%|██▎       | 91321/400000 [00:10<00:34, 8841.21it/s] 23%|██▎       | 92239/400000 [00:10<00:34, 8938.67it/s] 23%|██▎       | 93134/400000 [00:10<00:34, 8772.92it/s] 24%|██▎       | 94014/400000 [00:10<00:34, 8779.64it/s] 24%|██▎       | 94893/400000 [00:10<00:36, 8400.00it/s] 24%|██▍       | 95766/400000 [00:10<00:35, 8493.77it/s] 24%|██▍       | 96652/400000 [00:11<00:35, 8599.29it/s] 24%|██▍       | 97515/400000 [00:11<00:35, 8505.11it/s] 25%|██▍       | 98368/400000 [00:11<00:35, 8505.96it/s] 25%|██▍       | 99241/400000 [00:11<00:35, 8570.91it/s] 25%|██▌       | 100129/400000 [00:11<00:34, 8660.08it/s] 25%|██▌       | 101025/400000 [00:11<00:34, 8746.46it/s] 25%|██▌       | 101901/400000 [00:11<00:34, 8551.90it/s] 26%|██▌       | 102810/400000 [00:11<00:34, 8706.48it/s] 26%|██▌       | 103720/400000 [00:11<00:33, 8820.07it/s] 26%|██▌       | 104613/400000 [00:11<00:33, 8851.78it/s] 26%|██▋       | 105511/400000 [00:12<00:33, 8887.62it/s] 27%|██▋       | 106401/400000 [00:12<00:33, 8781.47it/s] 27%|██▋       | 107301/400000 [00:12<00:33, 8845.22it/s] 27%|██▋       | 108228/400000 [00:12<00:32, 8968.27it/s] 27%|██▋       | 109178/400000 [00:12<00:31, 9120.94it/s] 28%|██▊       | 110092/400000 [00:12<00:32, 8991.43it/s] 28%|██▊       | 110993/400000 [00:12<00:32, 8996.25it/s] 28%|██▊       | 111894/400000 [00:12<00:32, 8825.34it/s] 28%|██▊       | 112818/400000 [00:12<00:32, 8942.94it/s] 28%|██▊       | 113714/400000 [00:12<00:32, 8923.39it/s] 29%|██▊       | 114608/400000 [00:13<00:32, 8824.13it/s] 29%|██▉       | 115492/400000 [00:13<00:32, 8779.50it/s] 29%|██▉       | 116372/400000 [00:13<00:32, 8784.41it/s] 29%|██▉       | 117258/400000 [00:13<00:32, 8806.82it/s] 30%|██▉       | 118167/400000 [00:13<00:31, 8888.12it/s] 30%|██▉       | 119099/400000 [00:13<00:31, 9010.90it/s] 30%|███       | 120028/400000 [00:13<00:30, 9091.84it/s] 30%|███       | 120938/400000 [00:13<00:30, 9041.30it/s] 30%|███       | 121844/400000 [00:13<00:30, 9044.42it/s] 31%|███       | 122749/400000 [00:14<00:30, 8950.14it/s] 31%|███       | 123645/400000 [00:14<00:31, 8735.59it/s] 31%|███       | 124536/400000 [00:14<00:31, 8785.77it/s] 31%|███▏      | 125416/400000 [00:14<00:31, 8742.45it/s] 32%|███▏      | 126314/400000 [00:14<00:31, 8811.95it/s] 32%|███▏      | 127223/400000 [00:14<00:30, 8890.26it/s] 32%|███▏      | 128126/400000 [00:14<00:30, 8929.35it/s] 32%|███▏      | 129052/400000 [00:14<00:30, 9022.61it/s] 32%|███▏      | 129955/400000 [00:14<00:30, 8874.40it/s] 33%|███▎      | 130844/400000 [00:14<00:30, 8853.10it/s] 33%|███▎      | 131730/400000 [00:15<00:30, 8762.82it/s] 33%|███▎      | 132620/400000 [00:15<00:30, 8801.85it/s] 33%|███▎      | 133501/400000 [00:15<00:30, 8790.33it/s] 34%|███▎      | 134383/400000 [00:15<00:30, 8797.78it/s] 34%|███▍      | 135285/400000 [00:15<00:29, 8860.69it/s] 34%|███▍      | 136172/400000 [00:15<00:30, 8772.58it/s] 34%|███▍      | 137073/400000 [00:15<00:29, 8839.89it/s] 34%|███▍      | 137958/400000 [00:15<00:29, 8794.65it/s] 35%|███▍      | 138838/400000 [00:15<00:29, 8770.02it/s] 35%|███▍      | 139719/400000 [00:15<00:29, 8779.35it/s] 35%|███▌      | 140598/400000 [00:16<00:29, 8685.68it/s] 35%|███▌      | 141500/400000 [00:16<00:29, 8782.58it/s] 36%|███▌      | 142388/400000 [00:16<00:29, 8808.88it/s] 36%|███▌      | 143300/400000 [00:16<00:28, 8897.04it/s] 36%|███▌      | 144191/400000 [00:16<00:28, 8890.82it/s] 36%|███▋      | 145081/400000 [00:16<00:28, 8814.99it/s] 36%|███▋      | 145967/400000 [00:16<00:28, 8826.66it/s] 37%|███▋      | 146876/400000 [00:16<00:28, 8903.16it/s] 37%|███▋      | 147767/400000 [00:16<00:28, 8855.68it/s] 37%|███▋      | 148653/400000 [00:16<00:28, 8738.52it/s] 37%|███▋      | 149530/400000 [00:17<00:28, 8746.11it/s] 38%|███▊      | 150431/400000 [00:17<00:28, 8823.55it/s] 38%|███▊      | 151336/400000 [00:17<00:27, 8888.16it/s] 38%|███▊      | 152226/400000 [00:17<00:28, 8763.29it/s] 38%|███▊      | 153141/400000 [00:17<00:27, 8873.79it/s] 39%|███▊      | 154030/400000 [00:17<00:27, 8829.77it/s] 39%|███▊      | 154925/400000 [00:17<00:27, 8864.16it/s] 39%|███▉      | 155854/400000 [00:17<00:27, 8987.12it/s] 39%|███▉      | 156756/400000 [00:17<00:27, 8995.44it/s] 39%|███▉      | 157658/400000 [00:17<00:26, 9002.65it/s] 40%|███▉      | 158559/400000 [00:18<00:26, 8964.01it/s] 40%|███▉      | 159456/400000 [00:18<00:27, 8888.49it/s] 40%|████      | 160346/400000 [00:18<00:27, 8870.44it/s] 40%|████      | 161234/400000 [00:18<00:27, 8819.76it/s] 41%|████      | 162117/400000 [00:18<00:27, 8740.95it/s] 41%|████      | 163008/400000 [00:18<00:26, 8789.47it/s] 41%|████      | 163888/400000 [00:18<00:26, 8767.33it/s] 41%|████      | 164765/400000 [00:18<00:27, 8652.85it/s] 41%|████▏     | 165631/400000 [00:18<00:27, 8561.21it/s] 42%|████▏     | 166488/400000 [00:18<00:27, 8515.26it/s] 42%|████▏     | 167352/400000 [00:19<00:27, 8551.95it/s] 42%|████▏     | 168239/400000 [00:19<00:26, 8642.43it/s] 42%|████▏     | 169138/400000 [00:19<00:26, 8741.07it/s] 43%|████▎     | 170013/400000 [00:19<00:27, 8338.50it/s] 43%|████▎     | 170872/400000 [00:19<00:27, 8411.32it/s] 43%|████▎     | 171717/400000 [00:19<00:27, 8419.95it/s] 43%|████▎     | 172562/400000 [00:19<00:27, 8381.15it/s] 43%|████▎     | 173402/400000 [00:19<00:27, 8287.02it/s] 44%|████▎     | 174233/400000 [00:19<00:27, 8238.96it/s] 44%|████▍     | 175058/400000 [00:20<00:27, 8215.78it/s] 44%|████▍     | 175893/400000 [00:20<00:27, 8255.03it/s] 44%|████▍     | 176781/400000 [00:20<00:26, 8430.47it/s] 44%|████▍     | 177724/400000 [00:20<00:25, 8705.50it/s] 45%|████▍     | 178598/400000 [00:20<00:26, 8272.83it/s] 45%|████▍     | 179433/400000 [00:20<00:26, 8260.38it/s] 45%|████▌     | 180274/400000 [00:20<00:26, 8302.41it/s] 45%|████▌     | 181138/400000 [00:20<00:26, 8398.74it/s] 46%|████▌     | 182021/400000 [00:20<00:25, 8523.02it/s] 46%|████▌     | 182876/400000 [00:20<00:26, 8330.63it/s] 46%|████▌     | 183729/400000 [00:21<00:25, 8387.07it/s] 46%|████▌     | 184570/400000 [00:21<00:25, 8334.07it/s] 46%|████▋     | 185466/400000 [00:21<00:25, 8511.32it/s] 47%|████▋     | 186356/400000 [00:21<00:24, 8624.19it/s] 47%|████▋     | 187235/400000 [00:21<00:24, 8671.30it/s] 47%|████▋     | 188125/400000 [00:21<00:24, 8738.08it/s] 47%|████▋     | 189014/400000 [00:21<00:24, 8782.78it/s] 47%|████▋     | 189894/400000 [00:21<00:23, 8756.67it/s] 48%|████▊     | 190813/400000 [00:21<00:23, 8879.13it/s] 48%|████▊     | 191702/400000 [00:21<00:23, 8696.13it/s] 48%|████▊     | 192576/400000 [00:22<00:23, 8708.96it/s] 48%|████▊     | 193450/400000 [00:22<00:23, 8715.86it/s] 49%|████▊     | 194352/400000 [00:22<00:23, 8804.20it/s] 49%|████▉     | 195283/400000 [00:22<00:22, 8949.45it/s] 49%|████▉     | 196180/400000 [00:22<00:23, 8820.28it/s] 49%|████▉     | 197109/400000 [00:22<00:22, 8953.58it/s] 50%|████▉     | 198006/400000 [00:22<00:22, 8895.03it/s] 50%|████▉     | 198938/400000 [00:22<00:22, 9017.90it/s] 50%|████▉     | 199848/400000 [00:22<00:22, 9040.68it/s] 50%|█████     | 200753/400000 [00:22<00:22, 8935.65it/s] 50%|█████     | 201648/400000 [00:23<00:22, 8885.82it/s] 51%|█████     | 202538/400000 [00:23<00:22, 8781.87it/s] 51%|█████     | 203424/400000 [00:23<00:22, 8803.97it/s] 51%|█████     | 204305/400000 [00:23<00:22, 8654.13it/s] 51%|█████▏    | 205172/400000 [00:23<00:22, 8630.09it/s] 52%|█████▏    | 206098/400000 [00:23<00:22, 8809.59it/s] 52%|█████▏    | 206990/400000 [00:23<00:21, 8841.97it/s] 52%|█████▏    | 207876/400000 [00:23<00:22, 8568.36it/s] 52%|█████▏    | 208796/400000 [00:23<00:21, 8746.92it/s] 52%|█████▏    | 209674/400000 [00:23<00:21, 8655.70it/s] 53%|█████▎    | 210543/400000 [00:24<00:21, 8663.71it/s] 53%|█████▎    | 211415/400000 [00:24<00:21, 8679.17it/s] 53%|█████▎    | 212343/400000 [00:24<00:21, 8849.46it/s] 53%|█████▎    | 213266/400000 [00:24<00:20, 8959.63it/s] 54%|█████▎    | 214164/400000 [00:24<00:20, 8955.92it/s] 54%|█████▍    | 215062/400000 [00:24<00:20, 8963.09it/s] 54%|█████▍    | 215960/400000 [00:24<00:20, 8939.72it/s] 54%|█████▍    | 216855/400000 [00:24<00:20, 8783.99it/s] 54%|█████▍    | 217735/400000 [00:24<00:21, 8591.23it/s] 55%|█████▍    | 218596/400000 [00:25<00:21, 8502.33it/s] 55%|█████▍    | 219448/400000 [00:25<00:21, 8505.19it/s] 55%|█████▌    | 220338/400000 [00:25<00:20, 8619.68it/s] 55%|█████▌    | 221245/400000 [00:25<00:20, 8747.71it/s] 56%|█████▌    | 222121/400000 [00:25<00:20, 8732.52it/s] 56%|█████▌    | 222996/400000 [00:25<00:20, 8551.77it/s] 56%|█████▌    | 223873/400000 [00:25<00:20, 8613.98it/s] 56%|█████▌    | 224737/400000 [00:25<00:20, 8620.08it/s] 56%|█████▋    | 225612/400000 [00:25<00:20, 8657.73it/s] 57%|█████▋    | 226479/400000 [00:25<00:20, 8634.97it/s] 57%|█████▋    | 227343/400000 [00:26<00:20, 8575.19it/s] 57%|█████▋    | 228205/400000 [00:26<00:20, 8587.15it/s] 57%|█████▋    | 229086/400000 [00:26<00:19, 8650.00it/s] 57%|█████▋    | 229952/400000 [00:26<00:19, 8620.54it/s] 58%|█████▊    | 230828/400000 [00:26<00:19, 8658.86it/s] 58%|█████▊    | 231696/400000 [00:26<00:19, 8659.31it/s] 58%|█████▊    | 232563/400000 [00:26<00:19, 8599.21it/s] 58%|█████▊    | 233424/400000 [00:26<00:19, 8574.80it/s] 59%|█████▊    | 234284/400000 [00:26<00:19, 8580.21it/s] 59%|█████▉    | 235143/400000 [00:26<00:19, 8514.08it/s] 59%|█████▉    | 235995/400000 [00:27<00:19, 8504.68it/s] 59%|█████▉    | 236858/400000 [00:27<00:19, 8540.02it/s] 59%|█████▉    | 237713/400000 [00:27<00:19, 8415.15it/s] 60%|█████▉    | 238606/400000 [00:27<00:18, 8561.86it/s] 60%|█████▉    | 239472/400000 [00:27<00:18, 8590.88it/s] 60%|██████    | 240334/400000 [00:27<00:18, 8598.96it/s] 60%|██████    | 241199/400000 [00:27<00:18, 8613.82it/s] 61%|██████    | 242061/400000 [00:27<00:18, 8597.81it/s] 61%|██████    | 242922/400000 [00:27<00:18, 8583.66it/s] 61%|██████    | 243792/400000 [00:27<00:18, 8616.10it/s] 61%|██████    | 244654/400000 [00:28<00:18, 8565.74it/s] 61%|██████▏   | 245511/400000 [00:28<00:18, 8546.57it/s] 62%|██████▏   | 246366/400000 [00:28<00:17, 8544.06it/s] 62%|██████▏   | 247228/400000 [00:28<00:17, 8563.75it/s] 62%|██████▏   | 248085/400000 [00:28<00:17, 8508.92it/s] 62%|██████▏   | 248974/400000 [00:28<00:17, 8618.98it/s] 62%|██████▏   | 249891/400000 [00:28<00:17, 8775.86it/s] 63%|██████▎   | 250788/400000 [00:28<00:16, 8832.35it/s] 63%|██████▎   | 251673/400000 [00:28<00:16, 8767.77it/s] 63%|██████▎   | 252572/400000 [00:28<00:16, 8833.20it/s] 63%|██████▎   | 253456/400000 [00:29<00:16, 8713.45it/s] 64%|██████▎   | 254335/400000 [00:29<00:16, 8734.77it/s] 64%|██████▍   | 255246/400000 [00:29<00:16, 8841.92it/s] 64%|██████▍   | 256131/400000 [00:29<00:16, 8664.69it/s] 64%|██████▍   | 256999/400000 [00:29<00:16, 8616.83it/s] 64%|██████▍   | 257885/400000 [00:29<00:16, 8687.42it/s] 65%|██████▍   | 258803/400000 [00:29<00:15, 8829.04it/s] 65%|██████▍   | 259745/400000 [00:29<00:15, 8997.03it/s] 65%|██████▌   | 260647/400000 [00:29<00:15, 8937.90it/s] 65%|██████▌   | 261542/400000 [00:29<00:15, 8916.92it/s] 66%|██████▌   | 262435/400000 [00:30<00:15, 8639.03it/s] 66%|██████▌   | 263302/400000 [00:30<00:16, 8464.45it/s] 66%|██████▌   | 264230/400000 [00:30<00:15, 8692.57it/s] 66%|██████▋   | 265127/400000 [00:30<00:15, 8773.60it/s] 67%|██████▋   | 266031/400000 [00:30<00:15, 8851.44it/s] 67%|██████▋   | 266919/400000 [00:30<00:15, 8765.97it/s] 67%|██████▋   | 267798/400000 [00:30<00:15, 8616.63it/s] 67%|██████▋   | 268662/400000 [00:30<00:15, 8429.53it/s] 67%|██████▋   | 269512/400000 [00:30<00:15, 8449.69it/s] 68%|██████▊   | 270359/400000 [00:30<00:15, 8417.93it/s] 68%|██████▊   | 271213/400000 [00:31<00:15, 8451.78it/s] 68%|██████▊   | 272136/400000 [00:31<00:14, 8670.08it/s] 68%|██████▊   | 273005/400000 [00:31<00:14, 8663.22it/s] 68%|██████▊   | 273892/400000 [00:31<00:14, 8723.51it/s] 69%|██████▊   | 274766/400000 [00:31<00:14, 8601.32it/s] 69%|██████▉   | 275628/400000 [00:31<00:14, 8585.96it/s] 69%|██████▉   | 276515/400000 [00:31<00:14, 8667.58it/s] 69%|██████▉   | 277383/400000 [00:31<00:14, 8497.94it/s] 70%|██████▉   | 278288/400000 [00:31<00:14, 8655.67it/s] 70%|██████▉   | 279173/400000 [00:32<00:13, 8711.26it/s] 70%|███████   | 280046/400000 [00:32<00:13, 8653.15it/s] 70%|███████   | 280981/400000 [00:32<00:13, 8849.96it/s] 70%|███████   | 281903/400000 [00:32<00:13, 8956.42it/s] 71%|███████   | 282827/400000 [00:32<00:12, 9038.70it/s] 71%|███████   | 283733/400000 [00:32<00:12, 8980.09it/s] 71%|███████   | 284632/400000 [00:32<00:12, 8891.84it/s] 71%|███████▏  | 285523/400000 [00:32<00:12, 8829.89it/s] 72%|███████▏  | 286407/400000 [00:32<00:12, 8750.31it/s] 72%|███████▏  | 287283/400000 [00:32<00:12, 8739.31it/s] 72%|███████▏  | 288158/400000 [00:33<00:12, 8739.90it/s] 72%|███████▏  | 289033/400000 [00:33<00:12, 8713.41it/s] 72%|███████▏  | 289914/400000 [00:33<00:12, 8740.31it/s] 73%|███████▎  | 290789/400000 [00:33<00:12, 8722.88it/s] 73%|███████▎  | 291662/400000 [00:33<00:12, 8501.06it/s] 73%|███████▎  | 292514/400000 [00:33<00:12, 8385.86it/s] 73%|███████▎  | 293396/400000 [00:33<00:12, 8508.88it/s] 74%|███████▎  | 294263/400000 [00:33<00:12, 8553.86it/s] 74%|███████▍  | 295125/400000 [00:33<00:12, 8572.64it/s] 74%|███████▍  | 296033/400000 [00:33<00:11, 8716.98it/s] 74%|███████▍  | 296910/400000 [00:34<00:11, 8730.81it/s] 74%|███████▍  | 297784/400000 [00:34<00:11, 8658.64it/s] 75%|███████▍  | 298661/400000 [00:34<00:11, 8690.85it/s] 75%|███████▍  | 299584/400000 [00:34<00:11, 8844.00it/s] 75%|███████▌  | 300491/400000 [00:34<00:11, 8907.83it/s] 75%|███████▌  | 301436/400000 [00:34<00:10, 9063.14it/s] 76%|███████▌  | 302344/400000 [00:34<00:10, 8984.22it/s] 76%|███████▌  | 303244/400000 [00:34<00:10, 8986.28it/s] 76%|███████▌  | 304144/400000 [00:34<00:10, 8879.32it/s] 76%|███████▋  | 305055/400000 [00:34<00:10, 8945.92it/s] 76%|███████▋  | 305951/400000 [00:35<00:10, 8878.41it/s] 77%|███████▋  | 306840/400000 [00:35<00:10, 8740.88it/s] 77%|███████▋  | 307715/400000 [00:35<00:10, 8729.44it/s] 77%|███████▋  | 308621/400000 [00:35<00:10, 8823.96it/s] 77%|███████▋  | 309520/400000 [00:35<00:10, 8872.92it/s] 78%|███████▊  | 310408/400000 [00:35<00:10, 8760.76it/s] 78%|███████▊  | 311285/400000 [00:35<00:10, 8763.37it/s] 78%|███████▊  | 312162/400000 [00:35<00:10, 8734.08it/s] 78%|███████▊  | 313036/400000 [00:35<00:09, 8716.89it/s] 78%|███████▊  | 313908/400000 [00:35<00:09, 8690.20it/s] 79%|███████▊  | 314783/400000 [00:36<00:09, 8705.63it/s] 79%|███████▉  | 315654/400000 [00:36<00:09, 8609.20it/s] 79%|███████▉  | 316566/400000 [00:36<00:09, 8756.11it/s] 79%|███████▉  | 317493/400000 [00:36<00:09, 8903.86it/s] 80%|███████▉  | 318392/400000 [00:36<00:09, 8927.35it/s] 80%|███████▉  | 319286/400000 [00:36<00:09, 8800.02it/s] 80%|████████  | 320168/400000 [00:36<00:09, 8698.14it/s] 80%|████████  | 321039/400000 [00:36<00:09, 8681.08it/s] 80%|████████  | 321908/400000 [00:36<00:09, 8397.50it/s] 81%|████████  | 322789/400000 [00:36<00:09, 8515.91it/s] 81%|████████  | 323681/400000 [00:37<00:08, 8632.54it/s] 81%|████████  | 324547/400000 [00:37<00:08, 8633.11it/s] 81%|████████▏ | 325462/400000 [00:37<00:08, 8781.43it/s] 82%|████████▏ | 326418/400000 [00:37<00:08, 8999.29it/s] 82%|████████▏ | 327334/400000 [00:37<00:08, 9045.23it/s] 82%|████████▏ | 328268/400000 [00:37<00:07, 9129.48it/s] 82%|████████▏ | 329183/400000 [00:37<00:07, 8948.01it/s] 83%|████████▎ | 330099/400000 [00:37<00:07, 9009.44it/s] 83%|████████▎ | 331014/400000 [00:37<00:07, 9048.87it/s] 83%|████████▎ | 331920/400000 [00:37<00:07, 9035.33it/s] 83%|████████▎ | 332825/400000 [00:38<00:07, 8803.13it/s] 83%|████████▎ | 333708/400000 [00:38<00:07, 8768.28it/s] 84%|████████▎ | 334587/400000 [00:38<00:07, 8739.81it/s] 84%|████████▍ | 335462/400000 [00:38<00:07, 8629.61it/s] 84%|████████▍ | 336347/400000 [00:38<00:07, 8691.94it/s] 84%|████████▍ | 337243/400000 [00:38<00:07, 8769.42it/s] 85%|████████▍ | 338173/400000 [00:38<00:06, 8920.72it/s] 85%|████████▍ | 339076/400000 [00:38<00:06, 8952.43it/s] 85%|████████▍ | 339973/400000 [00:38<00:06, 8891.11it/s] 85%|████████▌ | 340863/400000 [00:39<00:06, 8802.98it/s] 85%|████████▌ | 341744/400000 [00:39<00:06, 8634.71it/s] 86%|████████▌ | 342663/400000 [00:39<00:06, 8793.79it/s] 86%|████████▌ | 343599/400000 [00:39<00:06, 8954.55it/s] 86%|████████▌ | 344530/400000 [00:39<00:06, 9056.91it/s] 86%|████████▋ | 345482/400000 [00:39<00:05, 9189.12it/s] 87%|████████▋ | 346434/400000 [00:39<00:05, 9285.22it/s] 87%|████████▋ | 347385/400000 [00:39<00:05, 9348.41it/s] 87%|████████▋ | 348321/400000 [00:39<00:05, 9242.23it/s] 87%|████████▋ | 349254/400000 [00:39<00:05, 9265.98it/s] 88%|████████▊ | 350182/400000 [00:40<00:05, 9127.80it/s] 88%|████████▊ | 351096/400000 [00:40<00:05, 8997.06it/s] 88%|████████▊ | 351997/400000 [00:40<00:05, 8831.89it/s] 88%|████████▊ | 352882/400000 [00:40<00:05, 8784.72it/s] 88%|████████▊ | 353773/400000 [00:40<00:05, 8819.29it/s] 89%|████████▊ | 354665/400000 [00:40<00:05, 8848.34it/s] 89%|████████▉ | 355576/400000 [00:40<00:04, 8923.03it/s] 89%|████████▉ | 356469/400000 [00:40<00:04, 8800.24it/s] 89%|████████▉ | 357350/400000 [00:40<00:04, 8762.69it/s] 90%|████████▉ | 358227/400000 [00:40<00:04, 8707.52it/s] 90%|████████▉ | 359100/400000 [00:41<00:04, 8711.64it/s] 90%|████████▉ | 359972/400000 [00:41<00:04, 8568.39it/s] 90%|█████████ | 360846/400000 [00:41<00:04, 8618.09it/s] 90%|█████████ | 361709/400000 [00:41<00:04, 8593.96it/s] 91%|█████████ | 362569/400000 [00:41<00:04, 8552.84it/s] 91%|█████████ | 363425/400000 [00:41<00:04, 8505.32it/s] 91%|█████████ | 364276/400000 [00:41<00:04, 8463.68it/s] 91%|█████████▏| 365123/400000 [00:41<00:04, 8393.02it/s] 91%|█████████▏| 365970/400000 [00:41<00:04, 8415.87it/s] 92%|█████████▏| 366815/400000 [00:41<00:03, 8423.48it/s] 92%|█████████▏| 367698/400000 [00:42<00:03, 8539.77it/s] 92%|█████████▏| 368577/400000 [00:42<00:03, 8610.62it/s] 92%|█████████▏| 369439/400000 [00:42<00:03, 8606.58it/s] 93%|█████████▎| 370329/400000 [00:42<00:03, 8691.00it/s] 93%|█████████▎| 371237/400000 [00:42<00:03, 8803.58it/s] 93%|█████████▎| 372157/400000 [00:42<00:03, 8917.36it/s] 93%|█████████▎| 373083/400000 [00:42<00:02, 9014.95it/s] 93%|█████████▎| 373986/400000 [00:42<00:02, 8934.22it/s] 94%|█████████▎| 374881/400000 [00:42<00:02, 8642.77it/s] 94%|█████████▍| 375768/400000 [00:42<00:02, 8709.32it/s] 94%|█████████▍| 376655/400000 [00:43<00:02, 8755.87it/s] 94%|█████████▍| 377546/400000 [00:43<00:02, 8798.77it/s] 95%|█████████▍| 378427/400000 [00:43<00:02, 8752.24it/s] 95%|█████████▍| 379304/400000 [00:43<00:02, 8709.80it/s] 95%|█████████▌| 380188/400000 [00:43<00:02, 8745.99it/s] 95%|█████████▌| 381071/400000 [00:43<00:02, 8769.02it/s] 95%|█████████▌| 381987/400000 [00:43<00:02, 8879.75it/s] 96%|█████████▌| 382891/400000 [00:43<00:01, 8925.90it/s] 96%|█████████▌| 383785/400000 [00:43<00:01, 8891.66it/s] 96%|█████████▌| 384689/400000 [00:43<00:01, 8935.28it/s] 96%|█████████▋| 385585/400000 [00:44<00:01, 8942.58it/s] 97%|█████████▋| 386502/400000 [00:44<00:01, 9009.18it/s] 97%|█████████▋| 387404/400000 [00:44<00:01, 8807.25it/s] 97%|█████████▋| 388286/400000 [00:44<00:01, 8655.95it/s] 97%|█████████▋| 389199/400000 [00:44<00:01, 8790.68it/s] 98%|█████████▊| 390117/400000 [00:44<00:01, 8901.71it/s] 98%|█████████▊| 391050/400000 [00:44<00:00, 9024.98it/s] 98%|█████████▊| 391954/400000 [00:44<00:00, 8738.23it/s] 98%|█████████▊| 392831/400000 [00:44<00:00, 8708.34it/s] 98%|█████████▊| 393722/400000 [00:45<00:00, 8766.74it/s] 99%|█████████▊| 394601/400000 [00:45<00:00, 8734.28it/s] 99%|█████████▉| 395502/400000 [00:45<00:00, 8813.72it/s] 99%|█████████▉| 396385/400000 [00:45<00:00, 8688.62it/s] 99%|█████████▉| 397255/400000 [00:45<00:00, 8648.86it/s]100%|█████████▉| 398163/400000 [00:45<00:00, 8772.47it/s]100%|█████████▉| 399071/400000 [00:45<00:00, 8861.31it/s]100%|█████████▉| 399963/400000 [00:45<00:00, 8877.59it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8746.23it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f7033906438>, <torchtext.data.dataset.TabularDataset object at 0x7f7033906588>, <torchtext.vocab.Vocab object at 0x7f70339064a8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.20 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.20 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.20 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.65 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.65 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.03 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.03 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.51 file/s]2020-06-28 18:18:46.636415: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-28 18:18:46.640965: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-28 18:18:46.641141: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5605d24f4e90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-28 18:18:46.641157: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|██▉       | 2965504/9912422 [00:00<00:00, 29217876.40it/s]9920512it [00:00, 33810203.90it/s]                             
0it [00:00, ?it/s]32768it [00:00, 656462.48it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 481551.44it/s]1654784it [00:00, 12152091.66it/s]                         
0it [00:00, ?it/s]8192it [00:00, 232460.39it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

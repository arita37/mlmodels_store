
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fedc56b50d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fedc56b50d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fee309ff1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fee309ff1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fee4ed46e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fee4ed46e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fedddd2c620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fedddd2c620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fedddd2c620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:21, 121766.83it/s] 78%|███████▊  | 7700480/9912422 [00:00<00:12, 173832.42it/s]9920512it [00:00, 38700200.77it/s]                           
0it [00:00, ?it/s]32768it [00:00, 603676.17it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162433.97it/s]1654784it [00:00, 10931613.35it/s]                         
0it [00:00, ?it/s]8192it [00:00, 189141.03it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fedc4d6c978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fedc4d60be0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fedddd2c268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fedddd2c268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fedddd2c268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:17,  2.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:17,  2.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:17,  2.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:16,  2.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:16,  2.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:15,  2.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:15,  2.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:53,  2.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:53,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:52,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:52,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:51,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:51,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:50,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:50,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:35,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:35,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:35,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:34,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:34,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:34,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:34,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:33,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:24,  5.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:24,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:22,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  7.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  7.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  7.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:15,  7.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:01<00:11, 10.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:11, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:10, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 10.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 14.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 14.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 14.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 14.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 14.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 14.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 19.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 19.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 19.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 19.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 24.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 24.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 24.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 24.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 24.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 24.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 24.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 24.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 30.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 30.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 30.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 30.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 30.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 30.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 37.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 37.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 37.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 37.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 37.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 37.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 37.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 37.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 37.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 44.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 44.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 44.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 44.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 44.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 44.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 44.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 44.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 44.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 50.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 56.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 56.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 56.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 56.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 56.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 56.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 56.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 56.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 56.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 56.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 62.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 66.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 67.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 67.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 67.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 67.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 67.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 67.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 67.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 67.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 67.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 65.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 65.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 65.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 65.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 65.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 65.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 62.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 61.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 61.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 61.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 61.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 61.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 61.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 61.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 61.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 60.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.78s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.78s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.78s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.78s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 60.49 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.25 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ url]
0 examples [00:00, ? examples/s]2020-07-11 18:08:20.957802: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-11 18:08:21.059745: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-11 18:08:21.059946: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f9949fe7c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-11 18:08:21.059966: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.28 examples/s]90 examples [00:00,  4.68 examples/s]191 examples [00:00,  6.68 examples/s]287 examples [00:00,  9.51 examples/s]395 examples [00:00, 13.54 examples/s]493 examples [00:00, 19.22 examples/s]598 examples [00:00, 27.25 examples/s]705 examples [00:01, 38.50 examples/s]807 examples [00:01, 54.12 examples/s]904 examples [00:01, 75.47 examples/s]1001 examples [00:01, 104.12 examples/s]1101 examples [00:01, 142.34 examples/s]1200 examples [00:01, 191.48 examples/s]1298 examples [00:01, 252.30 examples/s]1401 examples [00:01, 325.93 examples/s]1500 examples [00:01, 407.32 examples/s]1599 examples [00:01, 489.22 examples/s]1696 examples [00:02, 572.22 examples/s]1797 examples [00:02, 656.65 examples/s]1895 examples [00:02, 725.60 examples/s]1992 examples [00:02, 775.44 examples/s]2088 examples [00:02, 820.87 examples/s]2186 examples [00:02, 860.85 examples/s]2285 examples [00:02, 891.35 examples/s]2391 examples [00:02, 934.44 examples/s]2494 examples [00:02, 959.80 examples/s]2595 examples [00:02, 966.64 examples/s]2695 examples [00:03, 954.39 examples/s]2793 examples [00:03, 951.53 examples/s]2890 examples [00:03, 933.98 examples/s]2985 examples [00:03, 910.92 examples/s]3092 examples [00:03, 953.42 examples/s]3195 examples [00:03, 973.11 examples/s]3302 examples [00:03, 999.85 examples/s]3413 examples [00:03, 1028.72 examples/s]3517 examples [00:03, 1026.22 examples/s]3624 examples [00:03, 1038.43 examples/s]3729 examples [00:04, 1034.96 examples/s]3835 examples [00:04, 1041.21 examples/s]3940 examples [00:04, 1003.82 examples/s]4041 examples [00:04, 996.60 examples/s] 4141 examples [00:04, 950.13 examples/s]4237 examples [00:04, 922.88 examples/s]4330 examples [00:04, 882.00 examples/s]4420 examples [00:04, 883.32 examples/s]4510 examples [00:04, 886.20 examples/s]4604 examples [00:05, 900.17 examples/s]4701 examples [00:05, 919.55 examples/s]4800 examples [00:05, 939.04 examples/s]4900 examples [00:05, 955.23 examples/s]4996 examples [00:05, 951.47 examples/s]5092 examples [00:05, 941.35 examples/s]5195 examples [00:05, 963.90 examples/s]5301 examples [00:05, 987.96 examples/s]5404 examples [00:05, 998.27 examples/s]5505 examples [00:05, 976.97 examples/s]5611 examples [00:06, 1000.02 examples/s]5720 examples [00:06, 1023.26 examples/s]5823 examples [00:06, 1013.81 examples/s]5925 examples [00:06, 998.06 examples/s] 6029 examples [00:06, 1009.59 examples/s]6142 examples [00:06, 1040.19 examples/s]6247 examples [00:06, 1011.28 examples/s]6358 examples [00:06, 1038.59 examples/s]6468 examples [00:06, 1054.79 examples/s]6581 examples [00:06, 1076.07 examples/s]6695 examples [00:07, 1093.57 examples/s]6805 examples [00:07, 1063.82 examples/s]6912 examples [00:07, 1040.80 examples/s]7023 examples [00:07, 1059.05 examples/s]7130 examples [00:07, 1058.29 examples/s]7238 examples [00:07, 1062.19 examples/s]7349 examples [00:07, 1074.54 examples/s]7462 examples [00:07, 1089.15 examples/s]7576 examples [00:07, 1102.42 examples/s]7687 examples [00:08, 1081.08 examples/s]7799 examples [00:08, 1091.33 examples/s]7909 examples [00:08, 1078.93 examples/s]8018 examples [00:08, 1008.99 examples/s]8120 examples [00:08, 972.26 examples/s] 8227 examples [00:08, 999.04 examples/s]8334 examples [00:08, 1016.82 examples/s]8448 examples [00:08, 1049.50 examples/s]8556 examples [00:08, 1056.95 examples/s]8663 examples [00:08, 1059.76 examples/s]8770 examples [00:09, 1059.72 examples/s]8877 examples [00:09, 1030.52 examples/s]8981 examples [00:09, 1030.26 examples/s]9089 examples [00:09, 1040.97 examples/s]9196 examples [00:09, 1048.77 examples/s]9310 examples [00:09, 1073.80 examples/s]9423 examples [00:09, 1087.79 examples/s]9533 examples [00:09, 1083.78 examples/s]9642 examples [00:09, 1058.09 examples/s]9749 examples [00:10, 1001.59 examples/s]9850 examples [00:10, 991.14 examples/s] 9953 examples [00:10, 1001.16 examples/s]10054 examples [00:10, 932.54 examples/s]10164 examples [00:10, 975.04 examples/s]10267 examples [00:10, 989.45 examples/s]10380 examples [00:10, 1027.77 examples/s]10495 examples [00:10, 1061.54 examples/s]10605 examples [00:10, 1071.97 examples/s]10716 examples [00:10, 1081.52 examples/s]10825 examples [00:11, 1060.99 examples/s]10932 examples [00:11, 1031.48 examples/s]11036 examples [00:11, 1028.62 examples/s]11140 examples [00:11, 1030.57 examples/s]11248 examples [00:11, 1043.11 examples/s]11353 examples [00:11, 997.03 examples/s] 11463 examples [00:11, 1025.14 examples/s]11570 examples [00:11, 1036.22 examples/s]11675 examples [00:11, 1006.44 examples/s]11777 examples [00:12, 983.94 examples/s] 11880 examples [00:12, 995.57 examples/s]11983 examples [00:12, 1003.94 examples/s]12086 examples [00:12, 1008.53 examples/s]12188 examples [00:12, 990.43 examples/s] 12288 examples [00:12, 966.75 examples/s]12399 examples [00:12, 1004.75 examples/s]12505 examples [00:12, 1018.87 examples/s]12609 examples [00:12, 1023.92 examples/s]12712 examples [00:12, 1015.47 examples/s]12817 examples [00:13, 1023.38 examples/s]12920 examples [00:13, 1022.11 examples/s]13023 examples [00:13, 1017.75 examples/s]13128 examples [00:13, 1025.83 examples/s]13236 examples [00:13, 1040.19 examples/s]13341 examples [00:13, 1035.22 examples/s]13449 examples [00:13, 1047.82 examples/s]13555 examples [00:13, 1049.84 examples/s]13661 examples [00:13, 1046.51 examples/s]13766 examples [00:13, 1045.12 examples/s]13871 examples [00:14, 1028.14 examples/s]13974 examples [00:14, 1027.66 examples/s]14078 examples [00:14, 1029.99 examples/s]14182 examples [00:14, 1027.28 examples/s]14285 examples [00:14, 1023.23 examples/s]14392 examples [00:14, 1035.69 examples/s]14503 examples [00:14, 1054.79 examples/s]14612 examples [00:14, 1064.88 examples/s]14719 examples [00:14, 1054.41 examples/s]14834 examples [00:14, 1078.90 examples/s]14943 examples [00:15, 1072.87 examples/s]15051 examples [00:15, 1040.55 examples/s]15158 examples [00:15, 1048.78 examples/s]15272 examples [00:15, 1072.82 examples/s]15381 examples [00:15, 1077.81 examples/s]15489 examples [00:15, 1048.71 examples/s]15595 examples [00:15, 1049.90 examples/s]15705 examples [00:15, 1062.36 examples/s]15812 examples [00:15, 1039.11 examples/s]15917 examples [00:15, 1015.38 examples/s]16024 examples [00:16, 1029.53 examples/s]16132 examples [00:16, 1041.83 examples/s]16237 examples [00:16, 1032.62 examples/s]16341 examples [00:16, 1018.15 examples/s]16446 examples [00:16, 1026.77 examples/s]16549 examples [00:16, 982.99 examples/s] 16653 examples [00:16, 997.17 examples/s]16762 examples [00:16, 1021.20 examples/s]16866 examples [00:16, 1024.54 examples/s]16975 examples [00:17, 1041.98 examples/s]17080 examples [00:17, 1035.10 examples/s]17185 examples [00:17, 1037.33 examples/s]17289 examples [00:17, 1003.87 examples/s]17398 examples [00:17, 1027.47 examples/s]17511 examples [00:17, 1054.11 examples/s]17617 examples [00:17, 1039.94 examples/s]17722 examples [00:17, 1030.29 examples/s]17826 examples [00:17, 1013.53 examples/s]17928 examples [00:17, 1012.97 examples/s]18044 examples [00:18, 1051.49 examples/s]18150 examples [00:18, 1050.42 examples/s]18263 examples [00:18, 1070.56 examples/s]18371 examples [00:18, 1063.21 examples/s]18480 examples [00:18, 1069.86 examples/s]18588 examples [00:18, 1039.62 examples/s]18693 examples [00:18, 1020.12 examples/s]18809 examples [00:18, 1055.61 examples/s]18916 examples [00:18, 1046.98 examples/s]19022 examples [00:18, 1006.21 examples/s]19124 examples [00:19, 1005.35 examples/s]19234 examples [00:19, 1029.90 examples/s]19345 examples [00:19, 1052.58 examples/s]19456 examples [00:19, 1068.94 examples/s]19564 examples [00:19, 1042.74 examples/s]19669 examples [00:19, 1002.27 examples/s]19771 examples [00:19, 1006.49 examples/s]19876 examples [00:19, 1007.53 examples/s]19978 examples [00:19, 993.60 examples/s] 20078 examples [00:20, 954.28 examples/s]20174 examples [00:20, 949.10 examples/s]20270 examples [00:20, 903.36 examples/s]20362 examples [00:20, 884.73 examples/s]20467 examples [00:20, 928.17 examples/s]20572 examples [00:20, 959.32 examples/s]20669 examples [00:20, 959.33 examples/s]20776 examples [00:20, 988.57 examples/s]20883 examples [00:20, 1010.85 examples/s]20988 examples [00:20, 1020.93 examples/s]21095 examples [00:21, 1031.73 examples/s]21199 examples [00:21, 1019.06 examples/s]21304 examples [00:21, 1027.76 examples/s]21411 examples [00:21, 1039.67 examples/s]21518 examples [00:21, 1047.77 examples/s]21623 examples [00:21, 1031.27 examples/s]21727 examples [00:21, 1000.98 examples/s]21841 examples [00:21, 1038.61 examples/s]21953 examples [00:21, 1061.06 examples/s]22060 examples [00:22, 1059.72 examples/s]22167 examples [00:22, 1062.68 examples/s]22274 examples [00:22, 1060.84 examples/s]22385 examples [00:22, 1073.44 examples/s]22500 examples [00:22, 1094.34 examples/s]22610 examples [00:22, 1076.58 examples/s]22726 examples [00:22, 1098.03 examples/s]22837 examples [00:22, 1055.15 examples/s]22945 examples [00:22, 1061.20 examples/s]23052 examples [00:22, 1056.86 examples/s]23160 examples [00:23, 1063.28 examples/s]23267 examples [00:23, 1050.64 examples/s]23373 examples [00:23, 1038.80 examples/s]23478 examples [00:23, 1038.63 examples/s]23582 examples [00:23, 1037.23 examples/s]23686 examples [00:23, 1030.74 examples/s]23790 examples [00:23, 1033.01 examples/s]23894 examples [00:23, 1015.06 examples/s]23996 examples [00:23, 1001.43 examples/s]24097 examples [00:23, 991.96 examples/s] 24203 examples [00:24, 1010.52 examples/s]24307 examples [00:24, 1017.43 examples/s]24409 examples [00:24, 975.35 examples/s] 24507 examples [00:24, 932.52 examples/s]24604 examples [00:24, 942.77 examples/s]24705 examples [00:24, 961.09 examples/s]24803 examples [00:24, 965.88 examples/s]24902 examples [00:24, 970.34 examples/s]25011 examples [00:24, 1001.88 examples/s]25113 examples [00:24, 1004.93 examples/s]25217 examples [00:25, 1012.13 examples/s]25319 examples [00:25, 993.55 examples/s] 25419 examples [00:25, 989.52 examples/s]25519 examples [00:25, 970.33 examples/s]25617 examples [00:25, 956.43 examples/s]25722 examples [00:25, 982.66 examples/s]25824 examples [00:25, 992.69 examples/s]25926 examples [00:25, 999.69 examples/s]26027 examples [00:25, 977.13 examples/s]26131 examples [00:26, 994.74 examples/s]26231 examples [00:26, 984.67 examples/s]26330 examples [00:26, 974.54 examples/s]26434 examples [00:26, 992.72 examples/s]26545 examples [00:26, 1023.18 examples/s]26649 examples [00:26, 1026.30 examples/s]26754 examples [00:26, 1032.59 examples/s]26858 examples [00:26, 1023.49 examples/s]26962 examples [00:26, 1027.16 examples/s]27065 examples [00:26, 990.67 examples/s] 27165 examples [00:27, 986.16 examples/s]27267 examples [00:27, 993.51 examples/s]27367 examples [00:27, 985.17 examples/s]27466 examples [00:27, 979.71 examples/s]27568 examples [00:27, 989.37 examples/s]27671 examples [00:27, 1000.64 examples/s]27773 examples [00:27, 1005.52 examples/s]27874 examples [00:27, 992.27 examples/s] 27977 examples [00:27, 1002.54 examples/s]28078 examples [00:27, 988.50 examples/s] 28181 examples [00:28, 999.04 examples/s]28285 examples [00:28, 1010.40 examples/s]28387 examples [00:28, 985.48 examples/s] 28492 examples [00:28, 1001.34 examples/s]28598 examples [00:28, 1016.21 examples/s]28709 examples [00:28, 1042.23 examples/s]28821 examples [00:28, 1062.34 examples/s]28928 examples [00:28, 1039.57 examples/s]29033 examples [00:28, 1013.06 examples/s]29141 examples [00:29, 1030.87 examples/s]29245 examples [00:29, 1006.42 examples/s]29347 examples [00:29, 1006.20 examples/s]29448 examples [00:29, 968.27 examples/s] 29550 examples [00:29, 981.75 examples/s]29651 examples [00:29, 988.73 examples/s]29755 examples [00:29, 1002.74 examples/s]29860 examples [00:29, 1014.88 examples/s]29964 examples [00:29, 1021.85 examples/s]30067 examples [00:29, 974.32 examples/s] 30173 examples [00:30, 996.61 examples/s]30283 examples [00:30, 1024.01 examples/s]30386 examples [00:30, 1024.65 examples/s]30489 examples [00:30, 1024.76 examples/s]30595 examples [00:30, 1034.16 examples/s]30701 examples [00:30, 1041.36 examples/s]30806 examples [00:30, 1002.93 examples/s]30915 examples [00:30, 1025.90 examples/s]31019 examples [00:30, 1023.97 examples/s]31126 examples [00:30, 1035.00 examples/s]31230 examples [00:31, 1035.27 examples/s]31334 examples [00:31, 1033.02 examples/s]31438 examples [00:31, 1027.42 examples/s]31542 examples [00:31, 1029.59 examples/s]31650 examples [00:31, 1043.55 examples/s]31755 examples [00:31, 1043.62 examples/s]31860 examples [00:31, 1035.52 examples/s]31964 examples [00:31, 1034.73 examples/s]32068 examples [00:31, 1006.79 examples/s]32169 examples [00:32, 981.61 examples/s] 32273 examples [00:32, 996.51 examples/s]32373 examples [00:32, 985.22 examples/s]32472 examples [00:32, 972.96 examples/s]32570 examples [00:32, 969.02 examples/s]32671 examples [00:32, 978.24 examples/s]32775 examples [00:32, 994.26 examples/s]32878 examples [00:32, 1004.00 examples/s]32980 examples [00:32, 1007.12 examples/s]33081 examples [00:32, 1005.54 examples/s]33185 examples [00:33, 1014.15 examples/s]33287 examples [00:33, 1011.63 examples/s]33389 examples [00:33, 1005.77 examples/s]33490 examples [00:33, 990.50 examples/s] 33595 examples [00:33, 1006.06 examples/s]33696 examples [00:33, 1000.11 examples/s]33798 examples [00:33, 1003.93 examples/s]33899 examples [00:33, 991.05 examples/s] 33999 examples [00:33, 962.34 examples/s]34096 examples [00:33, 959.02 examples/s]34194 examples [00:34, 962.77 examples/s]34295 examples [00:34, 976.06 examples/s]34396 examples [00:34, 985.17 examples/s]34495 examples [00:34, 986.33 examples/s]34594 examples [00:34, 968.50 examples/s]34694 examples [00:34, 977.32 examples/s]34800 examples [00:34, 1000.39 examples/s]34906 examples [00:34, 1016.61 examples/s]35008 examples [00:34, 1008.57 examples/s]35114 examples [00:34, 1020.96 examples/s]35217 examples [00:35, 996.00 examples/s] 35324 examples [00:35, 1015.12 examples/s]35429 examples [00:35, 1024.72 examples/s]35532 examples [00:35, 996.18 examples/s] 35632 examples [00:35, 994.13 examples/s]35743 examples [00:35, 1025.57 examples/s]35851 examples [00:35, 1039.73 examples/s]35956 examples [00:35, 1032.40 examples/s]36060 examples [00:35, 1009.23 examples/s]36169 examples [00:35, 1030.54 examples/s]36276 examples [00:36, 1040.22 examples/s]36383 examples [00:36, 1046.49 examples/s]36488 examples [00:36, 1034.11 examples/s]36593 examples [00:36, 1037.06 examples/s]36697 examples [00:36, 1018.55 examples/s]36805 examples [00:36, 1034.57 examples/s]36909 examples [00:36, 1032.54 examples/s]37018 examples [00:36, 1046.33 examples/s]37123 examples [00:36, 1042.25 examples/s]37228 examples [00:37, 1005.76 examples/s]37331 examples [00:37, 1010.21 examples/s]37434 examples [00:37, 1014.83 examples/s]37538 examples [00:37, 1020.73 examples/s]37641 examples [00:37, 1011.37 examples/s]37746 examples [00:37, 1020.97 examples/s]37849 examples [00:37, 1019.35 examples/s]37959 examples [00:37, 1041.04 examples/s]38073 examples [00:37, 1068.67 examples/s]38181 examples [00:37, 1064.67 examples/s]38289 examples [00:38, 1067.23 examples/s]38396 examples [00:38, 1037.15 examples/s]38501 examples [00:38, 986.57 examples/s] 38601 examples [00:38, 965.84 examples/s]38699 examples [00:38, 949.21 examples/s]38802 examples [00:38, 971.44 examples/s]38906 examples [00:38, 990.74 examples/s]39019 examples [00:38, 1026.79 examples/s]39128 examples [00:38, 1042.69 examples/s]39234 examples [00:38, 1046.18 examples/s]39340 examples [00:39, 1049.46 examples/s]39446 examples [00:39, 1043.70 examples/s]39551 examples [00:39, 1037.21 examples/s]39662 examples [00:39, 1057.53 examples/s]39768 examples [00:39, 1043.38 examples/s]39879 examples [00:39, 1060.63 examples/s]39986 examples [00:39, 1039.37 examples/s]40091 examples [00:39, 986.84 examples/s] 40199 examples [00:39, 1012.35 examples/s]40301 examples [00:40, 1011.31 examples/s]40404 examples [00:40, 1015.13 examples/s]40506 examples [00:40, 1004.76 examples/s]40607 examples [00:40, 1003.21 examples/s]40708 examples [00:40, 918.72 examples/s] 40802 examples [00:40, 873.28 examples/s]40899 examples [00:40, 900.02 examples/s]40991 examples [00:40, 903.26 examples/s]41083 examples [00:40, 892.71 examples/s]41173 examples [00:41, 852.72 examples/s]41260 examples [00:41, 854.71 examples/s]41356 examples [00:41, 881.85 examples/s]41445 examples [00:41, 876.51 examples/s]41555 examples [00:41, 931.47 examples/s]41656 examples [00:41, 953.60 examples/s]41757 examples [00:41, 968.97 examples/s]41862 examples [00:41, 990.07 examples/s]41962 examples [00:41, 980.95 examples/s]42061 examples [00:41, 974.29 examples/s]42162 examples [00:42, 983.40 examples/s]42270 examples [00:42, 1009.17 examples/s]42376 examples [00:42, 1022.24 examples/s]42480 examples [00:42, 1026.01 examples/s]42583 examples [00:42, 993.88 examples/s] 42683 examples [00:42, 991.17 examples/s]42789 examples [00:42, 1009.18 examples/s]42897 examples [00:42, 1027.96 examples/s]43003 examples [00:42, 1036.43 examples/s]43108 examples [00:42, 1039.20 examples/s]43213 examples [00:43, 1019.89 examples/s]43319 examples [00:43, 1031.44 examples/s]43423 examples [00:43, 1024.04 examples/s]43527 examples [00:43, 1026.32 examples/s]43630 examples [00:43, 1016.09 examples/s]43735 examples [00:43, 1023.47 examples/s]43841 examples [00:43, 1031.77 examples/s]43947 examples [00:43, 1039.23 examples/s]44051 examples [00:43, 1023.56 examples/s]44157 examples [00:43, 1034.05 examples/s]44261 examples [00:44, 1016.36 examples/s]44366 examples [00:44, 1024.78 examples/s]44469 examples [00:44, 998.70 examples/s] 44571 examples [00:44, 1004.39 examples/s]44677 examples [00:44, 1018.48 examples/s]44780 examples [00:44, 1014.64 examples/s]44889 examples [00:44, 1033.93 examples/s]44995 examples [00:44, 1040.74 examples/s]45100 examples [00:44, 1043.10 examples/s]45205 examples [00:44, 1013.09 examples/s]45307 examples [00:45, 1012.51 examples/s]45417 examples [00:45, 1034.94 examples/s]45524 examples [00:45, 1044.14 examples/s]45634 examples [00:45, 1058.57 examples/s]45741 examples [00:45, 1057.96 examples/s]45847 examples [00:45, 1041.96 examples/s]45958 examples [00:45, 1060.05 examples/s]46065 examples [00:45, 1060.64 examples/s]46173 examples [00:45, 1065.74 examples/s]46280 examples [00:46, 1047.83 examples/s]46385 examples [00:46, 1017.36 examples/s]46490 examples [00:46, 1026.53 examples/s]46593 examples [00:46, 1021.93 examples/s]46699 examples [00:46, 1033.05 examples/s]46804 examples [00:46, 1037.29 examples/s]46908 examples [00:46, 944.50 examples/s] 47017 examples [00:46, 983.63 examples/s]47128 examples [00:46, 1016.90 examples/s]47232 examples [00:46, 1017.83 examples/s]47335 examples [00:47, 1010.64 examples/s]47440 examples [00:47, 1021.80 examples/s]47543 examples [00:47, 1013.44 examples/s]47647 examples [00:47, 1019.44 examples/s]47750 examples [00:47, 1014.53 examples/s]47852 examples [00:47, 999.49 examples/s] 47953 examples [00:47, 994.18 examples/s]48055 examples [00:47, 999.74 examples/s]48158 examples [00:47, 1006.63 examples/s]48262 examples [00:47, 1015.92 examples/s]48364 examples [00:48, 1006.97 examples/s]48465 examples [00:48, 991.15 examples/s] 48568 examples [00:48, 1001.89 examples/s]48673 examples [00:48, 1013.60 examples/s]48777 examples [00:48, 1019.45 examples/s]48881 examples [00:48, 1022.37 examples/s]48991 examples [00:48, 1042.25 examples/s]49097 examples [00:48, 1047.42 examples/s]49203 examples [00:48, 1049.71 examples/s]49309 examples [00:48, 1029.84 examples/s]49413 examples [00:49, 1019.79 examples/s]49521 examples [00:49, 1035.34 examples/s]49628 examples [00:49, 1045.05 examples/s]49733 examples [00:49, 1034.96 examples/s]49837 examples [00:49, 1035.50 examples/s]49943 examples [00:49, 1040.54 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 17%|█▋        | 8490/50000 [00:00<00:00, 84896.44 examples/s] 42%|████▏     | 21132/50000 [00:00<00:00, 94175.62 examples/s] 67%|██████▋   | 33494/50000 [00:00<00:00, 101421.48 examples/s] 95%|█████████▌| 47556/50000 [00:00<00:00, 110675.91 examples/s]                                                                0 examples [00:00, ? examples/s]85 examples [00:00, 848.64 examples/s]197 examples [00:00, 913.57 examples/s]302 examples [00:00, 949.38 examples/s]408 examples [00:00, 979.12 examples/s]516 examples [00:00, 1004.93 examples/s]610 examples [00:00, 982.14 examples/s] 713 examples [00:00, 994.08 examples/s]820 examples [00:00, 1015.12 examples/s]928 examples [00:00, 1031.43 examples/s]1040 examples [00:01, 1054.53 examples/s]1144 examples [00:01, 1037.30 examples/s]1247 examples [00:01, 991.86 examples/s] 1352 examples [00:01, 1006.86 examples/s]1461 examples [00:01, 1030.01 examples/s]1566 examples [00:01, 1035.86 examples/s]1670 examples [00:01, 1025.75 examples/s]1774 examples [00:01, 1027.98 examples/s]1879 examples [00:01, 1032.10 examples/s]1983 examples [00:01, 1017.06 examples/s]2094 examples [00:02, 1042.47 examples/s]2199 examples [00:02, 1026.68 examples/s]2306 examples [00:02, 1036.98 examples/s]2410 examples [00:02, 1030.61 examples/s]2519 examples [00:02, 1046.33 examples/s]2628 examples [00:02, 1056.08 examples/s]2734 examples [00:02, 1031.50 examples/s]2844 examples [00:02, 1048.56 examples/s]2950 examples [00:02, 1029.03 examples/s]3061 examples [00:02, 1051.29 examples/s]3172 examples [00:03, 1068.07 examples/s]3280 examples [00:03, 1041.69 examples/s]3385 examples [00:03, 1027.67 examples/s]3494 examples [00:03, 1044.11 examples/s]3604 examples [00:03, 1059.41 examples/s]3718 examples [00:03, 1081.99 examples/s]3827 examples [00:03, 1077.50 examples/s]3935 examples [00:03, 1077.81 examples/s]4050 examples [00:03, 1097.41 examples/s]4160 examples [00:03, 1087.63 examples/s]4269 examples [00:04, 1076.50 examples/s]4377 examples [00:04, 1051.17 examples/s]4488 examples [00:04, 1066.09 examples/s]4595 examples [00:04, 1058.99 examples/s]4702 examples [00:04, 1053.64 examples/s]4815 examples [00:04, 1074.60 examples/s]4923 examples [00:04, 1064.15 examples/s]5032 examples [00:04, 1070.84 examples/s]5140 examples [00:04, 1046.17 examples/s]5245 examples [00:05, 1030.40 examples/s]5354 examples [00:05, 1045.99 examples/s]5459 examples [00:05, 1014.98 examples/s]5564 examples [00:05, 1022.46 examples/s]5670 examples [00:05, 1031.49 examples/s]5776 examples [00:05, 1038.33 examples/s]5889 examples [00:05, 1062.58 examples/s]5996 examples [00:05, 1059.93 examples/s]6108 examples [00:05, 1076.85 examples/s]6222 examples [00:05, 1093.27 examples/s]6332 examples [00:06, 1077.87 examples/s]6440 examples [00:06, 1074.03 examples/s]6548 examples [00:06, 1016.50 examples/s]6651 examples [00:06, 993.18 examples/s] 6759 examples [00:06, 1015.39 examples/s]6870 examples [00:06, 1041.20 examples/s]6976 examples [00:06, 1046.24 examples/s]7082 examples [00:06, 1046.61 examples/s]7187 examples [00:06, 1043.20 examples/s]7297 examples [00:06, 1058.60 examples/s]7406 examples [00:07, 1065.41 examples/s]7514 examples [00:07, 1068.39 examples/s]7621 examples [00:07, 1036.15 examples/s]7727 examples [00:07, 1041.08 examples/s]7833 examples [00:07, 1045.59 examples/s]7938 examples [00:07, 1016.05 examples/s]8046 examples [00:07, 1032.91 examples/s]8150 examples [00:07, 1033.40 examples/s]8255 examples [00:07, 1038.26 examples/s]8359 examples [00:08, 1033.69 examples/s]8467 examples [00:08, 1044.67 examples/s]8572 examples [00:08, 1013.30 examples/s]8676 examples [00:08, 1020.99 examples/s]8779 examples [00:08, 1015.58 examples/s]8881 examples [00:08, 1006.64 examples/s]8984 examples [00:08, 1012.00 examples/s]9086 examples [00:08, 1005.78 examples/s]9195 examples [00:08, 1029.29 examples/s]9299 examples [00:08, 1014.45 examples/s]9407 examples [00:09, 1030.34 examples/s]9511 examples [00:09, 1019.39 examples/s]9614 examples [00:09, 996.02 examples/s] 9714 examples [00:09, 984.38 examples/s]9813 examples [00:09, 982.73 examples/s]9919 examples [00:09, 1002.98 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete93JW6A/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete93JW6A/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fedddd2c620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fedddd2c620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fedddd2c620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fed7c1da128>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fed7c10b5f8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fedddd2c620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fedddd2c620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fedddd2c620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fedc4d605c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fedc4d60358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fed6408a1e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fed6408a1e0> 

  function with postional parmater data_info <function split_train_valid at 0x7fed6408a1e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fed6408a2f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fed6408a2f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fed6408a2f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=c224219d5eb70860f816a661bc212cf440e2811256e790deb24cdb44982060b7
  Stored in directory: /tmp/pip-ephem-wheel-cache-qe3qfm8v/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:24:06, 13.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:06:41, 18.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:13:52, 25.9kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:28:14, 37.0kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:31:06, 52.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.88M/862M [00:01<3:08:29, 75.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.6M/862M [00:01<2:11:10, 108kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.9M/862M [00:01<1:31:13, 154kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.6M/862M [00:01<1:03:30, 219kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.0M/862M [00:01<44:24, 312kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 36.7M/862M [00:01<30:58, 444kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.5M/862M [00:02<21:37, 632kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.2M/862M [00:02<15:07, 897kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<11:09, 1.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<09:41, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:04<09:01, 1.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:05<06:47, 1.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:05<04:52, 2.74MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<1:04:28, 207kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<46:44, 286kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<33:00, 404kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<25:45, 516kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<20:49, 638kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:09<15:16, 869kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:09<10:50, 1.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<28:25, 465kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:10<21:12, 623kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:10<15:06, 873kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<13:41, 961kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:12<10:54, 1.21MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:12<07:56, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<08:39, 1.51MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<08:41, 1.51MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:14<06:38, 1.97MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:15<04:47, 2.72MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<12:31, 1.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<10:06, 1.29MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:16<07:20, 1.77MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<08:10, 1.58MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<06:49, 1.90MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:18<05:05, 2.53MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<06:36, 1.95MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<07:13, 1.78MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:20<05:41, 2.26MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<06:03, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<05:31, 2.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:22<04:11, 3.05MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<05:54, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<06:49, 1.87MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:24<05:25, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<03:55, 3.23MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<48:43, 260kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<35:23, 358kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<25:02, 505kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<20:26, 616kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<16:50, 748kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<12:19, 1.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<08:47, 1.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<11:10, 1.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<09:04, 1.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<06:37, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:33, 1.65MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:47, 1.60MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:59, 2.08MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<04:18, 2.88MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<17:24, 712kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<13:27, 921kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<09:42, 1.27MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<09:40, 1.27MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<09:21, 1.32MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:05, 1.74MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<05:04, 2.41MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<14:33, 842kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<11:14, 1.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<08:10, 1.50MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:32, 1.43MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:11, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<05:17, 2.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:35, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:03, 1.72MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:32, 2.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:49, 2.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:18, 2.27MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:00, 3.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:33, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:13, 1.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<04:56, 2.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:23, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:16, 1.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:00, 2.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<03:38, 3.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<31:08, 380kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<22:58, 515kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<16:20, 723kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<14:10, 831kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<11:07, 1.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<08:01, 1.46MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<08:21, 1.40MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:01, 1.66MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<05:12, 2.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<06:23, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:52, 1.69MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:23, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:38, 2.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:06, 2.26MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:48, 3.03MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:22, 2.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<04:55, 2.33MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:41, 3.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:18, 2.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:01, 1.90MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<04:47, 2.39MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:11, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:55, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:43, 2.40MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<03:26, 3.29MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<1:23:22, 136kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<59:27, 190kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<41:46, 270kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<31:47, 353kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<23:22, 480kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<16:36, 674kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<14:14, 784kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<11:05, 1.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<07:59, 1.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<08:13, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:52, 1.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<05:02, 2.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:08, 1.79MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:13, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<03:56, 2.79MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:19, 2.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:50, 2.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<03:36, 3.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:07, 2.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:40, 2.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<03:30, 3.10MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:04, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:45, 1.88MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:29, 2.41MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<03:15, 3.30MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<10:52, 989kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:41, 1.24MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<06:18, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<06:55, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<07:00, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:26, 1.96MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:31, 1.92MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:56, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:40, 2.88MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:02, 2.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:39, 1.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:29, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:49, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:38, 1.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:28, 2.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<03:15, 3.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<1:16:36, 136kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<54:40, 190kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<38:26, 270kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<29:14, 353kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<22:33, 458kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<16:17, 634kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<13:00, 789kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<10:09, 1.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<07:21, 1.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:31, 1.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:25, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:39, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<04:03, 2.50MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<09:34, 1.06MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<07:47, 1.30MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:42, 1.77MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:14, 1.61MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<06:34, 1.53MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:08, 1.95MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<03:42, 2.69MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<18:23, 544kB/s] .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<13:45, 726kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<09:52, 1.01MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<09:05, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<07:25, 1.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:24, 1.83MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:59, 1.64MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:16, 1.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:50, 2.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:31, 2.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:22, 1.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:29, 1.78MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:02, 2.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:01, 1.93MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:33, 1.75MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<04:19, 2.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:08, 3.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<06:47, 1.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:47, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:18, 2.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:08, 1.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:36, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<04:25, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<03:12, 2.97MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<30:53, 308kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<22:38, 420kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<16:03, 590kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<13:18, 709kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<11:18, 834kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<08:19, 1.13MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<05:56, 1.58MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<08:14, 1.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:46, 1.38MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:58, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<05:33, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:41, 1.98MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<03:31, 2.63MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:32, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:07, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:03, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<02:56, 3.11MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<31:10, 294kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<22:46, 402kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<16:06, 567kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<13:16, 685kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<11:12, 811kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<08:19, 1.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<05:53, 1.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<17:13, 524kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<13:01, 692kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<09:19, 964kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<08:30, 1.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<07:50, 1.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:56, 1.50MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<04:14, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<23:35, 376kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<17:26, 509kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<12:22, 716kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<10:38, 829kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<09:17, 948kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<06:58, 1.26MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<04:57, 1.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<14:27, 605kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<11:02, 791kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<07:56, 1.10MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<07:28, 1.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<07:03, 1.23MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:20, 1.62MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<03:50, 2.24MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<14:30, 593kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<11:03, 778kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<07:56, 1.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<07:27, 1.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<06:07, 1.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:30, 1.89MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:02, 1.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:18, 1.59MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:09, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<02:59, 2.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<26:24, 318kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<19:22, 433kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<13:42, 610kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<11:26, 728kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<09:45, 854kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<07:12, 1.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<05:06, 1.62MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<09:10, 900kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<07:09, 1.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<05:13, 1.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:28, 1.50MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:33, 1.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<04:14, 1.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:03, 2.65MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<06:26, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<05:21, 1.51MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<03:54, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:34, 1.76MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:53, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<03:50, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:46, 2.87MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<19:51, 402kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<14:36, 546kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<10:24, 764kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<09:03, 874kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<07:59, 989kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<06:00, 1.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<04:16, 1.83MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<27:34, 284kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<20:06, 390kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<14:12, 550kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<11:39, 667kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<09:52, 787kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<07:15, 1.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:46<05:10, 1.49MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<06:27, 1.19MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:19, 1.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<03:53, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:25, 1.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:53, 1.96MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<02:54, 2.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:47, 2.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:18, 1.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:25, 2.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<02:27, 3.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<12:31, 599kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<09:32, 785kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<06:49, 1.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<06:25, 1.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<06:03, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:34, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<03:17, 2.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:41, 1.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:45, 1.55MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<03:30, 2.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:05, 1.78MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:23, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:25, 2.13MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:00<02:27, 2.94MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<08:23, 860kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<06:31, 1.11MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<04:44, 1.52MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:54, 1.46MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:00, 1.43MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:49, 1.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:45, 2.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:55, 1.44MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:09, 1.70MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:05, 2.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<03:46, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<04:07, 1.70MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:12, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<02:18, 3.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<06:01, 1.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:57, 1.40MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<03:36, 1.91MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:03, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:13, 1.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:15, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<02:21, 2.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:45, 1.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:43, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<03:27, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:57, 1.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:15, 1.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:16, 2.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<02:24, 2.79MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:51, 1.73MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:24, 1.96MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:32, 2.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:15, 2.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:58, 2.21MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:15, 2.91MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:01, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:29, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:47, 2.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:22<02:01, 3.20MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<10:53, 593kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<08:16, 779kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<05:54, 1.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<05:36, 1.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<05:19, 1.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:01, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:52, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:22, 1.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:19, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<03:11, 1.97MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<03:35, 1.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<03:49, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:00, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<02:09, 2.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<13:04, 473kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<09:41, 637kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<06:55, 888kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<06:14, 980kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<05:42, 1.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:15, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<03:04, 1.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<02:31, 2.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<5:11:43, 19.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<3:38:34, 27.6kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<2:32:28, 39.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<1:47:31, 55.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<1:16:25, 78.0kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<53:38, 111kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<37:24, 158kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<28:54, 204kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<20:50, 283kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<14:41, 400kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<11:31, 506kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<09:17, 628kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<06:45, 861kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<04:45, 1.21MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<06:44, 855kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:19, 1.08MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:51, 1.48MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:57, 1.44MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:54, 1.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:01, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<02:10, 2.59MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<30:57, 182kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<22:08, 254kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<15:35, 359kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<10:56, 508kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<13:46, 403kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<10:51, 511kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<07:50, 707kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<05:33, 990kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<05:27, 1.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:23, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:12, 1.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:26, 1.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:36, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:48, 1.92MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<02:00, 2.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<08:07, 658kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<06:14, 855kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<04:28, 1.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<04:17, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<04:09, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:11, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:58<02:17, 2.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<09:48, 531kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<07:18, 711kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<05:14, 988kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<04:47, 1.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<04:28, 1.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:21, 1.52MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<02:24, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<04:18, 1.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:32, 1.43MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:35, 1.95MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:55, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:08, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:24, 2.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<01:44, 2.83MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:33, 1.38MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:00, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:13, 2.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:38, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:51, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:15, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<01:37, 2.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<07:09, 668kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<05:31, 866kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:57, 1.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:48, 1.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:39, 1.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:46, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<01:59, 2.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:36, 1.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:01, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:13, 2.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:34, 1.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:45, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:10, 2.10MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:33, 2.90MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<06:45, 668kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<05:12, 865kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:43, 1.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:35, 1.24MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:26, 1.29MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:36, 1.70MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:51, 2.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:56, 1.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:08, 1.39MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:19, 1.87MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<01:40, 2.57MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<06:43, 640kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<05:39, 761kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<04:10, 1.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:26<02:56, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<07:21, 575kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<05:31, 765kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:58, 1.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<02:49, 1.48MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<06:33, 634kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<05:27, 761kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:59, 1.04MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<02:49, 1.45MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<04:23, 932kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:30, 1.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:32, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:39, 1.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:40, 1.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:02, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:27, 2.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<03:20, 1.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:45, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:00, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:15, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:26, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:54, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:38<01:21, 2.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<06:43, 568kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<05:05, 749kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:38, 1.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<03:23, 1.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<03:09, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:24, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:42, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<06:50, 538kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<05:10, 710kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:40, 992kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<03:21, 1.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:43, 1.32MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:59, 1.80MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:12, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:17, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:47, 1.97MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:16, 2.72MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<06:13, 558kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<04:43, 734kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:22, 1.02MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<03:05, 1.10MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:52, 1.18MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:09, 1.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:32, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:20, 1.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:59, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:27, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:44, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:33, 2.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:55<01:09, 2.80MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:31, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:23, 2.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:03, 3.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:25, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:41, 1.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:18, 2.36MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<00:58, 3.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:35, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:23, 2.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:02, 2.89MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:23, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:37, 1.84MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:16, 2.34MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<00:56, 3.15MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:31, 1.91MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:22, 2.12MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:00, 2.84MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:20, 2.11MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:32, 1.84MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:13, 2.31MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<00:52, 3.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<08:37, 322kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<06:19, 438kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<04:27, 616kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<03:40, 735kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<03:08, 860kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<02:19, 1.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:37, 1.62MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<20:01, 132kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<14:16, 184kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<09:57, 262kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<07:26, 345kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<05:43, 448kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<04:05, 623kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<02:51, 880kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<03:41, 677kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<02:47, 892kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<02:02, 1.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<01:26, 1.70MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<04:28, 542kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<03:39, 662kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<02:40, 900kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:19<01:52, 1.26MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<05:01, 469kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<03:45, 626kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<02:39, 877kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<02:20, 976kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<02:07, 1.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:34, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<01:06, 2.01MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<02:16, 979kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:47, 1.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:17, 1.70MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:25<00:55, 2.33MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<03:25, 629kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<02:51, 750kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<02:06, 1.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:27<01:28, 1.42MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<04:19, 482kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<03:13, 643kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<02:17, 896kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<02:01, 995kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:50, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:21, 1.46MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<00:58, 2.02MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:18, 1.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:07, 1.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:49, 2.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:58, 1.91MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:03, 1.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:49, 2.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<00:35, 3.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:18, 1.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:05, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:48, 2.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:56, 1.84MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:02, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:48, 2.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:39<00:34, 2.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<03:01, 550kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<02:17, 725kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:37, 1.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:27, 1.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:21, 1.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:01, 1.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:43<00:42, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<11:26, 133kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<08:08, 187kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<05:38, 266kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<04:11, 349kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<03:13, 451kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<02:19, 622kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<01:35, 880kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<03:12, 434kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<02:22, 581kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<01:40, 813kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:26, 921kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:16, 1.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:56, 1.38MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:51<00:39, 1.92MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<10:42, 117kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<07:33, 165kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<05:13, 234kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<03:48, 311kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<02:54, 405kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<02:04, 561kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:55<01:24, 793kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<02:22, 468kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:46, 624kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<01:14, 874kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:04, 975kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:57, 1.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:42, 1.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<00:29, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:44, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:37, 1.56MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:26, 2.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:30, 1.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:32, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:25, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:17, 2.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:56, 882kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:44, 1.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:31, 1.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:31, 1.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:31, 1.45MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:23, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:16, 2.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:37, 1.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:30, 1.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:21, 1.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:22, 1.66MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:19, 1.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:14, 2.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:16, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:19, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:14, 2.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:09, 3.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:20, 1.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:17, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:12, 2.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:13, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:11, 2.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:08, 2.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:10, 2.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:11, 1.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:08, 2.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:05, 3.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<01:25, 198kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<01:00, 275kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:38, 389kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:26, 494kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:20, 614kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:14, 844kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:08, 1.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:09, 905kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:07, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:03, 1.49MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.47MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.91MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.65MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 909kB/s] .vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<123:16:02,  1.11s/it]  0%|          | 901/400000 [00:01<86:05:47,  1.29it/s]  0%|          | 1827/400000 [00:01<60:07:53,  1.84it/s]  1%|          | 2787/400000 [00:01<41:59:38,  2.63it/s]  1%|          | 3743/400000 [00:01<29:19:42,  3.75it/s]  1%|          | 4642/400000 [00:01<20:29:13,  5.36it/s]  1%|▏         | 5548/400000 [00:01<14:18:42,  7.66it/s]  2%|▏         | 6367/400000 [00:01<10:00:05, 10.93it/s]  2%|▏         | 7355/400000 [00:01<6:59:12, 15.61it/s]   2%|▏         | 8341/400000 [00:02<4:52:54, 22.29it/s]  2%|▏         | 9263/400000 [00:02<3:24:45, 31.80it/s]  3%|▎         | 10229/400000 [00:02<2:23:10, 45.37it/s]  3%|▎         | 11237/400000 [00:02<1:40:09, 64.69it/s]  3%|▎         | 12209/400000 [00:02<1:10:08, 92.15it/s]  3%|▎         | 13165/400000 [00:02<49:10, 131.09it/s]   4%|▎         | 14116/400000 [00:02<34:33, 186.08it/s]  4%|▍         | 15056/400000 [00:02<24:20, 263.59it/s]  4%|▍         | 15988/400000 [00:02<17:12, 371.75it/s]  4%|▍         | 16903/400000 [00:02<12:14, 521.81it/s]  4%|▍         | 17828/400000 [00:03<08:45, 727.82it/s]  5%|▍         | 18742/400000 [00:03<06:19, 1004.87it/s]  5%|▍         | 19693/400000 [00:03<04:36, 1373.32it/s]  5%|▌         | 20615/400000 [00:03<03:25, 1842.44it/s]  5%|▌         | 21533/400000 [00:03<02:36, 2418.43it/s]  6%|▌         | 22498/400000 [00:03<02:01, 3119.66it/s]  6%|▌         | 23425/400000 [00:03<01:37, 3878.99it/s]  6%|▌         | 24378/400000 [00:03<01:19, 4717.97it/s]  6%|▋         | 25314/400000 [00:03<01:07, 5541.91it/s]  7%|▋         | 26245/400000 [00:03<00:59, 6303.83it/s]  7%|▋         | 27193/400000 [00:04<00:53, 7007.42it/s]  7%|▋         | 28129/400000 [00:04<00:49, 7470.25it/s]  7%|▋         | 29072/400000 [00:04<00:46, 7966.06it/s]  7%|▋         | 29999/400000 [00:04<00:44, 8284.82it/s]  8%|▊         | 30922/400000 [00:04<00:44, 8342.89it/s]  8%|▊         | 31823/400000 [00:04<00:43, 8383.20it/s]  8%|▊         | 32714/400000 [00:04<00:43, 8533.94it/s]  8%|▊         | 33645/400000 [00:04<00:41, 8751.55it/s]  9%|▊         | 34545/400000 [00:04<00:42, 8655.42it/s]  9%|▉         | 35485/400000 [00:04<00:41, 8864.29it/s]  9%|▉         | 36437/400000 [00:05<00:40, 9049.47it/s]  9%|▉         | 37353/400000 [00:05<00:40, 8999.49it/s] 10%|▉         | 38330/400000 [00:05<00:39, 9215.58it/s] 10%|▉         | 39265/400000 [00:05<00:38, 9253.24it/s] 10%|█         | 40240/400000 [00:05<00:38, 9395.75it/s] 10%|█         | 41193/400000 [00:05<00:38, 9433.81it/s] 11%|█         | 42140/400000 [00:05<00:38, 9276.99it/s] 11%|█         | 43071/400000 [00:05<00:38, 9196.83it/s] 11%|█         | 44051/400000 [00:05<00:37, 9367.66it/s] 11%|█▏        | 45057/400000 [00:05<00:37, 9565.04it/s] 12%|█▏        | 46016/400000 [00:06<00:37, 9439.37it/s] 12%|█▏        | 46963/400000 [00:06<00:37, 9385.75it/s] 12%|█▏        | 47928/400000 [00:06<00:37, 9462.16it/s] 12%|█▏        | 48894/400000 [00:06<00:36, 9519.98it/s] 12%|█▏        | 49912/400000 [00:06<00:36, 9708.25it/s] 13%|█▎        | 50918/400000 [00:06<00:35, 9808.45it/s] 13%|█▎        | 51901/400000 [00:06<00:36, 9645.90it/s] 13%|█▎        | 52889/400000 [00:06<00:35, 9712.09it/s] 13%|█▎        | 53862/400000 [00:06<00:35, 9646.81it/s] 14%|█▎        | 54828/400000 [00:07<00:36, 9514.09it/s] 14%|█▍        | 55789/400000 [00:07<00:36, 9541.12it/s] 14%|█▍        | 56744/400000 [00:07<00:36, 9341.02it/s] 14%|█▍        | 57707/400000 [00:07<00:36, 9424.14it/s] 15%|█▍        | 58651/400000 [00:07<00:36, 9414.22it/s] 15%|█▍        | 59594/400000 [00:07<00:36, 9390.51it/s] 15%|█▌        | 60551/400000 [00:07<00:35, 9442.59it/s] 15%|█▌        | 61496/400000 [00:07<00:37, 9067.63it/s] 16%|█▌        | 62469/400000 [00:07<00:36, 9254.66it/s] 16%|█▌        | 63398/400000 [00:07<00:36, 9263.70it/s] 16%|█▌        | 64379/400000 [00:08<00:35, 9419.38it/s] 16%|█▋        | 65341/400000 [00:08<00:35, 9476.80it/s] 17%|█▋        | 66291/400000 [00:08<00:35, 9351.76it/s] 17%|█▋        | 67280/400000 [00:08<00:35, 9504.74it/s] 17%|█▋        | 68259/400000 [00:08<00:34, 9587.37it/s] 17%|█▋        | 69220/400000 [00:08<00:34, 9591.25it/s] 18%|█▊        | 70225/400000 [00:08<00:33, 9724.39it/s] 18%|█▊        | 71199/400000 [00:08<00:33, 9689.73it/s] 18%|█▊        | 72169/400000 [00:08<00:34, 9518.30it/s] 18%|█▊        | 73132/400000 [00:08<00:34, 9549.34it/s] 19%|█▊        | 74114/400000 [00:09<00:33, 9628.41it/s] 19%|█▉        | 75097/400000 [00:09<00:33, 9687.05it/s] 19%|█▉        | 76067/400000 [00:09<00:34, 9362.58it/s] 19%|█▉        | 77037/400000 [00:09<00:34, 9460.77it/s] 20%|█▉        | 78013/400000 [00:09<00:33, 9546.03it/s] 20%|█▉        | 78970/400000 [00:09<00:34, 9346.58it/s] 20%|█▉        | 79907/400000 [00:09<00:34, 9205.27it/s] 20%|██        | 80830/400000 [00:09<00:34, 9159.26it/s] 20%|██        | 81802/400000 [00:09<00:34, 9319.61it/s] 21%|██        | 82736/400000 [00:09<00:34, 9281.90it/s] 21%|██        | 83666/400000 [00:10<00:34, 9249.90it/s] 21%|██        | 84635/400000 [00:10<00:33, 9375.28it/s] 21%|██▏       | 85577/400000 [00:10<00:33, 9387.55it/s] 22%|██▏       | 86561/400000 [00:10<00:32, 9517.30it/s] 22%|██▏       | 87553/400000 [00:10<00:32, 9633.11it/s] 22%|██▏       | 88559/400000 [00:10<00:31, 9754.90it/s] 22%|██▏       | 89567/400000 [00:10<00:31, 9849.20it/s] 23%|██▎       | 90553/400000 [00:10<00:32, 9590.12it/s] 23%|██▎       | 91515/400000 [00:10<00:32, 9532.10it/s] 23%|██▎       | 92503/400000 [00:10<00:31, 9632.01it/s] 23%|██▎       | 93496/400000 [00:11<00:31, 9716.88it/s] 24%|██▎       | 94472/400000 [00:11<00:31, 9728.82it/s] 24%|██▍       | 95446/400000 [00:11<00:33, 9204.06it/s] 24%|██▍       | 96407/400000 [00:11<00:32, 9321.41it/s] 24%|██▍       | 97363/400000 [00:11<00:32, 9391.53it/s] 25%|██▍       | 98313/400000 [00:11<00:32, 9422.67it/s] 25%|██▍       | 99258/400000 [00:11<00:32, 9181.79it/s] 25%|██▌       | 100180/400000 [00:11<00:33, 9061.26it/s] 25%|██▌       | 101103/400000 [00:11<00:32, 9109.06it/s] 26%|██▌       | 102049/400000 [00:12<00:32, 9210.07it/s] 26%|██▌       | 102972/400000 [00:12<00:32, 9197.90it/s] 26%|██▌       | 103893/400000 [00:12<00:32, 9177.17it/s] 26%|██▌       | 104849/400000 [00:12<00:31, 9288.76it/s] 26%|██▋       | 105811/400000 [00:12<00:31, 9385.60it/s] 27%|██▋       | 106751/400000 [00:12<00:31, 9367.29it/s] 27%|██▋       | 107707/400000 [00:12<00:31, 9422.49it/s] 27%|██▋       | 108650/400000 [00:12<00:31, 9389.36it/s] 27%|██▋       | 109590/400000 [00:12<00:32, 9051.25it/s] 28%|██▊       | 110499/400000 [00:12<00:32, 9030.05it/s] 28%|██▊       | 111457/400000 [00:13<00:31, 9188.23it/s] 28%|██▊       | 112378/400000 [00:13<00:31, 9134.21it/s] 28%|██▊       | 113321/400000 [00:13<00:31, 9214.81it/s] 29%|██▊       | 114244/400000 [00:13<00:31, 9093.72it/s] 29%|██▉       | 115172/400000 [00:13<00:31, 9145.61it/s] 29%|██▉       | 116088/400000 [00:13<00:31, 9131.26it/s] 29%|██▉       | 117002/400000 [00:13<00:31, 8988.62it/s] 29%|██▉       | 117941/400000 [00:13<00:30, 9105.31it/s] 30%|██▉       | 118853/400000 [00:13<00:30, 9089.01it/s] 30%|██▉       | 119763/400000 [00:13<00:31, 8982.07it/s] 30%|███       | 120663/400000 [00:14<00:31, 8756.29it/s] 30%|███       | 121561/400000 [00:14<00:31, 8821.79it/s] 31%|███       | 122490/400000 [00:14<00:30, 8955.12it/s] 31%|███       | 123387/400000 [00:14<00:31, 8843.01it/s] 31%|███       | 124305/400000 [00:14<00:30, 8940.71it/s] 31%|███▏      | 125231/400000 [00:14<00:30, 9031.93it/s] 32%|███▏      | 126136/400000 [00:14<00:30, 9035.22it/s] 32%|███▏      | 127088/400000 [00:14<00:29, 9175.36it/s] 32%|███▏      | 128007/400000 [00:14<00:29, 9123.99it/s] 32%|███▏      | 129028/400000 [00:14<00:28, 9423.75it/s] 33%|███▎      | 130004/400000 [00:15<00:28, 9518.74it/s] 33%|███▎      | 131028/400000 [00:15<00:27, 9723.61it/s] 33%|███▎      | 132025/400000 [00:15<00:27, 9793.94it/s] 33%|███▎      | 133007/400000 [00:15<00:27, 9735.33it/s] 33%|███▎      | 133993/400000 [00:15<00:27, 9771.09it/s] 34%|███▍      | 135021/400000 [00:15<00:26, 9917.19it/s] 34%|███▍      | 136015/400000 [00:15<00:27, 9744.35it/s] 34%|███▍      | 137033/400000 [00:15<00:26, 9870.10it/s] 35%|███▍      | 138022/400000 [00:15<00:27, 9611.44it/s] 35%|███▍      | 139029/400000 [00:15<00:26, 9742.72it/s] 35%|███▌      | 140036/400000 [00:16<00:26, 9836.46it/s] 35%|███▌      | 141033/400000 [00:16<00:26, 9875.37it/s] 36%|███▌      | 142040/400000 [00:16<00:25, 9930.48it/s] 36%|███▌      | 143035/400000 [00:16<00:26, 9703.22it/s] 36%|███▌      | 144058/400000 [00:16<00:25, 9854.77it/s] 36%|███▋      | 145064/400000 [00:16<00:25, 9915.10it/s] 37%|███▋      | 146062/400000 [00:16<00:25, 9934.43it/s] 37%|███▋      | 147057/400000 [00:16<00:25, 9814.69it/s] 37%|███▋      | 148040/400000 [00:16<00:25, 9712.81it/s] 37%|███▋      | 149013/400000 [00:16<00:25, 9710.89it/s] 38%|███▊      | 150008/400000 [00:17<00:25, 9781.18it/s] 38%|███▊      | 150987/400000 [00:17<00:25, 9701.19it/s] 38%|███▊      | 151958/400000 [00:17<00:25, 9679.78it/s] 38%|███▊      | 152927/400000 [00:17<00:25, 9549.15it/s] 38%|███▊      | 153938/400000 [00:17<00:25, 9707.50it/s] 39%|███▊      | 154962/400000 [00:17<00:24, 9858.80it/s] 39%|███▉      | 155950/400000 [00:17<00:25, 9722.59it/s] 39%|███▉      | 156924/400000 [00:17<00:25, 9679.23it/s] 39%|███▉      | 157893/400000 [00:17<00:25, 9463.67it/s] 40%|███▉      | 158859/400000 [00:18<00:25, 9519.73it/s] 40%|███▉      | 159826/400000 [00:18<00:25, 9564.12it/s] 40%|████      | 160820/400000 [00:18<00:24, 9672.66it/s] 40%|████      | 161827/400000 [00:18<00:24, 9785.94it/s] 41%|████      | 162807/400000 [00:18<00:24, 9500.23it/s] 41%|████      | 163802/400000 [00:18<00:24, 9630.74it/s] 41%|████      | 164813/400000 [00:18<00:24, 9767.55it/s] 41%|████▏     | 165820/400000 [00:18<00:23, 9854.48it/s] 42%|████▏     | 166808/400000 [00:18<00:24, 9658.72it/s] 42%|████▏     | 167776/400000 [00:18<00:24, 9391.34it/s] 42%|████▏     | 168719/400000 [00:19<00:24, 9372.79it/s] 42%|████▏     | 169659/400000 [00:19<00:24, 9316.03it/s] 43%|████▎     | 170618/400000 [00:19<00:24, 9396.20it/s] 43%|████▎     | 171608/400000 [00:19<00:23, 9539.77it/s] 43%|████▎     | 172564/400000 [00:19<00:24, 9450.68it/s] 43%|████▎     | 173527/400000 [00:19<00:23, 9502.01it/s] 44%|████▎     | 174529/400000 [00:19<00:23, 9650.78it/s] 44%|████▍     | 175542/400000 [00:19<00:22, 9789.71it/s] 44%|████▍     | 176526/400000 [00:19<00:22, 9802.43it/s] 44%|████▍     | 177508/400000 [00:19<00:23, 9561.34it/s] 45%|████▍     | 178531/400000 [00:20<00:22, 9752.10it/s] 45%|████▍     | 179553/400000 [00:20<00:22, 9886.82it/s] 45%|████▌     | 180563/400000 [00:20<00:22, 9945.81it/s] 45%|████▌     | 181560/400000 [00:20<00:22, 9620.63it/s] 46%|████▌     | 182526/400000 [00:20<00:23, 9261.82it/s] 46%|████▌     | 183465/400000 [00:20<00:23, 9298.89it/s] 46%|████▌     | 184474/400000 [00:20<00:22, 9521.13it/s] 46%|████▋     | 185431/400000 [00:20<00:22, 9454.35it/s] 47%|████▋     | 186417/400000 [00:20<00:22, 9570.30it/s] 47%|████▋     | 187377/400000 [00:20<00:22, 9488.19it/s] 47%|████▋     | 188368/400000 [00:21<00:22, 9610.66it/s] 47%|████▋     | 189336/400000 [00:21<00:21, 9629.40it/s] 48%|████▊     | 190301/400000 [00:21<00:22, 9507.07it/s] 48%|████▊     | 191253/400000 [00:21<00:22, 9459.70it/s] 48%|████▊     | 192200/400000 [00:21<00:22, 9378.46it/s] 48%|████▊     | 193179/400000 [00:21<00:21, 9496.95it/s] 49%|████▊     | 194159/400000 [00:21<00:21, 9585.87it/s] 49%|████▉     | 195119/400000 [00:21<00:21, 9569.63it/s] 49%|████▉     | 196077/400000 [00:21<00:21, 9522.60it/s] 49%|████▉     | 197030/400000 [00:22<00:22, 9220.38it/s] 49%|████▉     | 197969/400000 [00:22<00:21, 9269.56it/s] 50%|████▉     | 198937/400000 [00:22<00:21, 9385.97it/s] 50%|████▉     | 199886/400000 [00:22<00:21, 9414.08it/s] 50%|█████     | 200829/400000 [00:22<00:21, 9205.17it/s] 50%|█████     | 201752/400000 [00:22<00:22, 8741.07it/s] 51%|█████     | 202633/400000 [00:22<00:22, 8748.29it/s] 51%|█████     | 203585/400000 [00:22<00:21, 8964.75it/s] 51%|█████     | 204527/400000 [00:22<00:21, 9094.91it/s] 51%|█████▏    | 205497/400000 [00:22<00:20, 9267.97it/s] 52%|█████▏    | 206428/400000 [00:23<00:21, 9124.05it/s] 52%|█████▏    | 207383/400000 [00:23<00:20, 9245.89it/s] 52%|█████▏    | 208341/400000 [00:23<00:20, 9340.63it/s] 52%|█████▏    | 209310/400000 [00:23<00:20, 9441.78it/s] 53%|█████▎    | 210271/400000 [00:23<00:19, 9490.59it/s] 53%|█████▎    | 211222/400000 [00:23<00:20, 9143.77it/s] 53%|█████▎    | 212140/400000 [00:23<00:20, 8975.91it/s] 53%|█████▎    | 213061/400000 [00:23<00:20, 9042.66it/s] 53%|█████▎    | 213972/400000 [00:23<00:20, 9060.94it/s] 54%|█████▎    | 214891/400000 [00:23<00:20, 9097.67it/s] 54%|█████▍    | 215802/400000 [00:24<00:20, 8959.92it/s] 54%|█████▍    | 216806/400000 [00:24<00:19, 9257.69it/s] 54%|█████▍    | 217843/400000 [00:24<00:19, 9563.51it/s] 55%|█████▍    | 218818/400000 [00:24<00:18, 9618.20it/s] 55%|█████▍    | 219819/400000 [00:24<00:18, 9729.72it/s] 55%|█████▌    | 220795/400000 [00:24<00:18, 9676.26it/s] 55%|█████▌    | 221825/400000 [00:24<00:18, 9853.39it/s] 56%|█████▌    | 222844/400000 [00:24<00:17, 9949.87it/s] 56%|█████▌    | 223841/400000 [00:24<00:17, 9884.16it/s] 56%|█████▌    | 224831/400000 [00:24<00:17, 9829.27it/s] 56%|█████▋    | 225815/400000 [00:25<00:18, 9499.55it/s] 57%|█████▋    | 226769/400000 [00:25<00:18, 9313.71it/s] 57%|█████▋    | 227772/400000 [00:25<00:18, 9515.17it/s] 57%|█████▋    | 228765/400000 [00:25<00:17, 9634.29it/s] 57%|█████▋    | 229788/400000 [00:25<00:17, 9803.44it/s] 58%|█████▊    | 230771/400000 [00:25<00:17, 9730.85it/s] 58%|█████▊    | 231794/400000 [00:25<00:17, 9874.41it/s] 58%|█████▊    | 232784/400000 [00:25<00:16, 9852.13it/s] 58%|█████▊    | 233771/400000 [00:25<00:16, 9816.13it/s] 59%|█████▊    | 234802/400000 [00:26<00:16, 9958.08it/s] 59%|█████▉    | 235799/400000 [00:26<00:16, 9820.12it/s] 59%|█████▉    | 236800/400000 [00:26<00:16, 9876.07it/s] 59%|█████▉    | 237838/400000 [00:26<00:16, 10021.99it/s] 60%|█████▉    | 238866/400000 [00:26<00:15, 10097.37it/s] 60%|█████▉    | 239896/400000 [00:26<00:15, 10155.69it/s] 60%|██████    | 240913/400000 [00:26<00:15, 9944.53it/s]  60%|██████    | 241909/400000 [00:26<00:15, 9894.93it/s] 61%|██████    | 242900/400000 [00:26<00:16, 9525.27it/s] 61%|██████    | 243879/400000 [00:26<00:16, 9600.22it/s] 61%|██████    | 244850/400000 [00:27<00:16, 9632.48it/s] 61%|██████▏   | 245816/400000 [00:27<00:16, 9533.70it/s] 62%|██████▏   | 246804/400000 [00:27<00:15, 9634.98it/s] 62%|██████▏   | 247774/400000 [00:27<00:15, 9651.52it/s] 62%|██████▏   | 248757/400000 [00:27<00:15, 9703.32it/s] 62%|██████▏   | 249745/400000 [00:27<00:15, 9753.02it/s] 63%|██████▎   | 250721/400000 [00:27<00:15, 9577.25it/s] 63%|██████▎   | 251726/400000 [00:27<00:15, 9712.31it/s] 63%|██████▎   | 252699/400000 [00:27<00:15, 9429.71it/s] 63%|██████▎   | 253645/400000 [00:27<00:15, 9373.63it/s] 64%|██████▎   | 254648/400000 [00:28<00:15, 9561.10it/s] 64%|██████▍   | 255607/400000 [00:28<00:15, 9427.81it/s] 64%|██████▍   | 256613/400000 [00:28<00:14, 9606.50it/s] 64%|██████▍   | 257576/400000 [00:28<00:15, 9437.57it/s] 65%|██████▍   | 258548/400000 [00:28<00:14, 9518.05it/s] 65%|██████▍   | 259567/400000 [00:28<00:14, 9708.25it/s] 65%|██████▌   | 260540/400000 [00:28<00:14, 9654.52it/s] 65%|██████▌   | 261537/400000 [00:28<00:14, 9746.66it/s] 66%|██████▌   | 262520/400000 [00:28<00:14, 9768.92it/s] 66%|██████▌   | 263534/400000 [00:28<00:13, 9876.76it/s] 66%|██████▌   | 264527/400000 [00:29<00:13, 9892.32it/s] 66%|██████▋   | 265517/400000 [00:29<00:14, 9571.51it/s] 67%|██████▋   | 266508/400000 [00:29<00:13, 9668.37it/s] 67%|██████▋   | 267477/400000 [00:29<00:13, 9633.01it/s] 67%|██████▋   | 268456/400000 [00:29<00:13, 9676.94it/s] 67%|██████▋   | 269433/400000 [00:29<00:13, 9703.29it/s] 68%|██████▊   | 270405/400000 [00:29<00:13, 9449.99it/s] 68%|██████▊   | 271353/400000 [00:29<00:13, 9405.18it/s] 68%|██████▊   | 272295/400000 [00:29<00:13, 9382.36it/s] 68%|██████▊   | 273235/400000 [00:29<00:13, 9365.53it/s] 69%|██████▊   | 274175/400000 [00:30<00:13, 9374.74it/s] 69%|██████▉   | 275113/400000 [00:30<00:13, 9022.60it/s] 69%|██████▉   | 276073/400000 [00:30<00:13, 9186.26it/s] 69%|██████▉   | 277062/400000 [00:30<00:13, 9386.23it/s] 70%|██████▉   | 278052/400000 [00:30<00:12, 9534.47it/s] 70%|██████▉   | 279038/400000 [00:30<00:12, 9627.14it/s] 70%|███████   | 280003/400000 [00:30<00:12, 9416.15it/s] 70%|███████   | 280948/400000 [00:30<00:12, 9414.31it/s] 70%|███████   | 281924/400000 [00:30<00:12, 9513.44it/s] 71%|███████   | 282895/400000 [00:31<00:12, 9570.76it/s] 71%|███████   | 283854/400000 [00:31<00:12, 9518.24it/s] 71%|███████   | 284807/400000 [00:31<00:12, 9345.38it/s] 71%|███████▏  | 285779/400000 [00:31<00:12, 9452.33it/s] 72%|███████▏  | 286726/400000 [00:31<00:12, 9408.98it/s] 72%|███████▏  | 287668/400000 [00:31<00:12, 9343.84it/s] 72%|███████▏  | 288604/400000 [00:31<00:12, 9170.29it/s] 72%|███████▏  | 289523/400000 [00:31<00:12, 8508.43it/s] 73%|███████▎  | 290385/400000 [00:31<00:13, 8206.36it/s] 73%|███████▎  | 291216/400000 [00:31<00:13, 7824.19it/s] 73%|███████▎  | 292009/400000 [00:32<00:13, 7773.72it/s] 73%|███████▎  | 292935/400000 [00:32<00:13, 8165.61it/s] 73%|███████▎  | 293858/400000 [00:32<00:12, 8456.94it/s] 74%|███████▎  | 294814/400000 [00:32<00:12, 8760.03it/s] 74%|███████▍  | 295782/400000 [00:32<00:11, 9016.71it/s] 74%|███████▍  | 296736/400000 [00:32<00:11, 9165.14it/s] 74%|███████▍  | 297660/400000 [00:32<00:11, 9090.39it/s] 75%|███████▍  | 298620/400000 [00:32<00:10, 9236.23it/s] 75%|███████▍  | 299594/400000 [00:32<00:10, 9379.49it/s] 75%|███████▌  | 300576/400000 [00:33<00:10, 9505.10it/s] 75%|███████▌  | 301530/400000 [00:33<00:10, 9503.36it/s] 76%|███████▌  | 302483/400000 [00:33<00:10, 9414.76it/s] 76%|███████▌  | 303492/400000 [00:33<00:10, 9606.14it/s] 76%|███████▌  | 304475/400000 [00:33<00:09, 9671.19it/s] 76%|███████▋  | 305477/400000 [00:33<00:09, 9773.16it/s] 77%|███████▋  | 306483/400000 [00:33<00:09, 9856.35it/s] 77%|███████▋  | 307470/400000 [00:33<00:09, 9789.75it/s] 77%|███████▋  | 308450/400000 [00:33<00:09, 9782.53it/s] 77%|███████▋  | 309429/400000 [00:33<00:09, 9748.19it/s] 78%|███████▊  | 310429/400000 [00:34<00:09, 9821.76it/s] 78%|███████▊  | 311448/400000 [00:34<00:08, 9928.86it/s] 78%|███████▊  | 312442/400000 [00:34<00:08, 9904.31it/s] 78%|███████▊  | 313433/400000 [00:34<00:08, 9625.67it/s] 79%|███████▊  | 314442/400000 [00:34<00:08, 9758.62it/s] 79%|███████▉  | 315420/400000 [00:34<00:08, 9722.21it/s] 79%|███████▉  | 316403/400000 [00:34<00:08, 9752.96it/s] 79%|███████▉  | 317380/400000 [00:34<00:08, 9639.47it/s] 80%|███████▉  | 318345/400000 [00:34<00:09, 9062.71it/s] 80%|███████▉  | 319260/400000 [00:34<00:09, 8941.88it/s] 80%|████████  | 320172/400000 [00:35<00:08, 8992.01it/s] 80%|████████  | 321114/400000 [00:35<00:08, 9115.95it/s] 81%|████████  | 322029/400000 [00:35<00:08, 8905.27it/s] 81%|████████  | 322965/400000 [00:35<00:08, 9036.10it/s] 81%|████████  | 323974/400000 [00:35<00:08, 9326.60it/s] 81%|████████  | 324998/400000 [00:35<00:07, 9581.57it/s] 81%|████████▏ | 325997/400000 [00:35<00:07, 9699.80it/s] 82%|████████▏ | 326971/400000 [00:35<00:07, 9413.95it/s] 82%|████████▏ | 327946/400000 [00:35<00:07, 9510.18it/s] 82%|████████▏ | 328913/400000 [00:35<00:07, 9555.89it/s] 82%|████████▏ | 329909/400000 [00:36<00:07, 9673.14it/s] 83%|████████▎ | 330879/400000 [00:36<00:07, 9667.02it/s] 83%|████████▎ | 331848/400000 [00:36<00:07, 9523.66it/s] 83%|████████▎ | 332802/400000 [00:36<00:07, 9340.04it/s] 83%|████████▎ | 333777/400000 [00:36<00:07, 9458.15it/s] 84%|████████▎ | 334725/400000 [00:36<00:07, 9209.14it/s] 84%|████████▍ | 335649/400000 [00:36<00:06, 9196.61it/s] 84%|████████▍ | 336571/400000 [00:36<00:06, 9102.73it/s] 84%|████████▍ | 337527/400000 [00:36<00:06, 9234.35it/s] 85%|████████▍ | 338499/400000 [00:37<00:06, 9372.48it/s] 85%|████████▍ | 339461/400000 [00:37<00:06, 9444.41it/s] 85%|████████▌ | 340448/400000 [00:37<00:06, 9566.83it/s] 85%|████████▌ | 341406/400000 [00:37<00:06, 9480.24it/s] 86%|████████▌ | 342396/400000 [00:37<00:05, 9601.44it/s] 86%|████████▌ | 343358/400000 [00:37<00:05, 9606.51it/s] 86%|████████▌ | 344365/400000 [00:37<00:05, 9738.84it/s] 86%|████████▋ | 345366/400000 [00:37<00:05, 9817.18it/s] 87%|████████▋ | 346349/400000 [00:37<00:05, 9441.18it/s] 87%|████████▋ | 347315/400000 [00:37<00:05, 9503.74it/s] 87%|████████▋ | 348269/400000 [00:38<00:05, 9323.10it/s] 87%|████████▋ | 349241/400000 [00:38<00:05, 9438.11it/s] 88%|████████▊ | 350227/400000 [00:38<00:05, 9558.61it/s] 88%|████████▊ | 351185/400000 [00:38<00:05, 9187.83it/s] 88%|████████▊ | 352109/400000 [00:38<00:05, 9040.44it/s] 88%|████████▊ | 353017/400000 [00:38<00:05, 8997.53it/s] 88%|████████▊ | 353935/400000 [00:38<00:05, 9049.57it/s] 89%|████████▊ | 354894/400000 [00:38<00:04, 9203.35it/s] 89%|████████▉ | 355817/400000 [00:38<00:04, 8957.54it/s] 89%|████████▉ | 356716/400000 [00:38<00:04, 8932.92it/s] 89%|████████▉ | 357644/400000 [00:39<00:04, 9034.20it/s] 90%|████████▉ | 358550/400000 [00:39<00:04, 8989.23it/s] 90%|████████▉ | 359451/400000 [00:39<00:04, 8874.12it/s] 90%|█████████ | 360340/400000 [00:39<00:04, 8740.68it/s] 90%|█████████ | 361292/400000 [00:39<00:04, 8960.06it/s] 91%|█████████ | 362191/400000 [00:39<00:04, 8820.00it/s] 91%|█████████ | 363104/400000 [00:39<00:04, 8908.53it/s] 91%|█████████ | 363997/400000 [00:39<00:04, 8856.32it/s] 91%|█████████ | 364884/400000 [00:39<00:04, 8774.08it/s] 91%|█████████▏| 365772/400000 [00:39<00:03, 8804.48it/s] 92%|█████████▏| 366664/400000 [00:40<00:03, 8837.74it/s] 92%|█████████▏| 367604/400000 [00:40<00:03, 8997.33it/s] 92%|█████████▏| 368520/400000 [00:40<00:03, 9044.12it/s] 92%|█████████▏| 369426/400000 [00:40<00:03, 8905.04it/s] 93%|█████████▎| 370324/400000 [00:40<00:03, 8925.31it/s] 93%|█████████▎| 371222/400000 [00:40<00:03, 8940.04it/s] 93%|█████████▎| 372174/400000 [00:40<00:03, 9105.35it/s] 93%|█████████▎| 373096/400000 [00:40<00:02, 9138.56it/s] 94%|█████████▎| 374011/400000 [00:40<00:02, 9017.34it/s] 94%|█████████▎| 374947/400000 [00:40<00:02, 9116.63it/s] 94%|█████████▍| 375860/400000 [00:41<00:02, 9056.91it/s] 94%|█████████▍| 376767/400000 [00:41<00:02, 8971.35it/s] 94%|█████████▍| 377665/400000 [00:41<00:02, 8746.53it/s] 95%|█████████▍| 378542/400000 [00:41<00:02, 8486.34it/s] 95%|█████████▍| 379453/400000 [00:41<00:02, 8663.07it/s] 95%|█████████▌| 380388/400000 [00:41<00:02, 8856.36it/s] 95%|█████████▌| 381277/400000 [00:41<00:02, 8785.56it/s] 96%|█████████▌| 382158/400000 [00:41<00:02, 8435.77it/s] 96%|█████████▌| 383007/400000 [00:41<00:02, 8451.04it/s] 96%|█████████▌| 383997/400000 [00:42<00:01, 8836.65it/s] 96%|█████████▋| 385006/400000 [00:42<00:01, 9178.65it/s] 96%|█████████▋| 385992/400000 [00:42<00:01, 9372.38it/s] 97%|█████████▋| 386950/400000 [00:42<00:01, 9432.94it/s] 97%|█████████▋| 387899/400000 [00:42<00:01, 9264.20it/s] 97%|█████████▋| 388892/400000 [00:42<00:01, 9454.35it/s] 97%|█████████▋| 389857/400000 [00:42<00:01, 9509.83it/s] 98%|█████████▊| 390811/400000 [00:42<00:00, 9397.90it/s] 98%|█████████▊| 391753/400000 [00:42<00:00, 9104.35it/s] 98%|█████████▊| 392667/400000 [00:42<00:00, 8627.75it/s] 98%|█████████▊| 393601/400000 [00:43<00:00, 8827.82it/s] 99%|█████████▊| 394502/400000 [00:43<00:00, 8880.29it/s] 99%|█████████▉| 395447/400000 [00:43<00:00, 9041.83it/s] 99%|█████████▉| 396356/400000 [00:43<00:00, 8982.42it/s] 99%|█████████▉| 397258/400000 [00:43<00:00, 8751.72it/s]100%|█████████▉| 398163/400000 [00:43<00:00, 8838.48it/s]100%|█████████▉| 399100/400000 [00:43<00:00, 8991.14it/s]100%|█████████▉| 399999/400000 [00:43<00:00, 9135.68it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fed640ac160>, <torchtext.data.dataset.TabularDataset object at 0x7fed640ac2b0>, <torchtext.vocab.Vocab object at 0x7fed640ac1d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.24 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.24 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.24 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.96 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.96 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.89 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.89 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.58 file/s]2020-07-11 18:17:23.796713: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-11 18:17:23.800832: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-11 18:17:23.801031: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5644b6c7c000 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-11 18:17:23.801064: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3637248/9912422 [00:00<00:00, 34449275.45it/s]9920512it [00:00, 32596384.33it/s]                             
0it [00:00, ?it/s]32768it [00:00, 519842.02it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 154608.54it/s]1654784it [00:00, 11334680.87it/s]                         
0it [00:00, ?it/s]8192it [00:00, 153861.52it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

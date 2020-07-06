
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fa1a24f6048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fa1a24f6048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fa20e09d1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fa20e09d1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fa22c3e5ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fa22c3e5ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa1bb3cb620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa1bb3cb620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fa1bb3cb620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 479764.00it/s] 89%|████████▊ | 8773632/9912422 [00:00<00:01, 683765.23it/s]9920512it [00:00, 47382866.37it/s]                           
0it [00:00, ?it/s]32768it [00:00, 658961.56it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1004185.11it/s]1654784it [00:00, 11358145.09it/s]                           
0it [00:00, ?it/s]8192it [00:00, 221209.05it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa1b87bf978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa1a23ffc18>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fa1bb3cb268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fa1bb3cb268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fa1bb3cb268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:08,  2.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:08,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:07,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:07,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:07,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:06,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:06,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:05,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:05,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:46,  3.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:46,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:45,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:45,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:45,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:44,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:44,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:44,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:43,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:43,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:43,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:30,  4.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:30,  4.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:30,  4.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:30,  4.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:29,  4.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:29,  4.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:29,  4.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:29,  4.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:29,  4.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:28,  4.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:20,  6.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:20,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:20,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:20,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:19,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:19,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:19,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:19,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:19,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:19,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:13,  9.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:13,  9.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:13,  9.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:13,  9.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:13,  9.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:13,  9.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:13,  9.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:13,  9.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:12,  9.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:12,  9.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:12,  9.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:00<00:09, 12.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:09, 12.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:09, 12.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:09, 12.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:08, 12.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:08, 12.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:08, 12.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 22.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 22.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 22.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 22.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 22.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 22.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 22.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 22.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 22.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 22.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 22.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 29.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 29.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 29.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 29.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 29.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 29.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 29.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 29.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 29.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 29.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 29.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 36.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 36.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 36.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 36.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 36.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 36.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 36.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 36.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 36.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 36.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 36.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 45.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 45.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 45.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 45.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 45.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 45.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 45.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 45.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 53.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 53.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 61.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 63.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 63.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 63.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 63.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 63.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 63.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 63.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 63.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 63.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 63.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 64.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 64.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 64.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 64.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 64.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 64.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 64.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 64.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 64.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 64.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 66.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 66.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 65.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 66.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.33s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.33s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.33s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.49s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.33s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 66.74 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.49s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.49s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.09 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.49s/ url]
0 examples [00:00, ? examples/s]2020-07-06 18:09:25.209478: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-06 18:09:25.251475: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-06 18:09:25.252152: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bf639cbf30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-06 18:09:25.252183: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.83 examples/s]97 examples [00:00,  4.04 examples/s]197 examples [00:00,  5.76 examples/s]297 examples [00:00,  8.21 examples/s]404 examples [00:00, 11.69 examples/s]510 examples [00:00, 16.62 examples/s]600 examples [00:00, 23.55 examples/s]707 examples [00:01, 33.33 examples/s]813 examples [00:01, 46.98 examples/s]920 examples [00:01, 65.87 examples/s]1027 examples [00:01, 91.68 examples/s]1129 examples [00:01, 126.00 examples/s]1235 examples [00:01, 171.26 examples/s]1341 examples [00:01, 228.77 examples/s]1446 examples [00:01, 298.78 examples/s]1554 examples [00:01, 381.47 examples/s]1661 examples [00:01, 472.31 examples/s]1768 examples [00:02, 567.23 examples/s]1874 examples [00:02, 656.37 examples/s]1980 examples [00:02, 738.19 examples/s]2085 examples [00:02, 800.90 examples/s]2191 examples [00:02, 864.24 examples/s]2297 examples [00:02, 913.73 examples/s]2402 examples [00:02, 933.67 examples/s]2508 examples [00:02, 967.02 examples/s]2616 examples [00:02, 998.11 examples/s]2721 examples [00:02, 1013.04 examples/s]2826 examples [00:03, 1020.35 examples/s]2933 examples [00:03, 1032.90 examples/s]3039 examples [00:03, 1034.82 examples/s]3146 examples [00:03, 1044.63 examples/s]3252 examples [00:03, 1014.28 examples/s]3361 examples [00:03, 1033.95 examples/s]3469 examples [00:03, 1044.39 examples/s]3574 examples [00:03, 1019.66 examples/s]3677 examples [00:03, 1009.75 examples/s]3779 examples [00:04, 981.73 examples/s] 3880 examples [00:04, 987.54 examples/s]3980 examples [00:04, 956.94 examples/s]4083 examples [00:04, 976.00 examples/s]4184 examples [00:04, 983.90 examples/s]4284 examples [00:04, 987.88 examples/s]4383 examples [00:04, 984.34 examples/s]4484 examples [00:04, 991.51 examples/s]4589 examples [00:04, 1007.09 examples/s]4696 examples [00:04, 1025.15 examples/s]4801 examples [00:05, 1031.77 examples/s]4905 examples [00:05, 1033.14 examples/s]5010 examples [00:05, 1037.86 examples/s]5118 examples [00:05, 1047.69 examples/s]5227 examples [00:05, 1059.52 examples/s]5335 examples [00:05, 1063.12 examples/s]5443 examples [00:05, 1065.93 examples/s]5553 examples [00:05, 1073.21 examples/s]5661 examples [00:05, 1071.68 examples/s]5771 examples [00:05, 1077.06 examples/s]5881 examples [00:06, 1081.18 examples/s]5990 examples [00:06, 1079.21 examples/s]6098 examples [00:06, 1074.68 examples/s]6206 examples [00:06, 1075.25 examples/s]6314 examples [00:06, 1072.02 examples/s]6422 examples [00:06, 1025.92 examples/s]6533 examples [00:06, 1047.12 examples/s]6641 examples [00:06, 1056.43 examples/s]6750 examples [00:06, 1064.85 examples/s]6858 examples [00:06, 1069.30 examples/s]6966 examples [00:07, 1063.71 examples/s]7073 examples [00:07, 1059.44 examples/s]7181 examples [00:07, 1065.41 examples/s]7289 examples [00:07, 1067.57 examples/s]7396 examples [00:07, 1065.59 examples/s]7503 examples [00:07, 1058.56 examples/s]7609 examples [00:07, 1051.59 examples/s]7718 examples [00:07, 1062.31 examples/s]7827 examples [00:07, 1068.70 examples/s]7935 examples [00:07, 1070.76 examples/s]8043 examples [00:08, 1055.46 examples/s]8149 examples [00:08, 1056.77 examples/s]8256 examples [00:08, 1059.84 examples/s]8363 examples [00:08, 1058.01 examples/s]8469 examples [00:08, 1056.52 examples/s]8576 examples [00:08, 1060.07 examples/s]8683 examples [00:08, 1058.82 examples/s]8789 examples [00:08, 1031.52 examples/s]8895 examples [00:08, 1037.29 examples/s]8999 examples [00:09, 1018.35 examples/s]9106 examples [00:09, 1031.55 examples/s]9215 examples [00:09, 1046.89 examples/s]9325 examples [00:09, 1059.86 examples/s]9434 examples [00:09, 1066.61 examples/s]9543 examples [00:09, 1072.18 examples/s]9651 examples [00:09, 1057.21 examples/s]9758 examples [00:09, 1060.48 examples/s]9867 examples [00:09, 1067.70 examples/s]9974 examples [00:09, 1061.29 examples/s]10081 examples [00:10, 1018.62 examples/s]10189 examples [00:10, 1033.81 examples/s]10298 examples [00:10, 1048.92 examples/s]10405 examples [00:10, 1052.55 examples/s]10511 examples [00:10, 1053.37 examples/s]10617 examples [00:10, 1047.45 examples/s]10723 examples [00:10, 1048.53 examples/s]10834 examples [00:10, 1063.10 examples/s]10941 examples [00:10, 1059.82 examples/s]11048 examples [00:10, 1057.09 examples/s]11155 examples [00:11, 1060.14 examples/s]11262 examples [00:11, 1059.42 examples/s]11368 examples [00:11, 1045.00 examples/s]11473 examples [00:11, 1044.34 examples/s]11578 examples [00:11, 1045.81 examples/s]11685 examples [00:11, 1050.08 examples/s]11791 examples [00:11, 1031.24 examples/s]11898 examples [00:11, 1041.63 examples/s]12006 examples [00:11, 1050.51 examples/s]12113 examples [00:11, 1053.57 examples/s]12221 examples [00:12, 1059.89 examples/s]12328 examples [00:12, 1046.76 examples/s]12433 examples [00:12, 992.85 examples/s] 12533 examples [00:12, 986.78 examples/s]12633 examples [00:12, 985.22 examples/s]12732 examples [00:12, 942.68 examples/s]12827 examples [00:12, 941.81 examples/s]12933 examples [00:12, 973.36 examples/s]13042 examples [00:12, 1003.32 examples/s]13152 examples [00:13, 1030.05 examples/s]13257 examples [00:13, 1035.57 examples/s]13363 examples [00:13, 1041.47 examples/s]13472 examples [00:13, 1046.96 examples/s]13578 examples [00:13, 1049.31 examples/s]13684 examples [00:13, 1051.48 examples/s]13790 examples [00:13, 1029.65 examples/s]13896 examples [00:13, 1038.39 examples/s]14004 examples [00:13, 1046.98 examples/s]14109 examples [00:13, 1036.82 examples/s]14217 examples [00:14, 1047.48 examples/s]14322 examples [00:14, 1025.41 examples/s]14428 examples [00:14, 1033.50 examples/s]14534 examples [00:14, 1038.81 examples/s]14642 examples [00:14, 1048.89 examples/s]14747 examples [00:14, 1038.70 examples/s]14857 examples [00:14, 1055.69 examples/s]14969 examples [00:14, 1071.51 examples/s]15077 examples [00:14, 1066.34 examples/s]15186 examples [00:14, 1071.79 examples/s]15294 examples [00:15, 1067.90 examples/s]15401 examples [00:15, 1021.92 examples/s]15508 examples [00:15, 1033.94 examples/s]15612 examples [00:15, 1031.59 examples/s]15716 examples [00:15, 1028.10 examples/s]15820 examples [00:15, 1031.35 examples/s]15924 examples [00:15, 1030.06 examples/s]16028 examples [00:15, 1026.25 examples/s]16134 examples [00:15, 1035.95 examples/s]16239 examples [00:15, 1037.40 examples/s]16347 examples [00:16, 1048.26 examples/s]16457 examples [00:16, 1060.82 examples/s]16565 examples [00:16, 1066.33 examples/s]16678 examples [00:16, 1082.67 examples/s]16787 examples [00:16, 1080.51 examples/s]16896 examples [00:16, 1076.57 examples/s]17004 examples [00:16, 1059.03 examples/s]17111 examples [00:16, 1040.98 examples/s]17216 examples [00:16, 1033.94 examples/s]17324 examples [00:16, 1047.13 examples/s]17429 examples [00:17, 1035.84 examples/s]17533 examples [00:17, 1027.38 examples/s]17639 examples [00:17, 1034.63 examples/s]17749 examples [00:17, 1051.57 examples/s]17858 examples [00:17, 1061.32 examples/s]17965 examples [00:17, 1061.13 examples/s]18075 examples [00:17, 1071.35 examples/s]18183 examples [00:17, 1067.30 examples/s]18292 examples [00:17, 1073.95 examples/s]18400 examples [00:18, 1068.31 examples/s]18507 examples [00:18, 1055.70 examples/s]18613 examples [00:18, 1054.98 examples/s]18719 examples [00:18, 1054.63 examples/s]18826 examples [00:18, 1058.15 examples/s]18934 examples [00:18, 1062.28 examples/s]19042 examples [00:18, 1066.90 examples/s]19151 examples [00:18, 1073.57 examples/s]19259 examples [00:18, 1044.12 examples/s]19364 examples [00:18, 1024.50 examples/s]19468 examples [00:19, 1026.71 examples/s]19577 examples [00:19, 1042.68 examples/s]19687 examples [00:19, 1059.08 examples/s]19797 examples [00:19, 1068.94 examples/s]19905 examples [00:19, 1071.30 examples/s]20013 examples [00:19, 1022.67 examples/s]20121 examples [00:19, 1038.95 examples/s]20235 examples [00:19, 1065.06 examples/s]20344 examples [00:19, 1071.89 examples/s]20456 examples [00:19, 1084.91 examples/s]20565 examples [00:20, 1083.91 examples/s]20674 examples [00:20, 1081.39 examples/s]20783 examples [00:20, 1082.75 examples/s]20892 examples [00:20, 1074.49 examples/s]21003 examples [00:20, 1082.28 examples/s]21112 examples [00:20, 1073.67 examples/s]21220 examples [00:20, 1052.54 examples/s]21327 examples [00:20, 1055.84 examples/s]21433 examples [00:20, 1041.75 examples/s]21538 examples [00:20, 1026.66 examples/s]21641 examples [00:21, 1024.03 examples/s]21744 examples [00:21, 1011.84 examples/s]21850 examples [00:21, 1023.91 examples/s]21960 examples [00:21, 1045.00 examples/s]22067 examples [00:21, 1049.82 examples/s]22173 examples [00:21, 1032.36 examples/s]22277 examples [00:21, 1029.44 examples/s]22386 examples [00:21, 1046.09 examples/s]22491 examples [00:21, 1003.17 examples/s]22600 examples [00:22, 1027.68 examples/s]22709 examples [00:22, 1044.03 examples/s]22814 examples [00:22, 1039.43 examples/s]22920 examples [00:22, 1044.14 examples/s]23027 examples [00:22, 1050.86 examples/s]23133 examples [00:22, 1022.65 examples/s]23238 examples [00:22, 1029.67 examples/s]23348 examples [00:22, 1048.82 examples/s]23459 examples [00:22, 1064.59 examples/s]23570 examples [00:22, 1075.18 examples/s]23681 examples [00:23, 1082.75 examples/s]23790 examples [00:23, 1076.78 examples/s]23900 examples [00:23, 1080.80 examples/s]24009 examples [00:23, 1065.23 examples/s]24116 examples [00:23, 1064.18 examples/s]24223 examples [00:23, 1053.21 examples/s]24329 examples [00:23, 1050.31 examples/s]24436 examples [00:23, 1052.94 examples/s]24543 examples [00:23, 1057.45 examples/s]24649 examples [00:23, 1054.14 examples/s]24758 examples [00:24, 1063.52 examples/s]24865 examples [00:24, 1064.76 examples/s]24972 examples [00:24, 1063.28 examples/s]25079 examples [00:24, 1063.78 examples/s]25186 examples [00:24, 1060.90 examples/s]25294 examples [00:24, 1064.75 examples/s]25401 examples [00:24, 1062.91 examples/s]25508 examples [00:24, 1061.52 examples/s]25615 examples [00:24, 1063.29 examples/s]25722 examples [00:24, 1003.14 examples/s]25829 examples [00:25, 1021.31 examples/s]25934 examples [00:25, 1028.18 examples/s]26038 examples [00:25, 1012.52 examples/s]26146 examples [00:25, 1030.49 examples/s]26252 examples [00:25, 1039.00 examples/s]26357 examples [00:25, 1023.06 examples/s]26462 examples [00:25, 1030.12 examples/s]26571 examples [00:25, 1044.71 examples/s]26677 examples [00:25, 1048.23 examples/s]26786 examples [00:25, 1059.15 examples/s]26897 examples [00:26, 1071.19 examples/s]27005 examples [00:26, 1065.51 examples/s]27115 examples [00:26, 1073.67 examples/s]27226 examples [00:26, 1083.90 examples/s]27335 examples [00:26, 1054.16 examples/s]27441 examples [00:26, 1052.36 examples/s]27549 examples [00:26, 1059.10 examples/s]27657 examples [00:26, 1065.24 examples/s]27764 examples [00:26, 1058.18 examples/s]27873 examples [00:27, 1065.80 examples/s]27983 examples [00:27, 1073.78 examples/s]28091 examples [00:27, 1067.63 examples/s]28198 examples [00:27, 1063.76 examples/s]28305 examples [00:27, 1064.41 examples/s]28415 examples [00:27, 1073.23 examples/s]28523 examples [00:27, 1042.64 examples/s]28632 examples [00:27, 1055.89 examples/s]28741 examples [00:27, 1064.44 examples/s]28849 examples [00:27, 1069.04 examples/s]28958 examples [00:28, 1075.08 examples/s]29066 examples [00:28, 1056.65 examples/s]29172 examples [00:28, 1032.01 examples/s]29279 examples [00:28, 1041.48 examples/s]29388 examples [00:28, 1053.33 examples/s]29497 examples [00:28, 1063.24 examples/s]29606 examples [00:28, 1066.76 examples/s]29713 examples [00:28, 1067.70 examples/s]29823 examples [00:28, 1076.03 examples/s]29931 examples [00:28, 1059.74 examples/s]30038 examples [00:29, 1002.85 examples/s]30140 examples [00:29, 1005.84 examples/s]30242 examples [00:29, 998.74 examples/s] 30350 examples [00:29, 1020.39 examples/s]30460 examples [00:29, 1040.99 examples/s]30565 examples [00:29, 1038.90 examples/s]30673 examples [00:29, 1048.39 examples/s]30782 examples [00:29, 1056.97 examples/s]30888 examples [00:29, 1052.81 examples/s]30998 examples [00:29, 1063.61 examples/s]31108 examples [00:30, 1074.08 examples/s]31216 examples [00:30, 1063.20 examples/s]31324 examples [00:30, 1066.29 examples/s]31432 examples [00:30, 1070.25 examples/s]31540 examples [00:30, 1071.67 examples/s]31649 examples [00:30, 1073.94 examples/s]31757 examples [00:30, 1074.96 examples/s]31866 examples [00:30, 1077.17 examples/s]31975 examples [00:30, 1080.19 examples/s]32084 examples [00:30, 1067.84 examples/s]32192 examples [00:31, 1071.18 examples/s]32300 examples [00:31, 1035.94 examples/s]32404 examples [00:31, 1036.68 examples/s]32512 examples [00:31, 1048.24 examples/s]32621 examples [00:31, 1059.84 examples/s]32729 examples [00:31, 1064.31 examples/s]32838 examples [00:31, 1069.30 examples/s]32952 examples [00:31, 1088.45 examples/s]33062 examples [00:31, 1090.99 examples/s]33172 examples [00:32, 1090.13 examples/s]33282 examples [00:32, 1081.13 examples/s]33391 examples [00:32, 1066.28 examples/s]33502 examples [00:32, 1077.56 examples/s]33613 examples [00:32, 1084.53 examples/s]33723 examples [00:32, 1087.32 examples/s]33832 examples [00:32, 1088.08 examples/s]33941 examples [00:32, 1079.08 examples/s]34053 examples [00:32, 1090.00 examples/s]34163 examples [00:32, 1084.92 examples/s]34272 examples [00:33, 1085.18 examples/s]34381 examples [00:33, 1079.05 examples/s]34489 examples [00:33, 1078.34 examples/s]34599 examples [00:33, 1082.80 examples/s]34711 examples [00:33, 1092.35 examples/s]34824 examples [00:33, 1100.88 examples/s]34935 examples [00:33, 1102.19 examples/s]35046 examples [00:33, 1094.95 examples/s]35156 examples [00:33, 1093.59 examples/s]35266 examples [00:33, 1074.13 examples/s]35377 examples [00:34, 1083.06 examples/s]35486 examples [00:34, 1076.32 examples/s]35594 examples [00:34, 1045.91 examples/s]35699 examples [00:34, 1039.41 examples/s]35808 examples [00:34, 1051.86 examples/s]35918 examples [00:34, 1064.16 examples/s]36027 examples [00:34, 1069.01 examples/s]36136 examples [00:34, 1073.76 examples/s]36246 examples [00:34, 1079.70 examples/s]36356 examples [00:34, 1083.77 examples/s]36468 examples [00:35, 1093.78 examples/s]36578 examples [00:35, 1091.38 examples/s]36688 examples [00:35, 1081.49 examples/s]36798 examples [00:35, 1086.22 examples/s]36907 examples [00:35, 1085.82 examples/s]37016 examples [00:35, 1083.05 examples/s]37125 examples [00:35, 1068.98 examples/s]37234 examples [00:35, 1073.02 examples/s]37343 examples [00:35, 1077.58 examples/s]37452 examples [00:35, 1079.72 examples/s]37560 examples [00:36, 1076.41 examples/s]37668 examples [00:36, 1076.52 examples/s]37778 examples [00:36, 1082.52 examples/s]37889 examples [00:36, 1088.02 examples/s]38002 examples [00:36, 1098.99 examples/s]38112 examples [00:36, 1092.88 examples/s]38223 examples [00:36, 1095.95 examples/s]38333 examples [00:36, 1092.44 examples/s]38444 examples [00:36, 1096.05 examples/s]38554 examples [00:36, 1092.35 examples/s]38666 examples [00:37, 1098.52 examples/s]38776 examples [00:37, 1080.81 examples/s]38885 examples [00:37, 1063.18 examples/s]38992 examples [00:37, 1046.87 examples/s]39105 examples [00:37, 1068.28 examples/s]39213 examples [00:37, 1071.43 examples/s]39321 examples [00:37, 1058.10 examples/s]39428 examples [00:37, 1060.17 examples/s]39537 examples [00:37, 1065.86 examples/s]39644 examples [00:38, 1053.99 examples/s]39750 examples [00:38, 1049.51 examples/s]39859 examples [00:38, 1060.72 examples/s]39966 examples [00:38, 1054.49 examples/s]40072 examples [00:38, 1011.82 examples/s]40183 examples [00:38, 1038.92 examples/s]40293 examples [00:38, 1054.48 examples/s]40402 examples [00:38, 1062.32 examples/s]40511 examples [00:38, 1068.82 examples/s]40619 examples [00:38, 1069.20 examples/s]40728 examples [00:39, 1073.58 examples/s]40836 examples [00:39, 1075.37 examples/s]40944 examples [00:39, 1063.31 examples/s]41051 examples [00:39, 1061.01 examples/s]41160 examples [00:39, 1067.62 examples/s]41270 examples [00:39, 1075.63 examples/s]41379 examples [00:39, 1078.10 examples/s]41487 examples [00:39, 1075.27 examples/s]41595 examples [00:39, 1053.50 examples/s]41702 examples [00:39, 1058.30 examples/s]41813 examples [00:40, 1072.26 examples/s]41925 examples [00:40, 1084.06 examples/s]42034 examples [00:40, 1062.33 examples/s]42141 examples [00:40, 1036.29 examples/s]42247 examples [00:40, 1043.00 examples/s]42357 examples [00:40, 1057.98 examples/s]42463 examples [00:40, 1034.59 examples/s]42572 examples [00:40, 1048.77 examples/s]42678 examples [00:40, 1045.75 examples/s]42783 examples [00:40, 1034.59 examples/s]42891 examples [00:41, 1045.43 examples/s]43004 examples [00:41, 1067.34 examples/s]43114 examples [00:41, 1075.70 examples/s]43222 examples [00:41, 1064.19 examples/s]43331 examples [00:41, 1071.68 examples/s]43439 examples [00:41, 1059.40 examples/s]43548 examples [00:41, 1066.79 examples/s]43655 examples [00:41, 1055.92 examples/s]43761 examples [00:41, 1054.68 examples/s]43874 examples [00:41, 1073.47 examples/s]43984 examples [00:42, 1079.18 examples/s]44093 examples [00:42, 1065.80 examples/s]44203 examples [00:42, 1075.17 examples/s]44311 examples [00:42, 1068.37 examples/s]44428 examples [00:42, 1096.05 examples/s]44538 examples [00:42, 1088.77 examples/s]44650 examples [00:42, 1096.24 examples/s]44760 examples [00:42, 1092.73 examples/s]44870 examples [00:42, 1088.74 examples/s]44980 examples [00:43, 1091.27 examples/s]45090 examples [00:43, 1091.89 examples/s]45200 examples [00:43, 1087.29 examples/s]45309 examples [00:43, 1082.00 examples/s]45418 examples [00:43, 1038.40 examples/s]45525 examples [00:43, 1047.38 examples/s]45633 examples [00:43, 1054.61 examples/s]45741 examples [00:43, 1061.26 examples/s]45850 examples [00:43, 1067.41 examples/s]45957 examples [00:43, 1051.98 examples/s]46067 examples [00:44, 1063.73 examples/s]46174 examples [00:44, 1049.32 examples/s]46281 examples [00:44, 1054.56 examples/s]46390 examples [00:44, 1064.70 examples/s]46497 examples [00:44, 1054.48 examples/s]46608 examples [00:44, 1070.34 examples/s]46716 examples [00:44, 1071.66 examples/s]46828 examples [00:44, 1083.02 examples/s]46937 examples [00:44, 1081.47 examples/s]47046 examples [00:44, 1053.61 examples/s]47155 examples [00:45, 1062.12 examples/s]47263 examples [00:45, 1065.06 examples/s]47370 examples [00:45, 1050.29 examples/s]47478 examples [00:45, 1057.40 examples/s]47586 examples [00:45, 1062.70 examples/s]47694 examples [00:45, 1067.19 examples/s]47803 examples [00:45, 1073.18 examples/s]47911 examples [00:45, 1072.67 examples/s]48019 examples [00:45, 1070.47 examples/s]48127 examples [00:45, 1054.97 examples/s]48239 examples [00:46, 1071.75 examples/s]48347 examples [00:46, 1060.67 examples/s]48455 examples [00:46, 1063.82 examples/s]48563 examples [00:46, 1068.15 examples/s]48670 examples [00:46, 1025.81 examples/s]48780 examples [00:46, 1045.70 examples/s]48890 examples [00:46, 1058.97 examples/s]48997 examples [00:46, 1024.25 examples/s]49106 examples [00:46, 1040.15 examples/s]49211 examples [00:47, 1040.00 examples/s]49322 examples [00:47, 1059.78 examples/s]49429 examples [00:47, 1034.33 examples/s]49538 examples [00:47, 1049.66 examples/s]49648 examples [00:47, 1062.11 examples/s]49759 examples [00:47, 1073.28 examples/s]49867 examples [00:47, 1070.58 examples/s]49976 examples [00:47, 1073.82 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6743/50000 [00:00<00:00, 67426.69 examples/s] 39%|███▉      | 19632/50000 [00:00<00:00, 78682.69 examples/s] 66%|██████▋   | 33137/50000 [00:00<00:00, 89944.78 examples/s] 93%|█████████▎| 46713/50000 [00:00<00:00, 100074.72 examples/s]                                                                0 examples [00:00, ? examples/s]88 examples [00:00, 875.48 examples/s]197 examples [00:00, 930.13 examples/s]307 examples [00:00, 973.81 examples/s]413 examples [00:00, 997.44 examples/s]523 examples [00:00, 1023.89 examples/s]632 examples [00:00, 1040.35 examples/s]741 examples [00:00, 1052.98 examples/s]840 examples [00:00, 1031.41 examples/s]941 examples [00:00, 1024.68 examples/s]1046 examples [00:01, 1029.82 examples/s]1153 examples [00:01, 1039.89 examples/s]1263 examples [00:01, 1054.72 examples/s]1369 examples [00:01, 1054.40 examples/s]1474 examples [00:01, 1033.19 examples/s]1585 examples [00:01, 1053.09 examples/s]1695 examples [00:01, 1066.39 examples/s]1806 examples [00:01, 1076.64 examples/s]1915 examples [00:01, 1078.07 examples/s]2026 examples [00:01, 1086.46 examples/s]2138 examples [00:02, 1094.97 examples/s]2249 examples [00:02, 1098.35 examples/s]2359 examples [00:02, 1097.87 examples/s]2469 examples [00:02, 1092.53 examples/s]2579 examples [00:02, 1091.29 examples/s]2689 examples [00:02, 1093.54 examples/s]2799 examples [00:02, 1095.14 examples/s]2909 examples [00:02, 1091.61 examples/s]3021 examples [00:02, 1098.01 examples/s]3131 examples [00:02, 1089.46 examples/s]3240 examples [00:03, 1089.06 examples/s]3351 examples [00:03, 1093.58 examples/s]3461 examples [00:03, 1086.18 examples/s]3570 examples [00:03, 1082.91 examples/s]3679 examples [00:03, 1081.36 examples/s]3788 examples [00:03, 1079.28 examples/s]3899 examples [00:03, 1086.35 examples/s]4012 examples [00:03, 1098.84 examples/s]4122 examples [00:03, 1093.22 examples/s]4232 examples [00:03, 1054.96 examples/s]4338 examples [00:04, 1050.11 examples/s]4446 examples [00:04, 1056.66 examples/s]4558 examples [00:04, 1072.25 examples/s]4670 examples [00:04, 1085.19 examples/s]4779 examples [00:04, 1041.91 examples/s]4885 examples [00:04, 1046.48 examples/s]4995 examples [00:04, 1059.92 examples/s]5107 examples [00:04, 1076.11 examples/s]5215 examples [00:04, 1072.92 examples/s]5328 examples [00:04, 1087.39 examples/s]5442 examples [00:05, 1100.77 examples/s]5553 examples [00:05, 1087.65 examples/s]5667 examples [00:05, 1100.75 examples/s]5779 examples [00:05, 1103.71 examples/s]5890 examples [00:05, 1093.36 examples/s]6002 examples [00:05, 1099.62 examples/s]6113 examples [00:05, 1089.92 examples/s]6223 examples [00:05, 1076.34 examples/s]6334 examples [00:05, 1084.37 examples/s]6443 examples [00:05, 1079.90 examples/s]6552 examples [00:06, 1070.72 examples/s]6663 examples [00:06, 1080.27 examples/s]6774 examples [00:06, 1087.22 examples/s]6885 examples [00:06, 1093.29 examples/s]6999 examples [00:06, 1105.88 examples/s]7110 examples [00:06, 1106.28 examples/s]7221 examples [00:06, 1099.27 examples/s]7331 examples [00:06, 1093.09 examples/s]7441 examples [00:06, 1089.56 examples/s]7550 examples [00:07, 1083.83 examples/s]7659 examples [00:07, 1083.80 examples/s]7768 examples [00:07, 1075.69 examples/s]7878 examples [00:07, 1080.37 examples/s]7987 examples [00:07, 1082.27 examples/s]8096 examples [00:07, 1042.77 examples/s]8202 examples [00:07, 1046.33 examples/s]8310 examples [00:07, 1056.13 examples/s]8422 examples [00:07, 1072.24 examples/s]8533 examples [00:07, 1082.14 examples/s]8646 examples [00:08, 1092.88 examples/s]8758 examples [00:08, 1099.40 examples/s]8869 examples [00:08, 1089.50 examples/s]8979 examples [00:08, 1083.49 examples/s]9088 examples [00:08, 1085.04 examples/s]9198 examples [00:08, 1088.50 examples/s]9307 examples [00:08, 1084.73 examples/s]9417 examples [00:08, 1087.96 examples/s]9529 examples [00:08, 1095.93 examples/s]9639 examples [00:08, 1063.54 examples/s]9748 examples [00:09, 1068.85 examples/s]9858 examples [00:09, 1075.89 examples/s]9968 examples [00:09, 1081.19 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteZZXCDV/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteZZXCDV/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa1bb3cb620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa1bb3cb620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fa1bb3cb620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa140834588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa140870908>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa1bb3cb620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa1bb3cb620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fa1bb3cb620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa1a23ff5f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa1a23ff390>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fa132e4b378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fa132e4b378> 

  function with postional parmater data_info <function split_train_valid at 0x7fa132e4b378> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fa132e4b488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fa132e4b488> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fa132e4b488> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=b953ff587f9f9f9132fb3f04b964e924eb295327809ec3a935c207397e5c3042
  Stored in directory: /tmp/pip-ephem-wheel-cache-1in4d_kh/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:07:11, 10.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:43:06, 15.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:03:24, 21.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:44:52, 30.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:24:35, 44.1kB/s].vector_cache/glove.6B.zip:   1%|          | 8.48M/862M [00:01<3:46:01, 63.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:01<2:37:34, 89.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:01<1:49:42, 128kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:01<1:16:37, 183kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.8M/862M [00:01<53:25, 261kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:01<37:20, 372kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.5M/862M [00:02<26:04, 529kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.1M/862M [00:02<18:17, 751kB/s].vector_cache/glove.6B.zip:   5%|▍         | 43.0M/862M [00:02<12:48, 1.07MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.6M/862M [00:02<09:02, 1.50MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.1M/862M [00:02<06:23, 2.12MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.7M/862M [00:03<06:18, 2.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<06:18, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<06:32, 2.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.2M/862M [00:05<05:05, 2.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<05:56, 2.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:07<05:55, 2.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<04:35, 2.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<05:47, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:09<05:37, 2.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.9M/862M [00:09<04:14, 3.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:11<05:46, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<06:20, 2.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:11<05:07, 2.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:11<03:48, 3.46MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:13<07:16, 1.81MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:13<06:26, 2.04MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.4M/862M [00:13<04:50, 2.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<06:26, 2.03MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:15<05:52, 2.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.5M/862M [00:15<04:26, 2.94MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<06:09, 2.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:17<05:38, 2.30MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.6M/862M [00:17<04:16, 3.03MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<06:02, 2.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:19<05:33, 2.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.8M/862M [00:19<04:12, 3.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<05:59, 2.15MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:21<05:30, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.9M/862M [00:21<04:10, 3.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<05:56, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:22<05:27, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.0M/862M [00:23<04:08, 3.08MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<05:55, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<06:46, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:25<05:23, 2.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<03:55, 3.23MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<3:06:39, 67.8kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<2:11:51, 96.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<1:32:26, 137kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:07:30, 187kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<48:18, 261kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<34:00, 370kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<23:55, 524kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<1:04:03, 196kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<46:06, 272kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<32:28, 385kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<25:36, 487kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<19:12, 648kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<13:44, 905kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<12:31, 990kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<10:01, 1.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<07:16, 1.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:00, 1.54MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:07, 1.51MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<06:14, 1.97MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<04:30, 2.72MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<11:51, 1.03MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<09:35, 1.28MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<07:00, 1.74MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:44, 1.57MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:40, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<04:58, 2.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:16, 1.93MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:52, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:24, 2.24MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:44, 2.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:16, 2.28MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<03:57, 3.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:31, 2.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:20, 1.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:02, 2.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:27, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:01, 2.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<03:48, 3.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:26, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:13, 1.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:51, 2.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<03:34, 3.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:18, 1.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:20, 1.86MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:41, 2.51MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:01, 1.94MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:38, 1.76MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:08, 2.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<03:45, 3.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:47, 1.49MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:26, 1.80MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:46, 2.43MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:56<03:28, 3.32MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<49:01, 236kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<36:40, 315kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<26:13, 440kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<20:08, 571kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<15:05, 761kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<10:48, 1.06MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<07:41, 1.49MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<35:21, 323kB/s] .vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<27:00, 423kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<19:22, 589kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<13:41, 831kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<14:28, 785kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<11:18, 1.00MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<08:11, 1.38MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:21, 1.35MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:09, 1.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:17, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<04:30, 2.49MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<10:51:00, 17.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<7:36:35, 24.6kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<5:19:09, 35.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<3:45:18, 49.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<2:39:57, 69.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<1:52:20, 99.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<1:18:32, 141kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<1:00:16, 184kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<43:17, 256kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<30:31, 362kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<23:52, 462kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<17:49, 618kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<12:43, 863kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<11:28, 954kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<08:58, 1.22MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:30, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:41, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<47:04, 231kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<35:11, 309kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<25:04, 433kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<17:40, 613kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<16:32, 653kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<12:43, 850kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<09:09, 1.18MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:54, 1.21MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<07:19, 1.47MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<05:23, 1.99MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:17, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:36, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<05:03, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<03:43, 2.86MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:18, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:31, 1.92MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<04:08, 2.56MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:20, 1.97MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:48, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<03:35, 2.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:58, 2.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:32, 2.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:26, 3.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:51, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:32, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:24, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:44, 2.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:22, 2.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:19, 3.10MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:43, 2.17MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:21, 2.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<03:18, 3.09MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:43, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:23, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:13, 2.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<03:03, 3.31MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<11:03, 916kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<08:46, 1.15MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<06:23, 1.58MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<06:49, 1.48MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:47, 1.73MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<04:18, 2.33MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<05:20, 1.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<04:45, 2.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:34, 2.78MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:50, 2.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:24, 1.83MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:13, 2.34MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<03:05, 3.19MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:20, 1.55MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:47<05:27, 1.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:03, 2.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:07, 1.91MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:36, 1.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:21, 2.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<03:09, 3.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<09:06, 1.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<07:23, 1.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:22, 1.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<06:00, 1.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:11, 1.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:45, 2.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<03:27, 2.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<07:14, 1.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<06:03, 1.58MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:28, 2.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:21, 1.78MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:41, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:27, 2.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<03:12, 2.94MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<9:07:17, 17.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<6:23:48, 24.6kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<4:28:08, 35.1kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<3:09:11, 49.5kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<2:13:18, 70.2kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<1:33:14, 100kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<1:07:14, 138kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<47:59, 194kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<33:41, 275kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<25:42, 359kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<19:52, 464kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<14:22, 642kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<11:29, 798kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<08:58, 1.02MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<06:30, 1.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:39, 1.37MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<06:32, 1.39MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:02, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:58, 1.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:25, 2.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<03:19, 2.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:23, 2.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:59, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<03:57, 2.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<02:51, 3.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<58:44, 151kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<42:00, 211kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<29:33, 300kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<22:40, 389kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<17:40, 499kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<12:44, 691kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<09:02, 971kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<09:19, 939kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<07:24, 1.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<05:22, 1.62MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:47, 1.50MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:55, 1.76MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<03:39, 2.37MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:35, 1.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:58, 1.73MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:52, 2.22MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<02:49, 3.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<05:48, 1.47MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:57, 1.72MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<03:39, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:31, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:02, 2.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<03:00, 2.81MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<04:04, 2.06MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<04:34, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:37, 2.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<02:36, 3.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<17:32, 475kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<13:07, 635kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<09:22, 886kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<08:27, 978kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<07:32, 1.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<05:41, 1.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<04:02, 2.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<39:38, 207kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<28:35, 287kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<20:09, 405kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<15:56, 510kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<12:49, 635kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<09:22, 866kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<07:49, 1.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<06:17, 1.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:36, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:05, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<04:22, 1.83MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<03:15, 2.45MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:08, 1.91MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:42, 2.14MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<02:47, 2.83MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:48, 2.07MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:27, 2.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<02:37, 2.99MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:40, 2.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:10, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:15, 2.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:21, 3.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<07:08, 1.08MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<05:47, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<04:14, 1.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:44, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:05, 1.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:01, 2.52MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:55, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:23, 2.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:30, 3.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<01:51, 4.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<31:36, 238kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<22:51, 329kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<16:08, 464kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<13:00, 573kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<09:51, 755kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<07:04, 1.05MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<06:40, 1.11MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<06:11, 1.19MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:42, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:27, 1.64MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:52, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<02:53, 2.52MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:42, 1.96MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:08, 1.75MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:16, 2.21MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<02:21, 3.03MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<53:01, 135kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<37:50, 189kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<26:34, 269kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<20:10, 352kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<15:38, 454kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<11:14, 631kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<07:55, 890kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<08:51, 794kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<06:56, 1.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<05:00, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<05:07, 1.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<04:17, 1.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<03:08, 2.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:50, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:22, 2.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<02:30, 2.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:22, 2.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<03:03, 2.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:18, 2.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:12, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:56, 2.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:11, 3.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:07, 2.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:52, 2.33MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:08, 3.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:04, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:49, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:08, 3.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:02, 2.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:28, 1.89MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:42, 2.41MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<01:57, 3.31MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:55, 2.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<5:03:25, 21.4kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<3:32:18, 30.5kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:23<2:27:30, 43.6kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<1:57:56, 54.5kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<1:23:51, 76.5kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<58:54, 109kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<41:55, 152kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<29:59, 212kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<21:04, 300kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<16:07, 390kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<12:35, 499kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<09:06, 688kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<06:23, 972kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<6:02:54, 17.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<4:14:24, 24.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<2:57:29, 34.8kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<2:04:58, 49.2kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<1:28:01, 69.8kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<1:01:30, 99.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<44:14, 137kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<31:33, 192kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<22:08, 273kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<16:50, 357kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<13:00, 462kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<09:23, 638kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<07:29, 794kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<05:50, 1.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<04:13, 1.40MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:18, 1.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:16, 1.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:17, 1.78MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<02:21, 2.46MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<43:02, 135kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<30:42, 189kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<21:31, 268kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<16:18, 352kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<12:35, 456kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<09:03, 632kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<06:20, 894kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<11:15, 504kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<08:27, 669kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<06:00, 937kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<05:30, 1.02MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:25, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:12, 1.74MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<03:33, 1.56MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:03, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:15, 2.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:52, 1.90MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:33, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<01:54, 2.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<01:23, 3.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<21:26, 252kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<16:03, 336kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<11:29, 468kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<08:49, 603kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<06:43, 790kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:48, 1.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<04:34, 1.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:44, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:44, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:07, 1.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:43, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<02:01, 2.55MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:37, 1.95MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:21, 2.17MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<01:46, 2.87MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:25, 2.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:13, 2.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:40, 2.99MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:20, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:08, 2.31MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:37, 3.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:17, 2.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:06, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:35, 3.07MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:15, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:59, 2.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:10<01:28, 3.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:10<01:05, 4.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<18:47, 254kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<14:07, 338kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<10:06, 471kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<07:45, 606kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<05:54, 796kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<04:13, 1.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<03:13, 1.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<3:32:40, 21.8kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<2:28:41, 31.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<1:42:56, 44.4kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<1:26:54, 52.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<1:01:43, 74.0kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<43:18, 105kB/s]   .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<30:42, 147kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<21:56, 205kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<15:23, 290kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<11:42, 379kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<09:05, 487kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<06:32, 675kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<04:36, 951kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<05:02, 865kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:58, 1.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:51, 1.51MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:59, 1.43MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:58, 1.44MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:15, 1.89MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:39, 2.57MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:27, 1.72MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:09, 1.96MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:35, 2.64MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:04, 2.00MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:52, 2.21MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:23, 2.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:56, 2.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:46, 2.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:19, 3.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:52, 2.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:08, 1.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:42, 2.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<01:49, 2.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<01:41, 2.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:15, 3.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:47, 2.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:38, 2.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:14, 3.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:45, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<01:37, 2.35MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:13, 3.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:44, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:58, 1.89MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:32, 2.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:07, 3.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:22, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:02, 1.80MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:30, 2.41MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:53, 1.90MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:41, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:15, 2.85MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:42, 2.06MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:33, 2.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:09, 3.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:37, 2.14MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:50, 1.88MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:26, 2.40MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:02, 3.28MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:42, 1.25MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:15, 1.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:38, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:55, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:40, 1.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:14, 2.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:44, 1.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:16, 2.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:35, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:26, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:04, 2.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:29, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:40, 1.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:19, 2.34MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:24, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:17, 2.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<00:58, 3.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:22, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:36, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:16, 2.34MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:03<00:54, 3.20MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<21:29, 136kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<15:19, 190kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<10:41, 270kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<07:33, 378kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<2:18:20, 20.6kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<1:36:32, 29.4kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<1:06:21, 42.0kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<54:38, 51.0kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<38:46, 71.8kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<27:08, 102kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<19:05, 142kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<13:37, 199kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<09:30, 282kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<07:10, 369kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<05:33, 475kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<03:59, 660kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<02:47, 930kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<03:18, 779kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:34, 999kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:51, 1.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<01:52, 1.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:33, 1.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:08, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:21, 1.79MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:12, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<00:53, 2.70MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:10, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:18, 1.81MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:01, 2.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:04, 2.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:59, 2.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:44, 3.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:02, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<00:57, 2.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:42, 3.11MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:00, 2.15MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:08, 1.89MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<00:53, 2.41MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:38, 3.30MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:45, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:26, 1.45MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:02, 1.99MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:11, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:15, 1.60MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:59, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<00:42, 2.81MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<14:28, 136kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<10:18, 190kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<07:10, 270kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<05:22, 353kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<04:08, 457kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<02:57, 635kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<02:03, 896kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<02:20, 780kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:47, 1.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:37<01:17, 1.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:17, 1.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:15, 1.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:57, 1.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:55, 1.82MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:49, 2.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<00:36, 2.72MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:47, 2.04MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:53, 1.83MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:41, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:43, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:40, 2.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:29, 3.05MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:41, 2.15MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:47, 1.88MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:36, 2.40MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:46<00:26, 3.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:26, 985kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:17, 1.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:58, 1.44MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:40, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<05:19, 253kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<03:50, 349kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<02:40, 492kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<02:06, 604kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:36, 793kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<01:07, 1.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:03, 1.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:51, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:37, 1.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:41, 1.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:43, 1.59MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:33, 2.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:56<00:22, 2.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<1:02:13, 17.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<43:23, 24.6kB/s]  .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<29:43, 35.1kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:58<20:05, 50.1kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<18:04, 55.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<12:40, 78.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<08:39, 112kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<06:03, 154kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<04:18, 216kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<02:57, 306kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<02:11, 396kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:36, 534kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<01:06, 749kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:56, 853kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:48, 983kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:35, 1.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:23, 1.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<42:13, 17.3kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<29:22, 24.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<19:51, 35.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<13:19, 49.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<09:25, 69.9kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<06:29, 99.4kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<04:16, 142kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<03:16, 181kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<02:19, 252kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<01:34, 356kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<01:09, 454kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:54, 573kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:38, 791kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:25, 1.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:30, 894kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:23, 1.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:16, 1.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:15, 1.46MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:09, 2.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:10, 1.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:11, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:08, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:07, 2.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:04, 2.98MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 2.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:24<00:05, 1.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:02, 3.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:04, 1.42MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.84MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 2.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.77MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<27:23:26,  4.06it/s]  0%|          | 890/400000 [00:00<19:08:04,  5.79it/s]  0%|          | 1711/400000 [00:00<13:22:14,  8.27it/s]  1%|          | 2591/400000 [00:00<9:20:33, 11.82it/s]   1%|          | 3466/400000 [00:00<6:31:45, 16.87it/s]  1%|          | 4368/400000 [00:00<4:33:49, 24.08it/s]  1%|▏         | 5250/400000 [00:00<3:11:28, 34.36it/s]  2%|▏         | 6129/400000 [00:00<2:13:57, 49.01it/s]  2%|▏         | 7037/400000 [00:01<1:33:46, 69.85it/s]  2%|▏         | 7934/400000 [00:01<1:05:42, 99.45it/s]  2%|▏         | 8838/400000 [00:01<46:06, 141.40it/s]   2%|▏         | 9708/400000 [00:01<32:25, 200.58it/s]  3%|▎         | 10573/400000 [00:01<22:53, 283.57it/s]  3%|▎         | 11440/400000 [00:01<16:12, 399.50it/s]  3%|▎         | 12320/400000 [00:01<11:32, 559.82it/s]  3%|▎         | 13221/400000 [00:01<08:16, 778.99it/s]  4%|▎         | 14112/400000 [00:01<05:59, 1072.62it/s]  4%|▎         | 14992/400000 [00:01<04:25, 1448.84it/s]  4%|▍         | 15847/400000 [00:02<03:19, 1928.61it/s]  4%|▍         | 16700/400000 [00:02<02:32, 2509.99it/s]  4%|▍         | 17580/400000 [00:02<01:59, 3194.83it/s]  5%|▍         | 18457/400000 [00:02<01:36, 3947.53it/s]  5%|▍         | 19322/400000 [00:02<01:20, 4711.72it/s]  5%|▌         | 20210/400000 [00:02<01:09, 5482.92it/s]  5%|▌         | 21101/400000 [00:02<01:01, 6197.11it/s]  5%|▌         | 21991/400000 [00:02<00:55, 6817.73it/s]  6%|▌         | 22872/400000 [00:02<00:51, 7278.63it/s]  6%|▌         | 23748/400000 [00:02<00:49, 7574.62it/s]  6%|▌         | 24620/400000 [00:03<00:47, 7884.29it/s]  6%|▋         | 25487/400000 [00:03<00:46, 8076.16it/s]  7%|▋         | 26378/400000 [00:03<00:44, 8308.33it/s]  7%|▋         | 27257/400000 [00:03<00:44, 8446.30it/s]  7%|▋         | 28131/400000 [00:03<00:43, 8501.63it/s]  7%|▋         | 29002/400000 [00:03<00:43, 8495.87it/s]  7%|▋         | 29918/400000 [00:03<00:42, 8682.95it/s]  8%|▊         | 30822/400000 [00:03<00:42, 8786.85it/s]  8%|▊         | 31709/400000 [00:03<00:42, 8671.00it/s]  8%|▊         | 32583/400000 [00:03<00:43, 8530.22it/s]  8%|▊         | 33443/400000 [00:04<00:42, 8550.22it/s]  9%|▊         | 34302/400000 [00:04<00:42, 8553.06it/s]  9%|▉         | 35185/400000 [00:04<00:42, 8633.21it/s]  9%|▉         | 36063/400000 [00:04<00:41, 8676.48it/s]  9%|▉         | 36932/400000 [00:04<00:42, 8596.17it/s]  9%|▉         | 37799/400000 [00:04<00:42, 8617.08it/s] 10%|▉         | 38662/400000 [00:04<00:42, 8571.86it/s] 10%|▉         | 39547/400000 [00:04<00:41, 8653.11it/s] 10%|█         | 40413/400000 [00:04<00:42, 8492.78it/s] 10%|█         | 41264/400000 [00:05<00:42, 8448.04it/s] 11%|█         | 42110/400000 [00:05<00:42, 8338.21it/s] 11%|█         | 42945/400000 [00:05<00:43, 8181.40it/s] 11%|█         | 43820/400000 [00:05<00:42, 8342.92it/s] 11%|█         | 44668/400000 [00:05<00:42, 8383.38it/s] 11%|█▏        | 45508/400000 [00:05<00:42, 8261.06it/s] 12%|█▏        | 46350/400000 [00:05<00:42, 8306.80it/s] 12%|█▏        | 47182/400000 [00:05<00:43, 8153.09it/s] 12%|█▏        | 48020/400000 [00:05<00:42, 8218.15it/s] 12%|█▏        | 48847/400000 [00:05<00:42, 8232.59it/s] 12%|█▏        | 49676/400000 [00:06<00:42, 8248.35it/s] 13%|█▎        | 50502/400000 [00:06<00:43, 8116.16it/s] 13%|█▎        | 51375/400000 [00:06<00:42, 8290.46it/s] 13%|█▎        | 52279/400000 [00:06<00:40, 8499.53it/s] 13%|█▎        | 53197/400000 [00:06<00:39, 8692.77it/s] 14%|█▎        | 54070/400000 [00:06<00:41, 8334.66it/s] 14%|█▎        | 54929/400000 [00:06<00:41, 8408.30it/s] 14%|█▍        | 55778/400000 [00:06<00:40, 8430.81it/s] 14%|█▍        | 56654/400000 [00:06<00:40, 8526.58it/s] 14%|█▍        | 57509/400000 [00:06<00:40, 8446.76it/s] 15%|█▍        | 58356/400000 [00:07<00:40, 8447.95it/s] 15%|█▍        | 59217/400000 [00:07<00:40, 8495.69it/s] 15%|█▌        | 60068/400000 [00:07<00:40, 8454.89it/s] 15%|█▌        | 60935/400000 [00:07<00:39, 8516.14it/s] 15%|█▌        | 61788/400000 [00:07<00:40, 8454.95it/s] 16%|█▌        | 62634/400000 [00:07<00:39, 8450.31it/s] 16%|█▌        | 63499/400000 [00:07<00:39, 8506.57it/s] 16%|█▌        | 64379/400000 [00:07<00:39, 8591.14it/s] 16%|█▋        | 65270/400000 [00:07<00:38, 8682.73it/s] 17%|█▋        | 66139/400000 [00:07<00:38, 8662.37it/s] 17%|█▋        | 67006/400000 [00:08<00:39, 8490.34it/s] 17%|█▋        | 67857/400000 [00:08<00:39, 8459.68it/s] 17%|█▋        | 68704/400000 [00:08<00:39, 8448.32it/s] 17%|█▋        | 69550/400000 [00:08<00:39, 8356.05it/s] 18%|█▊        | 70414/400000 [00:08<00:39, 8438.35it/s] 18%|█▊        | 71269/400000 [00:08<00:38, 8469.61it/s] 18%|█▊        | 72117/400000 [00:08<00:38, 8467.09it/s] 18%|█▊        | 72965/400000 [00:08<00:38, 8434.05it/s] 18%|█▊        | 73809/400000 [00:08<00:38, 8407.62it/s] 19%|█▊        | 74650/400000 [00:08<00:38, 8389.71it/s] 19%|█▉        | 75490/400000 [00:09<00:39, 8301.93it/s] 19%|█▉        | 76335/400000 [00:09<00:38, 8344.58it/s] 19%|█▉        | 77170/400000 [00:09<00:38, 8333.80it/s] 20%|█▉        | 78060/400000 [00:09<00:37, 8494.75it/s] 20%|█▉        | 78931/400000 [00:09<00:37, 8557.53it/s] 20%|█▉        | 79788/400000 [00:09<00:37, 8505.55it/s] 20%|██        | 80694/400000 [00:09<00:36, 8662.49it/s] 20%|██        | 81598/400000 [00:09<00:36, 8768.87it/s] 21%|██        | 82493/400000 [00:09<00:35, 8821.42it/s] 21%|██        | 83376/400000 [00:09<00:36, 8751.12it/s] 21%|██        | 84261/400000 [00:10<00:35, 8779.11it/s] 21%|██▏       | 85142/400000 [00:10<00:35, 8786.16it/s] 22%|██▏       | 86021/400000 [00:10<00:36, 8603.31it/s] 22%|██▏       | 86883/400000 [00:10<00:36, 8582.04it/s] 22%|██▏       | 87742/400000 [00:10<00:37, 8392.71it/s] 22%|██▏       | 88583/400000 [00:10<00:37, 8388.16it/s] 22%|██▏       | 89487/400000 [00:10<00:36, 8572.87it/s] 23%|██▎       | 90373/400000 [00:10<00:35, 8656.28it/s] 23%|██▎       | 91246/400000 [00:10<00:35, 8677.22it/s] 23%|██▎       | 92115/400000 [00:11<00:35, 8628.52it/s] 23%|██▎       | 92979/400000 [00:11<00:35, 8623.26it/s] 23%|██▎       | 93887/400000 [00:11<00:34, 8753.91it/s] 24%|██▎       | 94810/400000 [00:11<00:34, 8890.09it/s] 24%|██▍       | 95736/400000 [00:11<00:33, 8995.20it/s] 24%|██▍       | 96665/400000 [00:11<00:33, 9081.13it/s] 24%|██▍       | 97575/400000 [00:11<00:34, 8715.60it/s] 25%|██▍       | 98466/400000 [00:11<00:34, 8770.94it/s] 25%|██▍       | 99346/400000 [00:11<00:34, 8591.86it/s] 25%|██▌       | 100208/400000 [00:11<00:35, 8513.79it/s] 25%|██▌       | 101062/400000 [00:12<00:36, 8238.19it/s] 25%|██▌       | 101890/400000 [00:12<00:36, 8154.81it/s] 26%|██▌       | 102755/400000 [00:12<00:35, 8297.31it/s] 26%|██▌       | 103644/400000 [00:12<00:35, 8464.25it/s] 26%|██▌       | 104510/400000 [00:12<00:34, 8520.15it/s] 26%|██▋       | 105388/400000 [00:12<00:34, 8595.07it/s] 27%|██▋       | 106249/400000 [00:12<00:34, 8521.75it/s] 27%|██▋       | 107103/400000 [00:12<00:34, 8471.71it/s] 27%|██▋       | 107979/400000 [00:12<00:34, 8555.47it/s] 27%|██▋       | 108836/400000 [00:12<00:34, 8519.69it/s] 27%|██▋       | 109711/400000 [00:13<00:33, 8586.46it/s] 28%|██▊       | 110571/400000 [00:13<00:33, 8523.75it/s] 28%|██▊       | 111459/400000 [00:13<00:33, 8627.53it/s] 28%|██▊       | 112326/400000 [00:13<00:33, 8638.56it/s] 28%|██▊       | 113218/400000 [00:13<00:32, 8719.88it/s] 29%|██▊       | 114093/400000 [00:13<00:32, 8728.82it/s] 29%|██▊       | 114967/400000 [00:13<00:32, 8691.33it/s] 29%|██▉       | 115837/400000 [00:13<00:32, 8639.36it/s] 29%|██▉       | 116743/400000 [00:13<00:32, 8761.03it/s] 29%|██▉       | 117620/400000 [00:13<00:32, 8657.22it/s] 30%|██▉       | 118487/400000 [00:14<00:32, 8566.74it/s] 30%|██▉       | 119349/400000 [00:14<00:32, 8581.16it/s] 30%|███       | 120208/400000 [00:14<00:32, 8575.84it/s] 30%|███       | 121083/400000 [00:14<00:32, 8624.64it/s] 30%|███       | 121952/400000 [00:14<00:32, 8643.45it/s] 31%|███       | 122817/400000 [00:14<00:32, 8628.93it/s] 31%|███       | 123700/400000 [00:14<00:31, 8686.96it/s] 31%|███       | 124610/400000 [00:14<00:31, 8805.24it/s] 31%|███▏      | 125492/400000 [00:14<00:31, 8764.26it/s] 32%|███▏      | 126369/400000 [00:14<00:31, 8664.74it/s] 32%|███▏      | 127237/400000 [00:15<00:31, 8579.16it/s] 32%|███▏      | 128140/400000 [00:15<00:31, 8708.65it/s] 32%|███▏      | 129050/400000 [00:15<00:30, 8819.78it/s] 32%|███▏      | 129976/400000 [00:15<00:30, 8945.31it/s] 33%|███▎      | 130885/400000 [00:15<00:29, 8985.33it/s] 33%|███▎      | 131785/400000 [00:15<00:30, 8851.58it/s] 33%|███▎      | 132672/400000 [00:15<00:31, 8500.05it/s] 33%|███▎      | 133526/400000 [00:15<00:31, 8488.18it/s] 34%|███▎      | 134390/400000 [00:15<00:31, 8531.23it/s] 34%|███▍      | 135276/400000 [00:15<00:30, 8624.64it/s] 34%|███▍      | 136182/400000 [00:16<00:30, 8748.75it/s] 34%|███▍      | 137062/400000 [00:16<00:30, 8761.07it/s] 34%|███▍      | 137960/400000 [00:16<00:29, 8825.19it/s] 35%|███▍      | 138844/400000 [00:16<00:29, 8779.79it/s] 35%|███▍      | 139723/400000 [00:16<00:29, 8733.32it/s] 35%|███▌      | 140597/400000 [00:16<00:30, 8568.53it/s] 35%|███▌      | 141482/400000 [00:16<00:29, 8649.03it/s] 36%|███▌      | 142392/400000 [00:16<00:29, 8776.65it/s] 36%|███▌      | 143310/400000 [00:16<00:28, 8892.35it/s] 36%|███▌      | 144213/400000 [00:17<00:28, 8931.03it/s] 36%|███▋      | 145107/400000 [00:17<00:28, 8865.10it/s] 36%|███▋      | 145995/400000 [00:17<00:28, 8763.20it/s] 37%|███▋      | 146873/400000 [00:17<00:28, 8762.84it/s] 37%|███▋      | 147790/400000 [00:17<00:28, 8880.23it/s] 37%|███▋      | 148684/400000 [00:17<00:28, 8895.90it/s] 37%|███▋      | 149581/400000 [00:17<00:28, 8916.41it/s] 38%|███▊      | 150474/400000 [00:17<00:28, 8869.62it/s] 38%|███▊      | 151406/400000 [00:17<00:27, 8997.68it/s] 38%|███▊      | 152307/400000 [00:17<00:27, 8948.84it/s] 38%|███▊      | 153203/400000 [00:18<00:28, 8728.13it/s] 39%|███▊      | 154100/400000 [00:18<00:27, 8797.37it/s] 39%|███▊      | 154981/400000 [00:18<00:28, 8586.25it/s] 39%|███▉      | 155842/400000 [00:18<00:29, 8301.60it/s] 39%|███▉      | 156742/400000 [00:18<00:28, 8499.40it/s] 39%|███▉      | 157600/400000 [00:18<00:28, 8521.26it/s] 40%|███▉      | 158495/400000 [00:18<00:27, 8645.20it/s] 40%|███▉      | 159373/400000 [00:18<00:27, 8682.90it/s] 40%|████      | 160243/400000 [00:18<00:27, 8632.40it/s] 40%|████      | 161143/400000 [00:18<00:27, 8736.69it/s] 41%|████      | 162018/400000 [00:19<00:27, 8629.69it/s] 41%|████      | 162888/400000 [00:19<00:27, 8648.02it/s] 41%|████      | 163754/400000 [00:19<00:27, 8574.13it/s] 41%|████      | 164613/400000 [00:19<00:27, 8498.42it/s] 41%|████▏     | 165483/400000 [00:19<00:27, 8556.71it/s] 42%|████▏     | 166340/400000 [00:19<00:27, 8504.00it/s] 42%|████▏     | 167191/400000 [00:19<00:27, 8489.26it/s] 42%|████▏     | 168057/400000 [00:19<00:27, 8538.38it/s] 42%|████▏     | 168917/400000 [00:19<00:27, 8555.91it/s] 42%|████▏     | 169773/400000 [00:19<00:27, 8513.94it/s] 43%|████▎     | 170625/400000 [00:20<00:27, 8381.75it/s] 43%|████▎     | 171487/400000 [00:20<00:27, 8449.71it/s] 43%|████▎     | 172342/400000 [00:20<00:26, 8479.01it/s] 43%|████▎     | 173203/400000 [00:20<00:26, 8516.25it/s] 44%|████▎     | 174055/400000 [00:20<00:26, 8499.92it/s] 44%|████▎     | 174912/400000 [00:20<00:26, 8519.36it/s] 44%|████▍     | 175786/400000 [00:20<00:26, 8581.85it/s] 44%|████▍     | 176646/400000 [00:20<00:26, 8585.86it/s] 44%|████▍     | 177509/400000 [00:20<00:25, 8596.86it/s] 45%|████▍     | 178369/400000 [00:20<00:26, 8509.77it/s] 45%|████▍     | 179221/400000 [00:21<00:26, 8459.11it/s] 45%|████▌     | 180085/400000 [00:21<00:25, 8512.56it/s] 45%|████▌     | 180937/400000 [00:21<00:26, 8229.74it/s] 45%|████▌     | 181771/400000 [00:21<00:26, 8260.55it/s] 46%|████▌     | 182666/400000 [00:21<00:25, 8453.50it/s] 46%|████▌     | 183514/400000 [00:21<00:25, 8410.83it/s] 46%|████▌     | 184386/400000 [00:21<00:25, 8500.23it/s] 46%|████▋     | 185240/400000 [00:21<00:25, 8510.56it/s] 47%|████▋     | 186142/400000 [00:21<00:24, 8655.87it/s] 47%|████▋     | 187037/400000 [00:21<00:24, 8742.04it/s] 47%|████▋     | 187913/400000 [00:22<00:24, 8576.21it/s] 47%|████▋     | 188773/400000 [00:22<00:24, 8531.81it/s] 47%|████▋     | 189633/400000 [00:22<00:24, 8551.48it/s] 48%|████▊     | 190513/400000 [00:22<00:24, 8623.18it/s] 48%|████▊     | 191390/400000 [00:22<00:24, 8666.33it/s] 48%|████▊     | 192295/400000 [00:22<00:23, 8776.69it/s] 48%|████▊     | 193174/400000 [00:22<00:23, 8725.83it/s] 49%|████▊     | 194048/400000 [00:22<00:23, 8710.35it/s] 49%|████▊     | 194928/400000 [00:22<00:23, 8736.66it/s] 49%|████▉     | 195802/400000 [00:23<00:23, 8611.65it/s] 49%|████▉     | 196664/400000 [00:23<00:23, 8555.75it/s] 49%|████▉     | 197521/400000 [00:23<00:23, 8482.90it/s] 50%|████▉     | 198380/400000 [00:23<00:23, 8513.36it/s] 50%|████▉     | 199253/400000 [00:23<00:23, 8577.05it/s] 50%|█████     | 200138/400000 [00:23<00:23, 8655.47it/s] 50%|█████     | 201031/400000 [00:23<00:22, 8733.47it/s] 50%|█████     | 201919/400000 [00:23<00:22, 8776.27it/s] 51%|█████     | 202798/400000 [00:23<00:23, 8553.98it/s] 51%|█████     | 203655/400000 [00:23<00:23, 8260.02it/s] 51%|█████     | 204485/400000 [00:24<00:24, 8122.44it/s] 51%|█████▏    | 205338/400000 [00:24<00:23, 8239.53it/s] 52%|█████▏    | 206178/400000 [00:24<00:23, 8286.03it/s] 52%|█████▏    | 207020/400000 [00:24<00:23, 8325.54it/s] 52%|█████▏    | 207861/400000 [00:24<00:23, 8350.11it/s] 52%|█████▏    | 208703/400000 [00:24<00:22, 8370.02it/s] 52%|█████▏    | 209541/400000 [00:24<00:23, 8270.17it/s] 53%|█████▎    | 210412/400000 [00:24<00:22, 8395.65it/s] 53%|█████▎    | 211253/400000 [00:24<00:22, 8389.30it/s] 53%|█████▎    | 212093/400000 [00:24<00:22, 8380.04it/s] 53%|█████▎    | 212932/400000 [00:25<00:22, 8348.65it/s] 53%|█████▎    | 213789/400000 [00:25<00:22, 8412.28it/s] 54%|█████▎    | 214656/400000 [00:25<00:21, 8487.24it/s] 54%|█████▍    | 215510/400000 [00:25<00:21, 8500.79it/s] 54%|█████▍    | 216361/400000 [00:25<00:21, 8467.51it/s] 54%|█████▍    | 217208/400000 [00:25<00:21, 8363.35it/s] 55%|█████▍    | 218051/400000 [00:25<00:21, 8381.69it/s] 55%|█████▍    | 218895/400000 [00:25<00:21, 8397.53it/s] 55%|█████▍    | 219735/400000 [00:25<00:21, 8389.48it/s] 55%|█████▌    | 220583/400000 [00:25<00:21, 8415.08it/s] 55%|█████▌    | 221431/400000 [00:26<00:21, 8433.92it/s] 56%|█████▌    | 222275/400000 [00:26<00:21, 8399.16it/s] 56%|█████▌    | 223132/400000 [00:26<00:20, 8448.42it/s] 56%|█████▌    | 223977/400000 [00:26<00:20, 8439.37it/s] 56%|█████▌    | 224822/400000 [00:26<00:20, 8374.58it/s] 56%|█████▋    | 225660/400000 [00:26<00:20, 8371.71it/s] 57%|█████▋    | 226505/400000 [00:26<00:20, 8394.61it/s] 57%|█████▋    | 227405/400000 [00:26<00:20, 8565.83it/s] 57%|█████▋    | 228282/400000 [00:26<00:19, 8622.98it/s] 57%|█████▋    | 229146/400000 [00:26<00:19, 8621.49it/s] 58%|█████▊    | 230009/400000 [00:27<00:19, 8551.64it/s] 58%|█████▊    | 230884/400000 [00:27<00:19, 8609.95it/s] 58%|█████▊    | 231746/400000 [00:27<00:19, 8499.15it/s] 58%|█████▊    | 232597/400000 [00:27<00:19, 8371.09it/s] 58%|█████▊    | 233453/400000 [00:27<00:19, 8426.46it/s] 59%|█████▊    | 234297/400000 [00:27<00:19, 8411.38it/s] 59%|█████▉    | 235145/400000 [00:27<00:19, 8431.32it/s] 59%|█████▉    | 235999/400000 [00:27<00:19, 8462.52it/s] 59%|█████▉    | 236869/400000 [00:27<00:19, 8531.89it/s] 59%|█████▉    | 237747/400000 [00:27<00:18, 8603.33it/s] 60%|█████▉    | 238608/400000 [00:28<00:19, 8491.08it/s] 60%|█████▉    | 239472/400000 [00:28<00:18, 8534.19it/s] 60%|██████    | 240326/400000 [00:28<00:19, 8170.67it/s] 60%|██████    | 241147/400000 [00:28<00:19, 8143.21it/s] 61%|██████    | 242007/400000 [00:28<00:19, 8274.75it/s] 61%|██████    | 242867/400000 [00:28<00:18, 8369.10it/s] 61%|██████    | 243735/400000 [00:28<00:18, 8458.26it/s] 61%|██████    | 244604/400000 [00:28<00:18, 8525.64it/s] 61%|██████▏   | 245477/400000 [00:28<00:18, 8584.35it/s] 62%|██████▏   | 246348/400000 [00:28<00:17, 8619.44it/s] 62%|██████▏   | 247211/400000 [00:29<00:17, 8565.36it/s] 62%|██████▏   | 248069/400000 [00:29<00:17, 8541.46it/s] 62%|██████▏   | 248926/400000 [00:29<00:17, 8549.29it/s] 62%|██████▏   | 249782/400000 [00:29<00:18, 8278.62it/s] 63%|██████▎   | 250620/400000 [00:29<00:17, 8308.77it/s] 63%|██████▎   | 251453/400000 [00:29<00:17, 8289.33it/s] 63%|██████▎   | 252286/400000 [00:29<00:17, 8299.84it/s] 63%|██████▎   | 253148/400000 [00:29<00:17, 8391.48it/s] 63%|██████▎   | 253994/400000 [00:29<00:17, 8410.38it/s] 64%|██████▎   | 254836/400000 [00:30<00:17, 8391.35it/s] 64%|██████▍   | 255676/400000 [00:30<00:17, 8355.55it/s] 64%|██████▍   | 256512/400000 [00:30<00:17, 8347.74it/s] 64%|██████▍   | 257347/400000 [00:30<00:17, 8251.20it/s] 65%|██████▍   | 258173/400000 [00:30<00:17, 8173.12it/s] 65%|██████▍   | 259012/400000 [00:30<00:17, 8235.01it/s] 65%|██████▍   | 259865/400000 [00:30<00:16, 8319.45it/s] 65%|██████▌   | 260743/400000 [00:30<00:16, 8451.97it/s] 65%|██████▌   | 261611/400000 [00:30<00:16, 8515.77it/s] 66%|██████▌   | 262464/400000 [00:30<00:16, 8476.19it/s] 66%|██████▌   | 263330/400000 [00:31<00:16, 8528.30it/s] 66%|██████▌   | 264184/400000 [00:31<00:16, 8470.10it/s] 66%|██████▋   | 265043/400000 [00:31<00:15, 8503.56it/s] 66%|██████▋   | 265894/400000 [00:31<00:15, 8406.56it/s] 67%|██████▋   | 266749/400000 [00:31<00:15, 8447.97it/s] 67%|██████▋   | 267625/400000 [00:31<00:15, 8537.68it/s] 67%|██████▋   | 268504/400000 [00:31<00:15, 8611.69it/s] 67%|██████▋   | 269398/400000 [00:31<00:15, 8705.15it/s] 68%|██████▊   | 270302/400000 [00:31<00:14, 8802.38it/s] 68%|██████▊   | 271204/400000 [00:31<00:14, 8864.71it/s] 68%|██████▊   | 272092/400000 [00:32<00:14, 8800.18it/s] 68%|██████▊   | 272973/400000 [00:32<00:14, 8637.93it/s] 68%|██████▊   | 273841/400000 [00:32<00:14, 8650.33it/s] 69%|██████▊   | 274707/400000 [00:32<00:14, 8629.34it/s] 69%|██████▉   | 275589/400000 [00:32<00:14, 8685.19it/s] 69%|██████▉   | 276458/400000 [00:32<00:14, 8673.93it/s] 69%|██████▉   | 277326/400000 [00:32<00:14, 8611.90it/s] 70%|██████▉   | 278188/400000 [00:32<00:14, 8593.89it/s] 70%|██████▉   | 279060/400000 [00:32<00:14, 8628.94it/s] 70%|██████▉   | 279924/400000 [00:32<00:13, 8620.51it/s] 70%|███████   | 280787/400000 [00:33<00:13, 8563.13it/s] 70%|███████   | 281644/400000 [00:33<00:14, 8436.18it/s] 71%|███████   | 282490/400000 [00:33<00:13, 8440.50it/s] 71%|███████   | 283335/400000 [00:33<00:13, 8436.88it/s] 71%|███████   | 284187/400000 [00:33<00:13, 8461.31it/s] 71%|███████▏  | 285044/400000 [00:33<00:13, 8493.61it/s] 71%|███████▏  | 285901/400000 [00:33<00:13, 8514.16it/s] 72%|███████▏  | 286773/400000 [00:33<00:13, 8574.03it/s] 72%|███████▏  | 287631/400000 [00:33<00:13, 8549.10it/s] 72%|███████▏  | 288487/400000 [00:33<00:13, 8492.02it/s] 72%|███████▏  | 289337/400000 [00:34<00:13, 8431.37it/s] 73%|███████▎  | 290181/400000 [00:34<00:13, 8315.54it/s] 73%|███████▎  | 291022/400000 [00:34<00:13, 8342.68it/s] 73%|███████▎  | 291882/400000 [00:34<00:12, 8417.92it/s] 73%|███████▎  | 292734/400000 [00:34<00:12, 8446.05it/s] 73%|███████▎  | 293591/400000 [00:34<00:12, 8481.40it/s] 74%|███████▎  | 294440/400000 [00:34<00:12, 8466.42it/s] 74%|███████▍  | 295308/400000 [00:34<00:12, 8529.28it/s] 74%|███████▍  | 296171/400000 [00:34<00:12, 8558.74it/s] 74%|███████▍  | 297028/400000 [00:34<00:12, 8383.46it/s] 74%|███████▍  | 297868/400000 [00:35<00:12, 8157.97it/s] 75%|███████▍  | 298722/400000 [00:35<00:12, 8267.52it/s] 75%|███████▍  | 299593/400000 [00:35<00:11, 8395.35it/s] 75%|███████▌  | 300444/400000 [00:35<00:11, 8429.21it/s] 75%|███████▌  | 301301/400000 [00:35<00:11, 8468.21it/s] 76%|███████▌  | 302149/400000 [00:35<00:11, 8445.25it/s] 76%|███████▌  | 302995/400000 [00:35<00:11, 8411.15it/s] 76%|███████▌  | 303840/400000 [00:35<00:11, 8420.92it/s] 76%|███████▌  | 304687/400000 [00:35<00:11, 8433.23it/s] 76%|███████▋  | 305531/400000 [00:35<00:11, 8321.64it/s] 77%|███████▋  | 306391/400000 [00:36<00:11, 8402.84it/s] 77%|███████▋  | 307232/400000 [00:36<00:11, 8359.48it/s] 77%|███████▋  | 308069/400000 [00:36<00:11, 8229.43it/s] 77%|███████▋  | 308893/400000 [00:36<00:11, 7974.46it/s] 77%|███████▋  | 309693/400000 [00:36<00:11, 7956.03it/s] 78%|███████▊  | 310548/400000 [00:36<00:11, 8124.12it/s] 78%|███████▊  | 311363/400000 [00:36<00:10, 8119.23it/s] 78%|███████▊  | 312199/400000 [00:36<00:10, 8189.32it/s] 78%|███████▊  | 313020/400000 [00:36<00:10, 8173.31it/s] 78%|███████▊  | 313867/400000 [00:37<00:10, 8259.37it/s] 79%|███████▊  | 314733/400000 [00:37<00:10, 8374.52it/s] 79%|███████▉  | 315596/400000 [00:37<00:09, 8448.15it/s] 79%|███████▉  | 316472/400000 [00:37<00:09, 8538.82it/s] 79%|███████▉  | 317335/400000 [00:37<00:09, 8563.93it/s] 80%|███████▉  | 318192/400000 [00:37<00:09, 8262.35it/s] 80%|███████▉  | 319021/400000 [00:37<00:09, 8258.81it/s] 80%|███████▉  | 319857/400000 [00:37<00:09, 8287.54it/s] 80%|████████  | 320708/400000 [00:37<00:09, 8352.18it/s] 80%|████████  | 321558/400000 [00:37<00:09, 8395.58it/s] 81%|████████  | 322409/400000 [00:38<00:09, 8428.63it/s] 81%|████████  | 323253/400000 [00:38<00:09, 8358.60it/s] 81%|████████  | 324090/400000 [00:38<00:09, 8236.54it/s] 81%|████████  | 324915/400000 [00:38<00:09, 8226.90it/s] 81%|████████▏ | 325775/400000 [00:38<00:08, 8334.45it/s] 82%|████████▏ | 326621/400000 [00:38<00:08, 8370.86it/s] 82%|████████▏ | 327459/400000 [00:38<00:08, 8227.70it/s] 82%|████████▏ | 328331/400000 [00:38<00:08, 8367.13it/s] 82%|████████▏ | 329191/400000 [00:38<00:08, 8434.73it/s] 83%|████████▎ | 330088/400000 [00:38<00:08, 8583.62it/s] 83%|████████▎ | 330948/400000 [00:39<00:08, 8527.69it/s] 83%|████████▎ | 331802/400000 [00:39<00:08, 8344.92it/s] 83%|████████▎ | 332682/400000 [00:39<00:07, 8476.20it/s] 83%|████████▎ | 333585/400000 [00:39<00:07, 8633.89it/s] 84%|████████▎ | 334470/400000 [00:39<00:07, 8697.41it/s] 84%|████████▍ | 335342/400000 [00:39<00:07, 8618.25it/s] 84%|████████▍ | 336205/400000 [00:39<00:07, 8540.80it/s] 84%|████████▍ | 337079/400000 [00:39<00:07, 8598.91it/s] 84%|████████▍ | 337977/400000 [00:39<00:07, 8709.53it/s] 85%|████████▍ | 338867/400000 [00:39<00:06, 8762.63it/s] 85%|████████▍ | 339744/400000 [00:40<00:06, 8728.34it/s] 85%|████████▌ | 340618/400000 [00:40<00:06, 8658.85it/s] 85%|████████▌ | 341485/400000 [00:40<00:06, 8547.40it/s] 86%|████████▌ | 342341/400000 [00:40<00:06, 8529.19it/s] 86%|████████▌ | 343195/400000 [00:40<00:06, 8491.09it/s] 86%|████████▌ | 344117/400000 [00:40<00:06, 8695.46it/s] 86%|████████▌ | 344989/400000 [00:40<00:06, 8683.77it/s] 86%|████████▋ | 345882/400000 [00:40<00:06, 8753.16it/s] 87%|████████▋ | 346790/400000 [00:40<00:06, 8846.11it/s] 87%|████████▋ | 347679/400000 [00:40<00:05, 8858.09it/s] 87%|████████▋ | 348587/400000 [00:41<00:05, 8922.58it/s] 87%|████████▋ | 349507/400000 [00:41<00:05, 9002.57it/s] 88%|████████▊ | 350455/400000 [00:41<00:05, 9138.31it/s] 88%|████████▊ | 351370/400000 [00:41<00:05, 8940.55it/s] 88%|████████▊ | 352266/400000 [00:41<00:05, 8883.31it/s] 88%|████████▊ | 353156/400000 [00:41<00:05, 8812.80it/s] 89%|████████▊ | 354039/400000 [00:41<00:05, 8340.02it/s] 89%|████████▊ | 354879/400000 [00:41<00:05, 8258.32it/s] 89%|████████▉ | 355732/400000 [00:41<00:05, 8336.95it/s] 89%|████████▉ | 356588/400000 [00:41<00:05, 8401.58it/s] 89%|████████▉ | 357431/400000 [00:42<00:05, 8401.58it/s] 90%|████████▉ | 358273/400000 [00:42<00:04, 8347.64it/s] 90%|████████▉ | 359141/400000 [00:42<00:04, 8442.27it/s] 90%|█████████ | 360002/400000 [00:42<00:04, 8489.89it/s] 90%|█████████ | 360876/400000 [00:42<00:04, 8563.28it/s] 90%|█████████ | 361734/400000 [00:42<00:04, 8558.66it/s] 91%|█████████ | 362591/400000 [00:42<00:04, 8530.04it/s] 91%|█████████ | 363452/400000 [00:42<00:04, 8552.23it/s] 91%|█████████ | 364308/400000 [00:42<00:04, 8547.97it/s] 91%|█████████▏| 365187/400000 [00:43<00:04, 8618.15it/s] 92%|█████████▏| 366050/400000 [00:43<00:03, 8602.41it/s] 92%|█████████▏| 366911/400000 [00:43<00:03, 8481.10it/s] 92%|█████████▏| 367760/400000 [00:43<00:03, 8473.91it/s] 92%|█████████▏| 368608/400000 [00:43<00:03, 8466.68it/s] 92%|█████████▏| 369455/400000 [00:43<00:03, 8251.54it/s] 93%|█████████▎| 370305/400000 [00:43<00:03, 8324.48it/s] 93%|█████████▎| 371139/400000 [00:43<00:03, 8307.32it/s] 93%|█████████▎| 371979/400000 [00:43<00:03, 8333.77it/s] 93%|█████████▎| 372835/400000 [00:43<00:03, 8397.58it/s] 93%|█████████▎| 373695/400000 [00:44<00:03, 8454.93it/s] 94%|█████████▎| 374541/400000 [00:44<00:03, 8352.68it/s] 94%|█████████▍| 375377/400000 [00:44<00:02, 8301.12it/s] 94%|█████████▍| 376234/400000 [00:44<00:02, 8377.07it/s] 94%|█████████▍| 377111/400000 [00:44<00:02, 8489.46it/s] 94%|█████████▍| 377979/400000 [00:44<00:02, 8544.15it/s] 95%|█████████▍| 378835/400000 [00:44<00:02, 8508.35it/s] 95%|█████████▍| 379707/400000 [00:44<00:02, 8570.65it/s] 95%|█████████▌| 380565/400000 [00:44<00:02, 8523.95it/s] 95%|█████████▌| 381418/400000 [00:44<00:02, 8491.04it/s] 96%|█████████▌| 382268/400000 [00:45<00:02, 8477.97it/s] 96%|█████████▌| 383116/400000 [00:45<00:01, 8446.24it/s] 96%|█████████▌| 383961/400000 [00:45<00:01, 8405.14it/s] 96%|█████████▌| 384816/400000 [00:45<00:01, 8447.86it/s] 96%|█████████▋| 385662/400000 [00:45<00:01, 8448.95it/s] 97%|█████████▋| 386512/400000 [00:45<00:01, 8461.95it/s] 97%|█████████▋| 387359/400000 [00:45<00:01, 8391.06it/s] 97%|█████████▋| 388199/400000 [00:45<00:01, 8327.56it/s] 97%|█████████▋| 389033/400000 [00:45<00:01, 8329.55it/s] 97%|█████████▋| 389908/400000 [00:45<00:01, 8450.70it/s] 98%|█████████▊| 390765/400000 [00:46<00:01, 8485.62it/s] 98%|█████████▊| 391616/400000 [00:46<00:00, 8490.34it/s] 98%|█████████▊| 392466/400000 [00:46<00:00, 8479.17it/s] 98%|█████████▊| 393322/400000 [00:46<00:00, 8503.17it/s] 99%|█████████▊| 394175/400000 [00:46<00:00, 8508.80it/s] 99%|█████████▉| 395040/400000 [00:46<00:00, 8549.03it/s] 99%|█████████▉| 395896/400000 [00:46<00:00, 8498.91it/s] 99%|█████████▉| 396747/400000 [00:46<00:00, 8470.06it/s] 99%|█████████▉| 397663/400000 [00:46<00:00, 8664.27it/s]100%|█████████▉| 398531/400000 [00:46<00:00, 8576.56it/s]100%|█████████▉| 399403/400000 [00:47<00:00, 8618.38it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8488.97it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fa132e63cc0>, <torchtext.data.dataset.TabularDataset object at 0x7fa132e63e10>, <torchtext.vocab.Vocab object at 0x7fa132e63d30>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.13 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 14.98 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 14.98 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 12.89 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 12.89 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.20 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.20 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.71 file/s]2020-07-06 18:18:31.577472: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-06 18:18:31.581954: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-06 18:18:31.582136: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5624f1ce75d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-06 18:18:31.582150: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███▏      | 3104768/9912422 [00:00<00:00, 27002906.83it/s]9920512it [00:00, 35101715.00it/s]                             
0it [00:00, ?it/s]32768it [00:00, 706089.73it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1009461.45it/s]1654784it [00:00, 12363735.25it/s]                           
0it [00:00, ?it/s]8192it [00:00, 222243.53it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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


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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f1bda6f9d90> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f1bda6f9d90> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f1c454429d8> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f1c454429d8> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f1c637cce18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f1c637cce18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1bf2ce72f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1bf2ce72f0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1bf2ce72f0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 40%|████      | 3973120/9912422 [00:00<00:00, 39359721.46it/s]9920512it [00:00, 36162613.36it/s]                             
0it [00:00, ?it/s]32768it [00:00, 619697.33it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 146522.78it/s]1654784it [00:00, 10254622.85it/s]                         
0it [00:00, ?it/s]8192it [00:00, 212830.23it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bf00cf2b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bf00c09e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1bf2cdbea0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1bf2cdbea0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f1bf2cdbea0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:34,  1.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:34,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:34,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:33,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:33,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:32,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:32,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:31,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:04,  2.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:04,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:03,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:03,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:03,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:02,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:02,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:01,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:01,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:00,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:00,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:42,  3.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:42,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:42,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:41,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:41,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:41,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:41,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:40,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:40,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:40,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:39,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:27,  4.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:27,  4.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:27,  4.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:27,  4.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:27,  4.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:01<00:19,  6.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:19,  6.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:19,  6.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:19,  6.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:19,  6.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:18,  6.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:18,  6.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:13,  8.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:13,  8.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:13,  8.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:13,  8.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:13,  8.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:13,  8.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:13,  8.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:09, 11.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:09, 11.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:09, 11.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:09, 11.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 15.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 19.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 24.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 24.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 24.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 24.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 24.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 24.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 29.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 29.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 29.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 29.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 29.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 29.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 29.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 34.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 34.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 34.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 34.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 34.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 34.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 34.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 38.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 38.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 38.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 38.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 42.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 42.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 42.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 42.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 42.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 42.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 42.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 45.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 45.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 45.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 45.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 45.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 45.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 45.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 48.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 48.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 48.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 48.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 48.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 48.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 48.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 49.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 49.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 49.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 49.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 49.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 49.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 49.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 52.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 52.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 52.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 52.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 52.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 52.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 52.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 53.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 53.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 53.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 53.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 53.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 53.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 53.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 53.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 53.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 53.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 53.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 53.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 53.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 53.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 54.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 54.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 54.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 54.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 54.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 54.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 54.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 54.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 55.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 55.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 55.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 55.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 55.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 55.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 55.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 55.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 55.24 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 55.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 55.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 55.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 55.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 55.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 55.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 55.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 55.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 55.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 55.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 55.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.33s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.33s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 55.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.33s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 55.53 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.52s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.33s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 55.53 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.52s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.52s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 29.36 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.52s/ url]
0 examples [00:00, ? examples/s]2020-06-21 06:09:28.689698: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-21 06:09:28.703501: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-21 06:09:28.704134: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56031dc8db40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-21 06:09:28.704174: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
44 examples [00:00, 439.30 examples/s]153 examples [00:00, 535.06 examples/s]262 examples [00:00, 630.70 examples/s]367 examples [00:00, 715.60 examples/s]456 examples [00:00, 759.94 examples/s]548 examples [00:00, 801.16 examples/s]659 examples [00:00, 872.01 examples/s]771 examples [00:00, 932.89 examples/s]883 examples [00:00, 981.11 examples/s]991 examples [00:01, 1007.70 examples/s]1103 examples [00:01, 1038.32 examples/s]1209 examples [00:01, 1010.71 examples/s]1319 examples [00:01, 1034.01 examples/s]1424 examples [00:01, 1012.41 examples/s]1530 examples [00:01, 1025.87 examples/s]1642 examples [00:01, 1029.80 examples/s]1749 examples [00:01, 1041.18 examples/s]1854 examples [00:01, 1035.88 examples/s]1958 examples [00:01, 1033.25 examples/s]2062 examples [00:02, 1017.81 examples/s]2170 examples [00:02, 1033.87 examples/s]2280 examples [00:02, 1051.05 examples/s]2386 examples [00:02, 1030.45 examples/s]2490 examples [00:02, 995.84 examples/s] 2590 examples [00:02, 987.03 examples/s]2699 examples [00:02, 1013.57 examples/s]2803 examples [00:02, 1020.10 examples/s]2906 examples [00:02, 1023.05 examples/s]3014 examples [00:02, 1037.46 examples/s]3124 examples [00:03, 1053.20 examples/s]3235 examples [00:03, 1069.49 examples/s]3347 examples [00:03, 1082.24 examples/s]3459 examples [00:03, 1091.59 examples/s]3569 examples [00:03, 1079.67 examples/s]3678 examples [00:03, 1054.17 examples/s]3785 examples [00:03, 1056.53 examples/s]3896 examples [00:03, 1069.68 examples/s]4007 examples [00:03, 1081.30 examples/s]4118 examples [00:03, 1087.47 examples/s]4228 examples [00:04, 1089.18 examples/s]4338 examples [00:04, 1091.29 examples/s]4448 examples [00:04, 1054.29 examples/s]4557 examples [00:04, 1062.29 examples/s]4665 examples [00:04, 1066.73 examples/s]4773 examples [00:04, 1067.74 examples/s]4883 examples [00:04, 1076.84 examples/s]4994 examples [00:04, 1085.61 examples/s]5105 examples [00:04, 1090.12 examples/s]5216 examples [00:05, 1093.71 examples/s]5326 examples [00:05, 1088.17 examples/s]5435 examples [00:05, 1072.01 examples/s]5543 examples [00:05, 1067.59 examples/s]5650 examples [00:05, 1061.82 examples/s]5757 examples [00:05, 1053.47 examples/s]5866 examples [00:05, 1061.70 examples/s]5973 examples [00:05, 1045.42 examples/s]6080 examples [00:05, 1050.74 examples/s]6188 examples [00:05, 1058.46 examples/s]6299 examples [00:06, 1070.74 examples/s]6409 examples [00:06, 1077.04 examples/s]6521 examples [00:06, 1088.18 examples/s]6632 examples [00:06, 1094.33 examples/s]6742 examples [00:06, 1064.26 examples/s]6853 examples [00:06, 1077.46 examples/s]6961 examples [00:06, 1037.61 examples/s]7072 examples [00:06, 1058.28 examples/s]7185 examples [00:06, 1076.31 examples/s]7293 examples [00:06, 1072.77 examples/s]7404 examples [00:07, 1082.27 examples/s]7513 examples [00:07, 1080.18 examples/s]7622 examples [00:07, 1063.16 examples/s]7729 examples [00:07, 1007.39 examples/s]7831 examples [00:07, 998.28 examples/s] 7940 examples [00:07, 1023.78 examples/s]8043 examples [00:07, 1015.34 examples/s]8154 examples [00:07, 1039.66 examples/s]8264 examples [00:07, 1056.26 examples/s]8372 examples [00:07, 1061.44 examples/s]8482 examples [00:08, 1071.99 examples/s]8591 examples [00:08, 1074.86 examples/s]8699 examples [00:08, 1072.92 examples/s]8807 examples [00:08, 1073.28 examples/s]8917 examples [00:08, 1079.47 examples/s]9026 examples [00:08, 1078.86 examples/s]9138 examples [00:08, 1090.17 examples/s]9248 examples [00:08, 1076.03 examples/s]9359 examples [00:08, 1085.28 examples/s]9468 examples [00:09, 1060.17 examples/s]9575 examples [00:09, 1053.89 examples/s]9681 examples [00:09, 1044.48 examples/s]9786 examples [00:09, 1019.61 examples/s]9889 examples [00:09, 1015.51 examples/s]9998 examples [00:09, 1036.10 examples/s]10102 examples [00:09, 986.25 examples/s]10213 examples [00:09, 1018.14 examples/s]10324 examples [00:09, 1042.70 examples/s]10429 examples [00:09, 1038.84 examples/s]10540 examples [00:10, 1056.83 examples/s]10647 examples [00:10, 1059.41 examples/s]10754 examples [00:10, 1055.34 examples/s]10865 examples [00:10, 1068.89 examples/s]10973 examples [00:10, 1027.21 examples/s]11083 examples [00:10, 1046.09 examples/s]11189 examples [00:10, 1043.90 examples/s]11294 examples [00:10, 1014.38 examples/s]11396 examples [00:10, 1003.77 examples/s]11504 examples [00:10, 1025.24 examples/s]11616 examples [00:11, 1050.92 examples/s]11722 examples [00:11, 1053.46 examples/s]11832 examples [00:11, 1064.38 examples/s]11940 examples [00:11, 1067.15 examples/s]12049 examples [00:11, 1069.56 examples/s]12157 examples [00:11, 1065.70 examples/s]12265 examples [00:11, 1069.49 examples/s]12373 examples [00:11, 1053.05 examples/s]12484 examples [00:11, 1069.26 examples/s]12595 examples [00:11, 1080.00 examples/s]12707 examples [00:12, 1091.42 examples/s]12817 examples [00:12, 1054.93 examples/s]12923 examples [00:12, 1003.72 examples/s]13028 examples [00:12, 1014.68 examples/s]13131 examples [00:12, 1013.40 examples/s]13236 examples [00:12, 1021.34 examples/s]13341 examples [00:12, 1029.15 examples/s]13450 examples [00:12, 1044.89 examples/s]13561 examples [00:12, 1063.34 examples/s]13670 examples [00:13, 1071.13 examples/s]13778 examples [00:13, 1072.33 examples/s]13886 examples [00:13, 1071.53 examples/s]13997 examples [00:13, 1081.10 examples/s]14106 examples [00:13, 1061.29 examples/s]14213 examples [00:13, 1058.67 examples/s]14323 examples [00:13, 1068.40 examples/s]14430 examples [00:13, 1066.00 examples/s]14539 examples [00:13, 1072.10 examples/s]14649 examples [00:13, 1079.83 examples/s]14759 examples [00:14, 1083.16 examples/s]14868 examples [00:14, 1082.96 examples/s]14977 examples [00:14, 1067.60 examples/s]15088 examples [00:14, 1077.64 examples/s]15196 examples [00:14, 1074.74 examples/s]15307 examples [00:14, 1083.32 examples/s]15417 examples [00:14, 1086.47 examples/s]15526 examples [00:14, 1085.59 examples/s]15635 examples [00:14, 1059.77 examples/s]15742 examples [00:14, 1059.32 examples/s]15854 examples [00:15, 1074.07 examples/s]15963 examples [00:15, 1076.46 examples/s]16071 examples [00:15, 1053.42 examples/s]16177 examples [00:15, 1025.93 examples/s]16281 examples [00:15, 1028.21 examples/s]16393 examples [00:15, 1052.77 examples/s]16503 examples [00:15, 1064.15 examples/s]16610 examples [00:15, 1054.89 examples/s]16716 examples [00:15, 1024.51 examples/s]16825 examples [00:16, 1041.64 examples/s]16937 examples [00:16, 1062.98 examples/s]17048 examples [00:16, 1076.26 examples/s]17157 examples [00:16, 1079.44 examples/s]17267 examples [00:16, 1083.80 examples/s]17376 examples [00:16, 1016.61 examples/s]17483 examples [00:16, 1030.94 examples/s]17591 examples [00:16, 1044.72 examples/s]17703 examples [00:16, 1065.16 examples/s]17815 examples [00:16, 1079.71 examples/s]17925 examples [00:17, 1085.00 examples/s]18038 examples [00:17, 1096.43 examples/s]18149 examples [00:17, 1098.01 examples/s]18259 examples [00:17, 1093.47 examples/s]18369 examples [00:17, 1066.88 examples/s]18481 examples [00:17, 1081.67 examples/s]18593 examples [00:17, 1092.85 examples/s]18703 examples [00:17, 1090.95 examples/s]18813 examples [00:17, 1072.16 examples/s]18925 examples [00:17, 1083.52 examples/s]19036 examples [00:18, 1089.90 examples/s]19147 examples [00:18, 1095.58 examples/s]19257 examples [00:18, 1082.70 examples/s]19367 examples [00:18, 1085.15 examples/s]19476 examples [00:18, 1067.79 examples/s]19583 examples [00:18, 1054.46 examples/s]19694 examples [00:18, 1067.72 examples/s]19804 examples [00:18, 1075.48 examples/s]19912 examples [00:18, 1049.71 examples/s]20018 examples [00:18, 1009.49 examples/s]20130 examples [00:19, 1040.24 examples/s]20242 examples [00:19, 1060.57 examples/s]20351 examples [00:19, 1069.05 examples/s]20460 examples [00:19, 1072.76 examples/s]20568 examples [00:19, 1004.36 examples/s]20670 examples [00:19, 992.76 examples/s] 20780 examples [00:19, 1022.41 examples/s]20890 examples [00:19, 1042.72 examples/s]20996 examples [00:19, 1046.98 examples/s]21103 examples [00:20, 1053.17 examples/s]21209 examples [00:20, 1053.90 examples/s]21315 examples [00:20, 1053.51 examples/s]21424 examples [00:20, 1063.81 examples/s]21531 examples [00:20, 1056.19 examples/s]21637 examples [00:20, 1047.84 examples/s]21745 examples [00:20, 1054.82 examples/s]21855 examples [00:20, 1067.97 examples/s]21962 examples [00:20, 1067.01 examples/s]22073 examples [00:20, 1078.42 examples/s]22184 examples [00:21, 1086.41 examples/s]22296 examples [00:21, 1095.26 examples/s]22409 examples [00:21, 1103.25 examples/s]22520 examples [00:21, 1097.20 examples/s]22631 examples [00:21, 1099.53 examples/s]22742 examples [00:21, 1101.16 examples/s]22853 examples [00:21, 1091.38 examples/s]22963 examples [00:21, 1088.89 examples/s]23072 examples [00:21, 1075.51 examples/s]23180 examples [00:21, 1055.69 examples/s]23292 examples [00:22, 1072.31 examples/s]23403 examples [00:22, 1082.65 examples/s]23512 examples [00:22, 1084.32 examples/s]23621 examples [00:22, 1080.49 examples/s]23730 examples [00:22, 1065.40 examples/s]23837 examples [00:22, 1054.54 examples/s]23943 examples [00:22, 1032.19 examples/s]24053 examples [00:22, 1049.97 examples/s]24161 examples [00:22, 1056.21 examples/s]24271 examples [00:22, 1066.50 examples/s]24378 examples [00:23, 1019.53 examples/s]24484 examples [00:23, 1029.67 examples/s]24592 examples [00:23, 1043.86 examples/s]24698 examples [00:23, 1047.70 examples/s]24808 examples [00:23, 1061.71 examples/s]24920 examples [00:23, 1075.60 examples/s]25028 examples [00:23, 1054.30 examples/s]25137 examples [00:23, 1064.03 examples/s]25245 examples [00:23, 1068.14 examples/s]25352 examples [00:24, 1057.02 examples/s]25464 examples [00:24, 1073.70 examples/s]25572 examples [00:24, 1063.19 examples/s]25679 examples [00:24, 1054.12 examples/s]25785 examples [00:24, 1054.29 examples/s]25896 examples [00:24, 1068.69 examples/s]26006 examples [00:24, 1077.56 examples/s]26114 examples [00:24, 1026.64 examples/s]26225 examples [00:24, 1048.06 examples/s]26331 examples [00:24, 1045.77 examples/s]26438 examples [00:25, 1052.78 examples/s]26545 examples [00:25, 1055.52 examples/s]26654 examples [00:25, 1058.26 examples/s]26761 examples [00:25, 1061.46 examples/s]26868 examples [00:25, 1062.59 examples/s]26975 examples [00:25, 1063.67 examples/s]27082 examples [00:25, 1064.74 examples/s]27189 examples [00:25, 1060.67 examples/s]27296 examples [00:25, 1059.21 examples/s]27402 examples [00:25, 1022.24 examples/s]27509 examples [00:26, 1035.47 examples/s]27613 examples [00:26, 1031.09 examples/s]27722 examples [00:26, 1045.44 examples/s]27831 examples [00:26, 1056.64 examples/s]27937 examples [00:26, 1044.13 examples/s]28046 examples [00:26, 1055.01 examples/s]28152 examples [00:26, 1043.91 examples/s]28262 examples [00:26, 1059.71 examples/s]28372 examples [00:26, 1069.25 examples/s]28480 examples [00:26, 1060.24 examples/s]28589 examples [00:27, 1068.72 examples/s]28699 examples [00:27, 1076.69 examples/s]28808 examples [00:27, 1079.85 examples/s]28919 examples [00:27, 1086.52 examples/s]29028 examples [00:27, 1070.25 examples/s]29136 examples [00:27, 1034.87 examples/s]29240 examples [00:27, 1024.42 examples/s]29344 examples [00:27, 1026.32 examples/s]29447 examples [00:27, 1022.52 examples/s]29551 examples [00:27, 1025.47 examples/s]29659 examples [00:28, 1039.25 examples/s]29766 examples [00:28, 1046.27 examples/s]29878 examples [00:28, 1064.94 examples/s]29985 examples [00:28, 1055.24 examples/s]30091 examples [00:28, 947.43 examples/s] 30193 examples [00:28, 966.90 examples/s]30304 examples [00:28, 1005.63 examples/s]30414 examples [00:28, 1030.79 examples/s]30523 examples [00:28, 1046.15 examples/s]30634 examples [00:29, 1062.36 examples/s]30741 examples [00:29, 1036.74 examples/s]30852 examples [00:29, 1055.72 examples/s]30962 examples [00:29, 1065.94 examples/s]31069 examples [00:29, 1019.80 examples/s]31176 examples [00:29, 1032.84 examples/s]31283 examples [00:29, 1043.58 examples/s]31394 examples [00:29, 1060.09 examples/s]31505 examples [00:29, 1072.20 examples/s]31614 examples [00:29, 1077.03 examples/s]31722 examples [00:30, 1077.35 examples/s]31833 examples [00:30, 1086.73 examples/s]31945 examples [00:30, 1096.27 examples/s]32056 examples [00:30, 1098.02 examples/s]32167 examples [00:30, 1099.63 examples/s]32278 examples [00:30, 1081.58 examples/s]32388 examples [00:30, 1086.86 examples/s]32498 examples [00:30, 1089.18 examples/s]32609 examples [00:30, 1093.26 examples/s]32719 examples [00:30, 1093.06 examples/s]32829 examples [00:31, 1078.30 examples/s]32937 examples [00:31, 1030.94 examples/s]33045 examples [00:31, 1042.61 examples/s]33153 examples [00:31, 1052.31 examples/s]33262 examples [00:31, 1061.00 examples/s]33369 examples [00:31, 1038.91 examples/s]33478 examples [00:31, 1053.27 examples/s]33584 examples [00:31, 1032.64 examples/s]33689 examples [00:31, 1037.30 examples/s]33799 examples [00:32, 1052.84 examples/s]33906 examples [00:32, 1057.59 examples/s]34012 examples [00:32, 1039.26 examples/s]34123 examples [00:32, 1057.11 examples/s]34233 examples [00:32, 1068.08 examples/s]34340 examples [00:32, 1051.10 examples/s]34451 examples [00:32, 1065.88 examples/s]34562 examples [00:32, 1078.68 examples/s]34671 examples [00:32, 1022.42 examples/s]34774 examples [00:32, 1001.40 examples/s]34875 examples [00:33, 965.51 examples/s] 34982 examples [00:33, 992.49 examples/s]35085 examples [00:33, 1001.83 examples/s]35195 examples [00:33, 1027.55 examples/s]35305 examples [00:33, 1045.93 examples/s]35411 examples [00:33, 1042.92 examples/s]35516 examples [00:33, 1012.44 examples/s]35623 examples [00:33, 1028.90 examples/s]35733 examples [00:33, 1048.83 examples/s]35839 examples [00:33, 1035.72 examples/s]35943 examples [00:34, 1024.78 examples/s]36053 examples [00:34, 1045.58 examples/s]36164 examples [00:34, 1061.12 examples/s]36271 examples [00:34, 1061.37 examples/s]36378 examples [00:34, 1042.19 examples/s]36486 examples [00:34, 1051.44 examples/s]36592 examples [00:34, 1049.03 examples/s]36699 examples [00:34, 1053.52 examples/s]36809 examples [00:34, 1065.17 examples/s]36918 examples [00:35, 1071.05 examples/s]37026 examples [00:35, 1065.39 examples/s]37133 examples [00:35, 1035.32 examples/s]37243 examples [00:35, 1051.65 examples/s]37352 examples [00:35, 1062.09 examples/s]37461 examples [00:35, 1068.77 examples/s]37569 examples [00:35, 1064.17 examples/s]37676 examples [00:35, 1050.45 examples/s]37787 examples [00:35, 1067.20 examples/s]37897 examples [00:35, 1074.82 examples/s]38006 examples [00:36, 1078.44 examples/s]38114 examples [00:36, 1056.41 examples/s]38224 examples [00:36, 1067.26 examples/s]38334 examples [00:36, 1074.85 examples/s]38443 examples [00:36, 1077.96 examples/s]38553 examples [00:36, 1081.98 examples/s]38662 examples [00:36, 1078.77 examples/s]38773 examples [00:36, 1085.58 examples/s]38884 examples [00:36, 1092.08 examples/s]38994 examples [00:36, 1089.30 examples/s]39104 examples [00:37, 1091.03 examples/s]39214 examples [00:37, 1080.20 examples/s]39323 examples [00:37, 1078.65 examples/s]39434 examples [00:37, 1085.13 examples/s]39545 examples [00:37, 1090.36 examples/s]39656 examples [00:37, 1095.54 examples/s]39766 examples [00:37, 1087.14 examples/s]39875 examples [00:37, 1047.94 examples/s]39981 examples [00:37, 1032.30 examples/s]40085 examples [00:37, 989.55 examples/s] 40193 examples [00:38, 1014.17 examples/s]40295 examples [00:38, 1003.05 examples/s]40396 examples [00:38, 989.68 examples/s] 40507 examples [00:38, 1020.46 examples/s]40617 examples [00:38, 1042.27 examples/s]40726 examples [00:38, 1052.92 examples/s]40832 examples [00:38, 1042.01 examples/s]40943 examples [00:38, 1059.06 examples/s]41051 examples [00:38, 1063.51 examples/s]41158 examples [00:39, 1060.86 examples/s]41268 examples [00:39, 1070.00 examples/s]41377 examples [00:39, 1074.90 examples/s]41488 examples [00:39, 1083.63 examples/s]41597 examples [00:39, 1073.94 examples/s]41705 examples [00:39, 1070.87 examples/s]41813 examples [00:39, 1070.78 examples/s]41921 examples [00:39, 1054.16 examples/s]42031 examples [00:39, 1065.87 examples/s]42138 examples [00:39, 1058.00 examples/s]42244 examples [00:40, 952.08 examples/s] 42342 examples [00:40, 957.90 examples/s]42449 examples [00:40, 988.92 examples/s]42561 examples [00:40, 1023.91 examples/s]42672 examples [00:40, 1046.18 examples/s]42778 examples [00:40, 1009.97 examples/s]42880 examples [00:40, 1010.03 examples/s]42982 examples [00:40, 1008.99 examples/s]43091 examples [00:40, 1031.59 examples/s]43201 examples [00:40, 1049.11 examples/s]43307 examples [00:41, 1036.37 examples/s]43411 examples [00:41, 1022.99 examples/s]43514 examples [00:41, 1023.73 examples/s]43617 examples [00:41, 983.64 examples/s] 43716 examples [00:41, 958.94 examples/s]43825 examples [00:41, 994.30 examples/s]43928 examples [00:41, 1003.23 examples/s]44039 examples [00:41, 1032.44 examples/s]44148 examples [00:41, 1046.73 examples/s]44258 examples [00:42, 1061.91 examples/s]44367 examples [00:42, 1069.84 examples/s]44475 examples [00:42, 1066.97 examples/s]44587 examples [00:42, 1080.34 examples/s]44698 examples [00:42, 1087.39 examples/s]44807 examples [00:42, 1076.40 examples/s]44917 examples [00:42, 1082.17 examples/s]45026 examples [00:42, 1077.03 examples/s]45134 examples [00:42, 1046.70 examples/s]45239 examples [00:42, 1027.96 examples/s]45346 examples [00:43, 1038.21 examples/s]45457 examples [00:43, 1056.53 examples/s]45566 examples [00:43, 1063.87 examples/s]45673 examples [00:43, 1051.49 examples/s]45780 examples [00:43, 1054.44 examples/s]45886 examples [00:43, 1043.60 examples/s]45994 examples [00:43, 1053.51 examples/s]46100 examples [00:43, 1049.84 examples/s]46206 examples [00:43, 1027.97 examples/s]46319 examples [00:43, 1055.15 examples/s]46431 examples [00:44, 1072.47 examples/s]46542 examples [00:44, 1083.42 examples/s]46651 examples [00:44, 1080.58 examples/s]46760 examples [00:44, 1058.09 examples/s]46867 examples [00:44, 1049.18 examples/s]46973 examples [00:44, 1031.27 examples/s]47077 examples [00:44, 1005.65 examples/s]47188 examples [00:44, 1032.32 examples/s]47295 examples [00:44, 1041.65 examples/s]47406 examples [00:44, 1059.44 examples/s]47516 examples [00:45, 1069.98 examples/s]47627 examples [00:45, 1081.44 examples/s]47736 examples [00:45, 1080.14 examples/s]47846 examples [00:45, 1083.24 examples/s]47955 examples [00:45, 1062.19 examples/s]48062 examples [00:45, 1057.85 examples/s]48173 examples [00:45, 1071.12 examples/s]48281 examples [00:45, 1070.08 examples/s]48389 examples [00:45, 1041.32 examples/s]48497 examples [00:46, 1051.86 examples/s]48607 examples [00:46, 1065.00 examples/s]48714 examples [00:46, 1005.29 examples/s]48816 examples [00:46, 1007.29 examples/s]48927 examples [00:46, 1033.90 examples/s]49037 examples [00:46, 1052.17 examples/s]49143 examples [00:46, 1029.55 examples/s]49252 examples [00:46, 1045.78 examples/s]49357 examples [00:46, 1040.96 examples/s]49469 examples [00:46, 1062.36 examples/s]49581 examples [00:47, 1077.91 examples/s]49694 examples [00:47, 1091.57 examples/s]49806 examples [00:47, 1098.02 examples/s]49916 examples [00:47, 1095.07 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7262/50000 [00:00<00:00, 72618.68 examples/s] 41%|████▏     | 20686/50000 [00:00<00:00, 84215.94 examples/s] 69%|██████▊   | 34304/50000 [00:00<00:00, 95102.11 examples/s] 96%|█████████▌| 47825/50000 [00:00<00:00, 104390.84 examples/s]                                                                0 examples [00:00, ? examples/s]91 examples [00:00, 904.07 examples/s]204 examples [00:00, 961.20 examples/s]318 examples [00:00, 1007.27 examples/s]430 examples [00:00, 1037.72 examples/s]544 examples [00:00, 1063.92 examples/s]658 examples [00:00, 1083.34 examples/s]772 examples [00:00, 1099.46 examples/s]880 examples [00:00, 1091.52 examples/s]994 examples [00:00, 1103.33 examples/s]1102 examples [00:01, 1091.46 examples/s]1212 examples [00:01, 1091.43 examples/s]1325 examples [00:01, 1102.63 examples/s]1437 examples [00:01, 1106.73 examples/s]1550 examples [00:01, 1112.29 examples/s]1661 examples [00:01, 1101.93 examples/s]1776 examples [00:01, 1114.80 examples/s]1888 examples [00:01, 1108.78 examples/s]1999 examples [00:01, 1052.21 examples/s]2110 examples [00:01, 1067.28 examples/s]2220 examples [00:02, 1075.12 examples/s]2333 examples [00:02, 1089.97 examples/s]2443 examples [00:02, 1067.64 examples/s]2551 examples [00:02, 1056.46 examples/s]2657 examples [00:02, 1015.53 examples/s]2760 examples [00:02, 962.21 examples/s] 2860 examples [00:02, 971.97 examples/s]2971 examples [00:02, 1009.62 examples/s]3073 examples [00:02, 991.31 examples/s] 3180 examples [00:02, 1012.67 examples/s]3292 examples [00:03, 1041.80 examples/s]3403 examples [00:03, 1060.07 examples/s]3510 examples [00:03, 1058.79 examples/s]3617 examples [00:03, 1053.58 examples/s]3723 examples [00:03, 1049.64 examples/s]3829 examples [00:03, 1047.56 examples/s]3934 examples [00:03, 1044.77 examples/s]4048 examples [00:03, 1069.30 examples/s]4159 examples [00:03, 1081.06 examples/s]4269 examples [00:03, 1085.42 examples/s]4382 examples [00:04, 1096.54 examples/s]4496 examples [00:04, 1108.33 examples/s]4610 examples [00:04, 1116.05 examples/s]4725 examples [00:04, 1123.46 examples/s]4838 examples [00:04, 1114.53 examples/s]4950 examples [00:04, 1107.18 examples/s]5061 examples [00:04, 1094.26 examples/s]5171 examples [00:04, 1080.16 examples/s]5284 examples [00:04, 1093.97 examples/s]5395 examples [00:05, 1095.80 examples/s]5505 examples [00:05, 1034.38 examples/s]5613 examples [00:05, 1045.34 examples/s]5727 examples [00:05, 1070.77 examples/s]5841 examples [00:05, 1087.81 examples/s]5951 examples [00:05, 1078.26 examples/s]6060 examples [00:05, 1053.59 examples/s]6170 examples [00:05, 1066.81 examples/s]6283 examples [00:05, 1083.02 examples/s]6396 examples [00:05, 1094.03 examples/s]6506 examples [00:06, 1087.16 examples/s]6619 examples [00:06, 1097.17 examples/s]6733 examples [00:06, 1107.40 examples/s]6846 examples [00:06, 1113.43 examples/s]6960 examples [00:06, 1119.17 examples/s]7072 examples [00:06, 1078.34 examples/s]7181 examples [00:06, 1070.84 examples/s]7289 examples [00:06, 1054.23 examples/s]7397 examples [00:06, 1060.57 examples/s]7507 examples [00:06, 1070.90 examples/s]7617 examples [00:07, 1078.55 examples/s]7730 examples [00:07, 1092.95 examples/s]7840 examples [00:07, 1086.82 examples/s]7950 examples [00:07, 1090.22 examples/s]8060 examples [00:07, 1055.77 examples/s]8166 examples [00:07, 997.80 examples/s] 8267 examples [00:07, 993.30 examples/s]8377 examples [00:07, 1022.82 examples/s]8480 examples [00:07, 994.57 examples/s] 8593 examples [00:08, 1027.00 examples/s]8704 examples [00:08, 1045.54 examples/s]8815 examples [00:08, 1064.00 examples/s]8925 examples [00:08, 1071.96 examples/s]9037 examples [00:08, 1085.63 examples/s]9146 examples [00:08, 1072.58 examples/s]9256 examples [00:08, 1078.16 examples/s]9364 examples [00:08, 1054.69 examples/s]9477 examples [00:08, 1073.75 examples/s]9591 examples [00:08, 1092.44 examples/s]9701 examples [00:09, 1092.61 examples/s]9813 examples [00:09, 1098.28 examples/s]9927 examples [00:09, 1108.62 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteHMUBTO/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteHMUBTO/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1bf2ce72f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1bf2ce72f0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1bf2ce72f0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1c56ce6128>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1b7c174518>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1bf2ce72f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1bf2ce72f0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1bf2ce72f0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1b7c1743c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bf00d90f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f1b7402f268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f1b7402f268> 

  function with postional parmater data_info <function split_train_valid at 0x7f1b7402f268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f1b7402f378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f1b7402f378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f1b7402f378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=30404bc79b9f7b342c8874c51596a220a9b8902deb871ffc6a53fa48803235a6
  Stored in directory: /tmp/pip-ephem-wheel-cache-gt1duvsu/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:08:46, 13.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<12:56:04, 18.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:06:28, 26.3kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:23:04, 37.5kB/s].vector_cache/glove.6B.zip:   0%|          | 3.06M/862M [00:01<4:27:40, 53.5kB/s].vector_cache/glove.6B.zip:   1%|          | 6.46M/862M [00:01<3:06:45, 76.4kB/s].vector_cache/glove.6B.zip:   1%|          | 10.8M/862M [00:01<2:10:10, 109kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 15.2M/862M [00:01<1:30:44, 156kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.9M/862M [00:01<1:03:21, 222kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.5M/862M [00:01<44:12, 316kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.5M/862M [00:01<30:54, 450kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.7M/862M [00:01<21:39, 639kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.2M/862M [00:02<15:16, 903kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.5M/862M [00:02<10:43, 1.28MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.5M/862M [00:02<07:37, 1.79MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.7M/862M [00:02<05:26, 2.50MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.1M/862M [00:02<03:54, 3.46MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<02:50, 4.74MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:03<27:35, 489kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 54.4M/862M [00:03<19:33, 688kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<16:56, 793kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<16:31, 812kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:05<12:44, 1.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.0M/862M [00:05<09:09, 1.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<10:00, 1.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<09:20, 1.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:07<07:06, 1.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<07:14, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<07:19, 1.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:09<05:37, 2.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:09<04:08, 3.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<08:26, 1.56MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:11<08:13, 1.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.2M/862M [00:11<06:19, 2.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:11<04:33, 2.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<30:43, 428kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:13<23:50, 552kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:13<17:14, 762kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<14:15, 917kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:15<12:10, 1.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:15<09:01, 1.45MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:15<06:30, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<10:31, 1.24MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<09:37, 1.35MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:17<07:18, 1.78MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<07:19, 1.77MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<07:21, 1.76MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:19<05:42, 2.26MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:19<04:07, 3.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<27:57, 460kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<21:49, 590kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:21<15:48, 813kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<13:13, 968kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<11:30, 1.11MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:23<08:36, 1.48MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:23<06:10, 2.07MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<15:45, 809kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<13:10, 966kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:24<09:41, 1.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<07:05, 1.79MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:29, 1.49MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:08, 1.56MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:10, 2.05MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<04:30, 2.80MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:53, 1.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:22, 1.50MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:24, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:37, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:49, 1.83MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:14, 2.39MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<03:54, 3.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:57, 1.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:03, 1.77MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<05:28, 2.28MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<05:56, 2.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:15, 1.98MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:52, 2.54MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<03:36, 3.43MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:36, 1.62MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:28, 1.65MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:46, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<04:10, 2.94MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<15:30, 791kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<12:56, 947kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<09:28, 1.29MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<06:55, 1.77MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:12, 1.48MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:40, 1.59MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:49, 2.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<04:11, 2.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<40:28, 299kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<30:25, 398kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<21:43, 557kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<15:21, 786kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<16:27, 733kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<13:33, 889kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<10:00, 1.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<09:02, 1.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<08:25, 1.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:20, 1.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<04:35, 2.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<09:11, 1.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:32, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<06:29, 1.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<04:38, 2.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<3:53:58, 50.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<2:45:49, 71.4kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<1:56:26, 102kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<1:21:18, 145kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<1:33:07, 126kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<1:07:15, 175kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<47:29, 248kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<33:16, 352kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<32:15, 363kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<24:32, 477kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<17:40, 661kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<14:18, 813kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<12:03, 965kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<08:55, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<06:23, 1.81MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<12:10, 951kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<10:32, 1.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<07:51, 1.47MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<05:36, 2.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<23:43, 485kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<18:36, 618kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<13:28, 852kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<09:36, 1.19MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<11:05, 1.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<09:46, 1.17MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:19, 1.56MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<07:03, 1.61MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:53, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:18, 2.14MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<03:50, 2.94MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<12:41, 889kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<10:53, 1.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<08:02, 1.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<05:46, 1.95MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<09:54, 1.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<08:54, 1.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:43, 1.67MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:35, 1.69MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:47, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<05:16, 2.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<03:47, 2.92MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<30:23, 365kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<23:28, 472kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<17:00, 651kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<12:02, 916kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<13:02, 845kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<10:57, 1.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:42, 1.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:48, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:08, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:13, 1.76MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<07:23, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:00, 1.82MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:33, 2.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<03:22, 3.23MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:31, 1.44MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<08:12, 1.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:22, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<04:42, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:46, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:41, 1.62MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:22, 2.01MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<03:55, 2.74MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:06, 1.76MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<07:05, 1.51MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:38, 1.90MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:06, 2.60MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:29, 1.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<09:42, 1.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<07:45, 1.37MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:39, 1.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:42, 1.58MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<08:31, 1.24MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:47, 1.56MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:58, 2.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<03:39, 2.88MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<11:35, 908kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<11:33, 911kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<08:58, 1.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<06:31, 1.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:54, 2.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<07:32, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<10:14, 1.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<08:16, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<06:07, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<04:30, 2.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<03:22, 3.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<2:16:39, 76.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<1:40:18, 104kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<1:11:16, 146kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<50:26, 206kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<35:24, 292kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<24:59, 414kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<30:05, 343kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<25:39, 402kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<18:55, 545kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<13:29, 763kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<09:40, 1.06MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<10:21, 990kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<11:25, 897kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<09:04, 1.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<06:37, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<04:49, 2.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<10:37, 960kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<11:36, 877kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<09:09, 1.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<06:36, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:55, 2.06MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<07:18, 1.38MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<09:15, 1.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<07:30, 1.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<05:31, 1.83MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<04:00, 2.50MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<15:24, 652kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<14:56, 673kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<11:23, 882kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<08:15, 1.21MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:58, 1.67MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<07:58, 1.25MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<09:23, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:33, 1.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:32, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<04:04, 2.44MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<09:58, 993kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<11:02, 897kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<08:38, 1.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<06:24, 1.54MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:40, 2.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:40, 1.48MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<08:45, 1.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<07:02, 1.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<05:20, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<03:53, 2.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<10:01, 975kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<12:58, 753kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<10:37, 919kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<07:46, 1.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:40, 1.72MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:31, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<03:30, 2.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<25:12, 385kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<21:46, 446kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<16:10, 600kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<11:47, 821kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<08:33, 1.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<06:24, 1.51MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:47, 2.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<10:26, 923kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<11:14, 857kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<08:48, 1.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<06:31, 1.47MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:53, 1.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:42, 2.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<12:58, 737kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<12:43, 752kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<09:48, 975kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<07:09, 1.33MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<05:17, 1.80MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:08, 2.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<03:31, 2.69MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<29:16, 324kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<25:40, 370kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<19:12, 494kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<13:52, 683kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<10:06, 935kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<07:25, 1.27MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<05:43, 1.65MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<09:08, 1.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<11:02, 854kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<08:49, 1.07MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<06:35, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<05:06, 1.84MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:07, 2.27MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:27, 2.71MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<02:58, 3.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<1:59:40, 78.2kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<1:26:35, 108kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<1:01:18, 153kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<43:23, 215kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<30:49, 303kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<22:01, 423kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<16:02, 580kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<11:40, 796kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<19:38, 473kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<19:27, 478kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<14:56, 621kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<10:58, 845kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<08:09, 1.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<06:11, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:48, 1.92MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:50, 2.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<16:36, 556kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<14:08, 652kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<10:34, 871kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<07:53, 1.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<05:59, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<04:39, 1.97MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:40, 2.49MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<09:34, 956kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<11:39, 785kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<09:23, 974kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<07:06, 1.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<05:26, 1.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<04:15, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:22, 2.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<02:57, 3.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<13:20, 681kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<11:47, 770kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<08:55, 1.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<06:42, 1.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:10, 1.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<04:17, 2.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:32, 2.55MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<02:56, 3.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<07:45, 1.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<07:52, 1.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:11, 1.45MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:47, 1.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<03:46, 2.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<02:58, 3.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<02:49, 3.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<06:40, 1.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<10:09, 881kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<08:20, 1.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<06:19, 1.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:54, 1.82MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:50, 2.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:11, 2.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<02:38, 3.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<02:20, 3.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<12:43, 698kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<11:19, 784kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<08:29, 1.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:26, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:56, 1.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<03:54, 2.26MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<03:09, 2.79MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<03:13, 2.74MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<02:35, 3.39MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<5:55:20, 24.8kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<4:15:56, 34.4kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<3:00:50, 48.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<2:07:10, 69.2kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<1:29:32, 98.2kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<1:03:13, 139kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<44:44, 196kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<31:52, 275kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<22:48, 384kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<16:30, 530kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<12:02, 726kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<34:14, 255kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<26:38, 328kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<19:18, 452kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<14:01, 621kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<10:17, 847kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<07:48, 1.11MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<05:51, 1.49MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<05:21, 1.62MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<04:06, 2.11MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<18:39, 465kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<16:35, 522kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<12:31, 692kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<09:20, 926kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<07:16, 1.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:37, 1.54MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:40, 1.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:48, 2.26MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<03:23, 2.54MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<02:51, 3.01MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:46, 3.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<10:08, 849kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<08:39, 994kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:41, 1.28MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:09, 1.66MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:22, 1.96MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:39, 2.34MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:06, 2.75MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<02:46, 3.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<02:29, 3.43MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<02:20, 3.66MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<13:29, 633kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<12:41, 672kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<09:42, 878kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<07:20, 1.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:46, 1.47MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:45, 1.79MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:12, 2.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:36, 2.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:23, 2.50MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:02, 2.79MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:00, 2.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<02:45, 3.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:32, 2.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<1:37:27, 86.9kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<1:10:57, 119kB/s] .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<50:39, 167kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<36:32, 231kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<26:36, 318kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<19:36, 431kB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<14:38, 577kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<11:13, 752kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<08:46, 960kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<07:04, 1.19MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:50, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:01, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:22, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<04:02, 2.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:38, 2.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:30, 2.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<08:03, 1.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<06:54, 1.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:43, 1.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:10, 1.62MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:49, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:17, 1.96MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:49, 2.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:35, 2.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:19, 2.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:17, 2.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:02, 2.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:03, 2.73MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:54, 2.87MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:57, 2.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:52, 2.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<09:51, 845kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<08:08, 1.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<06:34, 1.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:27, 1.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:51, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:19, 1.92MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:53, 2.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:33, 2.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:24, 2.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:11, 2.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:04, 2.69MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<02:57, 2.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<02:55, 2.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<02:49, 2.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<11:33, 714kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<09:16, 890kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<07:19, 1.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:55, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:55, 1.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:13, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:43, 2.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:13, 2.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:14, 2.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:02, 2.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<02:43, 3.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<02:57, 2.77MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<48:57, 167kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<36:22, 225kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<26:17, 311kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<19:08, 427kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<14:00, 583kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<10:35, 771kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<08:14, 990kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<06:30, 1.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:16, 1.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<04:25, 1.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:47, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:21, 2.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<03:02, 2.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<50:12, 162kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<37:01, 219kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<26:40, 304kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<19:20, 419kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<14:13, 569kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<10:35, 764kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<08:05, 999kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<06:19, 1.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<05:05, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<04:12, 1.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<13:16, 607kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<11:02, 729kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<08:28, 949kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<06:24, 1.25MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<05:22, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:21, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:35, 2.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:01, 2.64MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<02:45, 2.90MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:23, 3.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<11:20, 704kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<10:44, 744kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<08:17, 963kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<06:15, 1.27MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<05:12, 1.53MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:04, 1.95MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:31, 2.26MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:02, 2.61MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:45, 2.88MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<07:40, 1.03MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<08:51, 893kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<07:05, 1.12MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<05:33, 1.42MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:24, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:44, 2.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:08, 2.51MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<02:50, 2.78MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<02:34, 3.05MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:20, 3.36MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:12, 3.56MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<10:40, 735kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<08:54, 881kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<06:48, 1.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:13, 1.50MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:10, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:28, 2.25MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<02:55, 2.67MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:33, 3.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:15, 3.44MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<07:53, 986kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<08:04, 963kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<06:21, 1.22MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<04:51, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:01, 1.93MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:15, 2.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:47, 2.77MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:22, 3.25MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:11, 3.51MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:01, 3.82MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<14:10, 544kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<12:36, 611kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<09:22, 821kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<07:16, 1.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<05:28, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:45, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:33, 1.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:44, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:32, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:20, 2.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:12, 2.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:06, 2.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<09:08, 836kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<08:35, 890kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<06:52, 1.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<05:38, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:44, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:07, 1.85MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:40, 2.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:20, 2.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:07, 2.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:57, 2.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:49, 2.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<02:44, 2.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<09:36, 788kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<07:51, 964kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<06:17, 1.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:09, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:15, 1.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:56, 1.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:30, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:16, 2.30MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:02, 2.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<02:55, 2.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<02:51, 2.64MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<02:45, 2.72MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:41, 2.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:38, 2.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<11:18, 664kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<09:01, 831kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<07:03, 1.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<05:39, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<04:40, 1.60MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:59, 1.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:23, 2.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:30, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:04, 2.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:57, 2.52MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:40, 2.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:41, 2.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:35, 2.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<2:41:51, 45.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<1:55:10, 64.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<1:21:18, 91.3kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<57:33, 129kB/s]   .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<40:59, 181kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<29:30, 251kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<21:37, 342kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<16:11, 457kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<12:02, 614kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<09:05, 812kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<07:01, 1.05MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:33, 1.33MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<04:32, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:50, 1.92MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<14:58, 492kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<11:20, 649kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<08:46, 839kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<06:41, 1.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:29, 1.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<04:24, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:46, 1.94MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:13, 2.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<02:56, 2.49MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<02:39, 2.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:30, 2.92MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:23, 3.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<06:51, 1.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<06:26, 1.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:14, 1.39MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:17, 1.69MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:33, 2.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:05, 2.35MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<02:44, 2.65MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<02:29, 2.91MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:17, 3.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:09, 3.35MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<11:10, 647kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<09:16, 778kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<07:05, 1.02MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<05:41, 1.27MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:47, 1.50MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:03, 1.77MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:31, 2.04MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:08, 2.29MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:49, 2.54MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:29, 2.88MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:36, 2.75MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<06:43, 1.06MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<06:17, 1.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:03, 1.41MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:08, 1.72MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:31, 2.03MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:13, 2.22MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<02:50, 2.51MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<02:34, 2.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<02:22, 3.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:13, 3.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:09, 3.29MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:02, 3.48MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<14:24, 492kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<11:31, 615kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<08:40, 815kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<06:36, 1.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:08, 1.37MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<04:04, 1.73MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:26, 2.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:55, 2.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:54, 2.42MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:39, 2.64MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:40, 2.63MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<07:41, 913kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<07:17, 964kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:42, 1.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<04:40, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:53, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:33, 1.97MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:03, 2.29MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<02:45, 2.54MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<02:41, 2.58MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<02:27, 2.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<02:31, 2.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:20, 2.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:17, 3.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<32:17, 215kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<24:13, 287kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<17:37, 394kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<12:56, 536kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<09:39, 718kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<07:21, 940kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<05:37, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:43, 1.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:47, 1.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:28, 1.99MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:56, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:45, 2.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<19:19, 356kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<15:07, 455kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<11:13, 613kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<08:26, 814kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<06:29, 1.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<05:07, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:11, 1.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:31, 1.94MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:54, 2.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:53, 2.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:29, 2.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<24:29, 278kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<18:27, 369kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<13:39, 498kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<10:02, 678kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<07:44, 878kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:59, 1.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:39, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<03:56, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:20, 2.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:47, 2.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:38, 2.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<06:32, 1.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<06:00, 1.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:48, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:54, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:17, 2.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:49, 2.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:30, 2.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:14, 2.99MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:09, 3.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:02, 3.28MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<06:38, 1.01MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<06:01, 1.11MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:47, 1.39MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:50, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:14, 2.05MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:44, 2.42MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:27, 2.70MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:13, 2.98MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:02, 3.25MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<01:57, 3.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<01:50, 3.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<07:36, 868kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<08:00, 826kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<06:16, 1.05MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:49, 1.37MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:03, 1.62MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:27, 1.90MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:53, 2.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:29, 2.63MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:12, 2.97MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:00, 3.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<01:50, 3.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<01:44, 3.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<1:38:18, 66.6kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<1:11:03, 92.0kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<50:20, 130kB/s]   .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<35:35, 183kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<25:21, 257kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<18:06, 360kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<13:06, 496kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<09:31, 681kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<07:05, 914kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<05:21, 1.21MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<11:49, 547kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<10:16, 629kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<07:44, 834kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<05:47, 1.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:24, 1.46MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:25, 1.88MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:42, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:13, 2.88MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<06:07, 1.04MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<07:47, 821kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<06:16, 1.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:57, 1.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:42, 1.72MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:24, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:42, 2.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:27, 2.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:20, 2.71MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:13, 2.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:07, 2.97MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<08:11, 773kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<07:15, 872kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:39, 1.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:30, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:40, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:06, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<02:35, 2.44MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<02:29, 2.52MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<02:09, 2.92MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<02:15, 2.79MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:00, 3.12MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<09:12, 680kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<07:47, 804kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<05:59, 1.04MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<04:41, 1.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<03:48, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:09, 1.98MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:42, 2.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:22, 2.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:11, 2.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:00, 3.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<01:54, 3.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<01:48, 3.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<10:15, 604kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<08:29, 729kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<06:28, 955kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:58, 1.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:56, 1.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:20, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:42, 2.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:34, 2.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:10, 2.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:11, 2.81MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<01:53, 3.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<11:57, 512kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<09:37, 637kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<07:15, 843kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<05:33, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:21, 1.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:31, 1.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:55, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:31, 2.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:13, 2.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<06:09, 983kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<05:33, 1.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<04:23, 1.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:27, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:52, 2.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:33, 2.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:14, 2.68MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<01:55, 3.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<01:56, 3.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<01:47, 3.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<06:17, 952kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<06:52, 872kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<05:28, 1.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:17, 1.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:25, 1.74MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:49, 2.11MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:23, 2.48MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:06, 2.82MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<01:51, 3.20MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<01:46, 3.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<36:44, 161kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<26:49, 221kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<19:12, 308kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<13:50, 426kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<10:02, 587kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<07:25, 793kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<05:29, 1.07MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<04:24, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<03:25, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<08:57, 654kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<08:29, 689kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<06:33, 893kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:58, 1.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:51, 1.51MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:01, 1.92MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:25, 2.40MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:08, 2.71MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<01:49, 3.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<17:02, 340kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<13:42, 422kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<10:05, 573kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<07:20, 785kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:30, 1.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<04:10, 1.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:13, 1.78MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:32, 2.25MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<11:16, 507kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<09:26, 605kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<07:04, 807kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<05:13, 1.09MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:55, 1.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:58, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:19, 2.42MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<21:52, 258kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<17:51, 316kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<13:14, 426kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<09:30, 591kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<06:51, 818kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<05:00, 1.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<05:50, 954kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<06:21, 877kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<05:01, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<03:44, 1.48MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:48, 1.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:06, 2.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<06:21, 867kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<06:32, 843kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<05:03, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<03:42, 1.48MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:44, 1.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:05, 2.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<05:51, 929kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<07:27, 730kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<06:04, 895kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<04:26, 1.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<03:19, 1.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:25, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:25, 1.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:39, 1.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:37, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<02:39, 2.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<01:59, 2.67MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<07:02, 754kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<07:30, 705kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<05:49, 910kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:20, 1.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<03:08, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:19, 2.25MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<05:27, 958kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<06:09, 851kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:52, 1.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<03:34, 1.46MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<02:35, 1.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<05:33, 931kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<06:42, 770kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<05:27, 946kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<03:59, 1.29MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<02:54, 1.76MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<03:57, 1.29MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:49, 1.33MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:28, 1.47MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:33, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:57, 2.58MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<01:28, 3.43MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<07:28, 673kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<07:58, 630kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<06:13, 806kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:29, 1.11MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<03:17, 1.52MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<02:24, 2.06MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<06:56, 715kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<07:17, 680kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<05:43, 865kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<04:09, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<02:59, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<04:26, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<05:31, 884kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<04:25, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:15, 1.50MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<02:22, 2.04MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<04:38, 1.04MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<05:25, 889kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<04:19, 1.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<03:09, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<02:17, 2.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<05:09, 922kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<05:44, 827kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:33, 1.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:20, 1.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<02:26, 1.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:02, 1.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<04:02, 1.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:20, 1.40MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:29, 1.87MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:50, 2.52MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:22, 3.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<48:20, 95.4kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<35:42, 129kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<25:27, 181kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<17:51, 257kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<12:31, 364kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<10:59, 413kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<09:35, 474kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<07:11, 631kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<05:08, 878kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<03:39, 1.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<05:29, 815kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<05:40, 788kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<04:26, 1.01MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<03:13, 1.38MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<02:19, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<08:04, 546kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<07:37, 578kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<05:48, 758kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<04:10, 1.05MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<03:02, 1.44MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<02:12, 1.96MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<07:55, 547kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<07:29, 579kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<05:41, 760kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<04:04, 1.05MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<02:55, 1.46MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<04:35, 928kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<03:18, 1.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<02:23, 1.77MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:49, 2.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<04:55, 852kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<04:35, 912kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:27, 1.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:30, 1.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:52, 2.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:25, 2.90MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<15:23, 268kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<13:02, 317kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<09:40, 426kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<06:52, 597kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<05:01, 816kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<03:34, 1.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<05:34, 729kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<05:25, 747kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:10, 972kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:02, 1.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:15, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<01:40, 2.38MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<13:22, 298kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<10:52, 367kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<07:57, 500kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<05:40, 697kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<04:05, 964kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<02:57, 1.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<12:08, 323kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<09:56, 394kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<07:18, 536kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<05:18, 736kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<03:50, 1.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<02:47, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<02:04, 1.86MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<06:14, 617kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<05:37, 684kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:12, 913kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:02, 1.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:16, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:43, 2.21MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:00, 1.26MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<03:20, 1.13MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:38, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:58, 1.91MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:27, 2.56MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:10, 3.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<05:03, 734kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<04:47, 775kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:40, 1.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:43, 1.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:00, 1.83MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:43<01:31, 2.41MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:39, 1.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:03, 1.19MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:25, 1.50MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:48, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:21, 2.64MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:39, 1.35MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<04:03, 879kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:24, 1.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:30, 1.42MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:49, 1.93MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:24, 2.49MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:06, 3.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<17:25, 201kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<13:21, 262kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<09:36, 364kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<06:50, 509kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<04:51, 713kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<03:30, 984kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<04:03, 846kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<04:57, 693kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<03:56, 872kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:53, 1.18MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:08, 1.59MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:34, 2.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:13, 2.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<06:16, 537kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<05:25, 621kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<04:02, 830kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:55, 1.14MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<02:08, 1.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:01, 1.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:09, 1.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:27, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:48, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:21, 2.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<02:30, 1.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<02:45, 1.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:10, 1.48MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:37, 1.98MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:12, 2.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<00:57, 3.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<06:34, 482kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<06:32, 484kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<05:04, 622kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<03:38, 862kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:40, 1.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:55, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:28, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<10:10, 304kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<08:04, 383kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<05:52, 525kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<04:10, 733kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<02:58, 1.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<02:11, 1.38MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<08:51, 341kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<08:03, 375kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<06:06, 494kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<04:23, 685kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<03:07, 952kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<02:14, 1.32MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<21:36, 137kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<18:00, 164kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<13:16, 222kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<09:24, 313kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<06:38, 440kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<04:42, 617kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<04:19, 667kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<03:50, 750kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<02:53, 996kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<02:05, 1.36MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:31, 1.86MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<02:27, 1.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<03:16, 860kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<02:41, 1.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:10<01:57, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:26, 1.92MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:04, 2.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<03:26, 800kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<03:07, 877kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:22, 1.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<01:43, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:15, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<04:09, 645kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<04:23, 610kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<03:25, 782kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:27, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:46, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:17, 2.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<09:09, 285kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<07:02, 370kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<05:03, 513kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<03:34, 720kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:33, 998kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<02:55, 869kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<03:11, 796kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<02:31, 1.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:50, 1.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:20, 1.86MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<00:58, 2.52MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<13:21, 185kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<10:28, 236kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<07:34, 325kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<05:19, 459kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<03:43, 648kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<05:02, 477kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<04:30, 532kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<03:21, 712kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<02:24, 985kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:43, 1.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<01:58, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<02:17, 1.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:48, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:18, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:57, 2.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:42, 1.33MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:59, 1.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:36, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:09, 1.93MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:50, 2.61MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<02:21, 931kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<02:26, 901kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:51, 1.18MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:23, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:00, 2.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:02, 2.06MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:33, 1.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<02:50, 746kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<02:24, 880kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:47, 1.18MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<01:36, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:30<01:08, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:53, 2.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:57, 1.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<02:09, 956kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:41, 1.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:14, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:55, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:42, 2.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<19:37, 101kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<14:25, 138kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<10:14, 193kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<07:09, 274kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<05:01, 386kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<04:11, 459kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<04:29, 428kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<03:28, 551kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<02:29, 762kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:46, 1.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:19, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<01:36, 1.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<01:43, 1.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:21, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:59, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:45, 2.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:34, 3.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<01:51, 960kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<02:33, 695kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<02:05, 848kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<01:31, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:08, 1.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:49, 2.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:38, 2.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:41<02:06, 815kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<02:02, 836kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<01:34, 1.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:08, 1.47MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:50, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<01:15, 1.31MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<01:53, 868kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<01:33, 1.05MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<01:08, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<00:52, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:39, 2.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:30, 3.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<01:52, 843kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<01:49, 863kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<01:23, 1.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<01:02, 1.50MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:45, 2.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:34, 2.61MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<01:02, 1.45MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<01:11, 1.27MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:57, 1.57MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:42, 2.09MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:31, 2.75MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:49<00:57, 1.50MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<01:08, 1.25MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:53, 1.59MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:41, 2.06MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:31, 2.69MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:23, 3.49MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<01:21, 1.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<01:23, 981kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<01:04, 1.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:47, 1.68MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:36, 2.20MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:28, 2.79MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:22, 3.48MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<01:42, 763kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<01:38, 789kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<01:15, 1.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:54, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<00:40, 1.87MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:29, 2.49MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<01:13, 1.01MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<01:15, 978kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<00:58, 1.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:42, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:32, 2.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:23, 2.93MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<01:21, 852kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<01:18, 884kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<01:00, 1.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:44, 1.55MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:33, 2.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:25, 2.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:20, 3.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<01:51, 587kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<01:40, 653kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<01:15, 865kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:53, 1.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:38, 1.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:28, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<02:14, 455kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<01:52, 542kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<01:22, 732kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:58, 1.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:42, 1.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:31, 1.83MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<01:00, 945kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<00:58, 971kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:03<00:44, 1.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:32, 1.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:23, 2.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:49, 1.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<01:03, 830kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<00:52, 1.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:37, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:27, 1.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:35, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:07<00:38, 1.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:29, 1.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:21, 2.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:15, 2.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:54, 825kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<01:02, 715kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<00:49, 901kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:09<00:35, 1.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:24, 1.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:33, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:11<00:43, 923kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:11<00:35, 1.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:25, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:17, 2.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:29, 1.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:37, 982kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:13<00:29, 1.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:21, 1.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:14, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:38, 840kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:41, 778kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:15<00:32, 993kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:22, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:15, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:42, 669kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:41, 685kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:17<00:31, 890kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:21, 1.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:14, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:43, 555kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:39, 610kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:29, 804kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:19, 1.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:17, 1.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:21, 916kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:21<00:17, 1.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:12, 1.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 2.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:13, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:14, 1.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:11, 1.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:07, 1.83MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:05, 2.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:03, 3.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:09, 1.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:10, 1.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:08, 1.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:25<00:07, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:25<00:07, 1.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:06, 1.60MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:06, 1.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:05, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:05, 1.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:05, 1.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:04, 1.94MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:04, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:04, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:04, 2.03MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:04, 1.98MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:04, 2.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:03, 2.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:03, 2.03MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:27<00:03, 2.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:27<00:03, 2.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:03, 2.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:03, 2.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:02, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:02, 2.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:02, 2.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:02, 2.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:28<00:02, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:28<00:01, 2.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:01, 2.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:01, 2.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:01, 2.30MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.28MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.32MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.38MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:29<00:01, 2.35MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:01, 2.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:00, 2.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:00, 2.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:00, 2.43MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 2.41MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 2.45MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 2.47MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 2.49MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 2.47MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:30<00:00, 1.58MB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<37:12:18,  2.99it/s]  0%|          | 831/400000 [00:00<25:59:36,  4.27it/s]  0%|          | 1717/400000 [00:00<18:09:31,  6.09it/s]  1%|          | 2591/400000 [00:00<12:41:13,  8.70it/s]  1%|          | 3466/400000 [00:00<8:51:54, 12.42it/s]   1%|          | 4334/400000 [00:00<6:11:45, 17.74it/s]  1%|▏         | 5179/400000 [00:00<4:19:54, 25.32it/s]  2%|▏         | 6051/400000 [00:01<3:01:45, 36.12it/s]  2%|▏         | 6919/400000 [00:01<2:07:10, 51.51it/s]  2%|▏         | 7772/400000 [00:01<1:29:03, 73.40it/s]  2%|▏         | 8642/400000 [00:01<1:02:25, 104.48it/s]  2%|▏         | 9482/400000 [00:01<43:50, 148.46it/s]    3%|▎         | 10349/400000 [00:01<30:50, 210.54it/s]  3%|▎         | 11210/400000 [00:01<21:46, 297.65it/s]  3%|▎         | 12083/400000 [00:01<15:25, 419.09it/s]  3%|▎         | 12951/400000 [00:01<10:59, 586.56it/s]  3%|▎         | 13812/400000 [00:01<07:54, 813.63it/s]  4%|▎         | 14695/400000 [00:02<05:44, 1118.12it/s]  4%|▍         | 15570/400000 [00:02<04:13, 1514.33it/s]  4%|▍         | 16447/400000 [00:02<03:10, 2014.24it/s]  4%|▍         | 17317/400000 [00:02<02:26, 2617.48it/s]  5%|▍         | 18187/400000 [00:02<01:55, 3305.11it/s]  5%|▍         | 19063/400000 [00:02<01:33, 4064.06it/s]  5%|▍         | 19941/400000 [00:02<01:18, 4844.47it/s]  5%|▌         | 20819/400000 [00:02<01:07, 5596.88it/s]  5%|▌         | 21692/400000 [00:02<01:00, 6265.10it/s]  6%|▌         | 22564/400000 [00:02<00:55, 6748.25it/s]  6%|▌         | 23444/400000 [00:03<00:51, 7253.94it/s]  6%|▌         | 24318/400000 [00:03<00:49, 7643.74it/s]  6%|▋         | 25196/400000 [00:03<00:47, 7950.79it/s]  7%|▋         | 26072/400000 [00:03<00:45, 8176.45it/s]  7%|▋         | 26944/400000 [00:03<00:44, 8295.91it/s]  7%|▋         | 27834/400000 [00:03<00:43, 8466.36it/s]  7%|▋         | 28710/400000 [00:03<00:43, 8551.99it/s]  7%|▋         | 29585/400000 [00:03<00:43, 8607.59it/s]  8%|▊         | 30462/400000 [00:03<00:42, 8654.12it/s]  8%|▊         | 31338/400000 [00:03<00:42, 8636.26it/s]  8%|▊         | 32222/400000 [00:04<00:42, 8695.14it/s]  8%|▊         | 33097/400000 [00:04<00:42, 8711.25it/s]  8%|▊         | 33972/400000 [00:04<00:42, 8710.72it/s]  9%|▊         | 34846/400000 [00:04<00:41, 8696.24it/s]  9%|▉         | 35718/400000 [00:04<00:42, 8658.88it/s]  9%|▉         | 36590/400000 [00:04<00:41, 8676.05it/s]  9%|▉         | 37467/400000 [00:04<00:41, 8703.17it/s] 10%|▉         | 38349/400000 [00:04<00:41, 8734.69it/s] 10%|▉         | 39238/400000 [00:04<00:41, 8777.81it/s] 10%|█         | 40117/400000 [00:04<00:41, 8690.57it/s] 10%|█         | 41008/400000 [00:05<00:41, 8755.19it/s] 10%|█         | 41895/400000 [00:05<00:40, 8788.52it/s] 11%|█         | 42775/400000 [00:05<00:40, 8772.28it/s] 11%|█         | 43653/400000 [00:05<00:40, 8760.05it/s] 11%|█         | 44530/400000 [00:05<00:40, 8732.39it/s] 11%|█▏        | 45408/400000 [00:05<00:40, 8744.87it/s] 12%|█▏        | 46284/400000 [00:05<00:40, 8747.33it/s] 12%|█▏        | 47176/400000 [00:05<00:40, 8798.36it/s] 12%|█▏        | 48074/400000 [00:05<00:39, 8850.03it/s] 12%|█▏        | 48960/400000 [00:05<00:39, 8779.83it/s] 12%|█▏        | 49858/400000 [00:06<00:39, 8837.12it/s] 13%|█▎        | 50742/400000 [00:06<00:40, 8673.28it/s] 13%|█▎        | 51611/400000 [00:06<00:41, 8452.26it/s] 13%|█▎        | 52479/400000 [00:06<00:40, 8518.65it/s] 13%|█▎        | 53353/400000 [00:06<00:40, 8581.54it/s] 14%|█▎        | 54232/400000 [00:06<00:40, 8642.74it/s] 14%|█▍        | 55111/400000 [00:06<00:39, 8685.00it/s] 14%|█▍        | 55994/400000 [00:06<00:39, 8726.45it/s] 14%|█▍        | 56868/400000 [00:06<00:39, 8692.19it/s] 14%|█▍        | 57742/400000 [00:06<00:39, 8705.61it/s] 15%|█▍        | 58639/400000 [00:07<00:38, 8780.90it/s] 15%|█▍        | 59518/400000 [00:07<00:38, 8758.44it/s] 15%|█▌        | 60395/400000 [00:07<00:38, 8746.50it/s] 15%|█▌        | 61271/400000 [00:07<00:38, 8749.02it/s] 16%|█▌        | 62147/400000 [00:07<00:38, 8718.75it/s] 16%|█▌        | 63022/400000 [00:07<00:38, 8727.31it/s] 16%|█▌        | 63899/400000 [00:07<00:38, 8739.26it/s] 16%|█▌        | 64788/400000 [00:07<00:38, 8782.20it/s] 16%|█▋        | 65667/400000 [00:07<00:38, 8740.79it/s] 17%|█▋        | 66542/400000 [00:07<00:38, 8712.90it/s] 17%|█▋        | 67414/400000 [00:08<00:38, 8709.44it/s] 17%|█▋        | 68286/400000 [00:08<00:38, 8707.58it/s] 17%|█▋        | 69166/400000 [00:08<00:37, 8734.55it/s] 18%|█▊        | 70041/400000 [00:08<00:37, 8736.81it/s] 18%|█▊        | 70915/400000 [00:08<00:37, 8728.83it/s] 18%|█▊        | 71790/400000 [00:08<00:37, 8732.83it/s] 18%|█▊        | 72667/400000 [00:08<00:37, 8743.52it/s] 18%|█▊        | 73548/400000 [00:08<00:37, 8760.84it/s] 19%|█▊        | 74426/400000 [00:08<00:37, 8763.76it/s] 19%|█▉        | 75303/400000 [00:08<00:37, 8722.78it/s] 19%|█▉        | 76187/400000 [00:09<00:36, 8756.88it/s] 19%|█▉        | 77076/400000 [00:09<00:36, 8794.69it/s] 19%|█▉        | 77960/400000 [00:09<00:36, 8806.03it/s] 20%|█▉        | 78841/400000 [00:09<00:36, 8771.52it/s] 20%|█▉        | 79719/400000 [00:09<00:36, 8719.49it/s] 20%|██        | 80592/400000 [00:09<00:37, 8604.97it/s] 20%|██        | 81453/400000 [00:09<00:37, 8489.48it/s] 21%|██        | 82325/400000 [00:09<00:37, 8556.48it/s] 21%|██        | 83204/400000 [00:09<00:36, 8624.56it/s] 21%|██        | 84069/400000 [00:10<00:36, 8631.56it/s] 21%|██        | 84955/400000 [00:10<00:36, 8698.45it/s] 21%|██▏       | 85838/400000 [00:10<00:35, 8736.93it/s] 22%|██▏       | 86713/400000 [00:10<00:35, 8735.73it/s] 22%|██▏       | 87596/400000 [00:10<00:35, 8762.01it/s] 22%|██▏       | 88477/400000 [00:10<00:35, 8776.29it/s] 22%|██▏       | 89356/400000 [00:10<00:35, 8778.85it/s] 23%|██▎       | 90246/400000 [00:10<00:35, 8813.58it/s] 23%|██▎       | 91132/400000 [00:10<00:34, 8826.80it/s] 23%|██▎       | 92024/400000 [00:10<00:34, 8851.78it/s] 23%|██▎       | 92910/400000 [00:11<00:34, 8783.82it/s] 23%|██▎       | 93797/400000 [00:11<00:34, 8807.60it/s] 24%|██▎       | 94680/400000 [00:11<00:34, 8811.60it/s] 24%|██▍       | 95562/400000 [00:11<00:34, 8797.22it/s] 24%|██▍       | 96443/400000 [00:11<00:34, 8800.08it/s] 24%|██▍       | 97324/400000 [00:11<00:34, 8751.55it/s] 25%|██▍       | 98206/400000 [00:11<00:34, 8771.15it/s] 25%|██▍       | 99089/400000 [00:11<00:34, 8785.68it/s] 25%|██▍       | 99968/400000 [00:11<00:34, 8781.93it/s] 25%|██▌       | 100855/400000 [00:11<00:33, 8806.15it/s] 25%|██▌       | 101736/400000 [00:12<00:34, 8764.11it/s] 26%|██▌       | 102623/400000 [00:12<00:33, 8793.32it/s] 26%|██▌       | 103503/400000 [00:12<00:33, 8783.72it/s] 26%|██▌       | 104382/400000 [00:12<00:33, 8778.38it/s] 26%|██▋       | 105267/400000 [00:12<00:33, 8798.67it/s] 27%|██▋       | 106147/400000 [00:12<00:33, 8774.65it/s] 27%|██▋       | 107028/400000 [00:12<00:33, 8782.83it/s] 27%|██▋       | 107919/400000 [00:12<00:33, 8818.76it/s] 27%|██▋       | 108801/400000 [00:12<00:33, 8804.65it/s] 27%|██▋       | 109682/400000 [00:12<00:33, 8756.04it/s] 28%|██▊       | 110558/400000 [00:13<00:33, 8711.90it/s] 28%|██▊       | 111441/400000 [00:13<00:32, 8746.56it/s] 28%|██▊       | 112316/400000 [00:13<00:32, 8737.93it/s] 28%|██▊       | 113190/400000 [00:13<00:32, 8712.54it/s] 29%|██▊       | 114076/400000 [00:13<00:32, 8755.26it/s] 29%|██▊       | 114957/400000 [00:13<00:32, 8770.55it/s] 29%|██▉       | 115842/400000 [00:13<00:32, 8793.85it/s] 29%|██▉       | 116725/400000 [00:13<00:32, 8802.50it/s] 29%|██▉       | 117606/400000 [00:13<00:32, 8796.47it/s] 30%|██▉       | 118497/400000 [00:13<00:31, 8828.29it/s] 30%|██▉       | 119380/400000 [00:14<00:31, 8793.88it/s] 30%|███       | 120262/400000 [00:14<00:31, 8799.68it/s] 30%|███       | 121158/400000 [00:14<00:31, 8845.18it/s] 31%|███       | 122043/400000 [00:14<00:31, 8828.94it/s] 31%|███       | 122926/400000 [00:14<00:31, 8808.85it/s] 31%|███       | 123807/400000 [00:14<00:31, 8762.37it/s] 31%|███       | 124684/400000 [00:14<00:31, 8718.22it/s] 31%|███▏      | 125567/400000 [00:14<00:31, 8750.63it/s] 32%|███▏      | 126454/400000 [00:14<00:31, 8785.81it/s] 32%|███▏      | 127354/400000 [00:14<00:30, 8846.51it/s] 32%|███▏      | 128239/400000 [00:15<00:30, 8812.09it/s] 32%|███▏      | 129140/400000 [00:15<00:30, 8868.91it/s] 33%|███▎      | 130028/400000 [00:15<00:30, 8867.10it/s] 33%|███▎      | 130918/400000 [00:15<00:30, 8876.20it/s] 33%|███▎      | 131806/400000 [00:15<00:30, 8857.86it/s] 33%|███▎      | 132692/400000 [00:15<00:30, 8826.17it/s] 33%|███▎      | 133577/400000 [00:15<00:30, 8831.41it/s] 34%|███▎      | 134461/400000 [00:15<00:30, 8826.22it/s] 34%|███▍      | 135346/400000 [00:15<00:29, 8832.67it/s] 34%|███▍      | 136230/400000 [00:15<00:29, 8804.61it/s] 34%|███▍      | 137111/400000 [00:16<00:29, 8778.52it/s] 34%|███▍      | 137989/400000 [00:16<00:30, 8697.44it/s] 35%|███▍      | 138859/400000 [00:16<00:30, 8495.79it/s] 35%|███▍      | 139743/400000 [00:16<00:30, 8595.71it/s] 35%|███▌      | 140629/400000 [00:16<00:29, 8672.21it/s] 35%|███▌      | 141498/400000 [00:16<00:29, 8650.03it/s] 36%|███▌      | 142374/400000 [00:16<00:29, 8681.69it/s] 36%|███▌      | 143250/400000 [00:16<00:29, 8702.87it/s] 36%|███▌      | 144127/400000 [00:16<00:29, 8721.97it/s] 36%|███▋      | 145012/400000 [00:16<00:29, 8757.38it/s] 36%|███▋      | 145888/400000 [00:17<00:29, 8491.51it/s] 37%|███▋      | 146781/400000 [00:17<00:29, 8616.94it/s] 37%|███▋      | 147653/400000 [00:17<00:29, 8644.42it/s] 37%|███▋      | 148548/400000 [00:17<00:28, 8731.79it/s] 37%|███▋      | 149444/400000 [00:17<00:28, 8797.84it/s] 38%|███▊      | 150326/400000 [00:17<00:28, 8804.14it/s] 38%|███▊      | 151214/400000 [00:17<00:28, 8825.59it/s] 38%|███▊      | 152098/400000 [00:17<00:28, 8716.21it/s] 38%|███▊      | 152989/400000 [00:17<00:28, 8771.80it/s] 38%|███▊      | 153874/400000 [00:17<00:27, 8793.61it/s] 39%|███▊      | 154754/400000 [00:18<00:27, 8766.91it/s] 39%|███▉      | 155643/400000 [00:18<00:27, 8800.79it/s] 39%|███▉      | 156524/400000 [00:18<00:28, 8682.91it/s] 39%|███▉      | 157393/400000 [00:18<00:28, 8552.81it/s] 40%|███▉      | 158274/400000 [00:18<00:28, 8627.29it/s] 40%|███▉      | 159156/400000 [00:18<00:27, 8682.77it/s] 40%|████      | 160040/400000 [00:18<00:27, 8726.80it/s] 40%|████      | 160923/400000 [00:18<00:27, 8754.39it/s] 40%|████      | 161829/400000 [00:18<00:26, 8842.29it/s] 41%|████      | 162731/400000 [00:18<00:26, 8893.91it/s] 41%|████      | 163621/400000 [00:19<00:26, 8869.41it/s] 41%|████      | 164509/400000 [00:19<00:26, 8866.67it/s] 41%|████▏     | 165396/400000 [00:19<00:26, 8862.62it/s] 42%|████▏     | 166283/400000 [00:19<00:26, 8856.89it/s] 42%|████▏     | 167169/400000 [00:19<00:26, 8839.39it/s] 42%|████▏     | 168054/400000 [00:19<00:26, 8800.74it/s] 42%|████▏     | 168935/400000 [00:19<00:26, 8690.33it/s] 42%|████▏     | 169817/400000 [00:19<00:26, 8726.97it/s] 43%|████▎     | 170699/400000 [00:19<00:26, 8753.62it/s] 43%|████▎     | 171582/400000 [00:19<00:26, 8773.77it/s] 43%|████▎     | 172460/400000 [00:20<00:26, 8744.97it/s] 43%|████▎     | 173335/400000 [00:20<00:25, 8730.82it/s] 44%|████▎     | 174223/400000 [00:20<00:25, 8773.72it/s] 44%|████▍     | 175109/400000 [00:20<00:25, 8798.47it/s] 44%|████▍     | 175989/400000 [00:20<00:25, 8776.07it/s] 44%|████▍     | 176867/400000 [00:20<00:25, 8767.57it/s] 44%|████▍     | 177744/400000 [00:20<00:25, 8751.41it/s] 45%|████▍     | 178643/400000 [00:20<00:25, 8819.59it/s] 45%|████▍     | 179543/400000 [00:20<00:24, 8870.79it/s] 45%|████▌     | 180439/400000 [00:20<00:24, 8896.15it/s] 45%|████▌     | 181332/400000 [00:21<00:24, 8903.25it/s] 46%|████▌     | 182223/400000 [00:21<00:24, 8785.33it/s] 46%|████▌     | 183102/400000 [00:21<00:24, 8767.39it/s] 46%|████▌     | 183989/400000 [00:21<00:24, 8796.42it/s] 46%|████▌     | 184877/400000 [00:21<00:24, 8819.76it/s] 46%|████▋     | 185760/400000 [00:21<00:24, 8799.78it/s] 47%|████▋     | 186641/400000 [00:21<00:24, 8778.21it/s] 47%|████▋     | 187523/400000 [00:21<00:24, 8790.51it/s] 47%|████▋     | 188411/400000 [00:21<00:24, 8815.86it/s] 47%|████▋     | 189297/400000 [00:21<00:23, 8829.02it/s] 48%|████▊     | 190180/400000 [00:22<00:23, 8756.24it/s] 48%|████▊     | 191062/400000 [00:22<00:23, 8773.27it/s] 48%|████▊     | 191951/400000 [00:22<00:23, 8805.88it/s] 48%|████▊     | 192833/400000 [00:22<00:23, 8809.38it/s] 48%|████▊     | 193723/400000 [00:22<00:23, 8834.10it/s] 49%|████▊     | 194607/400000 [00:22<00:23, 8784.24it/s] 49%|████▉     | 195486/400000 [00:22<00:23, 8757.01it/s] 49%|████▉     | 196374/400000 [00:22<00:23, 8791.46it/s] 49%|████▉     | 197265/400000 [00:22<00:22, 8824.52it/s] 50%|████▉     | 198148/400000 [00:22<00:22, 8820.32it/s] 50%|████▉     | 199037/400000 [00:23<00:22, 8838.18it/s] 50%|████▉     | 199921/400000 [00:23<00:22, 8750.20it/s] 50%|█████     | 200802/400000 [00:23<00:22, 8766.91it/s] 50%|█████     | 201680/400000 [00:23<00:22, 8768.58it/s] 51%|█████     | 202558/400000 [00:23<00:22, 8769.31it/s] 51%|█████     | 203436/400000 [00:23<00:22, 8741.49it/s] 51%|█████     | 204311/400000 [00:23<00:22, 8625.13it/s] 51%|█████▏    | 205192/400000 [00:23<00:22, 8677.23it/s] 52%|█████▏    | 206075/400000 [00:23<00:22, 8721.56it/s] 52%|█████▏    | 206960/400000 [00:24<00:22, 8758.38it/s] 52%|█████▏    | 207843/400000 [00:24<00:21, 8778.84it/s] 52%|█████▏    | 208726/400000 [00:24<00:21, 8791.46it/s] 52%|█████▏    | 209624/400000 [00:24<00:21, 8846.14it/s] 53%|█████▎    | 210509/400000 [00:24<00:21, 8755.83it/s] 53%|█████▎    | 211385/400000 [00:24<00:21, 8693.50it/s] 53%|█████▎    | 212255/400000 [00:24<00:21, 8608.55it/s] 53%|█████▎    | 213120/400000 [00:24<00:21, 8620.51it/s] 53%|█████▎    | 213995/400000 [00:24<00:21, 8657.20it/s] 54%|█████▎    | 214885/400000 [00:24<00:21, 8727.14it/s] 54%|█████▍    | 215759/400000 [00:25<00:21, 8678.52it/s] 54%|█████▍    | 216651/400000 [00:25<00:20, 8746.93it/s] 54%|█████▍    | 217527/400000 [00:25<00:21, 8685.84it/s] 55%|█████▍    | 218417/400000 [00:25<00:20, 8748.95it/s] 55%|█████▍    | 219310/400000 [00:25<00:20, 8801.72it/s] 55%|█████▌    | 220191/400000 [00:25<00:20, 8795.83it/s] 55%|█████▌    | 221083/400000 [00:25<00:20, 8830.22it/s] 55%|█████▌    | 221967/400000 [00:25<00:20, 8771.84it/s] 56%|█████▌    | 222845/400000 [00:25<00:20, 8757.51it/s] 56%|█████▌    | 223743/400000 [00:25<00:19, 8823.04it/s] 56%|█████▌    | 224638/400000 [00:26<00:19, 8858.53it/s] 56%|█████▋    | 225533/400000 [00:26<00:19, 8885.28it/s] 57%|█████▋    | 226431/400000 [00:26<00:19, 8911.28it/s] 57%|█████▋    | 227330/400000 [00:26<00:19, 8932.90it/s] 57%|█████▋    | 228228/400000 [00:26<00:19, 8944.90it/s] 57%|█████▋    | 229123/400000 [00:26<00:19, 8873.31it/s] 58%|█████▊    | 230011/400000 [00:26<00:19, 8874.83it/s] 58%|█████▊    | 230899/400000 [00:26<00:19, 8855.79it/s] 58%|█████▊    | 231785/400000 [00:26<00:19, 8836.56it/s] 58%|█████▊    | 232669/400000 [00:26<00:18, 8815.74it/s] 58%|█████▊    | 233551/400000 [00:27<00:18, 8787.56it/s] 59%|█████▊    | 234438/400000 [00:27<00:18, 8809.49it/s] 59%|█████▉    | 235327/400000 [00:27<00:18, 8831.51it/s] 59%|█████▉    | 236218/400000 [00:27<00:18, 8851.91it/s] 59%|█████▉    | 237104/400000 [00:27<00:18, 8842.66it/s] 60%|█████▉    | 238005/400000 [00:27<00:18, 8892.20it/s] 60%|█████▉    | 238895/400000 [00:27<00:18, 8863.11it/s] 60%|█████▉    | 239786/400000 [00:27<00:18, 8874.38it/s] 60%|██████    | 240675/400000 [00:27<00:17, 8877.51it/s] 60%|██████    | 241563/400000 [00:27<00:17, 8820.53it/s] 61%|██████    | 242465/400000 [00:28<00:17, 8878.72it/s] 61%|██████    | 243355/400000 [00:28<00:17, 8882.76it/s] 61%|██████    | 244252/400000 [00:28<00:17, 8907.44it/s] 61%|██████▏   | 245143/400000 [00:28<00:17, 8901.87it/s] 62%|██████▏   | 246034/400000 [00:28<00:17, 8903.91it/s] 62%|██████▏   | 246925/400000 [00:28<00:17, 8834.36it/s] 62%|██████▏   | 247809/400000 [00:28<00:17, 8830.07it/s] 62%|██████▏   | 248693/400000 [00:28<00:17, 8818.15it/s] 62%|██████▏   | 249575/400000 [00:28<00:17, 8806.42it/s] 63%|██████▎   | 250456/400000 [00:28<00:17, 8776.95it/s] 63%|██████▎   | 251350/400000 [00:29<00:16, 8823.47it/s] 63%|██████▎   | 252233/400000 [00:29<00:16, 8824.23it/s] 63%|██████▎   | 253120/400000 [00:29<00:16, 8837.26it/s] 64%|██████▎   | 254005/400000 [00:29<00:16, 8838.44it/s] 64%|██████▎   | 254896/400000 [00:29<00:16, 8856.90it/s] 64%|██████▍   | 255782/400000 [00:29<00:16, 8849.52it/s] 64%|██████▍   | 256667/400000 [00:29<00:16, 8802.92it/s] 64%|██████▍   | 257548/400000 [00:29<00:16, 8768.42it/s] 65%|██████▍   | 258425/400000 [00:29<00:16, 8762.98it/s] 65%|██████▍   | 259314/400000 [00:29<00:15, 8800.24it/s] 65%|██████▌   | 260198/400000 [00:30<00:15, 8811.26it/s] 65%|██████▌   | 261089/400000 [00:30<00:15, 8839.71it/s] 65%|██████▌   | 261981/400000 [00:30<00:15, 8863.59it/s] 66%|██████▌   | 262868/400000 [00:30<00:15, 8844.80it/s] 66%|██████▌   | 263753/400000 [00:30<00:15, 8844.79it/s] 66%|██████▌   | 264666/400000 [00:30<00:15, 8926.16it/s] 66%|██████▋   | 265571/400000 [00:30<00:15, 8961.55it/s] 67%|██████▋   | 266468/400000 [00:30<00:14, 8907.94it/s] 67%|██████▋   | 267359/400000 [00:30<00:14, 8865.96it/s] 67%|██████▋   | 268258/400000 [00:30<00:14, 8901.82it/s] 67%|██████▋   | 269149/400000 [00:31<00:14, 8742.11it/s] 68%|██████▊   | 270043/400000 [00:31<00:14, 8798.69it/s] 68%|██████▊   | 270924/400000 [00:31<00:14, 8705.35it/s] 68%|██████▊   | 271819/400000 [00:31<00:14, 8775.15it/s] 68%|██████▊   | 272699/400000 [00:31<00:14, 8781.09it/s] 68%|██████▊   | 273583/400000 [00:31<00:14, 8798.29it/s] 69%|██████▊   | 274468/400000 [00:31<00:14, 8811.42it/s] 69%|██████▉   | 275350/400000 [00:31<00:14, 8776.98it/s] 69%|██████▉   | 276228/400000 [00:31<00:14, 8768.73it/s] 69%|██████▉   | 277106/400000 [00:31<00:14, 8766.15it/s] 69%|██████▉   | 277992/400000 [00:32<00:13, 8791.24it/s] 70%|██████▉   | 278876/400000 [00:32<00:13, 8803.04it/s] 70%|██████▉   | 279757/400000 [00:32<00:13, 8767.42it/s] 70%|███████   | 280634/400000 [00:32<00:13, 8758.63it/s] 70%|███████   | 281517/400000 [00:32<00:13, 8779.26it/s] 71%|███████   | 282402/400000 [00:32<00:13, 8799.11it/s] 71%|███████   | 283283/400000 [00:32<00:13, 8800.32it/s] 71%|███████   | 284164/400000 [00:32<00:13, 8749.81it/s] 71%|███████▏  | 285040/400000 [00:32<00:13, 8746.20it/s] 71%|███████▏  | 285949/400000 [00:32<00:12, 8846.13it/s] 72%|███████▏  | 286847/400000 [00:33<00:12, 8885.35it/s] 72%|███████▏  | 287736/400000 [00:33<00:12, 8881.29it/s] 72%|███████▏  | 288625/400000 [00:33<00:12, 8825.00it/s] 72%|███████▏  | 289508/400000 [00:33<00:12, 8764.87it/s] 73%|███████▎  | 290403/400000 [00:33<00:12, 8816.80it/s] 73%|███████▎  | 291292/400000 [00:33<00:12, 8838.60it/s] 73%|███████▎  | 292177/400000 [00:33<00:12, 8832.80it/s] 73%|███████▎  | 293061/400000 [00:33<00:12, 8779.58it/s] 73%|███████▎  | 293943/400000 [00:33<00:12, 8791.03it/s] 74%|███████▎  | 294828/400000 [00:33<00:11, 8806.86it/s] 74%|███████▍  | 295718/400000 [00:34<00:11, 8833.05it/s] 74%|███████▍  | 296612/400000 [00:34<00:11, 8863.59it/s] 74%|███████▍  | 297499/400000 [00:34<00:11, 8807.64it/s] 75%|███████▍  | 298380/400000 [00:34<00:11, 8790.25it/s] 75%|███████▍  | 299260/400000 [00:34<00:11, 8782.94it/s] 75%|███████▌  | 300146/400000 [00:34<00:11, 8803.46it/s] 75%|███████▌  | 301031/400000 [00:34<00:11, 8814.79it/s] 75%|███████▌  | 301913/400000 [00:34<00:11, 8799.54it/s] 76%|███████▌  | 302793/400000 [00:34<00:11, 8763.42it/s] 76%|███████▌  | 303671/400000 [00:34<00:10, 8766.80it/s] 76%|███████▌  | 304560/400000 [00:35<00:10, 8802.09it/s] 76%|███████▋  | 305444/400000 [00:35<00:10, 8812.33it/s] 77%|███████▋  | 306341/400000 [00:35<00:10, 8858.11it/s] 77%|███████▋  | 307227/400000 [00:35<00:10, 8812.83it/s] 77%|███████▋  | 308114/400000 [00:35<00:10, 8828.29it/s] 77%|███████▋  | 308997/400000 [00:35<00:10, 8823.32it/s] 77%|███████▋  | 309880/400000 [00:35<00:10, 8819.62it/s] 78%|███████▊  | 310765/400000 [00:35<00:10, 8828.06it/s] 78%|███████▊  | 311648/400000 [00:35<00:10, 8802.67it/s] 78%|███████▊  | 312538/400000 [00:35<00:09, 8831.24it/s] 78%|███████▊  | 313447/400000 [00:36<00:09, 8906.57it/s] 79%|███████▊  | 314338/400000 [00:36<00:09, 8879.84it/s] 79%|███████▉  | 315236/400000 [00:36<00:09, 8907.86it/s] 79%|███████▉  | 316127/400000 [00:36<00:09, 8854.04it/s] 79%|███████▉  | 317013/400000 [00:36<00:09, 8844.33it/s] 79%|███████▉  | 317898/400000 [00:36<00:09, 8825.55it/s] 80%|███████▉  | 318781/400000 [00:36<00:09, 8785.50it/s] 80%|███████▉  | 319663/400000 [00:36<00:09, 8794.94it/s] 80%|████████  | 320543/400000 [00:36<00:09, 8655.79it/s] 80%|████████  | 321431/400000 [00:36<00:09, 8720.01it/s] 81%|████████  | 322315/400000 [00:37<00:08, 8754.52it/s] 81%|████████  | 323201/400000 [00:37<00:08, 8783.71it/s] 81%|████████  | 324080/400000 [00:37<00:08, 8784.63it/s] 81%|████████  | 324959/400000 [00:37<00:08, 8748.27it/s] 81%|████████▏ | 325846/400000 [00:37<00:08, 8782.21it/s] 82%|████████▏ | 326729/400000 [00:37<00:08, 8795.31it/s] 82%|████████▏ | 327613/400000 [00:37<00:08, 8805.79it/s] 82%|████████▏ | 328495/400000 [00:37<00:08, 8809.44it/s] 82%|████████▏ | 329377/400000 [00:37<00:08, 8799.79it/s] 83%|████████▎ | 330258/400000 [00:37<00:07, 8795.98it/s] 83%|████████▎ | 331138/400000 [00:38<00:07, 8783.33it/s] 83%|████████▎ | 332018/400000 [00:38<00:07, 8786.93it/s] 83%|████████▎ | 332904/400000 [00:38<00:07, 8808.27it/s] 83%|████████▎ | 333785/400000 [00:38<00:07, 8791.57it/s] 84%|████████▎ | 334678/400000 [00:38<00:07, 8831.63it/s] 84%|████████▍ | 335562/400000 [00:38<00:07, 8807.56it/s] 84%|████████▍ | 336443/400000 [00:38<00:07, 8804.22it/s] 84%|████████▍ | 337326/400000 [00:38<00:07, 8810.81it/s] 85%|████████▍ | 338208/400000 [00:38<00:07, 8776.35it/s] 85%|████████▍ | 339086/400000 [00:39<00:06, 8772.79it/s] 85%|████████▍ | 339964/400000 [00:39<00:06, 8756.33it/s] 85%|████████▌ | 340841/400000 [00:39<00:06, 8759.92it/s] 85%|████████▌ | 341718/400000 [00:39<00:06, 8742.91it/s] 86%|████████▌ | 342593/400000 [00:39<00:06, 8727.14it/s] 86%|████████▌ | 343486/400000 [00:39<00:06, 8784.92it/s] 86%|████████▌ | 344365/400000 [00:39<00:06, 8781.63it/s] 86%|████████▋ | 345256/400000 [00:39<00:06, 8817.50it/s] 87%|████████▋ | 346144/400000 [00:39<00:06, 8835.94it/s] 87%|████████▋ | 347032/400000 [00:39<00:05, 8846.45it/s] 87%|████████▋ | 347923/400000 [00:40<00:05, 8864.05it/s] 87%|████████▋ | 348810/400000 [00:40<00:05, 8838.28it/s] 87%|████████▋ | 349704/400000 [00:40<00:05, 8865.88it/s] 88%|████████▊ | 350600/400000 [00:40<00:05, 8892.40it/s] 88%|████████▊ | 351494/400000 [00:40<00:05, 8897.80it/s] 88%|████████▊ | 352384/400000 [00:40<00:05, 8852.70it/s] 88%|████████▊ | 353271/400000 [00:40<00:05, 8854.81it/s] 89%|████████▊ | 354159/400000 [00:40<00:05, 8860.71it/s] 89%|████████▉ | 355052/400000 [00:40<00:05, 8881.00it/s] 89%|████████▉ | 355944/400000 [00:40<00:04, 8890.23it/s] 89%|████████▉ | 356834/400000 [00:41<00:04, 8822.07it/s] 89%|████████▉ | 357717/400000 [00:41<00:04, 8668.19it/s] 90%|████████▉ | 358602/400000 [00:41<00:04, 8720.46it/s] 90%|████████▉ | 359484/400000 [00:41<00:04, 8749.95it/s] 90%|█████████ | 360360/400000 [00:41<00:04, 8731.90it/s] 90%|█████████ | 361234/400000 [00:41<00:04, 8709.56it/s] 91%|█████████ | 362110/400000 [00:41<00:04, 8722.37it/s] 91%|█████████ | 362991/400000 [00:41<00:04, 8747.52it/s] 91%|█████████ | 363871/400000 [00:41<00:04, 8760.69it/s] 91%|█████████ | 364748/400000 [00:41<00:04, 8744.50it/s] 91%|█████████▏| 365623/400000 [00:42<00:03, 8705.64it/s] 92%|█████████▏| 366501/400000 [00:42<00:03, 8727.66it/s] 92%|█████████▏| 367380/400000 [00:42<00:03, 8743.90it/s] 92%|█████████▏| 368261/400000 [00:42<00:03, 8762.52it/s] 92%|█████████▏| 369151/400000 [00:42<00:03, 8801.26it/s] 93%|█████████▎| 370033/400000 [00:42<00:03, 8804.27it/s] 93%|█████████▎| 370922/400000 [00:42<00:03, 8827.75it/s] 93%|█████████▎| 371816/400000 [00:42<00:03, 8860.20it/s] 93%|█████████▎| 372703/400000 [00:42<00:03, 8854.95it/s] 93%|█████████▎| 373599/400000 [00:42<00:02, 8883.93it/s] 94%|█████████▎| 374488/400000 [00:43<00:02, 8873.14it/s] 94%|█████████▍| 375376/400000 [00:43<00:02, 8838.66it/s] 94%|█████████▍| 376261/400000 [00:43<00:02, 8842.01it/s] 94%|█████████▍| 377154/400000 [00:43<00:02, 8865.78it/s] 95%|█████████▍| 378041/400000 [00:43<00:02, 8678.55it/s] 95%|█████████▍| 378910/400000 [00:43<00:02, 8565.41it/s] 95%|█████████▍| 379785/400000 [00:43<00:02, 8617.85it/s] 95%|█████████▌| 380668/400000 [00:43<00:02, 8678.88it/s] 95%|█████████▌| 381555/400000 [00:43<00:02, 8735.02it/s] 96%|█████████▌| 382443/400000 [00:43<00:02, 8775.71it/s] 96%|█████████▌| 383329/400000 [00:44<00:01, 8799.90it/s] 96%|█████████▌| 384220/400000 [00:44<00:01, 8831.91it/s] 96%|█████████▋| 385115/400000 [00:44<00:01, 8865.95it/s] 97%|█████████▋| 386002/400000 [00:44<00:01, 8854.71it/s] 97%|█████████▋| 386888/400000 [00:44<00:01, 8847.57it/s] 97%|█████████▋| 387773/400000 [00:44<00:01, 8809.21it/s] 97%|█████████▋| 388655/400000 [00:44<00:01, 8802.23it/s] 97%|█████████▋| 389544/400000 [00:44<00:01, 8828.27it/s] 98%|█████████▊| 390438/400000 [00:44<00:01, 8861.48it/s] 98%|█████████▊| 391325/400000 [00:44<00:00, 8839.12it/s] 98%|█████████▊| 392209/400000 [00:45<00:00, 8828.05it/s] 98%|█████████▊| 393096/400000 [00:45<00:00, 8839.30it/s] 98%|█████████▊| 393988/400000 [00:45<00:00, 8861.52it/s] 99%|█████████▊| 394875/400000 [00:45<00:00, 8843.60it/s] 99%|█████████▉| 395769/400000 [00:45<00:00, 8871.70it/s] 99%|█████████▉| 396657/400000 [00:45<00:00, 8843.82it/s] 99%|█████████▉| 397542/400000 [00:45<00:00, 8832.35it/s]100%|█████████▉| 398426/400000 [00:45<00:00, 8807.11it/s]100%|█████████▉| 399311/400000 [00:45<00:00, 8818.63it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8710.66it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f1b748637f0>, <torchtext.data.dataset.TabularDataset object at 0x7f1b74863940>, <torchtext.vocab.Vocab object at 0x7f1b74863860>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.60 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.60 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.60 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.44 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.44 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.49 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.49 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.17 file/s]2020-06-21 06:18:35.935359: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-21 06:18:35.939754: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-21 06:18:35.939937: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562c2a8b24f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-21 06:18:35.939951: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:21, 121838.94it/s] 74%|███████▍  | 7348224/9912422 [00:00<00:14, 173931.73it/s]9920512it [00:00, 38678760.12it/s]                           
0it [00:00, ?it/s]32768it [00:00, 610845.22it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 476869.77it/s]1654784it [00:00, 12026468.08it/s]                         
0it [00:00, ?it/s]8192it [00:00, 208627.75it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

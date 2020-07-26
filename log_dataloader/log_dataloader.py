
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/7d505dc1aa33c53468c5fb72ffa7b3f0bb7798b0', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '7d505dc1aa33c53468c5fb72ffa7b3f0bb7798b0', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/7d505dc1aa33c53468c5fb72ffa7b3f0bb7798b0

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/7d505dc1aa33c53468c5fb72ffa7b3f0bb7798b0

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/7d505dc1aa33c53468c5fb72ffa7b3f0bb7798b0

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f164d868a60> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f164d868a60> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f16b84da488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f16b84da488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f16d7706ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f16d7706ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f16658089d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f16658089d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f16658089d8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 11%|█         | 1105920/9912422 [00:00<00:00, 10661945.49it/s] 44%|████▍     | 4366336/9912422 [00:00<00:00, 13359027.72it/s]9920512it [00:00, 24976750.69it/s]                             
0it [00:00, ?it/s]32768it [00:00, 593727.25it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 143652.20it/s]1654784it [00:00, 10287148.60it/s]                         
0it [00:00, ?it/s]8192it [00:00, 185988.70it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f16442a57b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f16442b2a58>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1665808620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1665808620> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f1665808620> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:25,  1.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:25,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:24,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:23,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:23,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:22,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:22,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:21,  1.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:57,  2.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:57,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:57,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:56,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:56,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:56,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:55,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:55,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:55,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:54,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:54,  2.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:38,  3.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:38,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:37,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:37,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:37,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:36,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:36,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:36,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:36,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:35,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:25,  5.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:25,  5.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:25,  5.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:24,  5.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:24,  5.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:24,  5.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:24,  5.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:24,  5.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:23,  5.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:23,  5.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:23,  5.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:16,  7.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:16,  7.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:16,  7.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:16,  7.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:16,  7.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:15,  7.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:15,  7.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:15,  7.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:15,  7.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11, 10.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11, 10.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:11, 10.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:10, 10.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:10, 10.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:10, 10.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 18.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 18.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 18.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 18.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 18.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 18.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 18.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 18.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 18.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 18.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 18.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 24.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 24.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 24.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 24.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 24.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 24.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 24.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 24.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 24.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 24.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 24.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 31.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 31.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 31.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 31.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 31.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 31.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 31.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 31.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 31.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 31.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 31.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 39.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 39.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 39.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 39.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 39.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 39.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 39.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 39.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 39.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 39.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 39.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 47.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 56.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 56.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 56.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 56.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 56.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 56.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 56.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 56.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 56.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 56.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 56.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 64.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 71.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 76.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 76.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 76.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 76.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 76.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 76.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 76.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 81.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 81.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 81.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 81.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 81.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.26s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.26s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.26s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.23 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.36s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.26s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 81.23 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.36s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.36s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.17 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.36s/ url]
0 examples [00:00, ? examples/s]2020-07-26 13:57:20.101628: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-26 13:57:20.107541: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095230000 Hz
2020-07-26 13:57:20.108044: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d70a793c10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-26 13:57:20.108071: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
76 examples [00:00, 756.96 examples/s]190 examples [00:00, 841.43 examples/s]300 examples [00:00, 905.13 examples/s]408 examples [00:00, 949.66 examples/s]518 examples [00:00, 988.81 examples/s]626 examples [00:00, 1013.81 examples/s]737 examples [00:00, 1038.52 examples/s]841 examples [00:00, 1036.51 examples/s]947 examples [00:00, 1042.25 examples/s]1056 examples [00:01, 1055.77 examples/s]1162 examples [00:01, 1054.38 examples/s]1271 examples [00:01, 1064.52 examples/s]1377 examples [00:01, 1054.41 examples/s]1482 examples [00:01, 1038.92 examples/s]1587 examples [00:01, 1041.75 examples/s]1691 examples [00:01, 1019.83 examples/s]1803 examples [00:01, 1046.20 examples/s]1915 examples [00:01, 1065.82 examples/s]2030 examples [00:01, 1088.53 examples/s]2142 examples [00:02, 1096.19 examples/s]2252 examples [00:02, 1087.30 examples/s]2362 examples [00:02, 1088.69 examples/s]2474 examples [00:02, 1096.97 examples/s]2589 examples [00:02, 1109.99 examples/s]2701 examples [00:02, 1112.73 examples/s]2813 examples [00:02, 1085.23 examples/s]2922 examples [00:02, 1085.05 examples/s]3032 examples [00:02, 1087.27 examples/s]3143 examples [00:02, 1091.92 examples/s]3253 examples [00:03, 1089.09 examples/s]3363 examples [00:03, 1091.29 examples/s]3479 examples [00:03, 1109.21 examples/s]3592 examples [00:03, 1112.94 examples/s]3706 examples [00:03, 1119.31 examples/s]3822 examples [00:03, 1129.64 examples/s]3936 examples [00:03, 1128.10 examples/s]4049 examples [00:03, 1126.75 examples/s]4162 examples [00:03, 1123.72 examples/s]4275 examples [00:03, 1103.82 examples/s]4386 examples [00:04, 1077.84 examples/s]4494 examples [00:04, 1070.80 examples/s]4602 examples [00:04, 1066.77 examples/s]4709 examples [00:04, 1046.18 examples/s]4820 examples [00:04, 1063.92 examples/s]4932 examples [00:04, 1079.17 examples/s]5046 examples [00:04, 1096.21 examples/s]5159 examples [00:04, 1103.50 examples/s]5273 examples [00:04, 1113.06 examples/s]5386 examples [00:04, 1117.88 examples/s]5498 examples [00:05, 1109.04 examples/s]5612 examples [00:05, 1117.51 examples/s]5724 examples [00:05, 1104.35 examples/s]5839 examples [00:05, 1116.79 examples/s]5951 examples [00:05, 1112.65 examples/s]6063 examples [00:05, 1107.84 examples/s]6176 examples [00:05, 1113.67 examples/s]6288 examples [00:05, 1049.82 examples/s]6397 examples [00:05, 1059.02 examples/s]6504 examples [00:06, 1049.72 examples/s]6611 examples [00:06, 1055.64 examples/s]6722 examples [00:06, 1069.92 examples/s]6831 examples [00:06, 1075.29 examples/s]6943 examples [00:06, 1087.65 examples/s]7056 examples [00:06, 1099.87 examples/s]7168 examples [00:06, 1105.83 examples/s]7279 examples [00:06, 1089.15 examples/s]7389 examples [00:06, 1081.89 examples/s]7500 examples [00:06, 1087.76 examples/s]7612 examples [00:07, 1097.10 examples/s]7722 examples [00:07, 1084.33 examples/s]7831 examples [00:07, 1082.15 examples/s]7940 examples [00:07, 1075.71 examples/s]8048 examples [00:07, 1062.10 examples/s]8159 examples [00:07, 1074.43 examples/s]8269 examples [00:07, 1080.22 examples/s]8379 examples [00:07, 1085.92 examples/s]8490 examples [00:07, 1092.03 examples/s]8602 examples [00:07, 1098.22 examples/s]8714 examples [00:08, 1102.45 examples/s]8825 examples [00:08, 1102.46 examples/s]8940 examples [00:08, 1114.57 examples/s]9052 examples [00:08, 1108.69 examples/s]9165 examples [00:08, 1112.20 examples/s]9277 examples [00:08, 1113.51 examples/s]9389 examples [00:08, 1114.42 examples/s]9503 examples [00:08, 1120.16 examples/s]9618 examples [00:08, 1128.26 examples/s]9732 examples [00:08, 1129.86 examples/s]9846 examples [00:09, 1125.04 examples/s]9959 examples [00:09, 1111.73 examples/s]10071 examples [00:09, 1059.24 examples/s]10180 examples [00:09, 1068.00 examples/s]10288 examples [00:09, 1053.39 examples/s]10395 examples [00:09, 1058.14 examples/s]10503 examples [00:09, 1062.49 examples/s]10614 examples [00:09, 1073.52 examples/s]10722 examples [00:09, 1056.03 examples/s]10836 examples [00:09, 1078.56 examples/s]10950 examples [00:10, 1094.84 examples/s]11060 examples [00:10, 1088.87 examples/s]11170 examples [00:10, 1079.22 examples/s]11279 examples [00:10, 1058.30 examples/s]11389 examples [00:10, 1067.97 examples/s]11502 examples [00:10, 1085.50 examples/s]11611 examples [00:10, 1086.84 examples/s]11720 examples [00:10, 1062.67 examples/s]11833 examples [00:10, 1081.76 examples/s]11942 examples [00:11, 1080.00 examples/s]12056 examples [00:11, 1095.02 examples/s]12166 examples [00:11, 1064.91 examples/s]12273 examples [00:11, 1037.59 examples/s]12384 examples [00:11, 1058.03 examples/s]12495 examples [00:11, 1072.33 examples/s]12606 examples [00:11, 1081.76 examples/s]12715 examples [00:11, 1083.29 examples/s]12824 examples [00:11, 1066.86 examples/s]12938 examples [00:11, 1087.55 examples/s]13051 examples [00:12, 1097.56 examples/s]13163 examples [00:12, 1103.99 examples/s]13274 examples [00:12, 1088.89 examples/s]13387 examples [00:12, 1099.59 examples/s]13498 examples [00:12, 1095.73 examples/s]13608 examples [00:12, 1096.65 examples/s]13718 examples [00:12, 1082.30 examples/s]13827 examples [00:12, 1079.55 examples/s]13939 examples [00:12, 1089.51 examples/s]14052 examples [00:12, 1099.82 examples/s]14163 examples [00:13, 1093.68 examples/s]14277 examples [00:13, 1104.22 examples/s]14388 examples [00:13, 1088.64 examples/s]14500 examples [00:13, 1097.36 examples/s]14610 examples [00:13, 1074.43 examples/s]14724 examples [00:13, 1091.42 examples/s]14834 examples [00:13, 1062.57 examples/s]14944 examples [00:13, 1072.61 examples/s]15057 examples [00:13, 1087.49 examples/s]15172 examples [00:13, 1103.13 examples/s]15283 examples [00:14, 1103.81 examples/s]15394 examples [00:14, 1103.24 examples/s]15505 examples [00:14, 1087.91 examples/s]15619 examples [00:14, 1100.70 examples/s]15731 examples [00:14, 1104.02 examples/s]15845 examples [00:14, 1112.59 examples/s]15958 examples [00:14, 1115.84 examples/s]16070 examples [00:14, 1111.24 examples/s]16182 examples [00:14, 1111.48 examples/s]16296 examples [00:14, 1117.20 examples/s]16409 examples [00:15, 1119.41 examples/s]16523 examples [00:15, 1123.12 examples/s]16636 examples [00:15, 1085.42 examples/s]16750 examples [00:15, 1098.42 examples/s]16864 examples [00:15, 1110.07 examples/s]16976 examples [00:15, 1107.61 examples/s]17088 examples [00:15, 1111.26 examples/s]17200 examples [00:15, 1080.42 examples/s]17313 examples [00:15, 1092.54 examples/s]17427 examples [00:16, 1105.29 examples/s]17538 examples [00:16, 1102.77 examples/s]17649 examples [00:16, 1040.99 examples/s]17756 examples [00:16, 1049.18 examples/s]17869 examples [00:16, 1069.33 examples/s]17985 examples [00:16, 1092.56 examples/s]18098 examples [00:16, 1103.29 examples/s]18211 examples [00:16, 1109.98 examples/s]18323 examples [00:16, 1109.02 examples/s]18435 examples [00:16, 1094.59 examples/s]18545 examples [00:17, 1067.21 examples/s]18656 examples [00:17, 1078.33 examples/s]18767 examples [00:17, 1086.33 examples/s]18876 examples [00:17, 1062.29 examples/s]18988 examples [00:17, 1076.92 examples/s]19100 examples [00:17, 1088.47 examples/s]19210 examples [00:17, 1052.35 examples/s]19316 examples [00:17, 1032.82 examples/s]19420 examples [00:17, 1028.66 examples/s]19530 examples [00:17, 1048.33 examples/s]19641 examples [00:18, 1063.97 examples/s]19751 examples [00:18, 1074.43 examples/s]19864 examples [00:18, 1088.66 examples/s]19974 examples [00:18, 1089.40 examples/s]20084 examples [00:18, 1044.58 examples/s]20200 examples [00:18, 1076.41 examples/s]20313 examples [00:18, 1091.10 examples/s]20424 examples [00:18, 1095.41 examples/s]20536 examples [00:18, 1100.99 examples/s]20647 examples [00:19, 1076.20 examples/s]20755 examples [00:19, 1061.83 examples/s]20868 examples [00:19, 1079.38 examples/s]20977 examples [00:19, 1072.95 examples/s]21085 examples [00:19, 1073.01 examples/s]21193 examples [00:19, 1039.97 examples/s]21298 examples [00:19, 1042.57 examples/s]21409 examples [00:19, 1061.51 examples/s]21518 examples [00:19, 1068.71 examples/s]21629 examples [00:19, 1080.61 examples/s]21738 examples [00:20, 1064.89 examples/s]21851 examples [00:20, 1083.36 examples/s]21965 examples [00:20, 1097.70 examples/s]22075 examples [00:20, 1093.28 examples/s]22187 examples [00:20, 1098.98 examples/s]22298 examples [00:20, 1101.17 examples/s]22409 examples [00:20, 1056.92 examples/s]22516 examples [00:20, 1042.05 examples/s]22621 examples [00:20, 1038.06 examples/s]22731 examples [00:20, 1054.41 examples/s]22839 examples [00:21, 1061.16 examples/s]22949 examples [00:21, 1071.15 examples/s]23058 examples [00:21, 1074.15 examples/s]23166 examples [00:21, 1068.07 examples/s]23281 examples [00:21, 1089.96 examples/s]23396 examples [00:21, 1106.74 examples/s]23509 examples [00:21, 1112.96 examples/s]23621 examples [00:21, 1106.25 examples/s]23732 examples [00:21, 1091.02 examples/s]23846 examples [00:21, 1104.44 examples/s]23960 examples [00:22, 1114.75 examples/s]24072 examples [00:22, 1115.69 examples/s]24184 examples [00:22, 1059.29 examples/s]24295 examples [00:22, 1072.43 examples/s]24406 examples [00:22, 1081.86 examples/s]24518 examples [00:22, 1090.39 examples/s]24628 examples [00:22, 1086.43 examples/s]24737 examples [00:22, 1086.81 examples/s]24846 examples [00:22, 1070.13 examples/s]24954 examples [00:23, 1033.70 examples/s]25064 examples [00:23, 1050.82 examples/s]25175 examples [00:23, 1065.50 examples/s]25285 examples [00:23, 1074.25 examples/s]25393 examples [00:23, 1061.80 examples/s]25504 examples [00:23, 1074.76 examples/s]25615 examples [00:23, 1084.42 examples/s]25724 examples [00:23, 1085.63 examples/s]25834 examples [00:23, 1087.24 examples/s]25945 examples [00:23, 1091.92 examples/s]26058 examples [00:24, 1100.15 examples/s]26169 examples [00:24, 1091.47 examples/s]26279 examples [00:24, 1073.19 examples/s]26390 examples [00:24, 1083.08 examples/s]26499 examples [00:24, 1080.90 examples/s]26610 examples [00:24, 1088.62 examples/s]26723 examples [00:24, 1099.87 examples/s]26834 examples [00:24, 1095.07 examples/s]26945 examples [00:24, 1097.25 examples/s]27055 examples [00:24, 1090.47 examples/s]27167 examples [00:25, 1099.15 examples/s]27278 examples [00:25, 1102.36 examples/s]27389 examples [00:25, 1101.06 examples/s]27501 examples [00:25, 1104.31 examples/s]27612 examples [00:25, 1100.54 examples/s]27723 examples [00:25, 1099.80 examples/s]27833 examples [00:25, 1098.19 examples/s]27945 examples [00:25, 1104.55 examples/s]28059 examples [00:25, 1113.55 examples/s]28171 examples [00:25, 1094.22 examples/s]28281 examples [00:26, 1084.97 examples/s]28390 examples [00:26, 1081.87 examples/s]28499 examples [00:26, 1076.02 examples/s]28607 examples [00:26, 1057.60 examples/s]28718 examples [00:26, 1072.42 examples/s]28832 examples [00:26, 1091.30 examples/s]28943 examples [00:26, 1095.93 examples/s]29057 examples [00:26, 1106.90 examples/s]29168 examples [00:26, 1100.61 examples/s]29279 examples [00:26, 1086.63 examples/s]29388 examples [00:27, 1075.26 examples/s]29502 examples [00:27, 1091.04 examples/s]29615 examples [00:27, 1102.37 examples/s]29730 examples [00:27, 1113.57 examples/s]29842 examples [00:27, 1107.43 examples/s]29958 examples [00:27, 1122.03 examples/s]30071 examples [00:27, 1073.84 examples/s]30182 examples [00:27, 1081.64 examples/s]30295 examples [00:27, 1093.14 examples/s]30406 examples [00:28, 1096.48 examples/s]30518 examples [00:28, 1102.22 examples/s]30629 examples [00:28, 1103.65 examples/s]30744 examples [00:28, 1114.89 examples/s]30857 examples [00:28, 1118.97 examples/s]30969 examples [00:28, 1114.03 examples/s]31081 examples [00:28, 1099.44 examples/s]31193 examples [00:28, 1105.32 examples/s]31304 examples [00:28, 1085.87 examples/s]31413 examples [00:28, 1074.71 examples/s]31523 examples [00:29, 1079.80 examples/s]31632 examples [00:29, 1054.65 examples/s]31738 examples [00:29, 1044.89 examples/s]31843 examples [00:29, 1032.58 examples/s]31947 examples [00:29, 1028.65 examples/s]32054 examples [00:29, 1038.93 examples/s]32168 examples [00:29, 1066.68 examples/s]32281 examples [00:29, 1083.26 examples/s]32390 examples [00:29, 1063.40 examples/s]32497 examples [00:29, 1053.01 examples/s]32609 examples [00:30, 1072.00 examples/s]32721 examples [00:30, 1084.81 examples/s]32830 examples [00:30, 1070.37 examples/s]32938 examples [00:30, 1049.45 examples/s]33044 examples [00:30, 1051.74 examples/s]33152 examples [00:30, 1059.35 examples/s]33259 examples [00:30, 1056.56 examples/s]33370 examples [00:30, 1070.71 examples/s]33483 examples [00:30, 1087.25 examples/s]33592 examples [00:30, 1045.37 examples/s]33697 examples [00:31, 1045.96 examples/s]33802 examples [00:31, 1044.77 examples/s]33911 examples [00:31, 1055.47 examples/s]34017 examples [00:31, 1048.31 examples/s]34122 examples [00:31, 1037.86 examples/s]34227 examples [00:31, 1039.09 examples/s]34334 examples [00:31, 1045.63 examples/s]34443 examples [00:31, 1057.90 examples/s]34549 examples [00:31, 1046.71 examples/s]34657 examples [00:32, 1054.14 examples/s]34769 examples [00:32, 1071.69 examples/s]34878 examples [00:32, 1075.54 examples/s]34988 examples [00:32, 1080.31 examples/s]35100 examples [00:32, 1091.29 examples/s]35210 examples [00:32, 1079.75 examples/s]35319 examples [00:32, 1068.60 examples/s]35426 examples [00:32, 1050.72 examples/s]35532 examples [00:32, 1030.59 examples/s]35642 examples [00:32, 1049.93 examples/s]35749 examples [00:33, 1055.10 examples/s]35860 examples [00:33, 1070.35 examples/s]35971 examples [00:33, 1079.28 examples/s]36080 examples [00:33, 1056.98 examples/s]36190 examples [00:33, 1067.50 examples/s]36297 examples [00:33, 1067.63 examples/s]36404 examples [00:33, 1067.29 examples/s]36511 examples [00:33, 1066.02 examples/s]36625 examples [00:33, 1085.47 examples/s]36735 examples [00:33, 1089.28 examples/s]36845 examples [00:34, 1069.39 examples/s]36955 examples [00:34, 1077.75 examples/s]37063 examples [00:34, 1074.51 examples/s]37171 examples [00:34, 1065.91 examples/s]37283 examples [00:34, 1079.32 examples/s]37392 examples [00:34, 1078.19 examples/s]37500 examples [00:34, 1069.53 examples/s]37608 examples [00:34, 1068.42 examples/s]37721 examples [00:34, 1083.55 examples/s]37833 examples [00:34, 1092.50 examples/s]37943 examples [00:35, 1087.52 examples/s]38052 examples [00:35, 1081.48 examples/s]38161 examples [00:35, 1068.54 examples/s]38268 examples [00:35, 1059.56 examples/s]38375 examples [00:35, 1032.18 examples/s]38479 examples [00:35, 1030.61 examples/s]38589 examples [00:35, 1048.99 examples/s]38695 examples [00:35, 1035.32 examples/s]38805 examples [00:35, 1053.23 examples/s]38916 examples [00:35, 1066.95 examples/s]39023 examples [00:36, 1057.34 examples/s]39135 examples [00:36, 1073.06 examples/s]39243 examples [00:36, 1057.76 examples/s]39349 examples [00:36, 1050.11 examples/s]39455 examples [00:36, 1050.23 examples/s]39564 examples [00:36, 1059.46 examples/s]39674 examples [00:36, 1071.26 examples/s]39783 examples [00:36, 1075.25 examples/s]39896 examples [00:36, 1088.28 examples/s]40005 examples [00:37, 1032.64 examples/s]40110 examples [00:37, 1037.28 examples/s]40220 examples [00:37, 1053.54 examples/s]40332 examples [00:37, 1070.21 examples/s]40444 examples [00:37, 1083.99 examples/s]40553 examples [00:37, 1069.51 examples/s]40661 examples [00:37, 1062.76 examples/s]40774 examples [00:37, 1081.26 examples/s]40886 examples [00:37, 1091.15 examples/s]40998 examples [00:37, 1099.13 examples/s]41109 examples [00:38, 1101.91 examples/s]41220 examples [00:38, 1078.42 examples/s]41333 examples [00:38, 1091.06 examples/s]41444 examples [00:38, 1095.50 examples/s]41554 examples [00:38, 1092.26 examples/s]41664 examples [00:38, 1088.23 examples/s]41773 examples [00:38, 1057.78 examples/s]41883 examples [00:38, 1068.92 examples/s]41993 examples [00:38, 1075.97 examples/s]42101 examples [00:38, 1065.91 examples/s]42214 examples [00:39, 1081.76 examples/s]42323 examples [00:39, 1073.01 examples/s]42434 examples [00:39, 1083.48 examples/s]42547 examples [00:39, 1095.22 examples/s]42660 examples [00:39, 1105.10 examples/s]42773 examples [00:39, 1110.63 examples/s]42885 examples [00:39, 1104.65 examples/s]43000 examples [00:39, 1115.25 examples/s]43112 examples [00:39, 1111.14 examples/s]43224 examples [00:39, 1066.49 examples/s]43336 examples [00:40, 1081.52 examples/s]43446 examples [00:40, 1084.43 examples/s]43559 examples [00:40, 1096.65 examples/s]43670 examples [00:40, 1098.96 examples/s]43782 examples [00:40, 1102.80 examples/s]43897 examples [00:40, 1114.49 examples/s]44009 examples [00:40, 1071.26 examples/s]44121 examples [00:40, 1084.56 examples/s]44230 examples [00:40, 1082.94 examples/s]44339 examples [00:41, 1084.98 examples/s]44448 examples [00:41, 1082.01 examples/s]44557 examples [00:41, 1073.57 examples/s]44671 examples [00:41, 1090.01 examples/s]44783 examples [00:41, 1096.39 examples/s]44893 examples [00:41, 1094.46 examples/s]45003 examples [00:41, 1078.54 examples/s]45111 examples [00:41, 1077.04 examples/s]45222 examples [00:41, 1086.64 examples/s]45335 examples [00:41, 1097.99 examples/s]45445 examples [00:42, 1097.05 examples/s]45556 examples [00:42, 1098.88 examples/s]45666 examples [00:42, 1097.57 examples/s]45780 examples [00:42, 1108.55 examples/s]45894 examples [00:42, 1115.82 examples/s]46006 examples [00:42, 1116.34 examples/s]46119 examples [00:42, 1119.53 examples/s]46231 examples [00:42, 1104.18 examples/s]46343 examples [00:42, 1107.93 examples/s]46454 examples [00:42, 1095.60 examples/s]46564 examples [00:43, 1077.96 examples/s]46672 examples [00:43, 1070.07 examples/s]46781 examples [00:43, 1074.56 examples/s]46894 examples [00:43, 1088.44 examples/s]47004 examples [00:43, 1091.79 examples/s]47114 examples [00:43, 1091.77 examples/s]47227 examples [00:43, 1101.08 examples/s]47338 examples [00:43, 1087.77 examples/s]47447 examples [00:43, 1062.66 examples/s]47560 examples [00:43, 1081.35 examples/s]47674 examples [00:44, 1096.68 examples/s]47785 examples [00:44, 1098.11 examples/s]47895 examples [00:44, 1094.79 examples/s]48010 examples [00:44, 1108.71 examples/s]48125 examples [00:44, 1119.24 examples/s]48239 examples [00:44, 1124.70 examples/s]48352 examples [00:44, 1102.63 examples/s]48463 examples [00:44, 1097.08 examples/s]48577 examples [00:44, 1108.07 examples/s]48688 examples [00:44, 1096.10 examples/s]48798 examples [00:45, 1096.58 examples/s]48908 examples [00:45, 1096.51 examples/s]49018 examples [00:45, 1093.67 examples/s]49133 examples [00:45, 1109.60 examples/s]49246 examples [00:45, 1114.90 examples/s]49361 examples [00:45, 1123.33 examples/s]49477 examples [00:45, 1132.15 examples/s]49591 examples [00:45, 1123.02 examples/s]49706 examples [00:45, 1129.16 examples/s]49821 examples [00:45, 1133.23 examples/s]49935 examples [00:46, 1117.76 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6267/50000 [00:00<00:00, 62667.07 examples/s] 39%|███▉      | 19460/50000 [00:00<00:00, 74381.09 examples/s] 66%|██████▌   | 32950/50000 [00:00<00:00, 85948.01 examples/s] 91%|█████████ | 45381/50000 [00:00<00:00, 94715.25 examples/s]                                                               0 examples [00:00, ? examples/s]90 examples [00:00, 897.00 examples/s]196 examples [00:00, 939.91 examples/s]308 examples [00:00, 985.88 examples/s]416 examples [00:00, 1010.35 examples/s]526 examples [00:00, 1034.30 examples/s]637 examples [00:00, 1051.45 examples/s]749 examples [00:00, 1070.04 examples/s]864 examples [00:00, 1092.28 examples/s]980 examples [00:00, 1109.30 examples/s]1088 examples [00:01, 1084.72 examples/s]1203 examples [00:01, 1101.70 examples/s]1312 examples [00:01, 1093.25 examples/s]1424 examples [00:01, 1098.57 examples/s]1538 examples [00:01, 1108.99 examples/s]1654 examples [00:01, 1122.53 examples/s]1769 examples [00:01, 1129.99 examples/s]1882 examples [00:01, 1129.06 examples/s]1999 examples [00:01, 1138.28 examples/s]2115 examples [00:01, 1141.99 examples/s]2230 examples [00:02, 1143.26 examples/s]2345 examples [00:02, 1126.40 examples/s]2458 examples [00:02, 1124.08 examples/s]2571 examples [00:02, 1122.59 examples/s]2684 examples [00:02, 1116.84 examples/s]2798 examples [00:02, 1121.24 examples/s]2912 examples [00:02, 1124.02 examples/s]3025 examples [00:02, 1125.28 examples/s]3142 examples [00:02, 1135.66 examples/s]3256 examples [00:02, 1120.85 examples/s]3371 examples [00:03, 1128.21 examples/s]3486 examples [00:03, 1131.74 examples/s]3600 examples [00:03, 1116.37 examples/s]3715 examples [00:03, 1123.66 examples/s]3829 examples [00:03, 1126.45 examples/s]3942 examples [00:03, 1125.23 examples/s]4058 examples [00:03, 1130.32 examples/s]4172 examples [00:03, 1093.72 examples/s]4283 examples [00:03, 1097.34 examples/s]4393 examples [00:03, 1070.68 examples/s]4507 examples [00:04, 1089.49 examples/s]4619 examples [00:04, 1097.07 examples/s]4730 examples [00:04, 1099.08 examples/s]4842 examples [00:04, 1104.59 examples/s]4955 examples [00:04, 1110.80 examples/s]5067 examples [00:04, 1110.27 examples/s]5182 examples [00:04, 1120.36 examples/s]5295 examples [00:04, 1120.54 examples/s]5411 examples [00:04, 1131.28 examples/s]5527 examples [00:04, 1137.87 examples/s]5641 examples [00:05, 1136.92 examples/s]5755 examples [00:05, 1119.25 examples/s]5868 examples [00:05, 1119.79 examples/s]5983 examples [00:05, 1127.03 examples/s]6097 examples [00:05, 1129.11 examples/s]6210 examples [00:05, 1119.85 examples/s]6323 examples [00:05, 1108.09 examples/s]6434 examples [00:05, 1095.17 examples/s]6548 examples [00:05, 1107.16 examples/s]6659 examples [00:05, 1106.23 examples/s]6773 examples [00:06, 1113.21 examples/s]6887 examples [00:06, 1118.58 examples/s]7000 examples [00:06, 1120.63 examples/s]7114 examples [00:06, 1125.19 examples/s]7229 examples [00:06, 1130.77 examples/s]7343 examples [00:06, 1128.05 examples/s]7456 examples [00:06, 1100.15 examples/s]7568 examples [00:06, 1103.94 examples/s]7683 examples [00:06, 1116.37 examples/s]7795 examples [00:07, 1098.51 examples/s]7908 examples [00:07, 1105.82 examples/s]8019 examples [00:07, 1105.24 examples/s]8130 examples [00:07, 1103.90 examples/s]8241 examples [00:07, 1098.93 examples/s]8351 examples [00:07, 1093.89 examples/s]8464 examples [00:07, 1103.30 examples/s]8577 examples [00:07, 1108.40 examples/s]8688 examples [00:07, 1108.75 examples/s]8804 examples [00:07, 1123.57 examples/s]8920 examples [00:08, 1131.49 examples/s]9034 examples [00:08, 1129.03 examples/s]9148 examples [00:08, 1130.34 examples/s]9262 examples [00:08, 1124.36 examples/s]9375 examples [00:08, 1104.35 examples/s]9487 examples [00:08, 1106.03 examples/s]9598 examples [00:08, 1104.63 examples/s]9709 examples [00:08, 1097.12 examples/s]9819 examples [00:08, 1094.48 examples/s]9931 examples [00:08, 1100.24 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteL4TS4X/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteL4TS4X/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f16658089d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f16658089d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f16658089d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f16d0d10fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f16638e6c88>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f16658089d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f16658089d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f16658089d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f16442b20f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f16442b2160>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f15de016620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f15de016620> 

  function with postional parmater data_info <function split_train_valid at 0x7f15de016620> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f15de016730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f15de016730> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f15de016730> , (data_info, **args) 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:17:35, 13.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:02:08, 18.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:10:42, 26.1kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:26:02, 37.2kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:29:34, 53.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.93M/862M [00:01<3:07:24, 75.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.3M/862M [00:01<2:10:25, 108kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:01<1:31:07, 154kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.0M/862M [00:01<1:03:27, 220kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.4M/862M [00:01<44:12, 314kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.7M/862M [00:01<30:57, 447kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:01<21:36, 636kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.3M/862M [00:02<15:12, 900kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.4M/862M [00:02<10:39, 1.28MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.9M/862M [00:02<07:32, 1.79MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.2M/862M [00:02<05:20, 2.52MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.3M/862M [00:03<1:19:04, 170kB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.9M/862M [00:03<56:13, 239kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:03<39:26, 340kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.4M/862M [00:05<33:13, 403kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:05<26:30, 505kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:05<19:13, 696kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<13:42, 974kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:07<13:00, 1.02MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.0M/862M [00:07<10:27, 1.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:07<07:35, 1.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<08:28, 1.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.9M/862M [00:09<08:37, 1.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:09<06:36, 2.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:09<04:48, 2.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:11<08:58, 1.47MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:11<07:36, 1.73MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:11<05:39, 2.33MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:13<07:02, 1.86MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:13<07:35, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.9M/862M [00:13<05:58, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:13<04:19, 3.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:15<1:50:36, 118kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:15<1:18:42, 166kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:15<55:18, 235kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:17<41:39, 312kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:17<31:53, 407kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.1M/862M [00:17<22:53, 567kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:17<16:10, 800kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:19<16:41, 774kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.7M/862M [00:19<13:01, 991kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:19<09:22, 1.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:21<09:34, 1.34MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.8M/862M [00:21<07:59, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:21<05:54, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:23<07:09, 1.79MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:23<06:17, 2.03MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:23<04:43, 2.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:25<06:18, 2.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:25<07:00, 1.81MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<05:32, 2.29MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<05:55, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<05:26, 2.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<04:04, 3.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:27<03:39, 3.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<9:33:37, 21.9kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<6:41:45, 31.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:29<4:40:11, 44.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<3:56:58, 52.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<2:48:22, 74.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<1:58:21, 106kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<1:24:32, 147kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<1:01:47, 201kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<43:45, 284kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:33<30:43, 403kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<27:06, 456kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<20:16, 610kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<14:29, 852kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<12:56, 951kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<10:19, 1.19MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<07:31, 1.63MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<08:07, 1.51MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<06:56, 1.76MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:06, 2.38MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<06:26, 1.89MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<05:44, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<04:18, 2.81MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<05:53, 2.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<06:34, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:06, 2.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<03:44, 3.22MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<08:12, 1.46MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<07:00, 1.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:09, 2.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<06:24, 1.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<05:41, 2.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<04:14, 2.81MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:47, 2.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:15, 2.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<03:58, 2.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:34, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:18, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<04:55, 2.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<03:37, 3.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<07:06, 1.65MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<06:11, 1.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<04:34, 2.56MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:54, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<05:19, 2.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<04:00, 2.90MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:33, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<06:15, 1.86MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:57, 2.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<03:36, 3.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<11:02, 1.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<08:54, 1.30MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<06:30, 1.77MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<07:13, 1.59MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<06:13, 1.84MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<04:38, 2.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:56, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:18, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:56, 2.88MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:28, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:08, 1.85MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<04:45, 2.38MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<03:29, 3.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<07:18, 1.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:17, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<04:38, 2.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:52, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:23, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:57, 2.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<03:35, 3.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<09:14, 1.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<07:36, 1.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<05:33, 2.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:26, 1.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<05:26, 2.03MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:01, 2.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<02:59, 3.67MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<1:36:00, 115kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<1:08:15, 161kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<47:56, 229kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<36:02, 303kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<27:26, 398kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<19:38, 555kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:16<13:52, 784kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<10:46, 1.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<8:15:30, 21.9kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<5:46:56, 31.3kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:18<4:02:08, 44.6kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<2:54:35, 61.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<2:04:27, 86.7kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<1:27:35, 123kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<1:01:09, 175kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<1:16:52, 139kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<54:53, 195kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<38:36, 277kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<29:23, 362kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<22:46, 468kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<16:22, 650kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<11:32, 918kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<14:57, 708kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<11:31, 917kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<08:19, 1.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<08:16, 1.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<08:01, 1.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<06:09, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<04:24, 2.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<1:10:06, 149kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<50:33, 207kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<35:31, 293kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<27:19, 380kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<20:10, 514kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<14:21, 721kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<12:27, 828kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<10:48, 954kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<07:59, 1.29MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<05:43, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<08:58, 1.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<07:19, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:19, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<06:04, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:07, 1.99MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<03:46, 2.69MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:38<02:46, 3.63MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<40:08, 252kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<29:06, 347kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<20:34, 489kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<16:43, 600kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<13:44, 730kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<10:02, 998kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<07:08, 1.40MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<10:21, 963kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<08:15, 1.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<06:01, 1.65MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:30, 1.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:35, 1.77MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<04:09, 2.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:12, 1.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:38, 2.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:29, 2.81MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:44, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:18, 2.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<03:15, 2.99MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:34, 2.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:09, 1.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<04:01, 2.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<02:56, 3.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<06:30, 1.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:33, 1.73MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<04:07, 2.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<05:05, 1.87MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:33, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<03:25, 2.78MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:37, 2.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:09, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<04:05, 2.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<02:57, 3.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<36:11, 260kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<26:08, 360kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<18:29, 508kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<15:04, 620kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<12:26, 751kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<09:10, 1.02MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<07:53, 1.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<06:27, 1.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:42, 1.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:27, 1.69MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:44, 1.94MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:30, 2.62MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<02:56, 3.11MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<7:19:07, 20.8kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<5:07:26, 29.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<3:34:06, 42.4kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<2:49:15, 53.6kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<2:00:16, 75.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<1:24:26, 107kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<59:00, 153kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<45:42, 197kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<32:54, 274kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<23:12, 387kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<18:15, 490kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<14:39, 610kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<10:38, 839kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<07:32, 1.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<10:00, 887kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<07:54, 1.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<05:44, 1.54MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<06:02, 1.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<06:00, 1.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:38, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:39, 1.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:08, 2.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:04, 2.83MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<04:11, 2.07MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:42, 1.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<03:41, 2.35MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:40, 3.21MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<1:03:49, 135kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<45:31, 189kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<31:59, 268kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<24:17, 351kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<18:48, 453kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<13:35, 627kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<09:33, 886kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<1:00:47, 139kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<43:24, 195kB/s]  .vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<30:30, 276kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<23:12, 362kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<17:05, 491kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<12:08, 689kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<10:25, 798kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<08:08, 1.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<05:53, 1.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<06:04, 1.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:56, 1.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<03:39, 2.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:29, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:48, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<03:42, 2.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:41, 3.02MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<06:15, 1.30MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<05:11, 1.56MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<03:50, 2.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:33, 1.76MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:49, 1.67MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<03:47, 2.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:43, 2.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<32:43, 244kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<23:42, 336kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<16:44, 475kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<13:31, 585kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<10:16, 770kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<07:20, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<06:57, 1.13MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<06:27, 1.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<04:51, 1.61MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<03:28, 2.24MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<09:43, 800kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<07:35, 1.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<05:29, 1.41MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:38, 1.37MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:43, 1.63MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<03:27, 2.22MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:13, 1.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:26, 1.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:29, 2.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:50<02:31, 3.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<7:17:32, 17.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<5:06:47, 24.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<3:34:11, 35.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<2:30:58, 49.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<1:47:09, 70.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<1:15:12, 99.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:54<52:25, 142kB/s]   .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<44:05, 169kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<31:29, 236kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<22:09, 334kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<15:51, 464kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<5:59:38, 20.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<4:11:41, 29.2kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<2:55:00, 41.7kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<2:21:30, 51.6kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<1:40:29, 72.6kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<1:10:35, 103kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<50:14, 144kB/s]  .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<35:54, 201kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<25:12, 285kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<19:12, 373kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<14:11, 504kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<10:03, 708kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<08:38, 820kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<07:29, 945kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:32, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<03:56, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<07:35, 925kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<06:02, 1.16MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<04:21, 1.60MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:42, 1.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:22, 1.59MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:21, 2.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:26, 2.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:53, 1.41MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:06, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:02, 2.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:43, 1.83MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:55, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:02, 2.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:13, 3.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<04:03, 1.66MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:32, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:38, 2.55MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<03:24, 1.96MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<03:47, 1.76MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<02:59, 2.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:09, 3.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<43:53, 151kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<31:23, 210kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<22:03, 298kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<16:53, 387kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<13:08, 497kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<09:31, 685kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<06:42, 967kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<09:59, 647kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<07:39, 844kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<05:30, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<05:19, 1.20MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<05:01, 1.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<03:50, 1.67MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:41, 1.72MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:13, 1.96MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<02:24, 2.61MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:08, 1.99MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:50, 2.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:06, 2.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:57, 2.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:22, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:37, 2.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<01:55, 3.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:52, 1.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:20, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:27, 2.48MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:08, 1.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:25, 1.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:42, 2.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:51, 2.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:36, 2.30MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<01:58, 3.03MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:45, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:07, 1.89MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:29, 2.38MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:40, 2.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:28, 2.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<01:51, 3.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:39, 2.18MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:01, 1.91MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:24, 2.39MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:36, 2.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:58, 1.92MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:19, 2.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<01:41, 3.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:53, 3.00MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<4:11:12, 22.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<2:55:44, 32.1kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<2:01:58, 45.8kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<1:35:30, 58.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<1:07:59, 82.1kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<47:43, 117kB/s]   .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<33:21, 166kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<24:53, 222kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<17:58, 307kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<12:39, 433kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<10:05, 540kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<08:12, 663kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<05:58, 910kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:53<04:13, 1.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<05:58, 899kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:38, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:21, 1.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<02:24, 2.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<15:05, 352kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<11:05, 478kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<07:51, 671kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<06:42, 782kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<05:45, 910kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:17, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<03:01, 1.71MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<45:12, 114kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<32:07, 161kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<22:30, 228kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<16:50, 303kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<12:51, 397kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<09:12, 553kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<06:27, 782kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<07:33, 667kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:05<05:48, 866kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:10, 1.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<04:03, 1.22MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:53, 1.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:56, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<02:05, 2.34MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<04:50, 1.01MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<03:53, 1.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:48, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:05, 1.56MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<03:08, 1.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<02:26, 1.97MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:27, 1.93MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:12, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:39, 2.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:15, 2.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:03, 2.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<01:31, 3.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:09, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:26, 1.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:54, 2.42MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:23, 3.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:44, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:18, 1.97MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:43, 2.62MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:15, 1.99MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:29, 1.80MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:56, 2.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<01:24, 3.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:22, 1.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:48, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<02:02, 2.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:26, 1.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:35, 1.67MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:59, 2.17MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<01:26, 2.98MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<03:38, 1.17MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:59, 1.43MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<02:10, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:30, 1.68MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:36, 1.61MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:00, 2.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<01:26, 2.89MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<04:16, 970kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<03:24, 1.21MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:27, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:40, 1.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:16, 1.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:40, 2.42MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:07, 1.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:17, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:46, 2.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:34<01:16, 3.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<01:45, 2.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<2:59:38, 21.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<2:05:33, 31.3kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<1:26:48, 44.7kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<1:06:41, 58.0kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<47:25, 81.5kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<33:16, 116kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<23:36, 161kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<16:53, 225kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<11:49, 319kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<09:04, 412kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<07:06, 525kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<05:08, 723kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<04:08, 884kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:16, 1.12MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:21, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:28, 1.45MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:05, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:31, 2.32MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:54, 1.85MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:02, 1.72MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:34, 2.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:08, 3.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<03:10, 1.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:34, 1.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:51, 1.84MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<02:05, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:09, 1.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:40, 2.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:41, 1.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:31, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:07, 2.92MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:32, 2.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:44, 1.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:22, 2.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:27, 2.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:21, 2.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:01, 3.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:26, 2.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:19, 2.35MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<00:59, 3.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:24, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:17, 2.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<00:58, 3.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:22, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:34, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:14, 2.38MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<00:53, 3.28MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<2:48:57, 17.2kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<1:58:17, 24.6kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<1:22:08, 35.0kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<57:26, 49.5kB/s]  .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<40:45, 69.7kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<28:32, 99.0kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<20:03, 138kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<14:17, 194kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<09:58, 275kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<07:31, 359kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<05:48, 465kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<04:11, 642kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<03:18, 798kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:34, 1.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:51, 1.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:52, 1.36MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:50, 1.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:24, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:22, 1.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:13, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:54, 2.72MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:11, 2.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:04, 2.25MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:47, 3.01MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:06, 2.13MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:15, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:58, 2.40MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<00:42, 3.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:48, 1.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:29, 1.53MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:05, 2.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:16, 1.75MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:20, 1.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:02, 2.11MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<00:48, 2.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<1:37:03, 22.2kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<1:07:38, 31.7kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:27<46:14, 45.3kB/s]  .vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<36:23, 57.4kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<25:51, 80.6kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<18:05, 115kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<12:40, 159kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<09:03, 222kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<06:18, 315kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<04:47, 408kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<03:44, 521kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<02:41, 721kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<01:51, 1.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<02:23, 788kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:51, 1.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:19, 1.39MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:20, 1.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:07, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:48, 2.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:58, 1.80MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:51, 2.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:37, 2.74MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:49, 2.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:44, 2.23MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:33, 2.95MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:45, 2.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:41, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:30, 3.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:43, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:39, 2.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:29, 3.06MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:41, 2.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:32, 2.70MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:24, 3.49MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:35, 2.34MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:41, 2.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:33, 2.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:35, 2.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:31, 2.53MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:23, 3.33MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:17, 4.47MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<02:35, 489kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<02:03, 611kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<01:29, 840kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<01:01, 1.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:31, 787kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<01:10, 1.01MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:50, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:49, 1.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:48, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:36, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:25, 2.51MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:44, 1.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:37, 1.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:26, 2.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:32, 1.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:34, 1.72MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:26, 2.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:00<00:18, 3.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:40, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:34, 1.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:24, 2.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:28, 1.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:24, 2.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:18, 2.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:23, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:20, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:15, 2.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:20, 2.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:22, 1.87MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:17, 2.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:08<00:12, 3.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:32, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:26, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:18, 1.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:20, 1.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:17, 1.95MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:12, 2.60MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:15, 1.97MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:16, 1.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:12, 2.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:14<00:08, 3.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<02:58, 149kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<02:05, 208kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<01:23, 295kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:16<00:54, 412kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<18:12, 20.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<12:19, 29.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:18<07:23, 41.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<05:54, 51.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<04:09, 72.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<02:47, 103kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<01:38, 144kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<01:08, 202kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:43, 286kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:27, 374kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:20, 482kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:13, 666kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:07, 823kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:05, 1.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:02, 1.45MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.65MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<38:35:06,  2.88it/s]  0%|          | 907/400000 [00:00<26:57:07,  4.11it/s]  0%|          | 1745/400000 [00:00<18:49:50,  5.87it/s]  1%|          | 2587/400000 [00:00<13:09:27,  8.39it/s]  1%|          | 3509/400000 [00:00<9:11:33, 11.98it/s]   1%|          | 4391/400000 [00:00<6:25:27, 17.11it/s]  1%|▏         | 5269/400000 [00:00<4:29:26, 24.42it/s]  2%|▏         | 6160/400000 [00:01<3:08:24, 34.84it/s]  2%|▏         | 7056/400000 [00:01<2:11:48, 49.69it/s]  2%|▏         | 7963/400000 [00:01<1:32:15, 70.82it/s]  2%|▏         | 8821/400000 [00:01<1:04:40, 100.80it/s]  2%|▏         | 9674/400000 [00:01<45:24, 143.25it/s]    3%|▎         | 10517/400000 [00:01<31:57, 203.16it/s]  3%|▎         | 11371/400000 [00:01<22:32, 287.30it/s]  3%|▎         | 12222/400000 [00:01<15:58, 404.57it/s]  3%|▎         | 13112/400000 [00:01<11:22, 566.90it/s]  4%|▎         | 14019/400000 [00:01<08:09, 788.73it/s]  4%|▎         | 14903/400000 [00:02<05:54, 1085.25it/s]  4%|▍         | 15780/400000 [00:02<04:21, 1467.87it/s]  4%|▍         | 16660/400000 [00:02<03:15, 1957.05it/s]  4%|▍         | 17542/400000 [00:02<02:29, 2552.92it/s]  5%|▍         | 18483/400000 [00:02<01:56, 3266.83it/s]  5%|▍         | 19376/400000 [00:02<01:34, 4032.46it/s]  5%|▌         | 20268/400000 [00:02<01:18, 4809.30it/s]  5%|▌         | 21155/400000 [00:02<01:08, 5543.29it/s]  6%|▌         | 22034/400000 [00:02<01:01, 6178.32it/s]  6%|▌         | 22932/400000 [00:02<00:55, 6816.19it/s]  6%|▌         | 23811/400000 [00:03<00:51, 7306.83it/s]  6%|▌         | 24689/400000 [00:03<00:49, 7600.70it/s]  6%|▋         | 25574/400000 [00:03<00:47, 7935.28it/s]  7%|▋         | 26454/400000 [00:03<00:45, 8173.90it/s]  7%|▋         | 27398/400000 [00:03<00:43, 8516.51it/s]  7%|▋         | 28325/400000 [00:03<00:42, 8728.41it/s]  7%|▋         | 29230/400000 [00:03<00:42, 8697.64it/s]  8%|▊         | 30168/400000 [00:03<00:41, 8891.18it/s]  8%|▊         | 31108/400000 [00:03<00:40, 9036.50it/s]  8%|▊         | 32024/400000 [00:03<00:40, 9019.93it/s]  8%|▊         | 32937/400000 [00:04<00:40, 9051.59it/s]  8%|▊         | 33849/400000 [00:04<00:41, 8895.50it/s]  9%|▊         | 34744/400000 [00:04<00:41, 8889.13it/s]  9%|▉         | 35651/400000 [00:04<00:40, 8940.62it/s]  9%|▉         | 36552/400000 [00:04<00:40, 8959.58it/s]  9%|▉         | 37485/400000 [00:04<00:39, 9066.05it/s] 10%|▉         | 38411/400000 [00:04<00:39, 9120.15it/s] 10%|▉         | 39346/400000 [00:04<00:39, 9187.72it/s] 10%|█         | 40281/400000 [00:04<00:38, 9235.80it/s] 10%|█         | 41209/400000 [00:04<00:38, 9246.98it/s] 11%|█         | 42135/400000 [00:05<00:38, 9197.69it/s] 11%|█         | 43056/400000 [00:05<00:39, 9130.30it/s] 11%|█         | 43970/400000 [00:05<00:39, 9113.06it/s] 11%|█         | 44916/400000 [00:05<00:38, 9214.26it/s] 11%|█▏        | 45890/400000 [00:05<00:37, 9365.31it/s] 12%|█▏        | 46831/400000 [00:05<00:37, 9376.53it/s] 12%|█▏        | 47776/400000 [00:05<00:37, 9395.90it/s] 12%|█▏        | 48738/400000 [00:05<00:37, 9460.02it/s] 12%|█▏        | 49685/400000 [00:05<00:37, 9385.36it/s] 13%|█▎        | 50669/400000 [00:05<00:36, 9515.37it/s] 13%|█▎        | 51622/400000 [00:06<00:36, 9489.61it/s] 13%|█▎        | 52572/400000 [00:06<00:36, 9405.39it/s] 13%|█▎        | 53557/400000 [00:06<00:36, 9533.72it/s] 14%|█▎        | 54512/400000 [00:06<00:36, 9530.97it/s] 14%|█▍        | 55495/400000 [00:06<00:35, 9615.64it/s] 14%|█▍        | 56489/400000 [00:06<00:35, 9708.51it/s] 14%|█▍        | 57461/400000 [00:06<00:35, 9642.15it/s] 15%|█▍        | 58426/400000 [00:06<00:35, 9634.31it/s] 15%|█▍        | 59390/400000 [00:06<00:35, 9612.13it/s] 15%|█▌        | 60354/400000 [00:07<00:35, 9618.68it/s] 15%|█▌        | 61332/400000 [00:07<00:35, 9664.31it/s] 16%|█▌        | 62299/400000 [00:07<00:36, 9291.92it/s] 16%|█▌        | 63232/400000 [00:07<00:36, 9106.37it/s] 16%|█▌        | 64179/400000 [00:07<00:36, 9212.32it/s] 16%|█▋        | 65121/400000 [00:07<00:36, 9272.21it/s] 17%|█▋        | 66051/400000 [00:07<00:36, 9264.61it/s] 17%|█▋        | 66979/400000 [00:07<00:36, 9237.27it/s] 17%|█▋        | 67938/400000 [00:07<00:35, 9339.33it/s] 17%|█▋        | 68908/400000 [00:07<00:35, 9443.89it/s] 17%|█▋        | 69885/400000 [00:08<00:34, 9538.86it/s] 18%|█▊        | 70840/400000 [00:08<00:34, 9516.63it/s] 18%|█▊        | 71793/400000 [00:08<00:34, 9426.74it/s] 18%|█▊        | 72738/400000 [00:08<00:34, 9432.87it/s] 18%|█▊        | 73682/400000 [00:08<00:34, 9372.99it/s] 19%|█▊        | 74631/400000 [00:08<00:34, 9405.60it/s] 19%|█▉        | 75582/400000 [00:08<00:34, 9436.34it/s] 19%|█▉        | 76526/400000 [00:08<00:34, 9328.41it/s] 19%|█▉        | 77512/400000 [00:08<00:34, 9479.36it/s] 20%|█▉        | 78496/400000 [00:08<00:33, 9583.51it/s] 20%|█▉        | 79482/400000 [00:09<00:33, 9662.15it/s] 20%|██        | 80463/400000 [00:09<00:32, 9703.62it/s] 20%|██        | 81434/400000 [00:09<00:34, 9212.37it/s] 21%|██        | 82361/400000 [00:09<00:34, 9203.62it/s] 21%|██        | 83286/400000 [00:09<00:34, 9120.27it/s] 21%|██        | 84247/400000 [00:09<00:34, 9260.09it/s] 21%|██▏       | 85176/400000 [00:09<00:34, 9201.12it/s] 22%|██▏       | 86098/400000 [00:09<00:34, 9160.09it/s] 22%|██▏       | 87065/400000 [00:09<00:33, 9307.01it/s] 22%|██▏       | 88044/400000 [00:09<00:33, 9446.55it/s] 22%|██▏       | 88991/400000 [00:10<00:33, 9417.00it/s] 22%|██▏       | 89950/400000 [00:10<00:32, 9467.36it/s] 23%|██▎       | 90902/400000 [00:10<00:32, 9480.73it/s] 23%|██▎       | 91870/400000 [00:10<00:32, 9536.54it/s] 23%|██▎       | 92893/400000 [00:10<00:31, 9734.21it/s] 23%|██▎       | 93930/400000 [00:10<00:30, 9913.39it/s] 24%|██▎       | 94924/400000 [00:10<00:32, 9509.20it/s] 24%|██▍       | 95880/400000 [00:10<00:32, 9221.47it/s] 24%|██▍       | 96808/400000 [00:10<00:33, 9041.40it/s] 24%|██▍       | 97717/400000 [00:11<00:33, 8980.07it/s] 25%|██▍       | 98619/400000 [00:11<00:33, 8932.66it/s] 25%|██▍       | 99515/400000 [00:11<00:33, 8885.86it/s] 25%|██▌       | 100406/400000 [00:11<00:34, 8775.14it/s] 25%|██▌       | 101285/400000 [00:11<00:34, 8708.93it/s] 26%|██▌       | 102158/400000 [00:11<00:34, 8702.20it/s] 26%|██▌       | 103086/400000 [00:11<00:33, 8867.55it/s] 26%|██▌       | 104134/400000 [00:11<00:31, 9295.62it/s] 26%|██▋       | 105144/400000 [00:11<00:30, 9521.36it/s] 27%|██▋       | 106141/400000 [00:11<00:30, 9649.12it/s] 27%|██▋       | 107111/400000 [00:12<00:30, 9520.65it/s] 27%|██▋       | 108067/400000 [00:12<00:30, 9443.28it/s] 27%|██▋       | 109014/400000 [00:12<00:30, 9442.56it/s] 27%|██▋       | 109961/400000 [00:12<00:31, 9193.86it/s] 28%|██▊       | 110884/400000 [00:12<00:31, 9105.29it/s] 28%|██▊       | 111797/400000 [00:12<00:31, 9111.21it/s] 28%|██▊       | 112741/400000 [00:12<00:31, 9205.76it/s] 28%|██▊       | 113676/400000 [00:12<00:30, 9248.44it/s] 29%|██▊       | 114657/400000 [00:12<00:30, 9409.95it/s] 29%|██▉       | 115630/400000 [00:12<00:29, 9501.74it/s] 29%|██▉       | 116591/400000 [00:13<00:29, 9532.73it/s] 29%|██▉       | 117546/400000 [00:13<00:30, 9294.43it/s] 30%|██▉       | 118478/400000 [00:13<00:30, 9231.64it/s] 30%|██▉       | 119403/400000 [00:13<00:30, 9216.95it/s] 30%|███       | 120326/400000 [00:13<00:30, 9133.90it/s] 30%|███       | 121241/400000 [00:13<00:30, 9075.54it/s] 31%|███       | 122150/400000 [00:13<00:30, 9054.38it/s] 31%|███       | 123056/400000 [00:13<00:31, 8681.56it/s] 31%|███       | 123931/400000 [00:13<00:31, 8699.54it/s] 31%|███       | 124858/400000 [00:13<00:31, 8862.12it/s] 31%|███▏      | 125747/400000 [00:14<00:30, 8863.91it/s] 32%|███▏      | 126675/400000 [00:14<00:30, 8983.46it/s] 32%|███▏      | 127606/400000 [00:14<00:30, 9077.93it/s] 32%|███▏      | 128516/400000 [00:14<00:30, 8984.17it/s] 32%|███▏      | 129437/400000 [00:14<00:29, 9049.90it/s] 33%|███▎      | 130397/400000 [00:14<00:29, 9206.86it/s] 33%|███▎      | 131377/400000 [00:14<00:28, 9375.88it/s] 33%|███▎      | 132323/400000 [00:14<00:28, 9399.19it/s] 33%|███▎      | 133265/400000 [00:14<00:28, 9207.24it/s] 34%|███▎      | 134198/400000 [00:14<00:28, 9242.99it/s] 34%|███▍      | 135148/400000 [00:15<00:28, 9316.54it/s] 34%|███▍      | 136089/400000 [00:15<00:28, 9341.57it/s] 34%|███▍      | 137024/400000 [00:15<00:28, 9335.34it/s] 34%|███▍      | 137959/400000 [00:15<00:28, 9229.83it/s] 35%|███▍      | 138883/400000 [00:15<00:28, 9230.63it/s] 35%|███▍      | 139912/400000 [00:15<00:27, 9523.89it/s] 35%|███▌      | 140868/400000 [00:15<00:27, 9448.00it/s] 35%|███▌      | 141815/400000 [00:15<00:27, 9383.10it/s] 36%|███▌      | 142755/400000 [00:15<00:27, 9219.33it/s] 36%|███▌      | 143679/400000 [00:16<00:28, 9152.23it/s] 36%|███▌      | 144647/400000 [00:16<00:27, 9303.17it/s] 36%|███▋      | 145599/400000 [00:16<00:27, 9366.56it/s] 37%|███▋      | 146585/400000 [00:16<00:26, 9507.65it/s] 37%|███▋      | 147539/400000 [00:16<00:26, 9516.18it/s] 37%|███▋      | 148506/400000 [00:16<00:26, 9561.25it/s] 37%|███▋      | 149463/400000 [00:16<00:26, 9306.47it/s] 38%|███▊      | 150396/400000 [00:16<00:26, 9298.72it/s] 38%|███▊      | 151335/400000 [00:16<00:26, 9324.63it/s] 38%|███▊      | 152269/400000 [00:16<00:27, 9000.89it/s] 38%|███▊      | 153173/400000 [00:17<00:27, 8846.07it/s] 39%|███▊      | 154061/400000 [00:17<00:28, 8764.47it/s] 39%|███▊      | 154942/400000 [00:17<00:27, 8776.84it/s] 39%|███▉      | 155901/400000 [00:17<00:27, 9004.43it/s] 39%|███▉      | 156865/400000 [00:17<00:26, 9185.82it/s] 39%|███▉      | 157828/400000 [00:17<00:26, 9311.81it/s] 40%|███▉      | 158762/400000 [00:17<00:26, 9048.46it/s] 40%|███▉      | 159676/400000 [00:17<00:26, 9075.57it/s] 40%|████      | 160631/400000 [00:17<00:25, 9209.26it/s] 40%|████      | 161581/400000 [00:17<00:25, 9294.44it/s] 41%|████      | 162565/400000 [00:18<00:25, 9450.89it/s] 41%|████      | 163512/400000 [00:18<00:25, 9365.27it/s] 41%|████      | 164450/400000 [00:18<00:25, 9239.13it/s] 41%|████▏     | 165376/400000 [00:18<00:26, 8959.87it/s] 42%|████▏     | 166275/400000 [00:18<00:26, 8776.00it/s] 42%|████▏     | 167156/400000 [00:18<00:26, 8685.66it/s] 42%|████▏     | 168119/400000 [00:18<00:25, 8947.75it/s] 42%|████▏     | 169095/400000 [00:18<00:25, 9174.72it/s] 43%|████▎     | 170111/400000 [00:18<00:24, 9447.67it/s] 43%|████▎     | 171061/400000 [00:18<00:24, 9415.92it/s] 43%|████▎     | 172051/400000 [00:19<00:23, 9555.06it/s] 43%|████▎     | 173018/400000 [00:19<00:23, 9586.03it/s] 43%|████▎     | 173979/400000 [00:19<00:24, 9379.25it/s] 44%|████▎     | 174920/400000 [00:19<00:24, 9232.46it/s] 44%|████▍     | 175846/400000 [00:19<00:24, 9044.57it/s] 44%|████▍     | 176753/400000 [00:19<00:25, 8927.16it/s] 44%|████▍     | 177648/400000 [00:19<00:25, 8706.05it/s] 45%|████▍     | 178547/400000 [00:19<00:25, 8787.10it/s] 45%|████▍     | 179428/400000 [00:19<00:25, 8788.57it/s] 45%|████▌     | 180313/400000 [00:20<00:24, 8805.52it/s] 45%|████▌     | 181195/400000 [00:20<00:24, 8775.20it/s] 46%|████▌     | 182074/400000 [00:20<00:24, 8737.48it/s] 46%|████▌     | 182970/400000 [00:20<00:24, 8801.82it/s] 46%|████▌     | 183857/400000 [00:20<00:24, 8820.21it/s] 46%|████▌     | 184740/400000 [00:20<00:24, 8697.51it/s] 46%|████▋     | 185627/400000 [00:20<00:24, 8748.16it/s] 47%|████▋     | 186518/400000 [00:20<00:24, 8794.31it/s] 47%|████▋     | 187403/400000 [00:20<00:24, 8808.87it/s] 47%|████▋     | 188285/400000 [00:20<00:24, 8782.86it/s] 47%|████▋     | 189164/400000 [00:21<00:24, 8710.40it/s] 48%|████▊     | 190036/400000 [00:21<00:24, 8654.27it/s] 48%|████▊     | 190919/400000 [00:21<00:24, 8705.35it/s] 48%|████▊     | 191812/400000 [00:21<00:23, 8769.45it/s] 48%|████▊     | 192699/400000 [00:21<00:23, 8798.94it/s] 48%|████▊     | 193580/400000 [00:21<00:23, 8781.41it/s] 49%|████▊     | 194465/400000 [00:21<00:23, 8800.93it/s] 49%|████▉     | 195346/400000 [00:21<00:23, 8741.14it/s] 49%|████▉     | 196221/400000 [00:21<00:23, 8599.88it/s] 49%|████▉     | 197082/400000 [00:21<00:23, 8573.30it/s] 49%|████▉     | 197957/400000 [00:22<00:23, 8623.81it/s] 50%|████▉     | 198832/400000 [00:22<00:23, 8660.96it/s] 50%|████▉     | 199716/400000 [00:22<00:22, 8711.08it/s] 50%|█████     | 200600/400000 [00:22<00:22, 8747.58it/s] 50%|█████     | 201491/400000 [00:22<00:22, 8794.10it/s] 51%|█████     | 202371/400000 [00:22<00:22, 8760.38it/s] 51%|█████     | 203248/400000 [00:22<00:22, 8682.62it/s] 51%|█████     | 204117/400000 [00:22<00:22, 8576.74it/s] 51%|█████     | 204976/400000 [00:22<00:23, 8472.66it/s] 51%|█████▏    | 205825/400000 [00:22<00:22, 8475.43it/s] 52%|█████▏    | 206673/400000 [00:23<00:22, 8419.93it/s] 52%|█████▏    | 207522/400000 [00:23<00:22, 8439.95it/s] 52%|█████▏    | 208413/400000 [00:23<00:22, 8574.72it/s] 52%|█████▏    | 209282/400000 [00:23<00:22, 8601.22it/s] 53%|█████▎    | 210143/400000 [00:23<00:22, 8603.60it/s] 53%|█████▎    | 211025/400000 [00:23<00:21, 8665.64it/s] 53%|█████▎    | 211916/400000 [00:23<00:21, 8734.96it/s] 53%|█████▎    | 212812/400000 [00:23<00:21, 8798.87it/s] 53%|█████▎    | 213708/400000 [00:23<00:21, 8845.05it/s] 54%|█████▎    | 214593/400000 [00:23<00:21, 8811.73it/s] 54%|█████▍    | 215475/400000 [00:24<00:21, 8649.12it/s] 54%|█████▍    | 216344/400000 [00:24<00:21, 8660.31it/s] 54%|█████▍    | 217220/400000 [00:24<00:21, 8688.73it/s] 55%|█████▍    | 218090/400000 [00:24<00:21, 8645.27it/s] 55%|█████▍    | 218955/400000 [00:24<00:21, 8591.59it/s] 55%|█████▍    | 219815/400000 [00:24<00:21, 8564.49it/s] 55%|█████▌    | 220700/400000 [00:24<00:20, 8647.97it/s] 55%|█████▌    | 221588/400000 [00:24<00:20, 8713.80it/s] 56%|█████▌    | 222482/400000 [00:24<00:20, 8777.59it/s] 56%|█████▌    | 223363/400000 [00:24<00:20, 8786.59it/s] 56%|█████▌    | 224242/400000 [00:25<00:20, 8770.68it/s] 56%|█████▋    | 225125/400000 [00:25<00:19, 8786.77it/s] 57%|█████▋    | 226004/400000 [00:25<00:19, 8732.41it/s] 57%|█████▋    | 226896/400000 [00:25<00:19, 8787.60it/s] 57%|█████▋    | 227791/400000 [00:25<00:19, 8834.53it/s] 57%|█████▋    | 228675/400000 [00:25<00:19, 8665.42it/s] 57%|█████▋    | 229562/400000 [00:25<00:19, 8725.45it/s] 58%|█████▊    | 230457/400000 [00:25<00:19, 8791.36it/s] 58%|█████▊    | 231346/400000 [00:25<00:19, 8819.50it/s] 58%|█████▊    | 232229/400000 [00:25<00:19, 8819.67it/s] 58%|█████▊    | 233112/400000 [00:26<00:19, 8773.84it/s] 59%|█████▊    | 234001/400000 [00:26<00:18, 8805.91it/s] 59%|█████▊    | 234885/400000 [00:26<00:18, 8813.35it/s] 59%|█████▉    | 235773/400000 [00:26<00:18, 8830.57it/s] 59%|█████▉    | 236670/400000 [00:26<00:18, 8871.16it/s] 59%|█████▉    | 237558/400000 [00:26<00:18, 8855.82it/s] 60%|█████▉    | 238444/400000 [00:26<00:18, 8852.62it/s] 60%|█████▉    | 239334/400000 [00:26<00:18, 8866.46it/s] 60%|██████    | 240235/400000 [00:26<00:17, 8906.73it/s] 60%|██████    | 241126/400000 [00:26<00:17, 8901.72it/s] 61%|██████    | 242017/400000 [00:27<00:17, 8891.33it/s] 61%|██████    | 242907/400000 [00:27<00:17, 8856.17it/s] 61%|██████    | 243793/400000 [00:27<00:17, 8845.78it/s] 61%|██████    | 244678/400000 [00:27<00:17, 8811.57it/s] 61%|██████▏   | 245575/400000 [00:27<00:17, 8856.42it/s] 62%|██████▏   | 246461/400000 [00:27<00:17, 8845.45it/s] 62%|██████▏   | 247346/400000 [00:27<00:17, 8775.80it/s] 62%|██████▏   | 248245/400000 [00:27<00:17, 8837.48it/s] 62%|██████▏   | 249129/400000 [00:27<00:17, 8766.41it/s] 63%|██████▎   | 250006/400000 [00:27<00:17, 8655.47it/s] 63%|██████▎   | 250873/400000 [00:28<00:17, 8597.54it/s] 63%|██████▎   | 251734/400000 [00:28<00:17, 8581.03it/s] 63%|██████▎   | 252603/400000 [00:28<00:17, 8611.68it/s] 63%|██████▎   | 253478/400000 [00:28<00:16, 8650.64it/s] 64%|██████▎   | 254354/400000 [00:28<00:16, 8682.71it/s] 64%|██████▍   | 255223/400000 [00:28<00:16, 8682.17it/s] 64%|██████▍   | 256092/400000 [00:28<00:16, 8634.07it/s] 64%|██████▍   | 256987/400000 [00:28<00:16, 8723.65it/s] 64%|██████▍   | 257880/400000 [00:28<00:16, 8782.07it/s] 65%|██████▍   | 258766/400000 [00:28<00:16, 8803.19it/s] 65%|██████▍   | 259647/400000 [00:29<00:16, 8762.33it/s] 65%|██████▌   | 260524/400000 [00:29<00:16, 8616.46it/s] 65%|██████▌   | 261407/400000 [00:29<00:15, 8677.43it/s] 66%|██████▌   | 262294/400000 [00:29<00:15, 8733.28it/s] 66%|██████▌   | 263168/400000 [00:29<00:15, 8697.50it/s] 66%|██████▌   | 264050/400000 [00:29<00:15, 8732.46it/s] 66%|██████▌   | 264924/400000 [00:29<00:15, 8632.63it/s] 66%|██████▋   | 265810/400000 [00:29<00:15, 8698.71it/s] 67%|██████▋   | 266681/400000 [00:29<00:15, 8665.83it/s] 67%|██████▋   | 267571/400000 [00:30<00:15, 8732.31it/s] 67%|██████▋   | 268452/400000 [00:30<00:15, 8754.58it/s] 67%|██████▋   | 269328/400000 [00:30<00:15, 8664.71it/s] 68%|██████▊   | 270222/400000 [00:30<00:14, 8742.81it/s] 68%|██████▊   | 271097/400000 [00:30<00:14, 8646.38it/s] 68%|██████▊   | 271972/400000 [00:30<00:14, 8675.16it/s] 68%|██████▊   | 272840/400000 [00:30<00:14, 8635.32it/s] 68%|██████▊   | 273704/400000 [00:30<00:14, 8635.60it/s] 69%|██████▊   | 274602/400000 [00:30<00:14, 8733.39it/s] 69%|██████▉   | 275502/400000 [00:30<00:14, 8811.17it/s] 69%|██████▉   | 276384/400000 [00:31<00:14, 8731.31it/s] 69%|██████▉   | 277258/400000 [00:31<00:14, 8581.81it/s] 70%|██████▉   | 278118/400000 [00:31<00:14, 8487.41it/s] 70%|██████▉   | 278968/400000 [00:31<00:14, 8477.03it/s] 70%|██████▉   | 279818/400000 [00:31<00:14, 8481.68it/s] 70%|███████   | 280690/400000 [00:31<00:13, 8550.92it/s] 70%|███████   | 281555/400000 [00:31<00:13, 8577.75it/s] 71%|███████   | 282419/400000 [00:31<00:13, 8594.59it/s] 71%|███████   | 283279/400000 [00:31<00:13, 8538.24it/s] 71%|███████   | 284160/400000 [00:31<00:13, 8616.20it/s] 71%|███████▏  | 285023/400000 [00:32<00:13, 8618.33it/s] 71%|███████▏  | 285900/400000 [00:32<00:13, 8661.64it/s] 72%|███████▏  | 286772/400000 [00:32<00:13, 8676.67it/s] 72%|███████▏  | 287641/400000 [00:32<00:12, 8680.16it/s] 72%|███████▏  | 288528/400000 [00:32<00:12, 8735.19it/s] 72%|███████▏  | 289403/400000 [00:32<00:12, 8739.38it/s] 73%|███████▎  | 290293/400000 [00:32<00:12, 8785.08it/s] 73%|███████▎  | 291179/400000 [00:32<00:12, 8805.08it/s] 73%|███████▎  | 292060/400000 [00:32<00:12, 8795.29it/s] 73%|███████▎  | 292951/400000 [00:32<00:12, 8827.54it/s] 73%|███████▎  | 293834/400000 [00:33<00:12, 8817.33it/s] 74%|███████▎  | 294716/400000 [00:33<00:11, 8778.83it/s] 74%|███████▍  | 295594/400000 [00:33<00:11, 8715.04it/s] 74%|███████▍  | 296466/400000 [00:33<00:11, 8691.37it/s] 74%|███████▍  | 297336/400000 [00:33<00:11, 8685.99it/s] 75%|███████▍  | 298205/400000 [00:33<00:11, 8664.15it/s] 75%|███████▍  | 299072/400000 [00:33<00:11, 8598.69it/s] 75%|███████▍  | 299950/400000 [00:33<00:11, 8650.58it/s] 75%|███████▌  | 300816/400000 [00:33<00:11, 8617.43it/s] 75%|███████▌  | 301702/400000 [00:33<00:11, 8686.46it/s] 76%|███████▌  | 302587/400000 [00:34<00:11, 8734.43it/s] 76%|███████▌  | 303461/400000 [00:34<00:11, 8721.36it/s] 76%|███████▌  | 304347/400000 [00:34<00:10, 8761.46it/s] 76%|███████▋  | 305224/400000 [00:34<00:10, 8694.59it/s] 77%|███████▋  | 306098/400000 [00:34<00:10, 8705.96it/s] 77%|███████▋  | 306973/400000 [00:34<00:10, 8718.00it/s] 77%|███████▋  | 307845/400000 [00:34<00:10, 8692.11it/s] 77%|███████▋  | 308715/400000 [00:34<00:10, 8607.94it/s] 77%|███████▋  | 309577/400000 [00:34<00:10, 8599.45it/s] 78%|███████▊  | 310440/400000 [00:34<00:10, 8606.42it/s] 78%|███████▊  | 311316/400000 [00:35<00:10, 8651.16it/s] 78%|███████▊  | 312182/400000 [00:35<00:10, 8652.37it/s] 78%|███████▊  | 313051/400000 [00:35<00:10, 8662.40it/s] 78%|███████▊  | 313918/400000 [00:35<00:09, 8659.05it/s] 79%|███████▊  | 314802/400000 [00:35<00:09, 8710.69it/s] 79%|███████▉  | 315679/400000 [00:35<00:09, 8726.54it/s] 79%|███████▉  | 316557/400000 [00:35<00:09, 8739.79it/s] 79%|███████▉  | 317443/400000 [00:35<00:09, 8773.20it/s] 80%|███████▉  | 318321/400000 [00:35<00:09, 8743.36it/s] 80%|███████▉  | 319199/400000 [00:35<00:09, 8753.62it/s] 80%|████████  | 320088/400000 [00:36<00:09, 8791.84it/s] 80%|████████  | 320968/400000 [00:36<00:09, 8703.37it/s] 80%|████████  | 321856/400000 [00:36<00:08, 8754.25it/s] 81%|████████  | 322732/400000 [00:36<00:08, 8694.42it/s] 81%|████████  | 323607/400000 [00:36<00:08, 8709.41it/s] 81%|████████  | 324492/400000 [00:36<00:08, 8748.75it/s] 81%|████████▏ | 325370/400000 [00:36<00:08, 8756.01it/s] 82%|████████▏ | 326251/400000 [00:36<00:08, 8772.12it/s] 82%|████████▏ | 327129/400000 [00:36<00:08, 8692.97it/s] 82%|████████▏ | 327999/400000 [00:36<00:08, 8689.20it/s] 82%|████████▏ | 328869/400000 [00:37<00:08, 8643.83it/s] 82%|████████▏ | 329740/400000 [00:37<00:08, 8663.33it/s] 83%|████████▎ | 330624/400000 [00:37<00:07, 8713.92it/s] 83%|████████▎ | 331496/400000 [00:37<00:08, 8551.24it/s] 83%|████████▎ | 332371/400000 [00:37<00:07, 8606.96it/s] 83%|████████▎ | 333273/400000 [00:37<00:07, 8724.84it/s] 84%|████████▎ | 334147/400000 [00:37<00:07, 8704.75it/s] 84%|████████▍ | 335023/400000 [00:37<00:07, 8718.62it/s] 84%|████████▍ | 335896/400000 [00:37<00:07, 8707.96it/s] 84%|████████▍ | 336790/400000 [00:37<00:07, 8773.90it/s] 84%|████████▍ | 337668/400000 [00:38<00:07, 8676.95it/s] 85%|████████▍ | 338537/400000 [00:38<00:07, 8517.94it/s] 85%|████████▍ | 339421/400000 [00:38<00:07, 8611.54it/s] 85%|████████▌ | 340284/400000 [00:38<00:06, 8560.37it/s] 85%|████████▌ | 341141/400000 [00:38<00:06, 8512.01it/s] 86%|████████▌ | 342013/400000 [00:38<00:06, 8572.65it/s] 86%|████████▌ | 342892/400000 [00:38<00:06, 8635.90it/s] 86%|████████▌ | 343788/400000 [00:38<00:06, 8728.72it/s] 86%|████████▌ | 344662/400000 [00:38<00:06, 8687.91it/s] 86%|████████▋ | 345544/400000 [00:38<00:06, 8726.75it/s] 87%|████████▋ | 346426/400000 [00:39<00:06, 8754.32it/s] 87%|████████▋ | 347302/400000 [00:39<00:06, 8703.28it/s] 87%|████████▋ | 348173/400000 [00:39<00:06, 8546.41it/s] 87%|████████▋ | 349031/400000 [00:39<00:05, 8555.63it/s] 87%|████████▋ | 349915/400000 [00:39<00:05, 8636.29it/s] 88%|████████▊ | 350792/400000 [00:39<00:05, 8674.89it/s] 88%|████████▊ | 351669/400000 [00:39<00:05, 8700.58it/s] 88%|████████▊ | 352542/400000 [00:39<00:05, 8708.67it/s] 88%|████████▊ | 353414/400000 [00:39<00:05, 8636.95it/s] 89%|████████▊ | 354283/400000 [00:40<00:05, 8646.62it/s] 89%|████████▉ | 355164/400000 [00:40<00:05, 8692.99it/s] 89%|████████▉ | 356047/400000 [00:40<00:05, 8732.50it/s] 89%|████████▉ | 356935/400000 [00:40<00:04, 8773.88it/s] 89%|████████▉ | 357813/400000 [00:40<00:04, 8733.30it/s] 90%|████████▉ | 358687/400000 [00:40<00:04, 8729.10it/s] 90%|████████▉ | 359577/400000 [00:40<00:04, 8779.06it/s] 90%|█████████ | 360461/400000 [00:40<00:04, 8794.70it/s] 90%|█████████ | 361346/400000 [00:40<00:04, 8809.23it/s] 91%|█████████ | 362228/400000 [00:40<00:04, 8772.46it/s] 91%|█████████ | 363106/400000 [00:41<00:04, 8763.87it/s] 91%|█████████ | 363983/400000 [00:41<00:04, 8622.51it/s] 91%|█████████ | 364851/400000 [00:41<00:04, 8639.55it/s] 91%|█████████▏| 365716/400000 [00:41<00:03, 8629.15it/s] 92%|█████████▏| 366592/400000 [00:41<00:03, 8665.61it/s] 92%|█████████▏| 367474/400000 [00:41<00:03, 8710.65it/s] 92%|█████████▏| 368357/400000 [00:41<00:03, 8745.60it/s] 92%|█████████▏| 369240/400000 [00:41<00:03, 8767.87it/s] 93%|█████████▎| 370117/400000 [00:41<00:03, 8756.20it/s] 93%|█████████▎| 370993/400000 [00:41<00:03, 8754.50it/s] 93%|█████████▎| 371869/400000 [00:42<00:03, 8671.99it/s] 93%|█████████▎| 372737/400000 [00:42<00:03, 8553.88it/s] 93%|█████████▎| 373599/400000 [00:42<00:03, 8573.55it/s] 94%|█████████▎| 374457/400000 [00:42<00:03, 8338.61it/s] 94%|█████████▍| 375312/400000 [00:42<00:02, 8399.99it/s] 94%|█████████▍| 376176/400000 [00:42<00:02, 8469.93it/s] 94%|█████████▍| 377058/400000 [00:42<00:02, 8571.77it/s] 94%|█████████▍| 377939/400000 [00:42<00:02, 8641.36it/s] 95%|█████████▍| 378812/400000 [00:42<00:02, 8666.72it/s] 95%|█████████▍| 379687/400000 [00:42<00:02, 8691.31it/s] 95%|█████████▌| 380557/400000 [00:43<00:02, 8681.40it/s] 95%|█████████▌| 381426/400000 [00:43<00:02, 8623.54it/s] 96%|█████████▌| 382289/400000 [00:43<00:02, 8602.77it/s] 96%|█████████▌| 383150/400000 [00:43<00:01, 8533.97it/s] 96%|█████████▌| 384004/400000 [00:43<00:01, 8519.31it/s] 96%|█████████▌| 384857/400000 [00:43<00:01, 8457.93it/s] 96%|█████████▋| 385704/400000 [00:43<00:01, 8446.58it/s] 97%|█████████▋| 386549/400000 [00:43<00:01, 8169.83it/s] 97%|█████████▋| 387384/400000 [00:43<00:01, 8221.17it/s] 97%|█████████▋| 388224/400000 [00:43<00:01, 8271.71it/s] 97%|█████████▋| 389061/400000 [00:44<00:01, 8300.26it/s] 97%|█████████▋| 389892/400000 [00:44<00:01, 8268.43it/s] 98%|█████████▊| 390731/400000 [00:44<00:01, 8303.44it/s] 98%|█████████▊| 391574/400000 [00:44<00:01, 8338.79it/s] 98%|█████████▊| 392409/400000 [00:44<00:00, 8309.20it/s] 98%|█████████▊| 393252/400000 [00:44<00:00, 8343.92it/s] 99%|█████████▊| 394128/400000 [00:44<00:00, 8463.33it/s] 99%|█████████▊| 394983/400000 [00:44<00:00, 8489.11it/s] 99%|█████████▉| 395848/400000 [00:44<00:00, 8535.17it/s] 99%|█████████▉| 396706/400000 [00:44<00:00, 8547.52it/s] 99%|█████████▉| 397565/400000 [00:45<00:00, 8558.05it/s]100%|█████████▉| 398425/400000 [00:45<00:00, 8568.09it/s]100%|█████████▉| 399282/400000 [00:45<00:00, 8536.82it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8821.33it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f15de02a518>, <torchtext.data.dataset.TabularDataset object at 0x7f15de02a3c8>, <torchtext.vocab.Vocab object at 0x7f15de02a4a8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.22 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 22.49 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.22 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.22 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.41 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.41 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.72 file/s]2020-07-26 14:06:17.805474: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-26 14:06:17.809822: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095230000 Hz
2020-07-26 14:06:17.810005: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56351f9e53b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-26 14:06:17.810021: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'test', 'fashion-mnist_small.npy', 'mnist2', 'mnist_dataset_small.npy', 'train'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3874816/9912422 [00:00<00:00, 38500175.89it/s]9920512it [00:00, 32598223.00it/s]                             
0it [00:00, ?it/s]32768it [00:00, 589515.07it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 465329.00it/s]1654784it [00:00, 11021813.07it/s]                         
0it [00:00, ?it/s]8192it [00:00, 168307.16it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

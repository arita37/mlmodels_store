
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '76c59fd9a4bb974b18235d07ef6a03bf2361dc5c', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5df20e7598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5df20e7598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5e5ce0c400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5e5ce0c400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5e7c029ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5e7c029ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5e0a148400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5e0a148400> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5e0a148400> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:11, 138419.24it/s] 69%|██████▉   | 6864896/9912422 [00:00<00:15, 197560.07it/s]9920512it [00:00, 38782550.37it/s]                           
0it [00:00, ?it/s]32768it [00:00, 700804.39it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1036209.13it/s]1654784it [00:00, 12782829.10it/s]                           
0it [00:00, ?it/s]8192it [00:00, 229951.00it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5df135df60>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5df1363c88>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5e0a1480d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5e0a1480d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5e0a1480d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:01,  2.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:01,  2.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:00,  2.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:00,  2.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:59,  2.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:59,  2.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:59,  2.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:58,  2.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:41,  3.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:41,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:41,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:41,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:40,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:40,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:40,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:40,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:39,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:39,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:39,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:27,  5.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:27,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:27,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:27,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:27,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:26,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:26,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:26,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:26,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:18,  7.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:18,  7.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:18,  7.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:17,  7.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:17,  7.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:17,  7.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:12, 10.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:12, 10.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:12, 10.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:12, 10.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:12, 10.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:12, 10.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:12, 10.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:11, 10.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:11, 10.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:11, 10.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:00<00:08, 13.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:08, 13.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 13.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:08, 13.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:08, 13.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:08, 13.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:08, 13.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:08, 13.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:08, 13.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 13.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 13.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 18.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 18.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 18.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 18.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 18.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 18.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 23.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 30.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 30.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 30.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 30.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 30.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 30.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 30.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 30.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 30.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 38.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 38.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 46.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 46.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 46.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 46.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 46.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 46.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 46.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 46.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 46.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 46.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 46.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 54.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 54.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 54.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 54.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 54.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 54.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 54.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 54.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 54.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 61.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 61.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 61.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 61.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 61.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 61.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 61.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 61.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 61.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 61.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 61.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 67.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 67.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 67.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 67.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 67.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 67.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 67.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 67.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 67.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 67.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 67.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 74.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 74.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 74.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 74.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 74.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 74.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 74.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 74.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 74.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 74.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 74.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 78.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 78.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 78.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 78.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 78.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 78.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 78.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 78.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 78.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 82.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 82.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 82.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 82.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 82.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 82.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 82.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 82.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 82.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.18s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.74 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 82.74 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.54 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ url]
0 examples [00:00, ? examples/s]2020-08-31 12:07:10.299756: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-31 12:07:10.313296: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-08-31 12:07:10.313646: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f00d4d6f30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-31 12:07:10.313666: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
19 examples [00:00, 189.56 examples/s]119 examples [00:00, 250.39 examples/s]226 examples [00:00, 324.90 examples/s]337 examples [00:00, 412.16 examples/s]447 examples [00:00, 507.20 examples/s]558 examples [00:00, 605.07 examples/s]668 examples [00:00, 698.60 examples/s]779 examples [00:00, 784.83 examples/s]890 examples [00:00, 859.17 examples/s]1000 examples [00:01, 918.04 examples/s]1110 examples [00:01, 964.98 examples/s]1220 examples [00:01, 1001.22 examples/s]1328 examples [00:01, 1009.09 examples/s]1437 examples [00:01, 1029.58 examples/s]1549 examples [00:01, 1052.58 examples/s]1661 examples [00:01, 1070.80 examples/s]1771 examples [00:01, 1074.67 examples/s]1881 examples [00:01, 1080.22 examples/s]1992 examples [00:01, 1088.58 examples/s]2102 examples [00:02, 1091.69 examples/s]2212 examples [00:02, 1080.81 examples/s]2321 examples [00:02, 1075.53 examples/s]2429 examples [00:02, 1076.81 examples/s]2540 examples [00:02, 1085.57 examples/s]2651 examples [00:02, 1090.82 examples/s]2762 examples [00:02, 1095.55 examples/s]2872 examples [00:02, 1079.39 examples/s]2981 examples [00:02, 1069.22 examples/s]3090 examples [00:02, 1074.84 examples/s]3199 examples [00:03, 1077.46 examples/s]3307 examples [00:03, 1053.92 examples/s]3415 examples [00:03, 1059.31 examples/s]3522 examples [00:03, 1027.16 examples/s]3626 examples [00:03, 1016.43 examples/s]3730 examples [00:03, 1022.99 examples/s]3833 examples [00:03, 1005.07 examples/s]3939 examples [00:03, 1020.43 examples/s]4044 examples [00:03, 1027.47 examples/s]4152 examples [00:03, 1040.19 examples/s]4263 examples [00:04, 1057.76 examples/s]4370 examples [00:04, 1061.09 examples/s]4477 examples [00:04, 1061.40 examples/s]4584 examples [00:04, 1062.10 examples/s]4694 examples [00:04, 1070.95 examples/s]4804 examples [00:04, 1077.92 examples/s]4912 examples [00:04, 1066.95 examples/s]5019 examples [00:04, 1067.50 examples/s]5129 examples [00:04, 1075.03 examples/s]5237 examples [00:04, 1049.89 examples/s]5343 examples [00:05, 1028.69 examples/s]5452 examples [00:05, 1045.09 examples/s]5560 examples [00:05, 1053.84 examples/s]5669 examples [00:05, 1063.50 examples/s]5777 examples [00:05, 1068.38 examples/s]5886 examples [00:05, 1072.01 examples/s]5994 examples [00:05, 1073.35 examples/s]6102 examples [00:05, 1074.24 examples/s]6210 examples [00:05, 1075.88 examples/s]6322 examples [00:05, 1088.70 examples/s]6433 examples [00:06, 1091.02 examples/s]6543 examples [00:06, 1092.10 examples/s]6653 examples [00:06, 1092.35 examples/s]6763 examples [00:06, 1077.60 examples/s]6873 examples [00:06, 1083.78 examples/s]6982 examples [00:06, 1010.27 examples/s]7090 examples [00:06, 1028.55 examples/s]7200 examples [00:06, 1047.69 examples/s]7311 examples [00:06, 1064.19 examples/s]7421 examples [00:07, 1073.62 examples/s]7529 examples [00:07, 1072.78 examples/s]7639 examples [00:07, 1080.34 examples/s]7748 examples [00:07, 1052.15 examples/s]7858 examples [00:07, 1063.23 examples/s]7971 examples [00:07, 1080.60 examples/s]8080 examples [00:07, 1080.07 examples/s]8190 examples [00:07, 1084.16 examples/s]8299 examples [00:07, 1079.95 examples/s]8408 examples [00:07, 1081.54 examples/s]8518 examples [00:08, 1086.67 examples/s]8627 examples [00:08, 1077.44 examples/s]8737 examples [00:08, 1084.07 examples/s]8846 examples [00:08, 1079.55 examples/s]8954 examples [00:08, 1045.80 examples/s]9064 examples [00:08, 1060.18 examples/s]9173 examples [00:08, 1068.02 examples/s]9280 examples [00:08, 1058.15 examples/s]9386 examples [00:08, 1053.07 examples/s]9498 examples [00:08, 1070.13 examples/s]9609 examples [00:09, 1081.18 examples/s]9718 examples [00:09, 1071.70 examples/s]9827 examples [00:09, 1074.51 examples/s]9938 examples [00:09, 1082.73 examples/s]10047 examples [00:09, 971.77 examples/s]10157 examples [00:09, 1006.50 examples/s]10270 examples [00:09, 1040.42 examples/s]10378 examples [00:09, 1048.97 examples/s]10485 examples [00:09, 994.38 examples/s] 10596 examples [00:10, 1024.08 examples/s]10703 examples [00:10, 1037.15 examples/s]10814 examples [00:10, 1053.74 examples/s]10924 examples [00:10, 1065.16 examples/s]11032 examples [00:10, 1061.91 examples/s]11143 examples [00:10, 1075.82 examples/s]11256 examples [00:10, 1090.07 examples/s]11366 examples [00:10, 1087.65 examples/s]11479 examples [00:10, 1098.94 examples/s]11590 examples [00:10, 1073.74 examples/s]11700 examples [00:11, 1078.83 examples/s]11811 examples [00:11, 1086.94 examples/s]11923 examples [00:11, 1095.15 examples/s]12034 examples [00:11, 1097.87 examples/s]12144 examples [00:11, 1081.96 examples/s]12253 examples [00:11, 1066.08 examples/s]12360 examples [00:11, 1064.42 examples/s]12469 examples [00:11, 1071.53 examples/s]12577 examples [00:11, 1059.07 examples/s]12686 examples [00:11, 1066.40 examples/s]12793 examples [00:12, 1058.96 examples/s]12903 examples [00:12, 1069.43 examples/s]13013 examples [00:12, 1078.24 examples/s]13121 examples [00:12, 1076.15 examples/s]13229 examples [00:12, 1072.17 examples/s]13339 examples [00:12, 1080.28 examples/s]13448 examples [00:12, 1073.41 examples/s]13556 examples [00:12, 1065.02 examples/s]13664 examples [00:12, 1068.50 examples/s]13771 examples [00:12, 1066.97 examples/s]13878 examples [00:13, 1051.52 examples/s]13984 examples [00:13, 1038.59 examples/s]14092 examples [00:13, 1049.22 examples/s]14202 examples [00:13, 1061.98 examples/s]14309 examples [00:13, 1059.62 examples/s]14418 examples [00:13, 1066.39 examples/s]14528 examples [00:13, 1073.48 examples/s]14636 examples [00:13, 1066.19 examples/s]14744 examples [00:13, 1067.95 examples/s]14852 examples [00:14, 1070.67 examples/s]14960 examples [00:14, 1052.98 examples/s]15072 examples [00:14, 1069.94 examples/s]15182 examples [00:14, 1076.67 examples/s]15290 examples [00:14, 1062.50 examples/s]15401 examples [00:14, 1074.24 examples/s]15514 examples [00:14, 1087.74 examples/s]15623 examples [00:14, 1073.59 examples/s]15735 examples [00:14, 1086.25 examples/s]15849 examples [00:14, 1099.04 examples/s]15960 examples [00:15, 1079.15 examples/s]16070 examples [00:15, 1083.88 examples/s]16179 examples [00:15, 1061.82 examples/s]16289 examples [00:15, 1070.58 examples/s]16399 examples [00:15, 1079.02 examples/s]16508 examples [00:15, 1078.21 examples/s]16616 examples [00:15, 1074.28 examples/s]16728 examples [00:15, 1086.79 examples/s]16837 examples [00:15, 1080.61 examples/s]16949 examples [00:15, 1089.78 examples/s]17059 examples [00:16, 1089.95 examples/s]17171 examples [00:16, 1097.85 examples/s]17282 examples [00:16, 1100.67 examples/s]17393 examples [00:16, 1074.87 examples/s]17503 examples [00:16, 1081.33 examples/s]17612 examples [00:16, 1080.29 examples/s]17723 examples [00:16, 1088.75 examples/s]17832 examples [00:16, 1082.21 examples/s]17944 examples [00:16, 1092.23 examples/s]18055 examples [00:16, 1095.51 examples/s]18165 examples [00:17, 1075.00 examples/s]18275 examples [00:17, 1080.29 examples/s]18388 examples [00:17, 1093.16 examples/s]18498 examples [00:17, 1090.40 examples/s]18610 examples [00:17, 1096.56 examples/s]18720 examples [00:17, 1078.17 examples/s]18829 examples [00:17, 1079.10 examples/s]18939 examples [00:17, 1083.30 examples/s]19048 examples [00:17, 1070.37 examples/s]19158 examples [00:17, 1077.17 examples/s]19268 examples [00:18, 1082.42 examples/s]19377 examples [00:18, 1084.52 examples/s]19489 examples [00:18, 1087.49 examples/s]19599 examples [00:18, 1090.93 examples/s]19709 examples [00:18, 1093.34 examples/s]19819 examples [00:18, 1087.30 examples/s]19928 examples [00:18, 1085.20 examples/s]20037 examples [00:18, 1020.96 examples/s]20146 examples [00:18, 1038.88 examples/s]20256 examples [00:19, 1054.99 examples/s]20365 examples [00:19, 1063.81 examples/s]20475 examples [00:19, 1073.97 examples/s]20586 examples [00:19, 1084.14 examples/s]20697 examples [00:19, 1088.56 examples/s]20807 examples [00:19, 1091.53 examples/s]20917 examples [00:19, 1082.61 examples/s]21026 examples [00:19, 1078.75 examples/s]21134 examples [00:19, 1062.98 examples/s]21241 examples [00:19, 1044.50 examples/s]21351 examples [00:20, 1058.66 examples/s]21458 examples [00:20, 1041.84 examples/s]21567 examples [00:20, 1052.96 examples/s]21676 examples [00:20, 1062.28 examples/s]21787 examples [00:20, 1073.94 examples/s]21897 examples [00:20, 1079.55 examples/s]22007 examples [00:20, 1082.56 examples/s]22116 examples [00:20, 1065.93 examples/s]22225 examples [00:20, 1070.60 examples/s]22335 examples [00:20, 1076.78 examples/s]22447 examples [00:21, 1087.93 examples/s]22556 examples [00:21, 1078.77 examples/s]22664 examples [00:21, 1064.46 examples/s]22774 examples [00:21, 1072.29 examples/s]22884 examples [00:21, 1079.48 examples/s]22993 examples [00:21, 1034.60 examples/s]23097 examples [00:21, 1030.95 examples/s]23206 examples [00:21, 1047.07 examples/s]23315 examples [00:21, 1057.80 examples/s]23426 examples [00:21, 1070.30 examples/s]23535 examples [00:22, 1075.41 examples/s]23643 examples [00:22, 1057.39 examples/s]23754 examples [00:22, 1071.83 examples/s]23862 examples [00:22, 1056.41 examples/s]23973 examples [00:22, 1069.79 examples/s]24083 examples [00:22, 1078.04 examples/s]24195 examples [00:22, 1087.92 examples/s]24305 examples [00:22, 1091.39 examples/s]24417 examples [00:22, 1098.73 examples/s]24527 examples [00:22, 1094.74 examples/s]24637 examples [00:23, 1074.16 examples/s]24747 examples [00:23, 1081.10 examples/s]24856 examples [00:23, 1082.25 examples/s]24967 examples [00:23, 1090.37 examples/s]25078 examples [00:23, 1096.14 examples/s]25188 examples [00:23, 1092.44 examples/s]25298 examples [00:23, 1090.92 examples/s]25408 examples [00:23, 1086.30 examples/s]25520 examples [00:23, 1095.76 examples/s]25631 examples [00:24, 1096.90 examples/s]25745 examples [00:24, 1106.91 examples/s]25856 examples [00:24, 1098.67 examples/s]25966 examples [00:24, 1096.83 examples/s]26076 examples [00:24, 1097.69 examples/s]26186 examples [00:24, 1098.06 examples/s]26297 examples [00:24, 1100.19 examples/s]26408 examples [00:24, 1066.67 examples/s]26515 examples [00:24, 1064.81 examples/s]26624 examples [00:24, 1071.79 examples/s]26732 examples [00:25, 1063.46 examples/s]26842 examples [00:25, 1072.98 examples/s]26954 examples [00:25, 1085.79 examples/s]27063 examples [00:25, 1082.96 examples/s]27172 examples [00:25, 1070.63 examples/s]27286 examples [00:25, 1087.94 examples/s]27396 examples [00:25, 1089.99 examples/s]27508 examples [00:25, 1098.77 examples/s]27618 examples [00:25, 1092.62 examples/s]27730 examples [00:25, 1099.71 examples/s]27842 examples [00:26, 1104.93 examples/s]27953 examples [00:26, 1104.73 examples/s]28064 examples [00:26, 1093.65 examples/s]28174 examples [00:26, 1041.61 examples/s]28285 examples [00:26, 1059.88 examples/s]28397 examples [00:26, 1075.50 examples/s]28509 examples [00:26, 1085.64 examples/s]28619 examples [00:26, 1087.13 examples/s]28728 examples [00:26, 1082.10 examples/s]28838 examples [00:26, 1086.89 examples/s]28948 examples [00:27, 1088.51 examples/s]29057 examples [00:27, 1081.26 examples/s]29169 examples [00:27, 1089.33 examples/s]29278 examples [00:27, 1063.47 examples/s]29385 examples [00:27, 1061.21 examples/s]29495 examples [00:27, 1072.29 examples/s]29605 examples [00:27, 1078.67 examples/s]29713 examples [00:27, 1060.63 examples/s]29820 examples [00:27, 1059.97 examples/s]29927 examples [00:27, 1060.64 examples/s]30034 examples [00:28, 1000.79 examples/s]30139 examples [00:28, 1012.54 examples/s]30246 examples [00:28, 1027.73 examples/s]30351 examples [00:28, 1034.16 examples/s]30461 examples [00:28, 1051.76 examples/s]30570 examples [00:28, 1062.11 examples/s]30677 examples [00:28, 1061.70 examples/s]30784 examples [00:28, 1044.27 examples/s]30889 examples [00:28, 1036.38 examples/s]30997 examples [00:29, 1048.89 examples/s]31105 examples [00:29, 1055.56 examples/s]31215 examples [00:29, 1066.98 examples/s]31327 examples [00:29, 1080.90 examples/s]31436 examples [00:29, 1077.77 examples/s]31546 examples [00:29, 1083.17 examples/s]31657 examples [00:29, 1089.31 examples/s]31767 examples [00:29, 1090.93 examples/s]31877 examples [00:29, 1081.51 examples/s]31986 examples [00:29, 1069.30 examples/s]32093 examples [00:30, 1007.59 examples/s]32198 examples [00:30, 1017.80 examples/s]32307 examples [00:30, 1038.14 examples/s]32419 examples [00:30, 1061.30 examples/s]32526 examples [00:30, 1061.12 examples/s]32635 examples [00:30, 1067.39 examples/s]32742 examples [00:30, 1066.42 examples/s]32852 examples [00:30, 1073.85 examples/s]32960 examples [00:30, 1068.55 examples/s]33067 examples [00:30, 1068.42 examples/s]33179 examples [00:31, 1081.53 examples/s]33288 examples [00:31, 1064.21 examples/s]33396 examples [00:31, 1068.56 examples/s]33506 examples [00:31, 1075.13 examples/s]33614 examples [00:31, 1073.68 examples/s]33722 examples [00:31, 1075.15 examples/s]33830 examples [00:31, 1050.74 examples/s]33936 examples [00:31, 1044.31 examples/s]34043 examples [00:31, 1051.81 examples/s]34149 examples [00:31, 1044.29 examples/s]34257 examples [00:32, 1052.25 examples/s]34368 examples [00:32, 1067.18 examples/s]34477 examples [00:32, 1072.05 examples/s]34587 examples [00:32, 1079.10 examples/s]34695 examples [00:32, 1072.46 examples/s]34803 examples [00:32, 1069.03 examples/s]34914 examples [00:32, 1079.37 examples/s]35022 examples [00:32, 1057.70 examples/s]35128 examples [00:32, 1041.51 examples/s]35233 examples [00:33, 1026.60 examples/s]35339 examples [00:33, 1035.33 examples/s]35446 examples [00:33, 1044.98 examples/s]35551 examples [00:33, 1041.49 examples/s]35656 examples [00:33, 1018.99 examples/s]35764 examples [00:33, 1034.68 examples/s]35870 examples [00:33, 1041.61 examples/s]35976 examples [00:33, 1044.94 examples/s]36081 examples [00:33, 1005.19 examples/s]36183 examples [00:33, 1008.42 examples/s]36295 examples [00:34, 1037.07 examples/s]36402 examples [00:34, 1044.17 examples/s]36513 examples [00:34, 1060.52 examples/s]36623 examples [00:34, 1069.94 examples/s]36735 examples [00:34, 1082.35 examples/s]36847 examples [00:34, 1090.71 examples/s]36957 examples [00:34, 1088.28 examples/s]37068 examples [00:34, 1092.83 examples/s]37178 examples [00:34, 1082.22 examples/s]37287 examples [00:34, 1026.38 examples/s]37391 examples [00:35, 1011.89 examples/s]37500 examples [00:35, 1031.21 examples/s]37609 examples [00:35, 1047.22 examples/s]37718 examples [00:35, 1057.30 examples/s]37827 examples [00:35, 1057.85 examples/s]37936 examples [00:35, 1065.31 examples/s]38045 examples [00:35, 1069.81 examples/s]38155 examples [00:35, 1075.83 examples/s]38263 examples [00:35, 1071.02 examples/s]38373 examples [00:35, 1078.32 examples/s]38481 examples [00:36, 1075.59 examples/s]38591 examples [00:36, 1080.96 examples/s]38700 examples [00:36, 1074.43 examples/s]38808 examples [00:36, 1073.57 examples/s]38916 examples [00:36, 1043.53 examples/s]39021 examples [00:36, 1042.43 examples/s]39132 examples [00:36, 1059.20 examples/s]39245 examples [00:36, 1077.69 examples/s]39353 examples [00:36, 1071.31 examples/s]39463 examples [00:36, 1077.99 examples/s]39572 examples [00:37, 1080.40 examples/s]39681 examples [00:37, 1070.15 examples/s]39789 examples [00:37, 1070.50 examples/s]39897 examples [00:37, 1071.62 examples/s]40005 examples [00:37, 1021.88 examples/s]40113 examples [00:37, 1038.30 examples/s]40221 examples [00:37, 1050.13 examples/s]40331 examples [00:37, 1062.41 examples/s]40438 examples [00:37, 1051.88 examples/s]40547 examples [00:38, 1061.46 examples/s]40654 examples [00:38, 1035.56 examples/s]40763 examples [00:38, 1050.27 examples/s]40874 examples [00:38, 1066.23 examples/s]40981 examples [00:38, 1062.14 examples/s]41091 examples [00:38, 1072.46 examples/s]41203 examples [00:38, 1083.56 examples/s]41312 examples [00:38, 1074.90 examples/s]41420 examples [00:38, 1071.87 examples/s]41528 examples [00:38, 1061.28 examples/s]41637 examples [00:39, 1069.23 examples/s]41745 examples [00:39, 1072.37 examples/s]41855 examples [00:39, 1079.57 examples/s]41966 examples [00:39, 1086.70 examples/s]42075 examples [00:39, 1067.55 examples/s]42185 examples [00:39, 1075.78 examples/s]42293 examples [00:39, 1044.93 examples/s]42398 examples [00:39, 1044.62 examples/s]42503 examples [00:39, 1040.63 examples/s]42610 examples [00:39, 1047.68 examples/s]42719 examples [00:40, 1058.75 examples/s]42826 examples [00:40, 1059.45 examples/s]42933 examples [00:40, 1060.28 examples/s]43043 examples [00:40, 1069.66 examples/s]43152 examples [00:40, 1073.73 examples/s]43262 examples [00:40, 1079.45 examples/s]43372 examples [00:40, 1083.47 examples/s]43481 examples [00:40, 1083.06 examples/s]43590 examples [00:40, 1077.34 examples/s]43698 examples [00:40, 1064.79 examples/s]43809 examples [00:41, 1075.20 examples/s]43919 examples [00:41, 1081.53 examples/s]44028 examples [00:41, 1079.09 examples/s]44136 examples [00:41, 1076.19 examples/s]44244 examples [00:41, 1076.24 examples/s]44355 examples [00:41, 1085.96 examples/s]44467 examples [00:41, 1093.26 examples/s]44577 examples [00:41, 1093.90 examples/s]44687 examples [00:41, 1093.39 examples/s]44797 examples [00:41, 1087.43 examples/s]44906 examples [00:42, 1061.13 examples/s]45014 examples [00:42, 1065.60 examples/s]45122 examples [00:42, 1067.22 examples/s]45232 examples [00:42, 1076.20 examples/s]45340 examples [00:42, 1075.72 examples/s]45448 examples [00:42, 1070.04 examples/s]45558 examples [00:42, 1077.13 examples/s]45667 examples [00:42, 1078.31 examples/s]45775 examples [00:42, 1060.89 examples/s]45882 examples [00:43, 1051.79 examples/s]45990 examples [00:43, 1059.89 examples/s]46099 examples [00:43, 1066.34 examples/s]46206 examples [00:43, 1066.50 examples/s]46316 examples [00:43, 1073.90 examples/s]46424 examples [00:43, 1072.17 examples/s]46533 examples [00:43, 1076.24 examples/s]46643 examples [00:43, 1082.41 examples/s]46752 examples [00:43, 1070.50 examples/s]46860 examples [00:43, 1055.65 examples/s]46966 examples [00:44, 1050.90 examples/s]47072 examples [00:44, 1050.02 examples/s]47183 examples [00:44, 1065.05 examples/s]47292 examples [00:44, 1069.77 examples/s]47402 examples [00:44, 1076.00 examples/s]47510 examples [00:44, 1064.20 examples/s]47617 examples [00:44, 1025.85 examples/s]47726 examples [00:44, 1043.60 examples/s]47834 examples [00:44, 1053.42 examples/s]47940 examples [00:44, 1052.96 examples/s]48046 examples [00:45, 1054.73 examples/s]48154 examples [00:45, 1061.65 examples/s]48265 examples [00:45, 1075.49 examples/s]48373 examples [00:45, 1055.10 examples/s]48479 examples [00:45, 983.72 examples/s] 48586 examples [00:45, 1006.86 examples/s]48696 examples [00:45, 1030.66 examples/s]48806 examples [00:45, 1048.05 examples/s]48916 examples [00:45, 1061.25 examples/s]49025 examples [00:45, 1067.45 examples/s]49133 examples [00:46, 1070.59 examples/s]49242 examples [00:46, 1075.84 examples/s]49353 examples [00:46, 1084.50 examples/s]49462 examples [00:46, 1085.26 examples/s]49572 examples [00:46, 1088.63 examples/s]49682 examples [00:46, 1090.00 examples/s]49793 examples [00:46, 1095.76 examples/s]49903 examples [00:46, 1096.83 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7126/50000 [00:00<00:00, 71256.50 examples/s] 41%|████      | 20368/50000 [00:00<00:00, 82718.47 examples/s] 68%|██████▊   | 33814/50000 [00:00<00:00, 93513.70 examples/s] 95%|█████████▌| 47712/50000 [00:00<00:00, 103689.06 examples/s]                                                                0 examples [00:00, ? examples/s]83 examples [00:00, 826.09 examples/s]194 examples [00:00, 893.64 examples/s]304 examples [00:00, 944.74 examples/s]416 examples [00:00, 989.79 examples/s]526 examples [00:00, 1018.51 examples/s]636 examples [00:00, 1039.36 examples/s]744 examples [00:00, 1048.91 examples/s]851 examples [00:00, 1053.32 examples/s]961 examples [00:00, 1064.32 examples/s]1065 examples [00:01, 1056.83 examples/s]1177 examples [00:01, 1074.03 examples/s]1286 examples [00:01, 1077.71 examples/s]1393 examples [00:01, 1069.05 examples/s]1505 examples [00:01, 1081.89 examples/s]1613 examples [00:01, 1060.10 examples/s]1719 examples [00:01, 1058.74 examples/s]1828 examples [00:01, 1065.51 examples/s]1938 examples [00:01, 1074.43 examples/s]2050 examples [00:01, 1085.05 examples/s]2159 examples [00:02, 1057.79 examples/s]2269 examples [00:02, 1069.93 examples/s]2377 examples [00:02, 1043.10 examples/s]2483 examples [00:02, 1046.29 examples/s]2588 examples [00:02, 1045.17 examples/s]2696 examples [00:02, 1055.37 examples/s]2805 examples [00:02, 1054.04 examples/s]2911 examples [00:02, 1050.07 examples/s]3017 examples [00:02, 1045.43 examples/s]3126 examples [00:02, 1057.44 examples/s]3239 examples [00:03, 1076.93 examples/s]3347 examples [00:03, 1068.08 examples/s]3456 examples [00:03, 1072.14 examples/s]3564 examples [00:03, 1071.00 examples/s]3672 examples [00:03, 1063.55 examples/s]3781 examples [00:03, 1070.47 examples/s]3890 examples [00:03, 1074.97 examples/s]3998 examples [00:03, 1076.28 examples/s]4107 examples [00:03, 1077.85 examples/s]4217 examples [00:03, 1081.96 examples/s]4327 examples [00:04, 1085.53 examples/s]4436 examples [00:04, 1082.43 examples/s]4545 examples [00:04, 1066.62 examples/s]4652 examples [00:04, 1062.46 examples/s]4759 examples [00:04, 854.24 examples/s] 4867 examples [00:04, 909.56 examples/s]4976 examples [00:04, 956.28 examples/s]5086 examples [00:04, 992.88 examples/s]5197 examples [00:04, 1022.65 examples/s]5307 examples [00:05, 1043.08 examples/s]5417 examples [00:05, 1057.27 examples/s]5525 examples [00:05, 1048.72 examples/s]5635 examples [00:05, 1062.32 examples/s]5745 examples [00:05, 1072.67 examples/s]5853 examples [00:05, 1068.19 examples/s]5961 examples [00:05, 1065.79 examples/s]6068 examples [00:05, 1063.55 examples/s]6176 examples [00:05, 1067.44 examples/s]6286 examples [00:05, 1076.80 examples/s]6394 examples [00:06, 1073.96 examples/s]6505 examples [00:06, 1082.51 examples/s]6615 examples [00:06, 1086.34 examples/s]6724 examples [00:06, 1079.82 examples/s]6834 examples [00:06, 1085.46 examples/s]6945 examples [00:06, 1092.30 examples/s]7055 examples [00:06, 1092.23 examples/s]7165 examples [00:06, 1087.50 examples/s]7274 examples [00:06, 1060.98 examples/s]7381 examples [00:06, 1059.06 examples/s]7490 examples [00:07, 1066.71 examples/s]7604 examples [00:07, 1085.31 examples/s]7715 examples [00:07, 1090.74 examples/s]7825 examples [00:07, 1084.19 examples/s]7937 examples [00:07, 1092.25 examples/s]8047 examples [00:07, 1077.82 examples/s]8158 examples [00:07, 1087.09 examples/s]8268 examples [00:07, 1088.45 examples/s]8377 examples [00:07, 1082.96 examples/s]8486 examples [00:07, 1084.01 examples/s]8596 examples [00:08, 1086.91 examples/s]8707 examples [00:08, 1091.90 examples/s]8817 examples [00:08, 1084.45 examples/s]8926 examples [00:08, 1084.98 examples/s]9040 examples [00:08, 1099.18 examples/s]9150 examples [00:08, 1089.07 examples/s]9259 examples [00:08, 1071.63 examples/s]9370 examples [00:08, 1081.66 examples/s]9480 examples [00:08, 1085.76 examples/s]9591 examples [00:09, 1090.67 examples/s]9701 examples [00:09, 1085.88 examples/s]9812 examples [00:09, 1090.60 examples/s]9923 examples [00:09, 1095.13 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLB5F75/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLB5F75/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5e0a148400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5e0a148400> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5e0a148400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5d9032ccc0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5d903f30f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5e0a148400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5e0a148400> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5e0a148400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5d903f32e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5df1386208>), {}) 

  




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
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range

  




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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libtorch_cpu.so: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 248, in compute
    preprocessor_func = load_function(uri)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.48 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 19.13 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 19.13 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.43 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.43 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.39 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.39 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.72 file/s]2020-08-31 12:08:14.601564: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-31 12:08:14.605837: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-08-31 12:08:14.606064: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55564f30e750 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-31 12:08:14.606088: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist2', 'train', 'test', 'cifar10', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 480093.59it/s] 52%|█████▏    | 5201920/9912422 [00:00<00:06, 682945.01it/s]9920512it [00:00, 35525175.33it/s]                           
0it [00:00, ?it/s]32768it [00:00, 658664.71it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 468196.06it/s]1654784it [00:00, 11774020.17it/s]                         
0it [00:00, ?it/s]8192it [00:00, 156805.00it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

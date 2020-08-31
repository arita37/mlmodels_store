
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f497a0ce598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f497a0ce598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f49e4df3400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f49e4df3400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f4a04010ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f4a04010ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f499212a400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f499212a400> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f499212a400> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 22%|██▏       | 2228224/9912422 [00:00<00:00, 22278755.54it/s] 63%|██████▎   | 6291456/9912422 [00:00<00:00, 24753919.66it/s]9920512it [00:00, 23342056.43it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 258049.47it/s]           
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 476605.18it/s]1654784it [00:00, 11869866.69it/s]                         
0it [00:00, ?it/s]8192it [00:00, 201483.21it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4990623f28>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4979349b00>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f499212a0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f499212a0d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f499212a0d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:42,  1.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:42,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:42,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:41,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:40,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:40,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:39,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:39,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:38,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:09,  2.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:09,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:08,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:08,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:07,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:07,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:06,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:06,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:05,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:05,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<01:04,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<01:04,  2.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:45,  3.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:45,  3.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:44,  3.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:44,  3.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:44,  3.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:44,  3.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:43,  3.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:43,  3.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:43,  3.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:42,  3.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:42,  3.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:42,  3.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:29,  4.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:29,  4.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:29,  4.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:29,  4.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:28,  4.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:28,  4.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:28,  4.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:28,  4.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:28,  4.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:27,  4.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:27,  4.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:27,  4.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:19,  6.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:19,  6.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:19,  6.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:19,  6.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:18,  6.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:18,  6.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:18,  6.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:18,  6.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:18,  6.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:18,  6.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:17,  6.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:17,  6.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:12,  8.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:12,  8.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:12,  8.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:12,  8.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:12,  8.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:12,  8.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:12,  8.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:11,  8.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:11,  8.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:11,  8.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:11,  8.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:11,  8.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:08, 11.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:08, 11.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:08, 11.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:08, 11.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 11.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:07, 11.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:07, 11.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:07, 11.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:07, 11.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:07, 11.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:07, 11.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:07, 11.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 16.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:05, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:05, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:05, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:05, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:05, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:04, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:04, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:04, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:04, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:04, 16.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 21.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 21.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 21.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:03, 21.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:03, 21.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:03, 21.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:03, 21.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:03, 21.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:03, 21.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:03, 21.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:03, 21.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 28.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 28.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 28.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 28.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:02, 28.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:02, 28.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:02, 28.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:02, 28.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:02, 28.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:02, 28.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:02, 28.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 28.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 36.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 36.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 36.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 36.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 36.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 36.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 36.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 36.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 36.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 36.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:01, 36.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:01, 36.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 44.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 53.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 60.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 60.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 60.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 60.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 60.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 60.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 60.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 60.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 60.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 60.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 60.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 68.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 68.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 68.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 68.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 68.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 68.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 75.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 75.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 75.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 75.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 75.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.27s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.27s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.27s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.43 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.36s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.27s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 75.43 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.36s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.36s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.14 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.36s/ url]
0 examples [00:00, ? examples/s]2020-08-31 18:07:04.570218: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-31 18:07:04.584751: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-08-31 18:07:04.584997: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ccd296e9f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-31 18:07:04.585017: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
11 examples [00:00, 109.70 examples/s]124 examples [00:00, 150.44 examples/s]235 examples [00:00, 203.05 examples/s]348 examples [00:00, 269.17 examples/s]459 examples [00:00, 348.06 examples/s]571 examples [00:00, 438.48 examples/s]684 examples [00:00, 536.67 examples/s]795 examples [00:00, 634.54 examples/s]907 examples [00:00, 728.87 examples/s]1014 examples [00:01, 804.94 examples/s]1125 examples [00:01, 876.84 examples/s]1238 examples [00:01, 938.27 examples/s]1349 examples [00:01, 982.56 examples/s]1459 examples [00:01, 990.92 examples/s]1569 examples [00:01, 1020.13 examples/s]1681 examples [00:01, 1046.41 examples/s]1794 examples [00:01, 1067.69 examples/s]1906 examples [00:01, 1080.87 examples/s]2018 examples [00:01, 1092.13 examples/s]2131 examples [00:02, 1101.31 examples/s]2243 examples [00:02, 1101.79 examples/s]2355 examples [00:02, 1104.14 examples/s]2466 examples [00:02, 1100.08 examples/s]2577 examples [00:02, 1102.89 examples/s]2688 examples [00:02, 1100.19 examples/s]2799 examples [00:02, 1097.48 examples/s]2909 examples [00:02, 1093.02 examples/s]3021 examples [00:02, 1098.26 examples/s]3133 examples [00:02, 1102.69 examples/s]3245 examples [00:03, 1105.55 examples/s]3356 examples [00:03, 1098.79 examples/s]3468 examples [00:03, 1102.37 examples/s]3579 examples [00:03, 1037.93 examples/s]3690 examples [00:03, 1055.95 examples/s]3800 examples [00:03, 1066.63 examples/s]3912 examples [00:03, 1080.77 examples/s]4021 examples [00:03, 1080.44 examples/s]4132 examples [00:03, 1087.28 examples/s]4244 examples [00:03, 1095.75 examples/s]4355 examples [00:04, 1097.51 examples/s]4466 examples [00:04, 1101.05 examples/s]4577 examples [00:04, 1103.64 examples/s]4688 examples [00:04, 1102.63 examples/s]4800 examples [00:04, 1105.96 examples/s]4911 examples [00:04, 1105.40 examples/s]5022 examples [00:04, 1104.17 examples/s]5134 examples [00:04, 1107.46 examples/s]5246 examples [00:04, 1109.20 examples/s]5358 examples [00:04, 1111.09 examples/s]5470 examples [00:05, 1109.50 examples/s]5581 examples [00:05, 1103.10 examples/s]5692 examples [00:05, 1102.54 examples/s]5803 examples [00:05, 1098.28 examples/s]5916 examples [00:05, 1104.88 examples/s]6027 examples [00:05, 1105.07 examples/s]6138 examples [00:05, 1105.95 examples/s]6251 examples [00:05, 1111.58 examples/s]6364 examples [00:05, 1114.53 examples/s]6477 examples [00:05, 1117.31 examples/s]6590 examples [00:06, 1118.80 examples/s]6702 examples [00:06, 1116.27 examples/s]6816 examples [00:06, 1122.03 examples/s]6929 examples [00:06, 1123.05 examples/s]7042 examples [00:06, 1122.78 examples/s]7155 examples [00:06, 1122.37 examples/s]7268 examples [00:06, 1122.68 examples/s]7381 examples [00:06, 1121.62 examples/s]7494 examples [00:06, 1123.26 examples/s]7607 examples [00:06, 1121.15 examples/s]7720 examples [00:07, 1120.48 examples/s]7833 examples [00:07, 1119.93 examples/s]7948 examples [00:07, 1126.13 examples/s]8061 examples [00:07, 1125.60 examples/s]8174 examples [00:07, 1124.81 examples/s]8288 examples [00:07, 1126.82 examples/s]8401 examples [00:07, 1121.94 examples/s]8514 examples [00:07, 1120.16 examples/s]8627 examples [00:07, 1122.45 examples/s]8740 examples [00:07, 1120.94 examples/s]8853 examples [00:08, 1120.47 examples/s]8966 examples [00:08, 1113.21 examples/s]9078 examples [00:08, 1113.14 examples/s]9193 examples [00:08, 1122.83 examples/s]9306 examples [00:08, 1121.46 examples/s]9419 examples [00:08, 1117.71 examples/s]9531 examples [00:08, 1113.45 examples/s]9643 examples [00:08, 1113.20 examples/s]9755 examples [00:08, 1113.65 examples/s]9867 examples [00:08, 1115.04 examples/s]9979 examples [00:09, 1112.48 examples/s]10091 examples [00:09, 1053.24 examples/s]10204 examples [00:09, 1074.79 examples/s]10313 examples [00:09, 1066.49 examples/s]10423 examples [00:09, 1076.21 examples/s]10532 examples [00:09, 1080.00 examples/s]10643 examples [00:09, 1087.26 examples/s]10755 examples [00:09, 1096.49 examples/s]10868 examples [00:09, 1105.30 examples/s]10981 examples [00:10, 1112.59 examples/s]11093 examples [00:10, 1110.75 examples/s]11205 examples [00:10, 1110.25 examples/s]11317 examples [00:10, 1100.20 examples/s]11428 examples [00:10, 1101.10 examples/s]11540 examples [00:10, 1104.07 examples/s]11651 examples [00:10, 1101.85 examples/s]11762 examples [00:10, 1098.75 examples/s]11872 examples [00:10, 1099.09 examples/s]11982 examples [00:10, 1098.95 examples/s]12093 examples [00:11, 1101.27 examples/s]12204 examples [00:11, 1089.65 examples/s]12313 examples [00:11, 1088.54 examples/s]12422 examples [00:11, 1074.34 examples/s]12533 examples [00:11, 1084.72 examples/s]12646 examples [00:11, 1096.12 examples/s]12758 examples [00:11, 1102.29 examples/s]12869 examples [00:11, 1102.80 examples/s]12980 examples [00:11, 1042.77 examples/s]13090 examples [00:11, 1057.60 examples/s]13201 examples [00:12, 1071.57 examples/s]13312 examples [00:12, 1080.07 examples/s]13423 examples [00:12, 1086.59 examples/s]13532 examples [00:12, 1085.52 examples/s]13642 examples [00:12, 1089.19 examples/s]13752 examples [00:12, 1091.84 examples/s]13862 examples [00:12, 1093.17 examples/s]13973 examples [00:12, 1093.90 examples/s]14084 examples [00:12, 1096.29 examples/s]14194 examples [00:12, 1097.31 examples/s]14304 examples [00:13, 1098.12 examples/s]14414 examples [00:13, 1097.65 examples/s]14525 examples [00:13, 1098.81 examples/s]14635 examples [00:13, 1066.93 examples/s]14742 examples [00:13, 1065.09 examples/s]14849 examples [00:13, 1062.32 examples/s]14958 examples [00:13, 1068.77 examples/s]15065 examples [00:13, 1066.38 examples/s]15175 examples [00:13, 1074.07 examples/s]15288 examples [00:13, 1089.84 examples/s]15401 examples [00:14, 1099.79 examples/s]15514 examples [00:14, 1106.54 examples/s]15627 examples [00:14, 1111.61 examples/s]15739 examples [00:14, 1107.45 examples/s]15851 examples [00:14, 1109.76 examples/s]15963 examples [00:14, 1098.72 examples/s]16073 examples [00:14, 1059.26 examples/s]16180 examples [00:14, 1039.21 examples/s]16285 examples [00:14, 1033.87 examples/s]16398 examples [00:15, 1059.75 examples/s]16510 examples [00:15, 1075.91 examples/s]16622 examples [00:15, 1087.74 examples/s]16734 examples [00:15, 1095.53 examples/s]16844 examples [00:15, 1079.41 examples/s]16956 examples [00:15, 1089.40 examples/s]17068 examples [00:15, 1097.72 examples/s]17179 examples [00:15, 1101.18 examples/s]17291 examples [00:15, 1105.55 examples/s]17403 examples [00:15, 1107.14 examples/s]17515 examples [00:16, 1109.64 examples/s]17627 examples [00:16, 1111.61 examples/s]17740 examples [00:16, 1115.48 examples/s]17852 examples [00:16, 1115.91 examples/s]17964 examples [00:16, 1108.56 examples/s]18075 examples [00:16, 1107.00 examples/s]18186 examples [00:16, 1102.98 examples/s]18297 examples [00:16, 1085.06 examples/s]18410 examples [00:16, 1096.54 examples/s]18520 examples [00:16, 1096.44 examples/s]18630 examples [00:17, 1096.75 examples/s]18740 examples [00:17, 1094.92 examples/s]18850 examples [00:17, 1095.20 examples/s]18960 examples [00:17, 1094.69 examples/s]19070 examples [00:17, 1090.86 examples/s]19181 examples [00:17, 1094.16 examples/s]19291 examples [00:17, 1093.49 examples/s]19402 examples [00:17, 1095.46 examples/s]19513 examples [00:17, 1097.09 examples/s]19624 examples [00:17, 1099.44 examples/s]19734 examples [00:18, 1098.32 examples/s]19844 examples [00:18, 1098.78 examples/s]19955 examples [00:18, 1099.41 examples/s]20065 examples [00:18, 1041.94 examples/s]20174 examples [00:18, 1055.45 examples/s]20284 examples [00:18, 1066.93 examples/s]20395 examples [00:18, 1078.75 examples/s]20506 examples [00:18, 1085.97 examples/s]20617 examples [00:18, 1092.20 examples/s]20728 examples [00:18, 1095.33 examples/s]20838 examples [00:19, 1093.35 examples/s]20950 examples [00:19, 1100.59 examples/s]21062 examples [00:19, 1103.98 examples/s]21173 examples [00:19, 1104.10 examples/s]21284 examples [00:19, 1058.40 examples/s]21391 examples [00:19, 1056.72 examples/s]21501 examples [00:19, 1067.75 examples/s]21612 examples [00:19, 1079.36 examples/s]21723 examples [00:19, 1085.92 examples/s]21834 examples [00:19, 1091.93 examples/s]21944 examples [00:20, 1091.71 examples/s]22054 examples [00:20, 1093.37 examples/s]22165 examples [00:20, 1097.38 examples/s]22276 examples [00:20, 1100.98 examples/s]22387 examples [00:20, 1101.46 examples/s]22498 examples [00:20, 1099.93 examples/s]22609 examples [00:20, 1099.26 examples/s]22719 examples [00:20, 1098.26 examples/s]22829 examples [00:20, 1097.46 examples/s]22940 examples [00:20, 1098.51 examples/s]23050 examples [00:21, 1098.56 examples/s]23160 examples [00:21, 1093.59 examples/s]23271 examples [00:21, 1097.55 examples/s]23383 examples [00:21, 1102.07 examples/s]23494 examples [00:21, 1082.93 examples/s]23604 examples [00:21, 1085.45 examples/s]23713 examples [00:21, 1086.77 examples/s]23822 examples [00:21, 1037.75 examples/s]23932 examples [00:21, 1054.18 examples/s]24042 examples [00:22, 1067.48 examples/s]24152 examples [00:22, 1076.44 examples/s]24262 examples [00:22, 1080.77 examples/s]24372 examples [00:22, 1084.08 examples/s]24481 examples [00:22, 1060.35 examples/s]24591 examples [00:22, 1070.11 examples/s]24701 examples [00:22, 1078.24 examples/s]24810 examples [00:22, 1079.97 examples/s]24920 examples [00:22, 1085.75 examples/s]25030 examples [00:22, 1089.97 examples/s]25141 examples [00:23, 1094.97 examples/s]25251 examples [00:23, 1094.81 examples/s]25361 examples [00:23, 1093.71 examples/s]25471 examples [00:23, 1095.01 examples/s]25581 examples [00:23, 1094.73 examples/s]25691 examples [00:23, 1092.78 examples/s]25801 examples [00:23, 1089.81 examples/s]25910 examples [00:23, 1089.47 examples/s]26020 examples [00:23, 1092.17 examples/s]26130 examples [00:23, 1092.12 examples/s]26240 examples [00:24, 1091.18 examples/s]26350 examples [00:24, 1092.46 examples/s]26461 examples [00:24, 1097.23 examples/s]26571 examples [00:24, 1097.19 examples/s]26682 examples [00:24, 1098.31 examples/s]26792 examples [00:24, 1098.53 examples/s]26903 examples [00:24, 1100.52 examples/s]27015 examples [00:24, 1103.84 examples/s]27126 examples [00:24, 1101.23 examples/s]27237 examples [00:24, 1100.17 examples/s]27348 examples [00:25, 1099.78 examples/s]27458 examples [00:25, 1099.52 examples/s]27570 examples [00:25, 1104.72 examples/s]27681 examples [00:25, 1101.16 examples/s]27792 examples [00:25, 1090.91 examples/s]27904 examples [00:25, 1099.36 examples/s]28016 examples [00:25, 1103.69 examples/s]28127 examples [00:25, 1095.11 examples/s]28237 examples [00:25, 1089.06 examples/s]28346 examples [00:25, 1087.30 examples/s]28455 examples [00:26, 1086.87 examples/s]28565 examples [00:26, 1090.72 examples/s]28676 examples [00:26, 1095.04 examples/s]28786 examples [00:26, 1094.33 examples/s]28896 examples [00:26, 1095.41 examples/s]29006 examples [00:26, 1096.19 examples/s]29117 examples [00:26, 1097.54 examples/s]29227 examples [00:26, 1098.14 examples/s]29338 examples [00:26, 1100.12 examples/s]29450 examples [00:26, 1103.13 examples/s]29561 examples [00:27, 1103.02 examples/s]29672 examples [00:27, 1103.54 examples/s]29783 examples [00:27, 1099.64 examples/s]29893 examples [00:27, 1096.31 examples/s]30003 examples [00:27, 1031.41 examples/s]30114 examples [00:27, 1053.52 examples/s]30225 examples [00:27, 1067.53 examples/s]30336 examples [00:27, 1077.21 examples/s]30448 examples [00:27, 1086.92 examples/s]30562 examples [00:27, 1099.96 examples/s]30673 examples [00:28, 1102.25 examples/s]30785 examples [00:28, 1104.68 examples/s]30897 examples [00:28, 1107.00 examples/s]31008 examples [00:28, 1104.96 examples/s]31119 examples [00:28, 1103.73 examples/s]31230 examples [00:28, 1100.62 examples/s]31341 examples [00:28, 1096.73 examples/s]31451 examples [00:28, 1094.48 examples/s]31561 examples [00:28, 1087.90 examples/s]31670 examples [00:29, 1085.88 examples/s]31779 examples [00:29, 1085.63 examples/s]31888 examples [00:29, 1085.42 examples/s]31997 examples [00:29, 1081.09 examples/s]32106 examples [00:29, 1082.62 examples/s]32215 examples [00:29, 1076.55 examples/s]32324 examples [00:29, 1080.10 examples/s]32433 examples [00:29, 1082.22 examples/s]32542 examples [00:29, 1071.16 examples/s]32650 examples [00:29, 1065.18 examples/s]32759 examples [00:30, 1072.42 examples/s]32869 examples [00:30, 1080.06 examples/s]32980 examples [00:30, 1086.13 examples/s]33089 examples [00:30, 1074.53 examples/s]33197 examples [00:30, 1056.66 examples/s]33303 examples [00:30, 1053.86 examples/s]33409 examples [00:30, 1013.42 examples/s]33515 examples [00:30, 1025.31 examples/s]33623 examples [00:30, 1040.38 examples/s]33731 examples [00:30, 1050.71 examples/s]33841 examples [00:31, 1064.78 examples/s]33950 examples [00:31, 1069.75 examples/s]34060 examples [00:31, 1077.67 examples/s]34170 examples [00:31, 1083.52 examples/s]34279 examples [00:31, 1084.37 examples/s]34388 examples [00:31, 1072.93 examples/s]34498 examples [00:31, 1079.17 examples/s]34607 examples [00:31, 1081.03 examples/s]34717 examples [00:31, 1084.57 examples/s]34826 examples [00:31, 1079.15 examples/s]34934 examples [00:32, 1066.41 examples/s]35044 examples [00:32, 1075.07 examples/s]35154 examples [00:32, 1078.57 examples/s]35264 examples [00:32, 1083.60 examples/s]35373 examples [00:32, 1083.00 examples/s]35482 examples [00:32, 1064.86 examples/s]35592 examples [00:32, 1074.25 examples/s]35702 examples [00:32, 1081.47 examples/s]35813 examples [00:32, 1087.26 examples/s]35925 examples [00:32, 1094.93 examples/s]36036 examples [00:33, 1098.88 examples/s]36146 examples [00:33, 1093.17 examples/s]36256 examples [00:33, 1087.08 examples/s]36365 examples [00:33, 1064.91 examples/s]36472 examples [00:33, 1056.13 examples/s]36578 examples [00:33, 1043.20 examples/s]36689 examples [00:33, 1059.79 examples/s]36796 examples [00:33, 1055.69 examples/s]36902 examples [00:33, 1056.54 examples/s]37012 examples [00:33, 1067.65 examples/s]37119 examples [00:34, 1064.85 examples/s]37226 examples [00:34, 1049.13 examples/s]37332 examples [00:34, 1023.66 examples/s]37442 examples [00:34, 1045.38 examples/s]37551 examples [00:34, 1056.33 examples/s]37658 examples [00:34, 1059.14 examples/s]37768 examples [00:34, 1069.81 examples/s]37878 examples [00:34, 1076.70 examples/s]37988 examples [00:34, 1082.63 examples/s]38097 examples [00:35, 1082.48 examples/s]38207 examples [00:35, 1083.16 examples/s]38316 examples [00:35, 1079.01 examples/s]38424 examples [00:35, 1064.91 examples/s]38534 examples [00:35, 1074.46 examples/s]38643 examples [00:35, 1076.53 examples/s]38751 examples [00:35, 1071.39 examples/s]38859 examples [00:35, 979.37 examples/s] 38970 examples [00:35, 1013.87 examples/s]39081 examples [00:35, 1039.47 examples/s]39189 examples [00:36, 1050.82 examples/s]39299 examples [00:36, 1065.04 examples/s]39411 examples [00:36, 1080.35 examples/s]39523 examples [00:36, 1090.97 examples/s]39635 examples [00:36, 1099.18 examples/s]39747 examples [00:36, 1103.03 examples/s]39858 examples [00:36, 1105.08 examples/s]39969 examples [00:36, 1103.60 examples/s]40080 examples [00:36, 1041.12 examples/s]40190 examples [00:36, 1057.75 examples/s]40300 examples [00:37, 1068.43 examples/s]40411 examples [00:37, 1079.33 examples/s]40520 examples [00:37, 1081.54 examples/s]40631 examples [00:37, 1087.26 examples/s]40741 examples [00:37, 1089.17 examples/s]40852 examples [00:37, 1093.25 examples/s]40962 examples [00:37, 1092.79 examples/s]41072 examples [00:37, 1090.55 examples/s]41183 examples [00:37, 1093.73 examples/s]41293 examples [00:37, 1093.56 examples/s]41403 examples [00:38, 1092.62 examples/s]41513 examples [00:38, 1087.29 examples/s]41622 examples [00:38, 1087.60 examples/s]41732 examples [00:38, 1090.88 examples/s]41842 examples [00:38, 1093.48 examples/s]41952 examples [00:38, 1090.92 examples/s]42062 examples [00:38, 1090.18 examples/s]42172 examples [00:38, 1089.40 examples/s]42281 examples [00:38, 1083.26 examples/s]42390 examples [00:39, 1041.43 examples/s]42497 examples [00:39, 1048.71 examples/s]42609 examples [00:39, 1067.41 examples/s]42719 examples [00:39, 1074.55 examples/s]42829 examples [00:39, 1079.43 examples/s]42938 examples [00:39, 1081.07 examples/s]43047 examples [00:39, 1082.29 examples/s]43158 examples [00:39, 1088.29 examples/s]43267 examples [00:39, 1072.27 examples/s]43380 examples [00:39, 1086.89 examples/s]43492 examples [00:40, 1094.25 examples/s]43604 examples [00:40, 1099.18 examples/s]43714 examples [00:40, 1099.40 examples/s]43824 examples [00:40, 1095.82 examples/s]43934 examples [00:40, 1095.37 examples/s]44045 examples [00:40, 1097.41 examples/s]44155 examples [00:40, 1096.12 examples/s]44265 examples [00:40, 1093.90 examples/s]44375 examples [00:40, 1090.63 examples/s]44486 examples [00:40, 1095.74 examples/s]44596 examples [00:41, 1096.09 examples/s]44706 examples [00:41, 1093.99 examples/s]44816 examples [00:41, 1094.33 examples/s]44926 examples [00:41, 1091.76 examples/s]45036 examples [00:41, 1077.83 examples/s]45147 examples [00:41, 1084.61 examples/s]45257 examples [00:41, 1086.70 examples/s]45366 examples [00:41, 1074.94 examples/s]45474 examples [00:41, 1076.39 examples/s]45586 examples [00:41, 1089.03 examples/s]45698 examples [00:42, 1097.28 examples/s]45811 examples [00:42, 1105.88 examples/s]45922 examples [00:42, 1098.31 examples/s]46033 examples [00:42, 1099.50 examples/s]46144 examples [00:42, 1101.58 examples/s]46255 examples [00:42, 1103.38 examples/s]46367 examples [00:42, 1106.66 examples/s]46478 examples [00:42, 1105.47 examples/s]46589 examples [00:42, 1103.16 examples/s]46700 examples [00:42, 1098.58 examples/s]46810 examples [00:43, 1097.53 examples/s]46920 examples [00:43, 1098.24 examples/s]47031 examples [00:43, 1099.48 examples/s]47142 examples [00:43, 1100.39 examples/s]47253 examples [00:43, 1068.15 examples/s]47361 examples [00:43, 1051.37 examples/s]47470 examples [00:43, 1060.72 examples/s]47580 examples [00:43, 1070.03 examples/s]47689 examples [00:43, 1074.67 examples/s]47797 examples [00:43, 1076.05 examples/s]47907 examples [00:44, 1081.02 examples/s]48018 examples [00:44, 1086.97 examples/s]48128 examples [00:44, 1090.52 examples/s]48238 examples [00:44, 1061.91 examples/s]48346 examples [00:44, 1066.00 examples/s]48455 examples [00:44, 1072.60 examples/s]48565 examples [00:44, 1078.99 examples/s]48676 examples [00:44, 1086.84 examples/s]48787 examples [00:44, 1093.13 examples/s]48900 examples [00:44, 1102.61 examples/s]49011 examples [00:45, 1104.40 examples/s]49122 examples [00:45, 1104.96 examples/s]49233 examples [00:45, 1104.21 examples/s]49344 examples [00:45, 1094.79 examples/s]49454 examples [00:45, 1079.87 examples/s]49563 examples [00:45, 1064.36 examples/s]49670 examples [00:45, 1055.07 examples/s]49776 examples [00:45, 1035.56 examples/s]49885 examples [00:45, 1050.45 examples/s]49995 examples [00:46, 1062.71 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6102/50000 [00:00<00:00, 61015.40 examples/s] 38%|███▊      | 19214/50000 [00:00<00:00, 72671.65 examples/s] 65%|██████▍   | 32473/50000 [00:00<00:00, 84068.25 examples/s] 92%|█████████▏| 45794/50000 [00:00<00:00, 94528.49 examples/s]                                                               0 examples [00:00, ? examples/s]92 examples [00:00, 917.95 examples/s]204 examples [00:00, 968.71 examples/s]317 examples [00:00, 1010.10 examples/s]430 examples [00:00, 1041.39 examples/s]529 examples [00:00, 1024.26 examples/s]642 examples [00:00, 1051.99 examples/s]755 examples [00:00, 1071.86 examples/s]868 examples [00:00, 1086.83 examples/s]979 examples [00:00, 1093.54 examples/s]1087 examples [00:01, 1088.96 examples/s]1198 examples [00:01, 1094.58 examples/s]1311 examples [00:01, 1102.07 examples/s]1422 examples [00:01, 1102.02 examples/s]1535 examples [00:01, 1108.96 examples/s]1648 examples [00:01, 1114.84 examples/s]1760 examples [00:01, 1106.94 examples/s]1871 examples [00:01, 1100.94 examples/s]1981 examples [00:01, 1088.02 examples/s]2092 examples [00:01, 1094.30 examples/s]2203 examples [00:02, 1096.13 examples/s]2313 examples [00:02, 1095.75 examples/s]2424 examples [00:02, 1098.84 examples/s]2534 examples [00:02, 1065.68 examples/s]2646 examples [00:02, 1080.93 examples/s]2759 examples [00:02, 1092.86 examples/s]2873 examples [00:02, 1103.88 examples/s]2984 examples [00:02, 1102.16 examples/s]3097 examples [00:02, 1107.58 examples/s]3209 examples [00:02, 1110.51 examples/s]3321 examples [00:03, 1113.09 examples/s]3433 examples [00:03, 1092.61 examples/s]3543 examples [00:03, 1077.65 examples/s]3656 examples [00:03, 1091.78 examples/s]3770 examples [00:03, 1103.99 examples/s]3883 examples [00:03, 1109.07 examples/s]3996 examples [00:03, 1115.04 examples/s]4108 examples [00:03, 1115.75 examples/s]4220 examples [00:03, 1114.67 examples/s]4333 examples [00:03, 1117.15 examples/s]4447 examples [00:04, 1120.95 examples/s]4560 examples [00:04, 1122.69 examples/s]4674 examples [00:04, 1127.02 examples/s]4787 examples [00:04, 882.62 examples/s] 4898 examples [00:04, 939.93 examples/s]5009 examples [00:04, 983.16 examples/s]5121 examples [00:04, 1018.89 examples/s]5232 examples [00:04, 1042.15 examples/s]5344 examples [00:04, 1061.90 examples/s]5454 examples [00:05, 1071.77 examples/s]5564 examples [00:05, 1078.97 examples/s]5675 examples [00:05, 1086.75 examples/s]5785 examples [00:05, 1063.31 examples/s]5892 examples [00:05, 1063.48 examples/s]6005 examples [00:05, 1081.49 examples/s]6117 examples [00:05, 1091.10 examples/s]6229 examples [00:05, 1099.59 examples/s]6341 examples [00:05, 1103.92 examples/s]6453 examples [00:05, 1106.32 examples/s]6565 examples [00:06, 1108.80 examples/s]6677 examples [00:06, 1111.42 examples/s]6789 examples [00:06, 1113.80 examples/s]6901 examples [00:06, 1113.65 examples/s]7013 examples [00:06, 1114.52 examples/s]7125 examples [00:06, 1112.19 examples/s]7237 examples [00:06, 1106.76 examples/s]7349 examples [00:06, 1108.61 examples/s]7461 examples [00:06, 1110.90 examples/s]7574 examples [00:06, 1114.09 examples/s]7687 examples [00:07, 1116.80 examples/s]7799 examples [00:07, 1113.99 examples/s]7911 examples [00:07, 1114.58 examples/s]8023 examples [00:07, 1112.18 examples/s]8135 examples [00:07, 1110.37 examples/s]8248 examples [00:07, 1115.45 examples/s]8360 examples [00:07, 1116.74 examples/s]8473 examples [00:07, 1119.19 examples/s]8587 examples [00:07, 1123.03 examples/s]8700 examples [00:07, 1115.05 examples/s]8812 examples [00:08, 1103.92 examples/s]8923 examples [00:08, 1101.94 examples/s]9035 examples [00:08, 1105.27 examples/s]9147 examples [00:08, 1108.08 examples/s]9258 examples [00:08, 1106.14 examples/s]9370 examples [00:08, 1108.48 examples/s]9481 examples [00:08, 1107.71 examples/s]9592 examples [00:08, 1105.20 examples/s]9703 examples [00:08, 1104.10 examples/s]9814 examples [00:08, 1098.30 examples/s]9924 examples [00:09, 1096.40 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLQ7LLA/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLQ7LLA/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f499212a400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f499212a400> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f499212a400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f499018f240>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f49183f32e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f499212a400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f499212a400> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f499212a400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f49183f3198>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f497936b198>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.01 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.99 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.99 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 16.99 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00, 10.06 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00, 10.06 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.88 file/s]2020-08-31 18:08:07.426124: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-31 18:08:07.430326: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-08-31 18:08:07.431152: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5567e46920f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-31 18:08:07.431171: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:01, 161537.05it/s] 54%|█████▎    | 5316608/9912422 [00:00<00:19, 230466.17it/s]9920512it [00:00, 39823023.06it/s]                           
0it [00:00, ?it/s]32768it [00:00, 554012.85it/s]
0it [00:00, ?it/s]  5%|▌         | 90112/1648877 [00:00<00:01, 892460.74it/s]1654784it [00:00, 11523834.36it/s]                         
0it [00:00, ?it/s]8192it [00:00, 175665.98it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

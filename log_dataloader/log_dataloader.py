
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6ca6da91408244e26c157e9e6467cc18ede43e71', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f96f5dc1620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f96f5dc1620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f9760a73400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f9760a73400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f977fc8fea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f977fc8fea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f970ddaf0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f970ddaf0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f970ddaf0d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 11%|█         | 1097728/9912422 [00:00<00:00, 10745086.14it/s] 42%|████▏     | 4177920/9912422 [00:00<00:00, 13324909.70it/s] 97%|█████████▋| 9658368/9912422 [00:00<00:00, 17239211.56it/s]9920512it [00:00, 22794283.89it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1443459.05it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1033406.67it/s]1654784it [00:00, 12545581.18it/s]                           
0it [00:00, ?it/s]8192it [00:00, 253845.30it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f96ec7f66d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f970bf5c978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f970dda6d08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f970dda6d08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f970dda6d08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:08,  2.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:08,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 2/162 [00:00<00:52,  3.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:52,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:52,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<00:40,  3.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:40,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:39,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:30,  5.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:30,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:30,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:29,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:23,  6.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:23,  6.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:22,  6.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:22,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:01<00:17,  8.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:17,  8.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:17,  8.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:17,  8.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:13, 10.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:13, 10.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:13, 10.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:13, 10.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:01<00:11, 12.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:11, 12.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:11, 12.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:10, 12.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:01<00:09, 15.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:09, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:09, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:09, 15.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:01<00:07, 17.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:07, 17.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:07, 17.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:07, 17.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:01<00:06, 19.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:06, 19.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:06, 19.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:06, 19.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:01<00:06, 21.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:06, 21.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:06, 21.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:06, 21.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:01<00:05, 22.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:05, 22.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:05, 22.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:05, 22.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:05, 23.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:05, 23.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:02<00:05, 23.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:02<00:05, 23.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:02<00:04, 24.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:02<00:04, 24.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:02<00:04, 24.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:02<00:04, 24.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:02<00:04, 25.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:02<00:04, 25.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:02<00:04, 25.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:02<00:04, 25.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:02<00:04, 25.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:02<00:04, 25.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:02<00:04, 25.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:02<00:04, 25.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:02<00:04, 26.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:02<00:04, 26.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:02<00:04, 26.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:04, 26.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:02<00:04, 26.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:04, 26.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:04, 26.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:04, 26.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:02<00:04, 26.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:04, 26.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:04, 26.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:03, 26.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:02<00:03, 27.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:03, 27.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:03, 27.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:03, 27.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 27.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 27.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 27.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 27.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:02<00:03, 26.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:03, 26.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:03, 26.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:03<00:03, 26.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:03<00:03, 27.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:03<00:03, 27.66 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:03<00:03, 27.66 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:03<00:03, 27.66 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:03<00:03, 27.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:03<00:03, 27.68 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:03<00:03, 27.68 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:03<00:03, 27.68 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:03<00:03, 27.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:03<00:03, 27.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:03<00:03, 27.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:03<00:03, 27.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:03<00:03, 28.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:03<00:03, 28.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:03<00:03, 28.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:03<00:03, 28.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:03<00:02, 28.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:03<00:02, 28.49 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:03<00:02, 28.49 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:03<00:02, 28.49 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:03<00:02, 28.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:03<00:02, 28.25 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:03<00:02, 28.25 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:03<00:02, 28.25 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:03<00:02, 28.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:03<00:02, 28.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:03<00:02, 28.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:03<00:02, 28.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:03<00:02, 28.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:03<00:02, 28.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:03<00:02, 28.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:03<00:02, 28.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:03<00:02, 28.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:03<00:02, 28.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:03<00:02, 28.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:03<00:02, 28.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:04<00:02, 28.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:04<00:02, 28.43 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:04<00:02, 28.43 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:04<00:02, 28.43 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:04<00:02, 28.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:04<00:02, 28.78 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:04<00:02, 28.78 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:04<00:02, 28.78 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:04<00:02, 28.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:04<00:02, 28.86 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:04<00:02, 28.86 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:04<00:02, 28.86 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:04<00:02, 29.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:04<00:02, 29.05 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:04<00:02, 29.05 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:04<00:01, 29.05 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:04<00:01, 29.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:04<00:01, 29.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:04<00:01, 29.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:04<00:01, 29.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:04<00:01, 29.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:04<00:01, 29.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:04<00:01, 29.58 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:04<00:01, 29.58 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:04<00:01, 29.58 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:04<00:01, 29.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:04<00:01, 29.57 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:04<00:01, 29.57 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:04<00:01, 29.57 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:04<00:01, 29.57 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:04<00:01, 29.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:04<00:01, 29.60 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:04<00:01, 29.60 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:04<00:01, 29.60 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:04<00:01, 29.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:04<00:01, 29.70 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:04<00:01, 29.70 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:04<00:01, 29.70 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:04<00:01, 29.70 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:05<00:01, 29.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:05<00:01, 29.87 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:05<00:01, 29.87 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:05<00:01, 29.87 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:05<00:01, 29.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:05<00:01, 29.69 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:05<00:01, 29.69 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:05<00:01, 29.69 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:05<00:01, 29.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:05<00:01, 29.42 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:05<00:01, 29.42 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:05<00:01, 29.42 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:05<00:01, 29.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:05<00:01, 29.04 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:05<00:00, 29.04 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:05<00:00, 29.04 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:05<00:00, 29.04 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:05<00:00, 29.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:05<00:00, 29.30 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:05<00:00, 29.30 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:05<00:00, 29.30 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:05<00:00, 29.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:05<00:00, 29.35 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:05<00:00, 29.35 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:05<00:00, 29.35 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:05<00:00, 29.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:05<00:00, 29.39 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:05<00:00, 29.39 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:05<00:00, 29.39 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:05<00:00, 29.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:05<00:00, 29.53 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:05<00:00, 29.53 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:05<00:00, 29.53 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:05<00:00, 29.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:05<00:00, 29.51 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:05<00:00, 29.51 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:05<00:00, 29.51 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:05<00:00, 29.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:05<00:00, 29.42 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:06<00:00, 29.42 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:06<00:00, 29.42 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:06<00:00, 29.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:06<00:00, 29.23 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:06<00:00, 29.23 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:06<00:00, 29.23 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:06<00:00, 29.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:06<00:00, 29.44 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:06<00:00, 29.44 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:06<00:00, 29.44 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:06<00:00, 29.44 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:06<00:00, 29.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:06<00:00, 29.48 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 29.48 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.36s/ url]Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 29.48 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 29.48 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:06<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:08<00:00,  8.33s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:08<00:00,  6.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00, 29.48 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:08<00:00,  8.33s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:08<00:00,  8.33s/ file]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00, 19.45 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:08<00:00,  8.33s/ url]
0 examples [00:00, ? examples/s]2020-08-15 12:06:24.556255: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-15 12:06:24.569006: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-15 12:06:24.569269: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fb9a3ed660 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-15 12:06:24.569292: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.49 examples/s]107 examples [00:00,  4.98 examples/s]223 examples [00:00,  7.09 examples/s]331 examples [00:00, 10.11 examples/s]445 examples [00:00, 14.38 examples/s]558 examples [00:00, 20.43 examples/s]663 examples [00:00, 28.95 examples/s]771 examples [00:00, 40.89 examples/s]878 examples [00:01, 57.47 examples/s]995 examples [00:01, 80.40 examples/s]1108 examples [00:01, 111.46 examples/s]1221 examples [00:01, 152.72 examples/s]1334 examples [00:01, 206.20 examples/s]1445 examples [00:01, 272.07 examples/s]1557 examples [00:01, 352.00 examples/s]1668 examples [00:01, 440.77 examples/s]1781 examples [00:01, 539.48 examples/s]1896 examples [00:01, 641.55 examples/s]2008 examples [00:02, 727.49 examples/s]2119 examples [00:02, 811.21 examples/s]2230 examples [00:02, 876.63 examples/s]2340 examples [00:02, 914.83 examples/s]2448 examples [00:02, 951.95 examples/s]2555 examples [00:02, 979.64 examples/s]2662 examples [00:02, 985.99 examples/s]2778 examples [00:02, 1031.05 examples/s]2888 examples [00:02, 1050.32 examples/s]2997 examples [00:03, 1034.85 examples/s]3107 examples [00:03, 1051.50 examples/s]3217 examples [00:03, 1064.81 examples/s]3325 examples [00:03, 1040.85 examples/s]3434 examples [00:03, 1053.61 examples/s]3541 examples [00:03, 1057.51 examples/s]3648 examples [00:03, 1059.82 examples/s]3758 examples [00:03, 1070.58 examples/s]3873 examples [00:03, 1091.74 examples/s]3983 examples [00:03, 1092.64 examples/s]4093 examples [00:04, 1038.09 examples/s]4198 examples [00:04, 1040.87 examples/s]4307 examples [00:04, 1053.72 examples/s]4418 examples [00:04, 1069.99 examples/s]4530 examples [00:04, 1083.40 examples/s]4639 examples [00:04, 1082.79 examples/s]4748 examples [00:04, 1065.56 examples/s]4855 examples [00:04, 1064.37 examples/s]4963 examples [00:04, 1068.15 examples/s]5077 examples [00:04, 1086.28 examples/s]5186 examples [00:05, 1061.81 examples/s]5293 examples [00:05, 1045.49 examples/s]5407 examples [00:05, 1070.45 examples/s]5517 examples [00:05, 1078.66 examples/s]5626 examples [00:05, 1077.01 examples/s]5734 examples [00:05, 1061.92 examples/s]5843 examples [00:05, 1069.21 examples/s]5951 examples [00:05, 1047.40 examples/s]6064 examples [00:05, 1070.00 examples/s]6172 examples [00:06, 1068.78 examples/s]6280 examples [00:06, 1053.87 examples/s]6386 examples [00:06, 1044.89 examples/s]6491 examples [00:06, 1035.30 examples/s]6606 examples [00:06, 1065.74 examples/s]6716 examples [00:06, 1073.64 examples/s]6824 examples [00:06, 1064.13 examples/s]6931 examples [00:06, 1034.06 examples/s]7039 examples [00:06, 1046.69 examples/s]7147 examples [00:06, 1054.72 examples/s]7253 examples [00:07, 1025.52 examples/s]7356 examples [00:07, 993.31 examples/s] 7463 examples [00:07, 1013.63 examples/s]7577 examples [00:07, 1047.83 examples/s]7687 examples [00:07, 1061.59 examples/s]7797 examples [00:07, 1071.66 examples/s]7905 examples [00:07, 1036.30 examples/s]8011 examples [00:07, 1041.66 examples/s]8117 examples [00:07, 1044.29 examples/s]8224 examples [00:07, 1049.63 examples/s]8333 examples [00:08, 1060.74 examples/s]8440 examples [00:08, 1036.03 examples/s]8545 examples [00:08, 1038.26 examples/s]8651 examples [00:08, 1042.33 examples/s]8761 examples [00:08, 1057.07 examples/s]8869 examples [00:08, 1060.35 examples/s]8976 examples [00:08, 1047.21 examples/s]9081 examples [00:08, 1040.95 examples/s]9188 examples [00:08, 1048.13 examples/s]9298 examples [00:08, 1060.59 examples/s]9405 examples [00:09, 1031.71 examples/s]9509 examples [00:09, 1013.91 examples/s]9611 examples [00:09, 981.14 examples/s] 9710 examples [00:09, 980.76 examples/s]9816 examples [00:09, 1001.13 examples/s]9917 examples [00:09, 1001.60 examples/s]10018 examples [00:09, 936.22 examples/s]10117 examples [00:09, 949.51 examples/s]10223 examples [00:09, 979.58 examples/s]10334 examples [00:10, 1015.22 examples/s]10439 examples [00:10, 1023.79 examples/s]10543 examples [00:10, 1015.93 examples/s]10649 examples [00:10, 1027.21 examples/s]10753 examples [00:10, 1025.58 examples/s]10865 examples [00:10, 1050.67 examples/s]10972 examples [00:10, 1054.31 examples/s]11078 examples [00:10, 1030.92 examples/s]11183 examples [00:10, 1035.32 examples/s]11293 examples [00:10, 1051.56 examples/s]11399 examples [00:11, 1040.26 examples/s]11510 examples [00:11, 1058.99 examples/s]11622 examples [00:11, 1074.58 examples/s]11732 examples [00:11, 1081.39 examples/s]11841 examples [00:11, 1069.93 examples/s]11952 examples [00:11, 1080.34 examples/s]12061 examples [00:11, 1072.02 examples/s]12169 examples [00:11, 1062.50 examples/s]12276 examples [00:11, 1060.79 examples/s]12389 examples [00:11, 1079.71 examples/s]12499 examples [00:12, 1083.75 examples/s]12608 examples [00:12, 1074.04 examples/s]12716 examples [00:12, 1074.33 examples/s]12827 examples [00:12, 1083.71 examples/s]12936 examples [00:12, 1075.90 examples/s]13044 examples [00:12, 1040.55 examples/s]13149 examples [00:12, 1020.23 examples/s]13252 examples [00:12, 1000.23 examples/s]13353 examples [00:12, 998.65 examples/s] 13454 examples [00:13, 998.84 examples/s]13555 examples [00:13, 954.63 examples/s]13651 examples [00:13, 936.49 examples/s]13756 examples [00:13, 966.18 examples/s]13862 examples [00:13, 990.08 examples/s]13966 examples [00:13, 1003.41 examples/s]14070 examples [00:13, 1013.28 examples/s]14172 examples [00:13, 1002.01 examples/s]14278 examples [00:13, 1017.30 examples/s]14386 examples [00:13, 1034.68 examples/s]14490 examples [00:14, 1023.86 examples/s]14595 examples [00:14, 1030.63 examples/s]14699 examples [00:14, 1013.14 examples/s]14801 examples [00:14, 986.14 examples/s] 14900 examples [00:14, 979.07 examples/s]15004 examples [00:14, 994.39 examples/s]15108 examples [00:14, 1007.21 examples/s]15209 examples [00:14, 1007.45 examples/s]15318 examples [00:14, 1028.83 examples/s]15422 examples [00:14, 1002.27 examples/s]15532 examples [00:15, 1028.07 examples/s]15640 examples [00:15, 1042.29 examples/s]15746 examples [00:15, 1046.57 examples/s]15859 examples [00:15, 1067.87 examples/s]15967 examples [00:15, 1057.10 examples/s]16077 examples [00:15, 1063.20 examples/s]16184 examples [00:15, 1018.88 examples/s]16287 examples [00:15, 1007.63 examples/s]16393 examples [00:15, 1020.59 examples/s]16498 examples [00:16, 1026.88 examples/s]16607 examples [00:16, 1043.20 examples/s]16716 examples [00:16, 1056.68 examples/s]16822 examples [00:16, 1022.52 examples/s]16930 examples [00:16, 1038.46 examples/s]17048 examples [00:16, 1075.83 examples/s]17167 examples [00:16, 1106.23 examples/s]17282 examples [00:16, 1118.51 examples/s]17395 examples [00:16, 1106.69 examples/s]17507 examples [00:16, 1109.79 examples/s]17622 examples [00:17, 1120.03 examples/s]17737 examples [00:17, 1126.97 examples/s]17857 examples [00:17, 1145.74 examples/s]17972 examples [00:17, 1134.10 examples/s]18088 examples [00:17, 1141.26 examples/s]18207 examples [00:17, 1153.01 examples/s]18326 examples [00:17, 1159.31 examples/s]18443 examples [00:17, 1157.97 examples/s]18559 examples [00:17, 1146.15 examples/s]18674 examples [00:17, 1123.65 examples/s]18787 examples [00:18, 1117.86 examples/s]18901 examples [00:18, 1122.94 examples/s]19020 examples [00:18, 1140.36 examples/s]19135 examples [00:18, 1122.36 examples/s]19248 examples [00:18, 1119.67 examples/s]19361 examples [00:18, 1099.74 examples/s]19472 examples [00:18, 1102.54 examples/s]19583 examples [00:18, 1080.69 examples/s]19692 examples [00:18, 1065.54 examples/s]19801 examples [00:18, 1068.73 examples/s]19909 examples [00:19, 1071.31 examples/s]20017 examples [00:19, 1006.04 examples/s]20119 examples [00:19, 983.15 examples/s] 20221 examples [00:19, 993.00 examples/s]20321 examples [00:19, 965.72 examples/s]20423 examples [00:19, 979.77 examples/s]20530 examples [00:19, 1002.73 examples/s]20631 examples [00:19, 995.37 examples/s] 20731 examples [00:19, 975.42 examples/s]20829 examples [00:20, 942.87 examples/s]20924 examples [00:20, 926.66 examples/s]21021 examples [00:20, 937.72 examples/s]21116 examples [00:20, 926.42 examples/s]21220 examples [00:20, 955.69 examples/s]21319 examples [00:20, 963.63 examples/s]21422 examples [00:20, 980.01 examples/s]21531 examples [00:20, 1008.58 examples/s]21636 examples [00:20, 1018.61 examples/s]21743 examples [00:20, 1031.13 examples/s]21847 examples [00:21, 1030.54 examples/s]21956 examples [00:21, 1046.16 examples/s]22065 examples [00:21, 1057.03 examples/s]22171 examples [00:21, 1015.68 examples/s]22277 examples [00:21, 1025.37 examples/s]22388 examples [00:21, 1047.31 examples/s]22494 examples [00:21, 1036.26 examples/s]22598 examples [00:21, 1034.53 examples/s]22702 examples [00:21, 1028.11 examples/s]22805 examples [00:22, 1024.19 examples/s]22909 examples [00:22, 1028.62 examples/s]23015 examples [00:22, 1035.77 examples/s]23123 examples [00:22, 1048.43 examples/s]23228 examples [00:22, 1036.08 examples/s]23338 examples [00:22, 1053.19 examples/s]23447 examples [00:22, 1062.49 examples/s]23554 examples [00:22, 1049.95 examples/s]23660 examples [00:22, 1038.15 examples/s]23768 examples [00:22, 1049.84 examples/s]23878 examples [00:23, 1061.94 examples/s]23994 examples [00:23, 1089.26 examples/s]24106 examples [00:23, 1097.76 examples/s]24220 examples [00:23, 1107.44 examples/s]24331 examples [00:23, 1073.73 examples/s]24451 examples [00:23, 1108.69 examples/s]24563 examples [00:23, 1107.78 examples/s]24675 examples [00:23, 1101.78 examples/s]24789 examples [00:23, 1112.87 examples/s]24901 examples [00:23, 1096.92 examples/s]25018 examples [00:24, 1117.47 examples/s]25134 examples [00:24, 1129.63 examples/s]25250 examples [00:24, 1136.06 examples/s]25364 examples [00:24, 1097.16 examples/s]25475 examples [00:24, 1081.19 examples/s]25586 examples [00:24, 1086.95 examples/s]25695 examples [00:24, 1085.37 examples/s]25804 examples [00:24, 1046.48 examples/s]25914 examples [00:24, 1060.34 examples/s]26021 examples [00:24, 1047.96 examples/s]26127 examples [00:25, 989.29 examples/s] 26227 examples [00:25, 983.55 examples/s]26330 examples [00:25, 994.77 examples/s]26435 examples [00:25, 1008.97 examples/s]26551 examples [00:25, 1048.27 examples/s]26669 examples [00:25, 1083.82 examples/s]26788 examples [00:25, 1111.10 examples/s]26904 examples [00:25, 1125.19 examples/s]27023 examples [00:25, 1143.67 examples/s]27138 examples [00:26, 1140.53 examples/s]27258 examples [00:26, 1156.03 examples/s]27378 examples [00:26, 1166.34 examples/s]27495 examples [00:26, 1160.37 examples/s]27612 examples [00:26, 1152.25 examples/s]27728 examples [00:26, 1123.92 examples/s]27841 examples [00:26, 1115.85 examples/s]27953 examples [00:26, 1111.93 examples/s]28065 examples [00:26, 1073.23 examples/s]28173 examples [00:26, 1047.43 examples/s]28279 examples [00:27, 1038.45 examples/s]28384 examples [00:27, 1036.00 examples/s]28493 examples [00:27, 1049.18 examples/s]28599 examples [00:27, 971.02 examples/s] 28698 examples [00:27, 969.04 examples/s]28798 examples [00:27, 976.53 examples/s]28905 examples [00:27, 1001.10 examples/s]29008 examples [00:27, 1007.39 examples/s]29115 examples [00:27, 1024.92 examples/s]29218 examples [00:28, 1003.17 examples/s]29326 examples [00:28, 1023.44 examples/s]29435 examples [00:28, 1040.80 examples/s]29546 examples [00:28, 1059.50 examples/s]29653 examples [00:28, 1058.58 examples/s]29760 examples [00:28, 1031.30 examples/s]29864 examples [00:28, 1031.23 examples/s]29968 examples [00:28, 1029.95 examples/s]30072 examples [00:28, 973.76 examples/s] 30185 examples [00:28, 1015.86 examples/s]30292 examples [00:29, 1030.75 examples/s]30396 examples [00:29, 998.35 examples/s] 30497 examples [00:29, 978.00 examples/s]30596 examples [00:29, 969.47 examples/s]30705 examples [00:29, 1001.38 examples/s]30811 examples [00:29, 1017.75 examples/s]30925 examples [00:29, 1050.05 examples/s]31031 examples [00:29, 1048.82 examples/s]31137 examples [00:29, 1026.10 examples/s]31248 examples [00:29, 1048.30 examples/s]31354 examples [00:30, 1046.41 examples/s]31459 examples [00:30, 1037.56 examples/s]31564 examples [00:30, 1038.76 examples/s]31670 examples [00:30, 1044.10 examples/s]31779 examples [00:30, 1054.47 examples/s]31885 examples [00:30, 1027.03 examples/s]31992 examples [00:30, 1038.29 examples/s]32097 examples [00:30, 1029.98 examples/s]32205 examples [00:30, 1044.21 examples/s]32313 examples [00:30, 1052.08 examples/s]32419 examples [00:31, 1030.75 examples/s]32526 examples [00:31, 1039.74 examples/s]32631 examples [00:31, 1021.04 examples/s]32738 examples [00:31, 1032.89 examples/s]32844 examples [00:31, 1040.77 examples/s]32949 examples [00:31, 1020.15 examples/s]33052 examples [00:31, 1006.91 examples/s]33154 examples [00:31, 1010.27 examples/s]33258 examples [00:31, 1017.00 examples/s]33360 examples [00:32, 979.43 examples/s] 33463 examples [00:32, 993.17 examples/s]33569 examples [00:32, 1012.30 examples/s]33675 examples [00:32, 1024.16 examples/s]33778 examples [00:32, 1001.85 examples/s]33879 examples [00:32, 1003.07 examples/s]33981 examples [00:32, 1007.92 examples/s]34082 examples [00:32, 977.96 examples/s] 34185 examples [00:32, 992.39 examples/s]34296 examples [00:32, 1023.50 examples/s]34401 examples [00:33, 1029.42 examples/s]34510 examples [00:33, 1046.45 examples/s]34615 examples [00:33, 1044.01 examples/s]34722 examples [00:33, 1051.56 examples/s]34840 examples [00:33, 1084.83 examples/s]34950 examples [00:33, 1088.76 examples/s]35060 examples [00:33, 1043.35 examples/s]35165 examples [00:33, 1031.66 examples/s]35270 examples [00:33, 1033.73 examples/s]35374 examples [00:33, 1019.49 examples/s]35477 examples [00:34, 989.50 examples/s] 35580 examples [00:34, 1000.04 examples/s]35686 examples [00:34, 1015.48 examples/s]35798 examples [00:34, 1043.31 examples/s]35908 examples [00:34, 1059.07 examples/s]36015 examples [00:34, 1041.46 examples/s]36125 examples [00:34, 1056.01 examples/s]36231 examples [00:34, 1038.81 examples/s]36336 examples [00:34, 1025.35 examples/s]36439 examples [00:35, 1017.65 examples/s]36553 examples [00:35, 1051.16 examples/s]36674 examples [00:35, 1092.92 examples/s]36793 examples [00:35, 1118.33 examples/s]36913 examples [00:35, 1139.29 examples/s]37032 examples [00:35, 1151.61 examples/s]37148 examples [00:35, 1137.87 examples/s]37265 examples [00:35, 1144.71 examples/s]37381 examples [00:35, 1148.73 examples/s]37497 examples [00:35, 1149.38 examples/s]37615 examples [00:36, 1157.44 examples/s]37731 examples [00:36, 1142.31 examples/s]37846 examples [00:36, 1138.52 examples/s]37960 examples [00:36, 1124.39 examples/s]38073 examples [00:36, 1119.18 examples/s]38185 examples [00:36, 1114.78 examples/s]38297 examples [00:36, 1087.96 examples/s]38413 examples [00:36, 1108.10 examples/s]38526 examples [00:36, 1112.28 examples/s]38638 examples [00:36, 1083.05 examples/s]38747 examples [00:37, 1064.67 examples/s]38854 examples [00:37, 1038.94 examples/s]38959 examples [00:37, 1038.56 examples/s]39064 examples [00:37, 1032.72 examples/s]39168 examples [00:37, 1024.86 examples/s]39273 examples [00:37, 1030.45 examples/s]39377 examples [00:37, 1013.16 examples/s]39480 examples [00:37, 1015.33 examples/s]39583 examples [00:37, 1016.55 examples/s]39685 examples [00:38, 1006.65 examples/s]39787 examples [00:38, 1009.95 examples/s]39889 examples [00:38, 1001.01 examples/s]39993 examples [00:38, 1011.66 examples/s]40095 examples [00:38, 979.49 examples/s] 40211 examples [00:38, 1027.15 examples/s]40315 examples [00:38, 1020.23 examples/s]40418 examples [00:38, 998.75 examples/s] 40521 examples [00:38, 1006.62 examples/s]40624 examples [00:38, 1012.62 examples/s]40726 examples [00:39, 1004.93 examples/s]40827 examples [00:39, 1005.49 examples/s]40928 examples [00:39, 990.57 examples/s] 41028 examples [00:39, 973.91 examples/s]41129 examples [00:39, 983.12 examples/s]41243 examples [00:39, 1022.97 examples/s]41357 examples [00:39, 1052.86 examples/s]41465 examples [00:39, 1060.68 examples/s]41577 examples [00:39, 1076.37 examples/s]41685 examples [00:39, 1063.40 examples/s]41792 examples [00:40, 1052.59 examples/s]41898 examples [00:40, 1041.31 examples/s]42003 examples [00:40, 1023.19 examples/s]42108 examples [00:40, 1028.80 examples/s]42215 examples [00:40, 1039.32 examples/s]42320 examples [00:40, 1036.69 examples/s]42424 examples [00:40, 1035.84 examples/s]42528 examples [00:40, 1022.67 examples/s]42631 examples [00:40, 1021.43 examples/s]42734 examples [00:40, 1016.37 examples/s]42842 examples [00:41, 1033.61 examples/s]42962 examples [00:41, 1077.23 examples/s]43071 examples [00:41, 1057.37 examples/s]43190 examples [00:41, 1091.64 examples/s]43306 examples [00:41, 1109.46 examples/s]43425 examples [00:41, 1131.36 examples/s]43539 examples [00:41, 1110.11 examples/s]43651 examples [00:41, 1064.62 examples/s]43759 examples [00:41, 1055.24 examples/s]43866 examples [00:42, 1057.28 examples/s]43973 examples [00:42, 1052.46 examples/s]44079 examples [00:42, 1047.20 examples/s]44184 examples [00:42, 1022.18 examples/s]44291 examples [00:42, 1035.32 examples/s]44397 examples [00:42, 1041.61 examples/s]44502 examples [00:42, 1042.32 examples/s]44607 examples [00:42, 1014.65 examples/s]44709 examples [00:42, 1009.19 examples/s]44815 examples [00:42, 1021.44 examples/s]44932 examples [00:43, 1060.40 examples/s]45048 examples [00:43, 1087.32 examples/s]45163 examples [00:43, 1104.24 examples/s]45274 examples [00:43, 1082.01 examples/s]45390 examples [00:43, 1103.06 examples/s]45501 examples [00:43, 1097.43 examples/s]45612 examples [00:43, 1059.92 examples/s]45719 examples [00:43, 1033.68 examples/s]45825 examples [00:43, 1039.25 examples/s]45930 examples [00:43, 1037.12 examples/s]46038 examples [00:44, 1047.66 examples/s]46143 examples [00:44, 1038.46 examples/s]46247 examples [00:44, 1025.48 examples/s]46350 examples [00:44, 1021.12 examples/s]46455 examples [00:44, 1028.82 examples/s]46560 examples [00:44, 1032.73 examples/s]46666 examples [00:44, 1040.03 examples/s]46771 examples [00:44, 1028.57 examples/s]46880 examples [00:44, 1045.64 examples/s]46995 examples [00:44, 1073.92 examples/s]47111 examples [00:45, 1095.29 examples/s]47226 examples [00:45, 1108.85 examples/s]47338 examples [00:45, 1080.98 examples/s]47447 examples [00:45, 1057.67 examples/s]47554 examples [00:45, 1054.83 examples/s]47663 examples [00:45, 1063.78 examples/s]47774 examples [00:45, 1075.09 examples/s]47882 examples [00:45, 1064.22 examples/s]47991 examples [00:45, 1071.33 examples/s]48099 examples [00:46, 1073.78 examples/s]48207 examples [00:46, 1072.91 examples/s]48315 examples [00:46, 1068.59 examples/s]48422 examples [00:46, 1063.54 examples/s]48529 examples [00:46, 1060.82 examples/s]48636 examples [00:46, 1063.34 examples/s]48745 examples [00:46, 1071.13 examples/s]48853 examples [00:46, 1066.88 examples/s]48960 examples [00:46, 1052.14 examples/s]49066 examples [00:46, 1038.12 examples/s]49178 examples [00:47, 1060.60 examples/s]49289 examples [00:47, 1074.35 examples/s]49401 examples [00:47, 1085.24 examples/s]49510 examples [00:47, 1086.21 examples/s]49623 examples [00:47, 1097.84 examples/s]49739 examples [00:47, 1115.50 examples/s]49854 examples [00:47, 1122.18 examples/s]49968 examples [00:47, 1124.30 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 8792/50000 [00:00<00:00, 87916.52 examples/s] 48%|████▊     | 23783/50000 [00:00<00:00, 100366.79 examples/s] 77%|███████▋  | 38623/50000 [00:00<00:00, 111158.80 examples/s]                                                                0 examples [00:00, ? examples/s]95 examples [00:00, 949.69 examples/s]204 examples [00:00, 987.47 examples/s]311 examples [00:00, 1010.52 examples/s]426 examples [00:00, 1046.11 examples/s]543 examples [00:00, 1078.10 examples/s]658 examples [00:00, 1098.69 examples/s]769 examples [00:00, 1102.03 examples/s]885 examples [00:00, 1116.16 examples/s]1006 examples [00:00, 1140.89 examples/s]1127 examples [00:01, 1158.85 examples/s]1248 examples [00:01, 1172.29 examples/s]1364 examples [00:01, 1144.39 examples/s]1482 examples [00:01, 1152.41 examples/s]1599 examples [00:01, 1157.52 examples/s]1715 examples [00:01, 1150.32 examples/s]1830 examples [00:01, 1144.96 examples/s]1945 examples [00:01, 1128.36 examples/s]2065 examples [00:01, 1146.95 examples/s]2182 examples [00:01, 1152.85 examples/s]2301 examples [00:02, 1161.47 examples/s]2421 examples [00:02, 1171.77 examples/s]2539 examples [00:02, 1151.60 examples/s]2655 examples [00:02, 1132.99 examples/s]2769 examples [00:02, 1133.52 examples/s]2883 examples [00:02, 1114.25 examples/s]2999 examples [00:02, 1126.27 examples/s]3112 examples [00:02, 1117.95 examples/s]3224 examples [00:02, 1097.41 examples/s]3341 examples [00:02, 1117.49 examples/s]3461 examples [00:03, 1141.01 examples/s]3581 examples [00:03, 1156.31 examples/s]3697 examples [00:03, 1138.56 examples/s]3816 examples [00:03, 1151.05 examples/s]3936 examples [00:03, 1163.42 examples/s]4053 examples [00:03, 1162.25 examples/s]4174 examples [00:03, 1174.93 examples/s]4292 examples [00:03, 1155.76 examples/s]4410 examples [00:03, 1161.86 examples/s]4530 examples [00:03, 1172.55 examples/s]4650 examples [00:04, 1178.32 examples/s]4770 examples [00:04, 1182.73 examples/s]4889 examples [00:04, 1165.48 examples/s]5010 examples [00:04, 1176.46 examples/s]5129 examples [00:04, 1179.96 examples/s]5248 examples [00:04, 1152.21 examples/s]5369 examples [00:04, 1165.91 examples/s]5486 examples [00:04, 1138.08 examples/s]5606 examples [00:04, 1153.00 examples/s]5722 examples [00:04, 1143.92 examples/s]5837 examples [00:05, 1143.12 examples/s]5955 examples [00:05, 1152.81 examples/s]6071 examples [00:05, 1136.78 examples/s]6189 examples [00:05, 1148.58 examples/s]6307 examples [00:05, 1155.66 examples/s]6424 examples [00:05, 1159.88 examples/s]6541 examples [00:05, 1159.33 examples/s]6657 examples [00:05, 1135.10 examples/s]6778 examples [00:05, 1155.36 examples/s]6898 examples [00:06, 1166.79 examples/s]7018 examples [00:06, 1173.63 examples/s]7137 examples [00:06, 1176.68 examples/s]7255 examples [00:06, 1137.55 examples/s]7376 examples [00:06, 1157.80 examples/s]7494 examples [00:06, 1162.11 examples/s]7614 examples [00:06, 1172.53 examples/s]7732 examples [00:06, 1143.79 examples/s]7847 examples [00:06, 1107.40 examples/s]7959 examples [00:06, 1095.19 examples/s]8077 examples [00:07, 1117.64 examples/s]8198 examples [00:07, 1142.43 examples/s]8319 examples [00:07, 1159.72 examples/s]8436 examples [00:07, 1126.07 examples/s]8550 examples [00:07, 1128.32 examples/s]8668 examples [00:07, 1142.41 examples/s]8788 examples [00:07, 1156.16 examples/s]8905 examples [00:07, 1159.75 examples/s]9022 examples [00:07, 1137.90 examples/s]9141 examples [00:07, 1150.93 examples/s]9259 examples [00:08, 1158.46 examples/s]9377 examples [00:08, 1163.78 examples/s]9494 examples [00:08, 1158.55 examples/s]9610 examples [00:08, 1127.21 examples/s]9728 examples [00:08, 1141.74 examples/s]9843 examples [00:08, 1125.37 examples/s]9960 examples [00:08, 1135.96 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteC3DRK9/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteC3DRK9/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f970ddaf0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f970ddaf0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f970ddaf0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f975978d7b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f969803ce48>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f970ddaf0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f970ddaf0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f970ddaf0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f970bf5c358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f970bf5c1d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.98 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.98 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  5.98 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.54 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.54 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.18 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.18 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.76 file/s]2020-08-15 12:07:27.682110: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-15 12:07:27.685048: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-15 12:07:27.685231: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563224634c40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-15 12:07:27.685247: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  1%|          | 106496/9912422 [00:00<00:09, 1001335.18it/s] 95%|█████████▍| 9412608/9912422 [00:00<00:00, 1423780.15it/s]9920512it [00:00, 47177215.02it/s]                            
0it [00:00, ?it/s]32768it [00:00, 652132.84it/s]
0it [00:00, ?it/s]  5%|▍         | 81920/1648877 [00:00<00:01, 816518.18it/s] 81%|████████  | 1335296/1648877 [00:00<00:00, 1134736.36it/s]1654784it [00:00, 7691989.96it/s]                             
0it [00:00, ?it/s]8192it [00:00, 259634.26it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

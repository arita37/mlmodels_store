
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/13f78dac13e826a41e3a7922ab3568a9b02adef6', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '13f78dac13e826a41e3a7922ab3568a9b02adef6', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/13f78dac13e826a41e3a7922ab3568a9b02adef6

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/13f78dac13e826a41e3a7922ab3568a9b02adef6

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/13f78dac13e826a41e3a7922ab3568a9b02adef6

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fdff41991e0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fdff41991e0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe05eebf400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe05eebf400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe07e0dbea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe07e0dbea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe00c1fb0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe00c1fb0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe00c1fb0d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3629056/9912422 [00:00<00:00, 35927405.99it/s]9920512it [00:00, 35315652.30it/s]                             
0it [00:00, ?it/s]32768it [00:00, 670964.14it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 450476.75it/s]1654784it [00:00, 11710174.30it/s]                         
0it [00:00, ?it/s]8192it [00:00, 155105.47it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe00a3a7b38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe00a3bc1d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe00c1f3d08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe00c1f3d08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe00c1f3d08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:26,  1.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:26,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:25,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:25,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:24,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:24,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:23,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:22,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:58,  2.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:58,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:58,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:57,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:57,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:56,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:56,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:56,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:55,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:39,  3.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:39,  3.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:39,  3.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:38,  3.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:38,  3.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:38,  3.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:37,  3.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:37,  3.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:37,  3.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:26,  5.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:26,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:26,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:26,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:25,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:25,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:25,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:25,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:25,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:18,  7.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:18,  7.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:17,  7.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:17,  7.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:17,  7.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:17,  7.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:17,  7.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:17,  7.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:16,  7.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:12,  9.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:12,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:12,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:11,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:11,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:11,  9.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:08, 13.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:08, 13.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:08, 13.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:08, 13.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 13.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 13.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 13.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 13.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 13.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 17.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 17.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 17.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 17.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 17.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 17.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 17.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 17.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 17.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 23.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 23.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 23.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 23.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 23.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 23.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 23.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 23.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 23.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 23.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 29.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 29.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 29.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 29.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 29.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 29.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 29.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 29.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 29.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 29.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 36.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 36.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 36.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 36.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 36.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 36.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 36.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 36.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 36.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 42.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 42.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 42.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 42.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 42.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 42.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 42.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 42.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 42.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 48.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 48.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 48.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 48.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 48.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 48.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 48.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 48.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 48.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 52.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 52.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 52.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 52.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 52.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 52.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 52.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 54.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 54.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 59.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 59.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 59.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 59.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 59.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 59.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 59.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 59.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 59.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 61.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 61.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 61.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 61.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 61.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 61.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 61.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 61.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 62.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 62.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 62.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 62.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 62.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 62.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 62.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 62.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 64.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 64.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 64.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 64.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 64.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 64.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 64.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 64.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 65.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 65.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 65.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 65.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 65.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 65.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 65.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 65.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 65.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 67.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 67.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.74s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.74s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.74s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.06 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.19s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.74s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 67.06 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.19s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.19s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.18 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.20s/ url]
0 examples [00:00, ? examples/s]2020-08-24 06:07:42.874954: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-24 06:07:42.891993: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-24 06:07:42.892188: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563c57de8370 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-24 06:07:42.892208: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
7 examples [00:00, 69.84 examples/s]93 examples [00:00, 96.39 examples/s]183 examples [00:00, 131.61 examples/s]273 examples [00:00, 176.91 examples/s]366 examples [00:00, 233.58 examples/s]455 examples [00:00, 299.71 examples/s]548 examples [00:00, 376.20 examples/s]631 examples [00:00, 449.42 examples/s]714 examples [00:00, 520.72 examples/s]799 examples [00:01, 588.69 examples/s]891 examples [00:01, 658.43 examples/s]977 examples [00:01, 706.24 examples/s]1062 examples [00:01, 732.99 examples/s]1146 examples [00:01, 749.65 examples/s]1239 examples [00:01, 795.83 examples/s]1334 examples [00:01, 835.97 examples/s]1423 examples [00:01, 850.27 examples/s]1514 examples [00:01, 866.08 examples/s]1607 examples [00:01, 883.24 examples/s]1698 examples [00:02, 877.11 examples/s]1788 examples [00:02, 881.07 examples/s]1877 examples [00:02, 864.84 examples/s]1968 examples [00:02, 877.14 examples/s]2063 examples [00:02, 895.32 examples/s]2153 examples [00:02, 892.00 examples/s]2243 examples [00:02, 847.74 examples/s]2333 examples [00:02, 862.32 examples/s]2422 examples [00:02, 868.44 examples/s]2514 examples [00:02, 883.17 examples/s]2610 examples [00:03, 903.14 examples/s]2701 examples [00:03, 897.71 examples/s]2792 examples [00:03, 882.42 examples/s]2881 examples [00:03, 858.66 examples/s]2969 examples [00:03, 863.00 examples/s]3056 examples [00:03, 851.27 examples/s]3143 examples [00:03, 856.73 examples/s]3232 examples [00:03, 866.03 examples/s]3319 examples [00:03, 855.68 examples/s]3405 examples [00:03, 854.41 examples/s]3495 examples [00:04, 867.59 examples/s]3588 examples [00:04, 883.12 examples/s]3677 examples [00:04, 864.73 examples/s]3767 examples [00:04, 872.79 examples/s]3857 examples [00:04, 879.19 examples/s]3947 examples [00:04, 884.57 examples/s]4037 examples [00:04, 887.37 examples/s]4130 examples [00:04, 899.12 examples/s]4220 examples [00:04, 897.60 examples/s]4316 examples [00:05, 914.79 examples/s]4412 examples [00:05, 926.28 examples/s]4505 examples [00:05, 911.94 examples/s]4597 examples [00:05, 911.65 examples/s]4689 examples [00:05, 909.07 examples/s]4780 examples [00:05, 902.86 examples/s]4871 examples [00:05, 903.44 examples/s]4962 examples [00:05, 899.47 examples/s]5052 examples [00:05, 849.11 examples/s]5138 examples [00:05, 788.02 examples/s]5231 examples [00:06, 825.51 examples/s]5322 examples [00:06, 848.97 examples/s]5411 examples [00:06, 858.72 examples/s]5498 examples [00:06, 847.78 examples/s]5587 examples [00:06, 858.69 examples/s]5679 examples [00:06, 873.72 examples/s]5770 examples [00:06, 882.97 examples/s]5860 examples [00:06, 886.99 examples/s]5949 examples [00:06, 887.61 examples/s]6038 examples [00:06, 879.05 examples/s]6127 examples [00:07, 869.97 examples/s]6217 examples [00:07, 876.96 examples/s]6308 examples [00:07, 885.35 examples/s]6397 examples [00:07, 886.47 examples/s]6488 examples [00:07, 891.76 examples/s]6578 examples [00:07, 888.80 examples/s]6667 examples [00:07, 866.13 examples/s]6755 examples [00:07, 867.74 examples/s]6842 examples [00:07, 865.03 examples/s]6931 examples [00:07, 871.56 examples/s]7019 examples [00:08, 849.39 examples/s]7105 examples [00:08, 837.18 examples/s]7190 examples [00:08, 839.63 examples/s]7281 examples [00:08, 857.12 examples/s]7371 examples [00:08, 867.49 examples/s]7458 examples [00:08, 854.01 examples/s]7544 examples [00:08, 846.81 examples/s]7629 examples [00:08, 828.91 examples/s]7718 examples [00:08, 846.21 examples/s]7803 examples [00:09, 825.07 examples/s]7886 examples [00:09, 823.17 examples/s]7977 examples [00:09, 845.55 examples/s]8070 examples [00:09, 869.12 examples/s]8158 examples [00:09, 859.55 examples/s]8252 examples [00:09, 879.58 examples/s]8345 examples [00:09, 893.34 examples/s]8437 examples [00:09, 899.60 examples/s]8531 examples [00:09, 909.06 examples/s]8627 examples [00:09, 921.34 examples/s]8720 examples [00:10, 919.01 examples/s]8815 examples [00:10, 927.63 examples/s]8908 examples [00:10, 916.93 examples/s]9000 examples [00:10, 916.61 examples/s]9092 examples [00:10, 912.10 examples/s]9184 examples [00:10, 905.66 examples/s]9277 examples [00:10, 910.22 examples/s]9369 examples [00:10, 910.64 examples/s]9461 examples [00:10, 903.30 examples/s]9552 examples [00:10, 865.41 examples/s]9642 examples [00:11, 873.04 examples/s]9736 examples [00:11, 890.38 examples/s]9826 examples [00:11, 886.02 examples/s]9915 examples [00:11, 885.46 examples/s]10004 examples [00:11, 839.09 examples/s]10097 examples [00:11, 861.75 examples/s]10191 examples [00:11, 883.05 examples/s]10280 examples [00:11, 884.87 examples/s]10372 examples [00:11, 894.47 examples/s]10464 examples [00:12, 899.12 examples/s]10558 examples [00:12, 908.33 examples/s]10651 examples [00:12, 914.10 examples/s]10745 examples [00:12, 919.60 examples/s]10838 examples [00:12, 916.00 examples/s]10930 examples [00:12, 912.83 examples/s]11023 examples [00:12, 914.84 examples/s]11117 examples [00:12, 921.35 examples/s]11210 examples [00:12, 915.84 examples/s]11303 examples [00:12, 917.36 examples/s]11395 examples [00:13, 913.29 examples/s]11490 examples [00:13, 923.75 examples/s]11586 examples [00:13, 934.21 examples/s]11680 examples [00:13, 930.79 examples/s]11774 examples [00:13, 901.58 examples/s]11865 examples [00:13, 903.28 examples/s]11956 examples [00:13, 905.18 examples/s]12053 examples [00:13, 923.15 examples/s]12148 examples [00:13, 930.79 examples/s]12242 examples [00:13, 933.36 examples/s]12336 examples [00:14, 924.13 examples/s]12429 examples [00:14, 918.32 examples/s]12521 examples [00:14, 896.08 examples/s]12614 examples [00:14, 904.49 examples/s]12705 examples [00:14, 883.81 examples/s]12794 examples [00:14, 870.88 examples/s]12884 examples [00:14, 879.11 examples/s]12976 examples [00:14, 889.96 examples/s]13068 examples [00:14, 898.53 examples/s]13161 examples [00:14, 907.42 examples/s]13253 examples [00:15, 909.27 examples/s]13349 examples [00:15, 923.79 examples/s]13445 examples [00:15, 932.37 examples/s]13539 examples [00:15, 924.29 examples/s]13635 examples [00:15, 933.48 examples/s]13729 examples [00:15, 904.78 examples/s]13820 examples [00:15, 861.67 examples/s]13907 examples [00:15, 859.87 examples/s]13997 examples [00:15, 870.40 examples/s]14085 examples [00:16, 856.41 examples/s]14171 examples [00:16, 856.95 examples/s]14258 examples [00:16, 859.17 examples/s]14350 examples [00:16, 875.91 examples/s]14443 examples [00:16, 890.68 examples/s]14538 examples [00:16, 907.05 examples/s]14632 examples [00:16, 914.98 examples/s]14726 examples [00:16, 921.13 examples/s]14819 examples [00:16, 914.53 examples/s]14911 examples [00:16, 893.37 examples/s]15002 examples [00:17, 897.27 examples/s]15092 examples [00:17, 848.78 examples/s]15178 examples [00:17, 826.26 examples/s]15271 examples [00:17, 854.50 examples/s]15365 examples [00:17, 876.50 examples/s]15454 examples [00:17, 880.49 examples/s]15543 examples [00:17, 834.53 examples/s]15633 examples [00:17, 852.13 examples/s]15725 examples [00:17, 868.65 examples/s]15817 examples [00:17, 882.88 examples/s]15910 examples [00:18, 894.59 examples/s]16000 examples [00:18, 885.96 examples/s]16089 examples [00:18, 874.88 examples/s]16181 examples [00:18, 887.05 examples/s]16272 examples [00:18, 891.31 examples/s]16362 examples [00:18, 883.65 examples/s]16460 examples [00:18, 908.23 examples/s]16553 examples [00:18, 913.96 examples/s]16645 examples [00:18, 912.55 examples/s]16739 examples [00:18, 919.29 examples/s]16832 examples [00:19, 904.90 examples/s]16923 examples [00:19, 900.71 examples/s]17014 examples [00:19, 898.12 examples/s]17104 examples [00:19, 890.05 examples/s]17194 examples [00:19, 874.71 examples/s]17282 examples [00:19, 857.96 examples/s]17371 examples [00:19, 865.48 examples/s]17462 examples [00:19, 877.28 examples/s]17551 examples [00:19, 879.17 examples/s]17640 examples [00:20, 877.56 examples/s]17729 examples [00:20, 878.33 examples/s]17817 examples [00:20, 864.79 examples/s]17909 examples [00:20, 878.09 examples/s]18006 examples [00:20, 903.73 examples/s]18099 examples [00:20, 910.06 examples/s]18191 examples [00:20, 892.74 examples/s]18281 examples [00:20, 886.34 examples/s]18377 examples [00:20, 904.62 examples/s]18469 examples [00:20, 908.51 examples/s]18560 examples [00:21, 902.03 examples/s]18653 examples [00:21, 908.47 examples/s]18744 examples [00:21, 793.80 examples/s]18833 examples [00:21, 818.56 examples/s]18919 examples [00:21, 829.91 examples/s]19005 examples [00:21, 836.11 examples/s]19098 examples [00:21, 856.38 examples/s]19192 examples [00:21, 879.69 examples/s]19283 examples [00:21, 887.86 examples/s]19373 examples [00:22, 889.54 examples/s]19466 examples [00:22, 901.08 examples/s]19559 examples [00:22, 903.91 examples/s]19650 examples [00:22, 878.59 examples/s]19742 examples [00:22, 890.40 examples/s]19833 examples [00:22, 895.72 examples/s]19923 examples [00:22, 875.64 examples/s]20011 examples [00:22, 801.17 examples/s]20093 examples [00:22, 799.23 examples/s]20183 examples [00:22, 825.96 examples/s]20275 examples [00:23, 851.05 examples/s]20366 examples [00:23, 867.62 examples/s]20454 examples [00:23, 870.60 examples/s]20542 examples [00:23, 838.02 examples/s]20628 examples [00:23, 842.86 examples/s]20714 examples [00:23, 847.02 examples/s]20799 examples [00:23, 831.06 examples/s]20888 examples [00:23, 845.37 examples/s]20976 examples [00:23, 853.26 examples/s]21065 examples [00:23, 861.99 examples/s]21152 examples [00:24, 861.60 examples/s]21239 examples [00:24, 859.44 examples/s]21332 examples [00:24, 879.28 examples/s]21428 examples [00:24, 900.42 examples/s]21519 examples [00:24, 899.69 examples/s]21610 examples [00:24, 896.40 examples/s]21700 examples [00:24, 887.81 examples/s]21790 examples [00:24, 891.15 examples/s]21880 examples [00:24, 872.38 examples/s]21972 examples [00:25, 883.99 examples/s]22063 examples [00:25, 889.26 examples/s]22153 examples [00:25, 886.13 examples/s]22247 examples [00:25, 900.46 examples/s]22339 examples [00:25, 905.14 examples/s]22433 examples [00:25, 914.37 examples/s]22525 examples [00:25, 895.01 examples/s]22615 examples [00:25, 879.19 examples/s]22704 examples [00:25, 867.50 examples/s]22791 examples [00:25, 816.91 examples/s]22877 examples [00:26, 827.33 examples/s]22961 examples [00:26, 828.41 examples/s]23045 examples [00:26, 831.63 examples/s]23136 examples [00:26, 852.94 examples/s]23222 examples [00:26, 829.56 examples/s]23306 examples [00:26, 828.82 examples/s]23397 examples [00:26, 851.18 examples/s]23483 examples [00:26, 839.45 examples/s]23571 examples [00:26, 849.44 examples/s]23660 examples [00:26, 858.83 examples/s]23751 examples [00:27, 871.17 examples/s]23843 examples [00:27, 880.96 examples/s]23932 examples [00:27, 871.63 examples/s]24020 examples [00:27, 853.92 examples/s]24112 examples [00:27, 872.58 examples/s]24201 examples [00:27, 876.81 examples/s]24292 examples [00:27, 884.46 examples/s]24381 examples [00:27, 882.56 examples/s]24472 examples [00:27, 888.14 examples/s]24566 examples [00:27, 900.90 examples/s]24657 examples [00:28, 892.73 examples/s]24748 examples [00:28, 895.30 examples/s]24838 examples [00:28, 880.04 examples/s]24928 examples [00:28, 884.77 examples/s]25017 examples [00:28, 885.33 examples/s]25109 examples [00:28, 893.79 examples/s]25199 examples [00:28, 883.16 examples/s]25288 examples [00:28, 858.96 examples/s]25378 examples [00:28, 869.08 examples/s]25466 examples [00:29, 865.86 examples/s]25555 examples [00:29, 867.14 examples/s]25645 examples [00:29, 875.21 examples/s]25734 examples [00:29, 879.04 examples/s]25822 examples [00:29, 877.43 examples/s]25913 examples [00:29, 885.08 examples/s]26002 examples [00:29, 867.21 examples/s]26096 examples [00:29, 882.69 examples/s]26188 examples [00:29, 892.43 examples/s]26278 examples [00:29, 882.68 examples/s]26373 examples [00:30, 900.25 examples/s]26467 examples [00:30, 909.84 examples/s]26560 examples [00:30, 914.40 examples/s]26652 examples [00:30, 892.93 examples/s]26745 examples [00:30, 902.87 examples/s]26836 examples [00:30, 898.35 examples/s]26930 examples [00:30, 908.01 examples/s]27023 examples [00:30, 911.87 examples/s]27115 examples [00:30, 910.97 examples/s]27207 examples [00:30, 864.94 examples/s]27299 examples [00:31, 878.89 examples/s]27388 examples [00:31, 876.98 examples/s]27476 examples [00:31, 857.49 examples/s]27566 examples [00:31, 867.64 examples/s]27660 examples [00:31, 885.65 examples/s]27749 examples [00:31, 868.09 examples/s]27837 examples [00:31, 871.60 examples/s]27930 examples [00:31, 886.26 examples/s]28021 examples [00:31, 893.07 examples/s]28116 examples [00:32, 908.44 examples/s]28208 examples [00:32, 900.70 examples/s]28304 examples [00:32, 915.54 examples/s]28396 examples [00:32, 915.21 examples/s]28488 examples [00:32, 911.35 examples/s]28584 examples [00:32, 923.62 examples/s]28677 examples [00:32, 903.47 examples/s]28768 examples [00:32, 897.24 examples/s]28858 examples [00:32, 893.55 examples/s]28948 examples [00:32, 893.09 examples/s]29041 examples [00:33, 903.50 examples/s]29134 examples [00:33, 908.91 examples/s]29225 examples [00:33, 899.68 examples/s]29316 examples [00:33, 896.69 examples/s]29406 examples [00:33, 893.88 examples/s]29498 examples [00:33, 899.18 examples/s]29591 examples [00:33, 906.55 examples/s]29684 examples [00:33, 911.84 examples/s]29778 examples [00:33, 917.84 examples/s]29870 examples [00:33, 917.96 examples/s]29964 examples [00:34, 921.80 examples/s]30057 examples [00:34, 867.22 examples/s]30151 examples [00:34, 885.34 examples/s]30244 examples [00:34, 895.78 examples/s]30336 examples [00:34, 899.99 examples/s]30427 examples [00:34, 892.68 examples/s]30517 examples [00:34, 889.20 examples/s]30607 examples [00:34, 878.31 examples/s]30696 examples [00:34, 880.87 examples/s]30786 examples [00:34, 885.03 examples/s]30877 examples [00:35, 889.59 examples/s]30971 examples [00:35, 901.60 examples/s]31062 examples [00:35, 899.66 examples/s]31153 examples [00:35, 897.18 examples/s]31243 examples [00:35, 897.94 examples/s]31335 examples [00:35, 904.07 examples/s]31427 examples [00:35, 907.39 examples/s]31520 examples [00:35, 911.09 examples/s]31612 examples [00:35, 909.70 examples/s]31703 examples [00:35, 888.26 examples/s]31795 examples [00:36, 897.05 examples/s]31885 examples [00:36, 891.11 examples/s]31975 examples [00:36, 886.71 examples/s]32064 examples [00:36, 861.81 examples/s]32151 examples [00:36, 853.02 examples/s]32240 examples [00:36, 861.04 examples/s]32330 examples [00:36, 870.89 examples/s]32421 examples [00:36, 880.53 examples/s]32510 examples [00:36, 883.05 examples/s]32602 examples [00:37, 891.41 examples/s]32693 examples [00:37, 895.07 examples/s]32788 examples [00:37, 908.07 examples/s]32879 examples [00:37, 890.11 examples/s]32973 examples [00:37, 902.39 examples/s]33064 examples [00:37, 903.94 examples/s]33155 examples [00:37, 901.39 examples/s]33246 examples [00:37, 857.73 examples/s]33333 examples [00:37, 857.20 examples/s]33421 examples [00:37, 861.76 examples/s]33508 examples [00:38, 848.88 examples/s]33599 examples [00:38, 865.15 examples/s]33686 examples [00:38, 846.78 examples/s]33771 examples [00:38, 829.76 examples/s]33864 examples [00:38, 855.94 examples/s]33951 examples [00:38, 859.24 examples/s]34038 examples [00:38, 859.44 examples/s]34133 examples [00:38, 882.83 examples/s]34225 examples [00:38, 891.03 examples/s]34318 examples [00:38, 899.87 examples/s]34411 examples [00:39, 908.46 examples/s]34502 examples [00:39, 873.63 examples/s]34594 examples [00:39, 884.83 examples/s]34689 examples [00:39, 901.94 examples/s]34782 examples [00:39, 910.12 examples/s]34874 examples [00:39, 897.24 examples/s]34966 examples [00:39, 901.72 examples/s]35060 examples [00:39, 910.35 examples/s]35154 examples [00:39, 916.50 examples/s]35246 examples [00:39, 912.93 examples/s]35338 examples [00:40, 899.44 examples/s]35429 examples [00:40, 885.02 examples/s]35525 examples [00:40, 904.52 examples/s]35617 examples [00:40, 908.12 examples/s]35708 examples [00:40, 903.86 examples/s]35799 examples [00:40, 900.89 examples/s]35894 examples [00:40, 912.33 examples/s]35988 examples [00:40, 919.58 examples/s]36081 examples [00:40, 922.02 examples/s]36174 examples [00:41, 923.96 examples/s]36269 examples [00:41, 929.14 examples/s]36362 examples [00:41, 899.85 examples/s]36453 examples [00:41, 893.72 examples/s]36544 examples [00:41, 896.93 examples/s]36635 examples [00:41, 900.62 examples/s]36726 examples [00:41, 901.81 examples/s]36821 examples [00:41, 914.16 examples/s]36913 examples [00:41, 893.85 examples/s]37005 examples [00:41, 899.18 examples/s]37096 examples [00:42, 871.67 examples/s]37184 examples [00:42, 863.85 examples/s]37271 examples [00:42, 863.09 examples/s]37359 examples [00:42, 865.42 examples/s]37449 examples [00:42, 874.90 examples/s]37539 examples [00:42, 881.58 examples/s]37628 examples [00:42, 859.62 examples/s]37717 examples [00:42, 867.51 examples/s]37811 examples [00:42, 887.20 examples/s]37905 examples [00:42, 900.23 examples/s]37998 examples [00:43, 907.46 examples/s]38089 examples [00:43, 888.83 examples/s]38179 examples [00:43, 883.24 examples/s]38277 examples [00:43, 909.14 examples/s]38369 examples [00:43, 888.81 examples/s]38459 examples [00:43, 871.13 examples/s]38547 examples [00:43, 864.01 examples/s]38635 examples [00:43, 868.65 examples/s]38726 examples [00:43, 878.43 examples/s]38818 examples [00:44, 887.80 examples/s]38908 examples [00:44, 889.91 examples/s]38998 examples [00:44, 880.38 examples/s]39087 examples [00:44, 874.46 examples/s]39177 examples [00:44, 880.03 examples/s]39267 examples [00:44, 883.44 examples/s]39356 examples [00:44, 884.89 examples/s]39445 examples [00:44, 870.63 examples/s]39534 examples [00:44, 875.80 examples/s]39624 examples [00:44, 881.69 examples/s]39713 examples [00:45, 879.17 examples/s]39804 examples [00:45, 885.63 examples/s]39893 examples [00:45, 868.98 examples/s]39980 examples [00:45, 867.42 examples/s]40067 examples [00:45, 820.54 examples/s]40157 examples [00:45, 840.52 examples/s]40248 examples [00:45, 858.06 examples/s]40336 examples [00:45, 863.40 examples/s]40427 examples [00:45, 874.91 examples/s]40517 examples [00:45, 880.31 examples/s]40606 examples [00:46, 878.15 examples/s]40698 examples [00:46, 889.09 examples/s]40788 examples [00:46, 874.85 examples/s]40883 examples [00:46, 896.11 examples/s]40973 examples [00:46, 887.31 examples/s]41062 examples [00:46, 833.73 examples/s]41147 examples [00:46, 813.98 examples/s]41237 examples [00:46, 836.28 examples/s]41329 examples [00:46, 858.07 examples/s]41420 examples [00:47, 872.81 examples/s]41511 examples [00:47, 882.95 examples/s]41603 examples [00:47, 892.35 examples/s]41693 examples [00:47, 841.09 examples/s]41784 examples [00:47, 858.56 examples/s]41871 examples [00:47, 850.34 examples/s]41957 examples [00:47, 840.20 examples/s]42046 examples [00:47, 853.23 examples/s]42136 examples [00:47, 865.36 examples/s]42228 examples [00:47, 880.51 examples/s]42323 examples [00:48, 900.01 examples/s]42414 examples [00:48, 898.21 examples/s]42505 examples [00:48, 891.53 examples/s]42595 examples [00:48, 892.25 examples/s]42688 examples [00:48, 902.01 examples/s]42779 examples [00:48, 893.59 examples/s]42869 examples [00:48, 891.87 examples/s]42960 examples [00:48, 894.91 examples/s]43053 examples [00:48, 904.20 examples/s]43144 examples [00:48, 887.14 examples/s]43233 examples [00:49, 887.21 examples/s]43323 examples [00:49, 888.55 examples/s]43412 examples [00:49, 873.46 examples/s]43502 examples [00:49, 880.16 examples/s]43592 examples [00:49, 885.45 examples/s]43681 examples [00:49, 882.78 examples/s]43771 examples [00:49, 886.94 examples/s]43863 examples [00:49, 895.31 examples/s]43958 examples [00:49, 910.32 examples/s]44052 examples [00:49, 916.77 examples/s]44148 examples [00:50, 928.54 examples/s]44241 examples [00:50, 916.41 examples/s]44333 examples [00:50, 895.75 examples/s]44426 examples [00:50, 904.75 examples/s]44521 examples [00:50, 915.22 examples/s]44615 examples [00:50, 922.29 examples/s]44710 examples [00:50, 929.91 examples/s]44804 examples [00:50, 923.25 examples/s]44897 examples [00:50, 924.97 examples/s]44990 examples [00:50, 906.01 examples/s]45081 examples [00:51, 873.21 examples/s]45169 examples [00:51, 874.07 examples/s]45257 examples [00:51, 847.84 examples/s]45347 examples [00:51, 860.12 examples/s]45441 examples [00:51, 881.74 examples/s]45536 examples [00:51, 898.75 examples/s]45627 examples [00:51, 895.00 examples/s]45717 examples [00:51, 891.46 examples/s]45812 examples [00:51, 907.73 examples/s]45906 examples [00:52, 915.33 examples/s]46001 examples [00:52, 923.86 examples/s]46094 examples [00:52, 920.97 examples/s]46187 examples [00:52, 894.90 examples/s]46277 examples [00:52, 889.26 examples/s]46368 examples [00:52, 894.75 examples/s]46458 examples [00:52, 887.92 examples/s]46553 examples [00:52, 905.34 examples/s]46644 examples [00:52, 905.57 examples/s]46735 examples [00:52, 889.61 examples/s]46828 examples [00:53, 899.39 examples/s]46926 examples [00:53, 922.00 examples/s]47019 examples [00:53, 919.99 examples/s]47112 examples [00:53, 898.72 examples/s]47206 examples [00:53, 908.40 examples/s]47299 examples [00:53, 913.00 examples/s]47391 examples [00:53, 913.73 examples/s]47484 examples [00:53, 918.31 examples/s]47576 examples [00:53, 907.36 examples/s]47669 examples [00:53, 912.04 examples/s]47761 examples [00:54, 907.30 examples/s]47854 examples [00:54, 912.49 examples/s]47947 examples [00:54, 916.03 examples/s]48041 examples [00:54, 922.88 examples/s]48134 examples [00:54, 915.70 examples/s]48228 examples [00:54, 922.50 examples/s]48321 examples [00:54, 906.09 examples/s]48412 examples [00:54, 902.77 examples/s]48503 examples [00:54, 897.07 examples/s]48598 examples [00:54, 910.65 examples/s]48693 examples [00:55, 919.07 examples/s]48787 examples [00:55, 922.99 examples/s]48882 examples [00:55, 927.50 examples/s]48975 examples [00:55, 923.27 examples/s]49072 examples [00:55, 936.65 examples/s]49167 examples [00:55, 939.64 examples/s]49262 examples [00:55, 924.79 examples/s]49356 examples [00:55, 927.37 examples/s]49449 examples [00:55, 905.01 examples/s]49544 examples [00:56, 915.50 examples/s]49636 examples [00:56, 913.19 examples/s]49728 examples [00:56, 913.81 examples/s]49820 examples [00:56, 894.59 examples/s]49910 examples [00:56, 871.92 examples/s]50000 examples [00:56, 879.54 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5536/50000 [00:00<00:00, 55358.07 examples/s] 32%|███▏      | 15785/50000 [00:00<00:00, 64214.99 examples/s] 52%|█████▏    | 26247/50000 [00:00<00:00, 72629.14 examples/s] 73%|███████▎  | 36260/50000 [00:00<00:00, 79150.38 examples/s] 94%|█████████▎| 46784/50000 [00:00<00:00, 85509.23 examples/s]                                                               0 examples [00:00, ? examples/s]72 examples [00:00, 715.67 examples/s]162 examples [00:00, 760.61 examples/s]255 examples [00:00, 802.68 examples/s]335 examples [00:00, 801.52 examples/s]426 examples [00:00, 830.92 examples/s]498 examples [00:00, 758.42 examples/s]591 examples [00:00, 801.77 examples/s]682 examples [00:00, 830.85 examples/s]780 examples [00:00, 869.18 examples/s]876 examples [00:01, 893.22 examples/s]967 examples [00:01, 896.92 examples/s]1057 examples [00:01, 888.98 examples/s]1147 examples [00:01, 891.22 examples/s]1240 examples [00:01, 899.37 examples/s]1330 examples [00:01, 889.78 examples/s]1422 examples [00:01, 898.60 examples/s]1521 examples [00:01, 922.95 examples/s]1614 examples [00:01, 900.77 examples/s]1705 examples [00:01, 901.37 examples/s]1803 examples [00:02, 922.25 examples/s]1897 examples [00:02, 926.78 examples/s]1991 examples [00:02, 930.40 examples/s]2085 examples [00:02, 922.48 examples/s]2178 examples [00:02, 909.11 examples/s]2270 examples [00:02, 906.68 examples/s]2362 examples [00:02, 908.80 examples/s]2453 examples [00:02, 908.22 examples/s]2545 examples [00:02, 911.58 examples/s]2641 examples [00:02, 924.34 examples/s]2736 examples [00:03, 929.11 examples/s]2829 examples [00:03, 920.27 examples/s]2922 examples [00:03, 918.47 examples/s]3014 examples [00:03, 918.66 examples/s]3110 examples [00:03, 927.96 examples/s]3203 examples [00:03, 925.40 examples/s]3298 examples [00:03, 931.52 examples/s]3392 examples [00:03, 909.11 examples/s]3484 examples [00:03, 904.13 examples/s]3576 examples [00:03, 907.62 examples/s]3671 examples [00:04, 917.30 examples/s]3765 examples [00:04, 922.16 examples/s]3858 examples [00:04, 909.96 examples/s]3950 examples [00:04, 901.28 examples/s]4046 examples [00:04, 918.12 examples/s]4138 examples [00:04, 911.73 examples/s]4230 examples [00:04, 913.05 examples/s]4322 examples [00:04, 909.49 examples/s]4414 examples [00:04, 907.99 examples/s]4505 examples [00:04, 901.78 examples/s]4596 examples [00:05, 891.54 examples/s]4688 examples [00:05, 899.20 examples/s]4778 examples [00:05, 716.92 examples/s]4871 examples [00:05, 769.74 examples/s]4954 examples [00:05, 770.05 examples/s]5047 examples [00:05, 811.47 examples/s]5141 examples [00:05, 844.81 examples/s]5233 examples [00:05, 864.56 examples/s]5328 examples [00:05, 886.07 examples/s]5419 examples [00:06, 891.04 examples/s]5510 examples [00:06, 884.40 examples/s]5600 examples [00:06, 885.84 examples/s]5691 examples [00:06, 892.93 examples/s]5786 examples [00:06, 907.77 examples/s]5878 examples [00:06, 907.00 examples/s]5969 examples [00:06, 907.14 examples/s]6060 examples [00:06, 902.92 examples/s]6151 examples [00:06, 902.89 examples/s]6242 examples [00:07, 899.91 examples/s]6335 examples [00:07, 906.74 examples/s]6426 examples [00:07, 884.69 examples/s]6515 examples [00:07, 881.53 examples/s]6610 examples [00:07, 899.30 examples/s]6701 examples [00:07, 901.46 examples/s]6792 examples [00:07, 903.47 examples/s]6885 examples [00:07, 909.28 examples/s]6977 examples [00:07, 911.76 examples/s]7069 examples [00:07, 892.58 examples/s]7159 examples [00:08, 894.38 examples/s]7250 examples [00:08, 897.34 examples/s]7340 examples [00:08, 897.09 examples/s]7430 examples [00:08, 875.68 examples/s]7518 examples [00:08, 875.79 examples/s]7608 examples [00:08, 881.31 examples/s]7697 examples [00:08, 882.01 examples/s]7786 examples [00:08, 880.01 examples/s]7877 examples [00:08, 886.78 examples/s]7967 examples [00:08, 888.84 examples/s]8056 examples [00:09, 887.48 examples/s]8145 examples [00:09, 883.80 examples/s]8235 examples [00:09, 884.89 examples/s]8327 examples [00:09, 892.96 examples/s]8417 examples [00:09, 879.68 examples/s]8506 examples [00:09, 871.91 examples/s]8599 examples [00:09, 887.60 examples/s]8688 examples [00:09, 873.72 examples/s]8781 examples [00:09, 889.50 examples/s]8877 examples [00:09, 908.21 examples/s]8973 examples [00:10, 921.09 examples/s]9066 examples [00:10, 923.37 examples/s]9159 examples [00:10, 916.83 examples/s]9251 examples [00:10, 914.58 examples/s]9343 examples [00:10, 915.70 examples/s]9435 examples [00:10, 891.90 examples/s]9525 examples [00:10, 842.60 examples/s]9610 examples [00:10, 844.31 examples/s]9705 examples [00:10, 872.98 examples/s]9799 examples [00:11, 891.44 examples/s]9889 examples [00:11, 892.52 examples/s]9979 examples [00:11, 886.73 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 94%|█████████▍| 9398/10000 [00:00<00:00, 93972.47 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKNVJDO/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKNVJDO/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe00c1fb0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe00c1fb0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe00c1fb0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe057bd9eb8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe00a64cda0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe00c1fb0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe00c1fb0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe00c1fb0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe00a3cf0b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe00a3cf128>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.60 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.60 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  5.60 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.60 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.69 file/s]2020-08-24 06:08:59.226722: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-24 06:08:59.231255: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-24 06:08:59.231432: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558ea5072ff0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-24 06:08:59.231450: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 486839.30it/s] 84%|████████▎ | 8290304/9912422 [00:00<00:02, 693717.48it/s]9920512it [00:00, 45819391.39it/s]                           
0it [00:00, ?it/s]32768it [00:00, 630326.69it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 157290.60it/s]1654784it [00:00, 10569136.14it/s]                         
0it [00:00, ?it/s]8192it [00:00, 202802.06it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

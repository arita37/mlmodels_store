
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'ff21906d63b46d89670bd3ed148b01355ca5ff0e', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/ff21906d63b46d89670bd3ed148b01355ca5ff0e

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f329e6dc6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f329e6dc6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f3309405488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f3309405488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f3328624e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f3328624e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f32b673c510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f32b673c510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f32b673c510> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 33%|███▎      | 3309568/9912422 [00:00<00:00, 32847920.94it/s]9920512it [00:00, 36050791.39it/s]                             
0it [00:00, ?it/s]32768it [00:00, 592565.09it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 470463.30it/s]1654784it [00:00, 11514160.97it/s]                         
0it [00:00, ?it/s]8192it [00:00, 192732.31it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f329e60ff98>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f329d957cf8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f32b673c1e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f32b673c1e0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f32b673c1e0> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:20,  2.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:20,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:19,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:19,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:18,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:17,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:17,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:54,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:53,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:53,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:52,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:52,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:51,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  3.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:36,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:35,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:34,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:34,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:34,  3.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:23,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:22,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:22,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:22,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:14,  7.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 14.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 14.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 14.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 14.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 14.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 19.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 19.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 25.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 25.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 32.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 40.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 40.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 40.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 40.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 40.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 40.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 40.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 40.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 40.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 40.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 40.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 48.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 48.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 57.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 57.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 57.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 57.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 57.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 57.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 57.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 57.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 57.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 57.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 57.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 64.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 71.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 71.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 71.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 77.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 77.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 77.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 77.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 77.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 77.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 77.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 77.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 77.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 81.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 81.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 81.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 81.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 81.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 81.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.60 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 81.60 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.75 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ url]
0 examples [00:00, ? examples/s]2020-09-05 12:08:23.132277: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-09-05 12:08:23.147434: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-09-05 12:08:23.147697: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55703431bdb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-05 12:08:23.147720: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.58 examples/s]96 examples [00:00,  5.10 examples/s]199 examples [00:00,  7.28 examples/s]299 examples [00:00, 10.36 examples/s]400 examples [00:00, 14.74 examples/s]500 examples [00:00, 20.92 examples/s]597 examples [00:00, 29.62 examples/s]702 examples [00:00, 41.80 examples/s]806 examples [00:01, 58.70 examples/s]905 examples [00:01, 81.77 examples/s]1009 examples [00:01, 112.98 examples/s]1112 examples [00:01, 154.14 examples/s]1217 examples [00:01, 207.06 examples/s]1322 examples [00:01, 272.65 examples/s]1427 examples [00:01, 350.39 examples/s]1531 examples [00:01, 437.25 examples/s]1635 examples [00:01, 529.26 examples/s]1739 examples [00:01, 618.11 examples/s]1843 examples [00:02, 703.16 examples/s]1947 examples [00:02, 773.01 examples/s]2050 examples [00:02, 792.40 examples/s]2151 examples [00:02, 846.29 examples/s]2254 examples [00:02, 893.18 examples/s]2355 examples [00:02, 923.62 examples/s]2458 examples [00:02, 952.08 examples/s]2563 examples [00:02, 978.94 examples/s]2666 examples [00:02, 991.74 examples/s]2771 examples [00:03, 1007.00 examples/s]2874 examples [00:03, 986.25 examples/s] 2975 examples [00:03, 982.87 examples/s]3076 examples [00:03, 989.93 examples/s]3178 examples [00:03, 996.22 examples/s]3281 examples [00:03, 1005.09 examples/s]3384 examples [00:03, 1011.47 examples/s]3487 examples [00:03, 1015.50 examples/s]3589 examples [00:03, 997.83 examples/s] 3689 examples [00:03, 996.29 examples/s]3792 examples [00:04, 1003.95 examples/s]3897 examples [00:04, 1014.71 examples/s]4001 examples [00:04, 1020.52 examples/s]4106 examples [00:04, 1027.70 examples/s]4209 examples [00:04, 1025.98 examples/s]4314 examples [00:04, 1030.25 examples/s]4419 examples [00:04, 1033.78 examples/s]4525 examples [00:04, 1039.14 examples/s]4630 examples [00:04, 1041.27 examples/s]4735 examples [00:04, 1036.12 examples/s]4839 examples [00:05, 1034.69 examples/s]4943 examples [00:05, 1013.43 examples/s]5045 examples [00:05, 1003.98 examples/s]5146 examples [00:05, 1003.56 examples/s]5248 examples [00:05, 1007.52 examples/s]5351 examples [00:05, 1012.47 examples/s]5453 examples [00:05, 1012.68 examples/s]5555 examples [00:05, 1007.06 examples/s]5656 examples [00:05, 996.03 examples/s] 5756 examples [00:05, 988.72 examples/s]5857 examples [00:06, 991.90 examples/s]5957 examples [00:06, 957.30 examples/s]6057 examples [00:06, 967.86 examples/s]6156 examples [00:06, 973.31 examples/s]6257 examples [00:06, 981.32 examples/s]6359 examples [00:06, 992.39 examples/s]6461 examples [00:06, 998.83 examples/s]6564 examples [00:06, 1005.33 examples/s]6665 examples [00:06, 993.85 examples/s] 6766 examples [00:06, 996.58 examples/s]6869 examples [00:07, 1005.10 examples/s]6970 examples [00:07, 1002.19 examples/s]7071 examples [00:07, 973.94 examples/s] 7174 examples [00:07, 988.91 examples/s]7274 examples [00:07, 992.00 examples/s]7376 examples [00:07, 1000.06 examples/s]7478 examples [00:07, 1003.37 examples/s]7579 examples [00:07, 1005.11 examples/s]7682 examples [00:07, 1010.54 examples/s]7784 examples [00:08, 1000.08 examples/s]7885 examples [00:08, 958.27 examples/s] 7985 examples [00:08, 969.07 examples/s]8090 examples [00:08, 990.58 examples/s]8190 examples [00:08, 987.05 examples/s]8289 examples [00:08, 975.57 examples/s]8391 examples [00:08, 987.66 examples/s]8490 examples [00:08, 987.84 examples/s]8592 examples [00:08, 994.76 examples/s]8694 examples [00:08, 1000.85 examples/s]8795 examples [00:09, 997.45 examples/s] 8896 examples [00:09, 999.01 examples/s]8996 examples [00:09, 987.00 examples/s]9098 examples [00:09, 994.65 examples/s]9200 examples [00:09, 999.21 examples/s]9300 examples [00:09, 994.49 examples/s]9401 examples [00:09, 998.78 examples/s]9502 examples [00:09, 1000.64 examples/s]9603 examples [00:09, 988.94 examples/s] 9706 examples [00:09, 1000.69 examples/s]9807 examples [00:10, 1001.09 examples/s]9908 examples [00:10, 1003.42 examples/s]10009 examples [00:10, 949.50 examples/s]10113 examples [00:10, 972.46 examples/s]10214 examples [00:10, 981.32 examples/s]10315 examples [00:10, 988.98 examples/s]10418 examples [00:10, 999.18 examples/s]10519 examples [00:10, 992.90 examples/s]10622 examples [00:10, 1001.90 examples/s]10723 examples [00:10, 997.60 examples/s] 10823 examples [00:11, 989.81 examples/s]10923 examples [00:11, 976.99 examples/s]11023 examples [00:11, 982.70 examples/s]11124 examples [00:11, 989.58 examples/s]11227 examples [00:11, 999.95 examples/s]11328 examples [00:11, 997.78 examples/s]11428 examples [00:11, 997.55 examples/s]11530 examples [00:11, 1002.94 examples/s]11633 examples [00:11, 1008.11 examples/s]11735 examples [00:11, 1011.21 examples/s]11837 examples [00:12, 1012.04 examples/s]11939 examples [00:12, 1006.24 examples/s]12041 examples [00:12, 1010.29 examples/s]12144 examples [00:12, 1014.66 examples/s]12247 examples [00:12, 1016.55 examples/s]12349 examples [00:12, 1016.66 examples/s]12451 examples [00:12, 1012.08 examples/s]12553 examples [00:12, 1008.04 examples/s]12657 examples [00:12, 1015.17 examples/s]12761 examples [00:13, 1019.90 examples/s]12864 examples [00:13, 1020.99 examples/s]12967 examples [00:13, 1015.22 examples/s]13069 examples [00:13, 1016.12 examples/s]13172 examples [00:13, 1018.76 examples/s]13275 examples [00:13, 1021.25 examples/s]13378 examples [00:13, 1023.75 examples/s]13481 examples [00:13, 1017.69 examples/s]13583 examples [00:13, 1017.14 examples/s]13685 examples [00:13, 1010.48 examples/s]13789 examples [00:14, 1018.76 examples/s]13894 examples [00:14, 1025.38 examples/s]13997 examples [00:14, 1024.31 examples/s]14101 examples [00:14, 1027.70 examples/s]14206 examples [00:14, 1031.81 examples/s]14311 examples [00:14, 1035.61 examples/s]14416 examples [00:14, 1038.07 examples/s]14520 examples [00:14, 1030.55 examples/s]14624 examples [00:14, 1032.59 examples/s]14729 examples [00:14, 1035.70 examples/s]14833 examples [00:15, 1036.23 examples/s]14937 examples [00:15, 1036.06 examples/s]15041 examples [00:15, 998.12 examples/s] 15142 examples [00:15, 995.74 examples/s]15243 examples [00:15, 992.69 examples/s]15345 examples [00:15, 998.65 examples/s]15448 examples [00:15, 1007.49 examples/s]15549 examples [00:15, 1007.89 examples/s]15651 examples [00:15, 1009.20 examples/s]15755 examples [00:15, 1016.37 examples/s]15857 examples [00:16, 1013.32 examples/s]15959 examples [00:16, 1013.69 examples/s]16061 examples [00:16, 1014.18 examples/s]16164 examples [00:16, 1017.34 examples/s]16266 examples [00:16, 1013.99 examples/s]16370 examples [00:16, 1019.17 examples/s]16472 examples [00:16, 1011.99 examples/s]16575 examples [00:16, 1016.70 examples/s]16679 examples [00:16, 1023.29 examples/s]16783 examples [00:16, 1026.30 examples/s]16886 examples [00:17, 1025.20 examples/s]16989 examples [00:17, 1010.72 examples/s]17093 examples [00:17, 1018.27 examples/s]17199 examples [00:17, 1027.52 examples/s]17304 examples [00:17, 1032.83 examples/s]17409 examples [00:17, 1036.63 examples/s]17513 examples [00:17, 1037.18 examples/s]17617 examples [00:17, 1034.77 examples/s]17723 examples [00:17, 1040.65 examples/s]17828 examples [00:17, 1041.14 examples/s]17933 examples [00:18, 1043.18 examples/s]18038 examples [00:18, 1042.02 examples/s]18143 examples [00:18, 1036.29 examples/s]18247 examples [00:18, 1035.84 examples/s]18351 examples [00:18, 1035.74 examples/s]18455 examples [00:18, 1033.92 examples/s]18559 examples [00:18, 1032.35 examples/s]18663 examples [00:18, 1023.11 examples/s]18766 examples [00:18, 1022.44 examples/s]18869 examples [00:18, 1022.87 examples/s]18974 examples [00:19, 1029.47 examples/s]19079 examples [00:19, 1032.82 examples/s]19183 examples [00:19, 1007.95 examples/s]19288 examples [00:19, 1017.88 examples/s]19391 examples [00:19, 1019.71 examples/s]19496 examples [00:19, 1028.61 examples/s]19601 examples [00:19, 1034.68 examples/s]19705 examples [00:19, 1029.38 examples/s]19808 examples [00:19, 1029.52 examples/s]19911 examples [00:19, 1028.64 examples/s]20014 examples [00:20, 984.76 examples/s] 20117 examples [00:20, 997.73 examples/s]20218 examples [00:20, 998.54 examples/s]20319 examples [00:20, 984.16 examples/s]20424 examples [00:20, 1000.49 examples/s]20529 examples [00:20, 1013.67 examples/s]20632 examples [00:20, 1018.04 examples/s]20735 examples [00:20, 1019.97 examples/s]20840 examples [00:20, 1028.57 examples/s]20945 examples [00:21, 1033.06 examples/s]21050 examples [00:21, 1037.36 examples/s]21155 examples [00:21, 1040.68 examples/s]21260 examples [00:21, 1036.40 examples/s]21366 examples [00:21, 1040.59 examples/s]21471 examples [00:21, 1039.29 examples/s]21577 examples [00:21, 1043.49 examples/s]21682 examples [00:21, 1033.99 examples/s]21786 examples [00:21, 1023.88 examples/s]21889 examples [00:21, 1025.21 examples/s]21992 examples [00:22, 1021.78 examples/s]22095 examples [00:22, 1017.28 examples/s]22198 examples [00:22, 1020.71 examples/s]22301 examples [00:22, 1015.02 examples/s]22404 examples [00:22, 1018.52 examples/s]22506 examples [00:22, 1016.72 examples/s]22608 examples [00:22, 1010.69 examples/s]22710 examples [00:22, 1008.33 examples/s]22812 examples [00:22, 1009.75 examples/s]22914 examples [00:22, 1011.57 examples/s]23016 examples [00:23, 1011.23 examples/s]23118 examples [00:23, 991.71 examples/s] 23220 examples [00:23, 997.62 examples/s]23323 examples [00:23, 1003.57 examples/s]23426 examples [00:23, 1010.60 examples/s]23528 examples [00:23, 1012.30 examples/s]23631 examples [00:23, 1015.90 examples/s]23734 examples [00:23, 1017.92 examples/s]23836 examples [00:23, 1016.17 examples/s]23938 examples [00:23, 1009.81 examples/s]24040 examples [00:24, 1010.31 examples/s]24142 examples [00:24, 1012.17 examples/s]24244 examples [00:24, 1014.15 examples/s]24347 examples [00:24, 1016.67 examples/s]24449 examples [00:24, 1013.49 examples/s]24551 examples [00:24, 1012.66 examples/s]24653 examples [00:24, 1004.10 examples/s]24754 examples [00:24, 1000.78 examples/s]24856 examples [00:24, 1005.44 examples/s]24957 examples [00:24, 975.26 examples/s] 25055 examples [00:25, 975.54 examples/s]25157 examples [00:25, 985.97 examples/s]25256 examples [00:25, 983.36 examples/s]25360 examples [00:25, 998.23 examples/s]25461 examples [00:25, 1000.45 examples/s]25564 examples [00:25, 1007.34 examples/s]25667 examples [00:25, 1012.82 examples/s]25771 examples [00:25, 1017.46 examples/s]25875 examples [00:25, 1023.34 examples/s]25978 examples [00:25, 1022.43 examples/s]26082 examples [00:26, 1025.51 examples/s]26186 examples [00:26, 1027.83 examples/s]26289 examples [00:26, 1024.66 examples/s]26393 examples [00:26, 1027.71 examples/s]26496 examples [00:26, 1026.80 examples/s]26599 examples [00:26, 1026.54 examples/s]26702 examples [00:26, 1025.93 examples/s]26806 examples [00:26, 1027.98 examples/s]26909 examples [00:26, 1016.10 examples/s]27012 examples [00:26, 1017.54 examples/s]27116 examples [00:27, 1021.96 examples/s]27220 examples [00:27, 1025.27 examples/s]27325 examples [00:27, 1030.73 examples/s]27430 examples [00:27, 1033.73 examples/s]27534 examples [00:27, 1029.61 examples/s]27638 examples [00:27, 1032.31 examples/s]27742 examples [00:27, 1029.51 examples/s]27846 examples [00:27, 1030.65 examples/s]27950 examples [00:27, 1032.15 examples/s]28054 examples [00:28, 1023.24 examples/s]28158 examples [00:28, 1025.84 examples/s]28262 examples [00:28, 1027.46 examples/s]28365 examples [00:28, 1002.26 examples/s]28468 examples [00:28, 1010.31 examples/s]28570 examples [00:28, 1008.90 examples/s]28672 examples [00:28, 1010.99 examples/s]28775 examples [00:28, 1016.46 examples/s]28879 examples [00:28, 1021.01 examples/s]28982 examples [00:28, 1023.23 examples/s]29085 examples [00:29, 1019.00 examples/s]29189 examples [00:29, 1023.21 examples/s]29292 examples [00:29, 1012.68 examples/s]29394 examples [00:29, 1013.69 examples/s]29496 examples [00:29, 1004.74 examples/s]29597 examples [00:29, 1004.76 examples/s]29700 examples [00:29, 1011.25 examples/s]29803 examples [00:29, 1014.99 examples/s]29907 examples [00:29, 1020.72 examples/s]30010 examples [00:29, 979.53 examples/s] 30111 examples [00:30, 987.61 examples/s]30212 examples [00:30, 993.03 examples/s]30315 examples [00:30, 1003.43 examples/s]30418 examples [00:30, 1009.74 examples/s]30522 examples [00:30, 1016.79 examples/s]30625 examples [00:30, 1018.25 examples/s]30727 examples [00:30, 1017.74 examples/s]30829 examples [00:30, 991.28 examples/s] 30929 examples [00:30, 992.54 examples/s]31029 examples [00:30, 971.80 examples/s]31131 examples [00:31, 984.80 examples/s]31232 examples [00:31, 989.64 examples/s]31337 examples [00:31, 1006.87 examples/s]31443 examples [00:31, 1019.71 examples/s]31547 examples [00:31, 1023.57 examples/s]31651 examples [00:31, 1026.24 examples/s]31757 examples [00:31, 1033.51 examples/s]31861 examples [00:31, 1033.16 examples/s]31966 examples [00:31, 1036.32 examples/s]32070 examples [00:31, 1037.37 examples/s]32174 examples [00:32, 1016.30 examples/s]32278 examples [00:32, 1022.57 examples/s]32383 examples [00:32, 1029.97 examples/s]32488 examples [00:32, 1033.24 examples/s]32593 examples [00:32, 1035.93 examples/s]32697 examples [00:32, 1030.29 examples/s]32801 examples [00:32, 1026.38 examples/s]32905 examples [00:32, 1028.11 examples/s]33009 examples [00:32, 1029.98 examples/s]33113 examples [00:32, 1029.20 examples/s]33216 examples [00:33, 1018.69 examples/s]33318 examples [00:33, 983.70 examples/s] 33420 examples [00:33, 994.27 examples/s]33524 examples [00:33, 1006.86 examples/s]33628 examples [00:33, 1015.37 examples/s]33731 examples [00:33, 1017.02 examples/s]33833 examples [00:33, 996.40 examples/s] 33937 examples [00:33, 1008.60 examples/s]34041 examples [00:33, 1015.39 examples/s]34143 examples [00:34, 1005.68 examples/s]34246 examples [00:34, 1010.57 examples/s]34349 examples [00:34, 1016.21 examples/s]34451 examples [00:34, 999.12 examples/s] 34555 examples [00:34, 1008.81 examples/s]34657 examples [00:34, 1012.13 examples/s]34759 examples [00:34, 1008.55 examples/s]34862 examples [00:34, 1014.60 examples/s]34968 examples [00:34, 1027.13 examples/s]35074 examples [00:34, 1034.08 examples/s]35178 examples [00:35, 1035.30 examples/s]35282 examples [00:35, 1036.36 examples/s]35386 examples [00:35, 1011.86 examples/s]35489 examples [00:35, 1016.19 examples/s]35591 examples [00:35, 985.04 examples/s] 35695 examples [00:35, 999.83 examples/s]35800 examples [00:35, 1012.38 examples/s]35905 examples [00:35, 1020.94 examples/s]36009 examples [00:35, 1023.98 examples/s]36113 examples [00:35, 1026.82 examples/s]36216 examples [00:36, 1024.10 examples/s]36319 examples [00:36, 1024.55 examples/s]36424 examples [00:36, 1029.71 examples/s]36529 examples [00:36, 1035.69 examples/s]36634 examples [00:36, 1039.53 examples/s]36739 examples [00:36, 1042.42 examples/s]36844 examples [00:36, 1040.75 examples/s]36949 examples [00:36, 1040.15 examples/s]37054 examples [00:36, 1042.50 examples/s]37159 examples [00:36, 1042.42 examples/s]37265 examples [00:37, 1045.79 examples/s]37370 examples [00:37, 1027.41 examples/s]37473 examples [00:37, 1024.29 examples/s]37576 examples [00:37, 1024.67 examples/s]37679 examples [00:37, 1025.33 examples/s]37782 examples [00:37, 1022.38 examples/s]37885 examples [00:37, 1019.35 examples/s]37988 examples [00:37, 1020.10 examples/s]38091 examples [00:37, 1012.95 examples/s]38194 examples [00:37, 1016.47 examples/s]38296 examples [00:38, 1015.58 examples/s]38398 examples [00:38, 1012.15 examples/s]38501 examples [00:38, 1014.94 examples/s]38603 examples [00:38, 1012.98 examples/s]38706 examples [00:38, 1017.50 examples/s]38810 examples [00:38, 1021.35 examples/s]38913 examples [00:38, 1011.27 examples/s]39016 examples [00:38, 1014.03 examples/s]39119 examples [00:38, 1017.75 examples/s]39222 examples [00:38, 1020.03 examples/s]39325 examples [00:39, 1010.53 examples/s]39427 examples [00:39, 1008.27 examples/s]39528 examples [00:39, 997.38 examples/s] 39631 examples [00:39, 1006.62 examples/s]39734 examples [00:39, 1012.63 examples/s]39836 examples [00:39, 1011.04 examples/s]39938 examples [00:39, 1013.16 examples/s]40040 examples [00:39, 971.07 examples/s] 40142 examples [00:39, 984.74 examples/s]40245 examples [00:40, 996.07 examples/s]40345 examples [00:40, 953.60 examples/s]40450 examples [00:40, 978.04 examples/s]40554 examples [00:40, 993.19 examples/s]40658 examples [00:40, 1004.60 examples/s]40762 examples [00:40, 1014.83 examples/s]40866 examples [00:40, 1020.40 examples/s]40971 examples [00:40, 1028.09 examples/s]41076 examples [00:40, 1033.63 examples/s]41182 examples [00:40, 1038.73 examples/s]41286 examples [00:41, 1033.44 examples/s]41390 examples [00:41, 1026.56 examples/s]41493 examples [00:41, 1020.41 examples/s]41596 examples [00:41, 941.83 examples/s] 41692 examples [00:41, 898.41 examples/s]41784 examples [00:41, 868.91 examples/s]41872 examples [00:41, 861.51 examples/s]41959 examples [00:41, 838.05 examples/s]42047 examples [00:41, 849.49 examples/s]42133 examples [00:42, 847.13 examples/s]42220 examples [00:42, 852.20 examples/s]42308 examples [00:42, 858.47 examples/s]42407 examples [00:42, 893.86 examples/s]42511 examples [00:42, 931.03 examples/s]42613 examples [00:42, 955.95 examples/s]42716 examples [00:42, 974.51 examples/s]42816 examples [00:42, 979.95 examples/s]42919 examples [00:42, 994.36 examples/s]43020 examples [00:42, 998.76 examples/s]43124 examples [00:43, 1008.86 examples/s]43228 examples [00:43, 1015.47 examples/s]43330 examples [00:43, 1007.37 examples/s]43434 examples [00:43, 1015.56 examples/s]43536 examples [00:43, 1014.33 examples/s]43640 examples [00:43, 1019.19 examples/s]43744 examples [00:43, 1023.60 examples/s]43847 examples [00:43, 1020.33 examples/s]43951 examples [00:43, 1025.39 examples/s]44054 examples [00:43, 1021.49 examples/s]44157 examples [00:44, 975.88 examples/s] 44260 examples [00:44, 991.39 examples/s]44361 examples [00:44, 994.66 examples/s]44465 examples [00:44, 1005.37 examples/s]44566 examples [00:44, 1002.87 examples/s]44669 examples [00:44, 1010.49 examples/s]44771 examples [00:44, 1009.74 examples/s]44874 examples [00:44, 1014.43 examples/s]44980 examples [00:44, 1025.32 examples/s]45084 examples [00:44, 1029.21 examples/s]45189 examples [00:45, 1034.59 examples/s]45294 examples [00:45, 1038.66 examples/s]45398 examples [00:45, 1032.84 examples/s]45504 examples [00:45, 1038.15 examples/s]45608 examples [00:45, 1020.51 examples/s]45711 examples [00:45, 1015.90 examples/s]45816 examples [00:45, 1025.06 examples/s]45920 examples [00:45, 1026.90 examples/s]46025 examples [00:45, 1033.58 examples/s]46129 examples [00:45, 1024.05 examples/s]46235 examples [00:46, 1032.87 examples/s]46339 examples [00:46, 1015.47 examples/s]46442 examples [00:46, 1019.71 examples/s]46547 examples [00:46, 1026.62 examples/s]46650 examples [00:46, 1027.38 examples/s]46755 examples [00:46, 1032.74 examples/s]46860 examples [00:46, 1034.89 examples/s]46964 examples [00:46, 1025.09 examples/s]47067 examples [00:46, 1022.36 examples/s]47170 examples [00:46, 1023.20 examples/s]47273 examples [00:47, 1021.36 examples/s]47376 examples [00:47, 1022.22 examples/s]47479 examples [00:47, 1004.62 examples/s]47581 examples [00:47, 1007.31 examples/s]47684 examples [00:47, 1011.68 examples/s]47788 examples [00:47, 1017.38 examples/s]47892 examples [00:47, 1022.01 examples/s]47995 examples [00:47, 1015.72 examples/s]48099 examples [00:47, 1021.56 examples/s]48202 examples [00:48, 1022.03 examples/s]48305 examples [00:48, 1023.56 examples/s]48408 examples [00:48, 1024.41 examples/s]48511 examples [00:48, 1017.97 examples/s]48614 examples [00:48, 1021.47 examples/s]48717 examples [00:48, 1020.84 examples/s]48821 examples [00:48, 1025.32 examples/s]48924 examples [00:48, 1026.41 examples/s]49027 examples [00:48, 1020.58 examples/s]49131 examples [00:48, 1024.45 examples/s]49234 examples [00:49, 1021.37 examples/s]49337 examples [00:49, 1021.81 examples/s]49440 examples [00:49, 1021.44 examples/s]49543 examples [00:49, 1021.26 examples/s]49646 examples [00:49, 994.65 examples/s] 49749 examples [00:49, 1002.23 examples/s]49850 examples [00:49, 999.55 examples/s] 49951 examples [00:49, 993.73 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7251/50000 [00:00<00:00, 72508.86 examples/s] 39%|███▉      | 19441/50000 [00:00<00:00, 82542.10 examples/s] 63%|██████▎   | 31638/50000 [00:00<00:00, 91405.97 examples/s] 88%|████████▊ | 43940/50000 [00:00<00:00, 99040.39 examples/s]                                                               0 examples [00:00, ? examples/s]86 examples [00:00, 855.78 examples/s]190 examples [00:00, 903.72 examples/s]295 examples [00:00, 941.92 examples/s]400 examples [00:00, 969.69 examples/s]506 examples [00:00, 992.58 examples/s]609 examples [00:00, 1001.47 examples/s]715 examples [00:00, 1017.00 examples/s]818 examples [00:00, 1020.34 examples/s]924 examples [00:00, 1031.84 examples/s]1030 examples [00:01, 1037.72 examples/s]1134 examples [00:01, 1036.05 examples/s]1239 examples [00:01, 1038.35 examples/s]1344 examples [00:01, 1039.32 examples/s]1450 examples [00:01, 1043.47 examples/s]1555 examples [00:01, 1044.37 examples/s]1660 examples [00:01, 1036.08 examples/s]1764 examples [00:01, 1034.23 examples/s]1869 examples [00:01, 1036.41 examples/s]1973 examples [00:01, 1030.61 examples/s]2078 examples [00:02, 1035.89 examples/s]2182 examples [00:02, 1029.42 examples/s]2289 examples [00:02, 1038.81 examples/s]2396 examples [00:02, 1046.32 examples/s]2502 examples [00:02, 1050.26 examples/s]2608 examples [00:02, 1050.52 examples/s]2714 examples [00:02, 1046.09 examples/s]2821 examples [00:02, 1050.65 examples/s]2928 examples [00:02, 1054.02 examples/s]3034 examples [00:02, 1047.27 examples/s]3141 examples [00:03, 1052.84 examples/s]3247 examples [00:03, 1033.52 examples/s]3354 examples [00:03, 1042.58 examples/s]3459 examples [00:03, 1044.43 examples/s]3565 examples [00:03, 1046.23 examples/s]3670 examples [00:03, 1046.86 examples/s]3775 examples [00:03, 1040.33 examples/s]3880 examples [00:03, 1030.43 examples/s]3987 examples [00:03, 1041.13 examples/s]4093 examples [00:03, 1043.85 examples/s]4200 examples [00:04, 1049.84 examples/s]4306 examples [00:04, 1041.09 examples/s]4413 examples [00:04, 1047.17 examples/s]4519 examples [00:04, 1050.85 examples/s]4626 examples [00:04, 1056.12 examples/s]4733 examples [00:04, 1059.40 examples/s]4839 examples [00:04, 846.88 examples/s] 4942 examples [00:04, 894.17 examples/s]5047 examples [00:04, 933.55 examples/s]5152 examples [00:05, 963.27 examples/s]5255 examples [00:05, 981.57 examples/s]5360 examples [00:05, 999.43 examples/s]5462 examples [00:05, 981.56 examples/s]5569 examples [00:05, 1005.32 examples/s]5676 examples [00:05, 1023.33 examples/s]5781 examples [00:05, 1028.70 examples/s]5888 examples [00:05, 1040.68 examples/s]5993 examples [00:05, 1041.69 examples/s]6099 examples [00:05, 1046.94 examples/s]6206 examples [00:06, 1051.03 examples/s]6312 examples [00:06, 1045.78 examples/s]6417 examples [00:06, 1039.15 examples/s]6522 examples [00:06, 1034.60 examples/s]6626 examples [00:06, 1032.95 examples/s]6732 examples [00:06, 1038.13 examples/s]6836 examples [00:06, 1032.28 examples/s]6941 examples [00:06, 1035.73 examples/s]7045 examples [00:06, 1035.32 examples/s]7149 examples [00:06, 1033.82 examples/s]7254 examples [00:07, 1036.72 examples/s]7358 examples [00:07, 1036.00 examples/s]7464 examples [00:07, 1040.19 examples/s]7569 examples [00:07, 1033.80 examples/s]7674 examples [00:07, 1036.07 examples/s]7778 examples [00:07, 1035.63 examples/s]7882 examples [00:07, 1035.68 examples/s]7989 examples [00:07, 1044.01 examples/s]8095 examples [00:07, 1046.95 examples/s]8200 examples [00:07, 1018.95 examples/s]8303 examples [00:08, 987.94 examples/s] 8406 examples [00:08, 999.41 examples/s]8511 examples [00:08, 1013.07 examples/s]8616 examples [00:08, 1023.13 examples/s]8721 examples [00:08, 1028.47 examples/s]8826 examples [00:08, 1033.37 examples/s]8930 examples [00:08, 1022.93 examples/s]9035 examples [00:08, 1029.13 examples/s]9139 examples [00:08, 1031.04 examples/s]9244 examples [00:08, 1034.28 examples/s]9349 examples [00:09, 1038.39 examples/s]9453 examples [00:09, 1020.90 examples/s]9558 examples [00:09, 1027.47 examples/s]9663 examples [00:09, 1032.48 examples/s]9767 examples [00:09, 1034.14 examples/s]9871 examples [00:09, 988.73 examples/s] 9972 examples [00:09, 993.17 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteF1UYDB/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteF1UYDB/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f32b673c510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f32b673c510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f32b673c510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f323daa6748>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f323c9d8320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f32b673c510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f32b673c510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f32b673c510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f329d989358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f329d989208>), {}) 

  




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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.45 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.45 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.45 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.74 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.74 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.47 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.47 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.29 file/s]2020-09-05 12:09:30.733323: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-09-05 12:09:30.737343: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-09-05 12:09:30.737567: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5568782cd2e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-05 12:09:30.737589: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'mnist2', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy', 'train', 'test'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:00, 163287.34it/s] 77%|███████▋  | 7618560/9912422 [00:00<00:09, 233052.92it/s]9920512it [00:00, 45049486.83it/s]                           
0it [00:00, ?it/s]32768it [00:00, 549573.36it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 469080.26it/s]1654784it [00:00, 11764899.47it/s]                         
0it [00:00, ?it/s]8192it [00:00, 198119.91it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

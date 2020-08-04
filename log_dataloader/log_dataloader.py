
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f26a94f2ae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f26a94f2ae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f2714162510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f2714162510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f273338fea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f273338fea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f26c150aa60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f26c150aa60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f26c150aa60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 33%|███▎      | 3252224/9912422 [00:00<00:00, 31525828.44it/s]9920512it [00:00, 31954005.47it/s]                             
0it [00:00, ?it/s]32768it [00:00, 498436.76it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 161649.15it/s]1654784it [00:00, 11291531.27it/s]                         
0it [00:00, ?it/s]8192it [00:00, 194310.54it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f26c163e5c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f26a93bd9e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f26c150a6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f26c150a6a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f26c150a6a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:20,  2.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:20,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:19,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:19,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:18,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:17,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:17,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:54,  2.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:53,  2.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:52,  2.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:52,  2.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:51,  2.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:51,  2.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  3.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:35,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:34,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:34,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:34,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:23,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:22,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:22,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:22,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  7.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:15,  7.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:14,  7.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:14,  7.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 10.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 14.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 14.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 14.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 14.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 14.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 19.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 19.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 25.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 25.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 33.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 33.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 33.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 33.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 33.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 33.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 41.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 41.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 41.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 49.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 49.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 49.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 49.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 49.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 49.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 49.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 57.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 65.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 65.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 65.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 65.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 65.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 65.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 65.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 65.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 65.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 65.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 65.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 72.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 72.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 72.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 72.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 72.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 72.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 72.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 72.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 77.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 81.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 81.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 81.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 81.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 81.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 81.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.23s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.23s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.23s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.84 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.88s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  2.23s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 81.84 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.88s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.88s/ file]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 41.75 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.88s/ url]
0 examples [00:00, ? examples/s]2020-08-04 00:07:52.899541: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-04 00:07:52.915573: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-08-04 00:07:52.916102: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f6a71ad410 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-04 00:07:52.916133: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.20 examples/s]142 examples [00:00,  3.14 examples/s]284 examples [00:00,  4.48 examples/s]425 examples [00:00,  6.39 examples/s]567 examples [00:00,  9.12 examples/s]709 examples [00:00, 12.99 examples/s]853 examples [00:01, 18.48 examples/s]997 examples [00:01, 26.26 examples/s]1141 examples [00:01, 37.22 examples/s]1285 examples [00:01, 52.59 examples/s]1423 examples [00:01, 73.90 examples/s]1561 examples [00:01, 102.98 examples/s]1695 examples [00:01, 142.20 examples/s]1827 examples [00:01, 193.97 examples/s]1958 examples [00:01, 260.25 examples/s]2090 examples [00:01, 342.82 examples/s]2221 examples [00:02, 440.22 examples/s]2352 examples [00:02, 549.00 examples/s]2490 examples [00:02, 669.68 examples/s]2623 examples [00:02, 784.36 examples/s]2759 examples [00:02, 898.41 examples/s]2893 examples [00:02, 981.04 examples/s]3024 examples [00:02, 1047.62 examples/s]3156 examples [00:02, 1115.10 examples/s]3289 examples [00:02, 1170.61 examples/s]3422 examples [00:02, 1213.43 examples/s]3558 examples [00:03, 1253.18 examples/s]3691 examples [00:03, 1237.42 examples/s]3828 examples [00:03, 1272.59 examples/s]3964 examples [00:03, 1296.93 examples/s]4098 examples [00:03, 1308.94 examples/s]4231 examples [00:03, 1297.83 examples/s]4371 examples [00:03, 1325.86 examples/s]4509 examples [00:03, 1338.74 examples/s]4644 examples [00:03, 1339.42 examples/s]4779 examples [00:04, 1340.47 examples/s]4919 examples [00:04, 1356.40 examples/s]5057 examples [00:04, 1363.20 examples/s]5194 examples [00:04, 1278.51 examples/s]5324 examples [00:04, 1277.55 examples/s]5464 examples [00:04, 1309.96 examples/s]5596 examples [00:04, 1300.19 examples/s]5727 examples [00:04, 1275.75 examples/s]5856 examples [00:04, 1268.80 examples/s]5984 examples [00:04, 1270.21 examples/s]6126 examples [00:05, 1310.66 examples/s]6269 examples [00:05, 1341.97 examples/s]6404 examples [00:05, 1340.50 examples/s]6539 examples [00:05, 1327.87 examples/s]6673 examples [00:05, 1252.34 examples/s]6800 examples [00:05, 1247.54 examples/s]6932 examples [00:05, 1266.02 examples/s]7070 examples [00:05, 1295.93 examples/s]7210 examples [00:05, 1325.05 examples/s]7344 examples [00:05, 1321.91 examples/s]7484 examples [00:06, 1343.24 examples/s]7620 examples [00:06, 1348.00 examples/s]7758 examples [00:06, 1356.93 examples/s]7894 examples [00:06, 1344.84 examples/s]8029 examples [00:06, 1310.52 examples/s]8164 examples [00:06, 1321.05 examples/s]8304 examples [00:06, 1341.78 examples/s]8441 examples [00:06, 1348.15 examples/s]8576 examples [00:06, 1339.25 examples/s]8711 examples [00:07, 1327.46 examples/s]8845 examples [00:07, 1328.22 examples/s]8980 examples [00:07, 1332.62 examples/s]9119 examples [00:07, 1348.12 examples/s]9258 examples [00:07, 1359.94 examples/s]9395 examples [00:07, 1337.34 examples/s]9529 examples [00:07, 1333.58 examples/s]9670 examples [00:07, 1355.20 examples/s]9806 examples [00:07, 1346.17 examples/s]9941 examples [00:07, 1329.39 examples/s]10075 examples [00:08, 1239.20 examples/s]10203 examples [00:08, 1250.89 examples/s]10340 examples [00:08, 1284.36 examples/s]10476 examples [00:08, 1306.04 examples/s]10611 examples [00:08, 1316.23 examples/s]10744 examples [00:08, 1305.26 examples/s]10875 examples [00:08, 1293.10 examples/s]11011 examples [00:08, 1310.38 examples/s]11143 examples [00:08, 1309.80 examples/s]11275 examples [00:08, 1297.95 examples/s]11405 examples [00:09, 1276.10 examples/s]11533 examples [00:09, 1260.77 examples/s]11660 examples [00:09, 1214.00 examples/s]11793 examples [00:09, 1244.86 examples/s]11919 examples [00:09, 1219.61 examples/s]12042 examples [00:09, 1222.49 examples/s]12172 examples [00:09, 1242.94 examples/s]12298 examples [00:09, 1246.55 examples/s]12432 examples [00:09, 1273.11 examples/s]12564 examples [00:09, 1286.53 examples/s]12696 examples [00:10, 1296.25 examples/s]12830 examples [00:10, 1307.09 examples/s]12966 examples [00:10, 1320.57 examples/s]13103 examples [00:10, 1334.47 examples/s]13237 examples [00:10, 1321.90 examples/s]13370 examples [00:10, 1288.57 examples/s]13510 examples [00:10, 1318.56 examples/s]13648 examples [00:10, 1333.66 examples/s]13789 examples [00:10, 1353.76 examples/s]13925 examples [00:11, 1353.51 examples/s]14063 examples [00:11, 1359.12 examples/s]14201 examples [00:11, 1364.00 examples/s]14338 examples [00:11, 1322.95 examples/s]14471 examples [00:11, 1316.51 examples/s]14607 examples [00:11, 1326.45 examples/s]14740 examples [00:11, 1266.22 examples/s]14870 examples [00:11, 1273.56 examples/s]15003 examples [00:11, 1289.63 examples/s]15135 examples [00:11, 1296.81 examples/s]15265 examples [00:12, 1297.74 examples/s]15395 examples [00:12, 1270.14 examples/s]15532 examples [00:12, 1296.46 examples/s]15667 examples [00:12, 1309.59 examples/s]15802 examples [00:12, 1319.41 examples/s]15935 examples [00:12, 1301.07 examples/s]16071 examples [00:12, 1316.41 examples/s]16207 examples [00:12, 1328.91 examples/s]16344 examples [00:12, 1339.36 examples/s]16479 examples [00:12, 1289.95 examples/s]16609 examples [00:13, 1256.31 examples/s]16736 examples [00:13, 1258.39 examples/s]16880 examples [00:13, 1305.79 examples/s]17023 examples [00:13, 1338.81 examples/s]17168 examples [00:13, 1368.24 examples/s]17306 examples [00:13, 1348.77 examples/s]17450 examples [00:13, 1374.59 examples/s]17590 examples [00:13, 1379.75 examples/s]17733 examples [00:13, 1392.77 examples/s]17874 examples [00:13, 1396.87 examples/s]18014 examples [00:14, 1355.51 examples/s]18155 examples [00:14, 1369.56 examples/s]18294 examples [00:14, 1373.01 examples/s]18432 examples [00:14, 1364.28 examples/s]18572 examples [00:14, 1372.71 examples/s]18710 examples [00:14, 1294.52 examples/s]18841 examples [00:14, 1238.75 examples/s]18970 examples [00:14, 1251.03 examples/s]19111 examples [00:14, 1292.18 examples/s]19248 examples [00:15, 1314.00 examples/s]19381 examples [00:15, 1315.20 examples/s]19514 examples [00:15, 1304.34 examples/s]19655 examples [00:15, 1334.12 examples/s]19793 examples [00:15, 1346.91 examples/s]19929 examples [00:15, 1326.00 examples/s]20062 examples [00:15, 1245.05 examples/s]20198 examples [00:15, 1277.18 examples/s]20340 examples [00:15, 1316.26 examples/s]20483 examples [00:15, 1346.20 examples/s]20619 examples [00:16, 1329.44 examples/s]20753 examples [00:16, 1320.47 examples/s]20886 examples [00:16, 1306.10 examples/s]21027 examples [00:16, 1335.00 examples/s]21169 examples [00:16, 1358.00 examples/s]21306 examples [00:16, 1307.93 examples/s]21444 examples [00:16, 1328.16 examples/s]21587 examples [00:16, 1355.78 examples/s]21724 examples [00:16, 1351.82 examples/s]21860 examples [00:17, 1346.88 examples/s]21995 examples [00:17, 1319.04 examples/s]22128 examples [00:17, 1321.46 examples/s]22266 examples [00:17, 1338.30 examples/s]22407 examples [00:17, 1358.31 examples/s]22550 examples [00:17, 1378.04 examples/s]22689 examples [00:17, 1319.64 examples/s]22822 examples [00:17, 1274.82 examples/s]22952 examples [00:17, 1279.72 examples/s]23085 examples [00:17, 1293.67 examples/s]23216 examples [00:18, 1296.08 examples/s]23346 examples [00:18, 1276.62 examples/s]23484 examples [00:18, 1304.78 examples/s]23623 examples [00:18, 1327.74 examples/s]23763 examples [00:18, 1347.99 examples/s]23901 examples [00:18, 1356.09 examples/s]24037 examples [00:18, 1332.11 examples/s]24180 examples [00:18, 1360.00 examples/s]24323 examples [00:18, 1380.23 examples/s]24462 examples [00:18, 1368.66 examples/s]24600 examples [00:19, 1326.45 examples/s]24734 examples [00:19, 1236.06 examples/s]24860 examples [00:19, 1224.37 examples/s]24999 examples [00:19, 1267.70 examples/s]25141 examples [00:19, 1307.95 examples/s]25273 examples [00:19, 1308.60 examples/s]25405 examples [00:19, 1294.34 examples/s]25536 examples [00:19, 1256.53 examples/s]25675 examples [00:19, 1291.93 examples/s]25813 examples [00:20, 1314.91 examples/s]25952 examples [00:20, 1334.58 examples/s]26088 examples [00:20, 1341.99 examples/s]26227 examples [00:20, 1353.42 examples/s]26370 examples [00:20, 1375.40 examples/s]26508 examples [00:20, 1373.05 examples/s]26646 examples [00:20, 1357.03 examples/s]26782 examples [00:20, 1331.58 examples/s]26916 examples [00:20, 1318.05 examples/s]27053 examples [00:20, 1332.34 examples/s]27191 examples [00:21, 1344.37 examples/s]27326 examples [00:21, 1317.23 examples/s]27458 examples [00:21, 1296.67 examples/s]27594 examples [00:21, 1312.42 examples/s]27733 examples [00:21, 1333.66 examples/s]27867 examples [00:21, 1327.21 examples/s]28000 examples [00:21, 1284.08 examples/s]28134 examples [00:21, 1298.71 examples/s]28269 examples [00:21, 1311.78 examples/s]28408 examples [00:21, 1332.28 examples/s]28547 examples [00:22, 1348.92 examples/s]28683 examples [00:22, 1313.06 examples/s]28815 examples [00:22, 1314.14 examples/s]28949 examples [00:22, 1320.13 examples/s]29087 examples [00:22, 1335.49 examples/s]29221 examples [00:22, 1335.71 examples/s]29355 examples [00:22, 1320.56 examples/s]29488 examples [00:22, 1311.00 examples/s]29622 examples [00:22, 1316.70 examples/s]29760 examples [00:22, 1334.31 examples/s]29897 examples [00:23, 1343.81 examples/s]30032 examples [00:23, 1244.84 examples/s]30167 examples [00:23, 1273.25 examples/s]30304 examples [00:23, 1299.35 examples/s]30438 examples [00:23, 1309.35 examples/s]30571 examples [00:23, 1314.82 examples/s]30703 examples [00:23, 1242.15 examples/s]30834 examples [00:23, 1261.05 examples/s]30963 examples [00:23, 1267.80 examples/s]31092 examples [00:24, 1273.18 examples/s]31228 examples [00:24, 1295.63 examples/s]31358 examples [00:24, 1291.37 examples/s]31493 examples [00:24, 1307.00 examples/s]31624 examples [00:24, 1291.39 examples/s]31760 examples [00:24, 1309.41 examples/s]31892 examples [00:24, 1304.71 examples/s]32023 examples [00:24, 1249.46 examples/s]32149 examples [00:24, 1224.54 examples/s]32272 examples [00:24, 1196.74 examples/s]32401 examples [00:25, 1220.91 examples/s]32532 examples [00:25, 1246.24 examples/s]32658 examples [00:25, 1246.68 examples/s]32794 examples [00:25, 1278.37 examples/s]32932 examples [00:25, 1305.58 examples/s]33067 examples [00:25, 1316.84 examples/s]33204 examples [00:25, 1329.92 examples/s]33338 examples [00:25, 1314.76 examples/s]33475 examples [00:25, 1328.52 examples/s]33609 examples [00:25, 1329.19 examples/s]33743 examples [00:26, 1315.94 examples/s]33878 examples [00:26, 1323.06 examples/s]34011 examples [00:26, 1312.74 examples/s]34150 examples [00:26, 1333.06 examples/s]34284 examples [00:26, 1333.19 examples/s]34420 examples [00:26, 1339.87 examples/s]34555 examples [00:26, 1339.99 examples/s]34690 examples [00:26, 1316.51 examples/s]34822 examples [00:26, 1314.89 examples/s]34954 examples [00:27, 1308.12 examples/s]35093 examples [00:27, 1330.68 examples/s]35227 examples [00:27, 1317.73 examples/s]35359 examples [00:27, 1305.72 examples/s]35495 examples [00:27, 1321.21 examples/s]35634 examples [00:27, 1340.06 examples/s]35769 examples [00:27, 1292.75 examples/s]35899 examples [00:27, 1282.93 examples/s]36028 examples [00:27, 1284.68 examples/s]36157 examples [00:27, 1281.32 examples/s]36291 examples [00:28, 1296.97 examples/s]36428 examples [00:28, 1317.32 examples/s]36565 examples [00:28, 1329.93 examples/s]36699 examples [00:28, 1303.17 examples/s]36838 examples [00:28, 1325.32 examples/s]36971 examples [00:28, 1316.48 examples/s]37105 examples [00:28, 1319.35 examples/s]37242 examples [00:28, 1333.14 examples/s]37376 examples [00:28, 1291.58 examples/s]37508 examples [00:28, 1297.43 examples/s]37641 examples [00:29, 1305.21 examples/s]37775 examples [00:29, 1313.74 examples/s]37912 examples [00:29, 1327.49 examples/s]38045 examples [00:29, 1273.24 examples/s]38173 examples [00:29, 1259.85 examples/s]38309 examples [00:29, 1288.15 examples/s]38450 examples [00:29, 1320.75 examples/s]38583 examples [00:29, 1278.32 examples/s]38723 examples [00:29, 1311.47 examples/s]38861 examples [00:29, 1329.85 examples/s]38995 examples [00:30, 1318.44 examples/s]39128 examples [00:30, 1311.95 examples/s]39260 examples [00:30, 1291.06 examples/s]39398 examples [00:30, 1314.32 examples/s]39540 examples [00:30, 1343.63 examples/s]39679 examples [00:30, 1354.62 examples/s]39815 examples [00:30, 1353.68 examples/s]39953 examples [00:30, 1361.05 examples/s]40090 examples [00:30, 1283.87 examples/s]40229 examples [00:31, 1312.62 examples/s]40367 examples [00:31, 1331.40 examples/s]40506 examples [00:31, 1348.35 examples/s]40642 examples [00:31, 1336.28 examples/s]40785 examples [00:31, 1362.61 examples/s]40929 examples [00:31, 1382.50 examples/s]41069 examples [00:31, 1385.42 examples/s]41214 examples [00:31, 1402.50 examples/s]41355 examples [00:31, 1388.96 examples/s]41495 examples [00:31, 1392.16 examples/s]41635 examples [00:32, 1392.75 examples/s]41775 examples [00:32, 1377.76 examples/s]41913 examples [00:32, 1365.69 examples/s]42050 examples [00:32, 1347.41 examples/s]42187 examples [00:32, 1353.65 examples/s]42323 examples [00:32, 1329.80 examples/s]42457 examples [00:32, 1320.64 examples/s]42592 examples [00:32, 1326.32 examples/s]42725 examples [00:32, 1280.48 examples/s]42854 examples [00:32, 1281.21 examples/s]42983 examples [00:33, 1281.83 examples/s]43119 examples [00:33, 1303.67 examples/s]43256 examples [00:33, 1320.58 examples/s]43390 examples [00:33, 1325.93 examples/s]43532 examples [00:33, 1350.63 examples/s]43668 examples [00:33, 1325.35 examples/s]43801 examples [00:33, 1302.87 examples/s]43937 examples [00:33, 1318.15 examples/s]44070 examples [00:33, 1305.90 examples/s]44201 examples [00:33, 1298.58 examples/s]44334 examples [00:34, 1304.46 examples/s]44465 examples [00:34, 1293.24 examples/s]44602 examples [00:34, 1313.12 examples/s]44734 examples [00:34, 1303.24 examples/s]44870 examples [00:34, 1317.76 examples/s]45002 examples [00:34, 1287.27 examples/s]45141 examples [00:34, 1315.29 examples/s]45282 examples [00:34, 1339.52 examples/s]45417 examples [00:34, 1325.28 examples/s]45557 examples [00:35, 1344.66 examples/s]45697 examples [00:35, 1358.42 examples/s]45834 examples [00:35, 1332.51 examples/s]45968 examples [00:35, 1284.37 examples/s]46097 examples [00:35, 1278.96 examples/s]46226 examples [00:35, 1268.11 examples/s]46366 examples [00:35, 1302.68 examples/s]46499 examples [00:35, 1308.20 examples/s]46631 examples [00:35, 1285.61 examples/s]46761 examples [00:35, 1289.66 examples/s]46897 examples [00:36, 1307.86 examples/s]47032 examples [00:36, 1319.35 examples/s]47165 examples [00:36, 1317.35 examples/s]47303 examples [00:36, 1333.50 examples/s]47437 examples [00:36, 1328.80 examples/s]47571 examples [00:36, 1331.67 examples/s]47711 examples [00:36, 1350.00 examples/s]47850 examples [00:36, 1360.69 examples/s]47993 examples [00:36, 1378.63 examples/s]48131 examples [00:36, 1327.05 examples/s]48265 examples [00:37, 1314.51 examples/s]48397 examples [00:37, 1312.12 examples/s]48530 examples [00:37, 1316.82 examples/s]48665 examples [00:37, 1325.50 examples/s]48798 examples [00:37, 1299.47 examples/s]48929 examples [00:37, 1291.49 examples/s]49067 examples [00:37, 1315.36 examples/s]49202 examples [00:37, 1323.09 examples/s]49337 examples [00:37, 1329.07 examples/s]49471 examples [00:37, 1320.05 examples/s]49612 examples [00:38, 1343.81 examples/s]49750 examples [00:38, 1353.66 examples/s]49886 examples [00:38, 1350.36 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 21%|██        | 10428/50000 [00:00<00:00, 104278.11 examples/s] 52%|█████▏    | 25756/50000 [00:00<00:00, 115339.01 examples/s] 83%|████████▎ | 41580/50000 [00:00<00:00, 125548.50 examples/s]                                                                0 examples [00:00, ? examples/s]116 examples [00:00, 1158.50 examples/s]248 examples [00:00, 1198.67 examples/s]368 examples [00:00, 1197.44 examples/s]490 examples [00:00, 1201.40 examples/s]621 examples [00:00, 1229.91 examples/s]764 examples [00:00, 1281.52 examples/s]899 examples [00:00, 1301.08 examples/s]1032 examples [00:00, 1306.61 examples/s]1161 examples [00:00, 1299.11 examples/s]1289 examples [00:01, 1291.17 examples/s]1417 examples [00:01, 1287.09 examples/s]1546 examples [00:01, 1285.28 examples/s]1674 examples [00:01, 1282.72 examples/s]1806 examples [00:01, 1290.81 examples/s]1935 examples [00:01, 1290.03 examples/s]2066 examples [00:01, 1294.10 examples/s]2196 examples [00:01, 1295.38 examples/s]2329 examples [00:01, 1302.50 examples/s]2460 examples [00:01, 1296.67 examples/s]2590 examples [00:02, 1268.49 examples/s]2717 examples [00:02, 1254.89 examples/s]2843 examples [00:02, 1255.75 examples/s]2977 examples [00:02, 1279.36 examples/s]3122 examples [00:02, 1324.98 examples/s]3260 examples [00:02, 1339.99 examples/s]3395 examples [00:02, 1274.73 examples/s]3524 examples [00:02, 1252.93 examples/s]3651 examples [00:02, 1234.60 examples/s]3786 examples [00:02, 1266.49 examples/s]3922 examples [00:03, 1290.62 examples/s]4052 examples [00:03, 1252.71 examples/s]4178 examples [00:03, 1226.22 examples/s]4302 examples [00:03, 1221.30 examples/s]4428 examples [00:03, 1231.12 examples/s]4561 examples [00:03, 1258.90 examples/s]4697 examples [00:03, 1286.44 examples/s]4827 examples [00:03, 1258.35 examples/s]4954 examples [00:03, 1232.68 examples/s]5078 examples [00:04, 1174.19 examples/s]5209 examples [00:04, 1210.51 examples/s]5347 examples [00:04, 1255.59 examples/s]5479 examples [00:04, 1273.23 examples/s]5615 examples [00:04, 1296.68 examples/s]5746 examples [00:04, 1287.50 examples/s]5887 examples [00:04, 1321.95 examples/s]6028 examples [00:04, 1344.92 examples/s]6163 examples [00:04, 1323.82 examples/s]6302 examples [00:04, 1340.59 examples/s]6442 examples [00:05, 1356.18 examples/s]6578 examples [00:05, 1340.65 examples/s]6715 examples [00:05, 1347.11 examples/s]6850 examples [00:05, 1329.56 examples/s]6984 examples [00:05, 1315.90 examples/s]7116 examples [00:05, 1315.82 examples/s]7255 examples [00:05, 1335.52 examples/s]7392 examples [00:05, 1343.77 examples/s]7527 examples [00:05, 1322.42 examples/s]7661 examples [00:05, 1325.92 examples/s]7804 examples [00:06, 1353.35 examples/s]7943 examples [00:06, 1362.39 examples/s]8080 examples [00:06, 1354.89 examples/s]8216 examples [00:06, 1347.67 examples/s]8351 examples [00:06, 1334.39 examples/s]8485 examples [00:06, 1326.27 examples/s]8624 examples [00:06, 1341.86 examples/s]8759 examples [00:06, 1311.53 examples/s]8891 examples [00:06, 1264.43 examples/s]9036 examples [00:06, 1313.09 examples/s]9178 examples [00:07, 1341.66 examples/s]9322 examples [00:07, 1369.27 examples/s]9460 examples [00:07, 1368.44 examples/s]9599 examples [00:07, 1372.40 examples/s]9741 examples [00:07, 1382.83 examples/s]9885 examples [00:07, 1397.97 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSB09OO/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSB09OO/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f26c150aa60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f26c150aa60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f26c150aa60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f270ce77fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f272c99af98>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f26c150aa60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f26c150aa60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f26c150aa60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f26a93bd2b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f26a93bd048>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.50 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 23.45 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.26 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.26 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.41 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.41 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.89 file/s]2020-08-04 00:08:45.333280: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-04 00:08:45.336696: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-08-04 00:08:45.336843: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5577a19577c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-04 00:08:45.336855: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist2', 'test', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:11, 138956.92it/s] 40%|███▉      | 3923968/9912422 [00:00<00:30, 198201.99it/s] 54%|█████▍    | 5357568/9912422 [00:00<00:16, 281145.60it/s]9920512it [00:00, 27694306.52it/s]                           
0it [00:00, ?it/s]32768it [00:00, 506459.60it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160648.11it/s]1654784it [00:00, 10942798.86it/s]                         
0it [00:00, ?it/s]8192it [00:00, 188635.34it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

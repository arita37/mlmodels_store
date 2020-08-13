
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f3efaab2598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f3efaab2598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f3f65762488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f3f65762488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f3f84980ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f3f84980ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3f12a9d158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3f12a9d158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3f12a9d158> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:15, 130916.19it/s] 66%|██████▋   | 6569984/9912422 [00:00<00:17, 186863.00it/s]9920512it [00:00, 36165819.37it/s]                           
0it [00:00, ?it/s]32768it [00:00, 576798.43it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 139634.85it/s]1654784it [00:00, 10485658.61it/s]                         
0it [00:00, ?it/s]8192it [00:00, 116467.37it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3ef9cec6d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3f10c48978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f3f12a93d90> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f3f12a93d90> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f3f12a93d90> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:28,  1.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:28,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:28,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:27,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:27,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:26,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:26,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:25,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:00,  2.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:00,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:59,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:59,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:59,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:58,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:58,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:57,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:57,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:57,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:40,  3.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:40,  3.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:39,  3.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:39,  3.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:39,  3.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:39,  3.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:38,  3.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:38,  3.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:38,  3.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:27,  5.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:27,  5.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:26,  5.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:26,  5.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:26,  5.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:26,  5.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:26,  5.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:25,  5.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:25,  5.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:25,  5.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:18,  7.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:18,  7.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:18,  7.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:17,  7.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:17,  7.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:17,  7.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:17,  7.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:17,  7.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:17,  7.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:12,  9.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:12,  9.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:12,  9.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:12,  9.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:12,  9.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:12,  9.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11,  9.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11,  9.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 12.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 12.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:08, 12.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:08, 12.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:08, 12.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 12.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 17.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 17.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 17.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 17.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 17.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 17.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 17.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 17.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 17.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 22.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 22.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 22.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 22.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 22.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 22.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 22.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 22.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 27.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 27.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 27.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 27.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 27.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 27.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 27.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 27.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 33.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 33.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 33.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 33.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 33.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 33.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 39.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 39.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 39.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 39.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 39.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 39.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 39.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 39.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 45.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 45.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 45.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 45.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 45.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 49.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 49.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 49.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 49.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 49.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 49.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 49.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 49.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 54.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 54.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 54.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 54.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 54.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 54.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 54.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 54.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 54.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 58.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 62.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 62.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 62.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 62.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 62.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 62.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 62.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 62.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 62.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 67.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 68.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 68.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.84s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.84s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.84s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.31 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.38s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.84s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 68.31 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.38s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.38s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.09 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.38s/ url]
0 examples [00:00, ? examples/s]2020-08-13 18:07:26.689725: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-13 18:07:26.700991: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-13 18:07:26.701223: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5555ccc0f6f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-13 18:07:26.701244: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  9.74 examples/s]97 examples [00:00, 13.86 examples/s]186 examples [00:00, 19.66 examples/s]278 examples [00:00, 27.84 examples/s]365 examples [00:00, 39.23 examples/s]460 examples [00:00, 55.06 examples/s]553 examples [00:00, 76.71 examples/s]648 examples [00:00, 105.91 examples/s]740 examples [00:00, 144.11 examples/s]831 examples [00:01, 192.74 examples/s]924 examples [00:01, 252.87 examples/s]1015 examples [00:01, 322.05 examples/s]1109 examples [00:01, 400.95 examples/s]1204 examples [00:01, 484.57 examples/s]1297 examples [00:01, 565.32 examples/s]1389 examples [00:01, 633.33 examples/s]1480 examples [00:01, 660.79 examples/s]1566 examples [00:01, 703.61 examples/s]1664 examples [00:01, 768.13 examples/s]1756 examples [00:02, 806.55 examples/s]1849 examples [00:02, 837.56 examples/s]1940 examples [00:02, 848.75 examples/s]2031 examples [00:02, 865.54 examples/s]2125 examples [00:02, 884.23 examples/s]2218 examples [00:02, 896.95 examples/s]2310 examples [00:02, 898.51 examples/s]2404 examples [00:02, 908.92 examples/s]2498 examples [00:02, 916.01 examples/s]2591 examples [00:02, 918.98 examples/s]2684 examples [00:03, 910.97 examples/s]2776 examples [00:03, 906.43 examples/s]2867 examples [00:03, 898.88 examples/s]2960 examples [00:03, 907.76 examples/s]3052 examples [00:03, 910.78 examples/s]3144 examples [00:03, 902.69 examples/s]3235 examples [00:03, 877.35 examples/s]3329 examples [00:03, 893.56 examples/s]3422 examples [00:03, 901.78 examples/s]3515 examples [00:03, 909.55 examples/s]3608 examples [00:04, 915.36 examples/s]3702 examples [00:04, 921.97 examples/s]3795 examples [00:04, 901.66 examples/s]3886 examples [00:04, 890.55 examples/s]3980 examples [00:04, 904.38 examples/s]4071 examples [00:04, 878.08 examples/s]4162 examples [00:04, 885.63 examples/s]4257 examples [00:04, 903.13 examples/s]4351 examples [00:04, 912.85 examples/s]4446 examples [00:05, 922.41 examples/s]4542 examples [00:05, 932.31 examples/s]4642 examples [00:05, 951.06 examples/s]4738 examples [00:05, 947.31 examples/s]4833 examples [00:05, 927.38 examples/s]4926 examples [00:05, 893.12 examples/s]5016 examples [00:05, 894.05 examples/s]5106 examples [00:05, 894.68 examples/s]5196 examples [00:05, 889.15 examples/s]5286 examples [00:05, 882.21 examples/s]5378 examples [00:06, 890.83 examples/s]5469 examples [00:06, 895.39 examples/s]5562 examples [00:06, 904.60 examples/s]5654 examples [00:06, 907.31 examples/s]5745 examples [00:06, 898.57 examples/s]5839 examples [00:06, 909.70 examples/s]5931 examples [00:06, 904.58 examples/s]6022 examples [00:06, 880.41 examples/s]6115 examples [00:06, 893.41 examples/s]6205 examples [00:06, 876.72 examples/s]6295 examples [00:07, 882.86 examples/s]6388 examples [00:07, 896.11 examples/s]6482 examples [00:07, 908.54 examples/s]6575 examples [00:07, 912.11 examples/s]6669 examples [00:07, 918.68 examples/s]6766 examples [00:07, 931.20 examples/s]6860 examples [00:07, 908.71 examples/s]6952 examples [00:07, 911.07 examples/s]7044 examples [00:07, 913.69 examples/s]7136 examples [00:07, 879.50 examples/s]7229 examples [00:08, 892.91 examples/s]7319 examples [00:08, 894.76 examples/s]7413 examples [00:08, 906.53 examples/s]7509 examples [00:08, 918.93 examples/s]7602 examples [00:08, 913.25 examples/s]7694 examples [00:08, 911.02 examples/s]7786 examples [00:08, 907.28 examples/s]7877 examples [00:08, 895.90 examples/s]7972 examples [00:08, 909.40 examples/s]8064 examples [00:09, 868.45 examples/s]8158 examples [00:09, 886.93 examples/s]8251 examples [00:09, 897.10 examples/s]8342 examples [00:09, 900.46 examples/s]8437 examples [00:09, 911.38 examples/s]8529 examples [00:09, 892.40 examples/s]8619 examples [00:09, 870.59 examples/s]8707 examples [00:09, 866.44 examples/s]8794 examples [00:09, 822.09 examples/s]8877 examples [00:09, 814.72 examples/s]8968 examples [00:10, 840.37 examples/s]9057 examples [00:10, 853.40 examples/s]9148 examples [00:10, 868.00 examples/s]9236 examples [00:10, 866.45 examples/s]9323 examples [00:10, 860.75 examples/s]9417 examples [00:10, 881.65 examples/s]9506 examples [00:10, 879.59 examples/s]9595 examples [00:10, 861.59 examples/s]9690 examples [00:10, 886.11 examples/s]9784 examples [00:10, 899.83 examples/s]9878 examples [00:11, 910.00 examples/s]9970 examples [00:11, 906.80 examples/s]10061 examples [00:11, 845.92 examples/s]10154 examples [00:11, 868.49 examples/s]10248 examples [00:11, 887.47 examples/s]10338 examples [00:11, 853.39 examples/s]10427 examples [00:11, 862.99 examples/s]10522 examples [00:11, 885.15 examples/s]10615 examples [00:11, 896.09 examples/s]10706 examples [00:12, 882.35 examples/s]10797 examples [00:12, 890.25 examples/s]10891 examples [00:12, 901.87 examples/s]10983 examples [00:12, 905.27 examples/s]11076 examples [00:12, 911.34 examples/s]11170 examples [00:12, 917.14 examples/s]11263 examples [00:12, 918.58 examples/s]11355 examples [00:12, 916.04 examples/s]11447 examples [00:12, 916.72 examples/s]11539 examples [00:12, 915.05 examples/s]11632 examples [00:13, 917.41 examples/s]11724 examples [00:13, 917.82 examples/s]11817 examples [00:13, 919.07 examples/s]11912 examples [00:13, 927.02 examples/s]12006 examples [00:13, 928.88 examples/s]12099 examples [00:13, 928.83 examples/s]12192 examples [00:13, 921.81 examples/s]12285 examples [00:13, 923.79 examples/s]12378 examples [00:13, 907.02 examples/s]12470 examples [00:13, 910.16 examples/s]12564 examples [00:14, 918.10 examples/s]12657 examples [00:14, 918.76 examples/s]12749 examples [00:14, 910.61 examples/s]12841 examples [00:14, 896.41 examples/s]12931 examples [00:14, 897.16 examples/s]13021 examples [00:14, 895.63 examples/s]13111 examples [00:14, 894.04 examples/s]13205 examples [00:14, 906.29 examples/s]13300 examples [00:14, 916.28 examples/s]13392 examples [00:14, 891.06 examples/s]13482 examples [00:15, 885.33 examples/s]13571 examples [00:15, 886.67 examples/s]13668 examples [00:15, 908.40 examples/s]13762 examples [00:15, 916.70 examples/s]13855 examples [00:15, 920.41 examples/s]13948 examples [00:15, 918.80 examples/s]14040 examples [00:15, 914.22 examples/s]14132 examples [00:15, 900.44 examples/s]14223 examples [00:15, 894.65 examples/s]14313 examples [00:16, 884.51 examples/s]14402 examples [00:16, 884.41 examples/s]14496 examples [00:16, 898.34 examples/s]14588 examples [00:16, 894.43 examples/s]14678 examples [00:16, 882.88 examples/s]14767 examples [00:16, 884.31 examples/s]14860 examples [00:16, 895.20 examples/s]14950 examples [00:16, 895.99 examples/s]15040 examples [00:16, 871.80 examples/s]15129 examples [00:16, 875.72 examples/s]15217 examples [00:17, 876.87 examples/s]15305 examples [00:17, 851.38 examples/s]15391 examples [00:17, 852.26 examples/s]15478 examples [00:17, 857.22 examples/s]15564 examples [00:17, 858.01 examples/s]15659 examples [00:17, 883.09 examples/s]15754 examples [00:17, 900.74 examples/s]15845 examples [00:17, 886.93 examples/s]15939 examples [00:17, 902.12 examples/s]16035 examples [00:17, 916.30 examples/s]16131 examples [00:18, 928.20 examples/s]16228 examples [00:18, 937.95 examples/s]16323 examples [00:18, 940.07 examples/s]16419 examples [00:18, 945.39 examples/s]16514 examples [00:18, 930.69 examples/s]16608 examples [00:18, 918.81 examples/s]16705 examples [00:18, 931.26 examples/s]16799 examples [00:18, 929.84 examples/s]16893 examples [00:18, 908.68 examples/s]16987 examples [00:18, 915.19 examples/s]17079 examples [00:19, 914.98 examples/s]17174 examples [00:19, 922.39 examples/s]17267 examples [00:19, 922.87 examples/s]17361 examples [00:19, 927.83 examples/s]17454 examples [00:19, 918.71 examples/s]17546 examples [00:19, 914.43 examples/s]17638 examples [00:19, 907.94 examples/s]17733 examples [00:19, 918.42 examples/s]17828 examples [00:19, 925.58 examples/s]17921 examples [00:19, 923.47 examples/s]18014 examples [00:20, 923.85 examples/s]18108 examples [00:20, 928.63 examples/s]18202 examples [00:20, 929.03 examples/s]18295 examples [00:20, 928.89 examples/s]18388 examples [00:20, 928.93 examples/s]18481 examples [00:20, 927.47 examples/s]18578 examples [00:20, 937.59 examples/s]18674 examples [00:20, 943.15 examples/s]18769 examples [00:20, 930.41 examples/s]18863 examples [00:20, 928.84 examples/s]18956 examples [00:21, 926.74 examples/s]19049 examples [00:21, 916.93 examples/s]19145 examples [00:21, 928.76 examples/s]19238 examples [00:21, 928.32 examples/s]19333 examples [00:21, 934.71 examples/s]19430 examples [00:21, 944.56 examples/s]19526 examples [00:21, 948.62 examples/s]19621 examples [00:21, 934.72 examples/s]19715 examples [00:21, 931.78 examples/s]19809 examples [00:22, 933.61 examples/s]19903 examples [00:22, 935.09 examples/s]20001 examples [00:22, 889.77 examples/s]20095 examples [00:22, 903.88 examples/s]20186 examples [00:22, 893.99 examples/s]20277 examples [00:22, 898.53 examples/s]20368 examples [00:22, 876.80 examples/s]20459 examples [00:22, 884.17 examples/s]20552 examples [00:22, 894.83 examples/s]20647 examples [00:22, 909.17 examples/s]20739 examples [00:23, 909.72 examples/s]20834 examples [00:23, 919.24 examples/s]20930 examples [00:23, 930.03 examples/s]21024 examples [00:23, 919.05 examples/s]21117 examples [00:23, 917.22 examples/s]21212 examples [00:23, 925.12 examples/s]21306 examples [00:23, 929.45 examples/s]21403 examples [00:23, 939.28 examples/s]21501 examples [00:23, 949.83 examples/s]21597 examples [00:23, 947.05 examples/s]21692 examples [00:24, 946.67 examples/s]21789 examples [00:24, 951.41 examples/s]21885 examples [00:24, 950.12 examples/s]21981 examples [00:24, 939.46 examples/s]22075 examples [00:24, 909.62 examples/s]22168 examples [00:24, 912.86 examples/s]22260 examples [00:24, 909.38 examples/s]22352 examples [00:24, 865.71 examples/s]22443 examples [00:24, 876.28 examples/s]22532 examples [00:25, 862.32 examples/s]22627 examples [00:25, 884.71 examples/s]22716 examples [00:25, 839.49 examples/s]22806 examples [00:25, 854.59 examples/s]22893 examples [00:25, 844.04 examples/s]22983 examples [00:25, 859.15 examples/s]23074 examples [00:25, 872.97 examples/s]23170 examples [00:25, 896.31 examples/s]23265 examples [00:25, 909.94 examples/s]23359 examples [00:25, 917.13 examples/s]23455 examples [00:26, 927.39 examples/s]23554 examples [00:26, 943.85 examples/s]23653 examples [00:26, 956.04 examples/s]23749 examples [00:26, 948.43 examples/s]23844 examples [00:26, 934.55 examples/s]23943 examples [00:26, 948.49 examples/s]24039 examples [00:26, 922.34 examples/s]24132 examples [00:26, 900.69 examples/s]24226 examples [00:26, 910.29 examples/s]24322 examples [00:26, 921.97 examples/s]24418 examples [00:27, 921.49 examples/s]24513 examples [00:27, 929.47 examples/s]24610 examples [00:27, 938.12 examples/s]24706 examples [00:27, 943.34 examples/s]24801 examples [00:27, 940.37 examples/s]24898 examples [00:27, 947.86 examples/s]24993 examples [00:27, 918.63 examples/s]25086 examples [00:27, 870.65 examples/s]25176 examples [00:27, 877.80 examples/s]25266 examples [00:28, 882.36 examples/s]25361 examples [00:28, 901.60 examples/s]25457 examples [00:28, 917.61 examples/s]25550 examples [00:28, 893.28 examples/s]25640 examples [00:28, 892.94 examples/s]25734 examples [00:28, 906.24 examples/s]25825 examples [00:28, 883.81 examples/s]25919 examples [00:28, 899.58 examples/s]26016 examples [00:28, 917.31 examples/s]26109 examples [00:28, 914.48 examples/s]26204 examples [00:29, 923.77 examples/s]26302 examples [00:29, 937.64 examples/s]26398 examples [00:29, 944.04 examples/s]26494 examples [00:29, 947.05 examples/s]26591 examples [00:29, 950.19 examples/s]26687 examples [00:29, 947.90 examples/s]26784 examples [00:29, 951.28 examples/s]26882 examples [00:29, 958.43 examples/s]26979 examples [00:29, 959.41 examples/s]27078 examples [00:29, 958.23 examples/s]27174 examples [00:30, 920.55 examples/s]27267 examples [00:30, 903.87 examples/s]27360 examples [00:30, 910.87 examples/s]27455 examples [00:30, 922.15 examples/s]27548 examples [00:30, 900.14 examples/s]27643 examples [00:30, 912.74 examples/s]27742 examples [00:30, 933.38 examples/s]27840 examples [00:30, 946.02 examples/s]27937 examples [00:30, 950.57 examples/s]28033 examples [00:30, 951.26 examples/s]28133 examples [00:31, 962.64 examples/s]28230 examples [00:31, 959.49 examples/s]28328 examples [00:31, 964.45 examples/s]28425 examples [00:31, 964.73 examples/s]28522 examples [00:31, 951.99 examples/s]28619 examples [00:31, 955.11 examples/s]28715 examples [00:31, 940.17 examples/s]28813 examples [00:31, 951.05 examples/s]28909 examples [00:31, 943.80 examples/s]29004 examples [00:31, 940.62 examples/s]29108 examples [00:32, 966.89 examples/s]29205 examples [00:32, 959.06 examples/s]29302 examples [00:32, 949.91 examples/s]29399 examples [00:32, 955.10 examples/s]29495 examples [00:32, 944.69 examples/s]29590 examples [00:32, 935.05 examples/s]29684 examples [00:32, 920.98 examples/s]29777 examples [00:32, 919.33 examples/s]29870 examples [00:32, 920.23 examples/s]29963 examples [00:33, 917.25 examples/s]30055 examples [00:33, 862.44 examples/s]30148 examples [00:33, 881.48 examples/s]30247 examples [00:33, 910.79 examples/s]30340 examples [00:33, 915.90 examples/s]30433 examples [00:33, 895.63 examples/s]30529 examples [00:33, 911.96 examples/s]30626 examples [00:33, 927.90 examples/s]30721 examples [00:33, 932.96 examples/s]30815 examples [00:33, 916.83 examples/s]30909 examples [00:34, 922.14 examples/s]31006 examples [00:34, 932.98 examples/s]31100 examples [00:34, 922.75 examples/s]31193 examples [00:34, 911.51 examples/s]31288 examples [00:34, 921.48 examples/s]31381 examples [00:34, 922.17 examples/s]31477 examples [00:34, 933.01 examples/s]31575 examples [00:34, 946.18 examples/s]31670 examples [00:34, 942.33 examples/s]31765 examples [00:34, 914.63 examples/s]31857 examples [00:35, 915.96 examples/s]31952 examples [00:35, 922.95 examples/s]32050 examples [00:35, 937.29 examples/s]32144 examples [00:35, 931.61 examples/s]32238 examples [00:35, 903.69 examples/s]32329 examples [00:35, 904.32 examples/s]32426 examples [00:35, 913.22 examples/s]32521 examples [00:35, 922.34 examples/s]32616 examples [00:35, 927.77 examples/s]32709 examples [00:36, 928.29 examples/s]32807 examples [00:36, 941.33 examples/s]32905 examples [00:36, 952.42 examples/s]33002 examples [00:36, 956.48 examples/s]33098 examples [00:36, 946.45 examples/s]33193 examples [00:36, 942.24 examples/s]33288 examples [00:36, 934.33 examples/s]33382 examples [00:36, 935.49 examples/s]33476 examples [00:36, 920.43 examples/s]33570 examples [00:36, 924.85 examples/s]33665 examples [00:37, 930.73 examples/s]33760 examples [00:37, 934.67 examples/s]33857 examples [00:37, 944.62 examples/s]33953 examples [00:37, 946.45 examples/s]34048 examples [00:37, 943.75 examples/s]34146 examples [00:37, 951.87 examples/s]34242 examples [00:37, 943.98 examples/s]34337 examples [00:37, 928.75 examples/s]34433 examples [00:37, 935.74 examples/s]34527 examples [00:37, 932.16 examples/s]34621 examples [00:38, 921.41 examples/s]34714 examples [00:38, 920.58 examples/s]34807 examples [00:38, 913.60 examples/s]34903 examples [00:38, 925.95 examples/s]34999 examples [00:38, 935.77 examples/s]35093 examples [00:38, 930.22 examples/s]35191 examples [00:38, 943.66 examples/s]35287 examples [00:38, 945.77 examples/s]35384 examples [00:38, 950.31 examples/s]35480 examples [00:38, 939.69 examples/s]35575 examples [00:39, 938.74 examples/s]35669 examples [00:39, 937.04 examples/s]35763 examples [00:39, 931.46 examples/s]35858 examples [00:39, 936.68 examples/s]35953 examples [00:39, 940.18 examples/s]36050 examples [00:39, 948.09 examples/s]36145 examples [00:39, 939.91 examples/s]36240 examples [00:39, 938.22 examples/s]36338 examples [00:39, 949.94 examples/s]36434 examples [00:39, 926.62 examples/s]36527 examples [00:40, 877.99 examples/s]36616 examples [00:40, 869.92 examples/s]36708 examples [00:40, 883.43 examples/s]36797 examples [00:40, 877.44 examples/s]36886 examples [00:40, 874.33 examples/s]36977 examples [00:40, 883.29 examples/s]37066 examples [00:40, 874.83 examples/s]37161 examples [00:40, 895.81 examples/s]37255 examples [00:40, 907.53 examples/s]37346 examples [00:41, 898.17 examples/s]37439 examples [00:41, 904.29 examples/s]37532 examples [00:41, 909.31 examples/s]37627 examples [00:41, 919.82 examples/s]37720 examples [00:41, 920.14 examples/s]37816 examples [00:41, 929.00 examples/s]37912 examples [00:41, 937.29 examples/s]38006 examples [00:41, 918.05 examples/s]38102 examples [00:41, 927.80 examples/s]38197 examples [00:41, 932.61 examples/s]38291 examples [00:42, 933.36 examples/s]38385 examples [00:42, 931.44 examples/s]38479 examples [00:42, 927.29 examples/s]38573 examples [00:42, 929.31 examples/s]38666 examples [00:42, 912.72 examples/s]38761 examples [00:42, 922.15 examples/s]38854 examples [00:42, 911.41 examples/s]38946 examples [00:42, 907.23 examples/s]39038 examples [00:42, 908.43 examples/s]39133 examples [00:42, 918.44 examples/s]39231 examples [00:43, 934.52 examples/s]39325 examples [00:43, 933.34 examples/s]39419 examples [00:43, 894.56 examples/s]39509 examples [00:43, 873.09 examples/s]39599 examples [00:43, 879.87 examples/s]39692 examples [00:43, 892.87 examples/s]39786 examples [00:43, 906.42 examples/s]39877 examples [00:43, 906.20 examples/s]39970 examples [00:43, 913.20 examples/s]40062 examples [00:44, 852.62 examples/s]40156 examples [00:44, 877.06 examples/s]40246 examples [00:44, 883.07 examples/s]40337 examples [00:44, 888.17 examples/s]40428 examples [00:44, 892.04 examples/s]40519 examples [00:44, 897.19 examples/s]40615 examples [00:44, 913.57 examples/s]40707 examples [00:44, 912.25 examples/s]40799 examples [00:44, 912.57 examples/s]40892 examples [00:44, 915.98 examples/s]40984 examples [00:45, 879.12 examples/s]41077 examples [00:45, 891.36 examples/s]41169 examples [00:45, 897.30 examples/s]41260 examples [00:45, 900.83 examples/s]41353 examples [00:45, 907.56 examples/s]41444 examples [00:45, 891.26 examples/s]41536 examples [00:45, 897.68 examples/s]41626 examples [00:45, 891.11 examples/s]41716 examples [00:45, 890.19 examples/s]41809 examples [00:45, 900.83 examples/s]41901 examples [00:46, 906.30 examples/s]41993 examples [00:46, 909.96 examples/s]42085 examples [00:46, 899.50 examples/s]42176 examples [00:46, 900.74 examples/s]42268 examples [00:46, 903.48 examples/s]42359 examples [00:46, 896.17 examples/s]42451 examples [00:46, 901.18 examples/s]42542 examples [00:46, 892.45 examples/s]42632 examples [00:46, 889.01 examples/s]42721 examples [00:46, 884.78 examples/s]42811 examples [00:47, 888.50 examples/s]42903 examples [00:47, 895.53 examples/s]42993 examples [00:47, 893.50 examples/s]43086 examples [00:47, 901.52 examples/s]43177 examples [00:47, 902.07 examples/s]43268 examples [00:47, 899.92 examples/s]43359 examples [00:47, 902.76 examples/s]43452 examples [00:47, 910.74 examples/s]43546 examples [00:47, 917.65 examples/s]43638 examples [00:47, 916.64 examples/s]43730 examples [00:48, 915.50 examples/s]43825 examples [00:48, 922.81 examples/s]43918 examples [00:48, 924.04 examples/s]44012 examples [00:48, 926.68 examples/s]44105 examples [00:48, 892.87 examples/s]44195 examples [00:48, 865.36 examples/s]44289 examples [00:48, 884.80 examples/s]44381 examples [00:48, 892.75 examples/s]44472 examples [00:48, 895.67 examples/s]44565 examples [00:48, 904.62 examples/s]44656 examples [00:49, 904.19 examples/s]44748 examples [00:49, 907.64 examples/s]44839 examples [00:49, 908.10 examples/s]44933 examples [00:49, 916.95 examples/s]45025 examples [00:49, 906.42 examples/s]45117 examples [00:49, 909.06 examples/s]45212 examples [00:49, 919.77 examples/s]45305 examples [00:49, 909.00 examples/s]45396 examples [00:49, 899.60 examples/s]45487 examples [00:50, 868.40 examples/s]45579 examples [00:50, 882.05 examples/s]45669 examples [00:50, 886.07 examples/s]45760 examples [00:50, 891.44 examples/s]45852 examples [00:50, 897.03 examples/s]45942 examples [00:50, 884.55 examples/s]46037 examples [00:50, 903.10 examples/s]46138 examples [00:50, 932.10 examples/s]46232 examples [00:50, 924.01 examples/s]46326 examples [00:50, 926.71 examples/s]46419 examples [00:51, 915.60 examples/s]46512 examples [00:51, 918.77 examples/s]46604 examples [00:51, 869.10 examples/s]46692 examples [00:51, 871.87 examples/s]46782 examples [00:51, 879.95 examples/s]46874 examples [00:51, 889.15 examples/s]46967 examples [00:51, 898.45 examples/s]47059 examples [00:51, 902.97 examples/s]47150 examples [00:51, 896.93 examples/s]47241 examples [00:51, 898.48 examples/s]47335 examples [00:52, 908.50 examples/s]47431 examples [00:52, 921.36 examples/s]47525 examples [00:52, 925.53 examples/s]47618 examples [00:52, 910.60 examples/s]47710 examples [00:52, 879.41 examples/s]47799 examples [00:52, 874.33 examples/s]47890 examples [00:52, 883.79 examples/s]47979 examples [00:52, 873.05 examples/s]48072 examples [00:52, 887.15 examples/s]48161 examples [00:53, 887.28 examples/s]48254 examples [00:53, 898.64 examples/s]48347 examples [00:53, 906.40 examples/s]48440 examples [00:53, 911.25 examples/s]48534 examples [00:53, 916.97 examples/s]48636 examples [00:53, 942.87 examples/s]48731 examples [00:53, 932.59 examples/s]48825 examples [00:53, 905.10 examples/s]48916 examples [00:53, 901.64 examples/s]49008 examples [00:53, 904.04 examples/s]49099 examples [00:54, 892.07 examples/s]49190 examples [00:54, 896.04 examples/s]49282 examples [00:54, 901.43 examples/s]49374 examples [00:54, 904.69 examples/s]49465 examples [00:54, 902.86 examples/s]49556 examples [00:54, 904.68 examples/s]49650 examples [00:54, 914.33 examples/s]49746 examples [00:54, 926.12 examples/s]49839 examples [00:54, 920.82 examples/s]49932 examples [00:54, 910.49 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 10%|█         | 5231/50000 [00:00<00:00, 52306.56 examples/s] 30%|███       | 15191/50000 [00:00<00:00, 60993.31 examples/s] 50%|████▉     | 24920/50000 [00:00<00:00, 68680.09 examples/s] 71%|███████   | 35351/50000 [00:00<00:00, 76521.34 examples/s] 90%|█████████ | 45094/50000 [00:00<00:00, 81784.59 examples/s]                                                               0 examples [00:00, ? examples/s]71 examples [00:00, 703.24 examples/s]170 examples [00:00, 768.47 examples/s]268 examples [00:00, 820.43 examples/s]365 examples [00:00, 860.01 examples/s]460 examples [00:00, 883.53 examples/s]558 examples [00:00, 908.56 examples/s]654 examples [00:00, 923.14 examples/s]742 examples [00:00, 904.09 examples/s]830 examples [00:00, 888.73 examples/s]922 examples [00:01, 896.40 examples/s]1016 examples [00:01, 908.56 examples/s]1108 examples [00:01, 910.18 examples/s]1207 examples [00:01, 930.14 examples/s]1300 examples [00:01, 927.94 examples/s]1393 examples [00:01, 907.83 examples/s]1490 examples [00:01, 923.43 examples/s]1591 examples [00:01, 945.81 examples/s]1686 examples [00:01, 937.81 examples/s]1780 examples [00:01, 926.01 examples/s]1873 examples [00:02, 926.99 examples/s]1966 examples [00:02, 895.12 examples/s]2056 examples [00:02, 884.91 examples/s]2150 examples [00:02, 898.51 examples/s]2241 examples [00:02, 889.03 examples/s]2331 examples [00:02, 880.70 examples/s]2423 examples [00:02, 890.02 examples/s]2519 examples [00:02, 908.10 examples/s]2614 examples [00:02, 918.30 examples/s]2706 examples [00:02, 900.01 examples/s]2797 examples [00:03, 882.51 examples/s]2886 examples [00:03, 876.10 examples/s]2974 examples [00:03, 841.72 examples/s]3059 examples [00:03, 841.03 examples/s]3144 examples [00:03, 842.68 examples/s]3236 examples [00:03, 862.86 examples/s]3331 examples [00:03, 885.43 examples/s]3424 examples [00:03, 896.11 examples/s]3514 examples [00:03, 877.32 examples/s]3607 examples [00:04, 890.52 examples/s]3703 examples [00:04, 908.25 examples/s]3795 examples [00:04, 900.35 examples/s]3888 examples [00:04, 907.04 examples/s]3979 examples [00:04, 904.70 examples/s]4070 examples [00:04, 905.09 examples/s]4163 examples [00:04, 912.03 examples/s]4255 examples [00:04, 900.06 examples/s]4349 examples [00:04, 911.28 examples/s]4441 examples [00:04, 904.96 examples/s]4534 examples [00:05, 910.10 examples/s]4626 examples [00:05, 912.71 examples/s]4720 examples [00:05, 919.92 examples/s]4813 examples [00:05, 919.62 examples/s]4907 examples [00:05, 921.81 examples/s]5004 examples [00:05, 934.85 examples/s]5098 examples [00:05, 930.73 examples/s]5192 examples [00:05, 931.23 examples/s]5291 examples [00:05, 946.94 examples/s]5386 examples [00:05, 909.25 examples/s]5478 examples [00:06, 898.11 examples/s]5569 examples [00:06, 900.50 examples/s]5665 examples [00:06, 915.35 examples/s]5761 examples [00:06, 926.42 examples/s]5856 examples [00:06, 932.41 examples/s]5950 examples [00:06, 933.19 examples/s]6047 examples [00:06, 942.09 examples/s]6144 examples [00:06, 947.98 examples/s]6242 examples [00:06, 954.26 examples/s]6338 examples [00:06, 954.57 examples/s]6434 examples [00:07, 942.33 examples/s]6529 examples [00:07, 937.61 examples/s]6623 examples [00:07, 925.20 examples/s]6716 examples [00:07, 913.77 examples/s]6808 examples [00:07, 911.70 examples/s]6900 examples [00:07, 905.39 examples/s]6991 examples [00:07, 900.42 examples/s]7084 examples [00:07, 908.34 examples/s]7176 examples [00:07, 909.42 examples/s]7270 examples [00:07, 917.36 examples/s]7362 examples [00:08, 917.98 examples/s]7454 examples [00:08, 916.03 examples/s]7547 examples [00:08, 918.27 examples/s]7639 examples [00:08, 917.17 examples/s]7731 examples [00:08, 910.93 examples/s]7823 examples [00:08, 880.81 examples/s]7914 examples [00:08, 886.29 examples/s]8003 examples [00:08, 869.29 examples/s]8091 examples [00:08, 838.41 examples/s]8176 examples [00:09, 838.27 examples/s]8265 examples [00:09, 852.19 examples/s]8357 examples [00:09, 869.57 examples/s]8451 examples [00:09, 888.05 examples/s]8541 examples [00:09, 850.06 examples/s]8627 examples [00:09, 830.38 examples/s]8719 examples [00:09, 855.07 examples/s]8810 examples [00:09, 869.14 examples/s]8901 examples [00:09, 878.34 examples/s]8990 examples [00:09, 859.82 examples/s]9079 examples [00:10, 867.66 examples/s]9166 examples [00:10, 868.26 examples/s]9256 examples [00:10, 876.26 examples/s]9347 examples [00:10, 885.36 examples/s]9436 examples [00:10, 870.49 examples/s]9524 examples [00:10, 843.99 examples/s]9620 examples [00:10, 874.86 examples/s]9713 examples [00:10, 888.12 examples/s]9808 examples [00:10, 903.44 examples/s]9899 examples [00:10, 901.65 examples/s]9990 examples [00:11, 885.21 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 94%|█████████▍| 9427/10000 [00:00<00:00, 94265.59 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete8J4NVO/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete8J4NVO/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3f12a9d158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3f12a9d158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3f12a9d158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3f5e47a0b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3ed0dae8d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3f12a9d158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3f12a9d158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3f12a9d158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3f10c48358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3f10c481d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 10.82 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 14.67 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 14.67 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.07 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.07 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.87 file/s]2020-08-13 18:08:41.810873: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-13 18:08:41.814971: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-13 18:08:41.815134: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56499967ab30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-13 18:08:41.815151: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 156226.79it/s] 87%|████████▋ | 8650752/9912422 [00:00<00:05, 223008.17it/s]9920512it [00:00, 45513420.24it/s]                           
0it [00:00, ?it/s]32768it [00:00, 489263.30it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 148852.03it/s]1654784it [00:00, 10835328.44it/s]                         
0it [00:00, ?it/s]8192it [00:00, 166673.48it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

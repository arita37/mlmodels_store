
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f8426fc4ae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f8426fc4ae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f8491c34510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f8491c34510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f84b0e61ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f84b0e61ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f843efdca60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f843efdca60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f843efdca60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 18%|█▊        | 1753088/9912422 [00:00<00:00, 17425743.16it/s] 78%|███████▊  | 7766016/9912422 [00:00<00:00, 22143497.78it/s]9920512it [00:00, 27291955.70it/s]                             
0it [00:00, ?it/s]32768it [00:00, 707492.73it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 448879.93it/s]1654784it [00:00, 10422908.90it/s]                         
0it [00:00, ?it/s]8192it [00:00, 246615.74it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8426e80630>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8426e908d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f843efdc6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f843efdc6a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f843efdc6a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:24,  1.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:24,  1.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:23,  1.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:23,  1.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:22,  1.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:22,  1.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:21,  1.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:57,  2.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:57,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:57,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:56,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:56,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:56,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:55,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:55,  2.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:39,  3.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:39,  3.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:38,  3.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:38,  3.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:38,  3.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:37,  3.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:37,  3.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:26,  5.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:26,  5.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:25,  5.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:25,  5.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:25,  5.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:25,  5.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:18,  7.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:18,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:17,  7.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:17,  7.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:17,  7.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:17,  7.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:12, 10.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:12, 10.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:12, 10.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:12, 10.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:12, 10.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 17.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 17.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 17.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 17.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 17.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 17.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 17.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 17.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 17.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 23.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 23.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 23.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 23.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 23.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 23.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 23.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 29.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 29.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 29.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 29.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 29.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 29.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 29.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 29.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 29.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 35.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 35.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 35.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 35.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 35.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 35.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 35.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 35.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 41.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 41.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 41.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 41.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 41.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 41.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 41.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 41.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 46.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 46.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 46.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 46.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 46.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 46.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 50.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 50.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 50.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 50.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 50.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 50.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 50.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 50.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 52.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 52.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 52.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 52.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 52.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 52.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 56.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 56.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 56.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 56.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 56.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 56.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 56.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 56.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 56.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 63.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 63.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 63.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 63.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 63.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 63.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 63.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 63.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 63.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 63.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 63.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 63.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 63.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 63.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 63.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 63.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 62.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 62.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 62.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 64.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.88s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.88s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.88s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 64.27 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.25s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.88s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 64.27 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.25s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.25s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.86 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.25s/ url]
0 examples [00:00, ? examples/s]2020-07-28 00:11:03.801511: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-28 00:11:03.864677: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-28 00:11:03.910350: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c46942f090 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-28 00:11:03.910390: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.48 examples/s]79 examples [00:00,  3.53 examples/s]170 examples [00:00,  5.04 examples/s]262 examples [00:00,  7.18 examples/s]354 examples [00:00, 10.22 examples/s]442 examples [00:00, 14.53 examples/s]534 examples [00:01, 20.61 examples/s]627 examples [00:01, 29.16 examples/s]712 examples [00:01, 41.06 examples/s]800 examples [00:01, 57.49 examples/s]895 examples [00:01, 80.04 examples/s]990 examples [00:01, 110.33 examples/s]1087 examples [00:01, 150.23 examples/s]1184 examples [00:01, 201.17 examples/s]1277 examples [00:01, 261.99 examples/s]1369 examples [00:01, 332.44 examples/s]1462 examples [00:02, 411.76 examples/s]1554 examples [00:02, 486.55 examples/s]1648 examples [00:02, 568.54 examples/s]1739 examples [00:02, 638.24 examples/s]1831 examples [00:02, 702.24 examples/s]1925 examples [00:02, 759.05 examples/s]2024 examples [00:02, 815.50 examples/s]2120 examples [00:02, 853.60 examples/s]2215 examples [00:02, 873.65 examples/s]2309 examples [00:02, 883.37 examples/s]2405 examples [00:03, 902.64 examples/s]2500 examples [00:03, 914.72 examples/s]2598 examples [00:03, 932.54 examples/s]2693 examples [00:03, 936.62 examples/s]2788 examples [00:03, 933.31 examples/s]2883 examples [00:03, 931.13 examples/s]2980 examples [00:03, 940.03 examples/s]3075 examples [00:03, 934.40 examples/s]3171 examples [00:03, 939.94 examples/s]3266 examples [00:03, 928.95 examples/s]3360 examples [00:04, 932.22 examples/s]3454 examples [00:04, 933.59 examples/s]3548 examples [00:04, 913.52 examples/s]3640 examples [00:04, 881.81 examples/s]3729 examples [00:04, 878.21 examples/s]3821 examples [00:04, 888.91 examples/s]3915 examples [00:04, 902.47 examples/s]4009 examples [00:04, 912.22 examples/s]4101 examples [00:04, 913.44 examples/s]4193 examples [00:04, 912.88 examples/s]4285 examples [00:05, 911.52 examples/s]4377 examples [00:05, 913.12 examples/s]4469 examples [00:05, 912.79 examples/s]4561 examples [00:05, 905.07 examples/s]4652 examples [00:05, 894.27 examples/s]4742 examples [00:05, 893.48 examples/s]4833 examples [00:05, 897.70 examples/s]4923 examples [00:05, 880.27 examples/s]5012 examples [00:05, 878.66 examples/s]5101 examples [00:06, 880.04 examples/s]5195 examples [00:06, 895.46 examples/s]5290 examples [00:06, 908.42 examples/s]5386 examples [00:06, 921.63 examples/s]5481 examples [00:06, 929.75 examples/s]5575 examples [00:06, 901.28 examples/s]5666 examples [00:06, 892.35 examples/s]5764 examples [00:06, 914.88 examples/s]5856 examples [00:06, 909.01 examples/s]5948 examples [00:06, 884.24 examples/s]6037 examples [00:07, 880.77 examples/s]6129 examples [00:07, 889.01 examples/s]6221 examples [00:07, 895.84 examples/s]6316 examples [00:07, 908.96 examples/s]6408 examples [00:07, 903.58 examples/s]6499 examples [00:07, 898.82 examples/s]6592 examples [00:07, 907.16 examples/s]6685 examples [00:07, 911.72 examples/s]6777 examples [00:07, 905.50 examples/s]6868 examples [00:07, 906.65 examples/s]6959 examples [00:08, 905.40 examples/s]7050 examples [00:08, 906.63 examples/s]7142 examples [00:08, 910.42 examples/s]7239 examples [00:08, 925.99 examples/s]7333 examples [00:08, 925.65 examples/s]7426 examples [00:08, 909.94 examples/s]7521 examples [00:08, 920.75 examples/s]7617 examples [00:08, 930.70 examples/s]7711 examples [00:08, 930.24 examples/s]7806 examples [00:08, 933.67 examples/s]7900 examples [00:09, 924.11 examples/s]7993 examples [00:09, 897.57 examples/s]8083 examples [00:09, 885.14 examples/s]8172 examples [00:09, 865.98 examples/s]8259 examples [00:09, 862.37 examples/s]8346 examples [00:09, 853.70 examples/s]8439 examples [00:09, 874.93 examples/s]8532 examples [00:09, 889.83 examples/s]8623 examples [00:09, 893.79 examples/s]8714 examples [00:09, 898.04 examples/s]8804 examples [00:10, 873.65 examples/s]8892 examples [00:10, 864.82 examples/s]8979 examples [00:10, 848.10 examples/s]9071 examples [00:10, 865.20 examples/s]9164 examples [00:10, 880.08 examples/s]9259 examples [00:10, 897.41 examples/s]9353 examples [00:10, 909.77 examples/s]9448 examples [00:10, 919.58 examples/s]9541 examples [00:10, 919.80 examples/s]9634 examples [00:11, 920.60 examples/s]9727 examples [00:11, 920.37 examples/s]9821 examples [00:11, 924.21 examples/s]9914 examples [00:11, 904.25 examples/s]10005 examples [00:11, 829.28 examples/s]10090 examples [00:11, 826.76 examples/s]10177 examples [00:11, 839.19 examples/s]10266 examples [00:11, 851.88 examples/s]10359 examples [00:11, 872.91 examples/s]10454 examples [00:11, 894.32 examples/s]10544 examples [00:12, 893.71 examples/s]10642 examples [00:12, 916.12 examples/s]10737 examples [00:12, 924.57 examples/s]10830 examples [00:12, 922.95 examples/s]10923 examples [00:12, 923.76 examples/s]11016 examples [00:12, 899.07 examples/s]11108 examples [00:12, 904.36 examples/s]11201 examples [00:12, 911.19 examples/s]11293 examples [00:12, 906.54 examples/s]11386 examples [00:12, 913.35 examples/s]11478 examples [00:13, 903.37 examples/s]11572 examples [00:13, 911.94 examples/s]11664 examples [00:13, 885.84 examples/s]11758 examples [00:13, 900.74 examples/s]11852 examples [00:13, 910.93 examples/s]11946 examples [00:13, 916.03 examples/s]12044 examples [00:13, 933.27 examples/s]12139 examples [00:13, 936.32 examples/s]12233 examples [00:13, 925.76 examples/s]12328 examples [00:14, 930.31 examples/s]12422 examples [00:14, 919.60 examples/s]12515 examples [00:14, 913.42 examples/s]12607 examples [00:14, 906.50 examples/s]12698 examples [00:14, 898.86 examples/s]12788 examples [00:14, 889.81 examples/s]12880 examples [00:14, 897.15 examples/s]12976 examples [00:14, 914.59 examples/s]13070 examples [00:14, 920.85 examples/s]13163 examples [00:14, 888.58 examples/s]13254 examples [00:15, 894.49 examples/s]13344 examples [00:15, 895.09 examples/s]13437 examples [00:15, 904.08 examples/s]13530 examples [00:15, 911.01 examples/s]13622 examples [00:15, 911.10 examples/s]13716 examples [00:15, 917.99 examples/s]13808 examples [00:15, 906.67 examples/s]13900 examples [00:15, 909.83 examples/s]13992 examples [00:15, 906.31 examples/s]14083 examples [00:15, 905.18 examples/s]14174 examples [00:16, 904.32 examples/s]14265 examples [00:16, 890.85 examples/s]14359 examples [00:16, 904.33 examples/s]14451 examples [00:16, 908.10 examples/s]14548 examples [00:16, 923.95 examples/s]14641 examples [00:16, 907.19 examples/s]14732 examples [00:16, 900.57 examples/s]14825 examples [00:16, 908.10 examples/s]14918 examples [00:16, 913.49 examples/s]15011 examples [00:16, 916.42 examples/s]15106 examples [00:17, 926.15 examples/s]15199 examples [00:17, 919.97 examples/s]15297 examples [00:17, 934.56 examples/s]15394 examples [00:17, 943.75 examples/s]15489 examples [00:17, 937.81 examples/s]15583 examples [00:17, 915.72 examples/s]15675 examples [00:17, 913.24 examples/s]15767 examples [00:17, 899.57 examples/s]15858 examples [00:17, 898.72 examples/s]15951 examples [00:18, 907.10 examples/s]16042 examples [00:18, 905.18 examples/s]16134 examples [00:18, 908.09 examples/s]16228 examples [00:18, 917.41 examples/s]16320 examples [00:18, 908.85 examples/s]16417 examples [00:18, 925.19 examples/s]16511 examples [00:18, 926.96 examples/s]16604 examples [00:18, 917.06 examples/s]16699 examples [00:18, 923.73 examples/s]16792 examples [00:18, 921.40 examples/s]16885 examples [00:19, 920.19 examples/s]16978 examples [00:19, 923.07 examples/s]17071 examples [00:19, 912.85 examples/s]17163 examples [00:19, 901.88 examples/s]17254 examples [00:19, 901.23 examples/s]17346 examples [00:19, 906.14 examples/s]17437 examples [00:19, 903.87 examples/s]17528 examples [00:19, 894.08 examples/s]17619 examples [00:19, 898.18 examples/s]17709 examples [00:19, 897.34 examples/s]17801 examples [00:20, 902.37 examples/s]17892 examples [00:20, 897.44 examples/s]17984 examples [00:20, 902.06 examples/s]18075 examples [00:20, 900.53 examples/s]18166 examples [00:20, 881.77 examples/s]18257 examples [00:20, 888.64 examples/s]18352 examples [00:20, 906.16 examples/s]18446 examples [00:20, 914.51 examples/s]18541 examples [00:20, 924.63 examples/s]18634 examples [00:20, 920.71 examples/s]18727 examples [00:21, 921.55 examples/s]18820 examples [00:21, 917.04 examples/s]18912 examples [00:21, 899.52 examples/s]19003 examples [00:21, 886.25 examples/s]19095 examples [00:21, 894.06 examples/s]19188 examples [00:21, 903.32 examples/s]19281 examples [00:21, 910.88 examples/s]19373 examples [00:21, 881.26 examples/s]19464 examples [00:21, 885.88 examples/s]19560 examples [00:21, 903.61 examples/s]19651 examples [00:22, 880.53 examples/s]19742 examples [00:22, 887.01 examples/s]19833 examples [00:22, 893.27 examples/s]19923 examples [00:22, 891.85 examples/s]20013 examples [00:22, 840.33 examples/s]20108 examples [00:22, 870.36 examples/s]20202 examples [00:22, 887.46 examples/s]20292 examples [00:22, 861.45 examples/s]20384 examples [00:22, 876.47 examples/s]20476 examples [00:23, 887.77 examples/s]20570 examples [00:23, 900.15 examples/s]20663 examples [00:23, 907.76 examples/s]20755 examples [00:23, 909.89 examples/s]20847 examples [00:23, 910.25 examples/s]20940 examples [00:23, 915.32 examples/s]21035 examples [00:23, 925.26 examples/s]21128 examples [00:23, 914.40 examples/s]21220 examples [00:23, 899.48 examples/s]21311 examples [00:23, 898.22 examples/s]21401 examples [00:24, 896.93 examples/s]21491 examples [00:24, 894.00 examples/s]21581 examples [00:24, 893.35 examples/s]21671 examples [00:24, 890.14 examples/s]21761 examples [00:24, 857.74 examples/s]21854 examples [00:24, 877.84 examples/s]21943 examples [00:24, 869.97 examples/s]22036 examples [00:24, 884.81 examples/s]22125 examples [00:24, 862.13 examples/s]22213 examples [00:24, 866.63 examples/s]22302 examples [00:25, 871.64 examples/s]22392 examples [00:25, 877.17 examples/s]22480 examples [00:25, 875.99 examples/s]22568 examples [00:25, 873.07 examples/s]22656 examples [00:25, 859.98 examples/s]22749 examples [00:25, 879.14 examples/s]22842 examples [00:25, 892.06 examples/s]22935 examples [00:25, 901.65 examples/s]23026 examples [00:25, 885.81 examples/s]23119 examples [00:25, 896.08 examples/s]23210 examples [00:26, 898.91 examples/s]23305 examples [00:26, 912.31 examples/s]23403 examples [00:26, 929.16 examples/s]23499 examples [00:26, 937.70 examples/s]23593 examples [00:26, 916.01 examples/s]23685 examples [00:26, 885.45 examples/s]23774 examples [00:26, 878.42 examples/s]23863 examples [00:26, 870.78 examples/s]23953 examples [00:26, 876.35 examples/s]24043 examples [00:27, 883.00 examples/s]24138 examples [00:27, 900.67 examples/s]24231 examples [00:27, 908.36 examples/s]24322 examples [00:27, 901.12 examples/s]24413 examples [00:27, 881.18 examples/s]24506 examples [00:27, 894.06 examples/s]24603 examples [00:27, 914.92 examples/s]24698 examples [00:27, 923.60 examples/s]24791 examples [00:27, 923.78 examples/s]24885 examples [00:27, 926.24 examples/s]24981 examples [00:28, 933.21 examples/s]25075 examples [00:28, 932.66 examples/s]25169 examples [00:28, 922.99 examples/s]25262 examples [00:28, 921.74 examples/s]25355 examples [00:28, 899.72 examples/s]25450 examples [00:28, 911.28 examples/s]25542 examples [00:28, 886.30 examples/s]25634 examples [00:28, 895.87 examples/s]25729 examples [00:28, 909.09 examples/s]25821 examples [00:28, 891.32 examples/s]25916 examples [00:29, 906.80 examples/s]26008 examples [00:29, 908.79 examples/s]26100 examples [00:29, 900.64 examples/s]26191 examples [00:29, 874.16 examples/s]26279 examples [00:29, 865.80 examples/s]26376 examples [00:29, 893.31 examples/s]26475 examples [00:29, 916.94 examples/s]26573 examples [00:29, 933.18 examples/s]26667 examples [00:29, 921.65 examples/s]26761 examples [00:30, 927.00 examples/s]26854 examples [00:30, 921.09 examples/s]26950 examples [00:30, 931.16 examples/s]27045 examples [00:30, 934.68 examples/s]27140 examples [00:30, 938.62 examples/s]27234 examples [00:30, 934.82 examples/s]27330 examples [00:30, 940.21 examples/s]27425 examples [00:30, 939.45 examples/s]27519 examples [00:30, 939.07 examples/s]27613 examples [00:30, 913.54 examples/s]27705 examples [00:31, 890.58 examples/s]27795 examples [00:31, 872.08 examples/s]27883 examples [00:31, 871.47 examples/s]27971 examples [00:31, 870.33 examples/s]28059 examples [00:31, 867.49 examples/s]28148 examples [00:31, 873.75 examples/s]28243 examples [00:31, 893.70 examples/s]28337 examples [00:31, 905.91 examples/s]28432 examples [00:31, 916.61 examples/s]28524 examples [00:31, 912.73 examples/s]28616 examples [00:32, 905.70 examples/s]28707 examples [00:32, 895.50 examples/s]28797 examples [00:32, 875.59 examples/s]28885 examples [00:32, 853.31 examples/s]28974 examples [00:32, 863.55 examples/s]29064 examples [00:32, 871.36 examples/s]29157 examples [00:32, 887.34 examples/s]29248 examples [00:32, 888.47 examples/s]29338 examples [00:32, 891.61 examples/s]29428 examples [00:32, 884.32 examples/s]29517 examples [00:33, 883.89 examples/s]29607 examples [00:33, 888.20 examples/s]29700 examples [00:33, 898.39 examples/s]29790 examples [00:33, 898.67 examples/s]29880 examples [00:33, 895.43 examples/s]29970 examples [00:33, 891.42 examples/s]30060 examples [00:33, 843.01 examples/s]30157 examples [00:33, 875.95 examples/s]30255 examples [00:33, 903.23 examples/s]30349 examples [00:34, 912.65 examples/s]30444 examples [00:34, 921.51 examples/s]30537 examples [00:34, 911.99 examples/s]30629 examples [00:34, 909.24 examples/s]30721 examples [00:34, 899.14 examples/s]30812 examples [00:34, 889.68 examples/s]30902 examples [00:34, 876.34 examples/s]30993 examples [00:34, 883.82 examples/s]31082 examples [00:34, 885.51 examples/s]31171 examples [00:34, 882.88 examples/s]31260 examples [00:35, 883.97 examples/s]31349 examples [00:35, 884.04 examples/s]31441 examples [00:35, 891.98 examples/s]31531 examples [00:35, 889.82 examples/s]31622 examples [00:35, 895.76 examples/s]31712 examples [00:35, 887.06 examples/s]31805 examples [00:35, 898.12 examples/s]31895 examples [00:35, 881.24 examples/s]31984 examples [00:35, 870.08 examples/s]32072 examples [00:35, 851.73 examples/s]32158 examples [00:36, 852.33 examples/s]32253 examples [00:36, 877.44 examples/s]32347 examples [00:36, 893.31 examples/s]32438 examples [00:36, 895.96 examples/s]32528 examples [00:36, 894.95 examples/s]32618 examples [00:36, 894.22 examples/s]32708 examples [00:36, 893.71 examples/s]32800 examples [00:36, 898.98 examples/s]32892 examples [00:36, 904.97 examples/s]32983 examples [00:36, 906.40 examples/s]33074 examples [00:37, 904.23 examples/s]33165 examples [00:37, 901.55 examples/s]33258 examples [00:37, 909.59 examples/s]33349 examples [00:37, 879.48 examples/s]33438 examples [00:37, 880.91 examples/s]33528 examples [00:37, 884.91 examples/s]33617 examples [00:37, 884.41 examples/s]33709 examples [00:37, 894.34 examples/s]33799 examples [00:37, 886.15 examples/s]33888 examples [00:37, 881.51 examples/s]33977 examples [00:38, 883.10 examples/s]34069 examples [00:38, 892.52 examples/s]34161 examples [00:38, 898.38 examples/s]34254 examples [00:38, 905.31 examples/s]34345 examples [00:38, 894.99 examples/s]34435 examples [00:38, 895.17 examples/s]34526 examples [00:38, 899.41 examples/s]34620 examples [00:38, 910.61 examples/s]34712 examples [00:38, 901.34 examples/s]34803 examples [00:39, 892.80 examples/s]34893 examples [00:39, 893.32 examples/s]34983 examples [00:39, 892.04 examples/s]35073 examples [00:39, 873.51 examples/s]35161 examples [00:39, 862.43 examples/s]35248 examples [00:39, 858.09 examples/s]35338 examples [00:39, 869.91 examples/s]35430 examples [00:39, 883.66 examples/s]35519 examples [00:39, 854.73 examples/s]35605 examples [00:39, 848.43 examples/s]35691 examples [00:40, 843.93 examples/s]35776 examples [00:40, 843.31 examples/s]35865 examples [00:40, 854.95 examples/s]35956 examples [00:40, 868.58 examples/s]36050 examples [00:40, 885.71 examples/s]36139 examples [00:40, 855.94 examples/s]36225 examples [00:40, 849.66 examples/s]36313 examples [00:40, 856.79 examples/s]36404 examples [00:40, 869.60 examples/s]36492 examples [00:40, 872.39 examples/s]36580 examples [00:41, 873.86 examples/s]36669 examples [00:41, 877.12 examples/s]36762 examples [00:41, 891.17 examples/s]36854 examples [00:41, 897.94 examples/s]36947 examples [00:41, 906.91 examples/s]37039 examples [00:41, 908.19 examples/s]37132 examples [00:41, 913.81 examples/s]37224 examples [00:41, 913.88 examples/s]37316 examples [00:41, 899.49 examples/s]37410 examples [00:41, 908.70 examples/s]37501 examples [00:42, 897.31 examples/s]37591 examples [00:42, 886.91 examples/s]37683 examples [00:42, 894.22 examples/s]37774 examples [00:42, 897.27 examples/s]37864 examples [00:42, 893.86 examples/s]37957 examples [00:42, 902.15 examples/s]38048 examples [00:42, 892.75 examples/s]38138 examples [00:42, 889.68 examples/s]38228 examples [00:42, 891.21 examples/s]38323 examples [00:43, 905.93 examples/s]38418 examples [00:43, 916.36 examples/s]38514 examples [00:43, 925.53 examples/s]38609 examples [00:43, 928.48 examples/s]38708 examples [00:43, 944.43 examples/s]38803 examples [00:43, 932.85 examples/s]38897 examples [00:43, 928.22 examples/s]38992 examples [00:43, 933.38 examples/s]39090 examples [00:43, 945.22 examples/s]39185 examples [00:43, 929.71 examples/s]39279 examples [00:44, 926.55 examples/s]39372 examples [00:44, 919.44 examples/s]39465 examples [00:44, 915.27 examples/s]39557 examples [00:44, 908.78 examples/s]39649 examples [00:44, 909.57 examples/s]39740 examples [00:44, 905.40 examples/s]39832 examples [00:44, 908.98 examples/s]39926 examples [00:44, 916.79 examples/s]40018 examples [00:44, 861.14 examples/s]40108 examples [00:44, 871.12 examples/s]40201 examples [00:45, 886.34 examples/s]40295 examples [00:45, 899.93 examples/s]40392 examples [00:45, 917.29 examples/s]40485 examples [00:45, 920.57 examples/s]40580 examples [00:45, 927.30 examples/s]40675 examples [00:45, 933.82 examples/s]40769 examples [00:45, 933.46 examples/s]40863 examples [00:45, 922.71 examples/s]40956 examples [00:45, 908.31 examples/s]41047 examples [00:45, 867.75 examples/s]41140 examples [00:46, 883.57 examples/s]41235 examples [00:46, 900.43 examples/s]41330 examples [00:46, 913.39 examples/s]41427 examples [00:46, 927.75 examples/s]41521 examples [00:46, 910.62 examples/s]41613 examples [00:46, 889.86 examples/s]41703 examples [00:46, 884.36 examples/s]41797 examples [00:46, 898.54 examples/s]41888 examples [00:46, 870.74 examples/s]41976 examples [00:47, 866.98 examples/s]42063 examples [00:47, 839.25 examples/s]42150 examples [00:47, 847.00 examples/s]42241 examples [00:47, 863.07 examples/s]42331 examples [00:47, 872.04 examples/s]42426 examples [00:47, 891.52 examples/s]42517 examples [00:47, 894.80 examples/s]42611 examples [00:47, 905.60 examples/s]42703 examples [00:47, 907.44 examples/s]42797 examples [00:47, 914.76 examples/s]42891 examples [00:48, 922.14 examples/s]42986 examples [00:48, 927.95 examples/s]43083 examples [00:48, 938.46 examples/s]43180 examples [00:48, 946.88 examples/s]43275 examples [00:48, 946.98 examples/s]43372 examples [00:48, 951.61 examples/s]43468 examples [00:48, 939.38 examples/s]43566 examples [00:48, 950.39 examples/s]43665 examples [00:48, 957.95 examples/s]43763 examples [00:48, 963.23 examples/s]43860 examples [00:49, 962.85 examples/s]43957 examples [00:49, 954.43 examples/s]44053 examples [00:49, 950.47 examples/s]44149 examples [00:49, 940.12 examples/s]44244 examples [00:49, 929.00 examples/s]44339 examples [00:49, 933.15 examples/s]44433 examples [00:49, 933.01 examples/s]44531 examples [00:49, 945.26 examples/s]44628 examples [00:49, 949.70 examples/s]44726 examples [00:49, 956.56 examples/s]44822 examples [00:50, 952.04 examples/s]44918 examples [00:50, 929.49 examples/s]45012 examples [00:50, 918.55 examples/s]45107 examples [00:50, 924.47 examples/s]45200 examples [00:50, 924.97 examples/s]45293 examples [00:50, 924.04 examples/s]45386 examples [00:50, 919.98 examples/s]45479 examples [00:50, 921.65 examples/s]45572 examples [00:50, 922.88 examples/s]45665 examples [00:51, 907.13 examples/s]45756 examples [00:51, 902.46 examples/s]45847 examples [00:51, 891.94 examples/s]45940 examples [00:51, 901.71 examples/s]46033 examples [00:51, 908.05 examples/s]46124 examples [00:51, 907.11 examples/s]46217 examples [00:51, 912.18 examples/s]46309 examples [00:51, 893.06 examples/s]46400 examples [00:51, 895.71 examples/s]46492 examples [00:51, 902.30 examples/s]46587 examples [00:52, 914.91 examples/s]46683 examples [00:52, 927.30 examples/s]46776 examples [00:52, 910.49 examples/s]46869 examples [00:52, 913.76 examples/s]46962 examples [00:52, 916.81 examples/s]47055 examples [00:52, 918.83 examples/s]47147 examples [00:52, 901.68 examples/s]47241 examples [00:52, 911.19 examples/s]47333 examples [00:52, 906.23 examples/s]47424 examples [00:52, 893.04 examples/s]47515 examples [00:53, 895.86 examples/s]47605 examples [00:53, 892.66 examples/s]47695 examples [00:53, 891.36 examples/s]47788 examples [00:53, 900.99 examples/s]47881 examples [00:53, 907.78 examples/s]47974 examples [00:53, 913.48 examples/s]48068 examples [00:53, 919.32 examples/s]48160 examples [00:53, 914.29 examples/s]48255 examples [00:53, 924.14 examples/s]48348 examples [00:53, 898.27 examples/s]48439 examples [00:54, 879.86 examples/s]48528 examples [00:54, 862.22 examples/s]48616 examples [00:54, 866.14 examples/s]48703 examples [00:54, 865.67 examples/s]48790 examples [00:54, 859.53 examples/s]48885 examples [00:54, 882.39 examples/s]48975 examples [00:54, 885.92 examples/s]49066 examples [00:54, 892.09 examples/s]49156 examples [00:54, 878.59 examples/s]49250 examples [00:54, 893.56 examples/s]49345 examples [00:55, 906.90 examples/s]49441 examples [00:55, 919.83 examples/s]49534 examples [00:55, 906.14 examples/s]49627 examples [00:55, 913.06 examples/s]49720 examples [00:55, 915.76 examples/s]49816 examples [00:55, 926.90 examples/s]49911 examples [00:55, 932.56 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6503/50000 [00:00<00:00, 65026.96 examples/s] 36%|███▌      | 17927/50000 [00:00<00:00, 74669.26 examples/s] 59%|█████▉    | 29486/50000 [00:00<00:00, 83540.62 examples/s] 83%|████████▎ | 41283/50000 [00:00<00:00, 91555.95 examples/s]                                                               0 examples [00:00, ? examples/s]76 examples [00:00, 757.86 examples/s]172 examples [00:00, 807.84 examples/s]270 examples [00:00, 851.54 examples/s]361 examples [00:00, 865.91 examples/s]454 examples [00:00, 883.93 examples/s]548 examples [00:00, 900.02 examples/s]647 examples [00:00, 924.78 examples/s]748 examples [00:00, 946.58 examples/s]846 examples [00:00, 954.13 examples/s]940 examples [00:01, 948.12 examples/s]1037 examples [00:01, 951.91 examples/s]1134 examples [00:01, 956.71 examples/s]1229 examples [00:01, 938.53 examples/s]1324 examples [00:01, 940.50 examples/s]1418 examples [00:01, 924.24 examples/s]1511 examples [00:01, 922.21 examples/s]1604 examples [00:01, 921.36 examples/s]1698 examples [00:01, 925.96 examples/s]1791 examples [00:01, 921.84 examples/s]1884 examples [00:02, 907.68 examples/s]1977 examples [00:02, 912.53 examples/s]2078 examples [00:02, 937.93 examples/s]2177 examples [00:02, 952.11 examples/s]2274 examples [00:02, 955.39 examples/s]2373 examples [00:02, 964.01 examples/s]2470 examples [00:02, 951.13 examples/s]2566 examples [00:02, 945.21 examples/s]2663 examples [00:02, 951.11 examples/s]2759 examples [00:02, 945.92 examples/s]2854 examples [00:03, 936.32 examples/s]2948 examples [00:03, 932.45 examples/s]3044 examples [00:03, 938.20 examples/s]3140 examples [00:03, 943.81 examples/s]3235 examples [00:03, 940.05 examples/s]3333 examples [00:03, 947.84 examples/s]3430 examples [00:03, 954.31 examples/s]3526 examples [00:03, 942.63 examples/s]3621 examples [00:03, 934.58 examples/s]3715 examples [00:03, 927.04 examples/s]3810 examples [00:04, 931.88 examples/s]3904 examples [00:04, 930.53 examples/s]3999 examples [00:04, 935.30 examples/s]4095 examples [00:04, 939.94 examples/s]4190 examples [00:04, 932.46 examples/s]4284 examples [00:04, 915.19 examples/s]4376 examples [00:04, 906.22 examples/s]4472 examples [00:04, 920.11 examples/s]4570 examples [00:04, 936.29 examples/s]4664 examples [00:04, 935.51 examples/s]4758 examples [00:05, 926.64 examples/s]4851 examples [00:05, 921.78 examples/s]4944 examples [00:05, 923.19 examples/s]5037 examples [00:05, 919.70 examples/s]5130 examples [00:05, 914.42 examples/s]5222 examples [00:05, 912.67 examples/s]5318 examples [00:05, 925.25 examples/s]5414 examples [00:05, 933.62 examples/s]5508 examples [00:05, 929.54 examples/s]5601 examples [00:06, 928.24 examples/s]5696 examples [00:06, 933.29 examples/s]5791 examples [00:06, 936.29 examples/s]5890 examples [00:06, 951.08 examples/s]5986 examples [00:06, 943.92 examples/s]6081 examples [00:06, 912.67 examples/s]6173 examples [00:06, 904.84 examples/s]6266 examples [00:06, 912.23 examples/s]6358 examples [00:06, 913.73 examples/s]6451 examples [00:06, 917.44 examples/s]6545 examples [00:07, 921.06 examples/s]6638 examples [00:07, 907.32 examples/s]6729 examples [00:07, 901.30 examples/s]6820 examples [00:07, 886.55 examples/s]6913 examples [00:07, 899.02 examples/s]7008 examples [00:07, 911.85 examples/s]7100 examples [00:07, 868.66 examples/s]7189 examples [00:07, 873.91 examples/s]7282 examples [00:07, 889.62 examples/s]7374 examples [00:07, 896.40 examples/s]7465 examples [00:08, 896.20 examples/s]7555 examples [00:08, 889.46 examples/s]7649 examples [00:08, 902.82 examples/s]7743 examples [00:08, 913.45 examples/s]7836 examples [00:08, 917.16 examples/s]7931 examples [00:08, 924.40 examples/s]8028 examples [00:08, 937.58 examples/s]8126 examples [00:08, 949.13 examples/s]8222 examples [00:08, 948.10 examples/s]8319 examples [00:08, 953.20 examples/s]8415 examples [00:09, 944.35 examples/s]8512 examples [00:09, 950.09 examples/s]8608 examples [00:09, 950.91 examples/s]8704 examples [00:09, 947.70 examples/s]8799 examples [00:09, 935.45 examples/s]8893 examples [00:09, 910.97 examples/s]8985 examples [00:09, 892.67 examples/s]9076 examples [00:09, 897.14 examples/s]9170 examples [00:09, 909.00 examples/s]9269 examples [00:09, 929.62 examples/s]9365 examples [00:10, 938.33 examples/s]9460 examples [00:10, 940.08 examples/s]9560 examples [00:10, 956.30 examples/s]9656 examples [00:10, 935.30 examples/s]9750 examples [00:10, 915.22 examples/s]9842 examples [00:10, 884.02 examples/s]9936 examples [00:10, 899.87 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteQAEWX0/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteQAEWX0/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f843efdca60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f843efdca60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f843efdca60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f848a9490b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f83c92c9fd0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f843efdca60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f843efdca60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f843efdca60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8426e902b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8426e90048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f83b79252f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f83b79252f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f83b79252f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f83b7925400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f83b7925400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f83b7925400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=451e906d0e72f3764ad12a64744fda178764135235bb8e7ffdc8e700c361fc7f
  Stored in directory: /tmp/pip-ephem-wheel-cache-wi573ns5/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:15:08, 11.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:24:41, 16.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:08:31, 23.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:06:29, 33.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.59M/862M [00:01<4:57:49, 48.0kB/s].vector_cache/glove.6B.zip:   1%|          | 9.16M/862M [00:01<3:27:11, 68.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.7M/862M [00:01<2:24:33, 97.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.3M/862M [00:01<1:40:36, 140kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:01<1:10:19, 199kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.7M/862M [00:01<48:59, 284kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:01<34:18, 404kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:01<23:56, 576kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.5M/862M [00:02<16:49, 816kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.8M/862M [00:02<11:46, 1.16MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.0M/862M [00:02<08:20, 1.63MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<06:20, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<06:20, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<06:37, 2.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<05:06, 2.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:05<03:44, 3.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<12:11, 1.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:07<10:15, 1.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<07:35, 1.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<07:56, 1.67MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<08:16, 1.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:09<06:22, 2.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:09<04:37, 2.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<10:08, 1.30MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<08:26, 1.56MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:11<06:11, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<07:24, 1.77MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:12<06:18, 2.08MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:13<04:43, 2.78MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:13<03:27, 3.78MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<51:53, 252kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<38:59, 335kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:15<27:55, 468kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<21:35, 603kB/s].vector_cache/glove.6B.zip:  10%|▉         | 81.9M/862M [00:16<16:27, 790kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:17<11:49, 1.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<11:17, 1.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<09:12, 1.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:19<06:42, 1.93MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<07:45, 1.66MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<08:05, 1.59MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:20<06:14, 2.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:21<04:30, 2.85MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<17:38, 726kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<13:38, 938kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:22<09:48, 1.30MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<09:51, 1.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<08:11, 1.55MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<06:00, 2.11MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<05:06, 2.48MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<8:14:17, 25.6kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<5:45:54, 36.6kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<4:03:32, 51.8kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<2:53:22, 72.7kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<2:01:54, 103kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<1:25:20, 147kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:03:41, 197kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<45:50, 273kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<32:20, 387kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<25:32, 488kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<18:55, 659kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<13:32, 918kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<12:22, 1.00MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<11:13, 1.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:23, 1.48MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<06:01, 2.05MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<09:57, 1.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:13, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<06:03, 2.03MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:06, 1.72MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:28, 1.64MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:51, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:03, 2.01MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:30, 2.21MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:07, 2.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:43, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:16, 2.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<03:56, 3.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:34, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:23, 1.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:06, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<03:41, 3.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<48:46, 246kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<35:20, 339kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<24:59, 478kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<20:14, 589kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<15:23, 774kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<11:00, 1.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<10:29, 1.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<09:45, 1.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:19, 1.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<05:17, 2.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<08:31, 1.38MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<07:09, 1.64MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:18, 2.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:27, 1.81MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:54, 1.69MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:26, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<03:56, 2.95MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<1:37:46, 119kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<1:09:36, 167kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<48:55, 237kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<36:50, 314kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<26:57, 429kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<19:04, 605kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<16:03, 716kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<12:25, 926kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [00:59<08:58, 1.28MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:57, 1.28MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:36, 1.33MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<06:30, 1.75MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<04:41, 2.43MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<10:55, 1.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<08:49, 1.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<06:24, 1.77MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<05:21, 2.11MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<7:06:41, 26.5kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<4:58:32, 37.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<3:30:10, 53.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<2:29:50, 75.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<1:45:22, 107kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<1:13:46, 152kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<54:51, 204kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<39:37, 282kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<27:56, 399kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<21:53, 507kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<17:48, 624kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<13:04, 849kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<09:15, 1.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<19:50, 556kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<15:00, 735kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<10:43, 1.03MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<10:03, 1.09MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<09:16, 1.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:02, 1.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<06:40, 1.63MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:48, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:17, 2.53MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:33, 1.95MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:05, 1.78MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:49, 2.24MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:06, 2.11MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:40, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<03:32, 3.03MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:58, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:40, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:26, 2.40MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:12, 3.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<14:46, 719kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<11:26, 928kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<08:13, 1.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<08:13, 1.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:49, 1.54MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:02, 2.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<06:00, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:16, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:56, 2.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<05:13, 2.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:42, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:31, 2.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:55, 2.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:29, 2.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:24, 3.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:48, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:34<04:24, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:20, 3.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:45, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<04:21, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:16, 3.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:42, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:21, 2.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<03:17, 3.07MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:41, 2.15MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:17, 2.34MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:15, 3.08MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:38, 2.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:07, 2.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<03:08, 3.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:33, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:12, 2.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<03:08, 3.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<03:00, 3.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<6:05:15, 27.1kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<4:15:28, 38.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<2:59:48, 54.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<2:07:55, 76.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<1:29:57, 109kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:48<1:02:46, 155kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<1:13:11, 133kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<52:12, 186kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<36:42, 265kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<27:51, 347kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<21:28, 450kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<15:26, 626kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<10:54, 883kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<12:09, 790kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<09:29, 1.01MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<06:50, 1.40MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<07:00, 1.36MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:50, 1.39MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<05:16, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:12, 1.82MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:36, 2.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:25, 2.76MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:35, 2.04MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:08, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<03:59, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<02:55, 3.21MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:43, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:40, 1.64MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:11, 2.21MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<05:06, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:30, 2.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:23, 2.73MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:32, 2.02MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:03, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<03:59, 2.30MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<04:15, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<03:55, 2.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<02:58, 3.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:11, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:47, 1.89MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:49, 2.37MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:06, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<03:49, 2.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<02:54, 3.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:10, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<04:44, 1.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:46, 2.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:03, 2.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:44, 2.36MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<02:50, 3.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:02, 2.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:42, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<02:48, 3.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:01, 2.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:37, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:36, 2.42MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<02:38, 3.28MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:30, 1.57MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:43, 1.83MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<03:29, 2.47MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:27, 1.93MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:59, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<03:00, 2.85MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<02:43, 3.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<5:28:35, 25.9kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<3:49:44, 37.0kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<2:41:29, 52.3kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<1:55:04, 73.4kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<1:20:52, 104kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<56:32, 149kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<42:19, 198kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<30:35, 274kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<21:35, 387kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<16:47, 495kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<13:35, 611kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<09:57, 833kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<07:01, 1.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<17:19, 475kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<12:57, 635kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<09:15, 886kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<08:22, 976kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<06:41, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<04:50, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:17, 1.53MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:31, 1.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:22, 2.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:15, 1.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:36, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:38, 2.21MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<03:49, 2.08MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:29, 2.28MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<02:38, 3.01MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<03:41, 2.14MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:23, 2.33MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<02:32, 3.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<03:38, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:20, 2.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<02:31, 3.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:36, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:18, 2.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<02:30, 3.08MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<03:34, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:16, 2.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<02:27, 3.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:31, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:14, 2.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:25, 3.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:30, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:12, 2.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<02:23, 3.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:27, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<03:10, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:24, 3.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:26, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:09, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<02:23, 3.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:24, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:52, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:05, 2.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:13, 3.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<7:02:03, 17.3kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<4:55:56, 24.6kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<3:26:36, 35.1kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<2:25:35, 49.5kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<1:42:33, 70.3kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<1:11:42, 100kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<51:38, 138kB/s]  .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<36:50, 194kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<25:52, 275kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<19:42, 359kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<14:23, 491kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<10:14, 689kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<07:12, 972kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<31:30, 222kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<22:45, 308kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<16:01, 435kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<11:37, 597kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<4:53:01, 23.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<3:25:14, 33.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<2:22:52, 48.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<1:43:36, 66.4kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<1:14:01, 92.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<52:03, 132kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<36:23, 188kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<27:37, 246kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<20:00, 340kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<14:07, 479kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<11:25, 589kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<09:22, 719kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<06:50, 984kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<04:52, 1.37MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:53, 1.13MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:48, 1.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<03:31, 1.88MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:00, 1.65MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:27, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:34, 2.55MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:20, 1.95MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:00, 2.17MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:15, 2.88MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<03:06, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:29, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:43, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<01:59, 3.21MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:46, 1.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:17, 1.94MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:27, 2.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:11, 1.98MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:31, 1.79MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:47, 2.27MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<02:56, 2.12MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<02:42, 2.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:03, 3.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:52, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:16, 1.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:33, 2.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<01:53, 3.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:28, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:57, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:11, 2.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<01:36, 3.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<25:31, 237kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<19:06, 317kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<13:39, 442kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<10:26, 573kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<07:54, 755kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<05:39, 1.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<05:18, 1.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:42, 1.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:33, 1.66MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:38, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:11, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<02:50, 2.05MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:06, 2.76MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:49, 2.04MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:12, 1.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:32, 2.27MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<01:49, 3.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<37:50, 151kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<27:02, 211kB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<18:59, 299kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<14:32, 388kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<10:44, 525kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<07:36, 737kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<06:36, 842kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<05:45, 966kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<04:15, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<03:04, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:48, 1.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<3:18:45, 27.7kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<2:18:47, 39.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<1:37:19, 55.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<1:09:18, 78.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<48:39, 112kB/s]   .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<34:00, 159kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<25:13, 213kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<18:06, 296kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<12:45, 419kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<08:57, 593kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<11:26, 464kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<09:04, 584kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<06:37, 799kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:57<04:39, 1.12MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<40:37, 129kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<28:56, 181kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<20:17, 256kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<15:19, 337kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<11:46, 439kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<08:26, 611kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<05:56, 861kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<06:23, 797kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<05:00, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<03:37, 1.40MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:40, 1.37MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:35, 1.40MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:43, 1.84MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<01:57, 2.54MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<04:23, 1.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:35, 1.38MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:36, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:57, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:34, 1.90MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<01:54, 2.54MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:28, 1.95MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:40, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:04, 2.31MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:31, 3.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:53, 1.64MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:31, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:51, 2.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:23, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:09, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:35, 2.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:12, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:29, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<01:56, 2.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:26, 3.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:21, 1.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:06, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:35, 2.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:08, 2.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:22, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<01:51, 2.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:21, 3.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:44, 1.61MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:22, 1.86MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<01:45, 2.48MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:14, 1.94MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:56, 2.24MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:26, 3.00MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<01:03, 4.04MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<16:53, 253kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<12:41, 337kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<09:04, 470kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<06:57, 605kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<05:17, 794kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<03:46, 1.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<03:35, 1.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:56, 1.40MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<02:09, 1.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:26, 1.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:32, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:56, 2.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<01:24, 2.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:47, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<2:35:19, 25.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<1:48:18, 36.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<1:15:37, 52.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<53:44, 73.3kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<37:41, 104kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<26:09, 148kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<20:35, 188kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<14:47, 261kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<10:22, 370kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<08:05, 470kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<06:02, 628kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<04:17, 880kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<03:50, 970kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<03:27, 1.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:35, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:42<01:50, 2.00MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<3:33:10, 17.2kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<2:29:20, 24.5kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<1:43:52, 35.0kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<1:12:50, 49.4kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<51:15, 70.0kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<35:41, 99.8kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<25:34, 138kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<18:36, 189kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<13:07, 267kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<09:07, 380kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<08:28, 408kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<06:16, 550kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<04:27, 770kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<03:52, 874kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<03:24, 994kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:31, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<01:48, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:27, 1.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:03, 1.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:30, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:48, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:35, 2.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:10, 2.74MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:34, 2.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:25, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:03, 2.98MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:28, 2.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:20, 2.30MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:00, 3.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:25, 2.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:37, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:17, 2.36MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:22, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:15, 2.36MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<00:56, 3.12MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:20, 2.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:31, 1.90MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:12, 2.39MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:17, 2.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:12, 2.35MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<00:54, 3.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:16, 2.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:07, 2.46MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<00:50, 3.28MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<00:36, 4.39MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<11:20, 239kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<08:28, 319kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<06:01, 446kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<04:11, 633kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<04:50, 545kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<03:39, 720kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<02:35, 1.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:58, 1.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<1:44:17, 24.7kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<1:12:29, 35.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<50:15, 49.9kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<35:36, 70.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<24:53, 100kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<17:28, 140kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<12:29, 195kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<08:43, 276kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<06:30, 364kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<05:02, 469kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<03:37, 648kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<02:31, 914kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<20:29, 112kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<14:32, 158kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<10:07, 224kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<07:29, 298kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<05:27, 408kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<03:49, 574kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<03:08, 689kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<02:38, 818kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:56, 1.10MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:40, 1.25MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:22, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<01:00, 2.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:09, 1.75MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:13, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:57, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<00:40, 2.89MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<14:25, 136kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<10:16, 190kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<07:08, 269kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<05:20, 353kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<04:07, 457kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:57, 632kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<02:02, 892kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<14:24, 126kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<10:14, 177kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<07:06, 251kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<05:16, 331kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<03:51, 451kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<02:42, 635kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<02:15, 746kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:44, 962kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<01:14, 1.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<01:13, 1.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:01, 1.57MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:44, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:52, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:55, 1.66MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:42, 2.15MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:30, 2.96MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:15, 1.17MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:01, 1.43MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:44, 1.95MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:50, 1.68MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:43, 1.93MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:32, 2.57MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:40, 1.96MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:36, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:27, 2.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:36, 2.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:40, 1.86MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:31, 2.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:22, 3.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:14, 970kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:59, 1.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:42, 1.66MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:44, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:37, 1.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:27, 2.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:33, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:36, 1.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:28, 2.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:21, 2.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<38:33, 26.0kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<26:28, 37.1kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<17:47, 52.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<12:37, 73.8kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<08:46, 105kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<05:50, 149kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<04:54, 176kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<03:29, 246kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<02:23, 348kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:47, 444kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<01:18, 602kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:54, 846kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:36, 1.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<03:10, 229kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<02:21, 306kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<01:39, 429kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<01:06, 608kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:04, 614kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:48, 805kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:33, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:30, 1.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:28, 1.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:20, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:12<00:14, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:23, 1.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:19, 1.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:13, 2.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:16, 1.68MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 2.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:07, 2.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<22:21, 17.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<15:25, 24.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<10:08, 35.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<06:22, 50.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<05:42, 55.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<04:00, 78.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<02:42, 111kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<01:34, 158kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<15:24, 16.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<10:31, 22.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<06:35, 32.7kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:23<03:52, 46.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<02:41, 65.1kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<01:45, 92.7kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:56, 132kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:38, 170kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:26, 237kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:13, 336kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:05, 432kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:03, 580kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 812kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<14:29:59,  7.66it/s]  0%|          | 745/400000 [00:00<10:08:07, 10.94it/s]  0%|          | 1456/400000 [00:00<7:05:12, 15.62it/s]  1%|          | 2221/400000 [00:00<4:57:20, 22.30it/s]  1%|          | 3018/400000 [00:00<3:27:58, 31.81it/s]  1%|          | 3805/400000 [00:00<2:25:32, 45.37it/s]  1%|          | 4578/400000 [00:00<1:41:56, 64.65it/s]  1%|▏         | 5322/400000 [00:00<1:11:29, 92.02it/s]  2%|▏         | 6022/400000 [00:00<50:14, 130.69it/s]   2%|▏         | 6770/400000 [00:01<35:22, 185.31it/s]  2%|▏         | 7512/400000 [00:01<24:58, 261.92it/s]  2%|▏         | 8279/400000 [00:01<17:42, 368.77it/s]  2%|▏         | 9012/400000 [00:01<12:39, 515.09it/s]  2%|▏         | 9756/400000 [00:01<09:06, 714.63it/s]  3%|▎         | 10485/400000 [00:01<06:38, 978.59it/s]  3%|▎         | 11209/400000 [00:01<04:54, 1321.43it/s]  3%|▎         | 11932/400000 [00:01<03:42, 1746.92it/s]  3%|▎         | 12668/400000 [00:01<02:51, 2264.95it/s]  3%|▎         | 13440/400000 [00:01<02:14, 2874.06it/s]  4%|▎         | 14217/400000 [00:02<01:48, 3543.54it/s]  4%|▎         | 14989/400000 [00:02<01:31, 4229.47it/s]  4%|▍         | 15745/400000 [00:02<01:19, 4805.16it/s]  4%|▍         | 16486/400000 [00:02<01:12, 5324.42it/s]  4%|▍         | 17218/400000 [00:02<01:06, 5789.62it/s]  4%|▍         | 17984/400000 [00:02<01:01, 6245.82it/s]  5%|▍         | 18747/400000 [00:02<00:57, 6605.02it/s]  5%|▍         | 19495/400000 [00:02<00:55, 6796.61it/s]  5%|▌         | 20237/400000 [00:02<00:54, 6964.86it/s]  5%|▌         | 20981/400000 [00:02<00:53, 7100.63it/s]  5%|▌         | 21743/400000 [00:03<00:52, 7248.02it/s]  6%|▌         | 22491/400000 [00:03<00:51, 7304.89it/s]  6%|▌         | 23243/400000 [00:03<00:51, 7365.61it/s]  6%|▌         | 23992/400000 [00:03<00:51, 7252.64it/s]  6%|▌         | 24744/400000 [00:03<00:51, 7330.14it/s]  6%|▋         | 25512/400000 [00:03<00:50, 7430.23it/s]  7%|▋         | 26260/400000 [00:03<00:50, 7351.44it/s]  7%|▋         | 26999/400000 [00:03<00:51, 7304.92it/s]  7%|▋         | 27732/400000 [00:03<00:52, 7094.31it/s]  7%|▋         | 28445/400000 [00:03<00:53, 6962.04it/s]  7%|▋         | 29221/400000 [00:04<00:51, 7181.98it/s]  7%|▋         | 29993/400000 [00:04<00:50, 7333.97it/s]  8%|▊         | 30730/400000 [00:04<00:50, 7334.82it/s]  8%|▊         | 31466/400000 [00:04<00:50, 7254.38it/s]  8%|▊         | 32205/400000 [00:04<00:50, 7294.23it/s]  8%|▊         | 32972/400000 [00:04<00:49, 7402.87it/s]  8%|▊         | 33723/400000 [00:04<00:49, 7433.53it/s]  9%|▊         | 34470/400000 [00:04<00:49, 7442.23it/s]  9%|▉         | 35215/400000 [00:04<00:50, 7209.61it/s]  9%|▉         | 35954/400000 [00:05<00:50, 7262.63it/s]  9%|▉         | 36716/400000 [00:05<00:49, 7363.76it/s]  9%|▉         | 37486/400000 [00:05<00:48, 7460.04it/s] 10%|▉         | 38234/400000 [00:05<00:48, 7435.66it/s] 10%|▉         | 38979/400000 [00:05<00:48, 7389.26it/s] 10%|▉         | 39747/400000 [00:05<00:48, 7474.05it/s] 10%|█         | 40496/400000 [00:05<00:48, 7443.55it/s] 10%|█         | 41265/400000 [00:05<00:47, 7513.93it/s] 11%|█         | 42023/400000 [00:05<00:47, 7533.29it/s] 11%|█         | 42777/400000 [00:05<00:47, 7531.90it/s] 11%|█         | 43551/400000 [00:06<00:46, 7591.10it/s] 11%|█         | 44311/400000 [00:06<00:48, 7338.82it/s] 11%|█▏        | 45057/400000 [00:06<00:48, 7368.94it/s] 11%|█▏        | 45822/400000 [00:06<00:47, 7449.60it/s] 12%|█▏        | 46569/400000 [00:06<00:48, 7303.88it/s] 12%|█▏        | 47353/400000 [00:06<00:47, 7454.48it/s] 12%|█▏        | 48104/400000 [00:06<00:47, 7469.79it/s] 12%|█▏        | 48853/400000 [00:06<00:47, 7414.88it/s] 12%|█▏        | 49596/400000 [00:06<00:48, 7224.09it/s] 13%|█▎        | 50321/400000 [00:06<00:49, 7123.75it/s] 13%|█▎        | 51088/400000 [00:07<00:47, 7277.52it/s] 13%|█▎        | 51832/400000 [00:07<00:47, 7324.88it/s] 13%|█▎        | 52566/400000 [00:07<00:47, 7326.51it/s] 13%|█▎        | 53309/400000 [00:07<00:47, 7354.84it/s] 14%|█▎        | 54046/400000 [00:07<00:47, 7331.70it/s] 14%|█▎        | 54815/400000 [00:07<00:46, 7434.25it/s] 14%|█▍        | 55561/400000 [00:07<00:46, 7441.38it/s] 14%|█▍        | 56306/400000 [00:07<00:47, 7286.54it/s] 14%|█▍        | 57036/400000 [00:07<00:47, 7156.61it/s] 14%|█▍        | 57753/400000 [00:07<00:49, 6933.78it/s] 15%|█▍        | 58514/400000 [00:08<00:47, 7123.39it/s] 15%|█▍        | 59289/400000 [00:08<00:46, 7300.18it/s] 15%|█▌        | 60048/400000 [00:08<00:46, 7383.27it/s] 15%|█▌        | 60822/400000 [00:08<00:45, 7486.74it/s] 15%|█▌        | 61573/400000 [00:08<00:47, 7121.06it/s] 16%|█▌        | 62319/400000 [00:08<00:46, 7218.73it/s] 16%|█▌        | 63048/400000 [00:08<00:46, 7237.92it/s] 16%|█▌        | 63776/400000 [00:08<00:46, 7248.76it/s] 16%|█▌        | 64503/400000 [00:08<00:46, 7191.47it/s] 16%|█▋        | 65224/400000 [00:09<00:47, 7049.63it/s] 16%|█▋        | 65931/400000 [00:09<00:47, 7046.92it/s] 17%|█▋        | 66703/400000 [00:09<00:46, 7236.03it/s] 17%|█▋        | 67445/400000 [00:09<00:45, 7287.46it/s] 17%|█▋        | 68184/400000 [00:09<00:45, 7316.16it/s] 17%|█▋        | 68917/400000 [00:09<00:45, 7270.54it/s] 17%|█▋        | 69680/400000 [00:09<00:44, 7370.43it/s] 18%|█▊        | 70425/400000 [00:09<00:44, 7391.80it/s] 18%|█▊        | 71165/400000 [00:09<00:44, 7378.90it/s] 18%|█▊        | 71904/400000 [00:09<00:44, 7343.95it/s] 18%|█▊        | 72639/400000 [00:10<00:44, 7341.69it/s] 18%|█▊        | 73403/400000 [00:10<00:43, 7428.28it/s] 19%|█▊        | 74155/400000 [00:10<00:43, 7451.50it/s] 19%|█▊        | 74901/400000 [00:10<00:44, 7365.95it/s] 19%|█▉        | 75639/400000 [00:10<00:44, 7274.83it/s] 19%|█▉        | 76378/400000 [00:10<00:44, 7306.99it/s] 19%|█▉        | 77110/400000 [00:10<00:44, 7292.97it/s] 19%|█▉        | 77866/400000 [00:10<00:43, 7369.20it/s] 20%|█▉        | 78614/400000 [00:10<00:43, 7400.31it/s] 20%|█▉        | 79355/400000 [00:10<00:43, 7309.40it/s] 20%|██        | 80087/400000 [00:11<00:44, 7222.06it/s] 20%|██        | 80851/400000 [00:11<00:43, 7340.67it/s] 20%|██        | 81617/400000 [00:11<00:42, 7431.79it/s] 21%|██        | 82362/400000 [00:11<00:42, 7431.61it/s] 21%|██        | 83124/400000 [00:11<00:42, 7486.10it/s] 21%|██        | 83874/400000 [00:11<00:42, 7476.82it/s] 21%|██        | 84646/400000 [00:11<00:41, 7544.48it/s] 21%|██▏       | 85408/400000 [00:11<00:41, 7565.40it/s] 22%|██▏       | 86165/400000 [00:11<00:41, 7555.41it/s] 22%|██▏       | 86921/400000 [00:11<00:41, 7480.34it/s] 22%|██▏       | 87670/400000 [00:12<00:42, 7414.21it/s] 22%|██▏       | 88448/400000 [00:12<00:41, 7517.76it/s] 22%|██▏       | 89211/400000 [00:12<00:41, 7549.22it/s] 22%|██▏       | 89967/400000 [00:12<00:41, 7552.23it/s] 23%|██▎       | 90729/400000 [00:12<00:40, 7571.37it/s] 23%|██▎       | 91489/400000 [00:12<00:40, 7577.21it/s] 23%|██▎       | 92274/400000 [00:12<00:40, 7656.78it/s] 23%|██▎       | 93055/400000 [00:12<00:39, 7698.91it/s] 23%|██▎       | 93826/400000 [00:12<00:39, 7699.61it/s] 24%|██▎       | 94597/400000 [00:12<00:39, 7698.42it/s] 24%|██▍       | 95367/400000 [00:13<00:39, 7644.60it/s] 24%|██▍       | 96142/400000 [00:13<00:39, 7673.18it/s] 24%|██▍       | 96914/400000 [00:13<00:39, 7674.29it/s] 24%|██▍       | 97693/400000 [00:13<00:39, 7708.61it/s] 25%|██▍       | 98464/400000 [00:13<00:39, 7644.15it/s] 25%|██▍       | 99229/400000 [00:13<00:39, 7558.17it/s] 25%|██▍       | 99996/400000 [00:13<00:39, 7589.89it/s] 25%|██▌       | 100760/400000 [00:13<00:39, 7603.95it/s] 25%|██▌       | 101521/400000 [00:13<00:39, 7575.61it/s] 26%|██▌       | 102301/400000 [00:13<00:38, 7640.78it/s] 26%|██▌       | 103066/400000 [00:14<00:38, 7628.45it/s] 26%|██▌       | 103841/400000 [00:14<00:38, 7662.70it/s] 26%|██▌       | 104616/400000 [00:14<00:38, 7688.53it/s] 26%|██▋       | 105385/400000 [00:14<00:38, 7583.57it/s] 27%|██▋       | 106154/400000 [00:14<00:38, 7614.11it/s] 27%|██▋       | 106916/400000 [00:14<00:38, 7526.73it/s] 27%|██▋       | 107670/400000 [00:14<00:39, 7471.82it/s] 27%|██▋       | 108418/400000 [00:14<00:39, 7426.69it/s] 27%|██▋       | 109162/400000 [00:14<00:39, 7368.65it/s] 27%|██▋       | 109900/400000 [00:14<00:39, 7369.65it/s] 28%|██▊       | 110648/400000 [00:15<00:39, 7401.84it/s] 28%|██▊       | 111389/400000 [00:15<00:39, 7371.39it/s] 28%|██▊       | 112127/400000 [00:15<00:39, 7301.72it/s] 28%|██▊       | 112887/400000 [00:15<00:38, 7387.17it/s] 28%|██▊       | 113627/400000 [00:15<00:39, 7245.30it/s] 29%|██▊       | 114377/400000 [00:15<00:39, 7318.34it/s] 29%|██▉       | 115117/400000 [00:15<00:38, 7342.25it/s] 29%|██▉       | 115852/400000 [00:15<00:39, 7119.03it/s] 29%|██▉       | 116566/400000 [00:15<00:40, 7066.33it/s] 29%|██▉       | 117346/400000 [00:15<00:38, 7271.30it/s] 30%|██▉       | 118103/400000 [00:16<00:38, 7358.39it/s] 30%|██▉       | 118845/400000 [00:16<00:38, 7376.59it/s] 30%|██▉       | 119585/400000 [00:16<00:38, 7219.98it/s] 30%|███       | 120333/400000 [00:16<00:38, 7294.47it/s] 30%|███       | 121086/400000 [00:16<00:37, 7361.41it/s] 30%|███       | 121824/400000 [00:16<00:38, 7192.12it/s] 31%|███       | 122563/400000 [00:16<00:38, 7247.74it/s] 31%|███       | 123290/400000 [00:16<00:38, 7226.71it/s] 31%|███       | 124042/400000 [00:16<00:37, 7310.51it/s] 31%|███       | 124809/400000 [00:17<00:37, 7413.42it/s] 31%|███▏      | 125556/400000 [00:17<00:36, 7429.95it/s] 32%|███▏      | 126300/400000 [00:17<00:37, 7364.15it/s] 32%|███▏      | 127048/400000 [00:17<00:36, 7395.40it/s] 32%|███▏      | 127788/400000 [00:17<00:37, 7275.15it/s] 32%|███▏      | 128525/400000 [00:17<00:37, 7301.80it/s] 32%|███▏      | 129263/400000 [00:17<00:36, 7325.06it/s] 33%|███▎      | 130027/400000 [00:17<00:36, 7416.50it/s] 33%|███▎      | 130778/400000 [00:17<00:36, 7442.18it/s] 33%|███▎      | 131553/400000 [00:17<00:35, 7531.26it/s] 33%|███▎      | 132319/400000 [00:18<00:35, 7567.78it/s] 33%|███▎      | 133077/400000 [00:18<00:35, 7552.33it/s] 33%|███▎      | 133854/400000 [00:18<00:34, 7614.88it/s] 34%|███▎      | 134616/400000 [00:18<00:34, 7609.14it/s] 34%|███▍      | 135378/400000 [00:18<00:35, 7557.50it/s] 34%|███▍      | 136134/400000 [00:18<00:35, 7396.36it/s] 34%|███▍      | 136875/400000 [00:18<00:35, 7363.22it/s] 34%|███▍      | 137623/400000 [00:18<00:35, 7396.97it/s] 35%|███▍      | 138394/400000 [00:18<00:34, 7486.20it/s] 35%|███▍      | 139144/400000 [00:18<00:35, 7447.93it/s] 35%|███▍      | 139890/400000 [00:19<00:35, 7400.88it/s] 35%|███▌      | 140634/400000 [00:19<00:34, 7411.14it/s] 35%|███▌      | 141398/400000 [00:19<00:34, 7477.13it/s] 36%|███▌      | 142158/400000 [00:19<00:34, 7511.01it/s] 36%|███▌      | 142910/400000 [00:19<00:34, 7487.33it/s] 36%|███▌      | 143659/400000 [00:19<00:34, 7365.72it/s] 36%|███▌      | 144398/400000 [00:19<00:34, 7370.14it/s] 36%|███▋      | 145136/400000 [00:19<00:34, 7343.76it/s] 36%|███▋      | 145871/400000 [00:19<00:34, 7336.63it/s] 37%|███▋      | 146605/400000 [00:19<00:34, 7323.97it/s] 37%|███▋      | 147369/400000 [00:20<00:34, 7415.88it/s] 37%|███▋      | 148123/400000 [00:20<00:33, 7451.03it/s] 37%|███▋      | 148869/400000 [00:20<00:33, 7386.26it/s] 37%|███▋      | 149608/400000 [00:20<00:34, 7299.65it/s] 38%|███▊      | 150355/400000 [00:20<00:33, 7347.62it/s] 38%|███▊      | 151116/400000 [00:20<00:33, 7422.63it/s] 38%|███▊      | 151876/400000 [00:20<00:33, 7473.93it/s] 38%|███▊      | 152660/400000 [00:20<00:32, 7579.45it/s] 38%|███▊      | 153427/400000 [00:20<00:32, 7603.77it/s] 39%|███▊      | 154202/400000 [00:20<00:32, 7646.82it/s] 39%|███▊      | 154968/400000 [00:21<00:32, 7563.00it/s] 39%|███▉      | 155732/400000 [00:21<00:32, 7583.12it/s] 39%|███▉      | 156514/400000 [00:21<00:31, 7652.04it/s] 39%|███▉      | 157280/400000 [00:21<00:31, 7643.02it/s] 40%|███▉      | 158066/400000 [00:21<00:31, 7706.63it/s] 40%|███▉      | 158852/400000 [00:21<00:31, 7749.03it/s] 40%|███▉      | 159628/400000 [00:21<00:31, 7630.51it/s] 40%|████      | 160406/400000 [00:21<00:31, 7673.48it/s] 40%|████      | 161174/400000 [00:21<00:31, 7635.23it/s] 40%|████      | 161951/400000 [00:21<00:31, 7674.88it/s] 41%|████      | 162740/400000 [00:22<00:30, 7737.49it/s] 41%|████      | 163515/400000 [00:22<00:30, 7692.24it/s] 41%|████      | 164285/400000 [00:22<00:30, 7620.51it/s] 41%|████▏     | 165048/400000 [00:22<00:31, 7571.88it/s] 41%|████▏     | 165820/400000 [00:22<00:30, 7613.32it/s] 42%|████▏     | 166600/400000 [00:22<00:30, 7665.27it/s] 42%|████▏     | 167367/400000 [00:22<00:30, 7660.23it/s] 42%|████▏     | 168134/400000 [00:22<00:30, 7662.04it/s] 42%|████▏     | 168901/400000 [00:22<00:30, 7619.68it/s] 42%|████▏     | 169664/400000 [00:22<00:30, 7494.03it/s] 43%|████▎     | 170438/400000 [00:23<00:30, 7563.56it/s] 43%|████▎     | 171198/400000 [00:23<00:30, 7573.74it/s] 43%|████▎     | 171993/400000 [00:23<00:29, 7682.18it/s] 43%|████▎     | 172762/400000 [00:23<00:29, 7592.81it/s] 43%|████▎     | 173522/400000 [00:23<00:29, 7586.36it/s] 44%|████▎     | 174290/400000 [00:23<00:29, 7612.18it/s] 44%|████▍     | 175052/400000 [00:23<00:29, 7554.97it/s] 44%|████▍     | 175846/400000 [00:23<00:29, 7664.53it/s] 44%|████▍     | 176614/400000 [00:23<00:29, 7612.74it/s] 44%|████▍     | 177394/400000 [00:23<00:29, 7667.03it/s] 45%|████▍     | 178162/400000 [00:24<00:29, 7551.11it/s] 45%|████▍     | 178918/400000 [00:24<00:29, 7442.84it/s] 45%|████▍     | 179664/400000 [00:24<00:29, 7390.15it/s] 45%|████▌     | 180404/400000 [00:24<00:29, 7355.57it/s] 45%|████▌     | 181141/400000 [00:24<00:29, 7300.67it/s] 45%|████▌     | 181872/400000 [00:24<00:29, 7299.05it/s] 46%|████▌     | 182603/400000 [00:24<00:30, 7198.62it/s] 46%|████▌     | 183324/400000 [00:24<00:30, 7173.15it/s] 46%|████▌     | 184042/400000 [00:24<00:30, 7155.52it/s] 46%|████▌     | 184795/400000 [00:25<00:29, 7263.78it/s] 46%|████▋     | 185522/400000 [00:25<00:29, 7206.38it/s] 47%|████▋     | 186244/400000 [00:25<00:30, 7101.61it/s] 47%|████▋     | 186955/400000 [00:25<00:30, 7080.75it/s] 47%|████▋     | 187722/400000 [00:25<00:29, 7245.73it/s] 47%|████▋     | 188482/400000 [00:25<00:28, 7346.95it/s] 47%|████▋     | 189231/400000 [00:25<00:28, 7388.84it/s] 47%|████▋     | 189999/400000 [00:25<00:28, 7472.65it/s] 48%|████▊     | 190778/400000 [00:25<00:27, 7564.52it/s] 48%|████▊     | 191536/400000 [00:25<00:27, 7474.06it/s] 48%|████▊     | 192319/400000 [00:26<00:27, 7575.99it/s] 48%|████▊     | 193106/400000 [00:26<00:27, 7661.67it/s] 48%|████▊     | 193874/400000 [00:26<00:27, 7623.04it/s] 49%|████▊     | 194641/400000 [00:26<00:26, 7636.21it/s] 49%|████▉     | 195406/400000 [00:26<00:26, 7633.40it/s] 49%|████▉     | 196183/400000 [00:26<00:26, 7672.88it/s] 49%|████▉     | 196958/400000 [00:26<00:26, 7693.59it/s] 49%|████▉     | 197728/400000 [00:26<00:26, 7653.73it/s] 50%|████▉     | 198512/400000 [00:26<00:26, 7706.26it/s] 50%|████▉     | 199283/400000 [00:26<00:27, 7407.00it/s] 50%|█████     | 200070/400000 [00:27<00:26, 7539.04it/s] 50%|█████     | 200831/400000 [00:27<00:26, 7558.86it/s] 50%|█████     | 201589/400000 [00:27<00:26, 7431.24it/s] 51%|█████     | 202334/400000 [00:27<00:26, 7337.47it/s] 51%|█████     | 203086/400000 [00:27<00:26, 7389.97it/s] 51%|█████     | 203827/400000 [00:27<00:26, 7358.76it/s] 51%|█████     | 204596/400000 [00:27<00:26, 7453.63it/s] 51%|█████▏    | 205348/400000 [00:27<00:26, 7472.49it/s] 52%|█████▏    | 206098/400000 [00:27<00:25, 7480.15it/s] 52%|█████▏    | 206847/400000 [00:27<00:25, 7474.47it/s] 52%|█████▏    | 207602/400000 [00:28<00:25, 7494.72it/s] 52%|█████▏    | 208368/400000 [00:28<00:25, 7543.43it/s] 52%|█████▏    | 209123/400000 [00:28<00:25, 7511.02it/s] 52%|█████▏    | 209902/400000 [00:28<00:25, 7592.51it/s] 53%|█████▎    | 210678/400000 [00:28<00:24, 7641.26it/s] 53%|█████▎    | 211450/400000 [00:28<00:24, 7661.93it/s] 53%|█████▎    | 212217/400000 [00:28<00:24, 7596.07it/s] 53%|█████▎    | 212977/400000 [00:28<00:24, 7485.59it/s] 53%|█████▎    | 213727/400000 [00:28<00:25, 7409.21it/s] 54%|█████▎    | 214469/400000 [00:28<00:25, 7355.90it/s] 54%|█████▍    | 215232/400000 [00:29<00:24, 7432.43it/s] 54%|█████▍    | 215996/400000 [00:29<00:24, 7493.00it/s] 54%|█████▍    | 216746/400000 [00:29<00:25, 7302.02it/s] 54%|█████▍    | 217482/400000 [00:29<00:24, 7318.27it/s] 55%|█████▍    | 218244/400000 [00:29<00:24, 7404.61it/s] 55%|█████▍    | 219000/400000 [00:29<00:24, 7447.68it/s] 55%|█████▍    | 219746/400000 [00:29<00:24, 7338.17it/s] 55%|█████▌    | 220481/400000 [00:29<00:24, 7304.17it/s] 55%|█████▌    | 221251/400000 [00:29<00:24, 7416.82it/s] 56%|█████▌    | 222036/400000 [00:29<00:23, 7539.64it/s] 56%|█████▌    | 222792/400000 [00:30<00:23, 7437.66it/s] 56%|█████▌    | 223559/400000 [00:30<00:23, 7503.87it/s] 56%|█████▌    | 224311/400000 [00:30<00:24, 7147.96it/s] 56%|█████▋    | 225030/400000 [00:30<00:25, 6944.81it/s] 56%|█████▋    | 225792/400000 [00:30<00:24, 7132.40it/s] 57%|█████▋    | 226571/400000 [00:30<00:23, 7316.92it/s] 57%|█████▋    | 227314/400000 [00:30<00:23, 7349.38it/s] 57%|█████▋    | 228084/400000 [00:30<00:23, 7448.37it/s] 57%|█████▋    | 228851/400000 [00:30<00:22, 7511.15it/s] 57%|█████▋    | 229620/400000 [00:31<00:22, 7563.77it/s] 58%|█████▊    | 230378/400000 [00:31<00:22, 7468.77it/s] 58%|█████▊    | 231127/400000 [00:31<00:22, 7427.51it/s] 58%|█████▊    | 231883/400000 [00:31<00:22, 7465.41it/s] 58%|█████▊    | 232646/400000 [00:31<00:22, 7510.99it/s] 58%|█████▊    | 233398/400000 [00:31<00:22, 7384.44it/s] 59%|█████▊    | 234138/400000 [00:31<00:23, 7114.38it/s] 59%|█████▊    | 234881/400000 [00:31<00:22, 7205.58it/s] 59%|█████▉    | 235657/400000 [00:31<00:22, 7362.74it/s] 59%|█████▉    | 236410/400000 [00:31<00:22, 7409.71it/s] 59%|█████▉    | 237186/400000 [00:32<00:21, 7510.22it/s] 59%|█████▉    | 237966/400000 [00:32<00:21, 7592.30it/s] 60%|█████▉    | 238727/400000 [00:32<00:21, 7448.45it/s] 60%|█████▉    | 239494/400000 [00:32<00:21, 7513.05it/s] 60%|██████    | 240267/400000 [00:32<00:21, 7575.73it/s] 60%|██████    | 241052/400000 [00:32<00:20, 7654.72it/s] 60%|██████    | 241819/400000 [00:32<00:20, 7610.03it/s] 61%|██████    | 242581/400000 [00:32<00:20, 7537.44it/s] 61%|██████    | 243338/400000 [00:32<00:20, 7546.99it/s] 61%|██████    | 244101/400000 [00:32<00:20, 7568.83it/s] 61%|██████    | 244868/400000 [00:33<00:20, 7596.45it/s] 61%|██████▏   | 245628/400000 [00:33<00:20, 7526.74it/s] 62%|██████▏   | 246381/400000 [00:33<00:20, 7513.31it/s] 62%|██████▏   | 247146/400000 [00:33<00:20, 7551.70it/s] 62%|██████▏   | 247913/400000 [00:33<00:20, 7586.59it/s] 62%|██████▏   | 248675/400000 [00:33<00:19, 7596.19it/s] 62%|██████▏   | 249435/400000 [00:33<00:19, 7558.68it/s] 63%|██████▎   | 250192/400000 [00:33<00:20, 7477.55it/s] 63%|██████▎   | 250941/400000 [00:33<00:20, 7436.55it/s] 63%|██████▎   | 251701/400000 [00:33<00:19, 7483.06it/s] 63%|██████▎   | 252480/400000 [00:34<00:19, 7570.90it/s] 63%|██████▎   | 253238/400000 [00:34<00:19, 7550.69it/s] 63%|██████▎   | 253994/400000 [00:34<00:19, 7361.06it/s] 64%|██████▎   | 254732/400000 [00:34<00:20, 7250.93it/s] 64%|██████▍   | 255479/400000 [00:34<00:19, 7313.39it/s] 64%|██████▍   | 256212/400000 [00:34<00:19, 7310.25it/s] 64%|██████▍   | 256947/400000 [00:34<00:19, 7320.85it/s] 64%|██████▍   | 257680/400000 [00:34<00:19, 7286.30it/s] 65%|██████▍   | 258410/400000 [00:34<00:19, 7276.54it/s] 65%|██████▍   | 259181/400000 [00:34<00:19, 7399.19it/s] 65%|██████▍   | 259940/400000 [00:35<00:18, 7454.39it/s] 65%|██████▌   | 260705/400000 [00:35<00:18, 7511.07it/s] 65%|██████▌   | 261457/400000 [00:35<00:18, 7408.01it/s] 66%|██████▌   | 262207/400000 [00:35<00:18, 7433.63it/s] 66%|██████▌   | 262965/400000 [00:35<00:18, 7475.65it/s] 66%|██████▌   | 263745/400000 [00:35<00:18, 7568.46it/s] 66%|██████▌   | 264511/400000 [00:35<00:17, 7593.29it/s] 66%|██████▋   | 265271/400000 [00:35<00:17, 7547.86it/s] 67%|██████▋   | 266049/400000 [00:35<00:17, 7613.65it/s] 67%|██████▋   | 266827/400000 [00:35<00:17, 7660.54it/s] 67%|██████▋   | 267613/400000 [00:36<00:17, 7717.43it/s] 67%|██████▋   | 268387/400000 [00:36<00:17, 7722.07it/s] 67%|██████▋   | 269160/400000 [00:36<00:17, 7637.91it/s] 67%|██████▋   | 269925/400000 [00:36<00:17, 7561.22it/s] 68%|██████▊   | 270711/400000 [00:36<00:16, 7647.56it/s] 68%|██████▊   | 271499/400000 [00:36<00:16, 7713.69it/s] 68%|██████▊   | 272279/400000 [00:36<00:16, 7738.95it/s] 68%|██████▊   | 273054/400000 [00:36<00:16, 7699.41it/s] 68%|██████▊   | 273825/400000 [00:36<00:16, 7677.11it/s] 69%|██████▊   | 274597/400000 [00:37<00:16, 7689.25it/s] 69%|██████▉   | 275371/400000 [00:37<00:16, 7702.84it/s] 69%|██████▉   | 276142/400000 [00:37<00:17, 7248.79it/s] 69%|██████▉   | 276885/400000 [00:37<00:16, 7299.60it/s] 69%|██████▉   | 277619/400000 [00:37<00:16, 7232.36it/s] 70%|██████▉   | 278386/400000 [00:37<00:16, 7357.38it/s] 70%|██████▉   | 279165/400000 [00:37<00:16, 7479.34it/s] 70%|██████▉   | 279930/400000 [00:37<00:15, 7527.23it/s] 70%|███████   | 280685/400000 [00:37<00:16, 7445.43it/s] 70%|███████   | 281431/400000 [00:37<00:16, 7397.56it/s] 71%|███████   | 282201/400000 [00:38<00:15, 7484.62it/s] 71%|███████   | 282955/400000 [00:38<00:15, 7498.57it/s] 71%|███████   | 283738/400000 [00:38<00:15, 7592.87it/s] 71%|███████   | 284499/400000 [00:38<00:15, 7527.45it/s] 71%|███████▏  | 285253/400000 [00:38<00:15, 7491.92it/s] 72%|███████▏  | 286050/400000 [00:38<00:14, 7627.03it/s] 72%|███████▏  | 286824/400000 [00:38<00:14, 7659.77it/s] 72%|███████▏  | 287593/400000 [00:38<00:14, 7666.39it/s] 72%|███████▏  | 288366/400000 [00:38<00:14, 7685.25it/s] 72%|███████▏  | 289135/400000 [00:38<00:14, 7666.25it/s] 72%|███████▏  | 289918/400000 [00:39<00:14, 7712.49it/s] 73%|███████▎  | 290692/400000 [00:39<00:14, 7719.13it/s] 73%|███████▎  | 291471/400000 [00:39<00:14, 7737.31it/s] 73%|███████▎  | 292245/400000 [00:39<00:14, 7321.84it/s] 73%|███████▎  | 293001/400000 [00:39<00:14, 7391.29it/s] 73%|███████▎  | 293744/400000 [00:39<00:14, 7396.80it/s] 74%|███████▎  | 294516/400000 [00:39<00:14, 7489.54it/s] 74%|███████▍  | 295267/400000 [00:39<00:13, 7494.96it/s] 74%|███████▍  | 296021/400000 [00:39<00:13, 7508.18it/s] 74%|███████▍  | 296789/400000 [00:39<00:13, 7558.42it/s] 74%|███████▍  | 297546/400000 [00:40<00:13, 7530.17it/s] 75%|███████▍  | 298331/400000 [00:40<00:13, 7621.14it/s] 75%|███████▍  | 299094/400000 [00:40<00:13, 7612.68it/s] 75%|███████▍  | 299856/400000 [00:40<00:13, 7535.71it/s] 75%|███████▌  | 300611/400000 [00:40<00:13, 7459.42it/s] 75%|███████▌  | 301360/400000 [00:40<00:13, 7467.74it/s] 76%|███████▌  | 302148/400000 [00:40<00:12, 7583.82it/s] 76%|███████▌  | 302915/400000 [00:40<00:12, 7607.46it/s] 76%|███████▌  | 303684/400000 [00:40<00:12, 7630.45it/s] 76%|███████▌  | 304448/400000 [00:40<00:12, 7631.25it/s] 76%|███████▋  | 305212/400000 [00:41<00:12, 7562.73it/s] 76%|███████▋  | 305969/400000 [00:41<00:12, 7361.71it/s] 77%|███████▋  | 306707/400000 [00:41<00:13, 7175.04it/s] 77%|███████▋  | 307434/400000 [00:41<00:12, 7200.92it/s] 77%|███████▋  | 308170/400000 [00:41<00:12, 7247.79it/s] 77%|███████▋  | 308939/400000 [00:41<00:12, 7372.66it/s] 77%|███████▋  | 309735/400000 [00:41<00:11, 7537.28it/s] 78%|███████▊  | 310491/400000 [00:41<00:11, 7505.18it/s] 78%|███████▊  | 311249/400000 [00:41<00:11, 7524.17it/s] 78%|███████▊  | 312021/400000 [00:42<00:11, 7581.76it/s] 78%|███████▊  | 312797/400000 [00:42<00:11, 7634.06it/s] 78%|███████▊  | 313562/400000 [00:42<00:11, 7611.56it/s] 79%|███████▊  | 314336/400000 [00:42<00:11, 7649.23it/s] 79%|███████▉  | 315117/400000 [00:42<00:11, 7694.98it/s] 79%|███████▉  | 315887/400000 [00:42<00:11, 7570.69it/s] 79%|███████▉  | 316673/400000 [00:42<00:10, 7654.86it/s] 79%|███████▉  | 317458/400000 [00:42<00:10, 7712.34it/s] 80%|███████▉  | 318230/400000 [00:42<00:10, 7693.48it/s] 80%|███████▉  | 319000/400000 [00:42<00:10, 7580.39it/s] 80%|███████▉  | 319759/400000 [00:43<00:10, 7537.90it/s] 80%|████████  | 320526/400000 [00:43<00:10, 7575.28it/s] 80%|████████  | 321313/400000 [00:43<00:10, 7659.23it/s] 81%|████████  | 322091/400000 [00:43<00:10, 7693.49it/s] 81%|████████  | 322861/400000 [00:43<00:10, 7628.41it/s] 81%|████████  | 323625/400000 [00:43<00:10, 7585.35it/s] 81%|████████  | 324393/400000 [00:43<00:09, 7611.45it/s] 81%|████████▏ | 325155/400000 [00:43<00:10, 7427.00it/s] 81%|████████▏ | 325918/400000 [00:43<00:09, 7486.22it/s] 82%|████████▏ | 326692/400000 [00:43<00:09, 7559.95it/s] 82%|████████▏ | 327449/400000 [00:44<00:09, 7516.59it/s] 82%|████████▏ | 328237/400000 [00:44<00:09, 7621.83it/s] 82%|████████▏ | 329029/400000 [00:44<00:09, 7707.84it/s] 82%|████████▏ | 329801/400000 [00:44<00:09, 7534.20it/s] 83%|████████▎ | 330556/400000 [00:44<00:09, 7513.54it/s] 83%|████████▎ | 331309/400000 [00:44<00:09, 7491.84it/s] 83%|████████▎ | 332059/400000 [00:44<00:09, 7468.78it/s] 83%|████████▎ | 332822/400000 [00:44<00:08, 7515.56it/s] 83%|████████▎ | 333574/400000 [00:44<00:08, 7461.54it/s] 84%|████████▎ | 334321/400000 [00:44<00:08, 7378.45it/s] 84%|████████▍ | 335083/400000 [00:45<00:08, 7446.50it/s] 84%|████████▍ | 335858/400000 [00:45<00:08, 7534.24it/s] 84%|████████▍ | 336620/400000 [00:45<00:08, 7556.45it/s] 84%|████████▍ | 337377/400000 [00:45<00:08, 7471.25it/s] 85%|████████▍ | 338125/400000 [00:45<00:08, 7442.18it/s] 85%|████████▍ | 338909/400000 [00:45<00:08, 7555.93it/s] 85%|████████▍ | 339666/400000 [00:45<00:08, 7539.94it/s] 85%|████████▌ | 340421/400000 [00:45<00:07, 7526.47it/s] 85%|████████▌ | 341199/400000 [00:45<00:07, 7599.09it/s] 85%|████████▌ | 341960/400000 [00:45<00:07, 7598.23it/s] 86%|████████▌ | 342721/400000 [00:46<00:07, 7579.04it/s] 86%|████████▌ | 343500/400000 [00:46<00:07, 7639.32it/s] 86%|████████▌ | 344294/400000 [00:46<00:07, 7726.97it/s] 86%|████████▋ | 345068/400000 [00:46<00:07, 7591.98it/s] 86%|████████▋ | 345829/400000 [00:46<00:07, 7551.48it/s] 87%|████████▋ | 346585/400000 [00:46<00:07, 7268.54it/s] 87%|████████▋ | 347364/400000 [00:46<00:07, 7415.59it/s] 87%|████████▋ | 348135/400000 [00:46<00:06, 7499.68it/s] 87%|████████▋ | 348910/400000 [00:46<00:06, 7571.75it/s] 87%|████████▋ | 349674/400000 [00:46<00:06, 7591.03it/s] 88%|████████▊ | 350455/400000 [00:47<00:06, 7652.50it/s] 88%|████████▊ | 351222/400000 [00:47<00:06, 7604.55it/s] 88%|████████▊ | 351984/400000 [00:47<00:06, 7364.10it/s] 88%|████████▊ | 352724/400000 [00:47<00:06, 7374.25it/s] 88%|████████▊ | 353463/400000 [00:47<00:06, 7329.54it/s] 89%|████████▊ | 354207/400000 [00:47<00:06, 7362.08it/s] 89%|████████▊ | 354988/400000 [00:47<00:06, 7490.30it/s] 89%|████████▉ | 355761/400000 [00:47<00:05, 7560.07it/s] 89%|████████▉ | 356531/400000 [00:47<00:05, 7599.29it/s] 89%|████████▉ | 357292/400000 [00:47<00:05, 7584.03it/s] 90%|████████▉ | 358062/400000 [00:48<00:05, 7616.99it/s] 90%|████████▉ | 358825/400000 [00:48<00:05, 7487.72it/s] 90%|████████▉ | 359575/400000 [00:48<00:05, 7412.49it/s] 90%|█████████ | 360334/400000 [00:48<00:05, 7463.17it/s] 90%|█████████ | 361081/400000 [00:48<00:05, 7407.89it/s] 90%|█████████ | 361856/400000 [00:48<00:05, 7506.50it/s] 91%|█████████ | 362608/400000 [00:48<00:05, 7370.41it/s] 91%|█████████ | 363350/400000 [00:48<00:04, 7382.58it/s] 91%|█████████ | 364117/400000 [00:48<00:04, 7464.83it/s] 91%|█████████ | 364889/400000 [00:49<00:04, 7538.07it/s] 91%|█████████▏| 365651/400000 [00:49<00:04, 7560.33it/s] 92%|█████████▏| 366421/400000 [00:49<00:04, 7600.04it/s] 92%|█████████▏| 367182/400000 [00:49<00:04, 7477.88it/s] 92%|█████████▏| 367963/400000 [00:49<00:04, 7574.14it/s] 92%|█████████▏| 368722/400000 [00:49<00:04, 7482.61it/s] 92%|█████████▏| 369496/400000 [00:49<00:04, 7557.54it/s] 93%|█████████▎| 370253/400000 [00:49<00:03, 7514.63it/s] 93%|█████████▎| 371014/400000 [00:49<00:03, 7540.77it/s] 93%|█████████▎| 371789/400000 [00:49<00:03, 7599.85it/s] 93%|█████████▎| 372550/400000 [00:50<00:03, 7586.81it/s] 93%|█████████▎| 373310/400000 [00:50<00:03, 7586.54it/s] 94%|█████████▎| 374069/400000 [00:50<00:03, 7362.20it/s] 94%|█████████▎| 374807/400000 [00:50<00:03, 7295.01it/s] 94%|█████████▍| 375543/400000 [00:50<00:03, 7313.90it/s] 94%|█████████▍| 376276/400000 [00:50<00:03, 7282.63it/s] 94%|█████████▍| 377040/400000 [00:50<00:03, 7385.39it/s] 94%|█████████▍| 377817/400000 [00:50<00:02, 7493.66it/s] 95%|█████████▍| 378591/400000 [00:50<00:02, 7564.02it/s] 95%|█████████▍| 379392/400000 [00:50<00:02, 7691.63it/s] 95%|█████████▌| 380165/400000 [00:51<00:02, 7700.37it/s] 95%|█████████▌| 380936/400000 [00:51<00:02, 7650.57it/s] 95%|█████████▌| 381715/400000 [00:51<00:02, 7688.59it/s] 96%|█████████▌| 382485/400000 [00:51<00:02, 7653.78it/s] 96%|█████████▌| 383263/400000 [00:51<00:02, 7689.52it/s] 96%|█████████▌| 384033/400000 [00:51<00:02, 7645.33it/s] 96%|█████████▌| 384799/400000 [00:51<00:01, 7649.07it/s] 96%|█████████▋| 385567/400000 [00:51<00:01, 7654.63it/s] 97%|█████████▋| 386333/400000 [00:51<00:01, 7524.89it/s] 97%|█████████▋| 387115/400000 [00:51<00:01, 7610.74it/s] 97%|█████████▋| 387878/400000 [00:52<00:01, 7615.99it/s] 97%|█████████▋| 388656/400000 [00:52<00:01, 7662.59it/s] 97%|█████████▋| 389423/400000 [00:52<00:01, 7490.58it/s] 98%|█████████▊| 390187/400000 [00:52<00:01, 7533.11it/s] 98%|█████████▊| 390981/400000 [00:52<00:01, 7650.12it/s] 98%|█████████▊| 391759/400000 [00:52<00:01, 7686.46it/s] 98%|█████████▊| 392568/400000 [00:52<00:00, 7800.80it/s] 98%|█████████▊| 393359/400000 [00:52<00:00, 7832.67it/s] 99%|█████████▊| 394143/400000 [00:52<00:00, 7765.22it/s] 99%|█████████▊| 394943/400000 [00:52<00:00, 7831.78it/s] 99%|█████████▉| 395727/400000 [00:53<00:00, 7765.25it/s] 99%|█████████▉| 396505/400000 [00:53<00:00, 7745.70it/s] 99%|█████████▉| 397280/400000 [00:53<00:00, 7596.27it/s]100%|█████████▉| 398041/400000 [00:53<00:00, 7584.01it/s]100%|█████████▉| 398801/400000 [00:53<00:00, 7558.76it/s]100%|█████████▉| 399558/400000 [00:53<00:00, 7519.68it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7457.01it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f83b7c0cb70>, <torchtext.data.dataset.TabularDataset object at 0x7f83b7c0ccc0>, <torchtext.vocab.Vocab object at 0x7f83b7c0cbe0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.67 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 23.12 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.61 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.61 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.87 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.87 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.97 file/s]2020-07-28 00:20:26.201357: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-28 00:20:26.206072: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-28 00:20:26.206318: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c8b8aa0010 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-28 00:20:26.206337: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:24, 117696.54it/s] 51%|█████     | 5054464/9912422 [00:00<00:28, 167965.43it/s]9920512it [00:00, 33627975.84it/s]                           
0it [00:00, ?it/s]32768it [00:00, 631021.26it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 155559.45it/s]1654784it [00:00, 10438710.28it/s]                         
0it [00:00, ?it/s]8192it [00:00, 211031.57it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

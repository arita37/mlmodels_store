
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7feb56f2bb70> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7feb56f2bb70> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7febc1b9b510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7febc1b9b510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7febe0dc8ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7febe0dc8ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feb6ef43a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feb6ef43a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feb6ef43a60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 142338.24it/s] 56%|█████▋    | 5578752/9912422 [00:00<00:21, 203115.67it/s]9920512it [00:00, 36840037.22it/s]                           
0it [00:00, ?it/s]32768it [00:00, 631305.31it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:09, 163275.70it/s]1654784it [00:00, 11173844.69it/s]                         
0it [00:00, ?it/s]8192it [00:00, 179922.18it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feb56de86d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feb56df5978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7feb6ef436a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7feb6ef436a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7feb6ef436a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:43,  1.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:43,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:42,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:41,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:41,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:40,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:39,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:10,  2.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:10,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:09,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:09,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:08,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:08,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:08,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:07,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:07,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:47,  3.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:47,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:46,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:46,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:46,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:46,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:45,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:45,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:45,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:31,  4.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:31,  4.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:31,  4.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:31,  4.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:31,  4.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:30,  4.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:30,  4.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:30,  4.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:30,  4.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:21,  6.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:21,  6.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:21,  6.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:21,  6.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:21,  6.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:20,  6.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:20,  6.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:20,  6.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:20,  6.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:20,  6.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:14,  8.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:14,  8.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:14,  8.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:14,  8.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:14,  8.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:14,  8.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:13,  8.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:13,  8.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:13,  8.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:13,  8.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 11.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 15.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 20.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 20.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 20.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 20.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 33.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 33.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 33.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 33.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 41.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 49.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:01, 49.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 49.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 57.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 57.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 57.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 57.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 57.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 57.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 57.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 57.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 57.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 57.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 63.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 63.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 63.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 63.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 63.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 63.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 63.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 63.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 63.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 63.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 66.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 67.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 67.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 67.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 67.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 67.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 67.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.68s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.68s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.68s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.86 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.98s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.68s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 67.86 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.98s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.98s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.54 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.98s/ url]
0 examples [00:00, ? examples/s]2020-08-01 06:09:34.178063: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-01 06:09:34.191662: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-01 06:09:34.191893: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557894be2d30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-01 06:09:34.191930: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
43 examples [00:00, 427.54 examples/s]135 examples [00:00, 508.76 examples/s]226 examples [00:00, 585.20 examples/s]316 examples [00:00, 652.17 examples/s]405 examples [00:00, 707.65 examples/s]493 examples [00:00, 751.00 examples/s]580 examples [00:00, 782.76 examples/s]675 examples [00:00, 825.36 examples/s]767 examples [00:00, 850.82 examples/s]863 examples [00:01, 879.56 examples/s]958 examples [00:01, 897.78 examples/s]1050 examples [00:01, 904.14 examples/s]1145 examples [00:01, 916.03 examples/s]1243 examples [00:01, 933.05 examples/s]1338 examples [00:01, 937.30 examples/s]1434 examples [00:01, 941.66 examples/s]1529 examples [00:01, 934.90 examples/s]1625 examples [00:01, 941.32 examples/s]1721 examples [00:01, 945.10 examples/s]1817 examples [00:02, 947.85 examples/s]1913 examples [00:02, 948.77 examples/s]2008 examples [00:02, 939.65 examples/s]2104 examples [00:02, 942.89 examples/s]2200 examples [00:02, 946.68 examples/s]2295 examples [00:02, 944.79 examples/s]2393 examples [00:02, 954.32 examples/s]2489 examples [00:02, 954.27 examples/s]2585 examples [00:02, 955.30 examples/s]2681 examples [00:02, 940.23 examples/s]2776 examples [00:03, 937.50 examples/s]2870 examples [00:03, 922.44 examples/s]2963 examples [00:03, 924.18 examples/s]3056 examples [00:03, 919.19 examples/s]3150 examples [00:03, 925.05 examples/s]3243 examples [00:03, 899.82 examples/s]3340 examples [00:03, 918.09 examples/s]3433 examples [00:03, 920.74 examples/s]3529 examples [00:03, 931.19 examples/s]3625 examples [00:03, 938.07 examples/s]3720 examples [00:04, 940.16 examples/s]3815 examples [00:04, 941.80 examples/s]3910 examples [00:04, 937.65 examples/s]4007 examples [00:04, 945.64 examples/s]4103 examples [00:04, 948.49 examples/s]4201 examples [00:04, 955.05 examples/s]4297 examples [00:04, 950.07 examples/s]4393 examples [00:04, 945.67 examples/s]4488 examples [00:04, 921.47 examples/s]4581 examples [00:04, 914.23 examples/s]4673 examples [00:05, 913.15 examples/s]4765 examples [00:05, 911.62 examples/s]4857 examples [00:05, 899.32 examples/s]4948 examples [00:05, 902.40 examples/s]5040 examples [00:05, 906.07 examples/s]5134 examples [00:05, 915.29 examples/s]5226 examples [00:05, 906.98 examples/s]5317 examples [00:05, 898.27 examples/s]5410 examples [00:05, 906.29 examples/s]5501 examples [00:05, 898.91 examples/s]5591 examples [00:06, 898.30 examples/s]5681 examples [00:06, 898.11 examples/s]5771 examples [00:06, 895.49 examples/s]5863 examples [00:06, 900.29 examples/s]5955 examples [00:06, 905.92 examples/s]6046 examples [00:06, 903.27 examples/s]6138 examples [00:06, 906.12 examples/s]6229 examples [00:06, 896.05 examples/s]6319 examples [00:06, 895.02 examples/s]6409 examples [00:07, 895.98 examples/s]6501 examples [00:07, 902.73 examples/s]6595 examples [00:07, 910.66 examples/s]6687 examples [00:07, 894.70 examples/s]6777 examples [00:07, 892.61 examples/s]6868 examples [00:07, 895.78 examples/s]6959 examples [00:07, 898.74 examples/s]7049 examples [00:07, 896.43 examples/s]7139 examples [00:07, 894.25 examples/s]7233 examples [00:07, 905.56 examples/s]7324 examples [00:08, 905.94 examples/s]7417 examples [00:08, 910.11 examples/s]7509 examples [00:08, 901.11 examples/s]7600 examples [00:08, 890.01 examples/s]7692 examples [00:08, 897.06 examples/s]7784 examples [00:08, 900.36 examples/s]7877 examples [00:08, 907.82 examples/s]7969 examples [00:08, 908.44 examples/s]8060 examples [00:08, 894.33 examples/s]8150 examples [00:08, 889.22 examples/s]8242 examples [00:09, 897.06 examples/s]8332 examples [00:09, 897.84 examples/s]8424 examples [00:09, 902.93 examples/s]8515 examples [00:09, 903.84 examples/s]8606 examples [00:09, 901.84 examples/s]8697 examples [00:09, 899.75 examples/s]8787 examples [00:09, 899.67 examples/s]8879 examples [00:09, 902.81 examples/s]8970 examples [00:09, 895.07 examples/s]9060 examples [00:09, 894.78 examples/s]9151 examples [00:10, 897.84 examples/s]9243 examples [00:10, 902.44 examples/s]9335 examples [00:10, 905.66 examples/s]9426 examples [00:10, 896.26 examples/s]9516 examples [00:10, 894.63 examples/s]9608 examples [00:10, 900.51 examples/s]9699 examples [00:10, 900.29 examples/s]9790 examples [00:10, 901.48 examples/s]9882 examples [00:10, 905.03 examples/s]9973 examples [00:10, 906.30 examples/s]10064 examples [00:11, 848.23 examples/s]10156 examples [00:11, 867.49 examples/s]10248 examples [00:11, 880.82 examples/s]10337 examples [00:11, 882.23 examples/s]10426 examples [00:11, 884.44 examples/s]10515 examples [00:11, 884.27 examples/s]10607 examples [00:11, 893.77 examples/s]10699 examples [00:11, 898.74 examples/s]10789 examples [00:11, 895.04 examples/s]10881 examples [00:11, 900.71 examples/s]10972 examples [00:12, 898.02 examples/s]11064 examples [00:12, 901.74 examples/s]11156 examples [00:12, 905.97 examples/s]11247 examples [00:12, 902.37 examples/s]11338 examples [00:12, 882.51 examples/s]11429 examples [00:12, 888.12 examples/s]11527 examples [00:12, 913.26 examples/s]11622 examples [00:12, 923.86 examples/s]11715 examples [00:12, 923.28 examples/s]11808 examples [00:13, 905.71 examples/s]11900 examples [00:13, 908.79 examples/s]11996 examples [00:13, 921.78 examples/s]12092 examples [00:13, 932.85 examples/s]12187 examples [00:13, 935.60 examples/s]12284 examples [00:13, 944.54 examples/s]12383 examples [00:13, 956.88 examples/s]12479 examples [00:13, 955.56 examples/s]12576 examples [00:13, 957.33 examples/s]12672 examples [00:13, 956.45 examples/s]12771 examples [00:14, 964.45 examples/s]12868 examples [00:14, 959.05 examples/s]12965 examples [00:14, 960.40 examples/s]13062 examples [00:14, 958.34 examples/s]13158 examples [00:14, 946.52 examples/s]13253 examples [00:14, 939.58 examples/s]13348 examples [00:14, 932.49 examples/s]13442 examples [00:14, 922.66 examples/s]13538 examples [00:14, 932.30 examples/s]13632 examples [00:14, 915.98 examples/s]13724 examples [00:15, 916.95 examples/s]13820 examples [00:15, 928.07 examples/s]13913 examples [00:15, 924.27 examples/s]14010 examples [00:15, 936.45 examples/s]14106 examples [00:15, 942.52 examples/s]14201 examples [00:15, 943.67 examples/s]14296 examples [00:15, 937.44 examples/s]14393 examples [00:15, 945.65 examples/s]14490 examples [00:15, 950.95 examples/s]14586 examples [00:15, 939.90 examples/s]14682 examples [00:16, 944.16 examples/s]14777 examples [00:16, 939.53 examples/s]14872 examples [00:16, 940.83 examples/s]14969 examples [00:16, 948.88 examples/s]15064 examples [00:16, 898.89 examples/s]15155 examples [00:16, 900.23 examples/s]15247 examples [00:16, 905.89 examples/s]15338 examples [00:16, 902.89 examples/s]15430 examples [00:16, 907.50 examples/s]15521 examples [00:16, 902.64 examples/s]15614 examples [00:17, 909.83 examples/s]15707 examples [00:17, 915.75 examples/s]15800 examples [00:17, 917.70 examples/s]15892 examples [00:17, 912.72 examples/s]15984 examples [00:17, 908.20 examples/s]16077 examples [00:17, 913.25 examples/s]16169 examples [00:17, 908.05 examples/s]16262 examples [00:17, 911.53 examples/s]16355 examples [00:17, 916.04 examples/s]16447 examples [00:17, 912.82 examples/s]16540 examples [00:18, 917.71 examples/s]16632 examples [00:18, 917.83 examples/s]16725 examples [00:18, 921.43 examples/s]16818 examples [00:18, 913.48 examples/s]16910 examples [00:18, 899.93 examples/s]17004 examples [00:18, 910.83 examples/s]17096 examples [00:18, 909.89 examples/s]17191 examples [00:18, 920.53 examples/s]17284 examples [00:18, 922.54 examples/s]17377 examples [00:19, 916.78 examples/s]17472 examples [00:19, 923.34 examples/s]17565 examples [00:19, 908.18 examples/s]17662 examples [00:19, 923.80 examples/s]17757 examples [00:19, 930.50 examples/s]17851 examples [00:19, 926.95 examples/s]17944 examples [00:19, 926.08 examples/s]18039 examples [00:19, 931.24 examples/s]18133 examples [00:19, 932.42 examples/s]18227 examples [00:19, 927.73 examples/s]18321 examples [00:20, 927.95 examples/s]18416 examples [00:20, 932.75 examples/s]18512 examples [00:20, 939.76 examples/s]18607 examples [00:20, 941.80 examples/s]18702 examples [00:20, 940.90 examples/s]18797 examples [00:20, 938.31 examples/s]18893 examples [00:20, 943.07 examples/s]18988 examples [00:20, 939.90 examples/s]19084 examples [00:20, 943.20 examples/s]19179 examples [00:20, 941.77 examples/s]19274 examples [00:21, 919.71 examples/s]19369 examples [00:21, 925.85 examples/s]19463 examples [00:21, 929.53 examples/s]19559 examples [00:21, 937.73 examples/s]19654 examples [00:21, 939.67 examples/s]19749 examples [00:21, 937.83 examples/s]19847 examples [00:21, 949.16 examples/s]19943 examples [00:21, 949.02 examples/s]20038 examples [00:21, 893.41 examples/s]20130 examples [00:21, 900.84 examples/s]20224 examples [00:22, 910.42 examples/s]20316 examples [00:22, 899.81 examples/s]20407 examples [00:22, 869.17 examples/s]20496 examples [00:22, 874.00 examples/s]20591 examples [00:22, 894.11 examples/s]20685 examples [00:22, 906.95 examples/s]20776 examples [00:22, 889.01 examples/s]20866 examples [00:22, 886.47 examples/s]20964 examples [00:22, 911.61 examples/s]21056 examples [00:23, 895.12 examples/s]21146 examples [00:23, 894.63 examples/s]21239 examples [00:23, 904.80 examples/s]21333 examples [00:23, 913.44 examples/s]21427 examples [00:23, 920.26 examples/s]21520 examples [00:23, 920.01 examples/s]21613 examples [00:23, 915.40 examples/s]21706 examples [00:23, 919.22 examples/s]21798 examples [00:23, 911.04 examples/s]21895 examples [00:23, 925.97 examples/s]21993 examples [00:24, 940.63 examples/s]22090 examples [00:24, 947.95 examples/s]22187 examples [00:24, 953.28 examples/s]22284 examples [00:24, 958.01 examples/s]22380 examples [00:24, 952.57 examples/s]22477 examples [00:24, 956.30 examples/s]22573 examples [00:24, 956.01 examples/s]22669 examples [00:24, 954.95 examples/s]22765 examples [00:24, 939.60 examples/s]22860 examples [00:24, 931.25 examples/s]22956 examples [00:25, 936.75 examples/s]23053 examples [00:25, 944.88 examples/s]23149 examples [00:25, 946.66 examples/s]23246 examples [00:25, 951.49 examples/s]23343 examples [00:25, 955.71 examples/s]23446 examples [00:25, 974.73 examples/s]23544 examples [00:25, 958.32 examples/s]23641 examples [00:25, 960.85 examples/s]23738 examples [00:25, 953.36 examples/s]23835 examples [00:25, 955.72 examples/s]23931 examples [00:26, 954.21 examples/s]24027 examples [00:26, 950.32 examples/s]24123 examples [00:26, 927.35 examples/s]24216 examples [00:26, 898.93 examples/s]24307 examples [00:26, 852.86 examples/s]24400 examples [00:26, 872.80 examples/s]24495 examples [00:26, 892.53 examples/s]24592 examples [00:26, 914.37 examples/s]24687 examples [00:26, 923.36 examples/s]24783 examples [00:26, 932.80 examples/s]24878 examples [00:27, 936.15 examples/s]24973 examples [00:27, 938.81 examples/s]25068 examples [00:27, 937.41 examples/s]25162 examples [00:27, 935.36 examples/s]25256 examples [00:27, 918.17 examples/s]25348 examples [00:27, 910.99 examples/s]25440 examples [00:27, 909.98 examples/s]25532 examples [00:27, 905.51 examples/s]25623 examples [00:27, 889.89 examples/s]25713 examples [00:28, 857.73 examples/s]25800 examples [00:28, 844.01 examples/s]25890 examples [00:28, 857.64 examples/s]25978 examples [00:28, 861.92 examples/s]26067 examples [00:28, 867.76 examples/s]26155 examples [00:28, 868.96 examples/s]26242 examples [00:28, 867.62 examples/s]26335 examples [00:28, 884.13 examples/s]26424 examples [00:28, 877.87 examples/s]26514 examples [00:28, 883.87 examples/s]26603 examples [00:29, 884.59 examples/s]26692 examples [00:29, 885.83 examples/s]26783 examples [00:29, 889.94 examples/s]26878 examples [00:29, 906.89 examples/s]26973 examples [00:29, 917.79 examples/s]27068 examples [00:29, 925.93 examples/s]27161 examples [00:29, 925.04 examples/s]27258 examples [00:29, 937.22 examples/s]27354 examples [00:29, 942.66 examples/s]27450 examples [00:29, 947.23 examples/s]27545 examples [00:30, 947.52 examples/s]27640 examples [00:30, 935.71 examples/s]27734 examples [00:30, 933.80 examples/s]27828 examples [00:30, 919.62 examples/s]27923 examples [00:30, 927.96 examples/s]28019 examples [00:30, 935.22 examples/s]28113 examples [00:30, 908.24 examples/s]28212 examples [00:30, 931.16 examples/s]28307 examples [00:30, 935.65 examples/s]28401 examples [00:30, 935.76 examples/s]28497 examples [00:31, 940.30 examples/s]28592 examples [00:31, 935.88 examples/s]28689 examples [00:31, 944.55 examples/s]28784 examples [00:31, 940.63 examples/s]28879 examples [00:31, 937.32 examples/s]28973 examples [00:31, 910.79 examples/s]29066 examples [00:31, 913.64 examples/s]29161 examples [00:31, 923.71 examples/s]29255 examples [00:31, 928.49 examples/s]29348 examples [00:31, 926.36 examples/s]29441 examples [00:32, 918.25 examples/s]29535 examples [00:32, 921.45 examples/s]29633 examples [00:32, 936.46 examples/s]29730 examples [00:32, 943.44 examples/s]29825 examples [00:32, 899.98 examples/s]29920 examples [00:32, 912.98 examples/s]30012 examples [00:32, 864.64 examples/s]30108 examples [00:32, 888.13 examples/s]30202 examples [00:32, 901.44 examples/s]30293 examples [00:33, 897.36 examples/s]30384 examples [00:33, 899.20 examples/s]30475 examples [00:33, 894.93 examples/s]30567 examples [00:33, 902.16 examples/s]30662 examples [00:33, 913.20 examples/s]30758 examples [00:33, 925.68 examples/s]30855 examples [00:33, 935.63 examples/s]30949 examples [00:33, 929.61 examples/s]31045 examples [00:33, 938.31 examples/s]31142 examples [00:33, 946.44 examples/s]31239 examples [00:34, 951.79 examples/s]31335 examples [00:34, 937.43 examples/s]31429 examples [00:34, 934.45 examples/s]31528 examples [00:34, 950.41 examples/s]31624 examples [00:34, 947.08 examples/s]31719 examples [00:34, 947.69 examples/s]31815 examples [00:34, 948.45 examples/s]31910 examples [00:34, 944.74 examples/s]32005 examples [00:34, 935.08 examples/s]32100 examples [00:34, 937.62 examples/s]32198 examples [00:35, 948.41 examples/s]32296 examples [00:35, 955.47 examples/s]32392 examples [00:35, 951.93 examples/s]32489 examples [00:35, 955.83 examples/s]32587 examples [00:35, 960.60 examples/s]32685 examples [00:35, 964.45 examples/s]32782 examples [00:35, 961.40 examples/s]32879 examples [00:35, 958.94 examples/s]32976 examples [00:35, 961.61 examples/s]33073 examples [00:35, 956.55 examples/s]33169 examples [00:36, 945.94 examples/s]33264 examples [00:36, 936.61 examples/s]33358 examples [00:36, 934.45 examples/s]33452 examples [00:36, 934.86 examples/s]33548 examples [00:36, 941.59 examples/s]33643 examples [00:36, 896.75 examples/s]33734 examples [00:36, 890.40 examples/s]33828 examples [00:36, 901.41 examples/s]33919 examples [00:36, 901.70 examples/s]34013 examples [00:37, 911.37 examples/s]34106 examples [00:37, 914.30 examples/s]34198 examples [00:37, 912.11 examples/s]34290 examples [00:37, 891.53 examples/s]34382 examples [00:37, 897.05 examples/s]34472 examples [00:37, 895.23 examples/s]34562 examples [00:37, 890.69 examples/s]34652 examples [00:37, 881.90 examples/s]34744 examples [00:37, 892.60 examples/s]34838 examples [00:37, 904.42 examples/s]34932 examples [00:38, 912.67 examples/s]35024 examples [00:38, 913.84 examples/s]35119 examples [00:38, 921.92 examples/s]35212 examples [00:38, 913.27 examples/s]35304 examples [00:38, 900.34 examples/s]35395 examples [00:38, 888.15 examples/s]35486 examples [00:38, 892.49 examples/s]35578 examples [00:38, 898.29 examples/s]35668 examples [00:38, 890.80 examples/s]35760 examples [00:38, 898.29 examples/s]35851 examples [00:39, 899.61 examples/s]35944 examples [00:39, 906.69 examples/s]36035 examples [00:39, 907.31 examples/s]36126 examples [00:39, 895.71 examples/s]36217 examples [00:39, 898.25 examples/s]36307 examples [00:39, 898.67 examples/s]36397 examples [00:39, 898.45 examples/s]36489 examples [00:39, 903.03 examples/s]36580 examples [00:39, 897.53 examples/s]36672 examples [00:39, 902.26 examples/s]36763 examples [00:40, 903.03 examples/s]36854 examples [00:40, 876.87 examples/s]36943 examples [00:40, 879.62 examples/s]37032 examples [00:40, 882.00 examples/s]37122 examples [00:40, 886.86 examples/s]37213 examples [00:40, 893.14 examples/s]37306 examples [00:40, 902.91 examples/s]37402 examples [00:40, 916.54 examples/s]37497 examples [00:40, 925.62 examples/s]37590 examples [00:40, 915.90 examples/s]37682 examples [00:41, 914.07 examples/s]37774 examples [00:41, 912.09 examples/s]37868 examples [00:41, 918.68 examples/s]37960 examples [00:41, 911.14 examples/s]38053 examples [00:41, 916.44 examples/s]38146 examples [00:41, 919.86 examples/s]38242 examples [00:41, 929.46 examples/s]38337 examples [00:41, 934.40 examples/s]38431 examples [00:41, 935.64 examples/s]38526 examples [00:41, 937.73 examples/s]38622 examples [00:42, 942.34 examples/s]38720 examples [00:42, 951.86 examples/s]38817 examples [00:42, 956.28 examples/s]38913 examples [00:42, 949.79 examples/s]39010 examples [00:42, 955.72 examples/s]39108 examples [00:42, 960.72 examples/s]39205 examples [00:42, 961.69 examples/s]39302 examples [00:42, 957.29 examples/s]39398 examples [00:42, 952.54 examples/s]39495 examples [00:43, 956.35 examples/s]39591 examples [00:43, 949.28 examples/s]39686 examples [00:43, 932.31 examples/s]39780 examples [00:43, 914.24 examples/s]39872 examples [00:43, 879.14 examples/s]39966 examples [00:43, 894.78 examples/s]40056 examples [00:43, 851.13 examples/s]40154 examples [00:43, 885.36 examples/s]40251 examples [00:43, 906.89 examples/s]40348 examples [00:43, 922.90 examples/s]40443 examples [00:44, 928.32 examples/s]40543 examples [00:44, 948.04 examples/s]40639 examples [00:44, 942.82 examples/s]40736 examples [00:44, 948.68 examples/s]40832 examples [00:44, 946.29 examples/s]40929 examples [00:44, 952.75 examples/s]41025 examples [00:44, 953.01 examples/s]41121 examples [00:44, 951.03 examples/s]41219 examples [00:44, 958.51 examples/s]41315 examples [00:44, 949.77 examples/s]41412 examples [00:45, 954.52 examples/s]41508 examples [00:45, 947.42 examples/s]41607 examples [00:45, 957.28 examples/s]41703 examples [00:45, 953.34 examples/s]41799 examples [00:45, 943.83 examples/s]41894 examples [00:45, 944.54 examples/s]41989 examples [00:45, 945.87 examples/s]42084 examples [00:45, 931.26 examples/s]42178 examples [00:45, 924.01 examples/s]42271 examples [00:46, 887.08 examples/s]42363 examples [00:46, 894.40 examples/s]42459 examples [00:46, 911.56 examples/s]42555 examples [00:46, 923.44 examples/s]42653 examples [00:46, 938.32 examples/s]42748 examples [00:46, 929.73 examples/s]42842 examples [00:46, 874.20 examples/s]42936 examples [00:46, 891.55 examples/s]43030 examples [00:46, 904.25 examples/s]43122 examples [00:46, 908.12 examples/s]43214 examples [00:47, 901.41 examples/s]43309 examples [00:47, 912.79 examples/s]43401 examples [00:47, 909.50 examples/s]43493 examples [00:47, 910.72 examples/s]43589 examples [00:47, 923.78 examples/s]43682 examples [00:47, 921.05 examples/s]43780 examples [00:47, 934.91 examples/s]43877 examples [00:47, 943.80 examples/s]43972 examples [00:47, 944.91 examples/s]44069 examples [00:47, 950.54 examples/s]44165 examples [00:48, 949.34 examples/s]44260 examples [00:48, 943.40 examples/s]44356 examples [00:48, 948.04 examples/s]44453 examples [00:48, 953.90 examples/s]44551 examples [00:48, 960.12 examples/s]44648 examples [00:48, 954.35 examples/s]44744 examples [00:48, 947.83 examples/s]44839 examples [00:48, 934.81 examples/s]44934 examples [00:48, 937.98 examples/s]45028 examples [00:48, 926.37 examples/s]45121 examples [00:49, 920.52 examples/s]45214 examples [00:49, 921.72 examples/s]45308 examples [00:49, 926.47 examples/s]45401 examples [00:49, 926.75 examples/s]45494 examples [00:49, 921.75 examples/s]45587 examples [00:49, 916.08 examples/s]45682 examples [00:49, 925.99 examples/s]45775 examples [00:49, 925.87 examples/s]45871 examples [00:49, 933.48 examples/s]45966 examples [00:49, 937.42 examples/s]46062 examples [00:50, 942.48 examples/s]46157 examples [00:50, 939.16 examples/s]46251 examples [00:50, 928.56 examples/s]46347 examples [00:50, 936.72 examples/s]46442 examples [00:50, 938.85 examples/s]46536 examples [00:50, 935.37 examples/s]46630 examples [00:50, 923.78 examples/s]46723 examples [00:50, 916.58 examples/s]46815 examples [00:50, 912.04 examples/s]46907 examples [00:51, 887.73 examples/s]46996 examples [00:51, 858.28 examples/s]47085 examples [00:51, 865.66 examples/s]47173 examples [00:51, 868.22 examples/s]47263 examples [00:51, 875.26 examples/s]47354 examples [00:51, 882.07 examples/s]47443 examples [00:51, 874.13 examples/s]47534 examples [00:51, 882.24 examples/s]47626 examples [00:51, 891.13 examples/s]47716 examples [00:51, 867.63 examples/s]47803 examples [00:52, 854.28 examples/s]47891 examples [00:52, 860.87 examples/s]47983 examples [00:52, 876.45 examples/s]48078 examples [00:52, 896.60 examples/s]48171 examples [00:52, 904.68 examples/s]48267 examples [00:52, 918.91 examples/s]48360 examples [00:52, 909.22 examples/s]48452 examples [00:52, 910.68 examples/s]48546 examples [00:52, 917.23 examples/s]48638 examples [00:52, 911.40 examples/s]48731 examples [00:53, 915.85 examples/s]48823 examples [00:53, 903.00 examples/s]48918 examples [00:53, 914.67 examples/s]49012 examples [00:53, 920.80 examples/s]49105 examples [00:53, 904.65 examples/s]49196 examples [00:53, 879.19 examples/s]49288 examples [00:53, 890.26 examples/s]49382 examples [00:53, 903.83 examples/s]49475 examples [00:53, 910.19 examples/s]49567 examples [00:53, 907.75 examples/s]49664 examples [00:54, 923.28 examples/s]49760 examples [00:54, 932.75 examples/s]49856 examples [00:54, 939.12 examples/s]49951 examples [00:54, 936.68 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 6962/50000 [00:00<00:00, 69617.58 examples/s] 36%|███▌      | 18049/50000 [00:00<00:00, 78364.51 examples/s] 59%|█████▉    | 29486/50000 [00:00<00:00, 86536.70 examples/s] 83%|████████▎ | 41343/50000 [00:00<00:00, 94168.71 examples/s]                                                               0 examples [00:00, ? examples/s]76 examples [00:00, 755.62 examples/s]168 examples [00:00, 797.30 examples/s]261 examples [00:00, 830.84 examples/s]352 examples [00:00, 851.52 examples/s]453 examples [00:00, 892.87 examples/s]550 examples [00:00, 913.03 examples/s]645 examples [00:00, 923.13 examples/s]742 examples [00:00, 935.55 examples/s]840 examples [00:00, 947.37 examples/s]933 examples [00:01, 933.95 examples/s]1025 examples [00:01, 928.77 examples/s]1121 examples [00:01, 936.03 examples/s]1220 examples [00:01, 948.83 examples/s]1316 examples [00:01, 950.11 examples/s]1413 examples [00:01, 955.93 examples/s]1510 examples [00:01, 959.21 examples/s]1606 examples [00:01, 909.19 examples/s]1698 examples [00:01, 908.04 examples/s]1790 examples [00:01, 910.73 examples/s]1882 examples [00:02, 907.81 examples/s]1974 examples [00:02, 910.46 examples/s]2066 examples [00:02, 903.11 examples/s]2158 examples [00:02, 907.24 examples/s]2249 examples [00:02, 904.93 examples/s]2340 examples [00:02, 905.05 examples/s]2431 examples [00:02, 906.04 examples/s]2522 examples [00:02, 902.97 examples/s]2613 examples [00:02, 902.85 examples/s]2705 examples [00:02, 905.55 examples/s]2796 examples [00:03, 905.31 examples/s]2894 examples [00:03, 924.49 examples/s]2991 examples [00:03, 935.31 examples/s]3090 examples [00:03, 948.64 examples/s]3188 examples [00:03, 955.55 examples/s]3288 examples [00:03, 968.33 examples/s]3389 examples [00:03, 978.26 examples/s]3487 examples [00:03, 966.64 examples/s]3588 examples [00:03, 978.92 examples/s]3687 examples [00:03, 972.12 examples/s]3785 examples [00:04, 964.64 examples/s]3882 examples [00:04, 957.09 examples/s]3978 examples [00:04, 957.05 examples/s]4075 examples [00:04, 958.85 examples/s]4172 examples [00:04, 961.78 examples/s]4269 examples [00:04, 960.67 examples/s]4366 examples [00:04, 947.31 examples/s]4461 examples [00:04, 944.56 examples/s]4557 examples [00:04, 947.72 examples/s]4652 examples [00:04, 940.68 examples/s]4747 examples [00:05, 909.27 examples/s]4839 examples [00:05, 905.03 examples/s]4930 examples [00:05, 900.04 examples/s]5022 examples [00:05, 904.79 examples/s]5115 examples [00:05, 911.02 examples/s]5207 examples [00:05, 912.35 examples/s]5300 examples [00:05, 914.02 examples/s]5392 examples [00:05, 876.11 examples/s]5485 examples [00:05, 891.19 examples/s]5575 examples [00:06, 888.51 examples/s]5668 examples [00:06, 898.45 examples/s]5761 examples [00:06, 906.40 examples/s]5852 examples [00:06, 905.88 examples/s]5943 examples [00:06, 902.56 examples/s]6035 examples [00:06, 905.71 examples/s]6126 examples [00:06, 898.80 examples/s]6217 examples [00:06, 901.07 examples/s]6308 examples [00:06, 901.51 examples/s]6399 examples [00:06, 902.36 examples/s]6491 examples [00:07, 905.29 examples/s]6582 examples [00:07, 906.04 examples/s]6676 examples [00:07, 915.06 examples/s]6771 examples [00:07, 923.41 examples/s]6865 examples [00:07, 927.47 examples/s]6960 examples [00:07, 934.09 examples/s]7054 examples [00:07, 924.91 examples/s]7149 examples [00:07, 931.83 examples/s]7244 examples [00:07, 935.59 examples/s]7344 examples [00:07, 951.49 examples/s]7441 examples [00:08, 955.24 examples/s]7540 examples [00:08, 963.18 examples/s]7637 examples [00:08, 921.87 examples/s]7730 examples [00:08, 895.41 examples/s]7823 examples [00:08, 902.85 examples/s]7915 examples [00:08, 906.70 examples/s]8006 examples [00:08, 900.48 examples/s]8097 examples [00:08, 901.94 examples/s]8188 examples [00:08, 897.96 examples/s]8282 examples [00:08, 908.07 examples/s]8373 examples [00:09, 894.72 examples/s]8463 examples [00:09, 885.00 examples/s]8557 examples [00:09, 899.77 examples/s]8648 examples [00:09, 896.91 examples/s]8743 examples [00:09, 909.19 examples/s]8838 examples [00:09, 919.86 examples/s]8933 examples [00:09, 926.14 examples/s]9028 examples [00:09, 932.96 examples/s]9123 examples [00:09, 936.36 examples/s]9217 examples [00:09, 932.51 examples/s]9311 examples [00:10, 924.81 examples/s]9404 examples [00:10, 905.60 examples/s]9495 examples [00:10, 903.39 examples/s]9586 examples [00:10, 892.83 examples/s]9676 examples [00:10, 886.87 examples/s]9768 examples [00:10, 895.10 examples/s]9858 examples [00:10, 872.73 examples/s]9948 examples [00:10, 879.63 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteJ677CZ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteJ677CZ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feb6ef43a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feb6ef43a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feb6ef43a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7febba8b0cc0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feaf926bcf8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feb6ef43a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feb6ef43a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feb6ef43a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feb56df5358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feb56df51d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 10.90 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 21.63 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.90 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.90 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.51 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.51 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.95 file/s]2020-08-01 06:10:47.928207: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-01 06:10:47.932298: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-01 06:10:47.932448: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5579d43c5220 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-01 06:10:47.932465: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 469394.26it/s] 74%|███████▍  | 7323648/9912422 [00:00<00:03, 668662.07it/s]9920512it [00:00, 41384058.66it/s]                           
0it [00:00, ?it/s]32768it [00:00, 621932.30it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160549.02it/s]1654784it [00:00, 11346558.51it/s]                         
0it [00:00, ?it/s]8192it [00:00, 200362.35it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

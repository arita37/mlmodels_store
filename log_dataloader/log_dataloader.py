
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fe2ddb39598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fe2ddb39598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe3487eb400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe3487eb400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe367a07ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe367a07ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe2f5b260d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe2f5b260d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe2f5b260d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3710976/9912422 [00:00<00:00, 37101303.57it/s]9920512it [00:00, 36363260.96it/s]                             
0it [00:00, ?it/s]32768it [00:00, 362140.80it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 152164.86it/s]1654784it [00:00, 10929960.73it/s]                         
0it [00:00, ?it/s]8192it [00:00, 146957.07it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe2f3cf4710>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe2f3dbf978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe2f5b1ed08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe2f5b1ed08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe2f5b1ed08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:23,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:22,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:22,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:21,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:57,  2.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:57,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:57,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:56,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:56,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:56,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:55,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:39,  3.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:39,  3.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:39,  3.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:39,  3.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:38,  3.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:38,  3.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:27,  5.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:27,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:27,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:27,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:26,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:26,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:18,  7.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:18,  7.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:18,  7.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:18,  7.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:13,  9.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:13,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:13,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:13,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:13,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:13,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:12,  9.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:09, 12.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:09, 12.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:09, 12.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:09, 12.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:09, 12.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 12.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:09, 12.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 16.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 16.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 16.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 16.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 16.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 16.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 16.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 21.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 21.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 21.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 21.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 21.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 21.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 21.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 25.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 25.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 31.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 37.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 37.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 37.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 37.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 37.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 37.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 37.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 37.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 42.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 42.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 42.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 42.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 42.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 42.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 42.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 42.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 47.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 47.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 47.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 47.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 47.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 47.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 47.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 47.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 52.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 52.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 52.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 52.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 52.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 52.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 52.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 52.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 55.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 57.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 57.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 57.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 57.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 57.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 57.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 57.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 57.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 59.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 61.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 61.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 61.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 61.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 61.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 61.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 61.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 61.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 64.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 64.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 64.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 64.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 64.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 64.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 64.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 64.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 64.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 64.99 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 64.99 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 65.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 65.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 65.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 65.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 65.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 65.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 65.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 65.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 65.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 68.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.15s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.15s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.15s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 68.51 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.90s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.15s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 68.51 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.90s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.90s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 27.45 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.90s/ url]
0 examples [00:00, ? examples/s]2020-08-19 18:08:14.469965: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-19 18:08:14.483056: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-19 18:08:14.483271: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a9dd8146d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-19 18:08:14.483293: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
26 examples [00:00, 258.19 examples/s]111 examples [00:00, 326.05 examples/s]193 examples [00:00, 397.67 examples/s]279 examples [00:00, 473.56 examples/s]363 examples [00:00, 543.71 examples/s]436 examples [00:00, 587.23 examples/s]517 examples [00:00, 639.04 examples/s]606 examples [00:00, 697.95 examples/s]696 examples [00:00, 747.30 examples/s]781 examples [00:01, 775.17 examples/s]866 examples [00:01, 795.77 examples/s]949 examples [00:01, 780.29 examples/s]1030 examples [00:01, 785.78 examples/s]1111 examples [00:01, 791.06 examples/s]1199 examples [00:01, 815.49 examples/s]1290 examples [00:01, 840.06 examples/s]1378 examples [00:01, 850.18 examples/s]1468 examples [00:01, 863.30 examples/s]1556 examples [00:01, 865.28 examples/s]1643 examples [00:02, 859.26 examples/s]1730 examples [00:02, 861.02 examples/s]1817 examples [00:02, 861.48 examples/s]1904 examples [00:02, 835.23 examples/s]1989 examples [00:02, 837.36 examples/s]2073 examples [00:02, 833.16 examples/s]2157 examples [00:02, 823.95 examples/s]2240 examples [00:02, 808.07 examples/s]2321 examples [00:02, 808.39 examples/s]2410 examples [00:02, 831.16 examples/s]2496 examples [00:03, 839.46 examples/s]2587 examples [00:03, 858.08 examples/s]2674 examples [00:03, 834.06 examples/s]2758 examples [00:03, 833.66 examples/s]2844 examples [00:03, 838.80 examples/s]2933 examples [00:03, 853.31 examples/s]3024 examples [00:03, 869.07 examples/s]3112 examples [00:03, 859.51 examples/s]3199 examples [00:03, 852.07 examples/s]3285 examples [00:03, 830.65 examples/s]3369 examples [00:04, 832.86 examples/s]3453 examples [00:04, 821.51 examples/s]3536 examples [00:04, 821.16 examples/s]3627 examples [00:04, 843.49 examples/s]3714 examples [00:04, 849.86 examples/s]3800 examples [00:04, 845.37 examples/s]3886 examples [00:04, 849.11 examples/s]3976 examples [00:04, 862.06 examples/s]4064 examples [00:04, 866.75 examples/s]4153 examples [00:04, 872.36 examples/s]4241 examples [00:05, 871.34 examples/s]4329 examples [00:05, 834.92 examples/s]4413 examples [00:05, 768.45 examples/s]4495 examples [00:05, 782.45 examples/s]4581 examples [00:05, 803.30 examples/s]4670 examples [00:05, 825.51 examples/s]4754 examples [00:05, 812.24 examples/s]4836 examples [00:05, 806.44 examples/s]4919 examples [00:05, 810.97 examples/s]5001 examples [00:06, 812.96 examples/s]5083 examples [00:06, 812.76 examples/s]5168 examples [00:06, 823.56 examples/s]5255 examples [00:06, 835.05 examples/s]5339 examples [00:06, 824.47 examples/s]5425 examples [00:06, 831.15 examples/s]5509 examples [00:06, 829.91 examples/s]5593 examples [00:06, 819.84 examples/s]5676 examples [00:06, 811.48 examples/s]5762 examples [00:06, 823.28 examples/s]5845 examples [00:07, 818.62 examples/s]5927 examples [00:07, 799.79 examples/s]6008 examples [00:07, 790.69 examples/s]6089 examples [00:07, 796.20 examples/s]6179 examples [00:07, 822.69 examples/s]6264 examples [00:07, 830.64 examples/s]6352 examples [00:07, 843.06 examples/s]6437 examples [00:07, 843.18 examples/s]6524 examples [00:07, 848.52 examples/s]6609 examples [00:08, 833.92 examples/s]6693 examples [00:08, 831.49 examples/s]6777 examples [00:08, 816.13 examples/s]6859 examples [00:08, 777.13 examples/s]6943 examples [00:08, 793.94 examples/s]7028 examples [00:08, 809.33 examples/s]7113 examples [00:08, 818.61 examples/s]7196 examples [00:08, 797.06 examples/s]7282 examples [00:08, 813.44 examples/s]7366 examples [00:08, 820.12 examples/s]7451 examples [00:09, 827.52 examples/s]7537 examples [00:09, 836.01 examples/s]7622 examples [00:09, 839.88 examples/s]7709 examples [00:09, 848.09 examples/s]7794 examples [00:09, 844.78 examples/s]7879 examples [00:09, 842.14 examples/s]7964 examples [00:09, 841.05 examples/s]8049 examples [00:09, 838.94 examples/s]8135 examples [00:09, 844.97 examples/s]8220 examples [00:09, 844.64 examples/s]8305 examples [00:10, 822.31 examples/s]8388 examples [00:10, 812.04 examples/s]8470 examples [00:10, 788.36 examples/s]8555 examples [00:10, 803.55 examples/s]8644 examples [00:10, 826.30 examples/s]8729 examples [00:10, 833.22 examples/s]8816 examples [00:10, 843.56 examples/s]8904 examples [00:10, 852.01 examples/s]8990 examples [00:10, 852.00 examples/s]9076 examples [00:10, 847.31 examples/s]9162 examples [00:11, 849.80 examples/s]9248 examples [00:11, 844.17 examples/s]9336 examples [00:11, 853.66 examples/s]9422 examples [00:11, 804.63 examples/s]9504 examples [00:11, 783.79 examples/s]9589 examples [00:11, 801.75 examples/s]9675 examples [00:11, 816.26 examples/s]9760 examples [00:11, 825.82 examples/s]9843 examples [00:11, 824.61 examples/s]9928 examples [00:12, 831.40 examples/s]10012 examples [00:12, 781.66 examples/s]10097 examples [00:12, 799.35 examples/s]10178 examples [00:12, 796.15 examples/s]10259 examples [00:12, 786.99 examples/s]10339 examples [00:12, 782.43 examples/s]10425 examples [00:12, 801.86 examples/s]10506 examples [00:12, 796.64 examples/s]10591 examples [00:12, 811.24 examples/s]10681 examples [00:12, 834.89 examples/s]10769 examples [00:13, 845.51 examples/s]10856 examples [00:13, 851.68 examples/s]10942 examples [00:13, 836.73 examples/s]11026 examples [00:13, 825.20 examples/s]11110 examples [00:13, 828.36 examples/s]11198 examples [00:13, 842.03 examples/s]11285 examples [00:13, 847.96 examples/s]11370 examples [00:13, 828.00 examples/s]11458 examples [00:13, 842.38 examples/s]11547 examples [00:13, 853.32 examples/s]11633 examples [00:14, 855.18 examples/s]11719 examples [00:14, 853.44 examples/s]11805 examples [00:14, 853.97 examples/s]11891 examples [00:14, 817.73 examples/s]11974 examples [00:14, 793.40 examples/s]12063 examples [00:14, 817.63 examples/s]12151 examples [00:14, 832.74 examples/s]12238 examples [00:14, 841.72 examples/s]12323 examples [00:14, 842.09 examples/s]12409 examples [00:15, 847.26 examples/s]12494 examples [00:15, 839.21 examples/s]12579 examples [00:15, 826.93 examples/s]12664 examples [00:15, 832.65 examples/s]12748 examples [00:15, 827.11 examples/s]12833 examples [00:15, 833.19 examples/s]12919 examples [00:15, 840.50 examples/s]13004 examples [00:15, 836.01 examples/s]13090 examples [00:15, 842.40 examples/s]13175 examples [00:15, 842.81 examples/s]13260 examples [00:16, 839.72 examples/s]13345 examples [00:16, 842.13 examples/s]13430 examples [00:16, 838.80 examples/s]13515 examples [00:16, 840.74 examples/s]13604 examples [00:16, 852.60 examples/s]13694 examples [00:16, 864.38 examples/s]13781 examples [00:16, 864.13 examples/s]13869 examples [00:16, 868.15 examples/s]13956 examples [00:16, 825.28 examples/s]14039 examples [00:16, 798.39 examples/s]14127 examples [00:17, 819.85 examples/s]14214 examples [00:17, 831.41 examples/s]14300 examples [00:17, 837.65 examples/s]14385 examples [00:17, 767.12 examples/s]14474 examples [00:17, 798.81 examples/s]14562 examples [00:17, 820.50 examples/s]14649 examples [00:17, 832.91 examples/s]14736 examples [00:17, 841.31 examples/s]14821 examples [00:17, 838.58 examples/s]14906 examples [00:18, 837.55 examples/s]14991 examples [00:18, 838.87 examples/s]15076 examples [00:18, 834.55 examples/s]15160 examples [00:18, 832.30 examples/s]15248 examples [00:18, 843.49 examples/s]15337 examples [00:18, 856.86 examples/s]15425 examples [00:18, 861.49 examples/s]15512 examples [00:18, 863.03 examples/s]15599 examples [00:18, 856.37 examples/s]15685 examples [00:18, 855.19 examples/s]15771 examples [00:19, 842.43 examples/s]15856 examples [00:19, 842.69 examples/s]15942 examples [00:19, 845.59 examples/s]16027 examples [00:19, 839.60 examples/s]16111 examples [00:19, 835.94 examples/s]16195 examples [00:19, 829.43 examples/s]16283 examples [00:19, 843.43 examples/s]16368 examples [00:19, 841.02 examples/s]16453 examples [00:19, 821.57 examples/s]16536 examples [00:19, 818.98 examples/s]16624 examples [00:20, 835.48 examples/s]16708 examples [00:20, 835.99 examples/s]16792 examples [00:20, 836.20 examples/s]16876 examples [00:20, 774.56 examples/s]16962 examples [00:20, 796.58 examples/s]17050 examples [00:20, 818.46 examples/s]17139 examples [00:20, 836.66 examples/s]17224 examples [00:20, 839.44 examples/s]17313 examples [00:20, 852.80 examples/s]17402 examples [00:20, 862.43 examples/s]17489 examples [00:21, 851.30 examples/s]17575 examples [00:21, 848.01 examples/s]17660 examples [00:21, 847.72 examples/s]17748 examples [00:21, 854.99 examples/s]17834 examples [00:21, 850.94 examples/s]17921 examples [00:21, 854.88 examples/s]18007 examples [00:21, 838.73 examples/s]18092 examples [00:21, 840.86 examples/s]18177 examples [00:21, 834.69 examples/s]18261 examples [00:22, 813.21 examples/s]18343 examples [00:22, 811.09 examples/s]18429 examples [00:22, 823.30 examples/s]18513 examples [00:22, 826.54 examples/s]18596 examples [00:22, 825.21 examples/s]18679 examples [00:22, 824.68 examples/s]18764 examples [00:22, 830.66 examples/s]18850 examples [00:22, 837.83 examples/s]18934 examples [00:22, 822.81 examples/s]19018 examples [00:22, 827.88 examples/s]19103 examples [00:23, 833.97 examples/s]19191 examples [00:23, 847.19 examples/s]19277 examples [00:23, 850.00 examples/s]19364 examples [00:23, 854.00 examples/s]19450 examples [00:23, 800.99 examples/s]19533 examples [00:23, 807.11 examples/s]19621 examples [00:23, 826.67 examples/s]19708 examples [00:23, 838.69 examples/s]19793 examples [00:23, 839.36 examples/s]19883 examples [00:23, 854.35 examples/s]19974 examples [00:24, 869.95 examples/s]20062 examples [00:24, 817.87 examples/s]20153 examples [00:24, 842.06 examples/s]20241 examples [00:24, 852.81 examples/s]20331 examples [00:24, 865.10 examples/s]20418 examples [00:24, 803.97 examples/s]20501 examples [00:24, 811.11 examples/s]20585 examples [00:24, 817.48 examples/s]20668 examples [00:24, 816.32 examples/s]20751 examples [00:25, 814.01 examples/s]20835 examples [00:25, 819.38 examples/s]20922 examples [00:25, 832.74 examples/s]21006 examples [00:25, 834.77 examples/s]21090 examples [00:25, 833.42 examples/s]21176 examples [00:25, 841.04 examples/s]21261 examples [00:25, 838.91 examples/s]21345 examples [00:25, 838.40 examples/s]21429 examples [00:25, 832.03 examples/s]21516 examples [00:25, 840.70 examples/s]21601 examples [00:26, 826.87 examples/s]21686 examples [00:26, 833.32 examples/s]21770 examples [00:26, 820.33 examples/s]21853 examples [00:26, 822.22 examples/s]21936 examples [00:26, 759.90 examples/s]22022 examples [00:26, 786.75 examples/s]22102 examples [00:26, 787.79 examples/s]22190 examples [00:26, 811.34 examples/s]22272 examples [00:26, 795.01 examples/s]22354 examples [00:26, 800.56 examples/s]22441 examples [00:27, 819.50 examples/s]22527 examples [00:27, 829.21 examples/s]22617 examples [00:27, 848.11 examples/s]22703 examples [00:27, 846.08 examples/s]22788 examples [00:27, 832.01 examples/s]22874 examples [00:27, 839.10 examples/s]22961 examples [00:27, 845.68 examples/s]23047 examples [00:27, 848.62 examples/s]23132 examples [00:27, 828.63 examples/s]23216 examples [00:28, 827.41 examples/s]23302 examples [00:28, 834.67 examples/s]23386 examples [00:28, 833.23 examples/s]23470 examples [00:28, 808.37 examples/s]23556 examples [00:28, 820.40 examples/s]23639 examples [00:28, 818.94 examples/s]23724 examples [00:28, 827.70 examples/s]23811 examples [00:28, 839.15 examples/s]23898 examples [00:28, 847.25 examples/s]23983 examples [00:28, 833.32 examples/s]24067 examples [00:29, 832.61 examples/s]24157 examples [00:29, 850.43 examples/s]24245 examples [00:29, 856.47 examples/s]24331 examples [00:29, 677.08 examples/s]24415 examples [00:29, 717.64 examples/s]24498 examples [00:29, 746.67 examples/s]24580 examples [00:29, 766.52 examples/s]24660 examples [00:29, 771.36 examples/s]24749 examples [00:29, 802.43 examples/s]24835 examples [00:30, 817.44 examples/s]24922 examples [00:30, 831.88 examples/s]25014 examples [00:30, 856.13 examples/s]25101 examples [00:30, 852.80 examples/s]25187 examples [00:30, 839.85 examples/s]25273 examples [00:30, 843.02 examples/s]25359 examples [00:30, 847.10 examples/s]25446 examples [00:30, 853.11 examples/s]25532 examples [00:30, 840.31 examples/s]25617 examples [00:30, 826.95 examples/s]25705 examples [00:31, 840.70 examples/s]25790 examples [00:31, 842.02 examples/s]25879 examples [00:31, 855.86 examples/s]25968 examples [00:31, 863.67 examples/s]26055 examples [00:31, 860.97 examples/s]26142 examples [00:31, 863.63 examples/s]26229 examples [00:31, 853.15 examples/s]26316 examples [00:31, 857.02 examples/s]26402 examples [00:31, 854.41 examples/s]26488 examples [00:31, 852.31 examples/s]26580 examples [00:32, 869.39 examples/s]26669 examples [00:32, 873.49 examples/s]26757 examples [00:32, 867.54 examples/s]26844 examples [00:32, 853.03 examples/s]26930 examples [00:32, 825.83 examples/s]27013 examples [00:32, 808.71 examples/s]27095 examples [00:32, 803.54 examples/s]27178 examples [00:32, 808.73 examples/s]27262 examples [00:32, 815.81 examples/s]27349 examples [00:32, 831.24 examples/s]27433 examples [00:33, 825.67 examples/s]27519 examples [00:33, 832.91 examples/s]27606 examples [00:33, 842.86 examples/s]27691 examples [00:33, 824.74 examples/s]27774 examples [00:33, 811.96 examples/s]27858 examples [00:33, 819.31 examples/s]27944 examples [00:33, 827.60 examples/s]28030 examples [00:33, 835.70 examples/s]28119 examples [00:33, 849.76 examples/s]28207 examples [00:34, 858.17 examples/s]28294 examples [00:34, 858.28 examples/s]28380 examples [00:34, 857.24 examples/s]28467 examples [00:34, 859.34 examples/s]28553 examples [00:34, 848.25 examples/s]28638 examples [00:34, 833.81 examples/s]28725 examples [00:34, 841.50 examples/s]28811 examples [00:34, 845.08 examples/s]28896 examples [00:34, 845.74 examples/s]28981 examples [00:34, 833.74 examples/s]29068 examples [00:35, 844.22 examples/s]29153 examples [00:35, 836.89 examples/s]29240 examples [00:35, 845.51 examples/s]29325 examples [00:35, 841.14 examples/s]29410 examples [00:35, 828.37 examples/s]29493 examples [00:35, 805.76 examples/s]29578 examples [00:35, 815.96 examples/s]29662 examples [00:35, 820.10 examples/s]29748 examples [00:35, 830.98 examples/s]29834 examples [00:35, 838.45 examples/s]29918 examples [00:36, 815.91 examples/s]30001 examples [00:36, 767.97 examples/s]30091 examples [00:36, 801.22 examples/s]30180 examples [00:36, 824.93 examples/s]30268 examples [00:36, 838.19 examples/s]30353 examples [00:36, 819.18 examples/s]30437 examples [00:36, 824.21 examples/s]30521 examples [00:36, 828.41 examples/s]30605 examples [00:36, 823.91 examples/s]30690 examples [00:37, 829.33 examples/s]30774 examples [00:37, 821.72 examples/s]30857 examples [00:37, 821.96 examples/s]30941 examples [00:37, 825.47 examples/s]31024 examples [00:37, 820.56 examples/s]31108 examples [00:37, 825.92 examples/s]31193 examples [00:37, 830.59 examples/s]31278 examples [00:37, 836.26 examples/s]31362 examples [00:37, 836.08 examples/s]31447 examples [00:37, 839.92 examples/s]31532 examples [00:38, 840.40 examples/s]31617 examples [00:38, 835.81 examples/s]31705 examples [00:38, 846.12 examples/s]31790 examples [00:38, 846.03 examples/s]31877 examples [00:38, 850.87 examples/s]31963 examples [00:38, 769.83 examples/s]32042 examples [00:38, 772.71 examples/s]32121 examples [00:38, 753.75 examples/s]32205 examples [00:38, 776.60 examples/s]32292 examples [00:38, 799.73 examples/s]32377 examples [00:39, 813.22 examples/s]32461 examples [00:39, 815.16 examples/s]32550 examples [00:39, 836.01 examples/s]32635 examples [00:39, 830.54 examples/s]32721 examples [00:39, 838.01 examples/s]32806 examples [00:39, 841.14 examples/s]32891 examples [00:39, 842.76 examples/s]32979 examples [00:39, 852.23 examples/s]33065 examples [00:39, 791.10 examples/s]33151 examples [00:40, 808.94 examples/s]33236 examples [00:40, 819.65 examples/s]33319 examples [00:40, 820.19 examples/s]33406 examples [00:40, 833.01 examples/s]33494 examples [00:40, 844.67 examples/s]33582 examples [00:40, 853.63 examples/s]33672 examples [00:40, 864.47 examples/s]33759 examples [00:40, 863.13 examples/s]33848 examples [00:40, 868.47 examples/s]33936 examples [00:40, 871.10 examples/s]34024 examples [00:41, 851.73 examples/s]34110 examples [00:41, 820.29 examples/s]34193 examples [00:41, 822.60 examples/s]34276 examples [00:41, 824.40 examples/s]34359 examples [00:41, 821.17 examples/s]34442 examples [00:41, 753.66 examples/s]34527 examples [00:41, 773.54 examples/s]34616 examples [00:41, 804.71 examples/s]34707 examples [00:41, 832.92 examples/s]34792 examples [00:41, 829.06 examples/s]34876 examples [00:42, 820.21 examples/s]34959 examples [00:42, 813.88 examples/s]35047 examples [00:42, 829.99 examples/s]35134 examples [00:42, 841.50 examples/s]35221 examples [00:42, 846.28 examples/s]35306 examples [00:42, 840.50 examples/s]35391 examples [00:42, 819.37 examples/s]35478 examples [00:42, 832.41 examples/s]35565 examples [00:42, 840.80 examples/s]35654 examples [00:43, 853.25 examples/s]35740 examples [00:43, 849.75 examples/s]35827 examples [00:43, 854.37 examples/s]35916 examples [00:43, 862.99 examples/s]36003 examples [00:43, 864.27 examples/s]36090 examples [00:43, 856.60 examples/s]36176 examples [00:43, 852.87 examples/s]36262 examples [00:43, 847.66 examples/s]36347 examples [00:43, 835.80 examples/s]36431 examples [00:43, 813.93 examples/s]36517 examples [00:44, 825.23 examples/s]36605 examples [00:44, 840.85 examples/s]36693 examples [00:44, 850.78 examples/s]36779 examples [00:44, 844.29 examples/s]36865 examples [00:44, 846.90 examples/s]36950 examples [00:44, 842.35 examples/s]37035 examples [00:44, 795.39 examples/s]37116 examples [00:44, 798.31 examples/s]37198 examples [00:44, 802.57 examples/s]37279 examples [00:45, 735.81 examples/s]37366 examples [00:45, 770.77 examples/s]37453 examples [00:45, 796.27 examples/s]37540 examples [00:45, 814.09 examples/s]37623 examples [00:45, 817.67 examples/s]37712 examples [00:45, 836.41 examples/s]37798 examples [00:45, 842.18 examples/s]37883 examples [00:45, 821.58 examples/s]37971 examples [00:45, 836.53 examples/s]38058 examples [00:45, 843.33 examples/s]38143 examples [00:46, 834.04 examples/s]38227 examples [00:46, 825.16 examples/s]38312 examples [00:46, 831.45 examples/s]38397 examples [00:46, 835.43 examples/s]38483 examples [00:46, 840.35 examples/s]38569 examples [00:46, 846.05 examples/s]38654 examples [00:46, 837.56 examples/s]38739 examples [00:46, 840.84 examples/s]38824 examples [00:46, 837.26 examples/s]38908 examples [00:46, 834.53 examples/s]38993 examples [00:47, 838.77 examples/s]39077 examples [00:47, 823.81 examples/s]39160 examples [00:47, 790.32 examples/s]39240 examples [00:47, 776.72 examples/s]39322 examples [00:47, 788.30 examples/s]39411 examples [00:47, 814.13 examples/s]39493 examples [00:47, 770.17 examples/s]39576 examples [00:47, 785.54 examples/s]39656 examples [00:47, 779.24 examples/s]39736 examples [00:47, 784.24 examples/s]39818 examples [00:48, 793.04 examples/s]39903 examples [00:48, 808.83 examples/s]39992 examples [00:48, 829.58 examples/s]40076 examples [00:48, 787.99 examples/s]40164 examples [00:48, 812.52 examples/s]40246 examples [00:48, 812.21 examples/s]40329 examples [00:48, 816.62 examples/s]40413 examples [00:48, 820.59 examples/s]40496 examples [00:48, 811.62 examples/s]40579 examples [00:49, 816.80 examples/s]40664 examples [00:49, 825.24 examples/s]40747 examples [00:49, 823.25 examples/s]40830 examples [00:49, 823.21 examples/s]40918 examples [00:49, 838.78 examples/s]41002 examples [00:49, 826.02 examples/s]41085 examples [00:49, 806.39 examples/s]41166 examples [00:49, 801.19 examples/s]41247 examples [00:49, 799.38 examples/s]41332 examples [00:49, 812.93 examples/s]41415 examples [00:50, 815.80 examples/s]41498 examples [00:50, 818.05 examples/s]41580 examples [00:50, 815.40 examples/s]41662 examples [00:50, 814.44 examples/s]41744 examples [00:50, 805.14 examples/s]41829 examples [00:50, 817.71 examples/s]41911 examples [00:50, 758.30 examples/s]41995 examples [00:50, 780.57 examples/s]42078 examples [00:50, 793.91 examples/s]42160 examples [00:50, 800.73 examples/s]42243 examples [00:51, 807.05 examples/s]42328 examples [00:51, 818.16 examples/s]42415 examples [00:51, 830.74 examples/s]42500 examples [00:51, 834.86 examples/s]42586 examples [00:51, 841.97 examples/s]42671 examples [00:51, 828.15 examples/s]42755 examples [00:51, 828.93 examples/s]42840 examples [00:51, 833.38 examples/s]42924 examples [00:51, 827.07 examples/s]43007 examples [00:52, 815.82 examples/s]43089 examples [00:52, 793.02 examples/s]43169 examples [00:52, 783.45 examples/s]43255 examples [00:52, 804.32 examples/s]43337 examples [00:52, 807.63 examples/s]43418 examples [00:52, 801.30 examples/s]43499 examples [00:52, 756.74 examples/s]43576 examples [00:52, 750.17 examples/s]43652 examples [00:52, 712.76 examples/s]43726 examples [00:52, 719.66 examples/s]43800 examples [00:53, 724.46 examples/s]43882 examples [00:53, 748.75 examples/s]43958 examples [00:53, 751.98 examples/s]44037 examples [00:53, 761.65 examples/s]44114 examples [00:53, 754.11 examples/s]44196 examples [00:53, 772.05 examples/s]44274 examples [00:53, 729.79 examples/s]44363 examples [00:53, 770.40 examples/s]44453 examples [00:53, 804.82 examples/s]44539 examples [00:53, 820.28 examples/s]44631 examples [00:54, 847.54 examples/s]44720 examples [00:54, 859.69 examples/s]44811 examples [00:54, 871.86 examples/s]44899 examples [00:54, 863.32 examples/s]44986 examples [00:54, 853.36 examples/s]45072 examples [00:54, 854.47 examples/s]45158 examples [00:54, 844.47 examples/s]45243 examples [00:54, 843.11 examples/s]45328 examples [00:54, 833.82 examples/s]45412 examples [00:55, 833.40 examples/s]45499 examples [00:55, 841.95 examples/s]45584 examples [00:55, 843.96 examples/s]45672 examples [00:55, 853.18 examples/s]45762 examples [00:55, 866.57 examples/s]45849 examples [00:55, 842.41 examples/s]45936 examples [00:55, 848.79 examples/s]46022 examples [00:55, 846.55 examples/s]46110 examples [00:55, 853.78 examples/s]46200 examples [00:55, 864.94 examples/s]46287 examples [00:56, 860.07 examples/s]46374 examples [00:56, 858.22 examples/s]46460 examples [00:56, 844.60 examples/s]46545 examples [00:56, 836.36 examples/s]46629 examples [00:56, 833.20 examples/s]46713 examples [00:56, 818.74 examples/s]46796 examples [00:56, 821.61 examples/s]46879 examples [00:56, 765.93 examples/s]46964 examples [00:56, 786.64 examples/s]47044 examples [00:56, 785.81 examples/s]47124 examples [00:57, 785.83 examples/s]47206 examples [00:57, 793.63 examples/s]47288 examples [00:57, 799.53 examples/s]47371 examples [00:57, 806.01 examples/s]47455 examples [00:57, 813.80 examples/s]47538 examples [00:57, 816.24 examples/s]47620 examples [00:57, 805.93 examples/s]47703 examples [00:57, 810.41 examples/s]47788 examples [00:57, 819.35 examples/s]47871 examples [00:57, 821.02 examples/s]47957 examples [00:58, 831.38 examples/s]48041 examples [00:58, 820.72 examples/s]48124 examples [00:58, 821.32 examples/s]48213 examples [00:58, 840.42 examples/s]48299 examples [00:58, 846.05 examples/s]48388 examples [00:58, 858.09 examples/s]48479 examples [00:58, 870.61 examples/s]48567 examples [00:58, 869.02 examples/s]48654 examples [00:58, 863.90 examples/s]48741 examples [00:59, 862.95 examples/s]48828 examples [00:59, 858.82 examples/s]48914 examples [00:59, 850.47 examples/s]49000 examples [00:59, 849.24 examples/s]49086 examples [00:59, 851.73 examples/s]49173 examples [00:59, 854.59 examples/s]49259 examples [00:59, 794.97 examples/s]49340 examples [00:59, 742.04 examples/s]49424 examples [00:59, 766.58 examples/s]49507 examples [00:59, 783.07 examples/s]49587 examples [01:00, 778.16 examples/s]49671 examples [01:00, 795.26 examples/s]49752 examples [01:00, 796.96 examples/s]49836 examples [01:00, 808.62 examples/s]49918 examples [01:00, 780.24 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5429/50000 [00:00<00:00, 54288.63 examples/s] 32%|███▏      | 16007/50000 [00:00<00:00, 63570.60 examples/s] 54%|█████▎    | 26804/50000 [00:00<00:00, 72515.83 examples/s] 74%|███████▍  | 37059/50000 [00:00<00:00, 79500.48 examples/s] 96%|█████████▌| 48038/50000 [00:00<00:00, 86670.58 examples/s]                                                               0 examples [00:00, ? examples/s]63 examples [00:00, 623.49 examples/s]145 examples [00:00, 671.01 examples/s]233 examples [00:00, 721.31 examples/s]318 examples [00:00, 753.89 examples/s]400 examples [00:00, 770.45 examples/s]468 examples [00:00, 715.87 examples/s]541 examples [00:00, 718.40 examples/s]627 examples [00:00, 755.28 examples/s]718 examples [00:00, 794.48 examples/s]802 examples [00:01, 807.58 examples/s]887 examples [00:01, 819.51 examples/s]971 examples [00:01, 822.81 examples/s]1054 examples [00:01, 824.06 examples/s]1139 examples [00:01, 830.09 examples/s]1222 examples [00:01, 805.02 examples/s]1303 examples [00:01, 740.10 examples/s]1385 examples [00:01, 761.08 examples/s]1469 examples [00:01, 781.65 examples/s]1548 examples [00:01, 759.87 examples/s]1625 examples [00:02, 751.78 examples/s]1709 examples [00:02, 775.34 examples/s]1789 examples [00:02, 781.62 examples/s]1870 examples [00:02, 789.40 examples/s]1950 examples [00:02, 779.23 examples/s]2032 examples [00:02, 789.46 examples/s]2112 examples [00:02, 769.68 examples/s]2195 examples [00:02, 786.06 examples/s]2277 examples [00:02, 795.87 examples/s]2357 examples [00:03, 781.19 examples/s]2439 examples [00:03, 790.27 examples/s]2519 examples [00:03, 788.46 examples/s]2598 examples [00:03, 781.93 examples/s]2681 examples [00:03, 794.05 examples/s]2766 examples [00:03, 808.72 examples/s]2848 examples [00:03, 801.76 examples/s]2932 examples [00:03, 810.48 examples/s]3020 examples [00:03, 830.11 examples/s]3104 examples [00:03, 822.59 examples/s]3190 examples [00:04, 832.15 examples/s]3275 examples [00:04, 834.93 examples/s]3359 examples [00:04, 833.81 examples/s]3443 examples [00:04, 834.99 examples/s]3528 examples [00:04, 837.32 examples/s]3612 examples [00:04, 828.45 examples/s]3695 examples [00:04, 777.73 examples/s]3779 examples [00:04, 794.50 examples/s]3863 examples [00:04, 805.73 examples/s]3944 examples [00:04, 787.50 examples/s]4024 examples [00:05, 781.02 examples/s]4104 examples [00:05, 785.78 examples/s]4183 examples [00:05, 779.79 examples/s]4266 examples [00:05, 792.81 examples/s]4357 examples [00:05, 823.65 examples/s]4440 examples [00:05, 822.82 examples/s]4523 examples [00:05, 810.01 examples/s]4607 examples [00:05, 817.66 examples/s]4690 examples [00:05, 821.31 examples/s]4775 examples [00:05, 829.65 examples/s]4859 examples [00:06, 832.16 examples/s]4943 examples [00:06, 830.54 examples/s]5027 examples [00:06, 832.52 examples/s]5111 examples [00:06, 824.92 examples/s]5198 examples [00:06, 835.76 examples/s]5282 examples [00:06, 810.13 examples/s]5366 examples [00:06, 815.24 examples/s]5448 examples [00:06, 793.74 examples/s]5528 examples [00:06, 795.06 examples/s]5611 examples [00:07, 803.89 examples/s]5692 examples [00:07, 803.90 examples/s]5773 examples [00:07, 769.84 examples/s]5856 examples [00:07, 785.73 examples/s]5942 examples [00:07, 806.39 examples/s]6030 examples [00:07, 826.54 examples/s]6114 examples [00:07, 823.31 examples/s]6197 examples [00:07, 788.40 examples/s]6284 examples [00:07, 810.28 examples/s]6369 examples [00:07, 821.14 examples/s]6454 examples [00:08, 827.13 examples/s]6537 examples [00:08, 820.75 examples/s]6620 examples [00:08, 783.58 examples/s]6699 examples [00:08, 737.05 examples/s]6782 examples [00:08, 761.98 examples/s]6868 examples [00:08, 787.24 examples/s]6948 examples [00:08, 779.97 examples/s]7029 examples [00:08, 787.12 examples/s]7110 examples [00:08, 792.36 examples/s]7190 examples [00:08, 785.41 examples/s]7274 examples [00:09, 800.19 examples/s]7355 examples [00:09, 790.09 examples/s]7435 examples [00:09, 790.11 examples/s]7515 examples [00:09, 761.18 examples/s]7600 examples [00:09, 784.06 examples/s]7688 examples [00:09, 808.38 examples/s]7775 examples [00:09, 824.92 examples/s]7858 examples [00:09, 794.22 examples/s]7943 examples [00:09, 807.98 examples/s]8028 examples [00:10, 818.31 examples/s]8117 examples [00:10, 836.87 examples/s]8202 examples [00:10, 828.75 examples/s]8292 examples [00:10, 847.83 examples/s]8378 examples [00:10, 819.94 examples/s]8467 examples [00:10, 836.97 examples/s]8554 examples [00:10, 845.44 examples/s]8639 examples [00:10, 783.69 examples/s]8722 examples [00:10, 795.57 examples/s]8804 examples [00:10, 800.91 examples/s]8885 examples [00:11, 799.54 examples/s]8966 examples [00:11, 796.06 examples/s]9046 examples [00:11, 790.81 examples/s]9126 examples [00:11, 790.26 examples/s]9212 examples [00:11, 809.79 examples/s]9297 examples [00:11, 820.02 examples/s]9380 examples [00:11, 809.57 examples/s]9468 examples [00:11, 827.28 examples/s]9557 examples [00:11, 842.80 examples/s]9642 examples [00:12, 841.49 examples/s]9727 examples [00:12, 832.09 examples/s]9811 examples [00:12, 830.58 examples/s]9895 examples [00:12, 823.77 examples/s]9978 examples [00:12, 796.91 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteAI4FUC/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteAI4FUC/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe2f5b260d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe2f5b260d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe2f5b260d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe3415057b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe280b745c0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe2f5b260d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe2f5b260d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe2f5b260d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe2f3dbf2b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe2f3dbf048>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.47 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.47 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.47 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.86 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.86 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.19 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.19 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.03 file/s]2020-08-19 18:09:37.171007: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-19 18:09:37.175322: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-19 18:09:37.175569: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b5f19b63a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-19 18:09:37.175589: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:01, 160272.68it/s] 73%|███████▎  | 7249920/9912422 [00:00<00:11, 228743.75it/s]9920512it [00:00, 44406592.39it/s]                           
0it [00:00, ?it/s]32768it [00:00, 532300.61it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 8559541.86it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 55594.32it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

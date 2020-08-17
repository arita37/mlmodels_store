
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fc22ea60620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fc22ea60620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fc299712400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fc299712400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fc2b892eea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fc2b892eea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc246a4d0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc246a4d0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc246a4d0d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▊      | 3817472/9912422 [00:00<00:00, 38141655.38it/s]9920512it [00:00, 34289149.72it/s]                             
0it [00:00, ?it/s]32768it [00:00, 571374.33it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 470682.42it/s]1654784it [00:00, 11640668.57it/s]                         
0it [00:00, ?it/s]8192it [00:00, 127452.77it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc244aad4e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc244bf6a20>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fc246a45d08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fc246a45d08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fc246a45d08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:07,  2.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:07,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:07,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:06,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:06,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:05,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:05,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:46,  3.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:46,  3.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:46,  3.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:45,  3.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:45,  3.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:45,  3.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:44,  3.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:44,  3.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:44,  3.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:31,  4.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:31,  4.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:31,  4.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:31,  4.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:30,  4.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:30,  4.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:30,  4.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:30,  4.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:21,  6.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:21,  6.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:21,  6.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:21,  6.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:21,  6.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:20,  6.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:20,  6.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:20,  6.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:20,  6.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:14,  8.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:14,  8.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:14,  8.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:14,  8.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:14,  8.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  8.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:14,  8.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:14,  8.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:14,  8.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:10, 12.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:10, 12.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:10, 12.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:10, 12.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 12.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:09, 12.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:09, 12.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:09, 12.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:09, 12.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 16.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 16.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 16.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 16.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 16.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 16.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 16.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 16.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 16.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 21.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 21.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 21.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 21.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 21.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 21.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 21.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 21.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 21.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 26.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 26.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 26.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 26.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 26.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 26.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 26.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 26.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 26.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 33.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 33.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 33.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 33.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 33.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 33.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 33.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 33.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 33.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 39.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 39.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 39.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 39.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 39.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 39.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 39.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 39.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 39.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 45.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 50.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 50.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 55.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 55.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 55.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 55.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 55.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 55.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 55.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 55.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 55.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 59.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 59.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 59.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 59.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 59.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 59.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 59.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 59.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 59.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 62.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 62.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 62.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 62.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 62.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 62.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 62.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 65.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 65.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 65.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 65.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 65.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 66.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 67.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.33s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 67.18 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.33s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.34s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.36 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.34s/ url]
0 examples [00:00, ? examples/s]2020-08-17 12:07:20.129568: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-17 12:07:20.146962: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-17 12:07:20.147270: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562992267b10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-17 12:07:20.147339: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.85 examples/s]95 examples [00:00,  5.49 examples/s]189 examples [00:00,  7.83 examples/s]278 examples [00:00, 11.14 examples/s]375 examples [00:00, 15.84 examples/s]471 examples [00:00, 22.47 examples/s]565 examples [00:00, 31.77 examples/s]660 examples [00:00, 44.74 examples/s]755 examples [00:01, 62.65 examples/s]846 examples [00:01, 86.93 examples/s]939 examples [00:01, 119.37 examples/s]1030 examples [00:01, 160.65 examples/s]1126 examples [00:01, 214.13 examples/s]1219 examples [00:01, 278.24 examples/s]1322 examples [00:01, 356.00 examples/s]1417 examples [00:01, 436.98 examples/s]1516 examples [00:01, 524.27 examples/s]1612 examples [00:01, 603.86 examples/s]1707 examples [00:02, 670.67 examples/s]1801 examples [00:02, 723.14 examples/s]1893 examples [00:02, 753.26 examples/s]1984 examples [00:02, 793.11 examples/s]2079 examples [00:02, 832.26 examples/s]2171 examples [00:02, 833.96 examples/s]2270 examples [00:02, 873.95 examples/s]2366 examples [00:02, 897.06 examples/s]2462 examples [00:02, 914.37 examples/s]2558 examples [00:03, 927.05 examples/s]2653 examples [00:03, 916.83 examples/s]2746 examples [00:03, 916.39 examples/s]2840 examples [00:03, 918.76 examples/s]2933 examples [00:03, 914.72 examples/s]3029 examples [00:03, 924.94 examples/s]3122 examples [00:03, 925.72 examples/s]3215 examples [00:03, 918.81 examples/s]3313 examples [00:03, 935.25 examples/s]3413 examples [00:03, 951.85 examples/s]3513 examples [00:04, 965.35 examples/s]3611 examples [00:04, 964.80 examples/s]3709 examples [00:04, 967.87 examples/s]3806 examples [00:04, 965.58 examples/s]3903 examples [00:04, 963.85 examples/s]4008 examples [00:04, 986.55 examples/s]4107 examples [00:04, 971.76 examples/s]4205 examples [00:04, 956.59 examples/s]4303 examples [00:04, 962.96 examples/s]4400 examples [00:04, 961.94 examples/s]4497 examples [00:05, 950.24 examples/s]4593 examples [00:05, 931.22 examples/s]4687 examples [00:05, 923.00 examples/s]4780 examples [00:05, 904.26 examples/s]4877 examples [00:05, 921.16 examples/s]4976 examples [00:05, 939.60 examples/s]5079 examples [00:05, 964.76 examples/s]5176 examples [00:05, 954.87 examples/s]5274 examples [00:05, 958.64 examples/s]5371 examples [00:05, 952.78 examples/s]5467 examples [00:06, 940.77 examples/s]5562 examples [00:06, 932.01 examples/s]5660 examples [00:06, 942.92 examples/s]5758 examples [00:06, 951.85 examples/s]5854 examples [00:06, 943.24 examples/s]5949 examples [00:06, 922.99 examples/s]6042 examples [00:06, 914.34 examples/s]6134 examples [00:06, 913.98 examples/s]6228 examples [00:06, 918.81 examples/s]6321 examples [00:07, 920.70 examples/s]6414 examples [00:07, 911.94 examples/s]6506 examples [00:07, 910.57 examples/s]6598 examples [00:07, 908.28 examples/s]6692 examples [00:07, 916.92 examples/s]6789 examples [00:07, 929.75 examples/s]6883 examples [00:07, 905.83 examples/s]6974 examples [00:07, 892.04 examples/s]7064 examples [00:07, 892.15 examples/s]7154 examples [00:07, 891.37 examples/s]7244 examples [00:08, 891.08 examples/s]7336 examples [00:08, 897.14 examples/s]7426 examples [00:08, 892.26 examples/s]7516 examples [00:08, 880.28 examples/s]7605 examples [00:08, 877.33 examples/s]7695 examples [00:08, 882.53 examples/s]7784 examples [00:08, 825.97 examples/s]7878 examples [00:08, 854.39 examples/s]7967 examples [00:08, 862.72 examples/s]8054 examples [00:08, 839.70 examples/s]8150 examples [00:09, 871.59 examples/s]8241 examples [00:09, 881.83 examples/s]8330 examples [00:09, 882.66 examples/s]8424 examples [00:09, 897.23 examples/s]8517 examples [00:09, 904.25 examples/s]8608 examples [00:09, 894.20 examples/s]8698 examples [00:09, 892.04 examples/s]8789 examples [00:09, 895.72 examples/s]8880 examples [00:09, 897.94 examples/s]8970 examples [00:10, 888.23 examples/s]9059 examples [00:10, 884.44 examples/s]9151 examples [00:10, 892.48 examples/s]9242 examples [00:10, 894.75 examples/s]9337 examples [00:10, 910.63 examples/s]9429 examples [00:10, 909.85 examples/s]9521 examples [00:10, 903.96 examples/s]9618 examples [00:10, 920.24 examples/s]9712 examples [00:10, 923.46 examples/s]9805 examples [00:10, 919.13 examples/s]9898 examples [00:11, 920.85 examples/s]9991 examples [00:11, 916.05 examples/s]10083 examples [00:11, 837.05 examples/s]10177 examples [00:11, 864.31 examples/s]10273 examples [00:11, 889.15 examples/s]10367 examples [00:11, 902.27 examples/s]10458 examples [00:11, 892.42 examples/s]10556 examples [00:11, 916.47 examples/s]10649 examples [00:11, 911.34 examples/s]10742 examples [00:11, 913.35 examples/s]10834 examples [00:12, 912.50 examples/s]10927 examples [00:12, 916.72 examples/s]11022 examples [00:12, 926.00 examples/s]11115 examples [00:12, 921.72 examples/s]11208 examples [00:12, 921.71 examples/s]11304 examples [00:12, 930.81 examples/s]11398 examples [00:12, 926.76 examples/s]11491 examples [00:12, 917.83 examples/s]11583 examples [00:12, 913.63 examples/s]11677 examples [00:12, 920.05 examples/s]11770 examples [00:13, 921.46 examples/s]11863 examples [00:13, 916.22 examples/s]11955 examples [00:13, 914.22 examples/s]12048 examples [00:13, 918.70 examples/s]12141 examples [00:13, 921.30 examples/s]12234 examples [00:13, 913.01 examples/s]12326 examples [00:13, 895.62 examples/s]12424 examples [00:13, 917.70 examples/s]12518 examples [00:13, 922.41 examples/s]12611 examples [00:14, 894.31 examples/s]12702 examples [00:14, 898.24 examples/s]12793 examples [00:14, 896.96 examples/s]12886 examples [00:14, 906.06 examples/s]12979 examples [00:14, 910.58 examples/s]13074 examples [00:14, 921.46 examples/s]13168 examples [00:14, 924.59 examples/s]13261 examples [00:14, 921.75 examples/s]13354 examples [00:14, 920.37 examples/s]13447 examples [00:14, 915.35 examples/s]13541 examples [00:15, 918.79 examples/s]13637 examples [00:15, 927.91 examples/s]13730 examples [00:15, 924.02 examples/s]13823 examples [00:15, 918.65 examples/s]13917 examples [00:15, 924.90 examples/s]14012 examples [00:15, 931.35 examples/s]14106 examples [00:15, 926.95 examples/s]14199 examples [00:15, 919.63 examples/s]14293 examples [00:15, 922.61 examples/s]14386 examples [00:15, 914.02 examples/s]14478 examples [00:16, 884.59 examples/s]14567 examples [00:16, 881.61 examples/s]14656 examples [00:16, 880.43 examples/s]14748 examples [00:16, 889.83 examples/s]14838 examples [00:16, 887.47 examples/s]14931 examples [00:16, 898.23 examples/s]15025 examples [00:16, 908.30 examples/s]15116 examples [00:16, 907.74 examples/s]15210 examples [00:16, 914.28 examples/s]15304 examples [00:16, 919.43 examples/s]15397 examples [00:17, 921.82 examples/s]15491 examples [00:17, 926.70 examples/s]15585 examples [00:17, 928.21 examples/s]15680 examples [00:17, 932.17 examples/s]15775 examples [00:17, 937.28 examples/s]15873 examples [00:17, 949.32 examples/s]15968 examples [00:17, 933.04 examples/s]16063 examples [00:17, 935.83 examples/s]16157 examples [00:17, 929.45 examples/s]16251 examples [00:17, 931.45 examples/s]16345 examples [00:18, 931.85 examples/s]16441 examples [00:18, 938.64 examples/s]16535 examples [00:18, 935.32 examples/s]16630 examples [00:18, 937.39 examples/s]16726 examples [00:18, 941.70 examples/s]16822 examples [00:18, 946.43 examples/s]16917 examples [00:18, 947.29 examples/s]17012 examples [00:18, 946.53 examples/s]17107 examples [00:18, 946.40 examples/s]17202 examples [00:18, 946.24 examples/s]17297 examples [00:19, 945.55 examples/s]17392 examples [00:19, 931.57 examples/s]17486 examples [00:19, 923.40 examples/s]17579 examples [00:19, 923.96 examples/s]17673 examples [00:19, 925.67 examples/s]17773 examples [00:19, 944.24 examples/s]17868 examples [00:19, 942.80 examples/s]17963 examples [00:19, 939.82 examples/s]18058 examples [00:19, 849.21 examples/s]18148 examples [00:20, 861.03 examples/s]18239 examples [00:20, 872.67 examples/s]18334 examples [00:20, 894.07 examples/s]18425 examples [00:20, 891.72 examples/s]18515 examples [00:20, 881.80 examples/s]18611 examples [00:20, 903.02 examples/s]18709 examples [00:20, 922.71 examples/s]18811 examples [00:20, 947.78 examples/s]18907 examples [00:20, 936.72 examples/s]19006 examples [00:20, 950.90 examples/s]19102 examples [00:21, 936.29 examples/s]19201 examples [00:21, 950.68 examples/s]19301 examples [00:21, 963.79 examples/s]19398 examples [00:21, 947.39 examples/s]19496 examples [00:21, 956.33 examples/s]19596 examples [00:21, 968.79 examples/s]19694 examples [00:21, 957.26 examples/s]19790 examples [00:21, 904.67 examples/s]19887 examples [00:21, 922.88 examples/s]19987 examples [00:21, 943.70 examples/s]20082 examples [00:22, 895.87 examples/s]20180 examples [00:22, 919.49 examples/s]20277 examples [00:22, 932.59 examples/s]20371 examples [00:22, 925.53 examples/s]20466 examples [00:22, 932.44 examples/s]20560 examples [00:22, 925.47 examples/s]20655 examples [00:22, 931.78 examples/s]20749 examples [00:22, 872.08 examples/s]20838 examples [00:22, 870.61 examples/s]20929 examples [00:23, 880.99 examples/s]21018 examples [00:23, 881.72 examples/s]21107 examples [00:23, 882.38 examples/s]21201 examples [00:23, 897.96 examples/s]21292 examples [00:23, 899.97 examples/s]21386 examples [00:23, 910.24 examples/s]21481 examples [00:23, 919.86 examples/s]21578 examples [00:23, 934.06 examples/s]21674 examples [00:23, 941.33 examples/s]21769 examples [00:23, 937.91 examples/s]21864 examples [00:24, 940.87 examples/s]21960 examples [00:24, 943.65 examples/s]22055 examples [00:24, 942.83 examples/s]22150 examples [00:24, 933.89 examples/s]22244 examples [00:24, 923.91 examples/s]22337 examples [00:24, 916.88 examples/s]22434 examples [00:24, 929.88 examples/s]22534 examples [00:24, 947.95 examples/s]22637 examples [00:24, 971.07 examples/s]22735 examples [00:24, 956.23 examples/s]22831 examples [00:25, 941.78 examples/s]22926 examples [00:25, 936.58 examples/s]23023 examples [00:25, 944.03 examples/s]23119 examples [00:25, 948.14 examples/s]23214 examples [00:25, 948.39 examples/s]23321 examples [00:25, 979.67 examples/s]23424 examples [00:25, 991.39 examples/s]23524 examples [00:25, 988.36 examples/s]23626 examples [00:25, 994.89 examples/s]23726 examples [00:25, 981.09 examples/s]23825 examples [00:26, 968.76 examples/s]23923 examples [00:26, 955.61 examples/s]24020 examples [00:26, 959.00 examples/s]24119 examples [00:26, 965.82 examples/s]24216 examples [00:26, 959.95 examples/s]24318 examples [00:26, 976.90 examples/s]24416 examples [00:26, 976.05 examples/s]24514 examples [00:26, 965.19 examples/s]24611 examples [00:26, 959.24 examples/s]24707 examples [00:27, 943.53 examples/s]24802 examples [00:27, 940.19 examples/s]24897 examples [00:27, 939.23 examples/s]24993 examples [00:27, 944.21 examples/s]25088 examples [00:27, 936.28 examples/s]25182 examples [00:27, 926.32 examples/s]25279 examples [00:27, 936.58 examples/s]25373 examples [00:27, 921.86 examples/s]25466 examples [00:27, 920.96 examples/s]25563 examples [00:27, 934.83 examples/s]25658 examples [00:28, 938.61 examples/s]25756 examples [00:28, 948.68 examples/s]25853 examples [00:28, 953.09 examples/s]25950 examples [00:28, 956.75 examples/s]26047 examples [00:28, 958.92 examples/s]26143 examples [00:28, 957.29 examples/s]26239 examples [00:28, 946.25 examples/s]26334 examples [00:28, 941.47 examples/s]26429 examples [00:28, 941.32 examples/s]26524 examples [00:28, 942.68 examples/s]26619 examples [00:29, 904.62 examples/s]26713 examples [00:29, 913.98 examples/s]26806 examples [00:29, 917.15 examples/s]26898 examples [00:29, 897.96 examples/s]26992 examples [00:29, 907.78 examples/s]27083 examples [00:29, 900.27 examples/s]27174 examples [00:29, 899.06 examples/s]27271 examples [00:29, 917.87 examples/s]27367 examples [00:29, 928.14 examples/s]27462 examples [00:29, 933.33 examples/s]27556 examples [00:30, 920.73 examples/s]27652 examples [00:30, 931.51 examples/s]27746 examples [00:30, 930.23 examples/s]27840 examples [00:30, 921.82 examples/s]27934 examples [00:30, 926.58 examples/s]28027 examples [00:30, 927.42 examples/s]28120 examples [00:30, 923.29 examples/s]28214 examples [00:30, 925.42 examples/s]28307 examples [00:30, 920.67 examples/s]28400 examples [00:30, 920.49 examples/s]28493 examples [00:31, 916.17 examples/s]28585 examples [00:31, 915.32 examples/s]28677 examples [00:31, 905.03 examples/s]28768 examples [00:31, 902.71 examples/s]28864 examples [00:31, 919.12 examples/s]28957 examples [00:31, 914.46 examples/s]29049 examples [00:31, 895.01 examples/s]29139 examples [00:31, 887.89 examples/s]29230 examples [00:31, 891.41 examples/s]29327 examples [00:32, 913.10 examples/s]29420 examples [00:32, 916.32 examples/s]29515 examples [00:32, 925.44 examples/s]29608 examples [00:32, 910.52 examples/s]29700 examples [00:32, 910.99 examples/s]29792 examples [00:32, 899.91 examples/s]29884 examples [00:32, 904.45 examples/s]29976 examples [00:32, 908.19 examples/s]30067 examples [00:32, 861.41 examples/s]30161 examples [00:32, 883.56 examples/s]30256 examples [00:33, 901.51 examples/s]30350 examples [00:33, 910.86 examples/s]30444 examples [00:33, 918.12 examples/s]30538 examples [00:33, 924.12 examples/s]30631 examples [00:33, 919.54 examples/s]30725 examples [00:33, 923.14 examples/s]30819 examples [00:33, 925.33 examples/s]30914 examples [00:33, 931.20 examples/s]31008 examples [00:33, 917.06 examples/s]31103 examples [00:33, 926.11 examples/s]31197 examples [00:34, 927.55 examples/s]31290 examples [00:34, 920.48 examples/s]31385 examples [00:34, 927.15 examples/s]31483 examples [00:34, 940.08 examples/s]31578 examples [00:34, 938.45 examples/s]31676 examples [00:34, 950.13 examples/s]31772 examples [00:34, 931.64 examples/s]31866 examples [00:34, 926.99 examples/s]31963 examples [00:34, 937.41 examples/s]32057 examples [00:34, 937.20 examples/s]32154 examples [00:35, 944.66 examples/s]32249 examples [00:35, 934.77 examples/s]32343 examples [00:35, 932.21 examples/s]32437 examples [00:35, 924.97 examples/s]32531 examples [00:35, 927.69 examples/s]32627 examples [00:35, 934.55 examples/s]32721 examples [00:35, 883.50 examples/s]32816 examples [00:35, 900.81 examples/s]32916 examples [00:35, 926.15 examples/s]33014 examples [00:36, 940.27 examples/s]33109 examples [00:36, 939.61 examples/s]33206 examples [00:36, 947.42 examples/s]33307 examples [00:36, 962.83 examples/s]33405 examples [00:36, 966.96 examples/s]33504 examples [00:36, 971.82 examples/s]33602 examples [00:36, 970.49 examples/s]33700 examples [00:36, 960.93 examples/s]33797 examples [00:36, 943.27 examples/s]33898 examples [00:36, 961.14 examples/s]33995 examples [00:37, 961.00 examples/s]34092 examples [00:37, 936.91 examples/s]34186 examples [00:37, 921.35 examples/s]34279 examples [00:37, 922.00 examples/s]34374 examples [00:37, 927.44 examples/s]34471 examples [00:37, 937.34 examples/s]34566 examples [00:37, 939.31 examples/s]34663 examples [00:37, 946.89 examples/s]34759 examples [00:37, 949.21 examples/s]34856 examples [00:37, 953.66 examples/s]34952 examples [00:38, 945.40 examples/s]35047 examples [00:38, 930.99 examples/s]35142 examples [00:38, 934.86 examples/s]35236 examples [00:38, 930.73 examples/s]35330 examples [00:38, 923.76 examples/s]35425 examples [00:38, 931.46 examples/s]35521 examples [00:38, 937.22 examples/s]35615 examples [00:38, 925.85 examples/s]35708 examples [00:38, 917.87 examples/s]35804 examples [00:38, 927.57 examples/s]35901 examples [00:39, 937.86 examples/s]35995 examples [00:39, 933.33 examples/s]36093 examples [00:39, 944.93 examples/s]36188 examples [00:39, 935.09 examples/s]36282 examples [00:39, 929.71 examples/s]36379 examples [00:39, 940.54 examples/s]36474 examples [00:39, 938.44 examples/s]36568 examples [00:39, 933.54 examples/s]36662 examples [00:39, 897.70 examples/s]36753 examples [00:40, 894.81 examples/s]36845 examples [00:40, 902.00 examples/s]36937 examples [00:40, 905.47 examples/s]37030 examples [00:40, 911.64 examples/s]37122 examples [00:40, 903.70 examples/s]37213 examples [00:40, 904.32 examples/s]37304 examples [00:40, 903.96 examples/s]37395 examples [00:40, 889.59 examples/s]37485 examples [00:40, 888.79 examples/s]37574 examples [00:40, 861.67 examples/s]37665 examples [00:41, 872.97 examples/s]37760 examples [00:41, 894.18 examples/s]37850 examples [00:41, 893.69 examples/s]37945 examples [00:41, 909.12 examples/s]38038 examples [00:41, 913.39 examples/s]38132 examples [00:41, 920.35 examples/s]38225 examples [00:41, 911.89 examples/s]38317 examples [00:41, 910.13 examples/s]38411 examples [00:41, 917.37 examples/s]38510 examples [00:41, 936.68 examples/s]38604 examples [00:42, 903.35 examples/s]38695 examples [00:42, 886.08 examples/s]38786 examples [00:42, 890.05 examples/s]38879 examples [00:42, 899.53 examples/s]38973 examples [00:42, 910.72 examples/s]39074 examples [00:42, 936.51 examples/s]39171 examples [00:42, 944.48 examples/s]39268 examples [00:42, 948.86 examples/s]39368 examples [00:42, 961.25 examples/s]39467 examples [00:42, 967.91 examples/s]39564 examples [00:43, 966.77 examples/s]39661 examples [00:43, 958.27 examples/s]39757 examples [00:43, 946.64 examples/s]39852 examples [00:43, 945.96 examples/s]39947 examples [00:43, 941.96 examples/s]40042 examples [00:43, 882.86 examples/s]40132 examples [00:43, 865.75 examples/s]40220 examples [00:43, 862.79 examples/s]40315 examples [00:43, 884.63 examples/s]40408 examples [00:44, 896.13 examples/s]40503 examples [00:44, 911.03 examples/s]40595 examples [00:44, 899.14 examples/s]40686 examples [00:44, 895.46 examples/s]40778 examples [00:44, 900.39 examples/s]40874 examples [00:44, 916.94 examples/s]40970 examples [00:44, 927.03 examples/s]41065 examples [00:44, 932.64 examples/s]41159 examples [00:44, 910.03 examples/s]41255 examples [00:44, 923.23 examples/s]41348 examples [00:45, 909.40 examples/s]41444 examples [00:45, 922.08 examples/s]41539 examples [00:45, 929.31 examples/s]41633 examples [00:45, 925.01 examples/s]41726 examples [00:45, 921.54 examples/s]41819 examples [00:45, 922.89 examples/s]41915 examples [00:45, 931.52 examples/s]42009 examples [00:45, 927.63 examples/s]42102 examples [00:45, 927.17 examples/s]42196 examples [00:45, 928.85 examples/s]42291 examples [00:46, 933.87 examples/s]42386 examples [00:46, 935.26 examples/s]42483 examples [00:46, 943.64 examples/s]42578 examples [00:46, 928.27 examples/s]42672 examples [00:46, 931.64 examples/s]42772 examples [00:46, 948.85 examples/s]42869 examples [00:46, 952.67 examples/s]42965 examples [00:46, 942.71 examples/s]43060 examples [00:46, 939.61 examples/s]43155 examples [00:46, 913.55 examples/s]43248 examples [00:47, 917.93 examples/s]43344 examples [00:47, 928.28 examples/s]43442 examples [00:47, 941.58 examples/s]43537 examples [00:47, 943.60 examples/s]43632 examples [00:47, 937.61 examples/s]43726 examples [00:47, 928.22 examples/s]43819 examples [00:47, 926.50 examples/s]43918 examples [00:47, 943.54 examples/s]44013 examples [00:47, 936.03 examples/s]44107 examples [00:47, 925.58 examples/s]44202 examples [00:48, 932.63 examples/s]44296 examples [00:48, 931.94 examples/s]44390 examples [00:48, 916.63 examples/s]44485 examples [00:48, 924.77 examples/s]44578 examples [00:48, 922.52 examples/s]44671 examples [00:48, 913.32 examples/s]44765 examples [00:48, 920.74 examples/s]44858 examples [00:48, 919.59 examples/s]44953 examples [00:48, 926.22 examples/s]45049 examples [00:49, 933.86 examples/s]45147 examples [00:49, 944.26 examples/s]45242 examples [00:49, 911.87 examples/s]45335 examples [00:49, 916.60 examples/s]45429 examples [00:49, 921.83 examples/s]45524 examples [00:49, 927.45 examples/s]45617 examples [00:49, 909.27 examples/s]45709 examples [00:49, 900.04 examples/s]45800 examples [00:49, 881.14 examples/s]45899 examples [00:49, 909.14 examples/s]45991 examples [00:50, 907.38 examples/s]46085 examples [00:50, 914.28 examples/s]46180 examples [00:50, 922.91 examples/s]46273 examples [00:50, 914.74 examples/s]46365 examples [00:50, 908.97 examples/s]46459 examples [00:50, 916.77 examples/s]46556 examples [00:50, 930.36 examples/s]46651 examples [00:50, 935.28 examples/s]46745 examples [00:50, 929.41 examples/s]46841 examples [00:50, 935.93 examples/s]46935 examples [00:51, 935.12 examples/s]47029 examples [00:51, 911.84 examples/s]47130 examples [00:51, 938.51 examples/s]47225 examples [00:51, 921.89 examples/s]47319 examples [00:51, 926.41 examples/s]47418 examples [00:51, 942.84 examples/s]47513 examples [00:51, 944.04 examples/s]47608 examples [00:51, 932.10 examples/s]47702 examples [00:51, 930.58 examples/s]47797 examples [00:51, 933.98 examples/s]47893 examples [00:52, 939.17 examples/s]47989 examples [00:52, 943.98 examples/s]48084 examples [00:52, 936.18 examples/s]48178 examples [00:52, 928.46 examples/s]48273 examples [00:52, 934.40 examples/s]48367 examples [00:52, 934.65 examples/s]48461 examples [00:52, 932.99 examples/s]48555 examples [00:52, 916.06 examples/s]48647 examples [00:52, 910.05 examples/s]48744 examples [00:53, 924.77 examples/s]48838 examples [00:53, 926.85 examples/s]48933 examples [00:53, 933.63 examples/s]49028 examples [00:53, 936.07 examples/s]49125 examples [00:53, 944.85 examples/s]49220 examples [00:53, 939.09 examples/s]49316 examples [00:53, 943.08 examples/s]49414 examples [00:53, 951.11 examples/s]49510 examples [00:53, 939.88 examples/s]49605 examples [00:53, 922.47 examples/s]49698 examples [00:54, 902.30 examples/s]49789 examples [00:54, 904.31 examples/s]49883 examples [00:54, 913.11 examples/s]49978 examples [00:54, 921.81 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7053/50000 [00:00<00:00, 70524.69 examples/s] 37%|███▋      | 18461/50000 [00:00<00:00, 79640.55 examples/s] 62%|██████▏   | 30955/50000 [00:00<00:00, 89358.86 examples/s] 87%|████████▋ | 43577/50000 [00:00<00:00, 97937.89 examples/s]                                                               0 examples [00:00, ? examples/s]73 examples [00:00, 723.51 examples/s]167 examples [00:00, 776.86 examples/s]261 examples [00:00, 817.63 examples/s]355 examples [00:00, 849.93 examples/s]447 examples [00:00, 868.92 examples/s]538 examples [00:00, 879.37 examples/s]636 examples [00:00, 907.00 examples/s]732 examples [00:00, 919.92 examples/s]824 examples [00:00, 919.31 examples/s]923 examples [00:01, 938.28 examples/s]1018 examples [00:01, 941.06 examples/s]1118 examples [00:01, 956.12 examples/s]1214 examples [00:01, 950.20 examples/s]1312 examples [00:01, 957.03 examples/s]1408 examples [00:01, 954.18 examples/s]1504 examples [00:01, 945.35 examples/s]1599 examples [00:01, 912.94 examples/s]1692 examples [00:01, 916.18 examples/s]1784 examples [00:01, 914.36 examples/s]1878 examples [00:02, 921.26 examples/s]1972 examples [00:02, 925.26 examples/s]2066 examples [00:02, 928.98 examples/s]2159 examples [00:02, 925.93 examples/s]2255 examples [00:02, 934.15 examples/s]2349 examples [00:02, 929.15 examples/s]2442 examples [00:02, 916.29 examples/s]2538 examples [00:02, 926.52 examples/s]2631 examples [00:02, 913.20 examples/s]2725 examples [00:02, 917.83 examples/s]2819 examples [00:03, 922.81 examples/s]2912 examples [00:03, 891.66 examples/s]3009 examples [00:03, 911.22 examples/s]3101 examples [00:03, 906.39 examples/s]3195 examples [00:03, 914.68 examples/s]3288 examples [00:03, 918.54 examples/s]3382 examples [00:03, 924.06 examples/s]3477 examples [00:03, 931.15 examples/s]3573 examples [00:03, 938.06 examples/s]3670 examples [00:03, 945.94 examples/s]3765 examples [00:04, 941.88 examples/s]3860 examples [00:04, 935.77 examples/s]3958 examples [00:04, 946.57 examples/s]4053 examples [00:04, 937.63 examples/s]4151 examples [00:04, 947.72 examples/s]4246 examples [00:04, 945.06 examples/s]4341 examples [00:04, 937.30 examples/s]4437 examples [00:04, 941.30 examples/s]4533 examples [00:04, 944.83 examples/s]4630 examples [00:04, 951.88 examples/s]4731 examples [00:05, 967.20 examples/s]4828 examples [00:05, 957.30 examples/s]4926 examples [00:05, 962.55 examples/s]5023 examples [00:05, 964.15 examples/s]5121 examples [00:05, 967.34 examples/s]5219 examples [00:05, 968.34 examples/s]5316 examples [00:05, 947.93 examples/s]5411 examples [00:05, 945.90 examples/s]5511 examples [00:05, 960.60 examples/s]5609 examples [00:05, 963.71 examples/s]5706 examples [00:06, 962.87 examples/s]5803 examples [00:06, 944.61 examples/s]5899 examples [00:06, 947.18 examples/s]5994 examples [00:06, 929.67 examples/s]6088 examples [00:06, 906.65 examples/s]6182 examples [00:06, 915.19 examples/s]6279 examples [00:06, 930.29 examples/s]6375 examples [00:06, 938.05 examples/s]6469 examples [00:06, 919.91 examples/s]6563 examples [00:07, 925.07 examples/s]6658 examples [00:07, 930.07 examples/s]6752 examples [00:07, 924.99 examples/s]6845 examples [00:07, 911.73 examples/s]6937 examples [00:07, 895.19 examples/s]7027 examples [00:07, 878.89 examples/s]7116 examples [00:07, 880.07 examples/s]7208 examples [00:07, 890.64 examples/s]7302 examples [00:07, 903.06 examples/s]7395 examples [00:07, 908.59 examples/s]7487 examples [00:08, 911.43 examples/s]7580 examples [00:08, 915.92 examples/s]7672 examples [00:08, 916.79 examples/s]7766 examples [00:08, 920.98 examples/s]7859 examples [00:08, 917.04 examples/s]7951 examples [00:08, 911.11 examples/s]8047 examples [00:08, 923.89 examples/s]8140 examples [00:08, 903.70 examples/s]8235 examples [00:08, 916.73 examples/s]8334 examples [00:08, 936.86 examples/s]8429 examples [00:09, 940.06 examples/s]8524 examples [00:09, 935.58 examples/s]8618 examples [00:09, 933.22 examples/s]8715 examples [00:09, 943.77 examples/s]8812 examples [00:09, 950.43 examples/s]8908 examples [00:09, 949.08 examples/s]9004 examples [00:09, 950.74 examples/s]9100 examples [00:09, 943.79 examples/s]9196 examples [00:09, 948.03 examples/s]9292 examples [00:09, 948.78 examples/s]9389 examples [00:10, 953.54 examples/s]9485 examples [00:10, 949.73 examples/s]9580 examples [00:10, 949.35 examples/s]9675 examples [00:10, 934.02 examples/s]9774 examples [00:10, 948.86 examples/s]9869 examples [00:10, 947.12 examples/s]9964 examples [00:10, 912.22 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLBYQJK/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLBYQJK/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc246a4d0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc246a4d0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc246a4d0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc1d0d13b00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc29242c7b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc246a4d0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc246a4d0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc246a4d0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc244bf62e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc244bf6080>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.09 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 21.94 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.55 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.55 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.90 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.90 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.27 file/s]2020-08-17 12:08:33.484648: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-17 12:08:33.488635: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-17 12:08:33.488839: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a3e5a1b710 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-17 12:08:33.488855: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 457213.39it/s] 72%|███████▏  | 7094272/9912422 [00:00<00:04, 651350.11it/s]9920512it [00:00, 43212794.63it/s]                           
0it [00:00, ?it/s]32768it [00:00, 538581.33it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 146070.55it/s]1654784it [00:00, 10218554.69it/s]                         
0it [00:00, ?it/s]8192it [00:00, 177039.96it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

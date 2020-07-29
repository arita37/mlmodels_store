
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f385401fb70> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f385401fb70> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f38bec8f510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f38bec8f510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f38ddebcea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f38ddebcea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f386c037a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f386c037a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f386c037a60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3702784/9912422 [00:00<00:00, 36913727.29it/s]9920512it [00:00, 35708615.85it/s]                             
0it [00:00, ?it/s]32768it [00:00, 599976.22it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 486597.99it/s]1654784it [00:00, 12104702.85it/s]                         
0it [00:00, ?it/s]8192it [00:00, 256106.34it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f386bfc24a8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3853ee9320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f386c0376a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f386c0376a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f386c0376a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:29,  1.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:29,  1.81 MiB/s][A

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

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:00,  2.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:00,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:00,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:00,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:59,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:59,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:58,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:58,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:41,  3.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:41,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:41,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:40,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:40,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:40,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:39,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:39,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:28,  4.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:28,  4.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:28,  4.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:27,  4.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:27,  4.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:27,  4.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:27,  4.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:27,  4.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:19,  6.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:19,  6.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:19,  6.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:19,  6.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:18,  6.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:18,  6.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:18,  6.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:18,  6.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:13,  9.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:13,  9.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:13,  9.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:13,  9.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:13,  9.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:13,  9.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:12,  9.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:12,  9.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:09, 12.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:09, 12.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:09, 12.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:09, 12.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:09, 12.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:09, 12.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:09, 12.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 12.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 16.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 16.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 16.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 16.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 16.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 16.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 16.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 16.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 21.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 26.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 26.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 26.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 26.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 26.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 26.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 26.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 26.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 32.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 32.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 32.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 32.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 32.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 32.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 32.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 32.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 37.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 37.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 37.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 37.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 37.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 37.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 37.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 37.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 43.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 43.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 43.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 43.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 43.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 43.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 43.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 43.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 48.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 48.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 48.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 48.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 48.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 48.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 48.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 48.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 52.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 52.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 52.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 52.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 52.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 52.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 52.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 52.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 56.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 56.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 56.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 56.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 56.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 56.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 56.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 56.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 56.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 60.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 60.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 60.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 60.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 60.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 60.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 60.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 60.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 62.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 63.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 63.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 62.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 62.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 62.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 62.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 62.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 62.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 62.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 62.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 63.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 63.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 63.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 63.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 63.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 63.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 63.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 63.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 65.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 65.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 65.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 65.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 65.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 65.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 65.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 65.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 65.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.02s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 65.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 65.01 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.46s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 65.01 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.46s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.46s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 29.65 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.47s/ url]
0 examples [00:00, ? examples/s]2020-07-29 06:09:54.256192: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-29 06:09:54.269607: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-29 06:09:54.269871: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560f20ac3bc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-29 06:09:54.269892: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
18 examples [00:00, 178.87 examples/s]103 examples [00:00, 234.33 examples/s]187 examples [00:00, 298.72 examples/s]272 examples [00:00, 370.85 examples/s]360 examples [00:00, 448.08 examples/s]449 examples [00:00, 525.79 examples/s]535 examples [00:00, 593.93 examples/s]625 examples [00:00, 661.32 examples/s]713 examples [00:00, 713.52 examples/s]804 examples [00:01, 760.84 examples/s]896 examples [00:01, 800.44 examples/s]984 examples [00:01, 821.98 examples/s]1074 examples [00:01, 841.95 examples/s]1162 examples [00:01, 844.05 examples/s]1249 examples [00:01, 838.57 examples/s]1335 examples [00:01, 838.85 examples/s]1421 examples [00:01, 834.32 examples/s]1507 examples [00:01, 839.78 examples/s]1593 examples [00:01, 844.14 examples/s]1679 examples [00:02, 846.39 examples/s]1764 examples [00:02, 846.62 examples/s]1849 examples [00:02, 840.01 examples/s]1934 examples [00:02, 840.15 examples/s]2019 examples [00:02, 838.91 examples/s]2103 examples [00:02, 830.82 examples/s]2187 examples [00:02, 830.95 examples/s]2271 examples [00:02, 829.64 examples/s]2362 examples [00:02, 850.55 examples/s]2452 examples [00:02, 863.05 examples/s]2541 examples [00:03, 869.68 examples/s]2629 examples [00:03, 860.39 examples/s]2717 examples [00:03, 863.81 examples/s]2804 examples [00:03, 850.38 examples/s]2890 examples [00:03, 849.72 examples/s]2976 examples [00:03, 830.62 examples/s]3060 examples [00:03, 825.63 examples/s]3145 examples [00:03, 830.98 examples/s]3232 examples [00:03, 840.94 examples/s]3319 examples [00:03, 848.33 examples/s]3408 examples [00:04, 857.55 examples/s]3498 examples [00:04, 866.66 examples/s]3585 examples [00:04, 852.30 examples/s]3671 examples [00:04, 847.97 examples/s]3757 examples [00:04, 850.27 examples/s]3843 examples [00:04, 846.22 examples/s]3928 examples [00:04, 839.09 examples/s]4012 examples [00:04, 837.51 examples/s]4096 examples [00:04, 833.25 examples/s]4180 examples [00:04, 833.45 examples/s]4265 examples [00:05, 838.25 examples/s]4349 examples [00:05, 835.36 examples/s]4433 examples [00:05, 811.98 examples/s]4515 examples [00:05, 796.39 examples/s]4605 examples [00:05, 824.39 examples/s]4694 examples [00:05, 842.84 examples/s]4780 examples [00:05, 844.89 examples/s]4865 examples [00:05, 814.34 examples/s]4950 examples [00:05, 823.78 examples/s]5035 examples [00:06, 829.68 examples/s]5122 examples [00:06, 841.22 examples/s]5209 examples [00:06, 848.92 examples/s]5300 examples [00:06, 862.76 examples/s]5392 examples [00:06, 877.20 examples/s]5485 examples [00:06, 891.98 examples/s]5579 examples [00:06, 904.04 examples/s]5670 examples [00:06, 901.26 examples/s]5763 examples [00:06, 907.72 examples/s]5854 examples [00:06, 907.52 examples/s]5945 examples [00:07, 906.61 examples/s]6039 examples [00:07, 915.93 examples/s]6131 examples [00:07, 911.25 examples/s]6225 examples [00:07, 919.56 examples/s]6318 examples [00:07, 916.68 examples/s]6412 examples [00:07, 923.22 examples/s]6506 examples [00:07, 925.52 examples/s]6599 examples [00:07, 921.86 examples/s]6692 examples [00:07, 897.88 examples/s]6787 examples [00:07, 911.58 examples/s]6879 examples [00:08, 905.67 examples/s]6970 examples [00:08, 903.06 examples/s]7061 examples [00:08, 881.07 examples/s]7150 examples [00:08, 833.47 examples/s]7235 examples [00:08, 838.33 examples/s]7320 examples [00:08, 808.11 examples/s]7406 examples [00:08, 820.76 examples/s]7493 examples [00:08, 834.32 examples/s]7577 examples [00:08, 832.11 examples/s]7661 examples [00:09, 792.66 examples/s]7741 examples [00:09, 759.44 examples/s]7818 examples [00:09, 696.37 examples/s]7890 examples [00:09, 698.31 examples/s]7975 examples [00:09, 737.61 examples/s]8059 examples [00:09, 764.38 examples/s]8144 examples [00:09, 786.99 examples/s]8224 examples [00:09, 760.04 examples/s]8301 examples [00:09, 759.90 examples/s]8382 examples [00:09, 773.81 examples/s]8465 examples [00:10, 787.53 examples/s]8554 examples [00:10, 815.60 examples/s]8640 examples [00:10, 828.20 examples/s]8724 examples [00:10, 824.17 examples/s]8807 examples [00:10, 817.27 examples/s]8894 examples [00:10, 831.26 examples/s]8978 examples [00:10, 826.86 examples/s]9063 examples [00:10, 832.36 examples/s]9147 examples [00:10, 824.29 examples/s]9230 examples [00:11, 824.76 examples/s]9313 examples [00:11, 825.17 examples/s]9396 examples [00:11, 822.98 examples/s]9479 examples [00:11, 815.70 examples/s]9564 examples [00:11, 823.69 examples/s]9648 examples [00:11, 828.30 examples/s]9734 examples [00:11, 837.27 examples/s]9819 examples [00:11, 837.30 examples/s]9903 examples [00:11, 836.05 examples/s]9987 examples [00:11, 836.54 examples/s]10071 examples [00:12, 787.47 examples/s]10155 examples [00:12, 800.85 examples/s]10245 examples [00:12, 825.84 examples/s]10329 examples [00:12, 827.51 examples/s]10413 examples [00:12, 825.10 examples/s]10498 examples [00:12, 832.34 examples/s]10585 examples [00:12, 841.76 examples/s]10670 examples [00:12, 803.14 examples/s]10753 examples [00:12, 810.52 examples/s]10837 examples [00:12, 818.36 examples/s]10920 examples [00:13, 817.41 examples/s]11005 examples [00:13, 825.52 examples/s]11088 examples [00:13, 823.21 examples/s]11172 examples [00:13, 827.91 examples/s]11255 examples [00:13, 818.71 examples/s]11343 examples [00:13, 835.28 examples/s]11428 examples [00:13, 837.10 examples/s]11512 examples [00:13, 833.29 examples/s]11598 examples [00:13, 841.09 examples/s]11683 examples [00:13, 835.28 examples/s]11768 examples [00:14, 838.35 examples/s]11853 examples [00:14, 840.50 examples/s]11938 examples [00:14, 839.56 examples/s]12023 examples [00:14, 842.50 examples/s]12108 examples [00:14, 835.35 examples/s]12192 examples [00:14, 821.55 examples/s]12275 examples [00:14, 817.22 examples/s]12361 examples [00:14, 827.18 examples/s]12448 examples [00:14, 837.18 examples/s]12532 examples [00:14, 825.14 examples/s]12619 examples [00:15, 836.18 examples/s]12707 examples [00:15, 847.01 examples/s]12795 examples [00:15, 854.84 examples/s]12881 examples [00:15, 830.39 examples/s]12965 examples [00:15, 805.00 examples/s]13053 examples [00:15, 825.85 examples/s]13137 examples [00:15, 827.25 examples/s]13222 examples [00:15, 831.69 examples/s]13306 examples [00:15, 830.95 examples/s]13390 examples [00:16, 827.71 examples/s]13473 examples [00:16, 808.45 examples/s]13555 examples [00:16, 811.73 examples/s]13640 examples [00:16, 822.28 examples/s]13725 examples [00:16, 828.28 examples/s]13808 examples [00:16, 826.06 examples/s]13891 examples [00:16, 818.59 examples/s]13976 examples [00:16, 826.35 examples/s]14059 examples [00:16, 819.74 examples/s]14142 examples [00:16, 818.85 examples/s]14229 examples [00:17, 830.62 examples/s]14314 examples [00:17, 836.16 examples/s]14398 examples [00:17, 834.81 examples/s]14485 examples [00:17, 844.16 examples/s]14571 examples [00:17, 845.90 examples/s]14656 examples [00:17, 828.12 examples/s]14739 examples [00:17, 823.35 examples/s]14822 examples [00:17, 824.58 examples/s]14905 examples [00:17, 810.92 examples/s]14988 examples [00:17, 812.25 examples/s]15073 examples [00:18, 822.53 examples/s]15158 examples [00:18, 827.67 examples/s]15242 examples [00:18, 828.73 examples/s]15330 examples [00:18, 841.57 examples/s]15415 examples [00:18, 841.29 examples/s]15500 examples [00:18, 842.98 examples/s]15587 examples [00:18, 850.04 examples/s]15675 examples [00:18, 858.08 examples/s]15761 examples [00:18, 858.08 examples/s]15854 examples [00:18, 876.27 examples/s]15945 examples [00:19, 883.52 examples/s]16037 examples [00:19, 891.75 examples/s]16127 examples [00:19, 860.98 examples/s]16214 examples [00:19, 858.55 examples/s]16301 examples [00:19, 844.18 examples/s]16390 examples [00:19, 856.42 examples/s]16478 examples [00:19, 862.77 examples/s]16570 examples [00:19, 876.73 examples/s]16663 examples [00:19, 889.92 examples/s]16756 examples [00:19, 900.54 examples/s]16850 examples [00:20, 909.28 examples/s]16942 examples [00:20, 902.79 examples/s]17033 examples [00:20, 902.19 examples/s]17124 examples [00:20, 889.45 examples/s]17214 examples [00:20, 876.87 examples/s]17302 examples [00:20, 862.03 examples/s]17390 examples [00:20, 866.38 examples/s]17477 examples [00:20, 826.64 examples/s]17568 examples [00:20, 849.47 examples/s]17657 examples [00:21, 860.46 examples/s]17744 examples [00:21, 861.87 examples/s]17831 examples [00:21, 856.17 examples/s]17918 examples [00:21, 858.76 examples/s]18006 examples [00:21, 863.08 examples/s]18093 examples [00:21, 864.55 examples/s]18180 examples [00:21, 864.20 examples/s]18267 examples [00:21, 863.43 examples/s]18356 examples [00:21, 869.49 examples/s]18443 examples [00:21, 831.61 examples/s]18529 examples [00:22, 839.28 examples/s]18614 examples [00:22, 838.46 examples/s]18700 examples [00:22, 844.65 examples/s]18785 examples [00:22, 765.86 examples/s]18864 examples [00:22, 765.96 examples/s]18942 examples [00:22, 750.32 examples/s]19022 examples [00:22, 758.60 examples/s]19105 examples [00:22, 778.36 examples/s]19184 examples [00:22, 773.26 examples/s]19267 examples [00:23, 787.75 examples/s]19352 examples [00:23, 803.20 examples/s]19435 examples [00:23, 809.19 examples/s]19519 examples [00:23, 816.21 examples/s]19601 examples [00:23, 809.99 examples/s]19685 examples [00:23, 818.35 examples/s]19767 examples [00:23, 812.79 examples/s]19850 examples [00:23, 815.64 examples/s]19932 examples [00:23, 811.23 examples/s]20014 examples [00:23, 757.94 examples/s]20100 examples [00:24, 785.76 examples/s]20183 examples [00:24, 797.41 examples/s]20264 examples [00:24, 797.73 examples/s]20351 examples [00:24, 815.33 examples/s]20438 examples [00:24, 828.69 examples/s]20523 examples [00:24, 834.69 examples/s]20608 examples [00:24, 836.76 examples/s]20696 examples [00:24, 847.14 examples/s]20781 examples [00:24, 842.27 examples/s]20868 examples [00:24, 847.61 examples/s]20955 examples [00:25, 853.82 examples/s]21041 examples [00:25, 846.64 examples/s]21128 examples [00:25, 852.48 examples/s]21214 examples [00:25, 839.50 examples/s]21302 examples [00:25, 849.97 examples/s]21388 examples [00:25, 843.30 examples/s]21473 examples [00:25, 837.54 examples/s]21557 examples [00:25, 828.04 examples/s]21644 examples [00:25, 837.81 examples/s]21735 examples [00:25, 855.50 examples/s]21821 examples [00:26, 852.47 examples/s]21907 examples [00:26, 847.32 examples/s]22000 examples [00:26, 869.63 examples/s]22089 examples [00:26, 873.45 examples/s]22179 examples [00:26, 877.46 examples/s]22268 examples [00:26, 878.69 examples/s]22359 examples [00:26, 886.11 examples/s]22448 examples [00:26, 882.02 examples/s]22537 examples [00:26, 851.06 examples/s]22623 examples [00:27, 836.16 examples/s]22708 examples [00:27, 837.75 examples/s]22793 examples [00:27, 838.40 examples/s]22877 examples [00:27, 838.47 examples/s]22961 examples [00:27, 838.59 examples/s]23046 examples [00:27, 841.33 examples/s]23131 examples [00:27, 836.68 examples/s]23216 examples [00:27, 840.30 examples/s]23301 examples [00:27, 835.65 examples/s]23385 examples [00:27, 833.29 examples/s]23471 examples [00:28, 840.15 examples/s]23557 examples [00:28, 844.70 examples/s]23642 examples [00:28, 844.67 examples/s]23727 examples [00:28, 831.59 examples/s]23811 examples [00:28, 829.56 examples/s]23898 examples [00:28, 839.49 examples/s]23985 examples [00:28, 847.16 examples/s]24073 examples [00:28, 854.33 examples/s]24161 examples [00:28, 860.02 examples/s]24252 examples [00:28, 872.82 examples/s]24345 examples [00:29, 888.33 examples/s]24439 examples [00:29, 901.36 examples/s]24530 examples [00:29, 890.53 examples/s]24620 examples [00:29, 874.97 examples/s]24708 examples [00:29, 873.43 examples/s]24796 examples [00:29, 868.23 examples/s]24885 examples [00:29, 872.92 examples/s]24973 examples [00:29, 870.56 examples/s]25061 examples [00:29, 861.55 examples/s]25148 examples [00:29, 834.57 examples/s]25232 examples [00:30, 817.42 examples/s]25318 examples [00:30, 827.62 examples/s]25405 examples [00:30, 838.41 examples/s]25490 examples [00:30, 807.10 examples/s]25579 examples [00:30, 828.83 examples/s]25663 examples [00:30, 816.15 examples/s]25745 examples [00:30, 807.99 examples/s]25827 examples [00:30, 802.34 examples/s]25908 examples [00:30, 773.15 examples/s]25992 examples [00:31, 791.97 examples/s]26075 examples [00:31, 802.29 examples/s]26160 examples [00:31, 815.11 examples/s]26246 examples [00:31, 827.26 examples/s]26332 examples [00:31, 835.55 examples/s]26416 examples [00:31, 836.16 examples/s]26503 examples [00:31, 845.10 examples/s]26592 examples [00:31, 856.74 examples/s]26683 examples [00:31, 870.36 examples/s]26775 examples [00:31, 881.94 examples/s]26864 examples [00:32, 875.31 examples/s]26952 examples [00:32, 875.41 examples/s]27040 examples [00:32, 871.65 examples/s]27128 examples [00:32, 855.74 examples/s]27215 examples [00:32, 857.31 examples/s]27303 examples [00:32, 862.45 examples/s]27394 examples [00:32, 874.50 examples/s]27482 examples [00:32, 870.16 examples/s]27573 examples [00:32, 880.77 examples/s]27662 examples [00:32, 868.50 examples/s]27749 examples [00:33, 861.37 examples/s]27836 examples [00:33, 861.69 examples/s]27923 examples [00:33, 853.33 examples/s]28009 examples [00:33, 850.40 examples/s]28096 examples [00:33, 854.66 examples/s]28182 examples [00:33, 841.15 examples/s]28267 examples [00:33, 836.90 examples/s]28354 examples [00:33, 845.52 examples/s]28443 examples [00:33, 856.37 examples/s]28533 examples [00:33, 866.45 examples/s]28620 examples [00:34, 855.96 examples/s]28706 examples [00:34, 854.29 examples/s]28792 examples [00:34, 851.58 examples/s]28878 examples [00:34, 849.34 examples/s]28964 examples [00:34, 852.03 examples/s]29050 examples [00:34, 846.06 examples/s]29135 examples [00:34, 844.78 examples/s]29220 examples [00:34, 834.72 examples/s]29308 examples [00:34, 845.65 examples/s]29393 examples [00:35, 845.17 examples/s]29478 examples [00:35, 844.63 examples/s]29563 examples [00:35, 837.11 examples/s]29647 examples [00:35, 833.94 examples/s]29731 examples [00:35, 835.45 examples/s]29815 examples [00:35, 676.34 examples/s]29888 examples [00:35, 662.61 examples/s]29974 examples [00:35, 711.50 examples/s]30049 examples [00:35, 679.98 examples/s]30133 examples [00:36, 720.48 examples/s]30219 examples [00:36, 757.20 examples/s]30303 examples [00:36, 778.21 examples/s]30383 examples [00:36, 778.80 examples/s]30467 examples [00:36, 794.92 examples/s]30550 examples [00:36, 804.57 examples/s]30632 examples [00:36, 795.39 examples/s]30716 examples [00:36, 807.17 examples/s]30801 examples [00:36, 819.10 examples/s]30889 examples [00:36, 834.32 examples/s]30975 examples [00:37, 838.87 examples/s]31061 examples [00:37, 842.86 examples/s]31147 examples [00:37, 844.61 examples/s]31232 examples [00:37, 840.22 examples/s]31317 examples [00:37, 831.27 examples/s]31407 examples [00:37, 850.02 examples/s]31493 examples [00:37, 821.48 examples/s]31578 examples [00:37, 829.31 examples/s]31662 examples [00:37, 803.07 examples/s]31749 examples [00:37, 820.40 examples/s]31832 examples [00:38, 821.27 examples/s]31918 examples [00:38, 829.86 examples/s]32002 examples [00:38, 824.14 examples/s]32085 examples [00:38, 822.62 examples/s]32168 examples [00:38, 817.34 examples/s]32252 examples [00:38, 822.77 examples/s]32335 examples [00:38, 821.56 examples/s]32419 examples [00:38, 826.82 examples/s]32502 examples [00:38, 822.65 examples/s]32585 examples [00:38, 815.71 examples/s]32673 examples [00:39, 832.27 examples/s]32763 examples [00:39, 849.38 examples/s]32855 examples [00:39, 868.70 examples/s]32944 examples [00:39, 873.30 examples/s]33035 examples [00:39, 883.65 examples/s]33126 examples [00:39, 888.42 examples/s]33215 examples [00:39, 887.04 examples/s]33305 examples [00:39, 887.87 examples/s]33394 examples [00:39, 872.68 examples/s]33482 examples [00:40, 856.89 examples/s]33568 examples [00:40, 834.46 examples/s]33654 examples [00:40, 841.76 examples/s]33742 examples [00:40, 851.86 examples/s]33828 examples [00:40, 829.80 examples/s]33912 examples [00:40, 816.73 examples/s]33996 examples [00:40, 822.51 examples/s]34079 examples [00:40, 818.56 examples/s]34161 examples [00:40, 817.15 examples/s]34246 examples [00:40, 825.00 examples/s]34332 examples [00:41, 835.01 examples/s]34418 examples [00:41, 839.60 examples/s]34504 examples [00:41, 843.06 examples/s]34589 examples [00:41, 834.43 examples/s]34679 examples [00:41, 852.99 examples/s]34770 examples [00:41, 868.26 examples/s]34860 examples [00:41, 877.22 examples/s]34951 examples [00:41, 886.02 examples/s]35040 examples [00:41, 881.34 examples/s]35131 examples [00:41, 888.36 examples/s]35220 examples [00:42, 873.38 examples/s]35308 examples [00:42, 861.73 examples/s]35395 examples [00:42, 845.10 examples/s]35480 examples [00:42, 841.24 examples/s]35565 examples [00:42, 841.10 examples/s]35650 examples [00:42, 839.01 examples/s]35734 examples [00:42, 832.24 examples/s]35818 examples [00:42, 830.50 examples/s]35902 examples [00:42, 823.64 examples/s]35990 examples [00:42, 838.30 examples/s]36074 examples [00:43, 832.37 examples/s]36165 examples [00:43, 853.49 examples/s]36251 examples [00:43, 848.72 examples/s]36337 examples [00:43, 845.44 examples/s]36422 examples [00:43, 844.77 examples/s]36507 examples [00:43, 845.52 examples/s]36594 examples [00:43, 851.61 examples/s]36680 examples [00:43, 850.88 examples/s]36767 examples [00:43, 856.25 examples/s]36853 examples [00:43, 852.57 examples/s]36940 examples [00:44, 855.51 examples/s]37027 examples [00:44, 858.51 examples/s]37113 examples [00:44, 858.62 examples/s]37199 examples [00:44, 856.61 examples/s]37285 examples [00:44, 857.36 examples/s]37371 examples [00:44, 855.90 examples/s]37460 examples [00:44, 864.28 examples/s]37549 examples [00:44, 870.39 examples/s]37637 examples [00:44, 872.54 examples/s]37725 examples [00:45, 863.21 examples/s]37812 examples [00:45, 846.37 examples/s]37897 examples [00:45, 843.75 examples/s]37985 examples [00:45, 853.86 examples/s]38071 examples [00:45, 851.55 examples/s]38159 examples [00:45, 859.22 examples/s]38247 examples [00:45, 863.95 examples/s]38338 examples [00:45, 875.38 examples/s]38429 examples [00:45, 883.89 examples/s]38518 examples [00:45, 875.88 examples/s]38606 examples [00:46, 872.09 examples/s]38695 examples [00:46, 874.65 examples/s]38784 examples [00:46, 878.84 examples/s]38872 examples [00:46, 878.56 examples/s]38960 examples [00:46, 877.37 examples/s]39048 examples [00:46, 860.84 examples/s]39135 examples [00:46, 845.98 examples/s]39221 examples [00:46, 849.56 examples/s]39307 examples [00:46, 831.00 examples/s]39393 examples [00:46, 836.07 examples/s]39477 examples [00:47, 836.30 examples/s]39562 examples [00:47, 838.51 examples/s]39646 examples [00:47, 826.81 examples/s]39729 examples [00:47, 826.78 examples/s]39812 examples [00:47, 822.21 examples/s]39895 examples [00:47, 823.96 examples/s]39979 examples [00:47, 826.89 examples/s]40062 examples [00:47, 764.64 examples/s]40152 examples [00:47, 799.26 examples/s]40235 examples [00:47, 807.44 examples/s]40322 examples [00:48, 823.77 examples/s]40409 examples [00:48, 834.63 examples/s]40493 examples [00:48, 830.90 examples/s]40580 examples [00:48, 840.50 examples/s]40665 examples [00:48, 841.25 examples/s]40751 examples [00:48, 844.11 examples/s]40836 examples [00:48, 832.81 examples/s]40920 examples [00:48, 826.95 examples/s]41007 examples [00:48, 837.22 examples/s]41091 examples [00:49, 828.30 examples/s]41180 examples [00:49, 844.62 examples/s]41268 examples [00:49, 852.36 examples/s]41354 examples [00:49, 853.85 examples/s]41440 examples [00:49, 836.08 examples/s]41524 examples [00:49, 821.00 examples/s]41608 examples [00:49, 824.11 examples/s]41695 examples [00:49, 834.69 examples/s]41780 examples [00:49, 838.18 examples/s]41866 examples [00:49, 842.12 examples/s]41954 examples [00:50, 850.30 examples/s]42040 examples [00:50, 844.21 examples/s]42126 examples [00:50, 846.93 examples/s]42211 examples [00:50, 838.63 examples/s]42296 examples [00:50, 840.15 examples/s]42381 examples [00:50, 832.14 examples/s]42465 examples [00:50, 822.80 examples/s]42548 examples [00:50, 818.51 examples/s]42630 examples [00:50, 806.48 examples/s]42714 examples [00:50, 815.78 examples/s]42796 examples [00:51, 783.70 examples/s]42882 examples [00:51, 804.89 examples/s]42973 examples [00:51, 831.93 examples/s]43060 examples [00:51, 842.88 examples/s]43153 examples [00:51, 867.25 examples/s]43241 examples [00:51, 847.69 examples/s]43327 examples [00:51, 841.82 examples/s]43414 examples [00:51, 848.20 examples/s]43500 examples [00:51, 843.17 examples/s]43586 examples [00:51, 845.49 examples/s]43671 examples [00:52, 833.71 examples/s]43757 examples [00:52, 840.30 examples/s]43845 examples [00:52, 850.07 examples/s]43936 examples [00:52, 864.98 examples/s]44026 examples [00:52, 874.20 examples/s]44115 examples [00:52, 877.57 examples/s]44204 examples [00:52, 879.77 examples/s]44295 examples [00:52, 887.74 examples/s]44387 examples [00:52, 894.62 examples/s]44481 examples [00:52, 906.38 examples/s]44572 examples [00:53, 898.04 examples/s]44662 examples [00:53, 897.56 examples/s]44753 examples [00:53, 900.78 examples/s]44844 examples [00:53, 876.99 examples/s]44932 examples [00:53, 867.89 examples/s]45019 examples [00:53, 854.96 examples/s]45105 examples [00:53, 837.82 examples/s]45189 examples [00:53, 836.10 examples/s]45273 examples [00:53, 827.15 examples/s]45356 examples [00:54, 823.20 examples/s]45440 examples [00:54, 825.75 examples/s]45523 examples [00:54, 816.54 examples/s]45609 examples [00:54, 828.49 examples/s]45695 examples [00:54, 837.19 examples/s]45782 examples [00:54, 846.74 examples/s]45870 examples [00:54, 855.75 examples/s]45961 examples [00:54, 870.73 examples/s]46051 examples [00:54, 878.91 examples/s]46141 examples [00:54, 884.29 examples/s]46233 examples [00:55, 892.34 examples/s]46323 examples [00:55, 888.30 examples/s]46415 examples [00:55, 894.57 examples/s]46505 examples [00:55, 864.33 examples/s]46593 examples [00:55, 868.77 examples/s]46683 examples [00:55, 877.44 examples/s]46772 examples [00:55, 879.74 examples/s]46861 examples [00:55, 879.73 examples/s]46950 examples [00:55, 871.97 examples/s]47038 examples [00:55, 836.19 examples/s]47122 examples [00:56, 815.06 examples/s]47212 examples [00:56, 836.92 examples/s]47300 examples [00:56, 848.13 examples/s]47388 examples [00:56, 856.98 examples/s]47474 examples [00:56, 855.17 examples/s]47560 examples [00:56, 851.31 examples/s]47646 examples [00:56, 844.45 examples/s]47731 examples [00:56, 842.46 examples/s]47816 examples [00:56, 843.45 examples/s]47901 examples [00:57, 840.66 examples/s]47986 examples [00:57, 822.16 examples/s]48069 examples [00:57, 823.61 examples/s]48152 examples [00:57, 814.92 examples/s]48234 examples [00:57, 765.03 examples/s]48313 examples [00:57, 770.99 examples/s]48391 examples [00:57, 772.29 examples/s]48469 examples [00:57, 746.78 examples/s]48550 examples [00:57, 763.40 examples/s]48636 examples [00:57, 788.51 examples/s]48716 examples [00:58, 787.88 examples/s]48796 examples [00:58, 776.99 examples/s]48875 examples [00:58, 778.17 examples/s]48955 examples [00:58, 783.06 examples/s]49036 examples [00:58, 790.91 examples/s]49121 examples [00:58, 806.15 examples/s]49202 examples [00:58, 806.63 examples/s]49288 examples [00:58, 821.42 examples/s]49376 examples [00:58, 833.75 examples/s]49465 examples [00:58, 848.54 examples/s]49551 examples [00:59, 843.47 examples/s]49636 examples [00:59, 837.21 examples/s]49720 examples [00:59, 835.23 examples/s]49806 examples [00:59, 841.73 examples/s]49891 examples [00:59, 843.36 examples/s]49978 examples [00:59, 849.90 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6297/50000 [00:00<00:00, 62966.61 examples/s] 33%|███▎      | 16522/50000 [00:00<00:00, 71169.20 examples/s] 54%|█████▍    | 26962/50000 [00:00<00:00, 78682.01 examples/s] 77%|███████▋  | 38636/50000 [00:00<00:00, 87211.20 examples/s] 99%|█████████▉| 49615/50000 [00:00<00:00, 92944.05 examples/s]                                                               0 examples [00:00, ? examples/s]63 examples [00:00, 625.52 examples/s]143 examples [00:00, 667.69 examples/s]228 examples [00:00, 711.81 examples/s]316 examples [00:00, 753.31 examples/s]403 examples [00:00, 784.76 examples/s]488 examples [00:00, 801.08 examples/s]575 examples [00:00, 818.29 examples/s]665 examples [00:00, 839.21 examples/s]751 examples [00:00, 843.57 examples/s]835 examples [00:01, 840.41 examples/s]918 examples [00:01, 833.02 examples/s]1001 examples [00:01, 828.75 examples/s]1084 examples [00:01, 804.94 examples/s]1174 examples [00:01, 829.69 examples/s]1266 examples [00:01, 853.52 examples/s]1357 examples [00:01, 867.71 examples/s]1444 examples [00:01, 862.50 examples/s]1531 examples [00:01, 833.31 examples/s]1615 examples [00:01, 829.21 examples/s]1702 examples [00:02, 839.08 examples/s]1791 examples [00:02, 852.33 examples/s]1877 examples [00:02, 849.48 examples/s]1965 examples [00:02, 856.29 examples/s]2057 examples [00:02, 873.14 examples/s]2145 examples [00:02, 859.46 examples/s]2232 examples [00:02, 855.89 examples/s]2318 examples [00:02, 847.56 examples/s]2403 examples [00:02, 846.76 examples/s]2488 examples [00:02, 841.41 examples/s]2576 examples [00:03, 852.29 examples/s]2662 examples [00:03, 843.22 examples/s]2753 examples [00:03, 861.76 examples/s]2843 examples [00:03, 872.09 examples/s]2931 examples [00:03, 869.29 examples/s]3020 examples [00:03, 873.68 examples/s]3108 examples [00:03, 861.07 examples/s]3195 examples [00:03, 852.12 examples/s]3283 examples [00:03, 857.85 examples/s]3377 examples [00:03, 880.44 examples/s]3466 examples [00:04, 872.90 examples/s]3554 examples [00:04, 854.56 examples/s]3641 examples [00:04, 857.65 examples/s]3727 examples [00:04, 833.70 examples/s]3812 examples [00:04, 836.19 examples/s]3897 examples [00:04, 837.38 examples/s]3981 examples [00:04, 835.08 examples/s]4065 examples [00:04, 830.96 examples/s]4150 examples [00:04, 834.96 examples/s]4234 examples [00:05, 817.46 examples/s]4316 examples [00:05, 804.94 examples/s]4397 examples [00:05, 800.26 examples/s]4485 examples [00:05, 821.84 examples/s]4575 examples [00:05, 841.48 examples/s]4663 examples [00:05, 851.78 examples/s]4753 examples [00:05, 863.36 examples/s]4840 examples [00:05, 856.56 examples/s]4926 examples [00:05, 851.07 examples/s]5012 examples [00:05, 843.37 examples/s]5097 examples [00:06, 836.91 examples/s]5181 examples [00:06, 837.38 examples/s]5265 examples [00:06, 837.77 examples/s]5349 examples [00:06, 834.67 examples/s]5437 examples [00:06, 846.48 examples/s]5524 examples [00:06, 850.92 examples/s]5610 examples [00:06, 841.97 examples/s]5695 examples [00:06, 835.00 examples/s]5785 examples [00:06, 852.76 examples/s]5871 examples [00:06, 824.08 examples/s]5955 examples [00:07, 828.15 examples/s]6039 examples [00:07, 824.72 examples/s]6126 examples [00:07, 836.37 examples/s]6210 examples [00:07, 828.55 examples/s]6295 examples [00:07, 834.22 examples/s]6384 examples [00:07, 849.11 examples/s]6473 examples [00:07, 859.28 examples/s]6560 examples [00:07, 861.28 examples/s]6647 examples [00:07, 856.66 examples/s]6733 examples [00:07, 852.42 examples/s]6819 examples [00:08, 839.15 examples/s]6904 examples [00:08, 842.13 examples/s]6991 examples [00:08, 849.18 examples/s]7076 examples [00:08, 846.93 examples/s]7161 examples [00:08, 839.49 examples/s]7246 examples [00:08, 841.37 examples/s]7332 examples [00:08, 845.97 examples/s]7417 examples [00:08, 847.13 examples/s]7503 examples [00:08, 850.00 examples/s]7589 examples [00:08, 842.54 examples/s]7674 examples [00:09, 806.41 examples/s]7761 examples [00:09, 823.83 examples/s]7850 examples [00:09, 842.45 examples/s]7938 examples [00:09, 850.38 examples/s]8024 examples [00:09, 846.88 examples/s]8109 examples [00:09, 844.45 examples/s]8195 examples [00:09, 847.73 examples/s]8282 examples [00:09, 854.13 examples/s]8369 examples [00:09, 856.48 examples/s]8455 examples [00:10, 856.05 examples/s]8541 examples [00:10, 851.08 examples/s]8627 examples [00:10, 834.57 examples/s]8711 examples [00:10, 816.24 examples/s]8793 examples [00:10, 817.19 examples/s]8880 examples [00:10, 831.19 examples/s]8964 examples [00:10, 830.42 examples/s]9051 examples [00:10, 841.23 examples/s]9140 examples [00:10, 852.61 examples/s]9227 examples [00:10, 857.03 examples/s]9313 examples [00:11, 828.33 examples/s]9400 examples [00:11, 839.58 examples/s]9485 examples [00:11, 833.79 examples/s]9576 examples [00:11, 854.63 examples/s]9664 examples [00:11, 858.69 examples/s]9756 examples [00:11, 873.98 examples/s]9846 examples [00:11, 880.46 examples/s]9935 examples [00:11, 879.52 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteCJRME7/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteCJRME7/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f386c037a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f386c037a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f386c037a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f38b79a40b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f37f22fe400>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f386c037a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f386c037a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f386c037a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3853ee95f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3853ee9860>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.22 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 24.28 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.02 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.02 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.24 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.24 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.73 file/s]2020-07-29 06:11:14.898361: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-29 06:11:14.903014: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-29 06:11:14.903198: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561a49491170 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-29 06:11:14.903216: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:00, 162836.96it/s] 40%|████      | 4014080/9912422 [00:00<00:25, 232216.27it/s] 96%|█████████▌| 9519104/9912422 [00:00<00:01, 331134.11it/s]9920512it [00:00, 31821798.82it/s]                           
0it [00:00, ?it/s]32768it [00:00, 665306.19it/s]
0it [00:00, ?it/s]  5%|▌         | 90112/1648877 [00:00<00:01, 892819.13it/s]1654784it [00:00, 12396439.22it/s]                         
0it [00:00, ?it/s]8192it [00:00, 230444.52it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

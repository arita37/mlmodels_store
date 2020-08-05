
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f35bc1ccb70> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f35bc1ccb70> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f3626e3c510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f3626e3c510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f3646069ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f3646069ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f35d41e4a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f35d41e4a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f35d41e4a60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▍      | 3432448/9912422 [00:00<00:00, 34320094.15it/s]9920512it [00:00, 36282469.57it/s]                             
0it [00:00, ?it/s]32768it [00:00, 541396.65it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 148737.98it/s]1654784it [00:00, 10957708.18it/s]                         
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 74801.92it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f35d258d6a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f35d257e940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f35d41e46a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f35d41e46a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f35d41e46a8> , (data_info, **args) 

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

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 3/162 [00:00<00:58,  2.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:58,  2.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:58,  2.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<00:44,  3.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:44,  3.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:43,  3.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:34,  4.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:34,  4.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:33,  4.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:01<00:27,  5.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:27,  5.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:27,  5.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:01<00:22,  6.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:22,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:22,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:01<00:20,  7.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:20,  7.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:19,  7.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:17,  8.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:17,  8.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:17,  8.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:01<00:15,  9.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:15,  9.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:15,  9.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:01<00:14, 10.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:14, 10.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:02<00:14, 10.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:02<00:13, 10.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:02<00:13, 10.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:02<00:13, 10.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:02<00:12, 11.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:02<00:12, 11.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:02<00:12, 11.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:02<00:11, 11.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:02<00:11, 11.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:02<00:11, 11.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:02<00:11, 11.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:02<00:11, 11.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:02<00:11, 11.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:02<00:10, 12.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:02<00:10, 12.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:02<00:10, 12.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:02<00:10, 12.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:02<00:10, 12.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:02<00:10, 12.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:03<00:09, 13.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:03<00:09, 13.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:03<00:09, 13.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:03<00:09, 13.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:03<00:09, 13.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:03<00:09, 13.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:03<00:09, 13.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:03<00:09, 13.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:03<00:09, 13.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:03<00:09, 13.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:03<00:09, 13.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:03<00:09, 13.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:03<00:09, 13.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:03<00:09, 13.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:03<00:09, 13.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:03<00:09, 13.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:03<00:09, 13.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:03<00:09, 13.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:03<00:09, 12.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:03<00:09, 12.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:04<00:08, 12.94 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:04<00:08, 13.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:04<00:08, 13.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:04<00:08, 13.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:04<00:08, 14.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:04<00:08, 14.12 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:04<00:07, 14.12 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:04<00:07, 15.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:04<00:07, 15.21 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:04<00:07, 15.21 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:04<00:06, 16.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:04<00:06, 16.00 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:04<00:06, 16.00 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:04<00:06, 16.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:04<00:06, 16.60 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:04<00:06, 16.60 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:04<00:06, 16.60 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:04<00:05, 18.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:04<00:05, 18.00 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:04<00:05, 18.00 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:04<00:05, 18.00 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:04<00:05, 19.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:04<00:05, 19.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:04<00:05, 19.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:04<00:04, 19.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:04<00:04, 21.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:04<00:04, 21.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:04<00:04, 21.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:04<00:04, 21.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:05<00:04, 23.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:05<00:04, 23.00 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:05<00:04, 23.00 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:05<00:04, 23.00 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:05<00:03, 24.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:05<00:03, 24.62 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:05<00:03, 24.62 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:05<00:03, 24.62 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:05<00:03, 24.62 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:05<00:03, 26.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:05<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:05<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:05<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:05<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:05<00:03, 27.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:05<00:03, 27.08 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:05<00:03, 27.08 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:05<00:03, 27.08 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:05<00:02, 27.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:05<00:02, 27.82 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:05<00:02, 27.82 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:05<00:02, 27.82 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:05<00:02, 28.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:05<00:02, 28.41 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:05<00:02, 28.41 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:05<00:02, 28.41 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:05<00:02, 28.41 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:05<00:02, 29.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:05<00:02, 29.21 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:05<00:02, 29.21 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:05<00:02, 29.21 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:05<00:02, 29.21 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:05<00:02, 29.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:05<00:02, 29.85 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:05<00:02, 29.85 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:05<00:02, 29.85 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:05<00:02, 29.85 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:05<00:02, 30.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:05<00:02, 30.29 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:05<00:02, 30.29 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:06<00:02, 30.29 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:06<00:02, 30.29 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:06<00:02, 30.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:06<00:02, 30.10 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:06<00:02, 30.10 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:06<00:01, 30.10 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:06<00:01, 30.10 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:06<00:01, 29.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:06<00:01, 29.32 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:06<00:01, 29.32 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:06<00:01, 29.32 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:06<00:01, 28.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:06<00:01, 28.69 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:06<00:01, 28.69 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:06<00:01, 28.69 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:06<00:01, 27.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:06<00:01, 27.95 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:06<00:01, 27.95 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:06<00:01, 27.95 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:06<00:01, 27.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:06<00:01, 27.52 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:06<00:01, 27.52 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:06<00:01, 27.52 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:06<00:01, 27.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:06<00:01, 27.42 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:06<00:01, 27.42 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:06<00:01, 27.42 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:06<00:01, 27.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:06<00:01, 27.06 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:06<00:01, 27.06 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:06<00:01, 27.06 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:06<00:01, 27.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:06<00:01, 27.82 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:06<00:01, 27.82 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:06<00:01, 27.82 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:06<00:01, 27.82 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:07<00:01, 30.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:07<00:01, 30.34 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:07<00:01, 30.34 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:07<00:01, 30.34 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:07<00:01, 30.34 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:07<00:00, 32.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:07<00:00, 32.33 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:07<00:00, 32.33 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:07<00:00, 32.33 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:07<00:00, 32.33 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:07<00:00, 34.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:07<00:00, 34.02 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:07<00:00, 34.02 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:07<00:00, 34.02 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:07<00:00, 34.02 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:07<00:00, 35.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:07<00:00, 35.27 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:07<00:00, 35.27 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:07<00:00, 35.27 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:07<00:00, 35.27 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:07<00:00, 35.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:07<00:00, 35.39 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:07<00:00, 35.39 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:07<00:00, 35.39 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:07<00:00, 35.39 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:07<00:00, 35.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:07<00:00, 35.02 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:07<00:00, 35.02 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:07<00:00, 35.02 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:07<00:00, 35.02 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:07<00:00, 33.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:07<00:00, 33.73 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:07<00:00, 33.73 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:07<00:00, 33.73 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:07<00:00, 33.73 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:07<00:00, 31.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:07<00:00, 31.08 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:07<00:00, 31.08 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:07<00:00, 31.08 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:07<00:00, 31.08 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:07<00:00, 28.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:07<00:00, 28.83 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:08<00:00, 28.83 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:08<00:00, 28.83 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:08<00:00, 27.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:08<00:00, 27.06 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00, 27.06 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:08<00:00,  8.16s/ url]Dl Completed...: 100%|██████████| 1/1 [00:08<00:00,  8.16s/ url]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00, 27.06 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:08<00:00,  8.16s/ url]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00, 27.06 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:08<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:10<00:00, 10.28s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:10<00:00,  8.16s/ url]
Dl Size...: 100%|██████████| 162/162 [00:10<00:00, 27.06 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:10<00:00, 10.28s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:10<00:00, 10.28s/ file]
Dl Size...: 100%|██████████| 162/162 [00:10<00:00, 15.76 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:10<00:00, 10.28s/ url]
0 examples [00:00, ? examples/s]2020-08-05 18:09:51.379808: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-05 18:09:51.392112: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-08-05 18:09:51.392638: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564e636250f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-05 18:09:51.392664: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
41 examples [00:00, 408.36 examples/s]152 examples [00:00, 503.50 examples/s]263 examples [00:00, 601.94 examples/s]371 examples [00:00, 693.99 examples/s]461 examples [00:00, 744.51 examples/s]567 examples [00:00, 816.18 examples/s]669 examples [00:00, 866.35 examples/s]779 examples [00:00, 924.63 examples/s]887 examples [00:00, 964.95 examples/s]997 examples [00:01, 1000.04 examples/s]1100 examples [00:01, 988.18 examples/s]1208 examples [00:01, 1011.70 examples/s]1319 examples [00:01, 1037.85 examples/s]1425 examples [00:01, 1025.36 examples/s]1530 examples [00:01, 1030.64 examples/s]1640 examples [00:01, 1048.43 examples/s]1750 examples [00:01, 1062.62 examples/s]1861 examples [00:01, 1074.06 examples/s]1973 examples [00:01, 1085.54 examples/s]2084 examples [00:02, 1090.07 examples/s]2195 examples [00:02, 1093.33 examples/s]2305 examples [00:02, 1087.00 examples/s]2416 examples [00:02, 1093.26 examples/s]2528 examples [00:02, 1099.25 examples/s]2638 examples [00:02, 1098.28 examples/s]2748 examples [00:02, 1068.40 examples/s]2856 examples [00:02, 1060.62 examples/s]2964 examples [00:02, 1064.06 examples/s]3072 examples [00:02, 1066.26 examples/s]3182 examples [00:03, 1074.80 examples/s]3293 examples [00:03, 1082.17 examples/s]3405 examples [00:03, 1092.76 examples/s]3516 examples [00:03, 1095.83 examples/s]3626 examples [00:03, 1087.60 examples/s]3736 examples [00:03, 1090.14 examples/s]3847 examples [00:03, 1093.71 examples/s]3959 examples [00:03, 1098.74 examples/s]4071 examples [00:03, 1103.85 examples/s]4184 examples [00:03, 1109.60 examples/s]4295 examples [00:04, 1089.37 examples/s]4405 examples [00:04, 1091.25 examples/s]4517 examples [00:04, 1097.08 examples/s]4628 examples [00:04, 1099.80 examples/s]4739 examples [00:04, 1099.64 examples/s]4849 examples [00:04, 1099.68 examples/s]4959 examples [00:04, 1093.42 examples/s]5070 examples [00:04, 1097.49 examples/s]5181 examples [00:04, 1099.97 examples/s]5294 examples [00:04, 1108.20 examples/s]5405 examples [00:05, 1105.77 examples/s]5516 examples [00:05, 1104.89 examples/s]5627 examples [00:05, 1072.06 examples/s]5735 examples [00:05, 1071.71 examples/s]5844 examples [00:05, 1076.62 examples/s]5956 examples [00:05, 1086.86 examples/s]6065 examples [00:05, 1078.08 examples/s]6175 examples [00:05, 1082.57 examples/s]6288 examples [00:05, 1094.02 examples/s]6398 examples [00:05, 1083.44 examples/s]6507 examples [00:06, 1071.10 examples/s]6616 examples [00:06, 1075.91 examples/s]6728 examples [00:06, 1087.42 examples/s]6840 examples [00:06, 1095.99 examples/s]6952 examples [00:06, 1101.66 examples/s]7063 examples [00:06, 1102.38 examples/s]7174 examples [00:06, 1090.88 examples/s]7284 examples [00:06, 1089.78 examples/s]7394 examples [00:06, 1092.73 examples/s]7504 examples [00:06, 1094.76 examples/s]7614 examples [00:07, 1094.51 examples/s]7724 examples [00:07, 1091.17 examples/s]7836 examples [00:07, 1097.84 examples/s]7948 examples [00:07, 1102.09 examples/s]8059 examples [00:07, 1097.58 examples/s]8169 examples [00:07, 1087.61 examples/s]8278 examples [00:07, 1069.52 examples/s]8387 examples [00:07, 1075.32 examples/s]8497 examples [00:07, 1081.80 examples/s]8606 examples [00:08, 1084.00 examples/s]8715 examples [00:08, 1085.12 examples/s]8824 examples [00:08, 1084.35 examples/s]8936 examples [00:08, 1094.17 examples/s]9047 examples [00:08, 1097.65 examples/s]9159 examples [00:08, 1101.93 examples/s]9270 examples [00:08, 1102.90 examples/s]9381 examples [00:08, 1087.75 examples/s]9490 examples [00:08, 1074.57 examples/s]9599 examples [00:08, 1079.03 examples/s]9709 examples [00:09, 1083.68 examples/s]9820 examples [00:09, 1089.79 examples/s]9930 examples [00:09, 1078.73 examples/s]10038 examples [00:09, 1023.10 examples/s]10147 examples [00:09, 1040.43 examples/s]10257 examples [00:09, 1055.40 examples/s]10366 examples [00:09, 1064.85 examples/s]10473 examples [00:09, 1064.67 examples/s]10582 examples [00:09, 1071.47 examples/s]10691 examples [00:09, 1074.10 examples/s]10800 examples [00:10, 1078.59 examples/s]10909 examples [00:10, 1080.82 examples/s]11018 examples [00:10, 1073.30 examples/s]11127 examples [00:10, 1077.52 examples/s]11237 examples [00:10, 1083.59 examples/s]11346 examples [00:10, 1081.96 examples/s]11455 examples [00:10, 1082.77 examples/s]11564 examples [00:10, 1074.82 examples/s]11673 examples [00:10, 1076.72 examples/s]11783 examples [00:10, 1080.78 examples/s]11893 examples [00:11, 1084.79 examples/s]12002 examples [00:11, 1079.94 examples/s]12111 examples [00:11, 1070.74 examples/s]12221 examples [00:11, 1076.78 examples/s]12330 examples [00:11, 1079.70 examples/s]12440 examples [00:11, 1085.50 examples/s]12549 examples [00:11, 1085.13 examples/s]12658 examples [00:11, 1083.45 examples/s]12768 examples [00:11, 1087.29 examples/s]12879 examples [00:11, 1091.48 examples/s]12990 examples [00:12, 1094.65 examples/s]13100 examples [00:12, 1092.13 examples/s]13210 examples [00:12, 1085.29 examples/s]13320 examples [00:12, 1088.22 examples/s]13430 examples [00:12, 1089.21 examples/s]13540 examples [00:12, 1089.97 examples/s]13650 examples [00:12, 1091.05 examples/s]13760 examples [00:12, 1081.24 examples/s]13869 examples [00:12, 1081.72 examples/s]13979 examples [00:12, 1085.83 examples/s]14089 examples [00:13, 1087.39 examples/s]14198 examples [00:13, 1078.08 examples/s]14306 examples [00:13, 1070.63 examples/s]14414 examples [00:13, 1064.77 examples/s]14523 examples [00:13, 1070.34 examples/s]14631 examples [00:13, 1062.50 examples/s]14741 examples [00:13, 1072.43 examples/s]14849 examples [00:13, 1072.14 examples/s]14960 examples [00:13, 1082.91 examples/s]15071 examples [00:13, 1090.12 examples/s]15181 examples [00:14, 1091.47 examples/s]15292 examples [00:14, 1095.10 examples/s]15402 examples [00:14, 1074.56 examples/s]15513 examples [00:14, 1084.28 examples/s]15624 examples [00:14, 1091.43 examples/s]15735 examples [00:14, 1095.64 examples/s]15847 examples [00:14, 1100.31 examples/s]15958 examples [00:14, 1092.16 examples/s]16068 examples [00:14, 1084.79 examples/s]16178 examples [00:15, 1089.18 examples/s]16289 examples [00:15, 1092.66 examples/s]16399 examples [00:15, 1094.78 examples/s]16509 examples [00:15, 1057.29 examples/s]16620 examples [00:15, 1071.20 examples/s]16732 examples [00:15, 1082.94 examples/s]16841 examples [00:15, 1081.67 examples/s]16951 examples [00:15, 1085.71 examples/s]17061 examples [00:15, 1089.25 examples/s]17171 examples [00:15, 1086.67 examples/s]17282 examples [00:16, 1092.06 examples/s]17394 examples [00:16, 1099.51 examples/s]17505 examples [00:16, 1101.54 examples/s]17617 examples [00:16, 1105.53 examples/s]17729 examples [00:16, 1108.17 examples/s]17840 examples [00:16, 1105.92 examples/s]17953 examples [00:16, 1112.67 examples/s]18065 examples [00:16, 1111.12 examples/s]18177 examples [00:16, 1102.67 examples/s]18288 examples [00:16, 1101.55 examples/s]18399 examples [00:17, 1103.87 examples/s]18513 examples [00:17, 1112.37 examples/s]18625 examples [00:17, 1108.93 examples/s]18736 examples [00:17, 1103.44 examples/s]18847 examples [00:17, 1102.65 examples/s]18959 examples [00:17, 1107.51 examples/s]19070 examples [00:17, 1105.04 examples/s]19181 examples [00:17, 1103.30 examples/s]19292 examples [00:17, 1083.10 examples/s]19401 examples [00:17, 1078.90 examples/s]19510 examples [00:18, 1080.55 examples/s]19619 examples [00:18, 1080.62 examples/s]19728 examples [00:18, 1080.68 examples/s]19837 examples [00:18, 1069.42 examples/s]19944 examples [00:18, 1058.28 examples/s]20050 examples [00:18, 1015.18 examples/s]20160 examples [00:18, 1037.56 examples/s]20270 examples [00:18, 1054.34 examples/s]20376 examples [00:18, 1055.72 examples/s]20483 examples [00:18, 1058.85 examples/s]20592 examples [00:19, 1067.92 examples/s]20699 examples [00:19, 1048.36 examples/s]20808 examples [00:19, 1060.18 examples/s]20915 examples [00:19, 1060.24 examples/s]21022 examples [00:19, 1060.93 examples/s]21129 examples [00:19, 1048.06 examples/s]21237 examples [00:19, 1055.65 examples/s]21347 examples [00:19, 1067.92 examples/s]21454 examples [00:19, 1063.46 examples/s]21564 examples [00:19, 1073.98 examples/s]21673 examples [00:20, 1078.55 examples/s]21781 examples [00:20, 1069.85 examples/s]21892 examples [00:20, 1079.40 examples/s]22000 examples [00:20, 1068.06 examples/s]22107 examples [00:20, 1054.53 examples/s]22216 examples [00:20, 1063.94 examples/s]22324 examples [00:20, 1066.75 examples/s]22435 examples [00:20, 1078.05 examples/s]22543 examples [00:20, 1077.40 examples/s]22651 examples [00:21, 1063.20 examples/s]22760 examples [00:21, 1070.71 examples/s]22868 examples [00:21, 1070.63 examples/s]22976 examples [00:21, 1070.76 examples/s]23086 examples [00:21, 1077.82 examples/s]23198 examples [00:21, 1087.92 examples/s]23307 examples [00:21, 1082.69 examples/s]23417 examples [00:21, 1087.56 examples/s]23527 examples [00:21, 1091.18 examples/s]23637 examples [00:21, 1087.09 examples/s]23748 examples [00:22, 1092.70 examples/s]23859 examples [00:22, 1097.17 examples/s]23970 examples [00:22, 1100.01 examples/s]24081 examples [00:22, 1099.87 examples/s]24191 examples [00:22, 1095.59 examples/s]24303 examples [00:22, 1102.66 examples/s]24415 examples [00:22, 1106.61 examples/s]24526 examples [00:22, 1103.91 examples/s]24638 examples [00:22, 1107.50 examples/s]24749 examples [00:22, 1099.92 examples/s]24860 examples [00:23, 1098.05 examples/s]24972 examples [00:23, 1103.32 examples/s]25083 examples [00:23, 1102.24 examples/s]25194 examples [00:23, 1102.14 examples/s]25305 examples [00:23, 1096.92 examples/s]25415 examples [00:23, 1078.69 examples/s]25523 examples [00:23, 1054.55 examples/s]25629 examples [00:23, 1020.81 examples/s]25734 examples [00:23, 1027.56 examples/s]25841 examples [00:23, 1038.92 examples/s]25951 examples [00:24, 1054.70 examples/s]26061 examples [00:24, 1066.38 examples/s]26170 examples [00:24, 1072.91 examples/s]26281 examples [00:24, 1082.22 examples/s]26390 examples [00:24, 1084.49 examples/s]26499 examples [00:24, 1079.50 examples/s]26609 examples [00:24, 1083.24 examples/s]26718 examples [00:24, 1069.90 examples/s]26826 examples [00:24, 1052.84 examples/s]26932 examples [00:24, 1049.64 examples/s]27038 examples [00:25, 1009.48 examples/s]27145 examples [00:25, 1025.82 examples/s]27255 examples [00:25, 1045.45 examples/s]27363 examples [00:25, 1055.01 examples/s]27469 examples [00:25, 1049.67 examples/s]27577 examples [00:25, 1057.07 examples/s]27683 examples [00:25, 1041.17 examples/s]27792 examples [00:25, 1054.32 examples/s]27903 examples [00:25, 1067.78 examples/s]28012 examples [00:26, 1073.30 examples/s]28121 examples [00:26, 1076.13 examples/s]28231 examples [00:26, 1080.66 examples/s]28340 examples [00:26, 1077.44 examples/s]28450 examples [00:26, 1081.78 examples/s]28559 examples [00:26, 1078.73 examples/s]28668 examples [00:26, 1081.19 examples/s]28777 examples [00:26, 1074.71 examples/s]28886 examples [00:26, 1078.49 examples/s]28996 examples [00:26, 1084.45 examples/s]29105 examples [00:27, 1081.88 examples/s]29216 examples [00:27, 1087.47 examples/s]29328 examples [00:27, 1094.40 examples/s]29438 examples [00:27, 1095.48 examples/s]29549 examples [00:27, 1096.98 examples/s]29659 examples [00:27, 1092.17 examples/s]29769 examples [00:27, 1093.10 examples/s]29879 examples [00:27, 1085.63 examples/s]29988 examples [00:27, 1082.74 examples/s]30097 examples [00:27, 1024.81 examples/s]30206 examples [00:28, 1041.58 examples/s]30314 examples [00:28, 1051.68 examples/s]30424 examples [00:28, 1064.40 examples/s]30533 examples [00:28, 1070.15 examples/s]30643 examples [00:28, 1077.82 examples/s]30752 examples [00:28, 1081.10 examples/s]30861 examples [00:28, 1083.20 examples/s]30970 examples [00:28, 990.65 examples/s] 31075 examples [00:28, 1006.56 examples/s]31181 examples [00:28, 1021.77 examples/s]31288 examples [00:29, 1034.12 examples/s]31396 examples [00:29, 1046.47 examples/s]31505 examples [00:29, 1058.58 examples/s]31616 examples [00:29, 1071.45 examples/s]31724 examples [00:29, 1070.86 examples/s]31832 examples [00:29, 1061.50 examples/s]31941 examples [00:29, 1069.30 examples/s]32051 examples [00:29, 1076.02 examples/s]32162 examples [00:29, 1083.29 examples/s]32273 examples [00:29, 1090.60 examples/s]32383 examples [00:30, 1085.77 examples/s]32495 examples [00:30, 1094.32 examples/s]32606 examples [00:30, 1098.58 examples/s]32716 examples [00:30, 1097.91 examples/s]32826 examples [00:30, 1089.16 examples/s]32935 examples [00:30, 1077.91 examples/s]33043 examples [00:30, 1069.84 examples/s]33152 examples [00:30, 1075.27 examples/s]33260 examples [00:30, 1075.83 examples/s]33371 examples [00:31, 1084.95 examples/s]33480 examples [00:31, 1084.27 examples/s]33590 examples [00:31, 1086.55 examples/s]33701 examples [00:31, 1091.23 examples/s]33811 examples [00:31, 1087.93 examples/s]33921 examples [00:31, 1089.83 examples/s]34030 examples [00:31, 1086.38 examples/s]34139 examples [00:31, 1063.78 examples/s]34249 examples [00:31, 1072.45 examples/s]34357 examples [00:31, 1067.40 examples/s]34465 examples [00:32, 1070.43 examples/s]34573 examples [00:32, 1068.35 examples/s]34680 examples [00:32, 1046.35 examples/s]34788 examples [00:32, 1055.73 examples/s]34901 examples [00:32, 1074.24 examples/s]35012 examples [00:32, 1082.21 examples/s]35121 examples [00:32, 1080.03 examples/s]35231 examples [00:32, 1085.56 examples/s]35340 examples [00:32, 1068.67 examples/s]35447 examples [00:32, 1066.42 examples/s]35558 examples [00:33, 1077.45 examples/s]35666 examples [00:33, 1075.62 examples/s]35774 examples [00:33, 1076.93 examples/s]35882 examples [00:33, 1074.67 examples/s]35992 examples [00:33, 1079.87 examples/s]36101 examples [00:33, 1082.15 examples/s]36210 examples [00:33, 1075.49 examples/s]36321 examples [00:33, 1083.90 examples/s]36432 examples [00:33, 1090.51 examples/s]36543 examples [00:33, 1095.64 examples/s]36653 examples [00:34, 1090.49 examples/s]36763 examples [00:34, 1075.89 examples/s]36874 examples [00:34, 1083.56 examples/s]36986 examples [00:34, 1091.61 examples/s]37096 examples [00:34, 1091.78 examples/s]37206 examples [00:34, 1085.11 examples/s]37315 examples [00:34, 1076.28 examples/s]37423 examples [00:34, 1048.93 examples/s]37529 examples [00:34, 1045.61 examples/s]37636 examples [00:34, 1052.39 examples/s]37744 examples [00:35, 1058.66 examples/s]37852 examples [00:35, 1063.22 examples/s]37962 examples [00:35, 1073.29 examples/s]38074 examples [00:35, 1085.44 examples/s]38187 examples [00:35, 1096.36 examples/s]38298 examples [00:35, 1098.45 examples/s]38408 examples [00:35, 1063.23 examples/s]38515 examples [00:35, 1061.60 examples/s]38626 examples [00:35, 1074.50 examples/s]38736 examples [00:35, 1081.44 examples/s]38849 examples [00:36, 1092.77 examples/s]38959 examples [00:36, 1089.97 examples/s]39071 examples [00:36, 1097.57 examples/s]39183 examples [00:36, 1102.36 examples/s]39294 examples [00:36, 1101.89 examples/s]39405 examples [00:36, 1100.96 examples/s]39516 examples [00:36, 1072.56 examples/s]39625 examples [00:36, 1075.70 examples/s]39735 examples [00:36, 1082.80 examples/s]39846 examples [00:37, 1088.75 examples/s]39957 examples [00:37, 1092.18 examples/s]40067 examples [00:37, 1035.84 examples/s]40179 examples [00:37, 1058.90 examples/s]40290 examples [00:37, 1073.18 examples/s]40402 examples [00:37, 1086.21 examples/s]40511 examples [00:37, 1084.94 examples/s]40620 examples [00:37, 1041.23 examples/s]40729 examples [00:37, 1054.03 examples/s]40839 examples [00:37, 1067.41 examples/s]40948 examples [00:38, 1073.09 examples/s]41056 examples [00:38, 1070.53 examples/s]41164 examples [00:38, 1071.48 examples/s]41274 examples [00:38, 1077.03 examples/s]41384 examples [00:38, 1083.38 examples/s]41495 examples [00:38, 1090.32 examples/s]41605 examples [00:38, 1076.19 examples/s]41713 examples [00:38, 1058.63 examples/s]41819 examples [00:38, 1041.32 examples/s]41926 examples [00:38, 1048.87 examples/s]42035 examples [00:39, 1060.11 examples/s]42142 examples [00:39, 1060.72 examples/s]42249 examples [00:39, 1055.25 examples/s]42355 examples [00:39, 1051.83 examples/s]42461 examples [00:39, 1028.99 examples/s]42567 examples [00:39, 1037.49 examples/s]42672 examples [00:39, 1040.30 examples/s]42777 examples [00:39, 1034.47 examples/s]42881 examples [00:39, 1030.76 examples/s]42986 examples [00:39, 1036.39 examples/s]43090 examples [00:40, 1031.40 examples/s]43194 examples [00:40, 1023.88 examples/s]43304 examples [00:40, 1043.47 examples/s]43414 examples [00:40, 1057.93 examples/s]43524 examples [00:40, 1069.70 examples/s]43635 examples [00:40, 1080.64 examples/s]43744 examples [00:40, 1081.24 examples/s]43853 examples [00:40, 1071.31 examples/s]43964 examples [00:40, 1082.51 examples/s]44073 examples [00:41, 1050.59 examples/s]44182 examples [00:41, 1059.86 examples/s]44289 examples [00:41, 1053.70 examples/s]44397 examples [00:41, 1061.13 examples/s]44508 examples [00:41, 1072.83 examples/s]44619 examples [00:41, 1082.06 examples/s]44729 examples [00:41, 1084.80 examples/s]44838 examples [00:41, 1082.51 examples/s]44947 examples [00:41, 1061.02 examples/s]45058 examples [00:41, 1073.28 examples/s]45166 examples [00:42, 1060.06 examples/s]45276 examples [00:42, 1071.35 examples/s]45384 examples [00:42, 1062.71 examples/s]45494 examples [00:42, 1072.07 examples/s]45606 examples [00:42, 1083.76 examples/s]45717 examples [00:42, 1089.52 examples/s]45829 examples [00:42, 1097.30 examples/s]45939 examples [00:42, 1080.63 examples/s]46048 examples [00:42, 1081.77 examples/s]46157 examples [00:42, 1082.24 examples/s]46266 examples [00:43, 1074.37 examples/s]46379 examples [00:43, 1088.38 examples/s]46488 examples [00:43, 1083.17 examples/s]46598 examples [00:43, 1085.77 examples/s]46708 examples [00:43, 1087.30 examples/s]46819 examples [00:43, 1091.59 examples/s]46931 examples [00:43, 1098.18 examples/s]47041 examples [00:43, 1090.70 examples/s]47152 examples [00:43, 1094.30 examples/s]47264 examples [00:43, 1099.10 examples/s]47374 examples [00:44, 1097.52 examples/s]47485 examples [00:44, 1099.42 examples/s]47595 examples [00:44, 1091.03 examples/s]47705 examples [00:44, 1058.84 examples/s]47814 examples [00:44, 1066.40 examples/s]47922 examples [00:44, 1069.74 examples/s]48033 examples [00:44, 1080.56 examples/s]48142 examples [00:44, 1080.78 examples/s]48251 examples [00:44, 1073.75 examples/s]48360 examples [00:44, 1075.71 examples/s]48471 examples [00:45, 1084.55 examples/s]48580 examples [00:45, 1076.07 examples/s]48689 examples [00:45, 1079.02 examples/s]48797 examples [00:45, 1073.64 examples/s]48908 examples [00:45, 1083.59 examples/s]49019 examples [00:45, 1089.26 examples/s]49130 examples [00:45, 1093.55 examples/s]49240 examples [00:45, 1094.45 examples/s]49350 examples [00:45, 1085.02 examples/s]49462 examples [00:45, 1093.22 examples/s]49574 examples [00:46, 1098.88 examples/s]49684 examples [00:46, 1096.91 examples/s]49796 examples [00:46, 1101.76 examples/s]49907 examples [00:46, 1092.34 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6863/50000 [00:00<00:00, 68629.08 examples/s] 40%|████      | 20204/50000 [00:00<00:00, 80331.14 examples/s] 68%|██████▊   | 33791/50000 [00:00<00:00, 91558.32 examples/s] 95%|█████████▍| 47294/50000 [00:00<00:00, 101346.19 examples/s]                                                                0 examples [00:00, ? examples/s]91 examples [00:00, 907.63 examples/s]203 examples [00:00, 961.60 examples/s]315 examples [00:00, 1003.85 examples/s]418 examples [00:00, 1009.99 examples/s]525 examples [00:00, 1025.06 examples/s]640 examples [00:00, 1057.28 examples/s]755 examples [00:00, 1080.98 examples/s]867 examples [00:00, 1089.74 examples/s]977 examples [00:00, 1092.19 examples/s]1083 examples [00:01, 1052.30 examples/s]1195 examples [00:01, 1071.00 examples/s]1306 examples [00:01, 1081.26 examples/s]1417 examples [00:01, 1088.17 examples/s]1529 examples [00:01, 1096.07 examples/s]1639 examples [00:01, 1094.64 examples/s]1752 examples [00:01, 1104.92 examples/s]1867 examples [00:01, 1115.88 examples/s]1981 examples [00:01, 1120.46 examples/s]2095 examples [00:01, 1123.95 examples/s]2208 examples [00:02, 1114.36 examples/s]2320 examples [00:02, 1114.15 examples/s]2432 examples [00:02, 1049.44 examples/s]2538 examples [00:02, 1038.72 examples/s]2650 examples [00:02, 1059.36 examples/s]2761 examples [00:02, 1071.81 examples/s]2875 examples [00:02, 1088.87 examples/s]2985 examples [00:02, 1082.35 examples/s]3094 examples [00:02, 1082.86 examples/s]3206 examples [00:02, 1092.78 examples/s]3318 examples [00:03, 1099.48 examples/s]3429 examples [00:03, 1102.14 examples/s]3540 examples [00:03, 1089.68 examples/s]3652 examples [00:03, 1098.07 examples/s]3762 examples [00:03, 1083.58 examples/s]3873 examples [00:03, 1089.71 examples/s]3985 examples [00:03, 1096.11 examples/s]4098 examples [00:03, 1104.61 examples/s]4209 examples [00:03, 1093.48 examples/s]4319 examples [00:03, 1094.78 examples/s]4430 examples [00:04, 1096.88 examples/s]4543 examples [00:04, 1104.44 examples/s]4654 examples [00:04, 1104.83 examples/s]4765 examples [00:04, 1104.53 examples/s]4877 examples [00:04, 1107.66 examples/s]4988 examples [00:04, 1105.37 examples/s]5101 examples [00:04, 1110.65 examples/s]5214 examples [00:04, 1113.48 examples/s]5328 examples [00:04, 1119.42 examples/s]5440 examples [00:04, 1119.17 examples/s]5552 examples [00:05, 1112.46 examples/s]5665 examples [00:05, 1113.24 examples/s]5777 examples [00:05, 1114.28 examples/s]5890 examples [00:05, 1117.91 examples/s]6002 examples [00:05, 1115.75 examples/s]6114 examples [00:05, 1111.13 examples/s]6226 examples [00:05, 1101.06 examples/s]6339 examples [00:05, 1108.53 examples/s]6451 examples [00:05, 1110.90 examples/s]6563 examples [00:05, 1110.38 examples/s]6675 examples [00:06, 1111.12 examples/s]6788 examples [00:06, 1115.01 examples/s]6901 examples [00:06, 1117.29 examples/s]7013 examples [00:06, 1118.07 examples/s]7125 examples [00:06, 1117.56 examples/s]7237 examples [00:06, 1109.64 examples/s]7349 examples [00:06, 1112.24 examples/s]7461 examples [00:06, 1113.73 examples/s]7573 examples [00:06, 1111.83 examples/s]7686 examples [00:06, 1115.86 examples/s]7798 examples [00:07, 1113.39 examples/s]7911 examples [00:07, 1116.30 examples/s]8023 examples [00:07, 1113.31 examples/s]8135 examples [00:07, 1113.19 examples/s]8247 examples [00:07, 1114.31 examples/s]8359 examples [00:07, 1111.31 examples/s]8471 examples [00:07, 1112.25 examples/s]8583 examples [00:07, 1110.55 examples/s]8696 examples [00:07, 1114.15 examples/s]8808 examples [00:08, 1108.17 examples/s]8919 examples [00:08, 1106.62 examples/s]9031 examples [00:08, 1109.92 examples/s]9143 examples [00:08, 1107.93 examples/s]9256 examples [00:08, 1112.69 examples/s]9369 examples [00:08, 1115.98 examples/s]9481 examples [00:08, 1115.68 examples/s]9593 examples [00:08, 1116.85 examples/s]9705 examples [00:08, 1113.91 examples/s]9818 examples [00:08, 1118.68 examples/s]9930 examples [00:09, 1113.16 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteO7Z87C/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteO7Z87C/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f35d41e4a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f35d41e4a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f35d41e4a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f361fb51cc0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f355e47f240>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f35d41e4a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f35d41e4a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f35d41e4a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f35d257e278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f35d257e828>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 15.27 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 22.18 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.05 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.05 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.64 file/s]2020-08-05 18:10:55.002677: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-05 18:10:55.007046: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-08-05 18:10:55.007225: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563ddd60e060 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-05 18:10:55.007240: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 482203.59it/s] 88%|████████▊ | 8724480/9912422 [00:00<00:01, 687225.18it/s]9920512it [00:00, 46552702.19it/s]                           
0it [00:00, ?it/s]32768it [00:00, 395924.78it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 456594.67it/s]1654784it [00:00, 11704565.94it/s]                         
0it [00:00, ?it/s]8192it [00:00, 242295.89it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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


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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5c15d5a620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5c15d5a620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5c80a0c400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5c80a0c400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5c9fc28ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5c9fc28ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5c2dd480d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5c2dd480d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5c2dd480d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:24, 116808.87it/s] 54%|█████▍    | 5373952/9912422 [00:00<00:27, 166711.38it/s]9920512it [00:00, 33330671.10it/s]                           
0it [00:00, ?it/s]32768it [00:00, 501884.10it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 158439.47it/s]1654784it [00:00, 11032096.74it/s]                         
0it [00:00, ?it/s]8192it [00:00, 152096.83it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5c15ca8780>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5c2bef54a8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5c2dd3fd08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5c2dd3fd08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5c2dd3fd08> , (data_info, **args) 

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

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:36,  3.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:36,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:34,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:24,  5.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:24,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:23,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:15,  7.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:15,  7.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:15,  7.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:11, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 24.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 24.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 24.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 24.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 24.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 24.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 24.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 30.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 30.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 30.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 30.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 30.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 30.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 30.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 30.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 30.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 37.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 37.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 37.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 37.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 37.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 37.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 37.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 37.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 37.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 37.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 45.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 45.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 45.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 45.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 45.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 45.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 52.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 52.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 52.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 52.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 52.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 52.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 52.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 52.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 52.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 58.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 62.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 68.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 68.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 68.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 68.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 68.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 68.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 68.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 72.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 72.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 72.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 72.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 72.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 72.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 72.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 75.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.51s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.37 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.30s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 78.37 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.30s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.31s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.53 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.31s/ url]
0 examples [00:00, ? examples/s]2020-08-18 18:07:50.864724: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-18 18:07:50.880043: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-18 18:07:50.880253: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564f237e5300 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-18 18:07:50.880273: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
9 examples [00:00, 89.31 examples/s]96 examples [00:00, 122.20 examples/s]186 examples [00:00, 164.89 examples/s]275 examples [00:00, 218.09 examples/s]362 examples [00:00, 281.07 examples/s]432 examples [00:00, 342.55 examples/s]525 examples [00:00, 422.29 examples/s]614 examples [00:00, 501.04 examples/s]703 examples [00:00, 575.33 examples/s]785 examples [00:01, 620.50 examples/s]876 examples [00:01, 685.33 examples/s]966 examples [00:01, 737.81 examples/s]1055 examples [00:01, 776.55 examples/s]1148 examples [00:01, 816.19 examples/s]1237 examples [00:01, 834.56 examples/s]1331 examples [00:01, 862.60 examples/s]1421 examples [00:01, 872.51 examples/s]1511 examples [00:01, 854.57 examples/s]1606 examples [00:01, 880.98 examples/s]1696 examples [00:02, 883.71 examples/s]1786 examples [00:02, 885.31 examples/s]1876 examples [00:02, 812.84 examples/s]1965 examples [00:02, 832.39 examples/s]2053 examples [00:02, 843.97 examples/s]2143 examples [00:02, 859.59 examples/s]2230 examples [00:02, 802.04 examples/s]2313 examples [00:02, 808.88 examples/s]2401 examples [00:02, 826.85 examples/s]2491 examples [00:02, 845.76 examples/s]2579 examples [00:03, 854.77 examples/s]2669 examples [00:03, 866.08 examples/s]2757 examples [00:03, 869.86 examples/s]2845 examples [00:03, 840.31 examples/s]2936 examples [00:03, 858.77 examples/s]3023 examples [00:03, 845.89 examples/s]3108 examples [00:03, 831.68 examples/s]3193 examples [00:03, 835.87 examples/s]3286 examples [00:03, 859.93 examples/s]3377 examples [00:04, 872.57 examples/s]3473 examples [00:04, 894.76 examples/s]3564 examples [00:04, 897.07 examples/s]3654 examples [00:04, 887.11 examples/s]3744 examples [00:04, 889.81 examples/s]3834 examples [00:04, 874.68 examples/s]3922 examples [00:04, 854.61 examples/s]4012 examples [00:04, 867.03 examples/s]4105 examples [00:04, 883.51 examples/s]4195 examples [00:04, 886.66 examples/s]4286 examples [00:05, 892.80 examples/s]4376 examples [00:05, 847.59 examples/s]4462 examples [00:05, 849.16 examples/s]4548 examples [00:05, 827.00 examples/s]4636 examples [00:05, 841.67 examples/s]4723 examples [00:05, 848.86 examples/s]4818 examples [00:05, 874.85 examples/s]4906 examples [00:05, 869.00 examples/s]5001 examples [00:05, 890.86 examples/s]5091 examples [00:05, 867.87 examples/s]5179 examples [00:06, 849.78 examples/s]5270 examples [00:06, 865.97 examples/s]5357 examples [00:06, 858.28 examples/s]5444 examples [00:06, 851.23 examples/s]5531 examples [00:06, 856.18 examples/s]5617 examples [00:06, 850.47 examples/s]5703 examples [00:06, 846.26 examples/s]5788 examples [00:06, 810.69 examples/s]5873 examples [00:06, 821.98 examples/s]5958 examples [00:07, 829.61 examples/s]6046 examples [00:07, 844.02 examples/s]6138 examples [00:07, 865.22 examples/s]6232 examples [00:07, 884.46 examples/s]6321 examples [00:07, 852.43 examples/s]6407 examples [00:07, 822.83 examples/s]6498 examples [00:07, 847.18 examples/s]6591 examples [00:07, 869.40 examples/s]6679 examples [00:07, 866.47 examples/s]6777 examples [00:07, 895.99 examples/s]6868 examples [00:08, 897.38 examples/s]6960 examples [00:08, 901.88 examples/s]7051 examples [00:08, 896.44 examples/s]7141 examples [00:08, 868.04 examples/s]7230 examples [00:08, 874.35 examples/s]7318 examples [00:08, 872.04 examples/s]7406 examples [00:08, 866.52 examples/s]7495 examples [00:08, 872.18 examples/s]7583 examples [00:08, 847.97 examples/s]7673 examples [00:08, 862.00 examples/s]7760 examples [00:09, 851.82 examples/s]7851 examples [00:09, 867.13 examples/s]7938 examples [00:09, 857.19 examples/s]8026 examples [00:09, 862.42 examples/s]8117 examples [00:09, 875.36 examples/s]8209 examples [00:09, 888.26 examples/s]8298 examples [00:09, 866.05 examples/s]8385 examples [00:09, 845.94 examples/s]8470 examples [00:09, 820.05 examples/s]8553 examples [00:10, 800.01 examples/s]8643 examples [00:10, 826.12 examples/s]8727 examples [00:10, 828.69 examples/s]8812 examples [00:10, 832.54 examples/s]8896 examples [00:10, 820.72 examples/s]8983 examples [00:10, 833.39 examples/s]9067 examples [00:10, 833.48 examples/s]9151 examples [00:10, 809.29 examples/s]9237 examples [00:10, 823.85 examples/s]9327 examples [00:10, 843.23 examples/s]9421 examples [00:11, 868.84 examples/s]9518 examples [00:11, 896.18 examples/s]9609 examples [00:11, 892.54 examples/s]9699 examples [00:11, 883.39 examples/s]9788 examples [00:11, 859.31 examples/s]9875 examples [00:11, 860.72 examples/s]9962 examples [00:11, 859.12 examples/s]10049 examples [00:11, 771.83 examples/s]10138 examples [00:11, 803.31 examples/s]10220 examples [00:12, 800.16 examples/s]10315 examples [00:12, 837.89 examples/s]10409 examples [00:12, 864.88 examples/s]10500 examples [00:12, 876.80 examples/s]10589 examples [00:12, 869.59 examples/s]10678 examples [00:12, 872.94 examples/s]10770 examples [00:12, 885.62 examples/s]10859 examples [00:12, 878.76 examples/s]10950 examples [00:12, 885.62 examples/s]11039 examples [00:12, 854.59 examples/s]11125 examples [00:13, 826.90 examples/s]11215 examples [00:13, 847.17 examples/s]11304 examples [00:13, 858.72 examples/s]11391 examples [00:13, 854.94 examples/s]11477 examples [00:13, 789.28 examples/s]11568 examples [00:13, 819.85 examples/s]11661 examples [00:13, 847.65 examples/s]11753 examples [00:13, 865.77 examples/s]11845 examples [00:13, 880.44 examples/s]11934 examples [00:14, 865.04 examples/s]12027 examples [00:14, 881.39 examples/s]12121 examples [00:14, 896.47 examples/s]12212 examples [00:14, 884.17 examples/s]12302 examples [00:14, 887.35 examples/s]12396 examples [00:14, 901.35 examples/s]12487 examples [00:14, 897.00 examples/s]12577 examples [00:14, 836.57 examples/s]12662 examples [00:14, 804.04 examples/s]12750 examples [00:14, 823.38 examples/s]12834 examples [00:15, 826.90 examples/s]12918 examples [00:15, 823.24 examples/s]13001 examples [00:15, 817.77 examples/s]13086 examples [00:15, 826.94 examples/s]13170 examples [00:15, 829.31 examples/s]13254 examples [00:15, 821.13 examples/s]13340 examples [00:15, 831.22 examples/s]13430 examples [00:15, 848.11 examples/s]13515 examples [00:15, 848.52 examples/s]13600 examples [00:15, 842.28 examples/s]13690 examples [00:16, 857.91 examples/s]13780 examples [00:16, 869.36 examples/s]13869 examples [00:16, 874.35 examples/s]13957 examples [00:16, 830.35 examples/s]14041 examples [00:16, 806.96 examples/s]14123 examples [00:16, 792.85 examples/s]14203 examples [00:16, 790.15 examples/s]14292 examples [00:16, 815.80 examples/s]14374 examples [00:16, 793.38 examples/s]14454 examples [00:17, 794.01 examples/s]14542 examples [00:17, 815.66 examples/s]14633 examples [00:17, 839.40 examples/s]14720 examples [00:17, 846.41 examples/s]14808 examples [00:17, 855.62 examples/s]14899 examples [00:17, 869.47 examples/s]14987 examples [00:17, 872.36 examples/s]15075 examples [00:17, 848.97 examples/s]15163 examples [00:17, 856.33 examples/s]15251 examples [00:17, 860.51 examples/s]15342 examples [00:18, 871.26 examples/s]15430 examples [00:18, 860.18 examples/s]15517 examples [00:18, 844.28 examples/s]15602 examples [00:18, 837.65 examples/s]15686 examples [00:18, 821.85 examples/s]15778 examples [00:18, 847.89 examples/s]15864 examples [00:18, 820.83 examples/s]15953 examples [00:18, 839.79 examples/s]16042 examples [00:18, 852.20 examples/s]16132 examples [00:18, 864.67 examples/s]16219 examples [00:19, 853.61 examples/s]16307 examples [00:19, 858.62 examples/s]16398 examples [00:19, 872.01 examples/s]16486 examples [00:19, 874.09 examples/s]16576 examples [00:19, 881.64 examples/s]16665 examples [00:19, 883.08 examples/s]16754 examples [00:19, 850.11 examples/s]16840 examples [00:19, 851.06 examples/s]16926 examples [00:19, 790.72 examples/s]17012 examples [00:20, 810.01 examples/s]17094 examples [00:20, 811.41 examples/s]17178 examples [00:20, 819.49 examples/s]17270 examples [00:20, 846.86 examples/s]17356 examples [00:20, 849.45 examples/s]17442 examples [00:20, 815.72 examples/s]17530 examples [00:20, 831.43 examples/s]17615 examples [00:20, 834.52 examples/s]17707 examples [00:20, 856.51 examples/s]17801 examples [00:20, 878.21 examples/s]17890 examples [00:21, 876.41 examples/s]17979 examples [00:21, 875.17 examples/s]18070 examples [00:21, 882.59 examples/s]18159 examples [00:21, 882.20 examples/s]18248 examples [00:21, 819.66 examples/s]18331 examples [00:21, 821.93 examples/s]18426 examples [00:21, 856.02 examples/s]18514 examples [00:21, 860.63 examples/s]18603 examples [00:21, 867.26 examples/s]18691 examples [00:21, 848.47 examples/s]18777 examples [00:22, 847.53 examples/s]18864 examples [00:22, 852.68 examples/s]18950 examples [00:22, 813.34 examples/s]19037 examples [00:22, 829.30 examples/s]19124 examples [00:22, 840.87 examples/s]19211 examples [00:22, 848.73 examples/s]19297 examples [00:22, 833.66 examples/s]19381 examples [00:22, 823.64 examples/s]19471 examples [00:22, 844.87 examples/s]19556 examples [00:23, 817.16 examples/s]19639 examples [00:23, 802.41 examples/s]19725 examples [00:23, 816.98 examples/s]19814 examples [00:23, 835.40 examples/s]19902 examples [00:23, 847.80 examples/s]19989 examples [00:23, 852.39 examples/s]20075 examples [00:23, 788.98 examples/s]20155 examples [00:23, 785.00 examples/s]20235 examples [00:23, 788.87 examples/s]20319 examples [00:23, 802.32 examples/s]20411 examples [00:24, 833.79 examples/s]20500 examples [00:24, 848.63 examples/s]20588 examples [00:24, 855.94 examples/s]20677 examples [00:24, 863.37 examples/s]20770 examples [00:24, 879.85 examples/s]20861 examples [00:24, 886.44 examples/s]20950 examples [00:24, 864.30 examples/s]21040 examples [00:24, 874.42 examples/s]21128 examples [00:24, 856.04 examples/s]21219 examples [00:24, 868.97 examples/s]21307 examples [00:25, 838.48 examples/s]21393 examples [00:25, 843.30 examples/s]21478 examples [00:25, 833.76 examples/s]21563 examples [00:25, 836.81 examples/s]21651 examples [00:25, 847.05 examples/s]21740 examples [00:25, 856.92 examples/s]21832 examples [00:25, 873.69 examples/s]21920 examples [00:25, 835.72 examples/s]22008 examples [00:25, 847.78 examples/s]22098 examples [00:26, 861.28 examples/s]22190 examples [00:26, 876.11 examples/s]22278 examples [00:26, 871.45 examples/s]22366 examples [00:26, 852.41 examples/s]22452 examples [00:26, 852.40 examples/s]22538 examples [00:26, 849.37 examples/s]22624 examples [00:26, 844.47 examples/s]22709 examples [00:26, 820.84 examples/s]22793 examples [00:26, 824.21 examples/s]22879 examples [00:26, 832.29 examples/s]22964 examples [00:27, 836.45 examples/s]23055 examples [00:27, 855.78 examples/s]23144 examples [00:27, 864.52 examples/s]23234 examples [00:27, 873.92 examples/s]23325 examples [00:27, 884.40 examples/s]23414 examples [00:27, 879.25 examples/s]23504 examples [00:27, 884.33 examples/s]23593 examples [00:27, 881.24 examples/s]23682 examples [00:27, 879.85 examples/s]23771 examples [00:27, 878.33 examples/s]23859 examples [00:28, 874.71 examples/s]23948 examples [00:28, 877.24 examples/s]24036 examples [00:28, 837.52 examples/s]24125 examples [00:28, 851.60 examples/s]24211 examples [00:28, 827.19 examples/s]24295 examples [00:28, 769.95 examples/s]24374 examples [00:28, 769.22 examples/s]24459 examples [00:28, 783.37 examples/s]24545 examples [00:28, 803.67 examples/s]24626 examples [00:29, 767.42 examples/s]24704 examples [00:29, 752.28 examples/s]24780 examples [00:29, 739.33 examples/s]24862 examples [00:29, 760.58 examples/s]24942 examples [00:29, 769.93 examples/s]25028 examples [00:29, 793.58 examples/s]25108 examples [00:29, 784.87 examples/s]25197 examples [00:29, 813.63 examples/s]25285 examples [00:29, 827.81 examples/s]25369 examples [00:29, 817.72 examples/s]25452 examples [00:30, 820.09 examples/s]25538 examples [00:30, 831.21 examples/s]25623 examples [00:30, 835.91 examples/s]25710 examples [00:30, 845.40 examples/s]25795 examples [00:30, 832.10 examples/s]25884 examples [00:30, 846.59 examples/s]25976 examples [00:30, 865.47 examples/s]26063 examples [00:30, 855.59 examples/s]26150 examples [00:30, 859.22 examples/s]26237 examples [00:31, 858.05 examples/s]26323 examples [00:31, 855.94 examples/s]26409 examples [00:31, 816.98 examples/s]26497 examples [00:31, 833.91 examples/s]26589 examples [00:31, 855.34 examples/s]26675 examples [00:31, 840.68 examples/s]26760 examples [00:31, 836.64 examples/s]26844 examples [00:31, 826.76 examples/s]26927 examples [00:31, 809.34 examples/s]27010 examples [00:31, 814.70 examples/s]27092 examples [00:32, 789.05 examples/s]27181 examples [00:32, 815.35 examples/s]27271 examples [00:32, 838.43 examples/s]27357 examples [00:32, 843.00 examples/s]27443 examples [00:32, 845.77 examples/s]27528 examples [00:32, 843.63 examples/s]27613 examples [00:32, 827.28 examples/s]27697 examples [00:32, 830.33 examples/s]27783 examples [00:32, 837.89 examples/s]27873 examples [00:32, 852.98 examples/s]27963 examples [00:33, 866.54 examples/s]28050 examples [00:33, 841.29 examples/s]28137 examples [00:33, 849.05 examples/s]28223 examples [00:33, 848.95 examples/s]28311 examples [00:33, 856.30 examples/s]28397 examples [00:33, 846.49 examples/s]28482 examples [00:33, 834.88 examples/s]28568 examples [00:33, 838.58 examples/s]28656 examples [00:33, 848.39 examples/s]28743 examples [00:33, 852.59 examples/s]28833 examples [00:34, 863.89 examples/s]28920 examples [00:34, 857.10 examples/s]29006 examples [00:34, 855.47 examples/s]29092 examples [00:34, 851.93 examples/s]29178 examples [00:34, 853.52 examples/s]29264 examples [00:34, 848.85 examples/s]29349 examples [00:34, 816.86 examples/s]29438 examples [00:34, 835.12 examples/s]29526 examples [00:34, 846.62 examples/s]29617 examples [00:35, 860.82 examples/s]29707 examples [00:35, 870.02 examples/s]29798 examples [00:35, 879.87 examples/s]29887 examples [00:35, 862.39 examples/s]29979 examples [00:35, 876.13 examples/s]30067 examples [00:35, 824.81 examples/s]30156 examples [00:35, 840.91 examples/s]30248 examples [00:35, 860.82 examples/s]30335 examples [00:35, 851.24 examples/s]30423 examples [00:35, 858.36 examples/s]30510 examples [00:36, 859.53 examples/s]30597 examples [00:36, 858.21 examples/s]30683 examples [00:36, 835.54 examples/s]30767 examples [00:36, 794.40 examples/s]30848 examples [00:36, 797.03 examples/s]30938 examples [00:36, 823.01 examples/s]31028 examples [00:36, 843.87 examples/s]31116 examples [00:36, 852.94 examples/s]31204 examples [00:36, 858.29 examples/s]31293 examples [00:36, 865.34 examples/s]31383 examples [00:37, 872.68 examples/s]31472 examples [00:37, 877.19 examples/s]31560 examples [00:37, 861.56 examples/s]31647 examples [00:37, 858.99 examples/s]31733 examples [00:37, 850.53 examples/s]31819 examples [00:37, 811.75 examples/s]31907 examples [00:37, 830.85 examples/s]31991 examples [00:37, 832.60 examples/s]32075 examples [00:37, 822.62 examples/s]32160 examples [00:38, 830.03 examples/s]32248 examples [00:38, 842.28 examples/s]32333 examples [00:38, 842.48 examples/s]32418 examples [00:38, 827.42 examples/s]32512 examples [00:38, 856.56 examples/s]32603 examples [00:38, 871.81 examples/s]32693 examples [00:38, 879.60 examples/s]32782 examples [00:38, 850.65 examples/s]32868 examples [00:38, 824.60 examples/s]32951 examples [00:38, 790.44 examples/s]33040 examples [00:39, 817.66 examples/s]33125 examples [00:39, 825.53 examples/s]33209 examples [00:39, 820.77 examples/s]33303 examples [00:39, 851.30 examples/s]33389 examples [00:39, 822.85 examples/s]33472 examples [00:39, 797.26 examples/s]33553 examples [00:39, 793.58 examples/s]33638 examples [00:39, 808.78 examples/s]33725 examples [00:39, 825.81 examples/s]33808 examples [00:40, 785.39 examples/s]33899 examples [00:40, 817.22 examples/s]33982 examples [00:40, 817.25 examples/s]34072 examples [00:40, 840.26 examples/s]34157 examples [00:40, 837.52 examples/s]34245 examples [00:40, 847.22 examples/s]34331 examples [00:40, 837.41 examples/s]34416 examples [00:40, 835.44 examples/s]34500 examples [00:40, 822.34 examples/s]34583 examples [00:40, 812.94 examples/s]34673 examples [00:41, 836.08 examples/s]34757 examples [00:41, 832.24 examples/s]34841 examples [00:41, 832.52 examples/s]34931 examples [00:41, 850.88 examples/s]35019 examples [00:41, 858.07 examples/s]35108 examples [00:41, 866.90 examples/s]35195 examples [00:41, 855.07 examples/s]35281 examples [00:41, 847.22 examples/s]35369 examples [00:41, 855.33 examples/s]35458 examples [00:41, 863.97 examples/s]35548 examples [00:42, 873.22 examples/s]35637 examples [00:42, 875.86 examples/s]35725 examples [00:42, 837.19 examples/s]35813 examples [00:42, 848.41 examples/s]35899 examples [00:42, 842.96 examples/s]35990 examples [00:42, 859.93 examples/s]36080 examples [00:42, 869.09 examples/s]36168 examples [00:42, 859.58 examples/s]36256 examples [00:42, 863.59 examples/s]36343 examples [00:43, 849.33 examples/s]36429 examples [00:43, 849.32 examples/s]36515 examples [00:43, 827.35 examples/s]36605 examples [00:43, 847.67 examples/s]36691 examples [00:43, 846.38 examples/s]36777 examples [00:43, 847.74 examples/s]36862 examples [00:43, 815.74 examples/s]36944 examples [00:43, 810.56 examples/s]37026 examples [00:43, 794.62 examples/s]37115 examples [00:43, 819.79 examples/s]37206 examples [00:44, 844.56 examples/s]37291 examples [00:44, 825.26 examples/s]37374 examples [00:44, 790.90 examples/s]37464 examples [00:44, 820.60 examples/s]37551 examples [00:44, 833.23 examples/s]37644 examples [00:44, 857.56 examples/s]37736 examples [00:44, 873.25 examples/s]37827 examples [00:44, 881.18 examples/s]37916 examples [00:44, 863.71 examples/s]38004 examples [00:44, 866.69 examples/s]38091 examples [00:45, 856.38 examples/s]38181 examples [00:45, 868.21 examples/s]38275 examples [00:45, 888.16 examples/s]38365 examples [00:45, 890.44 examples/s]38455 examples [00:45, 864.36 examples/s]38542 examples [00:45, 818.72 examples/s]38633 examples [00:45, 842.14 examples/s]38724 examples [00:45, 860.19 examples/s]38811 examples [00:45, 862.63 examples/s]38898 examples [00:46, 821.62 examples/s]38981 examples [00:46, 823.04 examples/s]39072 examples [00:46, 845.51 examples/s]39158 examples [00:46, 836.03 examples/s]39242 examples [00:46, 817.90 examples/s]39334 examples [00:46, 844.03 examples/s]39421 examples [00:46, 850.80 examples/s]39511 examples [00:46, 862.35 examples/s]39601 examples [00:46, 872.38 examples/s]39689 examples [00:46, 841.10 examples/s]39780 examples [00:47, 858.88 examples/s]39871 examples [00:47, 872.05 examples/s]39962 examples [00:47, 882.61 examples/s]40051 examples [00:47, 810.52 examples/s]40140 examples [00:47, 831.40 examples/s]40228 examples [00:47, 844.63 examples/s]40316 examples [00:47, 854.01 examples/s]40402 examples [00:47, 855.08 examples/s]40492 examples [00:47, 867.02 examples/s]40582 examples [00:47, 874.57 examples/s]40670 examples [00:48, 853.08 examples/s]40760 examples [00:48, 864.60 examples/s]40851 examples [00:48, 876.13 examples/s]40943 examples [00:48, 887.01 examples/s]41034 examples [00:48, 892.24 examples/s]41126 examples [00:48, 897.94 examples/s]41216 examples [00:48, 888.89 examples/s]41305 examples [00:48, 879.76 examples/s]41396 examples [00:48, 886.27 examples/s]41485 examples [00:49, 870.16 examples/s]41573 examples [00:49, 845.98 examples/s]41658 examples [00:49, 807.82 examples/s]41750 examples [00:49, 836.24 examples/s]41838 examples [00:49, 846.77 examples/s]41929 examples [00:49, 864.62 examples/s]42019 examples [00:49, 873.35 examples/s]42108 examples [00:49, 876.30 examples/s]42196 examples [00:49, 872.99 examples/s]42284 examples [00:49, 842.93 examples/s]42370 examples [00:50, 845.63 examples/s]42458 examples [00:50, 853.17 examples/s]42547 examples [00:50, 863.87 examples/s]42638 examples [00:50, 875.14 examples/s]42727 examples [00:50, 877.49 examples/s]42815 examples [00:50, 865.86 examples/s]42903 examples [00:50, 867.99 examples/s]42990 examples [00:50, 825.37 examples/s]43079 examples [00:50, 842.97 examples/s]43164 examples [00:51, 841.99 examples/s]43252 examples [00:51, 851.41 examples/s]43338 examples [00:51, 835.05 examples/s]43426 examples [00:51, 846.77 examples/s]43511 examples [00:51, 830.25 examples/s]43595 examples [00:51, 820.66 examples/s]43678 examples [00:51, 815.32 examples/s]43760 examples [00:51, 807.75 examples/s]43845 examples [00:51, 817.27 examples/s]43934 examples [00:51, 835.95 examples/s]44018 examples [00:52, 827.64 examples/s]44107 examples [00:52, 844.00 examples/s]44196 examples [00:52, 855.11 examples/s]44283 examples [00:52, 857.40 examples/s]44373 examples [00:52, 867.64 examples/s]44461 examples [00:52, 867.55 examples/s]44548 examples [00:52, 841.76 examples/s]44634 examples [00:52, 846.89 examples/s]44719 examples [00:52, 811.37 examples/s]44801 examples [00:52, 806.85 examples/s]44888 examples [00:53, 822.72 examples/s]44972 examples [00:53, 826.70 examples/s]45055 examples [00:53, 819.62 examples/s]45142 examples [00:53, 833.61 examples/s]45226 examples [00:53, 829.92 examples/s]45310 examples [00:53, 796.98 examples/s]45394 examples [00:53, 808.76 examples/s]45484 examples [00:53, 833.61 examples/s]45568 examples [00:53, 824.98 examples/s]45658 examples [00:53, 844.87 examples/s]45747 examples [00:54, 857.78 examples/s]45837 examples [00:54, 869.35 examples/s]45925 examples [00:54, 868.98 examples/s]46013 examples [00:54, 859.03 examples/s]46105 examples [00:54, 873.86 examples/s]46193 examples [00:54, 837.93 examples/s]46278 examples [00:54, 837.04 examples/s]46370 examples [00:54, 858.87 examples/s]46457 examples [00:54, 861.55 examples/s]46545 examples [00:55, 865.67 examples/s]46632 examples [00:55, 825.06 examples/s]46718 examples [00:55, 833.26 examples/s]46807 examples [00:55, 846.47 examples/s]46897 examples [00:55, 859.96 examples/s]46989 examples [00:55, 876.37 examples/s]47080 examples [00:55, 885.79 examples/s]47172 examples [00:55, 895.26 examples/s]47262 examples [00:55, 890.71 examples/s]47353 examples [00:55, 893.80 examples/s]47443 examples [00:56, 891.82 examples/s]47533 examples [00:56, 889.36 examples/s]47623 examples [00:56, 890.83 examples/s]47713 examples [00:56, 891.54 examples/s]47803 examples [00:56, 891.66 examples/s]47893 examples [00:56, 874.58 examples/s]47981 examples [00:56, 875.16 examples/s]48071 examples [00:56, 879.64 examples/s]48161 examples [00:56, 883.33 examples/s]48252 examples [00:56, 889.85 examples/s]48342 examples [00:57, 885.00 examples/s]48432 examples [00:57, 886.82 examples/s]48521 examples [00:57, 883.28 examples/s]48610 examples [00:57, 883.76 examples/s]48699 examples [00:57, 872.91 examples/s]48791 examples [00:57, 884.06 examples/s]48884 examples [00:57, 895.16 examples/s]48974 examples [00:57, 895.85 examples/s]49066 examples [00:57, 901.42 examples/s]49158 examples [00:57, 903.99 examples/s]49249 examples [00:58, 902.58 examples/s]49340 examples [00:58, 895.93 examples/s]49430 examples [00:58, 892.10 examples/s]49520 examples [00:58, 892.96 examples/s]49610 examples [00:58, 884.07 examples/s]49699 examples [00:58, 864.37 examples/s]49789 examples [00:58, 871.41 examples/s]49882 examples [00:58, 887.78 examples/s]49971 examples [00:58, 885.74 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5470/50000 [00:00<00:00, 54696.53 examples/s] 33%|███▎      | 16514/50000 [00:00<00:00, 64456.30 examples/s] 55%|█████▍    | 27472/50000 [00:00<00:00, 73540.77 examples/s] 76%|███████▌  | 37951/50000 [00:00<00:00, 80765.90 examples/s] 98%|█████████▊| 48849/50000 [00:00<00:00, 87565.54 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 694.62 examples/s]156 examples [00:00, 734.73 examples/s]243 examples [00:00, 768.72 examples/s]336 examples [00:00, 809.67 examples/s]427 examples [00:00, 836.92 examples/s]517 examples [00:00, 852.79 examples/s]602 examples [00:00, 850.77 examples/s]685 examples [00:00, 843.21 examples/s]775 examples [00:00, 858.84 examples/s]865 examples [00:01, 868.33 examples/s]953 examples [00:01, 871.14 examples/s]1039 examples [00:01, 860.07 examples/s]1125 examples [00:01, 823.93 examples/s]1218 examples [00:01, 851.06 examples/s]1308 examples [00:01, 862.22 examples/s]1398 examples [00:01, 872.60 examples/s]1486 examples [00:01, 874.16 examples/s]1574 examples [00:01, 875.71 examples/s]1664 examples [00:01, 880.77 examples/s]1754 examples [00:02, 886.42 examples/s]1844 examples [00:02, 889.56 examples/s]1933 examples [00:02, 883.58 examples/s]2022 examples [00:02, 872.68 examples/s]2110 examples [00:02, 868.07 examples/s]2197 examples [00:02, 854.14 examples/s]2287 examples [00:02, 866.78 examples/s]2378 examples [00:02, 877.58 examples/s]2467 examples [00:02, 880.34 examples/s]2557 examples [00:02, 884.51 examples/s]2648 examples [00:03, 889.91 examples/s]2739 examples [00:03, 893.59 examples/s]2830 examples [00:03, 896.60 examples/s]2920 examples [00:03, 872.79 examples/s]3008 examples [00:03, 776.52 examples/s]3093 examples [00:03, 796.11 examples/s]3179 examples [00:03, 813.18 examples/s]3263 examples [00:03, 820.13 examples/s]3357 examples [00:03, 851.77 examples/s]3447 examples [00:04, 865.65 examples/s]3538 examples [00:04, 876.72 examples/s]3629 examples [00:04, 883.92 examples/s]3718 examples [00:04, 884.82 examples/s]3807 examples [00:04, 879.10 examples/s]3896 examples [00:04, 870.79 examples/s]3984 examples [00:04, 866.65 examples/s]4075 examples [00:04, 879.09 examples/s]4164 examples [00:04, 863.75 examples/s]4251 examples [00:04, 857.63 examples/s]4337 examples [00:05, 845.01 examples/s]4422 examples [00:05, 810.03 examples/s]4509 examples [00:05, 825.47 examples/s]4594 examples [00:05, 832.12 examples/s]4678 examples [00:05, 829.09 examples/s]4766 examples [00:05, 841.80 examples/s]4851 examples [00:05, 826.29 examples/s]4935 examples [00:05, 828.13 examples/s]5025 examples [00:05, 847.11 examples/s]5112 examples [00:05, 852.57 examples/s]5201 examples [00:06, 861.76 examples/s]5294 examples [00:06, 880.99 examples/s]5386 examples [00:06, 891.88 examples/s]5476 examples [00:06, 891.01 examples/s]5572 examples [00:06, 907.96 examples/s]5664 examples [00:06, 909.51 examples/s]5756 examples [00:06, 891.55 examples/s]5846 examples [00:06, 874.27 examples/s]5934 examples [00:06, 875.52 examples/s]6022 examples [00:06, 864.78 examples/s]6109 examples [00:07, 856.37 examples/s]6195 examples [00:07, 854.62 examples/s]6285 examples [00:07, 865.62 examples/s]6377 examples [00:07, 878.94 examples/s]6468 examples [00:07, 884.09 examples/s]6557 examples [00:07, 868.42 examples/s]6647 examples [00:07, 875.76 examples/s]6735 examples [00:07, 866.10 examples/s]6827 examples [00:07, 879.89 examples/s]6921 examples [00:08, 895.81 examples/s]7014 examples [00:08, 905.79 examples/s]7105 examples [00:08, 895.11 examples/s]7195 examples [00:08, 858.81 examples/s]7285 examples [00:08, 870.40 examples/s]7376 examples [00:08, 880.54 examples/s]7467 examples [00:08, 886.51 examples/s]7556 examples [00:08, 866.01 examples/s]7646 examples [00:08, 873.34 examples/s]7734 examples [00:08, 861.83 examples/s]7822 examples [00:09, 866.86 examples/s]7915 examples [00:09, 882.82 examples/s]8004 examples [00:09, 857.77 examples/s]8091 examples [00:09, 858.75 examples/s]8181 examples [00:09, 870.09 examples/s]8272 examples [00:09, 881.25 examples/s]8361 examples [00:09, 870.18 examples/s]8449 examples [00:09, 858.09 examples/s]8535 examples [00:09, 849.73 examples/s]8621 examples [00:09, 843.30 examples/s]8713 examples [00:10, 862.86 examples/s]8806 examples [00:10, 881.15 examples/s]8899 examples [00:10, 893.40 examples/s]8989 examples [00:10, 894.91 examples/s]9079 examples [00:10, 854.79 examples/s]9165 examples [00:10, 853.43 examples/s]9251 examples [00:10, 838.84 examples/s]9336 examples [00:10, 839.40 examples/s]9421 examples [00:10, 815.18 examples/s]9503 examples [00:11, 810.94 examples/s]9595 examples [00:11, 839.41 examples/s]9680 examples [00:11, 840.86 examples/s]9769 examples [00:11, 854.84 examples/s]9855 examples [00:11, 847.40 examples/s]9940 examples [00:11, 833.17 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRR4OH4/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRR4OH4/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5c2dd480d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5c2dd480d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5c2dd480d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5c79726828>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5bb3fc5550>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5c2dd480d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5c2dd480d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5c2dd480d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5c2bef55f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5c2bef57b8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.94 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 23.66 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.87 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.87 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.54 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.54 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.88 file/s]2020-08-18 18:09:10.906159: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-18 18:09:10.910560: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-18 18:09:10.910764: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555828707aa0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-18 18:09:10.910783: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:24, 116613.23it/s]9920512it [00:00, 43928403.97it/s]                         
0it [00:00, ?it/s]32768it [00:00, 488748.30it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 144720.42it/s]1654784it [00:00, 10099614.46it/s]                         
0it [00:00, ?it/s]8192it [00:00, 164597.55it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

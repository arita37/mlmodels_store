
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5ab3c84b70> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5ab3c84b70> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5b1e8f4510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5b1e8f4510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5b3db21ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5b3db21ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5acbc9ca60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5acbc9ca60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5acbc9ca60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 155909.56it/s] 71%|███████   | 7045120/9912422 [00:00<00:12, 222515.80it/s]9920512it [00:00, 42677738.94it/s]                           
0it [00:00, ?it/s]32768it [00:00, 578858.50it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 148120.52it/s]1654784it [00:00, 11458843.18it/s]                         
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 62369.40it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ab3b406d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ab3b4d978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5acbc9c6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5acbc9c6a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5acbc9c6a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:01,  2.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:01,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:00,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:00,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:59,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:59,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:59,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:58,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:41,  3.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:41,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:41,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:41,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:40,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:40,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:40,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:39,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:28,  5.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:28,  5.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:28,  5.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:28,  5.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:27,  5.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:27,  5.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:27,  5.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:27,  5.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:27,  5.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:19,  7.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:19,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:19,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:19,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:18,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:18,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:18,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:13,  9.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:13,  9.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:13,  9.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:12,  9.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:12,  9.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:12,  9.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:12,  9.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:12,  9.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:12,  9.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:12,  9.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 13.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:08, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:08, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:08, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:08, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 18.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 18.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 18.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 18.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 18.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 23.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 23.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 30.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 30.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 30.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 30.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 30.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 30.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 30.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 37.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 37.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 37.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 37.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 37.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 37.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 37.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 37.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 37.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 37.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 45.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 52.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 52.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 52.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 52.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 52.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 52.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 52.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 52.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 52.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 52.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 59.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 59.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 59.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 59.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 59.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 59.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 59.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 59.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 59.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 59.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 62.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 64.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 64.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 64.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 64.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 64.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 64.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 64.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 64.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 64.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 64.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 69.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 69.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 69.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 69.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 69.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 69.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 69.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 73.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 76.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 76.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 76.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 76.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 76.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 76.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.36s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.67 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.77s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 78.67 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.77s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.77s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.94 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.77s/ url]
0 examples [00:00, ? examples/s]2020-07-30 18:10:01.638301: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-30 18:10:01.651513: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-07-30 18:10:01.651727: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558ef4f41580 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-30 18:10:01.651750: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
15 examples [00:00, 149.07 examples/s]104 examples [00:00, 198.58 examples/s]201 examples [00:00, 260.58 examples/s]292 examples [00:00, 331.31 examples/s]383 examples [00:00, 408.57 examples/s]464 examples [00:00, 479.27 examples/s]553 examples [00:00, 555.84 examples/s]648 examples [00:00, 633.61 examples/s]740 examples [00:00, 698.33 examples/s]839 examples [00:01, 764.73 examples/s]936 examples [00:01, 815.38 examples/s]1035 examples [00:01, 860.44 examples/s]1133 examples [00:01, 892.94 examples/s]1233 examples [00:01, 921.39 examples/s]1330 examples [00:01, 933.21 examples/s]1427 examples [00:01, 938.34 examples/s]1526 examples [00:01, 952.53 examples/s]1623 examples [00:01, 957.13 examples/s]1720 examples [00:01, 958.61 examples/s]1817 examples [00:02, 958.02 examples/s]1914 examples [00:02, 940.77 examples/s]2014 examples [00:02, 955.50 examples/s]2112 examples [00:02, 961.51 examples/s]2213 examples [00:02, 973.37 examples/s]2312 examples [00:02, 975.67 examples/s]2410 examples [00:02, 972.56 examples/s]2511 examples [00:02, 983.38 examples/s]2610 examples [00:02, 983.75 examples/s]2709 examples [00:02, 979.47 examples/s]2808 examples [00:03, 976.34 examples/s]2906 examples [00:03, 967.85 examples/s]3004 examples [00:03, 968.75 examples/s]3103 examples [00:03, 972.66 examples/s]3201 examples [00:03, 963.34 examples/s]3298 examples [00:03, 954.43 examples/s]3394 examples [00:03, 927.83 examples/s]3487 examples [00:03, 921.83 examples/s]3581 examples [00:03, 925.76 examples/s]3675 examples [00:03, 927.50 examples/s]3775 examples [00:04, 946.31 examples/s]3872 examples [00:04, 951.13 examples/s]3973 examples [00:04, 966.67 examples/s]4071 examples [00:04, 968.82 examples/s]4169 examples [00:04, 972.03 examples/s]4270 examples [00:04, 982.30 examples/s]4369 examples [00:04, 976.91 examples/s]4467 examples [00:04, 962.96 examples/s]4564 examples [00:04, 959.82 examples/s]4661 examples [00:04, 962.67 examples/s]4758 examples [00:05, 963.95 examples/s]4855 examples [00:05, 960.57 examples/s]4953 examples [00:05, 964.04 examples/s]5050 examples [00:05, 962.08 examples/s]5147 examples [00:05, 963.05 examples/s]5244 examples [00:05, 961.41 examples/s]5341 examples [00:05, 959.64 examples/s]5438 examples [00:05, 960.43 examples/s]5536 examples [00:05, 964.88 examples/s]5634 examples [00:05, 969.14 examples/s]5731 examples [00:06, 967.68 examples/s]5828 examples [00:06, 944.21 examples/s]5923 examples [00:06, 937.68 examples/s]6020 examples [00:06, 942.74 examples/s]6118 examples [00:06, 952.89 examples/s]6217 examples [00:06, 963.15 examples/s]6314 examples [00:06, 955.83 examples/s]6413 examples [00:06, 964.47 examples/s]6511 examples [00:06, 967.06 examples/s]6610 examples [00:06, 971.18 examples/s]6708 examples [00:07, 967.80 examples/s]6805 examples [00:07, 966.54 examples/s]6902 examples [00:07, 965.46 examples/s]6999 examples [00:07, 963.49 examples/s]7100 examples [00:07, 975.38 examples/s]7201 examples [00:07, 985.32 examples/s]7300 examples [00:07, 980.95 examples/s]7399 examples [00:07, 973.80 examples/s]7497 examples [00:07, 966.93 examples/s]7594 examples [00:08, 949.86 examples/s]7693 examples [00:08, 959.07 examples/s]7789 examples [00:08, 950.73 examples/s]7885 examples [00:08, 951.59 examples/s]7984 examples [00:08, 960.81 examples/s]8081 examples [00:08, 960.98 examples/s]8179 examples [00:08, 966.55 examples/s]8276 examples [00:08, 963.98 examples/s]8373 examples [00:08, 962.99 examples/s]8471 examples [00:08, 968.01 examples/s]8568 examples [00:09, 966.92 examples/s]8665 examples [00:09, 964.39 examples/s]8762 examples [00:09, 954.66 examples/s]8858 examples [00:09, 954.00 examples/s]8954 examples [00:09, 938.38 examples/s]9051 examples [00:09, 946.03 examples/s]9147 examples [00:09, 947.71 examples/s]9242 examples [00:09, 930.30 examples/s]9336 examples [00:09, 919.01 examples/s]9436 examples [00:09, 939.22 examples/s]9535 examples [00:10, 951.81 examples/s]9634 examples [00:10, 962.83 examples/s]9731 examples [00:10, 945.54 examples/s]9826 examples [00:10, 926.48 examples/s]9919 examples [00:10, 907.32 examples/s]10010 examples [00:10, 876.61 examples/s]10108 examples [00:10, 901.54 examples/s]10199 examples [00:10, 874.36 examples/s]10297 examples [00:10, 902.39 examples/s]10395 examples [00:11, 921.82 examples/s]10494 examples [00:11, 940.13 examples/s]10593 examples [00:11, 943.75 examples/s]10690 examples [00:11, 950.80 examples/s]10789 examples [00:11, 960.60 examples/s]10886 examples [00:11, 923.97 examples/s]10979 examples [00:11, 925.47 examples/s]11072 examples [00:11, 914.65 examples/s]11164 examples [00:11, 895.44 examples/s]11254 examples [00:11, 888.33 examples/s]11351 examples [00:12, 908.85 examples/s]11451 examples [00:12, 932.83 examples/s]11547 examples [00:12, 938.52 examples/s]11645 examples [00:12, 950.38 examples/s]11743 examples [00:12, 957.63 examples/s]11843 examples [00:12, 969.56 examples/s]11941 examples [00:12, 971.90 examples/s]12039 examples [00:12, 970.74 examples/s]12137 examples [00:12, 970.91 examples/s]12235 examples [00:12, 971.66 examples/s]12335 examples [00:13, 979.45 examples/s]12434 examples [00:13, 980.68 examples/s]12533 examples [00:13, 979.90 examples/s]12632 examples [00:13, 964.31 examples/s]12729 examples [00:13, 956.04 examples/s]12826 examples [00:13, 957.21 examples/s]12924 examples [00:13, 961.18 examples/s]13023 examples [00:13, 966.09 examples/s]13120 examples [00:13, 952.66 examples/s]13218 examples [00:13, 959.13 examples/s]13314 examples [00:14, 926.50 examples/s]13412 examples [00:14, 939.35 examples/s]13507 examples [00:14, 933.07 examples/s]13601 examples [00:14, 929.40 examples/s]13695 examples [00:14, 915.69 examples/s]13787 examples [00:14, 911.93 examples/s]13879 examples [00:14, 885.55 examples/s]13973 examples [00:14, 900.75 examples/s]14071 examples [00:14, 921.87 examples/s]14168 examples [00:15, 933.81 examples/s]14267 examples [00:15, 947.58 examples/s]14364 examples [00:15, 952.30 examples/s]14460 examples [00:15, 935.09 examples/s]14560 examples [00:15, 952.47 examples/s]14657 examples [00:15, 956.29 examples/s]14756 examples [00:15, 963.59 examples/s]14853 examples [00:15, 955.56 examples/s]14949 examples [00:15, 932.05 examples/s]15043 examples [00:15, 928.63 examples/s]15137 examples [00:16, 923.42 examples/s]15235 examples [00:16, 937.15 examples/s]15332 examples [00:16, 946.07 examples/s]15427 examples [00:16, 947.13 examples/s]15522 examples [00:16, 947.50 examples/s]15617 examples [00:16, 932.00 examples/s]15714 examples [00:16, 940.91 examples/s]15810 examples [00:16, 945.49 examples/s]15905 examples [00:16, 935.77 examples/s]15999 examples [00:16, 927.30 examples/s]16096 examples [00:17, 936.21 examples/s]16192 examples [00:17, 941.79 examples/s]16291 examples [00:17, 954.72 examples/s]16387 examples [00:17, 950.73 examples/s]16484 examples [00:17, 954.94 examples/s]16580 examples [00:17, 955.25 examples/s]16678 examples [00:17, 961.67 examples/s]16777 examples [00:17, 969.56 examples/s]16874 examples [00:17, 968.45 examples/s]16971 examples [00:17, 968.21 examples/s]17071 examples [00:18, 976.59 examples/s]17170 examples [00:18, 980.00 examples/s]17269 examples [00:18, 964.72 examples/s]17366 examples [00:18, 946.39 examples/s]17461 examples [00:18, 940.48 examples/s]17556 examples [00:18, 940.62 examples/s]17653 examples [00:18, 948.63 examples/s]17752 examples [00:18, 958.86 examples/s]17848 examples [00:18, 952.00 examples/s]17947 examples [00:18, 960.64 examples/s]18046 examples [00:19, 969.21 examples/s]18143 examples [00:19, 968.56 examples/s]18240 examples [00:19, 967.20 examples/s]18337 examples [00:19, 963.25 examples/s]18434 examples [00:19, 917.29 examples/s]18527 examples [00:19, 918.65 examples/s]18621 examples [00:19, 924.26 examples/s]18714 examples [00:19, 847.24 examples/s]18802 examples [00:19, 854.48 examples/s]18901 examples [00:20, 890.61 examples/s]19001 examples [00:20, 918.80 examples/s]19095 examples [00:20, 924.57 examples/s]19194 examples [00:20, 941.34 examples/s]19289 examples [00:20, 932.72 examples/s]19384 examples [00:20, 935.31 examples/s]19481 examples [00:20, 942.78 examples/s]19577 examples [00:20, 947.42 examples/s]19676 examples [00:20, 957.93 examples/s]19772 examples [00:20, 949.38 examples/s]19868 examples [00:21, 913.69 examples/s]19966 examples [00:21, 931.90 examples/s]20060 examples [00:21, 887.69 examples/s]20151 examples [00:21, 893.80 examples/s]20241 examples [00:21, 883.33 examples/s]20335 examples [00:21, 897.31 examples/s]20433 examples [00:21, 918.92 examples/s]20529 examples [00:21, 930.78 examples/s]20626 examples [00:21, 941.89 examples/s]20722 examples [00:21, 946.57 examples/s]20821 examples [00:22, 958.12 examples/s]20921 examples [00:22, 968.91 examples/s]21019 examples [00:22, 962.48 examples/s]21116 examples [00:22, 926.98 examples/s]21210 examples [00:22, 862.72 examples/s]21298 examples [00:22, 856.93 examples/s]21385 examples [00:22, 833.22 examples/s]21477 examples [00:22, 856.00 examples/s]21565 examples [00:22, 861.51 examples/s]21657 examples [00:23, 876.97 examples/s]21746 examples [00:23, 873.79 examples/s]21845 examples [00:23, 903.91 examples/s]21939 examples [00:23, 911.94 examples/s]22036 examples [00:23, 927.92 examples/s]22135 examples [00:23, 944.57 examples/s]22233 examples [00:23, 952.29 examples/s]22332 examples [00:23, 960.68 examples/s]22429 examples [00:23, 955.89 examples/s]22525 examples [00:23, 901.38 examples/s]22616 examples [00:24, 897.29 examples/s]22713 examples [00:24, 917.41 examples/s]22806 examples [00:24, 908.97 examples/s]22905 examples [00:24, 929.59 examples/s]23000 examples [00:24, 934.00 examples/s]23099 examples [00:24, 948.36 examples/s]23199 examples [00:24, 960.13 examples/s]23296 examples [00:24, 960.72 examples/s]23393 examples [00:24, 938.96 examples/s]23489 examples [00:24, 942.63 examples/s]23584 examples [00:25, 941.93 examples/s]23682 examples [00:25, 952.17 examples/s]23780 examples [00:25, 957.62 examples/s]23876 examples [00:25, 952.59 examples/s]23974 examples [00:25, 959.40 examples/s]24072 examples [00:25, 964.21 examples/s]24170 examples [00:25, 966.22 examples/s]24267 examples [00:25, 964.10 examples/s]24366 examples [00:25, 969.81 examples/s]24464 examples [00:26, 942.31 examples/s]24559 examples [00:26, 940.67 examples/s]24658 examples [00:26, 952.97 examples/s]24757 examples [00:26, 960.34 examples/s]24856 examples [00:26, 968.09 examples/s]24953 examples [00:26, 963.59 examples/s]25050 examples [00:26, 960.59 examples/s]25149 examples [00:26, 967.59 examples/s]25246 examples [00:26, 912.63 examples/s]25338 examples [00:26, 906.56 examples/s]25430 examples [00:27, 903.17 examples/s]25524 examples [00:27, 911.72 examples/s]25621 examples [00:27, 927.61 examples/s]25715 examples [00:27, 929.41 examples/s]25809 examples [00:27, 931.01 examples/s]25906 examples [00:27, 941.13 examples/s]26003 examples [00:27, 947.49 examples/s]26101 examples [00:27, 954.69 examples/s]26197 examples [00:27, 955.47 examples/s]26293 examples [00:27, 954.30 examples/s]26389 examples [00:28, 913.66 examples/s]26482 examples [00:28, 916.54 examples/s]26576 examples [00:28, 922.93 examples/s]26671 examples [00:28, 929.24 examples/s]26765 examples [00:28, 928.48 examples/s]26858 examples [00:28, 925.43 examples/s]26951 examples [00:28, 912.52 examples/s]27045 examples [00:28, 918.94 examples/s]27140 examples [00:28, 925.03 examples/s]27235 examples [00:28, 931.18 examples/s]27329 examples [00:29, 906.66 examples/s]27424 examples [00:29, 916.92 examples/s]27522 examples [00:29, 932.41 examples/s]27619 examples [00:29, 942.82 examples/s]27715 examples [00:29, 945.98 examples/s]27810 examples [00:29, 929.11 examples/s]27904 examples [00:29, 922.99 examples/s]27997 examples [00:29, 923.46 examples/s]28090 examples [00:29, 918.07 examples/s]28188 examples [00:30, 932.89 examples/s]28282 examples [00:30, 933.62 examples/s]28380 examples [00:30, 945.05 examples/s]28475 examples [00:30, 942.18 examples/s]28573 examples [00:30, 952.51 examples/s]28671 examples [00:30, 958.15 examples/s]28767 examples [00:30, 949.99 examples/s]28864 examples [00:30, 953.83 examples/s]28960 examples [00:30, 928.83 examples/s]29058 examples [00:30, 941.02 examples/s]29156 examples [00:31, 949.97 examples/s]29252 examples [00:31, 915.85 examples/s]29351 examples [00:31, 934.01 examples/s]29446 examples [00:31, 936.03 examples/s]29544 examples [00:31, 946.96 examples/s]29640 examples [00:31, 947.43 examples/s]29736 examples [00:31, 950.20 examples/s]29834 examples [00:31, 956.45 examples/s]29931 examples [00:31, 960.35 examples/s]30028 examples [00:31, 890.87 examples/s]30119 examples [00:32, 879.83 examples/s]30208 examples [00:32, 864.73 examples/s]30297 examples [00:32, 871.13 examples/s]30386 examples [00:32, 874.56 examples/s]30474 examples [00:32, 870.58 examples/s]30562 examples [00:32, 869.66 examples/s]30656 examples [00:32, 888.98 examples/s]30755 examples [00:32, 915.64 examples/s]30852 examples [00:32, 931.03 examples/s]30947 examples [00:32, 935.61 examples/s]31041 examples [00:33, 924.65 examples/s]31134 examples [00:33, 910.87 examples/s]31226 examples [00:33, 886.81 examples/s]31319 examples [00:33, 897.19 examples/s]31414 examples [00:33, 911.53 examples/s]31510 examples [00:33, 925.42 examples/s]31606 examples [00:33, 935.17 examples/s]31700 examples [00:33, 932.58 examples/s]31798 examples [00:33, 944.19 examples/s]31893 examples [00:34, 931.20 examples/s]31987 examples [00:34, 915.40 examples/s]32079 examples [00:34, 912.22 examples/s]32171 examples [00:34, 902.53 examples/s]32262 examples [00:34, 900.71 examples/s]32358 examples [00:34, 915.87 examples/s]32450 examples [00:34, 887.73 examples/s]32547 examples [00:34, 910.86 examples/s]32644 examples [00:34, 918.87 examples/s]32740 examples [00:34, 929.36 examples/s]32838 examples [00:35, 939.15 examples/s]32933 examples [00:35, 928.15 examples/s]33032 examples [00:35, 944.43 examples/s]33131 examples [00:35, 955.07 examples/s]33227 examples [00:35, 947.20 examples/s]33327 examples [00:35, 960.29 examples/s]33426 examples [00:35, 968.96 examples/s]33526 examples [00:35, 976.43 examples/s]33625 examples [00:35, 977.94 examples/s]33724 examples [00:35, 980.20 examples/s]33823 examples [00:36, 978.55 examples/s]33921 examples [00:36, 973.25 examples/s]34019 examples [00:36, 966.08 examples/s]34118 examples [00:36, 971.29 examples/s]34216 examples [00:36, 960.45 examples/s]34314 examples [00:36, 966.03 examples/s]34411 examples [00:36, 960.31 examples/s]34509 examples [00:36, 965.66 examples/s]34606 examples [00:36, 957.65 examples/s]34702 examples [00:36, 954.00 examples/s]34798 examples [00:37, 939.98 examples/s]34893 examples [00:37, 913.75 examples/s]34985 examples [00:37, 892.22 examples/s]35075 examples [00:37, 886.93 examples/s]35174 examples [00:37, 913.03 examples/s]35274 examples [00:37, 935.23 examples/s]35368 examples [00:37, 927.74 examples/s]35463 examples [00:37, 933.03 examples/s]35561 examples [00:37, 945.67 examples/s]35659 examples [00:38, 954.90 examples/s]35756 examples [00:38, 958.01 examples/s]35852 examples [00:38, 944.82 examples/s]35947 examples [00:38, 946.09 examples/s]36043 examples [00:38, 949.39 examples/s]36138 examples [00:38, 932.59 examples/s]36235 examples [00:38, 942.48 examples/s]36330 examples [00:38, 922.29 examples/s]36423 examples [00:38, 919.20 examples/s]36516 examples [00:38, 911.44 examples/s]36616 examples [00:39, 934.67 examples/s]36713 examples [00:39, 944.84 examples/s]36808 examples [00:39, 926.96 examples/s]36901 examples [00:39, 916.45 examples/s]36996 examples [00:39, 926.00 examples/s]37094 examples [00:39, 941.08 examples/s]37189 examples [00:39, 923.68 examples/s]37286 examples [00:39, 937.01 examples/s]37383 examples [00:39, 944.94 examples/s]37478 examples [00:39, 907.38 examples/s]37570 examples [00:40, 907.00 examples/s]37661 examples [00:40, 903.08 examples/s]37755 examples [00:40, 913.59 examples/s]37852 examples [00:40, 927.69 examples/s]37948 examples [00:40, 936.77 examples/s]38044 examples [00:40, 941.42 examples/s]38139 examples [00:40, 931.58 examples/s]38235 examples [00:40, 938.79 examples/s]38334 examples [00:40, 951.81 examples/s]38430 examples [00:40, 943.03 examples/s]38527 examples [00:41, 950.10 examples/s]38623 examples [00:41, 933.40 examples/s]38722 examples [00:41, 949.23 examples/s]38818 examples [00:41, 924.22 examples/s]38917 examples [00:41, 940.98 examples/s]39016 examples [00:41, 953.08 examples/s]39112 examples [00:41, 950.34 examples/s]39210 examples [00:41, 958.14 examples/s]39309 examples [00:41, 965.39 examples/s]39406 examples [00:42, 961.89 examples/s]39505 examples [00:42, 969.09 examples/s]39602 examples [00:42, 961.67 examples/s]39699 examples [00:42, 961.02 examples/s]39796 examples [00:42, 931.33 examples/s]39890 examples [00:42, 916.13 examples/s]39982 examples [00:42, 891.85 examples/s]40072 examples [00:42, 864.49 examples/s]40169 examples [00:42, 892.22 examples/s]40267 examples [00:42, 916.29 examples/s]40366 examples [00:43, 935.76 examples/s]40463 examples [00:43, 945.75 examples/s]40558 examples [00:43, 939.61 examples/s]40656 examples [00:43, 951.18 examples/s]40755 examples [00:43, 961.56 examples/s]40852 examples [00:43, 948.30 examples/s]40947 examples [00:43, 943.71 examples/s]41042 examples [00:43, 942.23 examples/s]41140 examples [00:43, 953.22 examples/s]41236 examples [00:43, 954.20 examples/s]41334 examples [00:44, 960.04 examples/s]41431 examples [00:44, 957.68 examples/s]41527 examples [00:44, 939.47 examples/s]41622 examples [00:44, 930.73 examples/s]41716 examples [00:44, 901.58 examples/s]41807 examples [00:44, 896.43 examples/s]41900 examples [00:44, 904.18 examples/s]41991 examples [00:44, 901.51 examples/s]42089 examples [00:44, 921.24 examples/s]42188 examples [00:45, 939.93 examples/s]42283 examples [00:45, 941.65 examples/s]42378 examples [00:45, 926.51 examples/s]42471 examples [00:45, 913.83 examples/s]42567 examples [00:45, 924.79 examples/s]42665 examples [00:45, 938.81 examples/s]42764 examples [00:45, 953.23 examples/s]42860 examples [00:45, 954.32 examples/s]42956 examples [00:45, 938.60 examples/s]43050 examples [00:45, 928.93 examples/s]43144 examples [00:46, 926.94 examples/s]43238 examples [00:46, 928.29 examples/s]43331 examples [00:46, 916.43 examples/s]43423 examples [00:46, 902.51 examples/s]43514 examples [00:46, 895.07 examples/s]43605 examples [00:46, 899.11 examples/s]43702 examples [00:46, 918.42 examples/s]43794 examples [00:46, 917.92 examples/s]43889 examples [00:46, 924.40 examples/s]43989 examples [00:46, 945.85 examples/s]44085 examples [00:47, 949.63 examples/s]44181 examples [00:47, 936.32 examples/s]44275 examples [00:47, 917.47 examples/s]44367 examples [00:47, 905.67 examples/s]44462 examples [00:47, 917.67 examples/s]44556 examples [00:47, 923.63 examples/s]44649 examples [00:47, 917.47 examples/s]44741 examples [00:47, 903.38 examples/s]44834 examples [00:47, 909.71 examples/s]44931 examples [00:47, 924.46 examples/s]45029 examples [00:48, 940.20 examples/s]45127 examples [00:48, 950.19 examples/s]45223 examples [00:48, 951.75 examples/s]45320 examples [00:48, 954.51 examples/s]45420 examples [00:48, 966.30 examples/s]45520 examples [00:48, 975.47 examples/s]45621 examples [00:48, 984.54 examples/s]45720 examples [00:48, 980.65 examples/s]45820 examples [00:48, 984.98 examples/s]45919 examples [00:48, 986.34 examples/s]46018 examples [00:49, 984.81 examples/s]46117 examples [00:49, 981.68 examples/s]46216 examples [00:49, 961.57 examples/s]46313 examples [00:49, 938.99 examples/s]46408 examples [00:49, 926.84 examples/s]46501 examples [00:49, 914.00 examples/s]46593 examples [00:49, 907.81 examples/s]46686 examples [00:49, 913.66 examples/s]46781 examples [00:49, 923.58 examples/s]46879 examples [00:50, 938.60 examples/s]46977 examples [00:50, 948.56 examples/s]47076 examples [00:50, 958.10 examples/s]47172 examples [00:50, 949.20 examples/s]47268 examples [00:50, 932.22 examples/s]47362 examples [00:50, 915.13 examples/s]47455 examples [00:50, 916.93 examples/s]47554 examples [00:50, 936.20 examples/s]47649 examples [00:50, 938.26 examples/s]47743 examples [00:50, 937.53 examples/s]47837 examples [00:51, 932.52 examples/s]47931 examples [00:51, 926.14 examples/s]48024 examples [00:51, 921.12 examples/s]48117 examples [00:51, 895.86 examples/s]48214 examples [00:51, 915.69 examples/s]48310 examples [00:51, 926.77 examples/s]48408 examples [00:51, 941.30 examples/s]48506 examples [00:51, 950.58 examples/s]48602 examples [00:51, 951.26 examples/s]48701 examples [00:51, 959.89 examples/s]48801 examples [00:52, 969.95 examples/s]48899 examples [00:52, 963.81 examples/s]48997 examples [00:52, 968.33 examples/s]49094 examples [00:52, 960.13 examples/s]49191 examples [00:52, 939.25 examples/s]49286 examples [00:52, 932.25 examples/s]49384 examples [00:52, 943.87 examples/s]49479 examples [00:52, 914.53 examples/s]49571 examples [00:52, 869.42 examples/s]49660 examples [00:53, 873.13 examples/s]49750 examples [00:53, 880.56 examples/s]49840 examples [00:53, 885.87 examples/s]49929 examples [00:53, 871.30 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5483/50000 [00:00<00:00, 54826.52 examples/s] 31%|███▏      | 15645/50000 [00:00<00:00, 63613.44 examples/s] 52%|█████▏    | 26165/50000 [00:00<00:00, 72170.95 examples/s] 74%|███████▍  | 36899/50000 [00:00<00:00, 80037.21 examples/s] 96%|█████████▌| 48121/50000 [00:00<00:00, 87569.47 examples/s]                                                               0 examples [00:00, ? examples/s]80 examples [00:00, 796.62 examples/s]178 examples [00:00, 842.67 examples/s]275 examples [00:00, 875.43 examples/s]372 examples [00:00, 899.66 examples/s]466 examples [00:00, 909.48 examples/s]563 examples [00:00, 924.79 examples/s]660 examples [00:00, 935.31 examples/s]757 examples [00:00, 944.74 examples/s]856 examples [00:00, 955.48 examples/s]949 examples [00:01, 943.40 examples/s]1043 examples [00:01, 941.28 examples/s]1138 examples [00:01, 943.64 examples/s]1236 examples [00:01, 952.85 examples/s]1334 examples [00:01, 958.20 examples/s]1431 examples [00:01, 961.65 examples/s]1533 examples [00:01, 977.27 examples/s]1633 examples [00:01, 982.97 examples/s]1735 examples [00:01, 992.04 examples/s]1835 examples [00:01, 986.67 examples/s]1934 examples [00:02, 964.03 examples/s]2031 examples [00:02, 952.02 examples/s]2127 examples [00:02, 946.85 examples/s]2222 examples [00:02, 944.02 examples/s]2317 examples [00:02, 943.92 examples/s]2412 examples [00:02, 942.12 examples/s]2508 examples [00:02, 946.04 examples/s]2603 examples [00:02, 944.37 examples/s]2698 examples [00:02, 944.07 examples/s]2793 examples [00:02, 939.97 examples/s]2889 examples [00:03, 945.44 examples/s]2989 examples [00:03, 958.64 examples/s]3088 examples [00:03, 967.14 examples/s]3189 examples [00:03, 977.48 examples/s]3292 examples [00:03, 990.74 examples/s]3392 examples [00:03, 993.30 examples/s]3492 examples [00:03, 989.82 examples/s]3594 examples [00:03, 997.44 examples/s]3694 examples [00:03, 992.63 examples/s]3795 examples [00:03, 996.65 examples/s]3895 examples [00:04, 948.64 examples/s]3991 examples [00:04, 917.73 examples/s]4084 examples [00:04, 919.56 examples/s]4178 examples [00:04, 925.16 examples/s]4278 examples [00:04, 944.09 examples/s]4374 examples [00:04, 947.23 examples/s]4473 examples [00:04, 958.56 examples/s]4573 examples [00:04, 970.08 examples/s]4672 examples [00:04, 975.20 examples/s]4771 examples [00:04, 977.67 examples/s]4869 examples [00:05, 972.81 examples/s]4968 examples [00:05, 976.85 examples/s]5069 examples [00:05, 985.12 examples/s]5171 examples [00:05, 993.33 examples/s]5271 examples [00:05, 981.35 examples/s]5370 examples [00:05, 973.49 examples/s]5469 examples [00:05, 977.57 examples/s]5567 examples [00:05, 966.73 examples/s]5664 examples [00:05, 962.16 examples/s]5761 examples [00:06, 907.02 examples/s]5856 examples [00:06, 916.77 examples/s]5951 examples [00:06, 924.48 examples/s]6044 examples [00:06, 923.62 examples/s]6137 examples [00:06, 922.15 examples/s]6230 examples [00:06, 923.55 examples/s]6323 examples [00:06, 902.82 examples/s]6414 examples [00:06, 885.66 examples/s]6513 examples [00:06, 911.60 examples/s]6614 examples [00:06, 937.44 examples/s]6714 examples [00:07, 951.35 examples/s]6813 examples [00:07, 962.31 examples/s]6914 examples [00:07, 973.53 examples/s]7015 examples [00:07, 983.97 examples/s]7116 examples [00:07, 991.32 examples/s]7216 examples [00:07, 981.74 examples/s]7315 examples [00:07, 957.61 examples/s]7412 examples [00:07, 960.28 examples/s]7511 examples [00:07, 968.18 examples/s]7608 examples [00:07, 942.99 examples/s]7703 examples [00:08, 926.34 examples/s]7796 examples [00:08, 889.02 examples/s]7886 examples [00:08, 877.62 examples/s]7976 examples [00:08, 884.19 examples/s]8065 examples [00:08, 879.06 examples/s]8156 examples [00:08, 886.11 examples/s]8245 examples [00:08, 845.15 examples/s]8347 examples [00:08, 890.12 examples/s]8439 examples [00:08, 897.07 examples/s]8534 examples [00:09, 911.61 examples/s]8634 examples [00:09, 934.53 examples/s]8728 examples [00:09, 916.72 examples/s]8821 examples [00:09, 909.37 examples/s]8917 examples [00:09, 922.12 examples/s]9015 examples [00:09, 937.48 examples/s]9113 examples [00:09, 948.14 examples/s]9211 examples [00:09, 955.74 examples/s]9312 examples [00:09, 969.65 examples/s]9411 examples [00:09, 975.10 examples/s]9509 examples [00:10, 972.07 examples/s]9609 examples [00:10, 977.70 examples/s]9707 examples [00:10, 969.45 examples/s]9805 examples [00:10, 966.46 examples/s]9902 examples [00:10, 960.66 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteCI09LC/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteCI09LC/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5acbc9ca60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5acbc9ca60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5acbc9ca60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5b17609048>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5a51f5aeb8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5acbc9ca60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5acbc9ca60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5acbc9ca60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ab3b4d278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ab3b4d860>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.77 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 20.37 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.58 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.58 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.83 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.83 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.41 file/s]2020-07-30 18:11:14.748225: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-30 18:11:14.752630: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-07-30 18:11:14.753356: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5572fba69580 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-30 18:11:14.753378: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3792896/9912422 [00:00<00:00, 37392135.57it/s]9920512it [00:00, 15857617.56it/s]                             
0it [00:00, ?it/s]32768it [00:00, 535999.91it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 137557.53it/s]1654784it [00:00, 9752223.41it/s]                          
0it [00:00, ?it/s]8192it [00:00, 183628.99it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

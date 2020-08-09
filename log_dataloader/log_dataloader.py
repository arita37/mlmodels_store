
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f4975892510> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f4975892510> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f49e0542488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f49e0542488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f49ff760ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f49ff760ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f498d87d158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f498d87d158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f498d87d158> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 143141.42it/s] 70%|███████   | 6946816/9912422 [00:00<00:14, 204306.82it/s]9920512it [00:00, 41048995.98it/s]                           
0it [00:00, ?it/s]32768it [00:00, 564574.77it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 951813.59it/s]1654784it [00:00, 12222389.39it/s]                          
0it [00:00, ?it/s]8192it [00:00, 235606.94it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4974acd6d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f498bc65978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f498d872d90> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f498d872d90> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f498d872d90> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:39,  1.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:39,  1.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:39,  1.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:38,  1.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:37,  1.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:37,  1.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:36,  1.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:08,  2.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:08,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:07,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:07,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:06,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:06,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:05,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:05,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:05,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:45,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:45,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:44,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:44,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:44,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:43,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:43,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:30,  4.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:30,  4.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:30,  4.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:29,  4.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:29,  4.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:29,  4.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:29,  4.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:20,  6.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:19,  6.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:14,  8.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:14,  8.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:14,  8.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:14,  8.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:14,  8.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:13,  8.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:13,  8.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:13,  8.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 11.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 11.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:09, 11.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:09, 11.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:09, 11.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 15.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 26.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 26.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 26.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 26.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 33.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 33.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 33.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 33.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 33.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 33.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 33.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 40.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 40.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 40.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 40.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 40.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 40.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 40.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 40.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 40.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 46.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 46.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 46.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 46.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 46.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 46.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 46.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 46.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 46.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 52.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 54.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 54.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 57.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 57.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 57.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 57.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 57.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 57.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 57.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 57.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 57.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 62.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 70.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 70.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 70.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 70.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 70.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 70.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 70.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 68.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 68.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 68.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.84s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.84s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.84s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.97 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.09s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.84s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 68.97 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.09s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.09s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.82 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.09s/ url]
0 examples [00:00, ? examples/s]2020-08-09 18:07:26.479105: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-09 18:07:26.497187: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-08-09 18:07:26.497467: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5580492f48f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-09 18:07:26.497498: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.87 examples/s]80 examples [00:00,  4.10 examples/s]168 examples [00:00,  5.85 examples/s]253 examples [00:00,  8.33 examples/s]347 examples [00:00, 11.85 examples/s]450 examples [00:00, 16.84 examples/s]552 examples [00:00, 23.89 examples/s]650 examples [00:01, 33.78 examples/s]749 examples [00:01, 47.55 examples/s]850 examples [00:01, 66.57 examples/s]945 examples [00:01, 92.30 examples/s]1047 examples [00:01, 126.91 examples/s]1150 examples [00:01, 172.19 examples/s]1249 examples [00:01, 227.43 examples/s]1345 examples [00:01, 292.79 examples/s]1439 examples [00:01, 364.33 examples/s]1531 examples [00:01, 437.73 examples/s]1620 examples [00:02, 509.40 examples/s]1709 examples [00:02, 584.36 examples/s]1800 examples [00:02, 654.61 examples/s]1896 examples [00:02, 723.24 examples/s]1999 examples [00:02, 792.67 examples/s]2097 examples [00:02, 840.40 examples/s]2193 examples [00:02, 866.21 examples/s]2288 examples [00:02, 868.44 examples/s]2381 examples [00:02, 873.71 examples/s]2473 examples [00:03, 870.90 examples/s]2570 examples [00:03, 895.36 examples/s]2662 examples [00:03, 900.07 examples/s]2754 examples [00:03, 877.82 examples/s]2843 examples [00:03, 855.43 examples/s]2930 examples [00:03, 830.26 examples/s]3014 examples [00:03, 808.31 examples/s]3109 examples [00:03, 845.37 examples/s]3203 examples [00:03, 870.68 examples/s]3291 examples [00:03, 868.53 examples/s]3381 examples [00:04, 876.11 examples/s]3472 examples [00:04, 885.99 examples/s]3568 examples [00:04, 906.35 examples/s]3669 examples [00:04, 932.72 examples/s]3772 examples [00:04, 957.78 examples/s]3876 examples [00:04, 979.20 examples/s]3975 examples [00:04, 979.11 examples/s]4076 examples [00:04, 987.02 examples/s]4175 examples [00:04, 981.55 examples/s]4274 examples [00:04, 982.78 examples/s]4378 examples [00:05, 998.05 examples/s]4478 examples [00:05, 955.33 examples/s]4575 examples [00:05, 949.27 examples/s]4671 examples [00:05, 928.80 examples/s]4765 examples [00:05, 893.57 examples/s]4855 examples [00:05, 867.59 examples/s]4952 examples [00:05, 893.91 examples/s]5050 examples [00:05, 916.81 examples/s]5152 examples [00:05, 944.96 examples/s]5248 examples [00:06, 936.34 examples/s]5343 examples [00:06, 925.46 examples/s]5436 examples [00:06, 877.08 examples/s]5532 examples [00:06, 900.30 examples/s]5631 examples [00:06, 923.26 examples/s]5730 examples [00:06, 941.45 examples/s]5825 examples [00:06, 927.49 examples/s]5919 examples [00:06, 911.11 examples/s]6011 examples [00:06, 878.60 examples/s]6103 examples [00:06, 888.07 examples/s]6196 examples [00:07, 900.14 examples/s]6287 examples [00:07, 883.94 examples/s]6376 examples [00:07, 850.93 examples/s]6463 examples [00:07, 854.62 examples/s]6557 examples [00:07, 877.30 examples/s]6659 examples [00:07, 914.43 examples/s]6757 examples [00:07, 933.03 examples/s]6860 examples [00:07, 958.75 examples/s]6957 examples [00:07, 922.90 examples/s]7050 examples [00:08, 898.36 examples/s]7141 examples [00:08, 895.77 examples/s]7232 examples [00:08, 893.44 examples/s]7326 examples [00:08, 906.15 examples/s]7417 examples [00:08, 896.17 examples/s]7507 examples [00:08, 888.42 examples/s]7597 examples [00:08, 864.36 examples/s]7684 examples [00:08, 856.09 examples/s]7773 examples [00:08, 863.98 examples/s]7860 examples [00:08, 854.70 examples/s]7946 examples [00:09, 836.86 examples/s]8030 examples [00:09, 829.32 examples/s]8126 examples [00:09, 864.45 examples/s]8230 examples [00:09, 909.05 examples/s]8334 examples [00:09, 938.72 examples/s]8431 examples [00:09, 946.30 examples/s]8537 examples [00:09, 974.34 examples/s]8636 examples [00:09, 950.79 examples/s]8732 examples [00:09, 950.92 examples/s]8828 examples [00:10, 950.05 examples/s]8927 examples [00:10, 959.26 examples/s]9027 examples [00:10, 970.41 examples/s]9127 examples [00:10, 978.55 examples/s]9235 examples [00:10, 1004.57 examples/s]9338 examples [00:10, 1010.54 examples/s]9440 examples [00:10, 1010.57 examples/s]9542 examples [00:10, 1000.42 examples/s]9644 examples [00:10, 1006.13 examples/s]9745 examples [00:10, 1001.01 examples/s]9846 examples [00:11, 983.97 examples/s] 9945 examples [00:11, 948.88 examples/s]10041 examples [00:11, 872.46 examples/s]10137 examples [00:11, 896.33 examples/s]10230 examples [00:11, 904.22 examples/s]10322 examples [00:11, 907.52 examples/s]10414 examples [00:11, 887.64 examples/s]10507 examples [00:11, 899.09 examples/s]10598 examples [00:11, 886.45 examples/s]10687 examples [00:11, 864.26 examples/s]10774 examples [00:12, 853.50 examples/s]10860 examples [00:12, 828.68 examples/s]10945 examples [00:12, 832.30 examples/s]11040 examples [00:12, 862.25 examples/s]11134 examples [00:12, 883.71 examples/s]11223 examples [00:12, 863.11 examples/s]11310 examples [00:12, 847.06 examples/s]11397 examples [00:12, 853.65 examples/s]11486 examples [00:12, 862.05 examples/s]11577 examples [00:13, 874.15 examples/s]11665 examples [00:13, 875.67 examples/s]11763 examples [00:13, 902.20 examples/s]11854 examples [00:13, 902.43 examples/s]11960 examples [00:13, 943.92 examples/s]12066 examples [00:13, 975.35 examples/s]12165 examples [00:13, 972.44 examples/s]12267 examples [00:13, 983.96 examples/s]12370 examples [00:13, 993.85 examples/s]12478 examples [00:13, 1017.00 examples/s]12584 examples [00:14, 1029.40 examples/s]12688 examples [00:14, 1025.91 examples/s]12791 examples [00:14, 996.50 examples/s] 12891 examples [00:14, 974.82 examples/s]12996 examples [00:14, 994.65 examples/s]13097 examples [00:14, 997.87 examples/s]13198 examples [00:14, 975.24 examples/s]13296 examples [00:14, 964.85 examples/s]13395 examples [00:14, 969.29 examples/s]13499 examples [00:14, 987.96 examples/s]13602 examples [00:15, 997.17 examples/s]13705 examples [00:15, 1004.55 examples/s]13806 examples [00:15, 999.56 examples/s] 13907 examples [00:15, 990.53 examples/s]14007 examples [00:15, 987.16 examples/s]14112 examples [00:15, 1004.70 examples/s]14215 examples [00:15, 1011.84 examples/s]14317 examples [00:15, 1013.16 examples/s]14419 examples [00:15, 1015.15 examples/s]14526 examples [00:15, 1030.61 examples/s]14630 examples [00:16, 1032.83 examples/s]14734 examples [00:16, 1017.36 examples/s]14836 examples [00:16, 1014.44 examples/s]14938 examples [00:16, 1003.42 examples/s]15046 examples [00:16, 1023.08 examples/s]15150 examples [00:16, 1027.36 examples/s]15256 examples [00:16, 1036.90 examples/s]15360 examples [00:16, 1011.44 examples/s]15462 examples [00:16, 1000.71 examples/s]15563 examples [00:17, 968.68 examples/s] 15661 examples [00:17, 931.79 examples/s]15755 examples [00:17, 909.96 examples/s]15857 examples [00:17, 939.87 examples/s]15961 examples [00:17, 963.39 examples/s]16064 examples [00:17, 981.78 examples/s]16166 examples [00:17, 991.48 examples/s]16269 examples [00:17, 1002.33 examples/s]16370 examples [00:17, 1000.28 examples/s]16473 examples [00:17, 1006.31 examples/s]16577 examples [00:18, 1015.18 examples/s]16683 examples [00:18, 1024.30 examples/s]16786 examples [00:18, 1000.18 examples/s]16887 examples [00:18, 980.44 examples/s] 16986 examples [00:18, 949.57 examples/s]17088 examples [00:18, 967.65 examples/s]17190 examples [00:18, 980.80 examples/s]17290 examples [00:18, 984.16 examples/s]17390 examples [00:18, 984.55 examples/s]17495 examples [00:18, 1002.66 examples/s]17598 examples [00:19, 1008.54 examples/s]17699 examples [00:19, 1000.31 examples/s]17805 examples [00:19, 1016.98 examples/s]17907 examples [00:19, 999.69 examples/s] 18012 examples [00:19, 1013.82 examples/s]18114 examples [00:19, 1014.75 examples/s]18220 examples [00:19, 1025.30 examples/s]18330 examples [00:19, 1045.00 examples/s]18435 examples [00:19, 1024.34 examples/s]18538 examples [00:20, 978.63 examples/s] 18637 examples [00:20, 961.77 examples/s]18734 examples [00:20, 950.03 examples/s]18830 examples [00:20, 933.75 examples/s]18924 examples [00:20, 904.99 examples/s]19024 examples [00:20, 929.28 examples/s]19122 examples [00:20, 943.41 examples/s]19219 examples [00:20, 948.40 examples/s]19316 examples [00:20, 952.76 examples/s]19413 examples [00:20, 955.52 examples/s]19513 examples [00:21, 965.74 examples/s]19613 examples [00:21, 975.56 examples/s]19711 examples [00:21, 974.35 examples/s]19810 examples [00:21, 977.39 examples/s]19908 examples [00:21, 974.77 examples/s]20006 examples [00:21, 927.65 examples/s]20102 examples [00:21, 935.02 examples/s]20199 examples [00:21, 945.09 examples/s]20298 examples [00:21, 956.45 examples/s]20394 examples [00:21, 951.72 examples/s]20490 examples [00:22, 946.48 examples/s]20594 examples [00:22, 972.19 examples/s]20694 examples [00:22, 979.95 examples/s]20798 examples [00:22, 995.84 examples/s]20898 examples [00:22, 980.77 examples/s]21000 examples [00:22, 990.95 examples/s]21104 examples [00:22, 1003.60 examples/s]21209 examples [00:22, 1008.87 examples/s]21310 examples [00:22, 990.57 examples/s] 21410 examples [00:22, 991.15 examples/s]21510 examples [00:23, 991.76 examples/s]21610 examples [00:23, 969.76 examples/s]21708 examples [00:23, 945.32 examples/s]21803 examples [00:23, 900.55 examples/s]21894 examples [00:23, 893.93 examples/s]21986 examples [00:23, 899.11 examples/s]22078 examples [00:23, 904.86 examples/s]22169 examples [00:23, 905.32 examples/s]22265 examples [00:23, 919.11 examples/s]22360 examples [00:24, 924.15 examples/s]22453 examples [00:24, 911.36 examples/s]22545 examples [00:24, 892.55 examples/s]22635 examples [00:24, 886.57 examples/s]22724 examples [00:24, 882.68 examples/s]22816 examples [00:24, 891.91 examples/s]22921 examples [00:24, 933.34 examples/s]23021 examples [00:24, 950.78 examples/s]23121 examples [00:24, 963.77 examples/s]23221 examples [00:24, 972.75 examples/s]23327 examples [00:25, 996.16 examples/s]23432 examples [00:25, 1009.18 examples/s]23534 examples [00:25, 1010.76 examples/s]23638 examples [00:25, 1017.07 examples/s]23741 examples [00:25, 1018.92 examples/s]23843 examples [00:25, 981.21 examples/s] 23942 examples [00:25, 983.29 examples/s]24041 examples [00:25, 961.68 examples/s]24138 examples [00:25, 922.17 examples/s]24231 examples [00:26, 912.04 examples/s]24323 examples [00:26, 896.33 examples/s]24413 examples [00:26, 873.83 examples/s]24501 examples [00:26, 854.92 examples/s]24587 examples [00:26, 843.41 examples/s]24672 examples [00:26, 840.59 examples/s]24760 examples [00:26, 851.42 examples/s]24862 examples [00:26, 895.59 examples/s]24963 examples [00:26, 925.16 examples/s]25066 examples [00:26, 953.43 examples/s]25163 examples [00:27, 933.33 examples/s]25265 examples [00:27, 955.84 examples/s]25365 examples [00:27, 967.34 examples/s]25469 examples [00:27, 986.96 examples/s]25571 examples [00:27, 996.62 examples/s]25673 examples [00:27, 1001.90 examples/s]25775 examples [00:27, 1006.52 examples/s]25879 examples [00:27, 1016.33 examples/s]25981 examples [00:27, 1015.98 examples/s]26083 examples [00:27, 985.92 examples/s] 26182 examples [00:28, 971.31 examples/s]26280 examples [00:28, 957.56 examples/s]26376 examples [00:28, 949.55 examples/s]26475 examples [00:28, 961.07 examples/s]26581 examples [00:28, 988.39 examples/s]26681 examples [00:28, 979.75 examples/s]26780 examples [00:28, 976.68 examples/s]26883 examples [00:28, 989.79 examples/s]26987 examples [00:28, 1002.14 examples/s]27092 examples [00:28, 1016.01 examples/s]27194 examples [00:29, 1013.04 examples/s]27298 examples [00:29, 1019.12 examples/s]27402 examples [00:29, 1023.55 examples/s]27505 examples [00:29, 1018.06 examples/s]27612 examples [00:29, 1030.94 examples/s]27716 examples [00:29, 1019.00 examples/s]27819 examples [00:29, 1022.13 examples/s]27922 examples [00:29, 1020.24 examples/s]28028 examples [00:29, 1030.93 examples/s]28132 examples [00:30, 1006.90 examples/s]28233 examples [00:30, 965.46 examples/s] 28331 examples [00:30, 939.92 examples/s]28426 examples [00:30, 919.91 examples/s]28525 examples [00:30, 937.74 examples/s]28627 examples [00:30, 960.00 examples/s]28727 examples [00:30, 969.93 examples/s]28833 examples [00:30, 993.12 examples/s]28934 examples [00:30, 996.19 examples/s]29037 examples [00:30, 1005.42 examples/s]29138 examples [00:31, 1006.56 examples/s]29239 examples [00:31, 1004.16 examples/s]29340 examples [00:31, 994.86 examples/s] 29440 examples [00:31, 967.46 examples/s]29537 examples [00:31, 923.16 examples/s]29630 examples [00:31, 906.12 examples/s]29722 examples [00:31, 900.77 examples/s]29813 examples [00:31, 902.27 examples/s]29912 examples [00:31, 926.39 examples/s]30005 examples [00:31, 898.50 examples/s]30108 examples [00:32, 932.19 examples/s]30213 examples [00:32, 962.29 examples/s]30318 examples [00:32, 985.47 examples/s]30420 examples [00:32, 992.35 examples/s]30520 examples [00:32, 973.73 examples/s]30618 examples [00:32, 946.24 examples/s]30714 examples [00:32, 931.45 examples/s]30810 examples [00:32, 937.38 examples/s]30914 examples [00:32, 963.95 examples/s]31014 examples [00:33, 972.40 examples/s]31115 examples [00:33, 982.40 examples/s]31214 examples [00:33, 958.56 examples/s]31311 examples [00:33, 939.55 examples/s]31407 examples [00:33, 942.77 examples/s]31502 examples [00:33, 937.06 examples/s]31603 examples [00:33, 955.99 examples/s]31699 examples [00:33, 955.67 examples/s]31795 examples [00:33, 934.03 examples/s]31889 examples [00:33, 933.89 examples/s]31983 examples [00:34, 928.81 examples/s]32076 examples [00:34, 913.93 examples/s]32168 examples [00:34, 888.19 examples/s]32260 examples [00:34, 896.16 examples/s]32353 examples [00:34, 904.05 examples/s]32447 examples [00:34, 913.69 examples/s]32541 examples [00:34, 920.17 examples/s]32640 examples [00:34, 939.95 examples/s]32742 examples [00:34, 960.49 examples/s]32839 examples [00:34, 963.17 examples/s]32936 examples [00:35, 924.08 examples/s]33029 examples [00:35, 898.74 examples/s]33120 examples [00:35, 892.64 examples/s]33210 examples [00:35, 881.09 examples/s]33300 examples [00:35, 885.72 examples/s]33393 examples [00:35, 897.96 examples/s]33488 examples [00:35, 912.53 examples/s]33587 examples [00:35, 933.15 examples/s]33681 examples [00:35, 918.20 examples/s]33774 examples [00:36, 918.23 examples/s]33871 examples [00:36, 930.54 examples/s]33965 examples [00:36, 933.36 examples/s]34059 examples [00:36, 904.43 examples/s]34150 examples [00:36, 901.24 examples/s]34244 examples [00:36, 911.32 examples/s]34336 examples [00:36, 908.38 examples/s]34427 examples [00:36, 889.90 examples/s]34517 examples [00:36, 886.77 examples/s]34606 examples [00:36, 875.90 examples/s]34698 examples [00:37, 888.10 examples/s]34794 examples [00:37, 906.65 examples/s]34885 examples [00:37, 891.39 examples/s]34979 examples [00:37, 903.04 examples/s]35070 examples [00:37, 898.54 examples/s]35165 examples [00:37, 913.34 examples/s]35268 examples [00:37, 944.60 examples/s]35369 examples [00:37, 961.60 examples/s]35466 examples [00:37, 940.03 examples/s]35561 examples [00:37, 942.41 examples/s]35663 examples [00:38, 960.62 examples/s]35764 examples [00:38, 974.22 examples/s]35864 examples [00:38, 979.21 examples/s]35968 examples [00:38, 994.18 examples/s]36070 examples [00:38, 1000.99 examples/s]36173 examples [00:38, 1007.86 examples/s]36279 examples [00:38, 1021.77 examples/s]36382 examples [00:38, 1013.82 examples/s]36484 examples [00:38, 988.64 examples/s] 36584 examples [00:38, 973.27 examples/s]36682 examples [00:39, 973.62 examples/s]36788 examples [00:39, 995.74 examples/s]36888 examples [00:39, 972.28 examples/s]36986 examples [00:39, 929.41 examples/s]37080 examples [00:39, 873.72 examples/s]37169 examples [00:39, 839.88 examples/s]37257 examples [00:39, 850.11 examples/s]37358 examples [00:39, 891.10 examples/s]37458 examples [00:39, 919.13 examples/s]37558 examples [00:40, 941.36 examples/s]37663 examples [00:40, 969.89 examples/s]37763 examples [00:40, 977.09 examples/s]37864 examples [00:40, 984.30 examples/s]37968 examples [00:40, 998.57 examples/s]38071 examples [00:40, 1007.70 examples/s]38176 examples [00:40, 1017.50 examples/s]38278 examples [00:40, 999.95 examples/s] 38379 examples [00:40, 920.27 examples/s]38473 examples [00:41, 900.74 examples/s]38565 examples [00:41, 886.98 examples/s]38655 examples [00:41, 877.08 examples/s]38744 examples [00:41, 875.39 examples/s]38834 examples [00:41, 880.58 examples/s]38923 examples [00:41, 870.89 examples/s]39018 examples [00:41, 893.20 examples/s]39113 examples [00:41, 908.02 examples/s]39205 examples [00:41, 908.04 examples/s]39301 examples [00:41, 921.42 examples/s]39394 examples [00:42, 914.54 examples/s]39486 examples [00:42, 910.49 examples/s]39578 examples [00:42, 892.25 examples/s]39672 examples [00:42, 905.82 examples/s]39765 examples [00:42, 910.78 examples/s]39866 examples [00:42, 935.81 examples/s]39967 examples [00:42, 954.21 examples/s]40063 examples [00:42, 911.42 examples/s]40155 examples [00:42, 906.40 examples/s]40247 examples [00:42, 893.75 examples/s]40337 examples [00:43, 889.42 examples/s]40427 examples [00:43, 892.13 examples/s]40517 examples [00:43, 887.93 examples/s]40608 examples [00:43, 893.17 examples/s]40706 examples [00:43, 916.56 examples/s]40809 examples [00:43, 947.32 examples/s]40907 examples [00:43, 956.44 examples/s]41003 examples [00:43, 945.91 examples/s]41098 examples [00:43, 921.42 examples/s]41191 examples [00:43, 907.90 examples/s]41283 examples [00:44, 907.71 examples/s]41379 examples [00:44, 921.62 examples/s]41478 examples [00:44, 939.04 examples/s]41580 examples [00:44, 959.85 examples/s]41684 examples [00:44, 980.07 examples/s]41788 examples [00:44, 995.63 examples/s]41888 examples [00:44, 988.64 examples/s]41988 examples [00:44, 982.60 examples/s]42090 examples [00:44, 993.10 examples/s]42190 examples [00:45, 988.83 examples/s]42289 examples [00:45, 980.38 examples/s]42388 examples [00:45, 981.19 examples/s]42487 examples [00:45, 976.15 examples/s]42585 examples [00:45, 965.09 examples/s]42683 examples [00:45, 967.05 examples/s]42781 examples [00:45, 969.75 examples/s]42879 examples [00:45, 958.20 examples/s]42975 examples [00:45, 897.14 examples/s]43066 examples [00:45, 874.69 examples/s]43155 examples [00:46, 866.32 examples/s]43243 examples [00:46, 865.72 examples/s]43337 examples [00:46, 885.42 examples/s]43429 examples [00:46, 893.86 examples/s]43519 examples [00:46, 883.55 examples/s]43615 examples [00:46, 903.18 examples/s]43710 examples [00:46, 915.67 examples/s]43804 examples [00:46, 920.40 examples/s]43900 examples [00:46, 930.82 examples/s]43994 examples [00:46, 921.55 examples/s]44089 examples [00:47, 929.00 examples/s]44184 examples [00:47, 934.15 examples/s]44278 examples [00:47, 928.00 examples/s]44371 examples [00:47, 902.52 examples/s]44462 examples [00:47, 883.94 examples/s]44551 examples [00:47, 881.87 examples/s]44643 examples [00:47, 890.95 examples/s]44741 examples [00:47, 915.54 examples/s]44839 examples [00:47, 933.45 examples/s]44934 examples [00:48, 935.60 examples/s]45028 examples [00:48, 911.64 examples/s]45120 examples [00:48, 912.44 examples/s]45220 examples [00:48, 936.25 examples/s]45314 examples [00:48, 919.99 examples/s]45407 examples [00:48, 909.08 examples/s]45499 examples [00:48, 898.46 examples/s]45605 examples [00:48, 939.38 examples/s]45707 examples [00:48, 960.30 examples/s]45804 examples [00:48, 906.21 examples/s]45896 examples [00:49, 900.15 examples/s]45987 examples [00:49, 902.76 examples/s]46078 examples [00:49, 892.75 examples/s]46168 examples [00:49, 887.31 examples/s]46267 examples [00:49, 914.80 examples/s]46371 examples [00:49, 948.51 examples/s]46474 examples [00:49, 970.29 examples/s]46580 examples [00:49, 994.20 examples/s]46680 examples [00:49, 994.85 examples/s]46780 examples [00:49, 994.72 examples/s]46880 examples [00:50, 992.96 examples/s]46980 examples [00:50, 978.04 examples/s]47078 examples [00:50, 974.87 examples/s]47177 examples [00:50, 977.78 examples/s]47275 examples [00:50, 934.45 examples/s]47369 examples [00:50, 919.81 examples/s]47462 examples [00:50, 908.44 examples/s]47554 examples [00:50, 905.36 examples/s]47645 examples [00:50, 899.87 examples/s]47745 examples [00:51, 927.51 examples/s]47844 examples [00:51, 943.65 examples/s]47945 examples [00:51, 961.30 examples/s]48054 examples [00:51, 996.42 examples/s]48158 examples [00:51, 1006.08 examples/s]48260 examples [00:51, 985.19 examples/s] 48359 examples [00:51, 968.81 examples/s]48457 examples [00:51, 956.23 examples/s]48553 examples [00:51, 930.59 examples/s]48647 examples [00:51, 882.68 examples/s]48737 examples [00:52, 873.78 examples/s]48825 examples [00:52, 867.97 examples/s]48913 examples [00:52, 871.05 examples/s]49013 examples [00:52, 906.10 examples/s]49113 examples [00:52, 931.99 examples/s]49211 examples [00:52, 943.81 examples/s]49306 examples [00:52, 926.72 examples/s]49400 examples [00:52, 910.78 examples/s]49492 examples [00:52, 894.72 examples/s]49582 examples [00:52, 891.94 examples/s]49678 examples [00:53, 909.84 examples/s]49779 examples [00:53, 936.29 examples/s]49882 examples [00:53, 960.80 examples/s]49982 examples [00:53, 971.51 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7420/50000 [00:00<00:00, 74194.41 examples/s] 38%|███▊      | 18976/50000 [00:00<00:00, 83120.17 examples/s] 63%|██████▎   | 31392/50000 [00:00<00:00, 92269.21 examples/s] 87%|████████▋ | 43726/50000 [00:00<00:00, 99812.00 examples/s]                                                               0 examples [00:00, ? examples/s]78 examples [00:00, 774.30 examples/s]177 examples [00:00, 828.22 examples/s]282 examples [00:00, 883.25 examples/s]385 examples [00:00, 921.08 examples/s]483 examples [00:00, 935.39 examples/s]567 examples [00:00, 903.38 examples/s]660 examples [00:00, 906.81 examples/s]757 examples [00:00, 923.67 examples/s]853 examples [00:00, 933.67 examples/s]949 examples [00:01, 938.99 examples/s]1053 examples [00:01, 966.90 examples/s]1149 examples [00:01, 961.74 examples/s]1247 examples [00:01, 966.68 examples/s]1344 examples [00:01, 931.62 examples/s]1438 examples [00:01, 919.11 examples/s]1530 examples [00:01, 906.10 examples/s]1621 examples [00:01, 882.82 examples/s]1719 examples [00:01, 908.16 examples/s]1815 examples [00:01, 922.24 examples/s]1915 examples [00:02, 943.74 examples/s]2016 examples [00:02, 961.41 examples/s]2113 examples [00:02, 956.43 examples/s]2209 examples [00:02, 939.59 examples/s]2304 examples [00:02, 929.59 examples/s]2398 examples [00:02, 928.19 examples/s]2491 examples [00:02, 923.20 examples/s]2593 examples [00:02, 949.42 examples/s]2698 examples [00:02, 973.78 examples/s]2799 examples [00:02, 982.99 examples/s]2905 examples [00:03, 1002.35 examples/s]3006 examples [00:03, 969.65 examples/s] 3108 examples [00:03, 984.04 examples/s]3212 examples [00:03, 1000.10 examples/s]3318 examples [00:03, 1014.78 examples/s]3420 examples [00:03, 1002.24 examples/s]3525 examples [00:03, 1015.78 examples/s]3627 examples [00:03, 993.00 examples/s] 3735 examples [00:03, 1015.57 examples/s]3837 examples [00:04, 966.89 examples/s] 3935 examples [00:04, 933.06 examples/s]4030 examples [00:04, 931.42 examples/s]4124 examples [00:04, 891.25 examples/s]4214 examples [00:04, 865.82 examples/s]4302 examples [00:04, 868.91 examples/s]4397 examples [00:04, 891.25 examples/s]4494 examples [00:04, 909.41 examples/s]4595 examples [00:04, 935.44 examples/s]4700 examples [00:04, 966.49 examples/s]4803 examples [00:05, 983.86 examples/s]4902 examples [00:05, 983.58 examples/s]5001 examples [00:05, 984.71 examples/s]5100 examples [00:05, 954.37 examples/s]5199 examples [00:05, 963.13 examples/s]5296 examples [00:05, 945.60 examples/s]5391 examples [00:05, 921.93 examples/s]5484 examples [00:05, 904.19 examples/s]5575 examples [00:05, 887.62 examples/s]5666 examples [00:05, 892.48 examples/s]5756 examples [00:06, 889.66 examples/s]5856 examples [00:06, 919.74 examples/s]5953 examples [00:06, 933.26 examples/s]6057 examples [00:06, 960.80 examples/s]6154 examples [00:06, 959.85 examples/s]6251 examples [00:06, 960.85 examples/s]6357 examples [00:06, 987.69 examples/s]6465 examples [00:06, 1011.86 examples/s]6571 examples [00:06, 1023.48 examples/s]6674 examples [00:07, 1019.17 examples/s]6777 examples [00:07, 1014.56 examples/s]6879 examples [00:07, 993.74 examples/s] 6979 examples [00:07, 982.97 examples/s]7082 examples [00:07, 995.28 examples/s]7182 examples [00:07, 988.20 examples/s]7284 examples [00:07, 995.59 examples/s]7384 examples [00:07, 988.53 examples/s]7483 examples [00:07, 961.10 examples/s]7580 examples [00:07, 961.11 examples/s]7677 examples [00:08, 958.24 examples/s]7773 examples [00:08, 939.65 examples/s]7880 examples [00:08, 973.64 examples/s]7978 examples [00:08, 928.14 examples/s]8072 examples [00:08, 916.91 examples/s]8171 examples [00:08, 936.83 examples/s]8266 examples [00:08, 929.96 examples/s]8361 examples [00:08, 934.63 examples/s]8455 examples [00:08, 926.74 examples/s]8548 examples [00:08, 904.97 examples/s]8639 examples [00:09, 880.35 examples/s]8728 examples [00:09, 831.31 examples/s]8813 examples [00:09, 836.61 examples/s]8900 examples [00:09, 845.42 examples/s]8989 examples [00:09, 858.19 examples/s]9076 examples [00:09, 851.44 examples/s]9162 examples [00:09, 844.16 examples/s]9247 examples [00:09, 844.54 examples/s]9332 examples [00:09, 833.85 examples/s]9416 examples [00:10, 828.46 examples/s]9516 examples [00:10, 872.27 examples/s]9609 examples [00:10, 887.95 examples/s]9699 examples [00:10, 886.79 examples/s]9789 examples [00:10, 833.44 examples/s]9874 examples [00:10, 817.21 examples/s]9957 examples [00:10, 817.41 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 98%|█████████▊| 9772/10000 [00:00<00:00, 97716.37 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteN0YU98/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteN0YU98/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f498d87d158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f498d87d158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f498d87d158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f49d925b0b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f494c8e6f28>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f498d87d158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f498d87d158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f498d87d158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f498bc65278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f498bc650b8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 14.45 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 26.84 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.51 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.51 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.97 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.97 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.54 file/s]2020-08-09 18:08:38.678050: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-09 18:08:38.683049: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-08-09 18:08:38.683213: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557796a8e960 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-09 18:08:38.683230: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 143659.71it/s] 76%|███████▌  | 7536640/9912422 [00:00<00:11, 205060.27it/s]9920512it [00:00, 43451373.06it/s]                           
0it [00:00, ?it/s]32768it [00:00, 627099.79it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 489259.82it/s]1654784it [00:00, 12283149.19it/s]                         
0it [00:00, ?it/s]8192it [00:00, 202206.51it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

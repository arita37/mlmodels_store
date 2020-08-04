
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f4371e72ae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f4371e72ae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f43dcae2510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f43dcae2510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f43fbd0fea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f43fbd0fea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4389e8aa60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4389e8aa60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4389e8aa60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:14, 133076.96it/s] 69%|██████▉   | 6864896/9912422 [00:00<00:16, 189948.90it/s]9920512it [00:00, 38572448.56it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1735165.05it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1008882.33it/s]1654784it [00:00, 12671786.90it/s]                           
0it [00:00, ?it/s]8192it [00:00, 263210.32it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4371d2e6d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4371d3f940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f4389e8a6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f4389e8a6a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f4389e8a6a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:25,  1.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:25,  1.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 2/162 [00:00<01:05,  2.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:05,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:04,  2.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<00:49,  3.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:49,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:48,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:37,  4.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:37,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<00:37,  4.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:01<00:28,  5.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:28,  5.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:28,  5.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:28,  5.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:01<00:21,  6.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:21,  6.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:21,  6.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:21,  6.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:01<00:17,  8.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:17,  8.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:16,  8.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:16,  8.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:01<00:13, 11.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:13, 11.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:13, 11.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:12, 11.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:12, 11.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:01<00:10, 13.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:10, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:10, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:10, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:10, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:01<00:08, 16.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:08, 16.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:08, 16.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:08, 16.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:08, 16.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:01<00:06, 19.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:06, 19.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:06, 19.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:06, 19.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:06, 19.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:01<00:05, 21.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:05, 21.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:02<00:05, 21.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:02<00:05, 21.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:02<00:05, 21.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:02<00:05, 23.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:02<00:05, 23.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:02<00:05, 23.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:02<00:05, 23.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:02<00:04, 25.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:02<00:04, 25.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:02<00:04, 25.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:02<00:04, 25.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:02<00:04, 26.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:02<00:04, 26.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:02<00:04, 26.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:02<00:04, 26.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:02<00:04, 26.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:02<00:04, 26.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:02<00:04, 26.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:02<00:04, 26.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:02<00:04, 27.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:02<00:04, 27.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:04, 27.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:04, 27.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:02<00:04, 27.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:04, 27.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:03, 27.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:03, 27.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:02<00:03, 27.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:03, 27.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:03, 27.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:03, 27.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:02<00:03, 26.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 26.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 26.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:03<00:03, 26.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:03<00:03, 26.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:03<00:03, 25.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:03<00:03, 25.39 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:03<00:03, 25.39 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:03<00:03, 25.39 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:03<00:03, 24.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:03<00:03, 24.76 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:03<00:03, 24.76 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:03<00:03, 24.76 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:03<00:03, 23.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:03<00:03, 23.77 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:03<00:03, 23.77 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:03<00:03, 23.77 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:03<00:03, 22.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:03<00:03, 22.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:03<00:03, 22.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:03<00:03, 22.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:03<00:03, 22.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:03<00:03, 22.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:03<00:03, 22.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:03<00:03, 22.34 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:03<00:03, 22.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:03<00:03, 22.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:03<00:03, 22.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:03<00:03, 22.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:03<00:03, 21.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:03<00:03, 21.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:03<00:03, 21.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:04<00:03, 21.80 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:04<00:03, 21.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:04<00:03, 21.62 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:04<00:03, 21.62 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:04<00:03, 21.62 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:04<00:03, 21.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:04<00:03, 21.45 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:04<00:03, 21.45 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:04<00:03, 21.45 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:04<00:03, 21.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:04<00:03, 21.39 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:04<00:03, 21.39 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:04<00:03, 21.39 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:04<00:03, 21.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:04<00:03, 21.26 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:04<00:03, 21.26 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:04<00:03, 21.26 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:04<00:03, 21.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:04<00:03, 21.29 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:04<00:03, 21.29 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:04<00:02, 21.29 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:04<00:02, 21.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:04<00:02, 21.22 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:04<00:02, 21.22 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:04<00:02, 21.22 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:04<00:02, 21.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:04<00:02, 21.12 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:04<00:02, 21.12 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:05<00:02, 21.12 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:05<00:02, 20.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:05<00:02, 20.77 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:05<00:02, 20.77 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:05<00:02, 20.77 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:05<00:02, 19.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:05<00:02, 19.95 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:05<00:02, 19.95 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:05<00:02, 19.95 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:05<00:02, 19.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:05<00:02, 19.40 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:05<00:02, 19.40 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:05<00:02, 18.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:05<00:02, 18.93 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:05<00:02, 18.93 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:05<00:02, 18.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:05<00:02, 18.73 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:05<00:02, 18.73 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:05<00:02, 18.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:05<00:02, 18.54 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:05<00:02, 18.54 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:05<00:02, 18.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:05<00:02, 18.41 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:05<00:02, 18.41 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:05<00:02, 18.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:05<00:02, 18.63 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:06<00:02, 18.63 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:06<00:02, 18.63 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:06<00:01, 18.63 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:06<00:01, 21.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:06<00:01, 21.32 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:06<00:01, 21.32 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:06<00:01, 21.32 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:06<00:01, 21.32 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:06<00:01, 24.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:06<00:01, 24.25 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:06<00:01, 24.25 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:06<00:01, 24.25 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:06<00:01, 24.25 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:06<00:01, 25.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:06<00:01, 25.80 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:06<00:01, 25.80 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:06<00:01, 25.80 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:06<00:00, 26.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:06<00:00, 26.92 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:06<00:00, 26.92 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:06<00:00, 26.92 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:06<00:00, 27.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:06<00:00, 27.57 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:06<00:00, 27.57 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:06<00:00, 27.57 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:06<00:00, 28.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:06<00:00, 28.17 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:06<00:00, 28.17 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:06<00:00, 28.17 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:06<00:00, 28.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:06<00:00, 28.65 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:06<00:00, 28.65 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:06<00:00, 28.65 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:06<00:00, 28.65 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:06<00:00, 29.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:06<00:00, 29.11 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:06<00:00, 29.11 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:06<00:00, 29.11 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:06<00:00, 29.11 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:06<00:00, 29.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:06<00:00, 29.33 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:07<00:00, 29.33 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:07<00:00, 29.33 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:07<00:00, 28.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:07<00:00, 28.36 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:07<00:00, 28.36 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:07<00:00, 28.36 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:07<00:00, 28.36 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:07<00:00, 28.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:07<00:00, 28.83 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 28.83 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.28s/ url]Dl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 28.83 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 28.83 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:07<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:09<00:00,  9.43s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:09<00:00,  7.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:09<00:00, 28.83 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:09<00:00,  9.43s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:09<00:00,  9.43s/ file]
Dl Size...: 100%|██████████| 162/162 [00:09<00:00, 17.18 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:09<00:00,  9.43s/ url]
0 examples [00:00, ? examples/s]2020-08-04 12:09:07.204426: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-04 12:09:07.217986: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-04 12:09:07.218180: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cfd614a2d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-04 12:09:07.218218: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
18 examples [00:00, 179.25 examples/s]116 examples [00:00, 237.30 examples/s]219 examples [00:00, 308.50 examples/s]317 examples [00:00, 387.95 examples/s]413 examples [00:00, 472.07 examples/s]511 examples [00:00, 558.43 examples/s]614 examples [00:00, 646.43 examples/s]715 examples [00:00, 723.85 examples/s]813 examples [00:00, 784.28 examples/s]908 examples [00:01, 827.29 examples/s]1002 examples [00:01, 858.09 examples/s]1096 examples [00:01, 835.13 examples/s]1186 examples [00:01, 848.11 examples/s]1275 examples [00:01, 860.09 examples/s]1366 examples [00:01, 872.97 examples/s]1456 examples [00:01, 854.08 examples/s]1543 examples [00:01, 836.83 examples/s]1632 examples [00:01, 850.72 examples/s]1723 examples [00:01, 866.26 examples/s]1813 examples [00:02, 874.43 examples/s]1902 examples [00:02, 878.21 examples/s]1993 examples [00:02, 887.25 examples/s]2089 examples [00:02, 907.62 examples/s]2181 examples [00:02, 893.37 examples/s]2277 examples [00:02, 910.25 examples/s]2369 examples [00:02, 906.63 examples/s]2466 examples [00:02, 923.57 examples/s]2564 examples [00:02, 937.52 examples/s]2663 examples [00:02, 952.15 examples/s]2760 examples [00:03, 955.64 examples/s]2856 examples [00:03, 953.95 examples/s]2952 examples [00:03, 944.56 examples/s]3047 examples [00:03, 878.15 examples/s]3143 examples [00:03, 899.62 examples/s]3239 examples [00:03, 915.43 examples/s]3338 examples [00:03, 935.44 examples/s]3433 examples [00:03, 939.49 examples/s]3531 examples [00:03, 950.38 examples/s]3627 examples [00:03, 948.26 examples/s]3723 examples [00:04, 942.75 examples/s]3818 examples [00:04, 936.95 examples/s]3912 examples [00:04, 920.03 examples/s]4005 examples [00:04, 910.18 examples/s]4097 examples [00:04, 895.57 examples/s]4194 examples [00:04, 913.95 examples/s]4292 examples [00:04, 931.02 examples/s]4388 examples [00:04, 938.24 examples/s]4486 examples [00:04, 949.15 examples/s]4588 examples [00:05, 968.09 examples/s]4690 examples [00:05, 981.68 examples/s]4789 examples [00:05, 948.13 examples/s]4891 examples [00:05, 968.39 examples/s]4994 examples [00:05, 983.85 examples/s]5097 examples [00:05, 996.42 examples/s]5197 examples [00:05, 995.34 examples/s]5297 examples [00:05, 992.76 examples/s]5399 examples [00:05, 999.44 examples/s]5500 examples [00:05, 970.65 examples/s]5598 examples [00:06, 968.61 examples/s]5696 examples [00:06, 962.65 examples/s]5793 examples [00:06, 961.92 examples/s]5895 examples [00:06, 978.12 examples/s]5993 examples [00:06, 964.32 examples/s]6094 examples [00:06, 975.53 examples/s]6193 examples [00:06, 974.64 examples/s]6291 examples [00:06, 971.46 examples/s]6390 examples [00:06, 974.37 examples/s]6490 examples [00:06, 979.73 examples/s]6589 examples [00:07, 973.80 examples/s]6687 examples [00:07, 966.52 examples/s]6784 examples [00:07, 966.68 examples/s]6881 examples [00:07, 960.17 examples/s]6982 examples [00:07, 973.20 examples/s]7087 examples [00:07, 994.51 examples/s]7187 examples [00:07, 994.13 examples/s]7287 examples [00:07, 983.86 examples/s]7386 examples [00:07, 980.63 examples/s]7485 examples [00:07, 962.25 examples/s]7582 examples [00:08, 963.12 examples/s]7679 examples [00:08, 956.79 examples/s]7775 examples [00:08, 952.31 examples/s]7871 examples [00:08, 949.34 examples/s]7966 examples [00:08, 947.93 examples/s]8061 examples [00:08, 946.99 examples/s]8156 examples [00:08, 933.90 examples/s]8251 examples [00:08, 938.61 examples/s]8345 examples [00:08, 929.83 examples/s]8441 examples [00:09, 936.16 examples/s]8538 examples [00:09, 944.43 examples/s]8633 examples [00:09, 943.16 examples/s]8728 examples [00:09, 936.18 examples/s]8823 examples [00:09, 937.39 examples/s]8918 examples [00:09, 939.16 examples/s]9016 examples [00:09, 949.24 examples/s]9111 examples [00:09, 935.85 examples/s]9205 examples [00:09, 932.10 examples/s]9299 examples [00:09, 896.16 examples/s]9395 examples [00:10, 913.64 examples/s]9488 examples [00:10, 917.91 examples/s]9585 examples [00:10, 931.49 examples/s]9685 examples [00:10, 949.01 examples/s]9787 examples [00:10, 967.00 examples/s]9884 examples [00:10, 966.88 examples/s]9981 examples [00:10, 936.58 examples/s]10075 examples [00:10, 860.67 examples/s]10166 examples [00:10, 872.21 examples/s]10261 examples [00:10, 893.71 examples/s]10359 examples [00:11, 915.94 examples/s]10457 examples [00:11, 931.74 examples/s]10551 examples [00:11, 886.22 examples/s]10650 examples [00:11, 913.40 examples/s]10747 examples [00:11, 928.34 examples/s]10841 examples [00:11, 931.47 examples/s]10935 examples [00:11, 912.26 examples/s]11028 examples [00:11, 915.06 examples/s]11121 examples [00:11, 917.08 examples/s]11213 examples [00:12, 882.35 examples/s]11307 examples [00:12, 896.63 examples/s]11402 examples [00:12, 910.91 examples/s]11497 examples [00:12, 922.08 examples/s]11593 examples [00:12, 931.52 examples/s]11687 examples [00:12, 920.70 examples/s]11782 examples [00:12, 928.44 examples/s]11879 examples [00:12, 938.78 examples/s]11973 examples [00:12, 881.47 examples/s]12068 examples [00:12, 899.24 examples/s]12159 examples [00:13, 901.10 examples/s]12254 examples [00:13, 912.82 examples/s]12346 examples [00:13, 909.82 examples/s]12438 examples [00:13, 908.28 examples/s]12532 examples [00:13, 915.56 examples/s]12624 examples [00:13, 880.22 examples/s]12714 examples [00:13, 884.40 examples/s]12808 examples [00:13, 899.12 examples/s]12899 examples [00:13, 899.02 examples/s]12994 examples [00:13, 912.05 examples/s]13086 examples [00:14, 912.65 examples/s]13182 examples [00:14, 925.39 examples/s]13279 examples [00:14, 935.96 examples/s]13373 examples [00:14, 907.50 examples/s]13470 examples [00:14, 923.51 examples/s]13567 examples [00:14, 935.84 examples/s]13663 examples [00:14, 942.67 examples/s]13760 examples [00:14, 950.51 examples/s]13856 examples [00:14, 944.38 examples/s]13952 examples [00:14, 947.58 examples/s]14053 examples [00:15, 963.19 examples/s]14151 examples [00:15, 967.80 examples/s]14248 examples [00:15, 964.55 examples/s]14352 examples [00:15, 984.80 examples/s]14451 examples [00:15, 979.67 examples/s]14551 examples [00:15, 984.28 examples/s]14650 examples [00:15, 984.38 examples/s]14752 examples [00:15, 991.08 examples/s]14852 examples [00:15, 947.10 examples/s]14949 examples [00:16, 953.65 examples/s]15048 examples [00:16, 964.22 examples/s]15145 examples [00:16, 954.68 examples/s]15241 examples [00:16, 933.61 examples/s]15337 examples [00:16, 939.09 examples/s]15434 examples [00:16, 948.10 examples/s]15529 examples [00:16, 939.12 examples/s]15626 examples [00:16, 947.90 examples/s]15723 examples [00:16, 953.11 examples/s]15825 examples [00:16, 971.22 examples/s]15924 examples [00:17, 974.73 examples/s]16029 examples [00:17, 996.08 examples/s]16133 examples [00:17, 1006.51 examples/s]16235 examples [00:17, 1009.76 examples/s]16337 examples [00:17, 978.21 examples/s] 16436 examples [00:17, 967.98 examples/s]16534 examples [00:17, 966.12 examples/s]16631 examples [00:17, 949.75 examples/s]16727 examples [00:17, 943.16 examples/s]16828 examples [00:17, 962.26 examples/s]16926 examples [00:18, 967.43 examples/s]17027 examples [00:18, 977.07 examples/s]17128 examples [00:18, 985.98 examples/s]17227 examples [00:18, 969.96 examples/s]17325 examples [00:18, 970.97 examples/s]17423 examples [00:18, 967.68 examples/s]17525 examples [00:18, 980.21 examples/s]17624 examples [00:18, 976.10 examples/s]17722 examples [00:18, 957.21 examples/s]17818 examples [00:18, 957.08 examples/s]17914 examples [00:19, 954.83 examples/s]18014 examples [00:19, 966.03 examples/s]18111 examples [00:19, 962.82 examples/s]18208 examples [00:19, 947.76 examples/s]18306 examples [00:19, 956.94 examples/s]18406 examples [00:19, 968.71 examples/s]18505 examples [00:19, 974.53 examples/s]18604 examples [00:19, 976.88 examples/s]18702 examples [00:19, 954.44 examples/s]18798 examples [00:20, 945.16 examples/s]18898 examples [00:20, 958.00 examples/s]18999 examples [00:20, 972.70 examples/s]19101 examples [00:20, 986.21 examples/s]19200 examples [00:20, 960.06 examples/s]19297 examples [00:20, 951.57 examples/s]19393 examples [00:20, 946.61 examples/s]19488 examples [00:20, 944.63 examples/s]19584 examples [00:20, 948.13 examples/s]19679 examples [00:20, 929.54 examples/s]19774 examples [00:21, 934.60 examples/s]19868 examples [00:21, 935.51 examples/s]19962 examples [00:21, 935.40 examples/s]20056 examples [00:21, 871.13 examples/s]20148 examples [00:21, 884.26 examples/s]20243 examples [00:21, 902.43 examples/s]20336 examples [00:21, 908.05 examples/s]20428 examples [00:21, 908.71 examples/s]20529 examples [00:21, 936.46 examples/s]20625 examples [00:21, 942.07 examples/s]20723 examples [00:22, 950.33 examples/s]20823 examples [00:22, 964.03 examples/s]20920 examples [00:22, 950.18 examples/s]21016 examples [00:22, 924.94 examples/s]21109 examples [00:22, 921.71 examples/s]21202 examples [00:22, 919.88 examples/s]21295 examples [00:22, 894.04 examples/s]21386 examples [00:22, 896.41 examples/s]21479 examples [00:22, 904.16 examples/s]21570 examples [00:23, 905.09 examples/s]21661 examples [00:23, 902.42 examples/s]21758 examples [00:23, 920.13 examples/s]21851 examples [00:23, 908.89 examples/s]21944 examples [00:23, 913.28 examples/s]22036 examples [00:23, 880.71 examples/s]22134 examples [00:23, 906.11 examples/s]22229 examples [00:23, 916.10 examples/s]22321 examples [00:23, 887.47 examples/s]22411 examples [00:23, 889.83 examples/s]22505 examples [00:24, 902.45 examples/s]22596 examples [00:24, 901.15 examples/s]22691 examples [00:24, 912.24 examples/s]22783 examples [00:24, 911.41 examples/s]22879 examples [00:24, 925.02 examples/s]22972 examples [00:24, 909.49 examples/s]23064 examples [00:24, 895.24 examples/s]23154 examples [00:24, 876.48 examples/s]23244 examples [00:24, 880.10 examples/s]23333 examples [00:24, 877.69 examples/s]23421 examples [00:25, 876.07 examples/s]23509 examples [00:25, 862.64 examples/s]23601 examples [00:25, 875.71 examples/s]23692 examples [00:25, 883.76 examples/s]23781 examples [00:25, 879.63 examples/s]23870 examples [00:25, 880.49 examples/s]23959 examples [00:25, 836.97 examples/s]24045 examples [00:25, 842.70 examples/s]24135 examples [00:25, 857.99 examples/s]24226 examples [00:25, 872.13 examples/s]24314 examples [00:26, 853.91 examples/s]24409 examples [00:26, 880.49 examples/s]24504 examples [00:26, 897.96 examples/s]24597 examples [00:26, 905.93 examples/s]24688 examples [00:26, 847.43 examples/s]24779 examples [00:26, 863.72 examples/s]24870 examples [00:26, 875.04 examples/s]24965 examples [00:26, 894.16 examples/s]25059 examples [00:26, 907.27 examples/s]25151 examples [00:27, 897.75 examples/s]25243 examples [00:27, 902.65 examples/s]25334 examples [00:27, 895.54 examples/s]25424 examples [00:27, 884.36 examples/s]25522 examples [00:27, 908.36 examples/s]25614 examples [00:27, 897.61 examples/s]25706 examples [00:27, 902.78 examples/s]25801 examples [00:27, 914.57 examples/s]25893 examples [00:27, 914.44 examples/s]25988 examples [00:27, 922.78 examples/s]26082 examples [00:28, 927.57 examples/s]26175 examples [00:28, 918.32 examples/s]26271 examples [00:28, 929.27 examples/s]26365 examples [00:28, 923.29 examples/s]26458 examples [00:28, 922.67 examples/s]26556 examples [00:28, 937.79 examples/s]26651 examples [00:28, 940.05 examples/s]26754 examples [00:28, 964.48 examples/s]26851 examples [00:28, 926.81 examples/s]26948 examples [00:28, 936.49 examples/s]27042 examples [00:29, 934.24 examples/s]27136 examples [00:29, 924.25 examples/s]27231 examples [00:29, 931.59 examples/s]27329 examples [00:29, 942.97 examples/s]27431 examples [00:29, 964.17 examples/s]27528 examples [00:29, 950.71 examples/s]27624 examples [00:29, 945.83 examples/s]27719 examples [00:29, 940.41 examples/s]27814 examples [00:29, 933.26 examples/s]27908 examples [00:30, 921.94 examples/s]28001 examples [00:30, 911.08 examples/s]28095 examples [00:30, 917.18 examples/s]28188 examples [00:30, 920.61 examples/s]28281 examples [00:30, 921.77 examples/s]28375 examples [00:30, 925.45 examples/s]28468 examples [00:30, 920.47 examples/s]28561 examples [00:30, 908.03 examples/s]28656 examples [00:30, 918.22 examples/s]28752 examples [00:30, 930.01 examples/s]28850 examples [00:31, 942.93 examples/s]28945 examples [00:31, 930.71 examples/s]29041 examples [00:31, 938.90 examples/s]29136 examples [00:31, 939.01 examples/s]29230 examples [00:31, 926.79 examples/s]29331 examples [00:31, 950.21 examples/s]29427 examples [00:31, 945.78 examples/s]29522 examples [00:31, 946.53 examples/s]29627 examples [00:31, 975.25 examples/s]29727 examples [00:31, 981.76 examples/s]29828 examples [00:32, 988.18 examples/s]29927 examples [00:32, 978.17 examples/s]30025 examples [00:32, 886.29 examples/s]30125 examples [00:32, 916.82 examples/s]30219 examples [00:32, 896.72 examples/s]30318 examples [00:32, 921.21 examples/s]30412 examples [00:32, 895.88 examples/s]30503 examples [00:32, 896.69 examples/s]30599 examples [00:32, 912.97 examples/s]30694 examples [00:33, 920.43 examples/s]30791 examples [00:33, 933.87 examples/s]30886 examples [00:33, 938.36 examples/s]30990 examples [00:33, 966.42 examples/s]31088 examples [00:33, 968.07 examples/s]31186 examples [00:33, 941.90 examples/s]31281 examples [00:33, 942.79 examples/s]31376 examples [00:33, 924.27 examples/s]31470 examples [00:33, 928.23 examples/s]31569 examples [00:33, 945.13 examples/s]31665 examples [00:34, 937.94 examples/s]31759 examples [00:34, 927.96 examples/s]31852 examples [00:34, 920.22 examples/s]31945 examples [00:34, 902.96 examples/s]32036 examples [00:34, 887.73 examples/s]32136 examples [00:34, 918.55 examples/s]32234 examples [00:34, 934.89 examples/s]32332 examples [00:34, 947.31 examples/s]32428 examples [00:34, 939.76 examples/s]32525 examples [00:34, 947.24 examples/s]32623 examples [00:35, 955.07 examples/s]32721 examples [00:35, 961.26 examples/s]32818 examples [00:35, 961.86 examples/s]32915 examples [00:35, 932.11 examples/s]33009 examples [00:35, 893.74 examples/s]33105 examples [00:35, 910.47 examples/s]33197 examples [00:35, 909.28 examples/s]33298 examples [00:35, 937.00 examples/s]33398 examples [00:35, 953.67 examples/s]33494 examples [00:35, 952.72 examples/s]33598 examples [00:36, 976.21 examples/s]33696 examples [00:36, 960.40 examples/s]33793 examples [00:36, 944.61 examples/s]33894 examples [00:36, 960.30 examples/s]33996 examples [00:36, 976.81 examples/s]34094 examples [00:36, 976.17 examples/s]34196 examples [00:36, 986.56 examples/s]34297 examples [00:36, 991.37 examples/s]34397 examples [00:36, 993.35 examples/s]34497 examples [00:37, 985.31 examples/s]34596 examples [00:37, 981.14 examples/s]34695 examples [00:37, 982.39 examples/s]34794 examples [00:37, 975.57 examples/s]34892 examples [00:37, 973.10 examples/s]34990 examples [00:37, 945.54 examples/s]35085 examples [00:37, 945.57 examples/s]35180 examples [00:37, 940.19 examples/s]35278 examples [00:37, 950.63 examples/s]35374 examples [00:37, 950.17 examples/s]35474 examples [00:38, 964.22 examples/s]35579 examples [00:38, 986.09 examples/s]35678 examples [00:38, 983.35 examples/s]35777 examples [00:38, 953.56 examples/s]35873 examples [00:38, 951.74 examples/s]35969 examples [00:38, 942.69 examples/s]36068 examples [00:38, 953.88 examples/s]36164 examples [00:38, 947.73 examples/s]36259 examples [00:38, 937.47 examples/s]36353 examples [00:38, 935.19 examples/s]36447 examples [00:39, 910.52 examples/s]36539 examples [00:39, 900.01 examples/s]36630 examples [00:39, 898.39 examples/s]36726 examples [00:39, 913.64 examples/s]36820 examples [00:39, 920.07 examples/s]36913 examples [00:39, 897.68 examples/s]37011 examples [00:39, 918.49 examples/s]37104 examples [00:39, 883.19 examples/s]37197 examples [00:39, 896.13 examples/s]37287 examples [00:40, 885.68 examples/s]37379 examples [00:40, 893.58 examples/s]37478 examples [00:40, 919.49 examples/s]37571 examples [00:40, 913.83 examples/s]37666 examples [00:40, 924.16 examples/s]37759 examples [00:40, 909.89 examples/s]37853 examples [00:40, 916.29 examples/s]37949 examples [00:40, 928.13 examples/s]38042 examples [00:40, 898.37 examples/s]38145 examples [00:40, 933.04 examples/s]38250 examples [00:41, 965.19 examples/s]38353 examples [00:41, 982.69 examples/s]38452 examples [00:41, 928.11 examples/s]38548 examples [00:41, 936.92 examples/s]38648 examples [00:41, 954.88 examples/s]38745 examples [00:41, 947.67 examples/s]38841 examples [00:41, 933.23 examples/s]38935 examples [00:41, 905.99 examples/s]39027 examples [00:41, 897.81 examples/s]39119 examples [00:41, 903.03 examples/s]39218 examples [00:42, 926.06 examples/s]39317 examples [00:42, 943.37 examples/s]39413 examples [00:42, 947.76 examples/s]39515 examples [00:42, 967.22 examples/s]39612 examples [00:42, 961.05 examples/s]39710 examples [00:42, 965.52 examples/s]39807 examples [00:42, 955.63 examples/s]39903 examples [00:42, 942.46 examples/s]40001 examples [00:42, 886.43 examples/s]40096 examples [00:43, 902.25 examples/s]40191 examples [00:43, 914.32 examples/s]40294 examples [00:43, 944.08 examples/s]40390 examples [00:43, 945.59 examples/s]40485 examples [00:43, 938.47 examples/s]40583 examples [00:43, 950.46 examples/s]40679 examples [00:43, 946.34 examples/s]40779 examples [00:43, 959.58 examples/s]40876 examples [00:43, 933.05 examples/s]40973 examples [00:43, 941.55 examples/s]41068 examples [00:44, 940.49 examples/s]41163 examples [00:44, 941.41 examples/s]41258 examples [00:44, 942.50 examples/s]41357 examples [00:44, 955.59 examples/s]41456 examples [00:44, 964.46 examples/s]41555 examples [00:44, 970.13 examples/s]41654 examples [00:44, 974.68 examples/s]41752 examples [00:44, 958.97 examples/s]41848 examples [00:44, 945.29 examples/s]41943 examples [00:44, 943.61 examples/s]42038 examples [00:45, 934.48 examples/s]42132 examples [00:45, 918.04 examples/s]42226 examples [00:45, 924.50 examples/s]42321 examples [00:45, 930.02 examples/s]42423 examples [00:45, 955.04 examples/s]42524 examples [00:45, 969.20 examples/s]42626 examples [00:45, 983.73 examples/s]42726 examples [00:45, 984.74 examples/s]42825 examples [00:45, 972.58 examples/s]42927 examples [00:45, 986.29 examples/s]43027 examples [00:46, 990.29 examples/s]43128 examples [00:46, 994.25 examples/s]43228 examples [00:46, 978.78 examples/s]43326 examples [00:46, 978.08 examples/s]43424 examples [00:46, 958.27 examples/s]43523 examples [00:46, 966.41 examples/s]43620 examples [00:46, 923.44 examples/s]43713 examples [00:46, 904.79 examples/s]43804 examples [00:46, 889.47 examples/s]43894 examples [00:47, 892.22 examples/s]43991 examples [00:47, 911.93 examples/s]44090 examples [00:47, 932.55 examples/s]44190 examples [00:47, 951.76 examples/s]44286 examples [00:47, 950.34 examples/s]44388 examples [00:47, 967.70 examples/s]44490 examples [00:47, 982.30 examples/s]44589 examples [00:47, 947.51 examples/s]44685 examples [00:47, 942.62 examples/s]44780 examples [00:47, 939.92 examples/s]44875 examples [00:48, 931.47 examples/s]44969 examples [00:48, 931.01 examples/s]45067 examples [00:48, 944.05 examples/s]45167 examples [00:48, 958.28 examples/s]45264 examples [00:48, 959.55 examples/s]45362 examples [00:48, 964.26 examples/s]45459 examples [00:48, 965.81 examples/s]45556 examples [00:48, 950.48 examples/s]45652 examples [00:48, 953.14 examples/s]45748 examples [00:48, 936.38 examples/s]45842 examples [00:49, 927.40 examples/s]45935 examples [00:49, 920.16 examples/s]46028 examples [00:49, 919.14 examples/s]46121 examples [00:49, 921.18 examples/s]46214 examples [00:49, 918.37 examples/s]46317 examples [00:49, 948.71 examples/s]46413 examples [00:49, 937.98 examples/s]46508 examples [00:49, 935.16 examples/s]46602 examples [00:49, 926.44 examples/s]46696 examples [00:49, 928.16 examples/s]46794 examples [00:50, 942.06 examples/s]46889 examples [00:50, 895.18 examples/s]46983 examples [00:50, 907.82 examples/s]47075 examples [00:50, 902.15 examples/s]47169 examples [00:50, 912.48 examples/s]47266 examples [00:50, 928.26 examples/s]47360 examples [00:50, 931.47 examples/s]47454 examples [00:50, 925.15 examples/s]47552 examples [00:50, 938.95 examples/s]47647 examples [00:51, 936.16 examples/s]47742 examples [00:51, 938.61 examples/s]47836 examples [00:51, 936.76 examples/s]47930 examples [00:51, 928.90 examples/s]48023 examples [00:51, 905.89 examples/s]48115 examples [00:51, 909.31 examples/s]48211 examples [00:51, 922.83 examples/s]48307 examples [00:51, 933.61 examples/s]48401 examples [00:51, 928.59 examples/s]48494 examples [00:51, 917.17 examples/s]48586 examples [00:52, 908.46 examples/s]48681 examples [00:52, 919.87 examples/s]48774 examples [00:52, 920.66 examples/s]48867 examples [00:52, 916.69 examples/s]48965 examples [00:52, 933.02 examples/s]49059 examples [00:52, 915.00 examples/s]49153 examples [00:52, 916.88 examples/s]49252 examples [00:52, 937.31 examples/s]49353 examples [00:52, 957.61 examples/s]49455 examples [00:52, 974.80 examples/s]49553 examples [00:53, 956.34 examples/s]49654 examples [00:53, 971.13 examples/s]49754 examples [00:53, 979.30 examples/s]49853 examples [00:53, 981.75 examples/s]49957 examples [00:53, 995.96 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7361/50000 [00:00<00:00, 73608.67 examples/s] 40%|███▉      | 19867/50000 [00:00<00:00, 83971.74 examples/s] 65%|██████▍   | 32298/50000 [00:00<00:00, 93027.01 examples/s] 90%|█████████ | 45073/50000 [00:00<00:00, 101285.16 examples/s]                                                                0 examples [00:00, ? examples/s]71 examples [00:00, 709.60 examples/s]167 examples [00:00, 768.37 examples/s]264 examples [00:00, 818.08 examples/s]353 examples [00:00, 836.33 examples/s]452 examples [00:00, 875.07 examples/s]547 examples [00:00, 894.97 examples/s]645 examples [00:00, 918.35 examples/s]747 examples [00:00, 945.45 examples/s]844 examples [00:00, 951.84 examples/s]940 examples [00:01, 952.68 examples/s]1034 examples [00:01, 945.58 examples/s]1133 examples [00:01, 957.05 examples/s]1234 examples [00:01, 971.22 examples/s]1334 examples [00:01, 978.73 examples/s]1433 examples [00:01, 979.24 examples/s]1531 examples [00:01, 969.11 examples/s]1628 examples [00:01, 966.39 examples/s]1729 examples [00:01, 977.62 examples/s]1827 examples [00:01, 948.04 examples/s]1926 examples [00:02, 958.81 examples/s]2024 examples [00:02, 963.87 examples/s]2125 examples [00:02, 974.74 examples/s]2225 examples [00:02, 979.34 examples/s]2324 examples [00:02, 956.68 examples/s]2421 examples [00:02, 960.38 examples/s]2518 examples [00:02, 945.78 examples/s]2616 examples [00:02, 954.41 examples/s]2720 examples [00:02, 976.72 examples/s]2823 examples [00:02, 989.63 examples/s]2923 examples [00:03, 988.43 examples/s]3022 examples [00:03, 976.21 examples/s]3122 examples [00:03, 980.72 examples/s]3222 examples [00:03, 985.60 examples/s]3321 examples [00:03, 985.27 examples/s]3420 examples [00:03, 981.24 examples/s]3519 examples [00:03, 966.67 examples/s]3616 examples [00:03, 938.62 examples/s]3715 examples [00:03, 951.34 examples/s]3811 examples [00:03, 947.21 examples/s]3906 examples [00:04, 919.44 examples/s]4000 examples [00:04, 924.95 examples/s]4098 examples [00:04, 940.63 examples/s]4200 examples [00:04, 961.00 examples/s]4297 examples [00:04, 920.31 examples/s]4393 examples [00:04, 930.34 examples/s]4487 examples [00:04, 902.19 examples/s]4578 examples [00:04, 891.20 examples/s]4672 examples [00:04, 904.82 examples/s]4763 examples [00:05, 878.61 examples/s]4857 examples [00:05, 895.61 examples/s]4947 examples [00:05, 894.87 examples/s]5045 examples [00:05, 917.63 examples/s]5151 examples [00:05, 955.08 examples/s]5254 examples [00:05, 975.65 examples/s]5359 examples [00:05, 996.47 examples/s]5460 examples [00:05, 972.67 examples/s]5558 examples [00:05, 970.73 examples/s]5659 examples [00:05, 980.11 examples/s]5758 examples [00:06, 966.74 examples/s]5855 examples [00:06, 954.64 examples/s]5951 examples [00:06, 935.09 examples/s]6051 examples [00:06, 951.14 examples/s]6147 examples [00:06, 951.07 examples/s]6243 examples [00:06, 953.69 examples/s]6339 examples [00:06, 954.30 examples/s]6437 examples [00:06, 960.73 examples/s]6534 examples [00:06, 959.08 examples/s]6632 examples [00:06, 962.11 examples/s]6732 examples [00:07, 972.75 examples/s]6831 examples [00:07, 975.80 examples/s]6929 examples [00:07, 970.53 examples/s]7027 examples [00:07, 964.40 examples/s]7124 examples [00:07, 960.77 examples/s]7221 examples [00:07, 961.32 examples/s]7318 examples [00:07, 945.60 examples/s]7413 examples [00:07, 929.04 examples/s]7507 examples [00:07, 878.82 examples/s]7603 examples [00:08, 901.10 examples/s]7702 examples [00:08, 923.81 examples/s]7795 examples [00:08, 925.37 examples/s]7889 examples [00:08, 929.59 examples/s]7985 examples [00:08, 937.50 examples/s]8080 examples [00:08, 940.81 examples/s]8176 examples [00:08, 945.92 examples/s]8271 examples [00:08, 943.71 examples/s]8366 examples [00:08, 936.77 examples/s]8465 examples [00:08, 950.88 examples/s]8561 examples [00:09, 931.49 examples/s]8662 examples [00:09, 952.44 examples/s]8758 examples [00:09, 949.06 examples/s]8859 examples [00:09, 964.20 examples/s]8959 examples [00:09, 973.61 examples/s]9057 examples [00:09, 970.79 examples/s]9160 examples [00:09, 985.44 examples/s]9261 examples [00:09, 989.96 examples/s]9361 examples [00:09, 934.99 examples/s]9460 examples [00:09, 949.50 examples/s]9556 examples [00:10, 934.16 examples/s]9652 examples [00:10, 940.24 examples/s]9747 examples [00:10, 939.46 examples/s]9843 examples [00:10, 945.50 examples/s]9939 examples [00:10, 948.15 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteY0I29T/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteY0I29T/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4389e8aa60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4389e8aa60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4389e8aa60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f43d57f70b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f431ca86f98>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4389e8aa60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4389e8aa60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4389e8aa60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4371d3f240>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4371d3f208>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 15.75 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 17.87 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 17.87 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.68 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.68 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.22 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.22 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.19 file/s]2020-08-04 12:10:19.588917: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-04 12:10:19.593409: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-04 12:10:19.593581: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5621ece473f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-04 12:10:19.593597: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  1%|          | 106496/9912422 [00:00<00:09, 1009256.18it/s] 61%|██████    | 6004736/9912422 [00:00<00:02, 1431275.81it/s]9920512it [00:00, 35630368.50it/s]                            
0it [00:00, ?it/s]32768it [00:00, 632764.38it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 484545.09it/s]1654784it [00:00, 12393163.26it/s]                         
0it [00:00, ?it/s]8192it [00:00, 205322.77it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

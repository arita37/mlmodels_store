
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f8f9cfba598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f8f9cfba598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f9007c6a488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f9007c6a488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f9026e88ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f9026e88ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8fb4fa6158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8fb4fa6158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f8fb4fa6158> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3678208/9912422 [00:00<00:00, 36777117.12it/s]9920512it [00:00, 29090837.58it/s]                             
0it [00:00, ?it/s]32768it [00:00, 534361.92it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160807.50it/s]1654784it [00:00, 11784235.35it/s]                         
0it [00:00, ?it/s]8192it [00:00, 147983.68it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8fb3003358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8fb33a6940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f8fb4f9ad90> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f8fb4f9ad90> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f8fb4f9ad90> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:11,  2.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:11,  2.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 2/162 [00:00<00:58,  2.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:58,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 3/162 [00:00<00:49,  3.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:49,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<00:42,  3.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:42,  3.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:01<00:37,  4.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<00:37,  4.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:01<00:34,  4.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<00:34,  4.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:01<00:31,  4.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<00:31,  4.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:01<00:30,  5.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:30,  5.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:01<00:28,  5.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:28,  5.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:01<00:27,  5.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:02<00:27,  5.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:02<00:27,  5.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:02<00:27,  5.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:02<00:27,  5.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:02<00:27,  5.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:02<00:26,  5.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:02<00:26,  5.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:02<00:27,  5.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:02<00:27,  5.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:02<00:27,  5.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:02<00:27,  5.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:03<00:27,  5.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:03<00:27,  5.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:03<00:27,  5.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:03<00:27,  5.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:03<00:27,  5.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:03<00:27,  5.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:03<00:26,  5.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:03<00:26,  5.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:03<00:26,  5.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:03<00:26,  5.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:04<00:26,  5.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:04<00:26,  5.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:04<00:26,  5.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:04<00:26,  5.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:04<00:26,  5.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:04<00:26,  5.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:04<00:26,  5.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:04<00:26,  5.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:04<00:25,  5.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:04<00:25,  5.32 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:04<00:25,  5.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:04<00:25,  5.40 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:05<00:24,  5.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:05<00:24,  5.45 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:05<00:24,  5.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:05<00:24,  5.44 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:05<00:24,  5.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:05<00:24,  5.49 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:05<00:23,  5.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:05<00:23,  5.51 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:05<00:23,  5.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:05<00:23,  5.54 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:06<00:23,  5.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:06<00:23,  5.62 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:06<00:22,  5.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:06<00:22,  5.67 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:06<00:22,  5.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:06<00:22,  5.68 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:06<00:22,  5.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:06<00:22,  5.66 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:06<00:22,  5.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:06<00:22,  5.65 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:06<00:22,  5.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:06<00:22,  5.63 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:07<00:22,  5.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:07<00:22,  5.63 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:07<00:21,  5.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:07<00:21,  5.70 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:07<00:21,  5.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:07<00:21,  5.74 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:07<00:20,  5.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:07<00:20,  5.77 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:07<00:20,  5.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:07<00:20,  5.80 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:07<00:20,  5.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:07<00:20,  5.83 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:08<00:20,  5.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:08<00:20,  5.82 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:08<00:20,  5.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:08<00:20,  5.83 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:08<00:19,  5.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:08<00:19,  5.91 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:08<00:19,  5.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:08<00:19,  5.97 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:08<00:19,  6.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:08<00:19,  6.00 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:08<00:18,  5.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:08<00:18,  5.95 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:09<00:19,  5.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:09<00:19,  5.85 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:09<00:18,  5.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:09<00:18,  5.92 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:09<00:18,  5.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:09<00:18,  5.99 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:09<00:18,  6.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:09<00:18,  6.04 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:09<00:17,  6.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:09<00:17,  6.08 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:09<00:17,  6.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:09<00:17,  6.09 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:10<00:17,  6.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:10<00:17,  6.10 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:10<00:17,  6.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:10<00:17,  6.07 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:10<00:17,  6.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:10<00:17,  6.04 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:10<00:17,  6.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:10<00:17,  6.02 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:10<00:16,  6.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:10<00:16,  6.06 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:10<00:16,  6.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:10<00:16,  6.08 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:11<00:16,  6.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:11<00:16,  6.09 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:11<00:16,  6.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:11<00:16,  6.10 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:11<00:15,  6.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:11<00:15,  6.15 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:11<00:15,  6.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:11<00:15,  6.08 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:11<00:15,  6.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:11<00:15,  6.02 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:11<00:15,  5.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:11<00:15,  5.95 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:12<00:15,  5.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:12<00:15,  5.90 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:12<00:16,  5.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:12<00:16,  5.76 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:12<00:16,  5.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:12<00:16,  5.61 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:12<00:16,  5.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:12<00:16,  5.50 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:12<00:16,  5.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:12<00:16,  5.42 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:13<00:16,  5.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:13<00:16,  5.36 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:13<00:16,  5.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:13<00:16,  5.33 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:13<00:16,  5.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:13<00:16,  5.39 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:13<00:15,  5.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:13<00:15,  5.43 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:13<00:15,  5.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:13<00:15,  5.41 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:13<00:15,  5.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:13<00:15,  5.44 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:14<00:15,  5.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:14<00:15,  5.47 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:14<00:14,  5.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:14<00:14,  5.48 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:14<00:14,  5.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:14<00:14,  5.48 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:14<00:14,  5.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:14<00:14,  5.54 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:14<00:14,  5.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:14<00:14,  5.57 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:15<00:14,  5.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:15<00:14,  5.55 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:15<00:13,  5.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:15<00:13,  5.61 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:15<00:13,  5.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:15<00:13,  5.66 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:15<00:13,  5.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:15<00:13,  5.72 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:15<00:12,  5.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:15<00:12,  5.74 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:15<00:12,  5.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:15<00:12,  5.78 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:16<00:12,  5.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:16<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:16<00:12,  5.88 MiB/s][A

Extraction completed...: 0 file [00:16, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:16<00:11,  5.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:16<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:16<00:11,  5.96 MiB/s][A

Extraction completed...: 0 file [00:16, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:16<00:11,  6.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:16<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:16<00:11,  6.02 MiB/s][A

Extraction completed...: 0 file [00:16, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:16<00:11,  6.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:16<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:16<00:11,  6.07 MiB/s][A

Extraction completed...: 0 file [00:16, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:16<00:11,  6.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:16<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:16<00:11,  6.09 MiB/s][A

Extraction completed...: 0 file [00:16, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:16<00:10,  6.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:16<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:16<00:10,  6.12 MiB/s][A

Extraction completed...: 0 file [00:16, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:17<00:10,  6.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:17<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:17<00:10,  6.13 MiB/s][A

Extraction completed...: 0 file [00:17, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:17<00:10,  6.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:17<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:17<00:10,  6.11 MiB/s][A

Extraction completed...: 0 file [00:17, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:17<00:10,  6.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:17<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:17<00:10,  6.11 MiB/s][A

Extraction completed...: 0 file [00:17, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:17<00:10,  6.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:17<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:17<00:10,  6.10 MiB/s][A

Extraction completed...: 0 file [00:17, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:17<00:10,  6.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:17<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:17<00:10,  6.14 MiB/s][A

Extraction completed...: 0 file [00:17, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:17<00:09,  6.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:17<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:17<00:09,  6.25 MiB/s][A

Extraction completed...: 0 file [00:17, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:18<00:09,  6.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:18<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:18<00:09,  6.43 MiB/s][A

Extraction completed...: 0 file [00:18, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:18<00:08,  6.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:18<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:18<00:08,  6.59 MiB/s][A

Extraction completed...: 0 file [00:18, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:18<00:08,  6.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:18<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:18<00:08,  6.58 MiB/s][A

Extraction completed...: 0 file [00:18, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:18<00:08,  6.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:18<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:18<00:08,  6.68 MiB/s][A

Extraction completed...: 0 file [00:18, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:18<00:08,  6.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:18<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:18<00:08,  6.76 MiB/s][A

Extraction completed...: 0 file [00:18, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:18<00:08,  6.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:18<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:18<00:08,  6.86 MiB/s][A

Extraction completed...: 0 file [00:18, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:18<00:07,  7.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:18<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:18<00:07,  7.14 MiB/s][A

Extraction completed...: 0 file [00:18, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:18<00:07,  7.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:18<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:18<00:07,  7.42 MiB/s][A

Extraction completed...: 0 file [00:18, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:19<00:06,  7.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:19<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:19<00:06,  7.58 MiB/s][A

Extraction completed...: 0 file [00:19, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:19<00:06,  7.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:19<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:19<00:06,  7.55 MiB/s][A

Extraction completed...: 0 file [00:19, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:19<00:06,  7.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:19<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:19<00:06,  7.67 MiB/s][A

Extraction completed...: 0 file [00:19, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:19<00:06,  7.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:19<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:19<00:06,  7.75 MiB/s][A

Extraction completed...: 0 file [00:19, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:19<00:06,  7.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:19<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:19<00:06,  7.59 MiB/s][A

Extraction completed...: 0 file [00:19, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:19<00:06,  7.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:19<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:19<00:06,  7.54 MiB/s][A

Extraction completed...: 0 file [00:19, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:19<00:06,  7.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:19<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:19<00:06,  7.47 MiB/s][A

Extraction completed...: 0 file [00:19, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:20<00:06,  7.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:20<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:20<00:06,  7.44 MiB/s][A

Extraction completed...: 0 file [00:20, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:20<00:05,  7.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:20<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:20<00:05,  7.43 MiB/s][A

Extraction completed...: 0 file [00:20, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:20<00:05,  7.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:20<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:20<00:05,  7.43 MiB/s][A

Extraction completed...: 0 file [00:20, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:20<00:05,  7.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:20<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:20<00:05,  7.41 MiB/s][A

Extraction completed...: 0 file [00:20, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:20<00:05,  7.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:20<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:20<00:05,  7.43 MiB/s][A

Extraction completed...: 0 file [00:20, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:20<00:05,  7.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:20<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:20<00:05,  7.27 MiB/s][A

Extraction completed...: 0 file [00:20, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:20<00:05,  7.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:20<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:20<00:05,  7.05 MiB/s][A

Extraction completed...: 0 file [00:20, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:21<00:05,  6.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:21<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:21<00:05,  6.89 MiB/s][A

Extraction completed...: 0 file [00:21, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:21<00:05,  6.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:21<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:21<00:05,  6.80 MiB/s][A

Extraction completed...: 0 file [00:21, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:21<00:05,  6.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:21<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:21<00:05,  6.83 MiB/s][A

Extraction completed...: 0 file [00:21, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:21<00:05,  6.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:21<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:21<00:05,  6.89 MiB/s][A

Extraction completed...: 0 file [00:21, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:21<00:04,  6.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:21<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:21<00:04,  6.86 MiB/s][A

Extraction completed...: 0 file [00:21, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:21<00:04,  6.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:21<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:21<00:04,  6.85 MiB/s][A

Extraction completed...: 0 file [00:21, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:21<00:04,  6.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:21<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:21<00:04,  6.88 MiB/s][A

Extraction completed...: 0 file [00:21, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:22<00:04,  6.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:22<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:22<00:04,  6.79 MiB/s][A

Extraction completed...: 0 file [00:22, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:22<00:04,  6.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:22<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:22<00:04,  6.80 MiB/s][A

Extraction completed...: 0 file [00:22, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:22<00:04,  6.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:22<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:22<00:04,  6.86 MiB/s][A

Extraction completed...: 0 file [00:22, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:22<00:04,  6.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:22<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:22<00:04,  6.91 MiB/s][A

Extraction completed...: 0 file [00:22, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:22<00:03,  7.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:22<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:22<00:03,  7.07 MiB/s][A

Extraction completed...: 0 file [00:22, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:22<00:03,  7.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:22<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:22<00:03,  7.15 MiB/s][A

Extraction completed...: 0 file [00:22, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:22<00:03,  7.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:22<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:22<00:03,  7.24 MiB/s][A

Extraction completed...: 0 file [00:22, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:23<00:03,  7.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:23<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:23<00:03,  7.31 MiB/s][A

Extraction completed...: 0 file [00:23, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:23<00:03,  7.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:23<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:23<00:03,  7.30 MiB/s][A

Extraction completed...: 0 file [00:23, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:23<00:03,  7.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:23<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:23<00:03,  7.33 MiB/s][A

Extraction completed...: 0 file [00:23, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:23<00:02,  7.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:23<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:23<00:02,  7.36 MiB/s][A

Extraction completed...: 0 file [00:23, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:23<00:02,  7.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:23<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:23<00:02,  7.35 MiB/s][A

Extraction completed...: 0 file [00:23, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:23<00:02,  7.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:23<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:23<00:02,  7.38 MiB/s][A

Extraction completed...: 0 file [00:23, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:23<00:02,  7.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:23<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:23<00:02,  7.41 MiB/s][A

Extraction completed...: 0 file [00:23, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:23<00:02,  7.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:23<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:23<00:02,  7.40 MiB/s][A

Extraction completed...: 0 file [00:23, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:24<00:02,  7.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:24<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:24<00:02,  7.33 MiB/s][A

Extraction completed...: 0 file [00:24, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:24<00:02,  7.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:24<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:24<00:02,  7.20 MiB/s][A

Extraction completed...: 0 file [00:24, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:24<00:01,  7.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:24<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:24<00:01,  7.13 MiB/s][A

Extraction completed...: 0 file [00:24, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:24<00:01,  6.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:24<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:24<00:01,  6.97 MiB/s][A

Extraction completed...: 0 file [00:24, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:24<00:01,  6.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:24<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:24<00:01,  6.83 MiB/s][A

Extraction completed...: 0 file [00:24, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:24<00:01,  6.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:24<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:24<00:01,  6.76 MiB/s][A

Extraction completed...: 0 file [00:24, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:25<00:01,  6.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:25<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:25<00:01,  6.71 MiB/s][A

Extraction completed...: 0 file [00:25, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:25<00:01,  6.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:25<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:25<00:01,  6.64 MiB/s][A

Extraction completed...: 0 file [00:25, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:25<00:01,  6.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:25<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:25<00:01,  6.52 MiB/s][A

Extraction completed...: 0 file [00:25, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:25<00:01,  6.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:25<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:25<00:01,  6.43 MiB/s][A

Extraction completed...: 0 file [00:25, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:25<00:00,  6.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:25<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:25<00:00,  6.35 MiB/s][A

Extraction completed...: 0 file [00:25, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:25<00:00,  6.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:25<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:25<00:00,  6.31 MiB/s][A

Extraction completed...: 0 file [00:25, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:25<00:00,  6.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:25<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:25<00:00,  6.28 MiB/s][A

Extraction completed...: 0 file [00:25, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:26<00:00,  6.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:26<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:26<00:00,  6.23 MiB/s][A

Extraction completed...: 0 file [00:26, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:26<00:00,  6.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:26<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:26<00:00,  6.23 MiB/s][A

Extraction completed...: 0 file [00:26, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:26<00:00,  6.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:26<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:26<00:00,  6.20 MiB/s][A

Extraction completed...: 0 file [00:26, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:26<00:00,  6.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:26<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:26<00:00,  6.21 MiB/s][A

Extraction completed...: 0 file [00:26, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:26<00:00, 26.65s/ url]Dl Completed...: 100%|██████████| 1/1 [00:26<00:00, 26.65s/ url]
Dl Size...: 100%|██████████| 162/162 [00:26<00:00,  6.21 MiB/s][A

Extraction completed...: 0 file [00:26, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:26<00:00, 26.65s/ url]
Dl Size...: 100%|██████████| 162/162 [00:26<00:00,  6.21 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:26<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:28<00:00, 28.27s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:28<00:00, 26.65s/ url]
Dl Size...: 100%|██████████| 162/162 [00:28<00:00,  6.21 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:28<00:00, 28.27s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:28<00:00, 28.27s/ file]
Dl Size...: 100%|██████████| 162/162 [00:28<00:00,  5.73 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:28<00:00, 28.27s/ url]
0 examples [00:00, ? examples/s]2020-08-08 18:06:32.793605: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-08 18:06:32.805004: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-08-08 18:06:32.805166: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e7cdc6dc30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-08 18:06:32.805184: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
59 examples [00:00, 586.63 examples/s]198 examples [00:00, 709.01 examples/s]333 examples [00:00, 826.25 examples/s]465 examples [00:00, 930.61 examples/s]605 examples [00:00, 1033.06 examples/s]748 examples [00:00, 1126.02 examples/s]878 examples [00:00, 1171.20 examples/s]1000 examples [00:00, 1179.04 examples/s]1128 examples [00:00, 1205.63 examples/s]1269 examples [00:01, 1258.91 examples/s]1408 examples [00:01, 1294.79 examples/s]1550 examples [00:01, 1329.22 examples/s]1685 examples [00:01, 1329.90 examples/s]1820 examples [00:01, 1310.22 examples/s]1952 examples [00:01, 1297.33 examples/s]2095 examples [00:01, 1334.22 examples/s]2237 examples [00:01, 1358.62 examples/s]2377 examples [00:01, 1369.21 examples/s]2515 examples [00:01, 1357.09 examples/s]2655 examples [00:02, 1368.52 examples/s]2795 examples [00:02, 1376.15 examples/s]2933 examples [00:02, 1333.12 examples/s]3067 examples [00:02, 1307.34 examples/s]3199 examples [00:02, 1299.27 examples/s]3330 examples [00:02, 1262.84 examples/s]3460 examples [00:02, 1273.16 examples/s]3596 examples [00:02, 1297.98 examples/s]3730 examples [00:02, 1308.43 examples/s]3862 examples [00:02, 1302.83 examples/s]4002 examples [00:03, 1329.61 examples/s]4139 examples [00:03, 1338.34 examples/s]4274 examples [00:03, 1336.57 examples/s]4410 examples [00:03, 1343.12 examples/s]4545 examples [00:03, 1335.04 examples/s]4679 examples [00:03, 1329.38 examples/s]4819 examples [00:03, 1348.84 examples/s]4958 examples [00:03, 1360.30 examples/s]5097 examples [00:03, 1367.46 examples/s]5234 examples [00:03, 1293.47 examples/s]5365 examples [00:04, 1289.83 examples/s]5495 examples [00:04, 1278.76 examples/s]5625 examples [00:04, 1284.94 examples/s]5758 examples [00:04, 1297.50 examples/s]5897 examples [00:04, 1323.31 examples/s]6030 examples [00:04, 1309.13 examples/s]6162 examples [00:04, 1311.25 examples/s]6305 examples [00:04, 1342.41 examples/s]6440 examples [00:04, 1299.39 examples/s]6571 examples [00:05, 1301.22 examples/s]6703 examples [00:05, 1306.70 examples/s]6840 examples [00:05, 1323.58 examples/s]6975 examples [00:05, 1331.06 examples/s]7113 examples [00:05, 1343.33 examples/s]7249 examples [00:05, 1345.43 examples/s]7389 examples [00:05, 1359.43 examples/s]7526 examples [00:05, 1358.40 examples/s]7666 examples [00:05, 1367.89 examples/s]7803 examples [00:05, 1317.06 examples/s]7937 examples [00:06, 1323.13 examples/s]8080 examples [00:06, 1351.36 examples/s]8222 examples [00:06, 1369.34 examples/s]8361 examples [00:06, 1373.99 examples/s]8502 examples [00:06, 1382.69 examples/s]8641 examples [00:06, 1371.82 examples/s]8786 examples [00:06, 1393.50 examples/s]8927 examples [00:06, 1395.81 examples/s]9067 examples [00:06, 1391.18 examples/s]9213 examples [00:06, 1407.99 examples/s]9354 examples [00:07, 1378.31 examples/s]9493 examples [00:07, 1366.51 examples/s]9630 examples [00:07, 1332.97 examples/s]9764 examples [00:07, 1332.04 examples/s]9898 examples [00:07, 1304.05 examples/s]10029 examples [00:07, 1247.52 examples/s]10166 examples [00:07, 1281.38 examples/s]10295 examples [00:07, 1283.13 examples/s]10431 examples [00:07, 1303.45 examples/s]10562 examples [00:07, 1299.61 examples/s]10697 examples [00:08, 1312.52 examples/s]10837 examples [00:08, 1336.94 examples/s]10977 examples [00:08, 1354.98 examples/s]11113 examples [00:08, 1339.22 examples/s]11248 examples [00:08, 1338.15 examples/s]11387 examples [00:08, 1352.20 examples/s]11523 examples [00:08, 1314.16 examples/s]11656 examples [00:08, 1316.40 examples/s]11798 examples [00:08, 1343.56 examples/s]11933 examples [00:09, 1301.82 examples/s]12069 examples [00:09, 1317.01 examples/s]12202 examples [00:09, 1301.76 examples/s]12338 examples [00:09, 1318.51 examples/s]12482 examples [00:09, 1351.14 examples/s]12619 examples [00:09, 1356.60 examples/s]12758 examples [00:09, 1363.45 examples/s]12901 examples [00:09, 1382.64 examples/s]13044 examples [00:09, 1396.12 examples/s]13185 examples [00:09, 1398.18 examples/s]13325 examples [00:10, 1394.33 examples/s]13466 examples [00:10, 1396.97 examples/s]13607 examples [00:10, 1399.59 examples/s]13748 examples [00:10, 1383.19 examples/s]13887 examples [00:10, 1352.61 examples/s]14023 examples [00:10, 1327.81 examples/s]14159 examples [00:10, 1335.21 examples/s]14295 examples [00:10, 1341.85 examples/s]14430 examples [00:10, 1281.62 examples/s]14559 examples [00:10, 1272.11 examples/s]14695 examples [00:11, 1296.30 examples/s]14834 examples [00:11, 1321.80 examples/s]14978 examples [00:11, 1353.75 examples/s]15114 examples [00:11, 1347.84 examples/s]15258 examples [00:11, 1372.27 examples/s]15396 examples [00:11, 1368.49 examples/s]15534 examples [00:11, 1369.13 examples/s]15672 examples [00:11, 1364.53 examples/s]15812 examples [00:11, 1374.57 examples/s]15953 examples [00:11, 1380.55 examples/s]16092 examples [00:12, 1369.88 examples/s]16230 examples [00:12, 1363.00 examples/s]16369 examples [00:12, 1368.38 examples/s]16508 examples [00:12, 1374.39 examples/s]16646 examples [00:12, 1355.36 examples/s]16782 examples [00:12, 1328.08 examples/s]16919 examples [00:12, 1337.99 examples/s]17063 examples [00:12, 1364.90 examples/s]17201 examples [00:12, 1365.31 examples/s]17338 examples [00:12, 1351.53 examples/s]17474 examples [00:13, 1349.79 examples/s]17610 examples [00:13, 1322.47 examples/s]17745 examples [00:13, 1328.64 examples/s]17886 examples [00:13, 1351.54 examples/s]18029 examples [00:13, 1373.15 examples/s]18167 examples [00:13, 1341.42 examples/s]18302 examples [00:13, 1297.92 examples/s]18433 examples [00:13, 1269.44 examples/s]18568 examples [00:13, 1291.63 examples/s]18698 examples [00:14, 1261.99 examples/s]18825 examples [00:14, 1258.19 examples/s]18960 examples [00:14, 1282.36 examples/s]19092 examples [00:14, 1290.18 examples/s]19222 examples [00:14, 1286.17 examples/s]19361 examples [00:14, 1314.59 examples/s]19496 examples [00:14, 1324.78 examples/s]19632 examples [00:14, 1334.46 examples/s]19767 examples [00:14, 1336.30 examples/s]19909 examples [00:14, 1359.27 examples/s]20046 examples [00:15, 1294.95 examples/s]20177 examples [00:15, 1280.72 examples/s]20314 examples [00:15, 1303.84 examples/s]20445 examples [00:15, 1299.28 examples/s]20589 examples [00:15, 1335.99 examples/s]20733 examples [00:15, 1363.67 examples/s]20870 examples [00:15, 1354.34 examples/s]21006 examples [00:15, 1344.69 examples/s]21141 examples [00:15, 1338.37 examples/s]21276 examples [00:15, 1340.65 examples/s]21413 examples [00:16, 1348.83 examples/s]21549 examples [00:16, 1350.83 examples/s]21685 examples [00:16, 1345.08 examples/s]21826 examples [00:16, 1362.51 examples/s]21968 examples [00:16, 1377.40 examples/s]22110 examples [00:16, 1389.46 examples/s]22250 examples [00:16, 1377.58 examples/s]22391 examples [00:16, 1386.74 examples/s]22530 examples [00:16, 1379.74 examples/s]22669 examples [00:16, 1350.09 examples/s]22805 examples [00:17, 1346.36 examples/s]22940 examples [00:17, 1315.56 examples/s]23072 examples [00:17, 1282.03 examples/s]23213 examples [00:17, 1316.28 examples/s]23353 examples [00:17, 1333.18 examples/s]23487 examples [00:17, 1318.54 examples/s]23620 examples [00:17, 1318.61 examples/s]23756 examples [00:17, 1329.41 examples/s]23896 examples [00:17, 1347.71 examples/s]24036 examples [00:18, 1361.66 examples/s]24173 examples [00:18, 1361.45 examples/s]24310 examples [00:18, 1292.78 examples/s]24448 examples [00:18, 1315.24 examples/s]24588 examples [00:18, 1337.49 examples/s]24723 examples [00:18, 1305.39 examples/s]24864 examples [00:18, 1333.32 examples/s]24998 examples [00:18, 1327.15 examples/s]25133 examples [00:18, 1331.25 examples/s]25267 examples [00:18, 1324.91 examples/s]25406 examples [00:19, 1342.89 examples/s]25542 examples [00:19, 1346.72 examples/s]25677 examples [00:19, 1318.61 examples/s]25818 examples [00:19, 1343.95 examples/s]25961 examples [00:19, 1367.52 examples/s]26104 examples [00:19, 1383.24 examples/s]26247 examples [00:19, 1395.89 examples/s]26387 examples [00:19, 1379.10 examples/s]26532 examples [00:19, 1392.51 examples/s]26672 examples [00:19, 1393.87 examples/s]26815 examples [00:20, 1401.24 examples/s]26956 examples [00:20, 1375.75 examples/s]27094 examples [00:20, 1314.86 examples/s]27227 examples [00:20, 1278.43 examples/s]27366 examples [00:20, 1309.51 examples/s]27501 examples [00:20, 1320.93 examples/s]27637 examples [00:20, 1330.51 examples/s]27771 examples [00:20, 1279.62 examples/s]27900 examples [00:20, 1281.48 examples/s]28041 examples [00:21, 1317.42 examples/s]28183 examples [00:21, 1344.74 examples/s]28323 examples [00:21, 1360.28 examples/s]28460 examples [00:21, 1353.26 examples/s]28603 examples [00:21, 1374.54 examples/s]28745 examples [00:21, 1387.73 examples/s]28885 examples [00:21, 1387.72 examples/s]29026 examples [00:21, 1393.84 examples/s]29166 examples [00:21, 1388.21 examples/s]29311 examples [00:21, 1403.72 examples/s]29454 examples [00:22, 1411.01 examples/s]29596 examples [00:22, 1410.53 examples/s]29739 examples [00:22, 1412.81 examples/s]29881 examples [00:22, 1384.03 examples/s]30020 examples [00:22, 1296.72 examples/s]30152 examples [00:22, 1303.43 examples/s]30292 examples [00:22, 1329.46 examples/s]30427 examples [00:22, 1333.98 examples/s]30562 examples [00:22, 1336.56 examples/s]30700 examples [00:22, 1348.73 examples/s]30842 examples [00:23, 1366.95 examples/s]30983 examples [00:23, 1379.20 examples/s]31122 examples [00:23, 1372.87 examples/s]31260 examples [00:23, 1363.79 examples/s]31397 examples [00:23, 1303.80 examples/s]31530 examples [00:23, 1308.53 examples/s]31663 examples [00:23, 1312.95 examples/s]31795 examples [00:23, 1293.10 examples/s]31928 examples [00:23, 1302.37 examples/s]32063 examples [00:23, 1316.08 examples/s]32195 examples [00:24, 1284.91 examples/s]32324 examples [00:24, 1266.80 examples/s]32457 examples [00:24, 1283.19 examples/s]32595 examples [00:24, 1310.32 examples/s]32735 examples [00:24, 1335.96 examples/s]32869 examples [00:24, 1300.79 examples/s]33009 examples [00:24, 1328.30 examples/s]33143 examples [00:24, 1323.58 examples/s]33281 examples [00:24, 1337.74 examples/s]33416 examples [00:25, 1339.58 examples/s]33555 examples [00:25, 1354.23 examples/s]33691 examples [00:25, 1348.40 examples/s]33826 examples [00:25, 1346.50 examples/s]33961 examples [00:25, 1346.33 examples/s]34102 examples [00:25, 1362.93 examples/s]34244 examples [00:25, 1377.66 examples/s]34382 examples [00:25, 1368.08 examples/s]34519 examples [00:25, 1320.48 examples/s]34653 examples [00:25, 1323.34 examples/s]34794 examples [00:26, 1347.04 examples/s]34938 examples [00:26, 1372.79 examples/s]35079 examples [00:26, 1382.96 examples/s]35218 examples [00:26, 1365.89 examples/s]35355 examples [00:26, 1343.86 examples/s]35490 examples [00:26, 1322.66 examples/s]35623 examples [00:26, 1307.11 examples/s]35754 examples [00:26, 1280.99 examples/s]35886 examples [00:26, 1290.27 examples/s]36023 examples [00:26, 1313.18 examples/s]36162 examples [00:27, 1334.47 examples/s]36300 examples [00:27, 1347.50 examples/s]36435 examples [00:27, 1322.43 examples/s]36568 examples [00:27, 1290.61 examples/s]36698 examples [00:27, 1278.64 examples/s]36828 examples [00:27, 1283.63 examples/s]36971 examples [00:27, 1322.72 examples/s]37111 examples [00:27, 1343.85 examples/s]37249 examples [00:27, 1354.02 examples/s]37392 examples [00:27, 1373.80 examples/s]37531 examples [00:28, 1376.04 examples/s]37674 examples [00:28, 1389.73 examples/s]37816 examples [00:28, 1395.68 examples/s]37956 examples [00:28, 1385.39 examples/s]38095 examples [00:28, 1380.95 examples/s]38234 examples [00:28, 1382.82 examples/s]38373 examples [00:28, 1364.35 examples/s]38514 examples [00:28, 1376.56 examples/s]38652 examples [00:28, 1335.98 examples/s]38796 examples [00:29, 1362.63 examples/s]38933 examples [00:29, 1336.35 examples/s]39067 examples [00:29, 1307.19 examples/s]39205 examples [00:29, 1327.23 examples/s]39339 examples [00:29, 1330.59 examples/s]39479 examples [00:29, 1348.25 examples/s]39615 examples [00:29, 1316.41 examples/s]39747 examples [00:29, 1306.82 examples/s]39878 examples [00:29, 1295.53 examples/s]40008 examples [00:29, 1215.63 examples/s]40135 examples [00:30, 1230.41 examples/s]40264 examples [00:30, 1247.05 examples/s]40397 examples [00:30, 1270.18 examples/s]40525 examples [00:30, 1270.62 examples/s]40656 examples [00:30, 1281.65 examples/s]40787 examples [00:30, 1289.33 examples/s]40921 examples [00:30, 1303.44 examples/s]41052 examples [00:30, 1272.56 examples/s]41180 examples [00:30, 1256.05 examples/s]41314 examples [00:30, 1277.45 examples/s]41452 examples [00:31, 1305.38 examples/s]41593 examples [00:31, 1333.40 examples/s]41727 examples [00:31, 1325.28 examples/s]41861 examples [00:31, 1329.30 examples/s]41998 examples [00:31, 1340.87 examples/s]42138 examples [00:31, 1356.54 examples/s]42274 examples [00:31, 1303.37 examples/s]42405 examples [00:31, 1291.48 examples/s]42540 examples [00:31, 1306.06 examples/s]42677 examples [00:32, 1324.08 examples/s]42812 examples [00:32, 1329.49 examples/s]42953 examples [00:32, 1350.03 examples/s]43089 examples [00:32, 1341.09 examples/s]43224 examples [00:32, 1334.27 examples/s]43358 examples [00:32, 1284.84 examples/s]43487 examples [00:32, 1264.27 examples/s]43616 examples [00:32, 1270.77 examples/s]43745 examples [00:32, 1273.84 examples/s]43873 examples [00:32, 1252.09 examples/s]44011 examples [00:33, 1286.95 examples/s]44141 examples [00:33, 1269.05 examples/s]44270 examples [00:33, 1274.92 examples/s]44407 examples [00:33, 1300.81 examples/s]44544 examples [00:33, 1319.12 examples/s]44685 examples [00:33, 1343.44 examples/s]44829 examples [00:33, 1368.75 examples/s]44971 examples [00:33, 1383.35 examples/s]45110 examples [00:33, 1384.04 examples/s]45249 examples [00:33, 1382.56 examples/s]45388 examples [00:34, 1351.67 examples/s]45524 examples [00:34, 1346.38 examples/s]45659 examples [00:34, 1341.86 examples/s]45794 examples [00:34, 1312.25 examples/s]45929 examples [00:34, 1320.27 examples/s]46072 examples [00:34, 1349.86 examples/s]46214 examples [00:34, 1368.46 examples/s]46355 examples [00:34, 1380.53 examples/s]46496 examples [00:34, 1386.85 examples/s]46635 examples [00:34, 1350.90 examples/s]46774 examples [00:35, 1359.25 examples/s]46911 examples [00:35, 1321.48 examples/s]47047 examples [00:35, 1332.11 examples/s]47188 examples [00:35, 1353.14 examples/s]47324 examples [00:35, 1342.33 examples/s]47460 examples [00:35, 1346.76 examples/s]47597 examples [00:35, 1352.83 examples/s]47733 examples [00:35, 1349.41 examples/s]47869 examples [00:35, 1312.51 examples/s]48001 examples [00:36, 1286.29 examples/s]48131 examples [00:36, 1286.80 examples/s]48267 examples [00:36, 1307.09 examples/s]48406 examples [00:36, 1330.82 examples/s]48540 examples [00:36, 1312.81 examples/s]48672 examples [00:36, 1296.34 examples/s]48806 examples [00:36, 1308.17 examples/s]48937 examples [00:36, 1293.62 examples/s]49067 examples [00:36, 1269.61 examples/s]49195 examples [00:36, 1255.06 examples/s]49321 examples [00:37, 1244.31 examples/s]49446 examples [00:37, 1237.69 examples/s]49586 examples [00:37, 1281.58 examples/s]49715 examples [00:37, 1253.74 examples/s]49853 examples [00:37, 1286.36 examples/s]49983 examples [00:37, 1262.78 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 22%|██▏       | 10800/50000 [00:00<00:00, 107999.07 examples/s] 56%|█████▌    | 28034/50000 [00:00<00:00, 121619.65 examples/s] 91%|█████████ | 45510/50000 [00:00<00:00, 133825.78 examples/s]                                                                0 examples [00:00, ? examples/s]110 examples [00:00, 1091.92 examples/s]238 examples [00:00, 1142.15 examples/s]378 examples [00:00, 1208.41 examples/s]514 examples [00:00, 1247.77 examples/s]654 examples [00:00, 1288.10 examples/s]796 examples [00:00, 1322.56 examples/s]924 examples [00:00, 1307.10 examples/s]1061 examples [00:00, 1323.69 examples/s]1191 examples [00:00, 1313.55 examples/s]1336 examples [00:01, 1350.79 examples/s]1469 examples [00:01, 1332.81 examples/s]1601 examples [00:01, 1294.25 examples/s]1730 examples [00:01, 1278.03 examples/s]1863 examples [00:01, 1293.03 examples/s]1997 examples [00:01, 1306.63 examples/s]2128 examples [00:01, 1300.43 examples/s]2262 examples [00:01, 1309.77 examples/s]2393 examples [00:01, 1300.08 examples/s]2528 examples [00:01, 1313.62 examples/s]2671 examples [00:02, 1345.55 examples/s]2812 examples [00:02, 1360.86 examples/s]2949 examples [00:02, 1353.33 examples/s]3092 examples [00:02, 1375.04 examples/s]3234 examples [00:02, 1387.10 examples/s]3374 examples [00:02, 1388.83 examples/s]3513 examples [00:02, 1371.72 examples/s]3653 examples [00:02, 1378.70 examples/s]3791 examples [00:02, 1367.76 examples/s]3928 examples [00:02, 1311.30 examples/s]4070 examples [00:03, 1341.40 examples/s]4208 examples [00:03, 1351.99 examples/s]4350 examples [00:03, 1370.05 examples/s]4496 examples [00:03, 1393.81 examples/s]4637 examples [00:03, 1396.42 examples/s]4781 examples [00:03, 1409.19 examples/s]4923 examples [00:03, 1406.01 examples/s]5064 examples [00:03, 1394.04 examples/s]5204 examples [00:03, 1386.12 examples/s]5345 examples [00:03, 1391.95 examples/s]5487 examples [00:04, 1400.23 examples/s]5628 examples [00:04, 1397.86 examples/s]5768 examples [00:04, 1392.61 examples/s]5908 examples [00:04, 1331.09 examples/s]6042 examples [00:04, 1313.39 examples/s]6174 examples [00:04, 1302.56 examples/s]6305 examples [00:04, 1296.70 examples/s]6445 examples [00:04, 1325.80 examples/s]6586 examples [00:04, 1345.80 examples/s]6721 examples [00:04, 1342.13 examples/s]6856 examples [00:05, 1332.20 examples/s]6990 examples [00:05, 1331.92 examples/s]7131 examples [00:05, 1352.65 examples/s]7267 examples [00:05, 1325.24 examples/s]7405 examples [00:05, 1340.97 examples/s]7552 examples [00:05, 1374.62 examples/s]7690 examples [00:05, 1339.23 examples/s]7830 examples [00:05, 1355.27 examples/s]7970 examples [00:05, 1365.32 examples/s]8107 examples [00:06, 1355.39 examples/s]8253 examples [00:06, 1379.93 examples/s]8392 examples [00:06, 1291.06 examples/s]8531 examples [00:06, 1318.52 examples/s]8664 examples [00:06, 1315.15 examples/s]8800 examples [00:06, 1326.81 examples/s]8944 examples [00:06, 1358.14 examples/s]9081 examples [00:06, 1257.18 examples/s]9215 examples [00:06, 1279.65 examples/s]9347 examples [00:06, 1291.07 examples/s]9484 examples [00:07, 1313.11 examples/s]9628 examples [00:07, 1347.20 examples/s]9764 examples [00:07, 1334.53 examples/s]9899 examples [00:07, 1328.67 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteIZDPNH/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteIZDPNH/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8fb4fa6158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8fb4fa6158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f8fb4fa6158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f90009830b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8f7400d438>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8fb4fa6158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8fb4fa6158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f8fb4fa6158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8fb33a6438>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8fb33a64a8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 10.63 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 13.11 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 13.11 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.11 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.87 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.87 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.20 file/s]2020-08-08 18:07:24.632481: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-08 18:07:24.635891: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-08-08 18:07:24.636024: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55740ae0fc10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-08 18:07:24.636036: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:05, 151611.61it/s] 78%|███████▊  | 7749632/9912422 [00:00<00:09, 216404.30it/s]9920512it [00:00, 44486210.51it/s]                           
0it [00:00, ?it/s]32768it [00:00, 529944.37it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160515.27it/s]1654784it [00:00, 11040098.67it/s]                         
0it [00:00, ?it/s]8192it [00:00, 208386.08it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

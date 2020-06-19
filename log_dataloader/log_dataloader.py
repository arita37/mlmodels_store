
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'bd7dfc233939710e05244f8e6a394a7ce12a3485', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fe019238ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fe019238ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe0847ef048> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe0847ef048> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe09e51de18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe09e51de18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe03206e8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe03206e8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe03206e8c8> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 428, in load
    fid = open(os_fspath(file), "rb")
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 33%|███▎      | 3276800/9912422 [00:00<00:00, 32764359.78it/s]9920512it [00:00, 35444467.87it/s]                             
0it [00:00, ?it/s]32768it [00:00, 739642.84it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1032386.79it/s]1654784it [00:00, 12523171.32it/s]                           
0it [00:00, ?it/s]8192it [00:00, 261826.39it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe02f5db550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe02f6a67f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe03206e510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe03206e510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe03206e510> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:06,  2.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:06,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:06,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:05,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:05,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:04,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:04,  2.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:45,  3.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:45,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:45,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:45,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:44,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:44,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:44,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:43,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:43,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:30,  4.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:30,  4.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:30,  4.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:30,  4.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:30,  4.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:30,  4.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:29,  4.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:29,  4.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:29,  4.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:21,  6.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:21,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:20,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:20,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:20,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:20,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:20,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:20,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:19,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:14,  9.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:14,  9.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:14,  9.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:14,  9.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  9.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:13,  9.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:13,  9.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:13,  9.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:13,  9.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:09, 12.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 16.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 16.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 16.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 16.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 16.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 16.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 16.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 16.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 16.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 21.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 21.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 21.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 21.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 21.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 21.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 21.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 21.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 27.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 27.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 27.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 27.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 27.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 27.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 27.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 27.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 32.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 32.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 32.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 32.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 32.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 32.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 32.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 32.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 32.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 39.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 39.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 39.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 39.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 39.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 39.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 39.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 39.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 39.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 45.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 45.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 45.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 45.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 45.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 45.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 45.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 45.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 45.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 56.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 56.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 61.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 64.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 64.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 64.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 64.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 64.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 64.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 64.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 65.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 65.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 65.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 64.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 64.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 64.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 64.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 67.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 68.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 70.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 70.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 70.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 70.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 70.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 70.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.67s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.67s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.67s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.01 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.67s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 70.01 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 32.18 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.04s/ url]
0 examples [00:00, ? examples/s]2020-06-19 06:09:18.898071: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-19 06:09:18.903345: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-19 06:09:18.903490: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e4f3868950 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-19 06:09:18.903525: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
58 examples [00:00, 571.93 examples/s]148 examples [00:00, 640.94 examples/s]237 examples [00:00, 698.67 examples/s]321 examples [00:00, 735.18 examples/s]408 examples [00:00, 769.64 examples/s]501 examples [00:00, 811.42 examples/s]591 examples [00:00, 834.41 examples/s]677 examples [00:00, 841.52 examples/s]764 examples [00:00, 846.84 examples/s]848 examples [00:01, 843.68 examples/s]942 examples [00:01, 868.13 examples/s]1033 examples [00:01, 880.24 examples/s]1121 examples [00:01, 877.21 examples/s]1209 examples [00:01, 872.99 examples/s]1297 examples [00:01, 864.44 examples/s]1388 examples [00:01, 875.79 examples/s]1476 examples [00:01, 876.57 examples/s]1565 examples [00:01, 878.65 examples/s]1656 examples [00:01, 886.75 examples/s]1745 examples [00:02, 861.62 examples/s]1838 examples [00:02, 880.69 examples/s]1927 examples [00:02, 878.87 examples/s]2016 examples [00:02, 879.93 examples/s]2107 examples [00:02, 888.35 examples/s]2197 examples [00:02, 888.16 examples/s]2286 examples [00:02, 881.69 examples/s]2375 examples [00:02, 861.92 examples/s]2463 examples [00:02, 863.91 examples/s]2553 examples [00:02, 871.77 examples/s]2641 examples [00:03, 872.24 examples/s]2729 examples [00:03, 833.51 examples/s]2817 examples [00:03, 846.02 examples/s]2902 examples [00:03, 844.74 examples/s]2987 examples [00:03, 842.20 examples/s]3076 examples [00:03, 853.20 examples/s]3167 examples [00:03, 867.25 examples/s]3254 examples [00:03, 855.12 examples/s]3340 examples [00:03, 846.30 examples/s]3426 examples [00:03, 848.48 examples/s]3513 examples [00:04, 852.77 examples/s]3599 examples [00:04, 853.92 examples/s]3688 examples [00:04, 861.85 examples/s]3776 examples [00:04, 866.96 examples/s]3868 examples [00:04, 881.52 examples/s]3962 examples [00:04, 896.94 examples/s]4052 examples [00:04, 871.67 examples/s]4141 examples [00:04, 875.36 examples/s]4229 examples [00:04, 856.28 examples/s]4316 examples [00:04, 859.24 examples/s]4403 examples [00:05, 851.99 examples/s]4489 examples [00:05, 846.60 examples/s]4574 examples [00:05, 843.34 examples/s]4659 examples [00:05, 839.85 examples/s]4751 examples [00:05, 859.67 examples/s]4842 examples [00:05, 872.52 examples/s]4934 examples [00:05, 884.69 examples/s]5030 examples [00:05, 904.59 examples/s]5124 examples [00:05, 914.37 examples/s]5216 examples [00:06, 900.39 examples/s]5307 examples [00:06, 892.32 examples/s]5397 examples [00:06, 842.30 examples/s]5487 examples [00:06, 858.71 examples/s]5578 examples [00:06, 872.07 examples/s]5670 examples [00:06, 885.51 examples/s]5759 examples [00:06, 879.46 examples/s]5848 examples [00:06, 860.33 examples/s]5935 examples [00:06, 852.06 examples/s]6021 examples [00:06, 842.41 examples/s]6108 examples [00:07, 848.32 examples/s]6200 examples [00:07, 866.02 examples/s]6289 examples [00:07, 871.87 examples/s]6378 examples [00:07, 876.84 examples/s]6472 examples [00:07, 894.18 examples/s]6564 examples [00:07, 901.29 examples/s]6655 examples [00:07, 902.27 examples/s]6746 examples [00:07, 895.95 examples/s]6836 examples [00:07, 884.25 examples/s]6925 examples [00:07, 863.23 examples/s]7012 examples [00:08, 831.42 examples/s]7101 examples [00:08, 845.41 examples/s]7188 examples [00:08, 850.27 examples/s]7277 examples [00:08, 861.04 examples/s]7364 examples [00:08, 862.77 examples/s]7451 examples [00:08, 855.29 examples/s]7537 examples [00:08, 834.17 examples/s]7621 examples [00:08, 832.78 examples/s]7707 examples [00:08, 839.87 examples/s]7794 examples [00:09, 847.42 examples/s]7879 examples [00:09, 848.04 examples/s]7969 examples [00:09, 862.48 examples/s]8056 examples [00:09, 841.93 examples/s]8141 examples [00:09, 839.30 examples/s]8230 examples [00:09, 852.39 examples/s]8319 examples [00:09, 862.15 examples/s]8406 examples [00:09, 863.79 examples/s]8493 examples [00:09, 861.14 examples/s]8580 examples [00:09, 859.81 examples/s]8668 examples [00:10, 863.51 examples/s]8756 examples [00:10, 865.58 examples/s]8843 examples [00:10, 863.64 examples/s]8930 examples [00:10, 859.19 examples/s]9018 examples [00:10, 864.88 examples/s]9105 examples [00:10, 862.18 examples/s]9199 examples [00:10, 881.95 examples/s]9288 examples [00:10, 870.40 examples/s]9376 examples [00:10, 872.19 examples/s]9467 examples [00:10, 883.13 examples/s]9556 examples [00:11, 879.72 examples/s]9645 examples [00:11, 872.85 examples/s]9734 examples [00:11, 875.88 examples/s]9823 examples [00:11, 878.73 examples/s]9915 examples [00:11, 886.57 examples/s]10004 examples [00:11, 835.08 examples/s]10089 examples [00:11, 833.40 examples/s]10173 examples [00:11, 834.45 examples/s]10261 examples [00:11, 846.76 examples/s]10346 examples [00:11, 842.54 examples/s]10432 examples [00:12, 845.93 examples/s]10518 examples [00:12, 849.73 examples/s]10604 examples [00:12, 840.69 examples/s]10692 examples [00:12, 850.75 examples/s]10778 examples [00:12, 840.83 examples/s]10869 examples [00:12, 860.39 examples/s]10956 examples [00:12, 855.37 examples/s]11042 examples [00:12, 835.07 examples/s]11126 examples [00:12, 823.39 examples/s]11210 examples [00:13, 827.20 examples/s]11293 examples [00:13, 815.86 examples/s]11379 examples [00:13, 825.57 examples/s]11462 examples [00:13, 815.85 examples/s]11545 examples [00:13, 813.90 examples/s]11630 examples [00:13, 821.98 examples/s]11713 examples [00:13, 820.79 examples/s]11802 examples [00:13, 838.53 examples/s]11889 examples [00:13, 846.07 examples/s]11976 examples [00:13, 851.53 examples/s]12064 examples [00:14, 858.89 examples/s]12154 examples [00:14, 869.71 examples/s]12242 examples [00:14, 858.08 examples/s]12328 examples [00:14, 853.64 examples/s]12414 examples [00:14, 851.84 examples/s]12501 examples [00:14, 856.82 examples/s]12590 examples [00:14, 862.74 examples/s]12677 examples [00:14, 837.02 examples/s]12761 examples [00:14, 825.34 examples/s]12844 examples [00:14, 816.20 examples/s]12927 examples [00:15, 820.13 examples/s]13013 examples [00:15, 829.40 examples/s]13106 examples [00:15, 855.51 examples/s]13196 examples [00:15, 867.36 examples/s]13285 examples [00:15, 873.69 examples/s]13373 examples [00:15, 847.99 examples/s]13460 examples [00:15, 852.04 examples/s]13546 examples [00:15, 848.42 examples/s]13631 examples [00:15, 841.37 examples/s]13719 examples [00:15, 850.22 examples/s]13805 examples [00:16, 835.31 examples/s]13889 examples [00:16, 829.22 examples/s]13976 examples [00:16, 840.05 examples/s]14061 examples [00:16, 836.14 examples/s]14146 examples [00:16, 839.95 examples/s]14232 examples [00:16, 844.92 examples/s]14318 examples [00:16, 848.79 examples/s]14406 examples [00:16, 855.22 examples/s]14494 examples [00:16, 860.24 examples/s]14582 examples [00:16, 863.41 examples/s]14669 examples [00:17, 865.37 examples/s]14757 examples [00:17, 868.89 examples/s]14844 examples [00:17, 860.54 examples/s]14931 examples [00:17, 860.88 examples/s]15019 examples [00:17, 864.30 examples/s]15108 examples [00:17, 869.41 examples/s]15195 examples [00:17, 856.22 examples/s]15284 examples [00:17, 865.95 examples/s]15372 examples [00:17, 867.92 examples/s]15461 examples [00:18, 872.76 examples/s]15549 examples [00:18, 857.04 examples/s]15635 examples [00:18, 831.06 examples/s]15728 examples [00:18, 855.86 examples/s]15816 examples [00:18, 857.82 examples/s]15904 examples [00:18, 863.55 examples/s]15991 examples [00:18, 827.41 examples/s]16087 examples [00:18, 859.64 examples/s]16174 examples [00:18, 846.85 examples/s]16260 examples [00:18, 824.43 examples/s]16345 examples [00:19, 829.73 examples/s]16430 examples [00:19, 833.18 examples/s]16519 examples [00:19, 847.48 examples/s]16604 examples [00:19, 847.80 examples/s]16689 examples [00:19, 846.71 examples/s]16774 examples [00:19, 844.22 examples/s]16859 examples [00:19, 836.90 examples/s]16946 examples [00:19, 845.52 examples/s]17033 examples [00:19, 849.12 examples/s]17118 examples [00:19, 844.34 examples/s]17207 examples [00:20, 856.57 examples/s]17298 examples [00:20, 869.84 examples/s]17394 examples [00:20, 892.43 examples/s]17485 examples [00:20, 897.17 examples/s]17575 examples [00:20, 896.01 examples/s]17665 examples [00:20, 891.33 examples/s]17755 examples [00:20, 884.75 examples/s]17845 examples [00:20, 887.19 examples/s]17935 examples [00:20, 888.58 examples/s]18024 examples [00:21, 824.92 examples/s]18116 examples [00:21, 850.62 examples/s]18206 examples [00:21, 862.32 examples/s]18293 examples [00:21, 842.93 examples/s]18384 examples [00:21, 860.73 examples/s]18471 examples [00:21, 841.13 examples/s]18560 examples [00:21, 849.64 examples/s]18646 examples [00:21, 840.58 examples/s]18734 examples [00:21, 848.16 examples/s]18820 examples [00:21, 846.97 examples/s]18910 examples [00:22, 859.99 examples/s]18997 examples [00:22, 853.62 examples/s]19083 examples [00:22, 842.66 examples/s]19170 examples [00:22, 850.64 examples/s]19256 examples [00:22, 844.90 examples/s]19341 examples [00:22, 842.77 examples/s]19430 examples [00:22, 853.88 examples/s]19519 examples [00:22, 862.69 examples/s]19610 examples [00:22, 874.75 examples/s]19698 examples [00:22, 867.90 examples/s]19785 examples [00:23, 867.49 examples/s]19872 examples [00:23, 857.66 examples/s]19960 examples [00:23, 862.95 examples/s]20047 examples [00:23, 807.19 examples/s]20130 examples [00:23, 812.12 examples/s]20215 examples [00:23, 821.66 examples/s]20301 examples [00:23, 832.66 examples/s]20386 examples [00:23, 834.90 examples/s]20474 examples [00:23, 847.48 examples/s]20559 examples [00:24, 844.09 examples/s]20646 examples [00:24, 851.64 examples/s]20733 examples [00:24, 854.90 examples/s]20821 examples [00:24, 859.95 examples/s]20908 examples [00:24, 842.34 examples/s]20996 examples [00:24, 853.12 examples/s]21082 examples [00:24, 845.99 examples/s]21167 examples [00:24, 834.52 examples/s]21251 examples [00:24, 835.70 examples/s]21336 examples [00:24, 837.58 examples/s]21420 examples [00:25, 834.75 examples/s]21506 examples [00:25, 840.48 examples/s]21592 examples [00:25, 845.47 examples/s]21682 examples [00:25, 859.86 examples/s]21772 examples [00:25, 870.24 examples/s]21860 examples [00:25, 850.59 examples/s]21946 examples [00:25, 850.82 examples/s]22032 examples [00:25, 850.68 examples/s]22118 examples [00:25, 840.65 examples/s]22207 examples [00:25, 852.54 examples/s]22293 examples [00:26, 854.42 examples/s]22385 examples [00:26, 871.93 examples/s]22480 examples [00:26, 892.81 examples/s]22575 examples [00:26, 908.16 examples/s]22667 examples [00:26, 908.82 examples/s]22759 examples [00:26, 901.87 examples/s]22850 examples [00:26, 901.66 examples/s]22941 examples [00:26, 902.71 examples/s]23032 examples [00:26, 890.41 examples/s]23122 examples [00:26, 848.24 examples/s]23208 examples [00:27, 847.40 examples/s]23303 examples [00:27, 874.55 examples/s]23393 examples [00:27, 880.98 examples/s]23486 examples [00:27, 895.03 examples/s]23581 examples [00:27, 909.64 examples/s]23674 examples [00:27, 914.18 examples/s]23768 examples [00:27, 919.15 examples/s]23861 examples [00:27, 891.46 examples/s]23953 examples [00:27, 898.28 examples/s]24044 examples [00:27, 897.43 examples/s]24134 examples [00:28, 879.30 examples/s]24223 examples [00:28, 876.24 examples/s]24311 examples [00:28, 853.82 examples/s]24397 examples [00:28, 834.95 examples/s]24481 examples [00:28, 831.79 examples/s]24565 examples [00:28, 833.45 examples/s]24653 examples [00:28, 846.66 examples/s]24740 examples [00:28, 852.30 examples/s]24828 examples [00:28, 858.25 examples/s]24916 examples [00:29, 863.26 examples/s]25004 examples [00:29, 865.25 examples/s]25091 examples [00:29, 866.22 examples/s]25179 examples [00:29, 869.78 examples/s]25267 examples [00:29, 871.55 examples/s]25355 examples [00:29, 872.57 examples/s]25443 examples [00:29, 858.23 examples/s]25530 examples [00:29, 859.45 examples/s]25616 examples [00:29, 852.76 examples/s]25703 examples [00:29, 856.63 examples/s]25790 examples [00:30, 860.53 examples/s]25877 examples [00:30, 847.76 examples/s]25963 examples [00:30, 851.13 examples/s]26056 examples [00:30, 872.31 examples/s]26144 examples [00:30, 870.14 examples/s]26232 examples [00:30, 866.12 examples/s]26319 examples [00:30, 857.10 examples/s]26407 examples [00:30, 861.12 examples/s]26494 examples [00:30, 816.23 examples/s]26581 examples [00:30, 828.97 examples/s]26667 examples [00:31, 835.32 examples/s]26757 examples [00:31, 851.17 examples/s]26846 examples [00:31, 861.72 examples/s]26937 examples [00:31, 872.84 examples/s]27032 examples [00:31, 893.51 examples/s]27123 examples [00:31, 898.21 examples/s]27214 examples [00:31, 892.41 examples/s]27304 examples [00:31, 894.54 examples/s]27394 examples [00:31, 887.86 examples/s]27483 examples [00:31, 886.38 examples/s]27575 examples [00:32, 895.22 examples/s]27667 examples [00:32, 901.38 examples/s]27760 examples [00:32, 908.24 examples/s]27852 examples [00:32, 909.33 examples/s]27944 examples [00:32, 912.35 examples/s]28036 examples [00:32, 890.71 examples/s]28126 examples [00:32, 886.72 examples/s]28215 examples [00:32, 879.96 examples/s]28309 examples [00:32, 895.46 examples/s]28401 examples [00:33, 901.63 examples/s]28492 examples [00:33, 902.19 examples/s]28585 examples [00:33, 909.86 examples/s]28677 examples [00:33, 910.47 examples/s]28769 examples [00:33, 899.13 examples/s]28859 examples [00:33, 891.20 examples/s]28949 examples [00:33, 881.70 examples/s]29039 examples [00:33, 886.14 examples/s]29130 examples [00:33, 890.31 examples/s]29220 examples [00:33, 851.89 examples/s]29311 examples [00:34, 867.32 examples/s]29402 examples [00:34, 877.70 examples/s]29491 examples [00:34, 881.30 examples/s]29585 examples [00:34, 896.20 examples/s]29678 examples [00:34, 904.30 examples/s]29772 examples [00:34, 912.36 examples/s]29864 examples [00:34, 899.25 examples/s]29955 examples [00:34, 897.00 examples/s]30045 examples [00:34, 837.89 examples/s]30140 examples [00:34, 865.77 examples/s]30234 examples [00:35, 884.41 examples/s]30324 examples [00:35, 858.53 examples/s]30413 examples [00:35, 866.55 examples/s]30503 examples [00:35, 876.30 examples/s]30598 examples [00:35, 894.30 examples/s]30692 examples [00:35, 906.61 examples/s]30783 examples [00:35, 889.40 examples/s]30873 examples [00:35, 886.09 examples/s]30962 examples [00:35, 852.74 examples/s]31048 examples [00:36, 852.95 examples/s]31140 examples [00:36, 869.35 examples/s]31231 examples [00:36, 881.05 examples/s]31325 examples [00:36, 895.08 examples/s]31415 examples [00:36, 885.61 examples/s]31504 examples [00:36, 878.45 examples/s]31595 examples [00:36, 885.83 examples/s]31687 examples [00:36, 893.62 examples/s]31783 examples [00:36, 911.32 examples/s]31875 examples [00:36, 889.07 examples/s]31965 examples [00:37, 883.08 examples/s]32054 examples [00:37, 882.67 examples/s]32145 examples [00:37, 887.86 examples/s]32237 examples [00:37, 894.99 examples/s]32329 examples [00:37, 901.51 examples/s]32425 examples [00:37, 916.68 examples/s]32519 examples [00:37, 920.87 examples/s]32612 examples [00:37, 915.34 examples/s]32707 examples [00:37, 924.94 examples/s]32801 examples [00:37, 925.80 examples/s]32896 examples [00:38, 930.66 examples/s]32990 examples [00:38, 912.66 examples/s]33082 examples [00:38, 900.16 examples/s]33173 examples [00:38, 885.54 examples/s]33262 examples [00:38, 857.22 examples/s]33349 examples [00:38, 860.35 examples/s]33439 examples [00:38, 869.69 examples/s]33527 examples [00:38, 871.69 examples/s]33615 examples [00:38, 868.69 examples/s]33702 examples [00:38, 856.70 examples/s]33797 examples [00:39, 881.67 examples/s]33890 examples [00:39, 895.00 examples/s]33980 examples [00:39, 893.59 examples/s]34072 examples [00:39, 901.35 examples/s]34163 examples [00:39, 900.54 examples/s]34254 examples [00:39, 894.43 examples/s]34347 examples [00:39, 902.36 examples/s]34440 examples [00:39, 909.58 examples/s]34532 examples [00:39, 892.33 examples/s]34622 examples [00:40, 848.31 examples/s]34715 examples [00:40, 870.94 examples/s]34803 examples [00:40, 868.78 examples/s]34892 examples [00:40, 871.01 examples/s]34981 examples [00:40, 876.40 examples/s]35069 examples [00:40, 871.80 examples/s]35157 examples [00:40, 864.34 examples/s]35251 examples [00:40, 883.87 examples/s]35342 examples [00:40, 890.96 examples/s]35435 examples [00:40, 900.17 examples/s]35527 examples [00:41, 905.96 examples/s]35618 examples [00:41, 905.48 examples/s]35709 examples [00:41, 887.18 examples/s]35802 examples [00:41, 897.31 examples/s]35895 examples [00:41, 905.17 examples/s]35990 examples [00:41, 916.42 examples/s]36088 examples [00:41, 933.12 examples/s]36182 examples [00:41, 931.36 examples/s]36276 examples [00:41, 931.61 examples/s]36370 examples [00:41, 932.97 examples/s]36464 examples [00:42, 929.97 examples/s]36558 examples [00:42, 927.58 examples/s]36651 examples [00:42, 923.36 examples/s]36744 examples [00:42, 904.50 examples/s]36836 examples [00:42, 906.48 examples/s]36927 examples [00:42, 876.38 examples/s]37015 examples [00:42, 864.85 examples/s]37102 examples [00:42, 864.87 examples/s]37193 examples [00:42, 876.05 examples/s]37282 examples [00:42, 878.42 examples/s]37370 examples [00:43, 827.12 examples/s]37454 examples [00:43, 810.40 examples/s]37536 examples [00:43, 803.72 examples/s]37627 examples [00:43, 832.15 examples/s]37711 examples [00:43, 811.43 examples/s]37801 examples [00:43, 834.41 examples/s]37889 examples [00:43, 847.13 examples/s]37980 examples [00:43, 864.95 examples/s]38073 examples [00:43, 882.86 examples/s]38162 examples [00:44, 884.04 examples/s]38255 examples [00:44, 897.25 examples/s]38345 examples [00:44, 888.17 examples/s]38434 examples [00:44, 887.40 examples/s]38526 examples [00:44, 895.64 examples/s]38620 examples [00:44, 907.15 examples/s]38718 examples [00:44, 927.74 examples/s]38811 examples [00:44, 914.61 examples/s]38903 examples [00:44, 897.28 examples/s]38993 examples [00:44, 894.51 examples/s]39083 examples [00:45, 889.45 examples/s]39173 examples [00:45, 891.76 examples/s]39263 examples [00:45, 857.74 examples/s]39352 examples [00:45, 865.61 examples/s]39444 examples [00:45, 878.68 examples/s]39540 examples [00:45, 899.82 examples/s]39631 examples [00:45, 887.80 examples/s]39721 examples [00:45, 885.87 examples/s]39810 examples [00:45, 873.17 examples/s]39900 examples [00:45, 880.77 examples/s]39994 examples [00:46, 895.72 examples/s]40084 examples [00:46, 838.19 examples/s]40175 examples [00:46, 857.87 examples/s]40263 examples [00:46, 863.90 examples/s]40350 examples [00:46, 864.21 examples/s]40437 examples [00:46, 863.89 examples/s]40527 examples [00:46, 871.96 examples/s]40620 examples [00:46, 887.67 examples/s]40711 examples [00:46, 892.37 examples/s]40804 examples [00:47, 900.70 examples/s]40896 examples [00:47, 903.53 examples/s]40988 examples [00:47, 907.70 examples/s]41080 examples [00:47, 909.96 examples/s]41173 examples [00:47, 914.37 examples/s]41266 examples [00:47, 918.58 examples/s]41364 examples [00:47, 934.86 examples/s]41458 examples [00:47, 931.61 examples/s]41552 examples [00:47, 917.42 examples/s]41644 examples [00:47, 901.99 examples/s]41735 examples [00:48, 883.19 examples/s]41824 examples [00:48, 881.32 examples/s]41918 examples [00:48, 897.02 examples/s]42008 examples [00:48, 883.12 examples/s]42097 examples [00:48, 877.97 examples/s]42185 examples [00:48, 865.82 examples/s]42272 examples [00:48, 851.15 examples/s]42370 examples [00:48, 883.86 examples/s]42464 examples [00:48, 899.27 examples/s]42555 examples [00:48, 899.05 examples/s]42646 examples [00:49, 868.01 examples/s]42735 examples [00:49, 873.25 examples/s]42823 examples [00:49, 848.17 examples/s]42913 examples [00:49, 860.95 examples/s]43004 examples [00:49, 872.83 examples/s]43096 examples [00:49, 885.18 examples/s]43187 examples [00:49, 892.31 examples/s]43277 examples [00:49, 883.84 examples/s]43366 examples [00:49, 881.49 examples/s]43455 examples [00:49, 869.96 examples/s]43549 examples [00:50, 888.34 examples/s]43645 examples [00:50, 906.31 examples/s]43736 examples [00:50, 875.67 examples/s]43824 examples [00:50, 857.59 examples/s]43916 examples [00:50, 874.61 examples/s]44008 examples [00:50, 887.52 examples/s]44098 examples [00:50, 888.27 examples/s]44188 examples [00:50, 882.79 examples/s]44277 examples [00:50, 873.99 examples/s]44365 examples [00:51, 872.17 examples/s]44457 examples [00:51, 885.24 examples/s]44547 examples [00:51, 888.38 examples/s]44638 examples [00:51, 891.86 examples/s]44729 examples [00:51, 895.60 examples/s]44821 examples [00:51, 901.90 examples/s]44916 examples [00:51, 914.84 examples/s]45010 examples [00:51, 919.65 examples/s]45103 examples [00:51, 913.33 examples/s]45195 examples [00:51, 909.84 examples/s]45287 examples [00:52, 889.92 examples/s]45379 examples [00:52, 896.43 examples/s]45469 examples [00:52, 880.08 examples/s]45558 examples [00:52, 873.96 examples/s]45650 examples [00:52, 886.44 examples/s]45746 examples [00:52, 906.46 examples/s]45840 examples [00:52, 913.67 examples/s]45932 examples [00:52, 914.97 examples/s]46024 examples [00:52, 909.31 examples/s]46116 examples [00:52, 892.53 examples/s]46209 examples [00:53, 900.94 examples/s]46302 examples [00:53, 907.51 examples/s]46397 examples [00:53, 918.15 examples/s]46489 examples [00:53, 914.55 examples/s]46581 examples [00:53, 907.77 examples/s]46672 examples [00:53, 898.38 examples/s]46762 examples [00:53, 888.71 examples/s]46855 examples [00:53, 900.25 examples/s]46946 examples [00:53, 896.14 examples/s]47036 examples [00:53, 889.28 examples/s]47126 examples [00:54, 889.53 examples/s]47222 examples [00:54, 908.75 examples/s]47317 examples [00:54, 918.09 examples/s]47409 examples [00:54, 898.71 examples/s]47500 examples [00:54, 851.72 examples/s]47586 examples [00:54, 846.19 examples/s]47675 examples [00:54, 858.12 examples/s]47764 examples [00:54, 865.42 examples/s]47854 examples [00:54, 873.70 examples/s]47942 examples [00:55, 861.58 examples/s]48037 examples [00:55, 884.67 examples/s]48129 examples [00:55, 891.30 examples/s]48222 examples [00:55, 899.66 examples/s]48313 examples [00:55, 874.03 examples/s]48401 examples [00:55, 870.58 examples/s]48498 examples [00:55, 897.45 examples/s]48594 examples [00:55, 913.48 examples/s]48688 examples [00:55, 919.70 examples/s]48781 examples [00:55, 906.65 examples/s]48872 examples [00:56, 886.78 examples/s]48961 examples [00:56, 865.31 examples/s]49051 examples [00:56, 872.90 examples/s]49143 examples [00:56, 885.96 examples/s]49232 examples [00:56, 882.67 examples/s]49322 examples [00:56, 887.06 examples/s]49412 examples [00:56, 888.20 examples/s]49502 examples [00:56, 891.40 examples/s]49594 examples [00:56, 898.14 examples/s]49688 examples [00:56, 908.28 examples/s]49779 examples [00:57, 906.78 examples/s]49871 examples [00:57, 910.51 examples/s]49963 examples [00:57, 911.82 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6817/50000 [00:00<00:00, 68153.65 examples/s] 37%|███▋      | 18552/50000 [00:00<00:00, 77957.43 examples/s] 60%|██████    | 30088/50000 [00:00<00:00, 86355.66 examples/s] 85%|████████▍ | 42284/50000 [00:00<00:00, 94644.22 examples/s]                                                               0 examples [00:00, ? examples/s]71 examples [00:00, 709.20 examples/s]160 examples [00:00, 754.69 examples/s]250 examples [00:00, 793.05 examples/s]341 examples [00:00, 824.13 examples/s]431 examples [00:00, 843.17 examples/s]519 examples [00:00, 848.91 examples/s]604 examples [00:00, 846.27 examples/s]689 examples [00:00, 846.79 examples/s]778 examples [00:00, 856.38 examples/s]872 examples [00:01, 879.00 examples/s]970 examples [00:01, 905.66 examples/s]1060 examples [00:01, 899.88 examples/s]1154 examples [00:01, 911.29 examples/s]1249 examples [00:01, 921.68 examples/s]1341 examples [00:01, 919.88 examples/s]1433 examples [00:01, 916.86 examples/s]1527 examples [00:01, 922.75 examples/s]1620 examples [00:01, 910.37 examples/s]1713 examples [00:01, 915.09 examples/s]1805 examples [00:02, 910.84 examples/s]1900 examples [00:02, 920.44 examples/s]1997 examples [00:02, 932.75 examples/s]2091 examples [00:02, 910.65 examples/s]2184 examples [00:02, 914.71 examples/s]2276 examples [00:02, 915.10 examples/s]2368 examples [00:02, 912.47 examples/s]2460 examples [00:02, 913.19 examples/s]2552 examples [00:02, 897.54 examples/s]2642 examples [00:02, 897.62 examples/s]2732 examples [00:03, 876.30 examples/s]2820 examples [00:03, 828.31 examples/s]2908 examples [00:03, 841.20 examples/s]2996 examples [00:03, 849.95 examples/s]3082 examples [00:03, 844.70 examples/s]3171 examples [00:03, 855.54 examples/s]3257 examples [00:03, 840.49 examples/s]3348 examples [00:03, 858.87 examples/s]3438 examples [00:03, 868.57 examples/s]3529 examples [00:03, 879.91 examples/s]3619 examples [00:04, 883.94 examples/s]3708 examples [00:04, 885.21 examples/s]3799 examples [00:04, 889.98 examples/s]3893 examples [00:04, 903.47 examples/s]3984 examples [00:04, 905.36 examples/s]4075 examples [00:04, 897.78 examples/s]4172 examples [00:04, 916.24 examples/s]4268 examples [00:04, 926.54 examples/s]4361 examples [00:04, 919.59 examples/s]4455 examples [00:04, 924.53 examples/s]4548 examples [00:05, 924.94 examples/s]4642 examples [00:05, 928.11 examples/s]4735 examples [00:05, 904.74 examples/s]4826 examples [00:05, 879.88 examples/s]4915 examples [00:05, 864.60 examples/s]5006 examples [00:05, 877.62 examples/s]5094 examples [00:05, 865.64 examples/s]5183 examples [00:05, 869.42 examples/s]5271 examples [00:05, 866.56 examples/s]5358 examples [00:06, 866.40 examples/s]5445 examples [00:06, 861.67 examples/s]5539 examples [00:06, 882.75 examples/s]5633 examples [00:06, 897.89 examples/s]5726 examples [00:06, 905.47 examples/s]5817 examples [00:06, 891.03 examples/s]5910 examples [00:06, 900.05 examples/s]6001 examples [00:06, 868.78 examples/s]6090 examples [00:06, 873.47 examples/s]6180 examples [00:06, 879.49 examples/s]6269 examples [00:07, 869.67 examples/s]6357 examples [00:07, 846.53 examples/s]6443 examples [00:07, 847.20 examples/s]6532 examples [00:07, 858.89 examples/s]6621 examples [00:07, 866.70 examples/s]6710 examples [00:07, 872.24 examples/s]6798 examples [00:07, 869.34 examples/s]6886 examples [00:07, 865.93 examples/s]6974 examples [00:07, 868.43 examples/s]7061 examples [00:07, 863.25 examples/s]7150 examples [00:08, 868.72 examples/s]7238 examples [00:08, 871.39 examples/s]7327 examples [00:08, 873.46 examples/s]7418 examples [00:08, 881.01 examples/s]7507 examples [00:08, 863.42 examples/s]7602 examples [00:08, 887.59 examples/s]7694 examples [00:08, 895.33 examples/s]7784 examples [00:08, 887.87 examples/s]7873 examples [00:08, 884.12 examples/s]7962 examples [00:09, 872.95 examples/s]8050 examples [00:09, 864.67 examples/s]8137 examples [00:09, 840.94 examples/s]8222 examples [00:09, 837.75 examples/s]8309 examples [00:09, 844.55 examples/s]8398 examples [00:09, 857.50 examples/s]8491 examples [00:09, 875.76 examples/s]8584 examples [00:09, 889.30 examples/s]8674 examples [00:09, 878.93 examples/s]8763 examples [00:09, 880.07 examples/s]8854 examples [00:10, 888.75 examples/s]8943 examples [00:10, 884.07 examples/s]9032 examples [00:10, 882.15 examples/s]9121 examples [00:10, 844.79 examples/s]9206 examples [00:10, 835.09 examples/s]9294 examples [00:10, 846.84 examples/s]9379 examples [00:10, 846.20 examples/s]9464 examples [00:10, 833.56 examples/s]9551 examples [00:10, 842.59 examples/s]9638 examples [00:10, 848.73 examples/s]9724 examples [00:11, 850.54 examples/s]9810 examples [00:11, 848.21 examples/s]9898 examples [00:11, 856.34 examples/s]9989 examples [00:11, 871.03 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteBTZQ64/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteBTZQ64/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe03206e8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe03206e8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe03206e8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe07da5c748>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe01aeda128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe03206e8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe03206e8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe03206e8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe02f6a63c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe02f6a6240>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fdfa9b12400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fdfa9b12400> 

  function with postional parmater data_info <function split_train_valid at 0x7fdfa9b12400> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fdfa9b12510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fdfa9b12510> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fdfa9b12510> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.4.5.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=45f33a1efd898476b350082481f81f3daff8ba0955255f9f64b5193ef2d393aa
  Stored in directory: /tmp/pip-ephem-wheel-cache-v0xy1jk6/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.0
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<19:32:08, 12.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:55:35, 17.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:48:22, 24.4kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:52:27, 34.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.35M/862M [00:01<4:48:04, 49.7kB/s].vector_cache/glove.6B.zip:   1%|          | 6.65M/862M [00:01<3:21:00, 70.9kB/s].vector_cache/glove.6B.zip:   1%|          | 9.13M/862M [00:01<2:20:30, 101kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 13.6M/862M [00:01<1:37:55, 144kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:01<1:08:19, 206kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.5M/862M [00:01<47:44, 294kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 25.7M/862M [00:01<33:20, 418kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.8M/862M [00:01<23:19, 595kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.1M/862M [00:02<16:23, 843kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.2M/862M [00:02<11:31, 1.19MB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.4M/862M [00:02<08:08, 1.68MB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.7M/862M [00:02<05:52, 2.32MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.0M/862M [00:02<04:11, 3.24MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.9M/862M [00:02<03:36, 3.74MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.0M/862M [00:03<02:36, 5.15MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.1M/862M [00:04<7:09:28, 31.3kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.3M/862M [00:04<5:02:49, 44.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<3:32:16, 63.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.1M/862M [00:05<2:28:12, 90.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:06<3:53:07, 57.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<2:45:16, 80.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<1:56:03, 115kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.3M/862M [00:08<1:23:23, 160kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.6M/862M [00:08<1:00:18, 221kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<42:33, 312kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<29:55, 443kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.5M/862M [00:10<29:09, 454kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:10<22:03, 600kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<15:46, 837kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:11<11:13, 1.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.7M/862M [00:12<20:27, 644kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:12<15:35, 845kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<11:41, 1.13MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:13<08:19, 1.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.8M/862M [00:14<12:30, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:14<10:16, 1.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<07:31, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.0M/862M [00:16<08:09, 1.60MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<07:06, 1.83MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:16<05:15, 2.47MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:17<04:18, 3.01MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.1M/862M [00:18<07:11, 1.81MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:18<08:08, 1.59MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<06:20, 2.04MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:19<04:50, 2.67MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.3M/862M [00:20<06:04, 2.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:20<07:21, 1.75MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<05:56, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:21<04:17, 2.99MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:22<10:35, 1.21MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:22<10:37, 1.21MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<08:12, 1.56MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:23<05:53, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:24<11:18, 1.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:24<10:43, 1.19MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<08:11, 1.55MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<05:52, 2.16MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<12:34, 1.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<11:37, 1.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:48, 1.44MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<06:17, 2.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<17:22, 726kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<14:49, 851kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<11:01, 1.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<07:50, 1.60MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<15:32, 808kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<13:30, 929kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<10:05, 1.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<07:11, 1.74MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<15:10, 823kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<13:14, 942kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<09:53, 1.26MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<07:01, 1.77MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<49:42, 250kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<37:19, 332kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<26:37, 465kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<18:51, 656kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<16:56, 729kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<14:16, 865kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<10:43, 1.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<07:37, 1.61MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<15:10, 809kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<13:00, 943kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<09:41, 1.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<06:54, 1.77MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<18:23, 664kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<15:48, 772kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<11:41, 1.04MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<08:26, 1.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:31, 1.42MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<10:54, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<08:41, 1.39MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<06:42, 1.81MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<04:48, 2.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<26:24, 456kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<21:24, 563kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<15:41, 767kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<11:06, 1.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<13:34, 883kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<12:23, 967kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<09:15, 1.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<06:46, 1.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:31, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:30, 1.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<05:48, 2.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<04:10, 2.84MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<25:44, 460kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<20:51, 568kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<15:18, 773kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<10:50, 1.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<12:43, 925kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<11:45, 1.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:49, 1.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<06:27, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:15, 1.61MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:48, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:01, 1.94MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:55<04:33, 2.56MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:44, 2.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:52, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<04:51, 2.39MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:37, 3.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:44, 2.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:47, 1.70MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:19, 2.17MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<04:02, 2.86MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:24, 2.13MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<09:22, 1.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:00<08:14, 1.39MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<07:46, 1.48MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<08:04, 1.42MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<08:05, 1.42MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<07:55, 1.45MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<07:27, 1.54MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<07:50, 1.46MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<07:23, 1.55MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<07:48, 1.47MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<07:21, 1.56MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:02<07:33, 1.52MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:02<07:11, 1.59MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<07:26, 1.54MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<07:02, 1.63MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<07:18, 1.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<06:57, 1.64MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<07:12, 1.59MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:40, 1.71MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:14, 1.58MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<06:42, 1.70MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<07:16, 1.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<06:52, 1.66MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<06:59, 1.63MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:03<06:41, 1.71MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<06:45, 1.69MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<06:30, 1.75MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<06:37, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<06:20, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<06:31, 1.75MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:04<06:15, 1.82MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:04<06:22, 1.79MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:04<06:07, 1.86MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:04<06:19, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<05:53, 1.94MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<07:57, 1.43MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<07:05, 1.60MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<06:57, 1.64MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:05<07:46, 1.46MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<07:16, 1.56MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<08:02, 1.42MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<09:15, 1.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<08:01, 1.42MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<09:12, 1.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<09:04, 1.25MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<09:23, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<09:13, 1.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<09:46, 1.16MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<09:18, 1.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<09:29, 1.20MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<09:01, 1.26MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<09:22, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<08:56, 1.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:06<09:17, 1.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:06<08:42, 1.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:06<09:17, 1.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:06<08:35, 1.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:07<09:05, 1.25MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:07<08:41, 1.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:07<08:45, 1.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:07<08:20, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:07<08:34, 1.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:07<16:46, 675kB/s] .vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:07<13:16, 853kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:07<11:39, 970kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:08<10:28, 1.08MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<09:20, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<08:21, 1.35MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<08:38, 1.31MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<07:56, 1.42MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<08:03, 1.40MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<07:38, 1.48MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<07:36, 1.48MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<07:14, 1.56MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<07:18, 1.54MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<07:07, 1.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:09<07:09, 1.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:09<06:55, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:09<06:58, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:09<06:58, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:09<06:47, 1.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:09<06:50, 1.65MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:09<06:37, 1.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:10<06:37, 1.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:10<06:27, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:10<06:30, 1.73MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:10<06:22, 1.77MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:10<06:23, 1.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<06:15, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<06:13, 1.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<06:05, 1.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<06:04, 1.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<05:58, 1.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<06:03, 1.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<05:54, 1.90MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<05:58, 1.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<05:48, 1.93MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<05:52, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<05:41, 1.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<05:45, 1.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<05:35, 2.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:12<05:42, 1.96MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:12<05:28, 2.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:12<05:41, 1.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:12<05:24, 2.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:12<05:35, 2.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<05:21, 2.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<05:29, 2.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<05:17, 2.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<05:23, 2.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<05:10, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<05:18, 2.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<05:07, 2.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<05:13, 2.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<05:04, 2.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:13<05:04, 2.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:13<04:53, 2.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:13<04:51, 2.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:13<04:40, 2.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:13<04:38, 2.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:14<04:26, 2.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:14<04:24, 2.52MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<04:14, 2.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<05:12, 2.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<04:29, 2.47MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<04:37, 2.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<04:45, 2.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<04:53, 2.27MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<04:43, 2.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<04:51, 2.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<04:39, 2.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<04:45, 2.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<04:32, 2.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:15<04:38, 2.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:15<04:24, 2.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:15<04:30, 2.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:15<04:14, 2.60MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<04:24, 2.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:16<04:10, 2.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:16<04:18, 2.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<04:02, 2.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<04:11, 2.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<03:59, 2.76MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<04:06, 2.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<05:28, 2.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<04:44, 2.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<04:50, 2.27MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<05:03, 2.17MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:17<05:00, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:17<07:28, 1.47MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:17<06:21, 1.73MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:17<06:42, 1.64MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:17<06:39, 1.65MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:17<06:51, 1.60MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:17<06:39, 1.65MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:17<06:44, 1.63MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<06:31, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<06:37, 1.66MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:18<06:28, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:18<06:32, 1.67MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<06:27, 1.70MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<06:23, 1.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<06:17, 1.74MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<06:15, 1.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:18<06:13, 1.76MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<06:12, 1.76MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<06:06, 1.79MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<06:07, 1.78MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<06:05, 1.79MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:19<06:02, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:19<05:58, 1.83MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:19<06:02, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:19<05:58, 1.83MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:19<05:59, 1.82MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:19<05:51, 1.86MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<05:48, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<05:47, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<05:45, 1.89MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<05:43, 1.90MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<05:38, 1.93MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<05:37, 1.94MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<05:35, 1.95MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<05:30, 1.98MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<05:27, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<05:24, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<05:23, 2.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<05:17, 2.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<05:17, 2.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<05:11, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<05:12, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<05:08, 2.11MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:21<05:06, 2.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:21<05:04, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:22<05:01, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:22<04:57, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:22<04:59, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<04:56, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<04:55, 2.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<04:50, 2.24MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<04:47, 2.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<04:45, 2.27MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<04:46, 2.26MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<04:42, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<04:42, 2.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<04:38, 2.32MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<04:36, 2.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<04:35, 2.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<04:32, 2.37MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:23<04:30, 2.39MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:23<04:30, 2.39MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:23<04:26, 2.42MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:23<04:29, 2.40MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<04:20, 2.47MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<04:25, 2.43MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<04:18, 2.50MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<04:21, 2.46MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<04:13, 2.54MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<04:19, 2.48MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<04:09, 2.58MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<04:15, 2.52MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<04:06, 2.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:25<04:12, 2.54MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:25<04:05, 2.62MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:25<04:09, 2.58MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:25<03:59, 2.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:25<04:07, 2.59MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<03:57, 2.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<04:01, 2.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<03:50, 2.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:25<03:53, 2.74MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<03:42, 2.87MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<03:46, 2.82MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:26<03:34, 2.98MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:26<03:36, 2.95MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:26<03:23, 3.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<03:27, 3.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<03:15, 3.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<03:17, 3.24MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<03:04, 3.45MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<03:07, 3.39MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:27<02:55, 3.64MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:27<02:59, 3.55MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:27<02:46, 3.82MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<18:09, 584kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<14:46, 716kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<11:12, 944kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:28<08:33, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<06:33, 1.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<05:12, 2.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<04:19, 2.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<03:39, 2.88MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:30<09:27, 1.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:30<09:42, 1.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:30<07:39, 1.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:30<05:58, 1.76MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<04:43, 2.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<03:48, 2.76MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<03:08, 3.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:32<1:26:47, 120kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:32<1:03:15, 165kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:32<44:55, 232kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:32<31:50, 327kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<22:37, 460kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<16:08, 644kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<26:52, 386kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<23:03, 450kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:34<17:10, 604kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:34<12:24, 835kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<08:58, 1.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<06:36, 1.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:35<17:23, 593kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<15:36, 661kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<11:38, 886kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<08:43, 1.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:36<06:21, 1.62MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<04:50, 2.12MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:37<08:25, 1.22MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<12:39, 810kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<12:18, 832kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:38<09:21, 1.09MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<06:52, 1.49MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<05:07, 1.99MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:39<07:54, 1.29MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:40<08:01, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:40<07:17, 1.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:40<05:25, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<04:12, 2.41MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<03:11, 3.17MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:41<11:29, 880kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:42<10:22, 974kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:42<07:45, 1.30MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:42<05:48, 1.73MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<04:14, 2.37MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:43<10:26, 962kB/s] .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:44<09:11, 1.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:44<06:53, 1.45MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<05:09, 1.94MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<03:56, 2.53MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:45<06:47, 1.47MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:46<09:33, 1.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:46<07:37, 1.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<05:30, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<04:13, 2.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:47<06:39, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:47<06:13, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:48<05:19, 1.86MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<03:54, 2.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<02:58, 3.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:49<1:08:33, 143kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:49<50:49, 193kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:50<36:15, 271kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<25:27, 384kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:51<21:16, 459kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:51<17:27, 559kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:52<12:45, 764kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:52<09:08, 1.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:53<08:39, 1.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:53<08:23, 1.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:54<06:21, 1.52MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<04:42, 2.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:55<05:26, 1.77MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:55<05:56, 1.62MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:56<04:41, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<03:25, 2.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:57<10:23, 920kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:57<10:04, 948kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:58<07:44, 1.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<05:31, 1.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:59<07:35, 1.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:59<07:09, 1.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:00<05:27, 1.73MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<03:56, 2.39MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:01<2:15:19, 69.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:01<1:37:12, 96.9kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:01<1:08:29, 137kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:02<48:10, 195kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:03<35:26, 264kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:03<27:02, 346kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:04<19:28, 479kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<13:42, 677kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<17:44, 523kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:05<14:33, 637kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:06<10:43, 864kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<07:35, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:07<10:23, 886kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:07<08:47, 1.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<06:28, 1.42MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<04:44, 1.93MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:09<06:04, 1.50MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:09<08:17, 1.10MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:09<06:45, 1.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<04:54, 1.85MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<03:40, 2.47MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<07:00, 1.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<08:36, 1.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<06:56, 1.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<05:04, 1.78MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<05:39, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<07:39, 1.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:13<06:05, 1.48MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:13<04:41, 1.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<03:24, 2.62MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:15<07:35, 1.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:15<08:40, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:15<06:45, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:15<05:09, 1.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<03:44, 2.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<08:25, 1.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<09:14, 958kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:17<07:06, 1.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:17<05:23, 1.64MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:18<03:53, 2.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:19<09:51, 893kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:19<09:59, 881kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:19<07:37, 1.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<05:43, 1.53MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<04:07, 2.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<08:58, 973kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<09:34, 911kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<07:23, 1.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<05:34, 1.56MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<04:00, 2.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:23<07:51, 1.10MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:23<08:31, 1.02MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:23<06:34, 1.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<04:54, 1.76MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<03:33, 2.42MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:25<11:03, 777kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:25<10:34, 812kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<07:57, 1.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<05:55, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:13, 2.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<16:31, 516kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<14:22, 592kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:27<10:45, 791kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<07:41, 1.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<07:50, 1.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<08:16, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:29<06:21, 1.33MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<04:42, 1.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<03:24, 2.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<18:50, 445kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:31<15:46, 531kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:31<11:41, 716kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<08:21, 999kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<08:09, 1.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<08:26, 985kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<06:29, 1.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<04:46, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<03:42, 2.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:35<04:59, 1.65MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:35<07:27, 1.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:35<06:15, 1.32MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<04:37, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<03:24, 2.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:37<10:25, 784kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:37<11:12, 729kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:37<08:49, 925kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<06:24, 1.27MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:37, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:39<14:54, 543kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:39<12:17, 659kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:39<09:27, 855kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:39<07:29, 1.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:39<05:56, 1.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<05:08, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<04:19, 1.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<04:02, 1.99MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:40<03:32, 2.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:40<03:23, 2.37MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:40<03:05, 2.61MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:40<03:06, 2.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:40<02:51, 2.81MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<10:19, 778kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<08:19, 963kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:41<06:36, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:41<05:23, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<04:33, 1.76MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<03:58, 2.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<04:13, 1.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<03:48, 2.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<03:52, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<03:30, 2.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<03:41, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:42<03:27, 2.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:42<03:27, 2.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:42<03:14, 2.46MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:43<09:44, 818kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:43<07:55, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:43<06:45, 1.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:43<05:39, 1.40MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:43<04:52, 1.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<04:17, 1.85MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<03:53, 2.04MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<03:37, 2.19MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<03:25, 2.32MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<03:15, 2.42MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:44<02:55, 2.71MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:44<03:21, 2.35MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:44<03:13, 2.45MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:45<38:57, 203kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:45<28:33, 276kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:45<20:52, 378kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:45<15:28, 509kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<11:41, 673kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<08:58, 877kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<06:58, 1.13MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<05:58, 1.32MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<04:53, 1.60MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<04:26, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:46<03:52, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:46<03:44, 2.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:46<03:23, 2.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:46<19:38, 399kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:47<22:03, 355kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:47<16:38, 470kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:47<12:28, 626kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<09:33, 817kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<07:29, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<06:02, 1.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<05:01, 1.55MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<04:17, 1.81MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:48<03:46, 2.06MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:48<03:18, 2.35MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:48<03:15, 2.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<02:56, 2.64MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<03:00, 2.58MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<02:45, 2.80MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:49<3:08:37, 41.1kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:49<2:14:02, 57.8kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:49<1:34:38, 81.7kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:50<1:06:49, 116kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:50<47:35, 162kB/s]  .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:50<33:57, 227kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:50<24:35, 314kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:50<17:51, 431kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:50<13:20, 577kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:50<09:58, 771kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:50<07:53, 975kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:50<06:07, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:50<05:11, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<12:25, 618kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<10:37, 723kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<08:12, 934kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:52<06:27, 1.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:52<05:04, 1.51MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:52<04:30, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<03:41, 2.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<07:33, 1.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<06:12, 1.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:52<05:07, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:52<04:38, 1.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:52<04:55, 1.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:53<04:29, 1.70MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:53<04:43, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<04:20, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<1:15:19, 101kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<53:52, 141kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<38:54, 195kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<28:25, 267kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:54<21:01, 361kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:54<15:51, 478kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:54<12:13, 620kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:54<09:41, 781kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:54<07:53, 959kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:54<06:31, 1.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:54<05:29, 1.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:55<05:27, 1.39MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:55<04:46, 1.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:55<04:22, 1.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:55<04:24, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:55<04:03, 1.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:55<03:52, 1.95MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:55<03:50, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:55<03:40, 2.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<03:49, 1.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<03:43, 2.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<03:39, 2.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<03:28, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<03:23, 2.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<03:25, 2.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<03:22, 2.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:56<03:24, 2.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:56<03:19, 2.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<03:21, 2.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<03:16, 2.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<03:18, 2.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<03:18, 2.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<03:19, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<04:10, 1.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<03:39, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<03:58, 1.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<03:58, 1.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:58<04:06, 1.82MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:58<04:01, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:58<04:05, 1.82MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<03:55, 1.90MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<03:59, 1.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<03:50, 1.94MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<03:53, 1.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:58<03:42, 2.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:58<03:45, 1.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<03:35, 2.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<05:07, 1.45MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<04:20, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:59<04:41, 1.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:59<04:32, 1.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:59<04:42, 1.57MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:59<04:33, 1.63MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:59<04:41, 1.58MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:59<04:29, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:59<04:34, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<04:24, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<04:26, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<04:20, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<04:22, 1.69MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<04:13, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<04:19, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<04:10, 1.77MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<04:15, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [03:00<04:05, 1.80MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<04:13, 1.74MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<04:03, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<04:11, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<04:02, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<04:11, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<04:02, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<04:07, 1.78MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<03:56, 1.86MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<04:00, 1.83MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<03:55, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<03:56, 1.86MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<03:51, 1.90MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<03:54, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<03:46, 1.94MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<03:48, 1.92MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<03:43, 1.97MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<03:47, 1.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<03:38, 2.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:03<03:43, 1.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<03:35, 2.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<03:37, 2.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<03:33, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<03:33, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<03:26, 2.11MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<03:30, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<03:23, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<03:27, 2.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<03:22, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:04<03:25, 2.12MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:04<03:18, 2.20MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:04<03:19, 2.19MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:04<03:14, 2.24MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<03:15, 2.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<03:12, 2.26MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<03:12, 2.26MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<03:08, 2.30MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:04<03:07, 2.31MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<03:06, 2.33MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<03:04, 2.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<03:05, 2.33MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:05<03:01, 2.38MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:05<03:01, 2.38MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:05<02:59, 2.40MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:05<02:59, 2.40MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:05<02:57, 2.44MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:05<02:57, 2.43MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:06<02:54, 2.46MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:06<02:55, 2.45MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:06<02:53, 2.49MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:06<02:53, 2.48MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:06<02:49, 2.54MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<02:51, 2.51MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<02:47, 2.56MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<02:48, 2.54MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<02:44, 2.61MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:07<03:43, 1.91MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:07<03:24, 2.10MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:07<03:13, 2.22MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:07<04:49, 1.48MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<04:01, 1.77MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<04:48, 1.48MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<04:31, 1.57MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<05:00, 1.42MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<04:37, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:08<05:03, 1.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<04:37, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<05:03, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<04:36, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<04:57, 1.43MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<04:30, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:08<04:53, 1.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:08<04:27, 1.59MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:08<04:51, 1.46MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<04:22, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<04:44, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<04:20, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<04:43, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<04:17, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<04:41, 1.51MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<04:15, 1.66MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<04:32, 1.55MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<04:09, 1.70MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<04:28, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<04:03, 1.74MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<04:20, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<03:59, 1.76MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<04:17, 1.64MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:10<03:55, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:10<04:10, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:10<03:50, 1.83MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:10<04:06, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<03:45, 1.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<03:58, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<03:41, 1.90MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<03:54, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<03:36, 1.94MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:11<03:52, 1.81MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:11<03:34, 1.96MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:11<03:47, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:11<03:30, 2.00MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:12<03:42, 1.89MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<03:25, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<03:42, 1.89MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<03:25, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<03:37, 1.93MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:12<03:21, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:12<03:34, 1.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:12<03:18, 2.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:12<03:29, 2.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<03:13, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<03:23, 2.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<03:12, 2.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<03:20, 2.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<03:09, 2.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<03:14, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<03:04, 2.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<03:10, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<03:00, 2.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:13<03:08, 2.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<03:02, 2.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<03:33, 1.94MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<03:35, 1.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<03:50, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:14<03:42, 1.86MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:14<03:50, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:14<03:44, 1.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:14<03:44, 1.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<03:35, 1.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<03:39, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<03:30, 1.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<03:32, 1.94MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<03:24, 2.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<03:26, 2.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<03:16, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<03:27, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<03:36, 1.91MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<04:00, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<03:55, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<05:10, 1.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<04:39, 1.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<04:45, 1.44MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<05:23, 1.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<05:06, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<05:39, 1.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:16<05:17, 1.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:16<05:37, 1.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<05:16, 1.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<05:36, 1.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<05:11, 1.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<05:32, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<05:07, 1.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:17<05:24, 1.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:17<05:03, 1.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:17<05:20, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:17<04:57, 1.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:18<05:14, 1.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:18<04:56, 1.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<05:08, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<04:47, 1.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<04:58, 1.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<04:40, 1.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<04:52, 1.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<04:33, 1.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<04:53, 1.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:18<04:30, 1.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:19<04:46, 1.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:19<04:22, 1.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:19<04:49, 1.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:19<04:21, 1.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:19<04:48, 1.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:19<04:19, 1.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:19<04:36, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:19<04:10, 1.62MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:19<04:33, 1.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:20<04:05, 1.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<04:38, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<04:08, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<04:28, 1.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<04:02, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<04:19, 1.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:20<03:55, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:20<04:14, 1.59MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:20<03:50, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:21<04:05, 1.64MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:21<03:49, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:21<03:56, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:21<03:36, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:21<03:52, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:21<03:45, 1.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:21<03:41, 1.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:21<03:38, 1.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:21<03:34, 1.88MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:22<03:31, 1.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:22<03:28, 1.93MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:22<03:26, 1.94MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:22<03:25, 1.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:22<03:20, 2.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:22<03:21, 1.99MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:22<03:16, 2.04MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:22<03:17, 2.03MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:22<03:14, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:23<03:15, 2.05MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:23<03:11, 2.09MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:23<03:13, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:23<03:08, 2.12MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:23<03:09, 2.11MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:23<03:04, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:23<03:06, 2.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:23<03:01, 2.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:23<03:02, 2.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:23<02:58, 2.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:24<03:00, 2.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:24<02:56, 2.25MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:24<02:57, 2.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:24<02:52, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:24<02:52, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<02:47, 2.37MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<02:45, 2.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<02:38, 2.50MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<02:38, 2.50MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:25<02:30, 2.64MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:25<02:28, 2.65MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:25<02:21, 2.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:25<02:20, 2.80MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:25<02:13, 2.96MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:25<02:12, 2.96MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<02:05, 3.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<02:05, 3.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:25<01:57, 3.33MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:26<01:58, 3.31MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:26<01:51, 3.53MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:26<01:51, 3.51MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:26<01:44, 3.75MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:26<01:46, 3.65MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:26<01:52, 3.47MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:26<02:00, 3.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:26<02:00, 3.24MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:26<02:04, 3.13MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:27<02:01, 3.21MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:27<02:04, 3.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:27<01:58, 3.27MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:27<02:01, 3.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:27<06:37, 976kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<05:27, 1.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:28<04:22, 1.47MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:28<03:32, 1.82MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:28<03:00, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<02:43, 2.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<02:22, 2.70MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<02:10, 2.94MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:28<02:03, 3.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:28<01:56, 3.31MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:29<07:10, 892kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:29<06:21, 1.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:30<05:00, 1.28MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:30<03:55, 1.62MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:30<03:22, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:30<02:45, 2.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:30<02:32, 2.50MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<02:15, 2.81MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<01:56, 3.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<02:00, 3.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<07:14, 874kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<06:20, 996kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<04:52, 1.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:32<03:52, 1.62MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:32<03:16, 1.93MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:32<02:40, 2.35MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:32<02:26, 2.57MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:32<02:05, 2.99MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:32<02:02, 3.08MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:32<01:48, 3.45MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:33<12:13, 512kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:33<10:55, 572kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:33<08:22, 746kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:34<06:22, 980kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:34<04:55, 1.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:34<03:53, 1.60MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:34<03:10, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:34<02:40, 2.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<02:16, 2.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:35<10:30, 589kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:35<09:07, 677kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:35<07:06, 868kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:36<05:50, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:36<04:55, 1.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:36<04:06, 1.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:36<03:55, 1.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:36<03:24, 1.81MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:36<03:24, 1.80MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:36<03:01, 2.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:36<03:07, 1.97MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:36<02:50, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<02:52, 2.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<02:39, 2.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:37<02:43, 2.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:37<02:33, 2.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:37<10:07, 604kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:37<07:47, 784kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:37<06:09, 990kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:38<04:58, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:38<04:00, 1.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:38<03:50, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:38<03:12, 1.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:38<03:14, 1.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<02:46, 2.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<02:58, 2.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<02:38, 2.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<02:42, 2.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<02:27, 2.46MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<02:35, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<06:13, 973kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<05:23, 1.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:39<04:20, 1.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:39<03:57, 1.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:40<03:27, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:40<03:06, 1.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:40<02:50, 2.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<02:31, 2.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<02:39, 2.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<02:24, 2.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<02:31, 2.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<02:18, 2.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<02:28, 2.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<04:09, 1.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:41<03:50, 1.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:41<03:20, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:41<02:58, 2.01MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:42<02:42, 2.19MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:42<02:27, 2.42MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<02:13, 2.67MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<02:18, 2.57MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<02:08, 2.78MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<02:12, 2.69MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<02:08, 2.77MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<02:05, 2.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<02:04, 2.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:43<55:49, 106kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<39:56, 148kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<28:26, 207kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<20:25, 289kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<14:59, 393kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:44<10:59, 535kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<08:15, 712kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<06:24, 917kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<05:02, 1.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:44<04:01, 1.45MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:44<03:23, 1.73MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<02:54, 2.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:45<45:30, 128kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:45<33:18, 175kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:45<23:58, 243kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:45<17:18, 337kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:46<12:38, 460kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:46<09:22, 619kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:46<07:11, 807kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:46<05:33, 1.04MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<04:35, 1.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<03:52, 1.50MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<03:14, 1.78MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<03:07, 1.85MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:47<16:03, 360kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:47<12:10, 474kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:47<09:09, 629kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:47<07:01, 819kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:48<05:31, 1.04MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:48<04:27, 1.29MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:48<03:38, 1.57MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:48<03:05, 1.85MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<02:46, 2.07MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<02:28, 2.31MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<02:20, 2.44MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:48<02:10, 2.62MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:49<06:21, 897kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:49<05:18, 1.07MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:49<04:14, 1.34MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:49<03:40, 1.55MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:49<03:00, 1.89MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:49<02:46, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:50<02:22, 2.39MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:50<02:18, 2.46MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:50<02:02, 2.78MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<02:03, 2.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<01:52, 3.02MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<01:57, 2.89MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:51<04:20, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:51<04:28, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:51<03:37, 1.55MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:51<03:06, 1.81MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:51<02:45, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:51<02:20, 2.40MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:52<02:17, 2.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:52<02:01, 2.77MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<02:05, 2.67MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<01:53, 2.96MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<01:55, 2.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:52<01:46, 3.13MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:53<06:40, 834kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:53<05:59, 928kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:53<04:46, 1.16MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:53<03:50, 1.44MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:53<03:11, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:54<02:44, 2.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:54<02:24, 2.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:54<02:11, 2.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<02:01, 2.71MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<01:55, 2.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:55<07:23, 744kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:55<06:30, 845kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:55<05:09, 1.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:55<04:02, 1.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<03:13, 1.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<02:54, 1.89MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:56<02:29, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<02:15, 2.41MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<02:02, 2.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<01:56, 2.82MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<01:48, 3.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<01:46, 3.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<07:41, 706kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<06:37, 820kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:57<05:11, 1.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:57<04:01, 1.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<03:22, 1.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<02:46, 1.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<02:28, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:58<02:08, 2.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:58<02:00, 2.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<01:48, 2.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<01:50, 2.93MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:59<04:12, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:59<04:07, 1.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:59<03:23, 1.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:59<02:52, 1.85MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:59<02:37, 2.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:59<02:25, 2.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<02:16, 2.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<02:04, 2.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<01:53, 2.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<01:59, 2.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<01:50, 2.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<01:54, 2.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:01<08:41, 609kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:01<06:50, 773kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:01<05:17, 997kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:01<04:11, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:01<03:24, 1.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<02:48, 1.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<02:29, 2.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:02<02:10, 2.42MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<02:01, 2.58MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<01:50, 2.84MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<01:47, 2.93MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:03<08:29, 616kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:03<07:07, 733kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:03<05:32, 942kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:03<04:20, 1.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:03<03:28, 1.50MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:03<02:51, 1.82MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:04<02:25, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<02:05, 2.47MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<01:48, 2.86MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<01:48, 2.86MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:05<07:42, 669kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:05<06:31, 790kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:05<05:03, 1.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:05<03:52, 1.33MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:05<03:16, 1.56MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<02:38, 1.94MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<02:23, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:05<02:01, 2.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:06<01:56, 2.62MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:06<01:43, 2.97MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<01:43, 2.96MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:07<07:53, 645kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:07<10:02, 507kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:07<08:12, 620kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:07<06:14, 813kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:08<04:47, 1.06MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:08<03:47, 1.33MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:08<02:57, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:08<02:47, 1.81MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<02:17, 2.20MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<02:13, 2.26MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<01:57, 2.57MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<01:52, 2.67MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:09<31:49, 158kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:09<23:00, 218kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:09<16:33, 302kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:09<12:01, 415kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:10<08:46, 569kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:10<06:44, 738kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<05:04, 979kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<04:09, 1.19MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:10<03:15, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:10<02:54, 1.71MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:11<05:20, 927kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:11<04:56, 1.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:11<03:59, 1.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:11<03:14, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<02:33, 1.92MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<02:30, 1.96MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<02:01, 2.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<02:01, 2.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<02:00, 2.44MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<01:47, 2.74MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<01:37, 3.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:13<15:19, 319kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:13<11:43, 416kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:13<08:35, 567kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:13<06:21, 765kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:13<04:46, 1.01MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<03:38, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<02:46, 1.74MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<02:25, 1.99MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<02:03, 2.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:15<08:57, 538kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:15<07:57, 604kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:15<06:05, 789kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:15<04:30, 1.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:15<03:29, 1.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<02:42, 1.76MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<02:14, 2.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<01:48, 2.62MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<01:34, 3.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:17<05:31, 859kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:17<05:15, 902kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:17<04:05, 1.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:17<03:07, 1.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<02:25, 1.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<01:54, 2.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<01:31, 3.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:19<06:09, 760kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:19<05:36, 833kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:19<04:15, 1.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:19<03:11, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:19<02:25, 1.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:19<01:51, 2.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<01:32, 2.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:21<23:27, 196kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:21<18:37, 247kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:21<13:32, 340kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:21<09:39, 475kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<06:53, 663kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:21<05:02, 904kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:23<05:16, 860kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:23<05:32, 819kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:23<04:19, 1.05MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:23<03:13, 1.40MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:23<02:29, 1.81MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:23<01:52, 2.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:25<05:02, 888kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:25<05:55, 755kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:25<04:35, 971kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:25<03:30, 1.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<02:36, 1.70MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<02:05, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:25<01:38, 2.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<01:31, 2.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:27<03:38, 1.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:27<04:00, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:27<03:10, 1.39MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:27<02:28, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:27<01:58, 2.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:27<01:35, 2.75MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<01:25, 3.04MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<01:12, 3.60MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:29<19:22, 224kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:29<14:50, 292kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:29<10:42, 403kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:29<07:42, 559kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:29<05:34, 769kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:29<04:06, 1.04MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:29<03:09, 1.35MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:31<04:34, 930kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:31<04:23, 972kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:31<03:20, 1.27MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:31<02:34, 1.65MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:31<02:00, 2.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<01:38, 2.57MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<01:26, 2.94MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<01:15, 3.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<01:13, 3.43MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:33<16:35, 253kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:33<13:11, 318kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:33<09:38, 434kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:33<07:01, 594kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:33<05:11, 802kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:33<03:51, 1.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:33<03:04, 1.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<02:22, 1.75MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<02:03, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:35<04:19, 954kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:35<04:28, 920kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:35<03:30, 1.17MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:35<02:40, 1.53MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:35<02:12, 1.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:35<01:45, 2.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:35<01:34, 2.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:35<01:18, 3.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<01:15, 3.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:37<03:23, 1.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:37<03:48, 1.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:37<02:59, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:37<02:20, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:37<01:53, 2.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:37<01:35, 2.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:37<01:21, 2.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<01:11, 3.38MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<01:04, 3.69MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:39<03:48, 1.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:39<03:53, 1.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:39<03:05, 1.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:39<02:24, 1.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:39<01:55, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<01:35, 2.48MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<01:21, 2.91MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<01:09, 3.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<35:40, 110kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:41<26:15, 149kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:41<18:42, 209kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:41<13:18, 293kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<09:28, 410kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<07:09, 542kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:41<05:09, 749kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<04:00, 962kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<03:06, 1.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<05:54, 651kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:43<05:50, 658kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:43<04:31, 849kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:43<03:25, 1.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:43<02:38, 1.45MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:43<02:13, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:43<01:47, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<01:33, 2.45MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:43<01:20, 2.83MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<01:15, 3.02MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<03:30, 1.08MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:45<03:09, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:45<02:31, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<01:59, 1.89MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<01:35, 2.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<01:26, 2.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<01:12, 3.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:45<01:08, 3.24MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:45<01:00, 3.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<05:17, 701kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:47<05:00, 740kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:47<03:53, 953kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:47<02:56, 1.26MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:47<02:22, 1.56MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<01:50, 1.99MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<01:36, 2.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<01:20, 2.71MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<01:10, 3.12MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:48<05:21, 679kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:48<04:57, 734kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:49<03:49, 949kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:49<02:54, 1.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:49<02:15, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<01:48, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<01:28, 2.43MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<01:15, 2.86MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:50<13:24, 267kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:50<10:29, 341kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:51<07:40, 464kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:51<05:33, 639kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<04:03, 873kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<03:06, 1.14MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:51<02:21, 1.49MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:51<01:55, 1.83MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<01:33, 2.25MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:52<21:26, 163kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:52<16:10, 217kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:53<11:36, 301kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:53<08:19, 419kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:53<06:01, 577kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<04:24, 784kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<03:17, 1.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<02:29, 1.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<1:02:24, 55.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<44:47, 76.7kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:55<31:35, 108kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:55<22:15, 153kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:55<15:44, 216kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<11:09, 304kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<07:59, 423kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<05:47, 582kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:56<06:58, 483kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:56<05:58, 563kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:56<04:27, 755kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:57<03:21, 996kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:57<02:51, 1.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:57<02:08, 1.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<01:46, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<01:34, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<01:19, 2.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<01:12, 2.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<03:03, 1.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<02:50, 1.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:58<02:12, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:59<01:52, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:59<01:30, 2.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:59<01:22, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<01:08, 2.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<01:05, 2.97MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<00:57, 3.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<00:57, 3.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:00<03:10, 1.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:00<03:11, 1.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:00<02:49, 1.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:00<02:10, 1.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:01<01:49, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:01<01:26, 2.22MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:01<01:17, 2.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<01:04, 2.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<01:01, 3.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<00:52, 3.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<03:24, 929kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:02<03:26, 919kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:02<02:43, 1.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:03<02:07, 1.48MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:03<01:40, 1.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:03<01:21, 2.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<01:08, 2.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<00:59, 3.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:04<06:20, 488kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:04<05:27, 566kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:04<04:07, 749kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:05<03:03, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:05<02:17, 1.34MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:05<01:50, 1.65MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<01:28, 2.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<01:12, 2.49MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:06<02:59, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:06<03:06, 974kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:06<02:27, 1.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:07<01:54, 1.57MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<01:31, 1.97MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<01:14, 2.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<01:02, 2.83MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<00:53, 3.32MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<11:10, 265kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<08:47, 336kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<06:25, 459kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:08<04:40, 628kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:09<03:26, 850kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:09<02:32, 1.14MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<02:00, 1.45MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:32, 1.87MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:18, 2.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<03:54, 738kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<03:41, 782kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:10<02:50, 1.01MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:11<02:08, 1.33MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:11<01:39, 1.71MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<01:20, 2.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<01:06, 2.55MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<00:56, 3.00MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<00:51, 3.29MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<03:18, 852kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<03:10, 885kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:12<02:29, 1.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:13<01:55, 1.46MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:13<01:28, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<01:14, 2.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<01:02, 2.67MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<00:53, 3.11MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:14<03:08, 874kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:14<03:03, 897kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:14<02:23, 1.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:14<01:48, 1.51MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:15<01:25, 1.90MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<01:10, 2.31MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<00:56, 2.85MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<00:49, 3.27MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:16<02:40, 1.00MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:16<02:41, 992kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:16<02:06, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:16<01:35, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:14, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:02, 2.51MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<00:51, 3.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<00:44, 3.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:18<07:29, 348kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:18<05:58, 436kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:18<04:22, 593kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:18<03:11, 810kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<02:19, 1.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:43, 1.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:21, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<03:14, 784kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<03:36, 703kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<02:50, 892kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:20<02:05, 1.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:20<01:44, 1.45MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<01:18, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<01:04, 2.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<00:53, 2.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<00:47, 3.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:22<04:35, 538kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:22<03:58, 620kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:22<02:56, 835kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:22<02:13, 1.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:22<01:42, 1.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<01:17, 1.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<01:03, 2.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<00:51, 2.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<02:47, 859kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<02:39, 901kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:24<02:00, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:24<01:33, 1.53MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:24<01:13, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<00:57, 2.47MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<00:49, 2.85MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:25<00:40, 3.49MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:26<07:20, 317kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:26<05:45, 405kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:26<04:11, 553kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<03:01, 764kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<02:14, 1.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:26<01:39, 1.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<01:17, 1.77MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:28<02:01, 1.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:28<02:37, 862kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:28<02:04, 1.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:28<02:59, 754kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:28<02:08, 1.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:29<01:38, 1.36MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<01:20, 1.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<01:04, 2.05MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<00:55, 2.38MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<00:47, 2.76MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:30<02:28, 887kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:30<02:08, 1.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:30<01:39, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:30<01:18, 1.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:30<01:02, 2.07MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:30<00:59, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<00:54, 2.39MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<00:47, 2.70MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<00:48, 2.64MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<00:43, 2.94MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:32<02:33, 833kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:32<02:17, 926kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:32<01:45, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:32<01:26, 1.46MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:32<01:10, 1.79MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:32<01:02, 2.03MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:32<00:52, 2.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:33<00:49, 2.54MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<00:43, 2.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<00:42, 2.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<00:38, 3.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<00:39, 3.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:34<03:54, 525kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:34<03:11, 644kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:34<02:24, 848kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:34<01:50, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:34<01:26, 1.41MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:34<01:08, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:35<01:00, 1.99MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<00:51, 2.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<00:43, 2.74MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:35<00:42, 2.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:36<02:25, 819kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:36<02:07, 937kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:36<01:39, 1.19MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:36<01:18, 1.50MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:36<01:04, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:36<00:54, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:37<00:47, 2.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<00:42, 2.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<00:38, 2.98MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:38<02:38, 727kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:38<02:15, 847kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:38<01:45, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:38<01:21, 1.40MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:38<01:04, 1.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:38<00:57, 1.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:38<00:49, 2.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<00:42, 2.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<00:37, 2.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<00:35, 3.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:40<02:44, 676kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:40<02:17, 807kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:40<01:43, 1.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:40<01:24, 1.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:40<01:07, 1.62MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:40<00:55, 1.97MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:40<00:46, 2.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<00:39, 2.75MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<00:38, 2.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<00:33, 3.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:42<06:30, 274kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:42<04:54, 363kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:42<03:32, 499kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:42<02:36, 677kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:42<01:58, 888kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:42<01:29, 1.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<01:10, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<00:57, 1.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<00:47, 2.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<00:39, 2.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:44<05:01, 341kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:44<04:12, 406kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:44<03:07, 546kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:44<02:18, 737kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:44<01:43, 978kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:44<01:19, 1.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:44<01:01, 1.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:45<00:48, 2.05MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:45<00:41, 2.37MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:46<02:04, 790kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:46<02:01, 809kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:46<01:35, 1.03MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:46<01:13, 1.34MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:46<00:56, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:46<00:44, 2.18MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:46<00:38, 2.50MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<00:31, 2.99MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<00:29, 3.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:48<02:44, 575kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:48<02:24, 654kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:48<01:49, 860kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:48<01:21, 1.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:48<01:01, 1.50MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<00:47, 1.93MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<00:36, 2.48MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:49<01:32, 976kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:50<01:29, 1.00MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:50<01:09, 1.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:50<00:53, 1.67MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:50<00:41, 2.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<00:32, 2.68MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:51<01:03, 1.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:52<01:25, 1.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:52<01:09, 1.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:52<00:51, 1.64MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<00:40, 2.08MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:52<00:30, 2.69MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:52<00:25, 3.19MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:53<01:03, 1.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:54<01:17, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:54<01:03, 1.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:54<00:46, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:54<00:35, 2.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:54<00:27, 2.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:21, 3.56MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<04:49, 270kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:56<03:49, 339kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:56<02:45, 467kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<01:57, 650kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<01:24, 898kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:56<01:00, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:57<01:17, 958kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:58<01:16, 961kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:58<00:59, 1.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:58<00:43, 1.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:31, 2.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<01:11, 977kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:59<01:26, 804kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:00<01:10, 986kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:00<00:50, 1.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<00:36, 1.82MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:26, 2.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<05:41, 192kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<04:14, 257kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:02<02:59, 360kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:02<02:05, 507kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<01:27, 712kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<01:33, 657kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<01:32, 663kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:04<01:11, 855kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:50, 1.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:36, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<00:43, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<00:53, 1.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:06<00:43, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:30, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<00:22, 2.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:39, 1.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:46, 1.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:08<00:36, 1.45MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:08<00:27, 1.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:19, 2.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<00:41, 1.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<00:46, 1.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:10<00:36, 1.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:10<00:25, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:18, 2.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<01:52, 398kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<01:35, 468kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:12<01:10, 627kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:12<00:49, 868kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:13<00:41, 971kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:13<00:50, 797kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:13<00:40, 986kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:14<00:28, 1.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<00:20, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:24, 1.47MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:26, 1.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:20, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<00:14, 2.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:16<00:10, 3.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:17<01:01, 526kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:17<00:58, 552kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:17<00:44, 721kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:18<00:30, 997kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<00:20, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:46, 609kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:38, 724kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:19<00:27, 977kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:18, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:20, 1.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:25, 954kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:21<00:19, 1.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:13, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:09, 2.21MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<02:01, 165kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<01:32, 215kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:23<01:05, 299kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:23<00:45, 418kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:28, 591kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:24, 637kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:24, 649kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:25<00:17, 859kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:25<00:12, 1.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:26<00:08, 1.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:05, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<01:29, 131kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<01:06, 174kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<00:45, 243kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<00:32, 336kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:28<00:19, 475kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:13, 661kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:10, 690kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:10, 718kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:07, 950kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:29<00:04, 1.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:29<00:02, 1.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:30<00:01, 2.22MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<00:02, 1.38MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<00:02, 1.16MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<00:01, 1.44MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:31<00:00, 1.95MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:32<00:00, 2.50MB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<15:54:10,  6.99it/s]  0%|          | 711/400000 [00:00<11:07:00,  9.98it/s]  0%|          | 1434/400000 [00:00<7:46:20, 14.24it/s]  1%|          | 2116/400000 [00:00<5:26:10, 20.33it/s]  1%|          | 2847/400000 [00:00<3:48:10, 29.01it/s]  1%|          | 3542/400000 [00:00<2:39:43, 41.37it/s]  1%|          | 4279/400000 [00:00<1:51:52, 58.95it/s]  1%|▏         | 5027/400000 [00:00<1:18:25, 83.94it/s]  1%|▏         | 5713/400000 [00:00<55:05, 119.28it/s]   2%|▏         | 6443/400000 [00:01<38:45, 169.22it/s]  2%|▏         | 7187/400000 [00:01<27:20, 239.41it/s]  2%|▏         | 7950/400000 [00:01<19:21, 337.47it/s]  2%|▏         | 8710/400000 [00:01<13:47, 473.10it/s]  2%|▏         | 9444/400000 [00:01<09:53, 657.57it/s]  3%|▎         | 10177/400000 [00:01<07:11, 904.39it/s]  3%|▎         | 10909/400000 [00:01<05:19, 1216.98it/s]  3%|▎         | 11646/400000 [00:01<03:59, 1623.56it/s]  3%|▎         | 12425/400000 [00:01<03:02, 2128.99it/s]  3%|▎         | 13192/400000 [00:01<02:22, 2717.52it/s]  3%|▎         | 13934/400000 [00:02<01:57, 3293.04it/s]  4%|▎         | 14650/400000 [00:02<01:38, 3929.15it/s]  4%|▍         | 15365/400000 [00:02<01:25, 4525.01it/s]  4%|▍         | 16117/400000 [00:02<01:14, 5139.00it/s]  4%|▍         | 16896/400000 [00:02<01:06, 5722.02it/s]  4%|▍         | 17636/400000 [00:02<01:02, 6117.96it/s]  5%|▍         | 18373/400000 [00:02<00:59, 6361.02it/s]  5%|▍         | 19099/400000 [00:02<00:58, 6517.64it/s]  5%|▍         | 19839/400000 [00:02<00:56, 6759.37it/s]  5%|▌         | 20562/400000 [00:02<00:55, 6835.97it/s]  5%|▌         | 21279/400000 [00:03<00:56, 6722.32it/s]  5%|▌         | 21975/400000 [00:03<00:56, 6657.25it/s]  6%|▌         | 22658/400000 [00:03<00:56, 6691.40it/s]  6%|▌         | 23359/400000 [00:03<00:55, 6783.28it/s]  6%|▌         | 24046/400000 [00:03<00:57, 6490.63it/s]  6%|▌         | 24704/400000 [00:03<00:58, 6441.31it/s]  6%|▋         | 25375/400000 [00:03<00:57, 6518.84it/s]  7%|▋         | 26075/400000 [00:03<00:56, 6654.59it/s]  7%|▋         | 26745/400000 [00:03<00:56, 6604.04it/s]  7%|▋         | 27409/400000 [00:04<00:57, 6483.32it/s]  7%|▋         | 28126/400000 [00:04<00:55, 6674.45it/s]  7%|▋         | 28859/400000 [00:04<00:54, 6856.90it/s]  7%|▋         | 29567/400000 [00:04<00:53, 6921.26it/s]  8%|▊         | 30289/400000 [00:04<00:52, 7005.90it/s]  8%|▊         | 31036/400000 [00:04<00:51, 7137.66it/s]  8%|▊         | 31752/400000 [00:04<00:52, 7064.55it/s]  8%|▊         | 32465/400000 [00:04<00:51, 7082.94it/s]  8%|▊         | 33190/400000 [00:04<00:51, 7129.97it/s]  8%|▊         | 33904/400000 [00:04<00:51, 7118.97it/s]  9%|▊         | 34617/400000 [00:05<00:51, 7059.98it/s]  9%|▉         | 35360/400000 [00:05<00:50, 7166.87it/s]  9%|▉         | 36078/400000 [00:05<00:52, 6887.50it/s]  9%|▉         | 36770/400000 [00:05<00:52, 6862.40it/s]  9%|▉         | 37481/400000 [00:05<00:52, 6933.23it/s] 10%|▉         | 38194/400000 [00:05<00:51, 6990.95it/s] 10%|▉         | 38895/400000 [00:05<00:53, 6796.38it/s] 10%|▉         | 39577/400000 [00:05<00:54, 6642.60it/s] 10%|█         | 40244/400000 [00:05<00:54, 6598.82it/s] 10%|█         | 40906/400000 [00:05<00:56, 6410.23it/s] 10%|█         | 41558/400000 [00:06<00:55, 6440.74it/s] 11%|█         | 42256/400000 [00:06<00:54, 6591.92it/s] 11%|█         | 42954/400000 [00:06<00:53, 6702.00it/s] 11%|█         | 43653/400000 [00:06<00:52, 6784.20it/s] 11%|█         | 44368/400000 [00:06<00:51, 6889.84it/s] 11%|█▏        | 45059/400000 [00:06<00:51, 6860.73it/s] 11%|█▏        | 45747/400000 [00:06<00:53, 6613.04it/s] 12%|█▏        | 46411/400000 [00:06<00:53, 6555.12it/s] 12%|█▏        | 47110/400000 [00:06<00:52, 6678.22it/s] 12%|█▏        | 47780/400000 [00:07<00:54, 6505.75it/s] 12%|█▏        | 48433/400000 [00:07<00:54, 6436.19it/s] 12%|█▏        | 49122/400000 [00:07<00:53, 6565.61it/s] 12%|█▏        | 49781/400000 [00:07<00:53, 6568.87it/s] 13%|█▎        | 50452/400000 [00:07<00:52, 6609.88it/s] 13%|█▎        | 51142/400000 [00:07<00:52, 6692.76it/s] 13%|█▎        | 51839/400000 [00:07<00:51, 6771.70it/s] 13%|█▎        | 52518/400000 [00:07<00:53, 6469.26it/s] 13%|█▎        | 53169/400000 [00:07<00:54, 6378.52it/s] 13%|█▎        | 53841/400000 [00:07<00:53, 6475.55it/s] 14%|█▎        | 54519/400000 [00:08<00:52, 6562.93it/s] 14%|█▍        | 55180/400000 [00:08<00:52, 6575.45it/s] 14%|█▍        | 55848/400000 [00:08<00:52, 6603.65it/s] 14%|█▍        | 56510/400000 [00:08<00:52, 6515.50it/s] 14%|█▍        | 57224/400000 [00:08<00:51, 6690.92it/s] 14%|█▍        | 57954/400000 [00:08<00:49, 6861.18it/s] 15%|█▍        | 58643/400000 [00:08<00:50, 6792.19it/s] 15%|█▍        | 59366/400000 [00:08<00:49, 6917.67it/s] 15%|█▌        | 60060/400000 [00:08<00:50, 6790.26it/s] 15%|█▌        | 60761/400000 [00:08<00:49, 6854.06it/s] 15%|█▌        | 61451/400000 [00:09<00:49, 6866.69it/s] 16%|█▌        | 62139/400000 [00:09<00:49, 6805.75it/s] 16%|█▌        | 62821/400000 [00:09<00:50, 6696.21it/s] 16%|█▌        | 63509/400000 [00:09<00:49, 6747.63it/s] 16%|█▌        | 64185/400000 [00:09<00:49, 6751.24it/s] 16%|█▌        | 64914/400000 [00:09<00:48, 6902.11it/s] 16%|█▋        | 65606/400000 [00:09<00:49, 6798.46it/s] 17%|█▋        | 66288/400000 [00:09<00:49, 6744.63it/s] 17%|█▋        | 67001/400000 [00:09<00:48, 6854.97it/s] 17%|█▋        | 67688/400000 [00:09<00:48, 6821.40it/s] 17%|█▋        | 68381/400000 [00:10<00:48, 6853.00it/s] 17%|█▋        | 69067/400000 [00:10<00:48, 6837.23it/s] 17%|█▋        | 69765/400000 [00:10<00:48, 6877.88it/s] 18%|█▊        | 70454/400000 [00:10<00:49, 6702.41it/s] 18%|█▊        | 71127/400000 [00:10<00:49, 6709.96it/s] 18%|█▊        | 71817/400000 [00:10<00:48, 6764.27it/s] 18%|█▊        | 72495/400000 [00:10<00:48, 6714.01it/s] 18%|█▊        | 73197/400000 [00:10<00:48, 6784.18it/s] 18%|█▊        | 73877/400000 [00:10<00:50, 6480.94it/s] 19%|█▊        | 74539/400000 [00:11<00:49, 6520.62it/s] 19%|█▉        | 75236/400000 [00:11<00:48, 6648.70it/s] 19%|█▉        | 75904/400000 [00:11<00:49, 6609.35it/s] 19%|█▉        | 76603/400000 [00:11<00:48, 6717.06it/s] 19%|█▉        | 77277/400000 [00:11<00:48, 6661.84it/s] 19%|█▉        | 77984/400000 [00:11<00:47, 6777.87it/s] 20%|█▉        | 78688/400000 [00:11<00:46, 6853.96it/s] 20%|█▉        | 79404/400000 [00:11<00:46, 6942.30it/s] 20%|██        | 80100/400000 [00:11<00:46, 6945.10it/s] 20%|██        | 80800/400000 [00:11<00:45, 6959.52it/s] 20%|██        | 81539/400000 [00:12<00:45, 7070.34it/s] 21%|██        | 82247/400000 [00:12<00:45, 6989.75it/s] 21%|██        | 82947/400000 [00:12<00:52, 6047.06it/s] 21%|██        | 83622/400000 [00:12<00:50, 6240.48it/s] 21%|██        | 84312/400000 [00:12<00:49, 6422.34it/s] 21%|██▏       | 85007/400000 [00:12<00:47, 6571.88it/s] 21%|██▏       | 85699/400000 [00:12<00:47, 6671.99it/s] 22%|██▏       | 86386/400000 [00:12<00:46, 6727.89it/s] 22%|██▏       | 87096/400000 [00:12<00:45, 6834.76it/s] 22%|██▏       | 87794/400000 [00:12<00:45, 6876.29it/s] 22%|██▏       | 88485/400000 [00:13<00:45, 6794.81it/s] 22%|██▏       | 89206/400000 [00:13<00:44, 6912.06it/s] 22%|██▏       | 89931/400000 [00:13<00:44, 7009.33it/s] 23%|██▎       | 90634/400000 [00:13<00:44, 7013.23it/s] 23%|██▎       | 91337/400000 [00:13<00:45, 6739.44it/s] 23%|██▎       | 92015/400000 [00:13<00:45, 6713.55it/s] 23%|██▎       | 92689/400000 [00:13<00:45, 6689.25it/s] 23%|██▎       | 93360/400000 [00:13<00:46, 6530.90it/s] 24%|██▎       | 94029/400000 [00:13<00:46, 6575.96it/s] 24%|██▎       | 94689/400000 [00:14<00:48, 6350.57it/s] 24%|██▍       | 95393/400000 [00:14<00:46, 6541.96it/s] 24%|██▍       | 96051/400000 [00:14<00:46, 6530.60it/s] 24%|██▍       | 96736/400000 [00:14<00:45, 6622.05it/s] 24%|██▍       | 97401/400000 [00:14<00:47, 6362.99it/s] 25%|██▍       | 98041/400000 [00:14<00:48, 6201.10it/s] 25%|██▍       | 98734/400000 [00:14<00:47, 6401.49it/s] 25%|██▍       | 99379/400000 [00:14<00:46, 6412.76it/s] 25%|██▌       | 100092/400000 [00:14<00:45, 6610.86it/s] 25%|██▌       | 100760/400000 [00:14<00:45, 6627.87it/s] 25%|██▌       | 101426/400000 [00:15<00:47, 6265.59it/s] 26%|██▌       | 102111/400000 [00:15<00:46, 6427.90it/s] 26%|██▌       | 102825/400000 [00:15<00:44, 6624.37it/s] 26%|██▌       | 103511/400000 [00:15<00:44, 6692.97it/s] 26%|██▌       | 104185/400000 [00:15<00:44, 6630.12it/s] 26%|██▌       | 104909/400000 [00:15<00:43, 6801.65it/s] 26%|██▋       | 105593/400000 [00:15<00:44, 6650.96it/s] 27%|██▋       | 106261/400000 [00:15<00:44, 6597.65it/s] 27%|██▋       | 106966/400000 [00:15<00:43, 6725.96it/s] 27%|██▋       | 107641/400000 [00:15<00:45, 6492.41it/s] 27%|██▋       | 108326/400000 [00:16<00:44, 6594.95it/s] 27%|██▋       | 108989/400000 [00:16<00:45, 6406.16it/s] 27%|██▋       | 109651/400000 [00:16<00:44, 6468.70it/s] 28%|██▊       | 110301/400000 [00:16<00:46, 6229.80it/s] 28%|██▊       | 110989/400000 [00:16<00:45, 6409.53it/s] 28%|██▊       | 111754/400000 [00:16<00:42, 6735.84it/s] 28%|██▊       | 112513/400000 [00:16<00:41, 6970.05it/s] 28%|██▊       | 113243/400000 [00:16<00:40, 7064.30it/s] 29%|██▊       | 114030/400000 [00:16<00:39, 7287.69it/s] 29%|██▊       | 114776/400000 [00:17<00:38, 7337.65it/s] 29%|██▉       | 115514/400000 [00:17<00:39, 7256.84it/s] 29%|██▉       | 116243/400000 [00:17<00:39, 7111.11it/s] 29%|██▉       | 116957/400000 [00:17<00:41, 6860.98it/s] 29%|██▉       | 117647/400000 [00:17<00:41, 6862.36it/s] 30%|██▉       | 118338/400000 [00:17<00:40, 6875.19it/s] 30%|██▉       | 119032/400000 [00:17<00:40, 6892.17it/s] 30%|██▉       | 119772/400000 [00:17<00:39, 7035.32it/s] 30%|███       | 120478/400000 [00:17<00:40, 6937.42it/s] 30%|███       | 121174/400000 [00:17<00:41, 6795.55it/s] 30%|███       | 121881/400000 [00:18<00:40, 6874.93it/s] 31%|███       | 122577/400000 [00:18<00:40, 6898.17it/s] 31%|███       | 123315/400000 [00:18<00:39, 7034.89it/s] 31%|███       | 124027/400000 [00:18<00:39, 7058.95it/s] 31%|███       | 124734/400000 [00:18<00:39, 7034.12it/s] 31%|███▏      | 125439/400000 [00:18<00:40, 6825.65it/s] 32%|███▏      | 126129/400000 [00:18<00:40, 6845.37it/s] 32%|███▏      | 126815/400000 [00:18<00:40, 6816.42it/s] 32%|███▏      | 127498/400000 [00:18<00:40, 6781.18it/s] 32%|███▏      | 128186/400000 [00:18<00:39, 6809.44it/s] 32%|███▏      | 128868/400000 [00:19<00:40, 6712.64it/s] 32%|███▏      | 129584/400000 [00:19<00:39, 6840.86it/s] 33%|███▎      | 130325/400000 [00:19<00:38, 7000.53it/s] 33%|███▎      | 131064/400000 [00:19<00:37, 7112.31it/s] 33%|███▎      | 131812/400000 [00:19<00:37, 7218.37it/s] 33%|███▎      | 132536/400000 [00:19<00:37, 7162.56it/s] 33%|███▎      | 133310/400000 [00:19<00:36, 7324.58it/s] 34%|███▎      | 134045/400000 [00:19<00:36, 7296.81it/s] 34%|███▎      | 134797/400000 [00:19<00:36, 7361.34it/s] 34%|███▍      | 135584/400000 [00:19<00:35, 7505.09it/s] 34%|███▍      | 136336/400000 [00:20<00:35, 7501.11it/s] 34%|███▍      | 137109/400000 [00:20<00:34, 7565.88it/s] 34%|███▍      | 137929/400000 [00:20<00:33, 7744.58it/s] 35%|███▍      | 138751/400000 [00:20<00:33, 7881.23it/s] 35%|███▍      | 139541/400000 [00:20<00:33, 7847.39it/s] 35%|███▌      | 140327/400000 [00:20<00:33, 7685.81it/s] 35%|███▌      | 141098/400000 [00:20<00:33, 7687.71it/s] 35%|███▌      | 141868/400000 [00:20<00:34, 7572.06it/s] 36%|███▌      | 142664/400000 [00:20<00:33, 7681.95it/s] 36%|███▌      | 143459/400000 [00:20<00:33, 7757.43it/s] 36%|███▌      | 144236/400000 [00:21<00:33, 7647.26it/s] 36%|███▋      | 145010/400000 [00:21<00:33, 7673.45it/s] 36%|███▋      | 145779/400000 [00:21<00:33, 7636.93it/s] 37%|███▋      | 146544/400000 [00:21<00:33, 7596.80it/s] 37%|███▋      | 147305/400000 [00:21<00:33, 7578.63it/s] 37%|███▋      | 148064/400000 [00:21<00:33, 7563.18it/s] 37%|███▋      | 148846/400000 [00:21<00:32, 7636.18it/s] 37%|███▋      | 149634/400000 [00:21<00:32, 7705.76it/s] 38%|███▊      | 150416/400000 [00:21<00:32, 7735.22it/s] 38%|███▊      | 151193/400000 [00:22<00:32, 7745.57it/s] 38%|███▊      | 151977/400000 [00:22<00:31, 7770.92it/s] 38%|███▊      | 152787/400000 [00:22<00:31, 7864.21it/s] 38%|███▊      | 153587/400000 [00:22<00:31, 7902.76it/s] 39%|███▊      | 154396/400000 [00:22<00:30, 7955.82it/s] 39%|███▉      | 155231/400000 [00:22<00:30, 8069.13it/s] 39%|███▉      | 156039/400000 [00:22<00:30, 7949.33it/s] 39%|███▉      | 156835/400000 [00:22<00:30, 7894.14it/s] 39%|███▉      | 157626/400000 [00:22<00:31, 7701.34it/s] 40%|███▉      | 158402/400000 [00:22<00:31, 7718.75it/s] 40%|███▉      | 159198/400000 [00:23<00:30, 7788.94it/s] 40%|███▉      | 159978/400000 [00:23<00:30, 7774.97it/s] 40%|████      | 160764/400000 [00:23<00:30, 7798.18it/s] 40%|████      | 161545/400000 [00:23<00:30, 7737.45it/s] 41%|████      | 162338/400000 [00:23<00:30, 7791.29it/s] 41%|████      | 163128/400000 [00:23<00:30, 7823.51it/s] 41%|████      | 163911/400000 [00:23<00:30, 7676.06it/s] 41%|████      | 164680/400000 [00:23<00:31, 7553.29it/s] 41%|████▏     | 165437/400000 [00:23<00:31, 7481.11it/s] 42%|████▏     | 166186/400000 [00:23<00:31, 7421.42it/s] 42%|████▏     | 166960/400000 [00:24<00:31, 7514.18it/s] 42%|████▏     | 167730/400000 [00:24<00:30, 7567.88it/s] 42%|████▏     | 168511/400000 [00:24<00:30, 7637.72it/s] 42%|████▏     | 169307/400000 [00:24<00:29, 7729.29it/s] 43%|████▎     | 170081/400000 [00:24<00:29, 7690.82it/s] 43%|████▎     | 170851/400000 [00:24<00:30, 7557.07it/s] 43%|████▎     | 171608/400000 [00:24<00:30, 7480.94it/s] 43%|████▎     | 172377/400000 [00:24<00:30, 7541.74it/s] 43%|████▎     | 173132/400000 [00:24<00:30, 7369.68it/s] 43%|████▎     | 173884/400000 [00:24<00:30, 7413.19it/s] 44%|████▎     | 174635/400000 [00:25<00:30, 7439.43it/s] 44%|████▍     | 175380/400000 [00:25<00:30, 7425.94it/s] 44%|████▍     | 176163/400000 [00:25<00:29, 7541.73it/s] 44%|████▍     | 176935/400000 [00:25<00:29, 7593.66it/s] 44%|████▍     | 177725/400000 [00:25<00:28, 7680.16it/s] 45%|████▍     | 178494/400000 [00:25<00:28, 7677.53it/s] 45%|████▍     | 179263/400000 [00:25<00:29, 7580.93it/s] 45%|████▌     | 180023/400000 [00:25<00:29, 7579.59it/s] 45%|████▌     | 180786/400000 [00:25<00:28, 7592.18it/s] 45%|████▌     | 181572/400000 [00:25<00:28, 7669.92it/s] 46%|████▌     | 182353/400000 [00:26<00:28, 7710.93it/s] 46%|████▌     | 183125/400000 [00:26<00:28, 7531.76it/s] 46%|████▌     | 183904/400000 [00:26<00:28, 7604.23it/s] 46%|████▌     | 184686/400000 [00:26<00:28, 7667.21it/s] 46%|████▋     | 185459/400000 [00:26<00:27, 7684.94it/s] 47%|████▋     | 186229/400000 [00:26<00:27, 7670.83it/s] 47%|████▋     | 187004/400000 [00:26<00:27, 7692.95it/s] 47%|████▋     | 187774/400000 [00:26<00:27, 7674.92it/s] 47%|████▋     | 188568/400000 [00:26<00:27, 7751.61it/s] 47%|████▋     | 189352/400000 [00:26<00:27, 7775.14it/s] 48%|████▊     | 190130/400000 [00:27<00:27, 7755.02it/s] 48%|████▊     | 190906/400000 [00:27<00:27, 7712.33it/s] 48%|████▊     | 191696/400000 [00:27<00:26, 7763.75it/s] 48%|████▊     | 192473/400000 [00:27<00:26, 7759.79it/s] 48%|████▊     | 193255/400000 [00:27<00:26, 7776.30it/s] 49%|████▊     | 194041/400000 [00:27<00:26, 7801.20it/s] 49%|████▊     | 194822/400000 [00:27<00:26, 7759.89it/s] 49%|████▉     | 195630/400000 [00:27<00:26, 7852.46it/s] 49%|████▉     | 196416/400000 [00:27<00:26, 7651.75it/s] 49%|████▉     | 197183/400000 [00:27<00:26, 7641.95it/s] 49%|████▉     | 197961/400000 [00:28<00:26, 7682.71it/s] 50%|████▉     | 198730/400000 [00:28<00:26, 7612.03it/s] 50%|████▉     | 199492/400000 [00:28<00:26, 7595.39it/s] 50%|█████     | 200272/400000 [00:28<00:26, 7653.80it/s] 50%|█████     | 201091/400000 [00:28<00:25, 7806.09it/s] 50%|█████     | 201887/400000 [00:28<00:25, 7849.11it/s] 51%|█████     | 202673/400000 [00:28<00:25, 7593.43it/s] 51%|█████     | 203460/400000 [00:28<00:25, 7672.19it/s] 51%|█████     | 204230/400000 [00:28<00:25, 7668.58it/s] 51%|█████     | 204999/400000 [00:29<00:25, 7541.22it/s] 51%|█████▏    | 205791/400000 [00:29<00:25, 7648.84it/s] 52%|█████▏    | 206558/400000 [00:29<00:25, 7563.93it/s] 52%|█████▏    | 207357/400000 [00:29<00:25, 7685.21it/s] 52%|█████▏    | 208127/400000 [00:29<00:25, 7668.98it/s] 52%|█████▏    | 208895/400000 [00:29<00:25, 7501.14it/s] 52%|█████▏    | 209697/400000 [00:29<00:24, 7649.36it/s] 53%|█████▎    | 210464/400000 [00:29<00:24, 7618.73it/s] 53%|█████▎    | 211237/400000 [00:29<00:24, 7651.09it/s] 53%|█████▎    | 212023/400000 [00:29<00:24, 7711.16it/s] 53%|█████▎    | 212795/400000 [00:30<00:24, 7650.09it/s] 53%|█████▎    | 213565/400000 [00:30<00:24, 7663.61it/s] 54%|█████▎    | 214332/400000 [00:30<00:24, 7585.76it/s] 54%|█████▍    | 215139/400000 [00:30<00:23, 7723.84it/s] 54%|█████▍    | 215917/400000 [00:30<00:23, 7738.95it/s] 54%|█████▍    | 216707/400000 [00:30<00:23, 7786.41it/s] 54%|█████▍    | 217487/400000 [00:30<00:23, 7744.26it/s] 55%|█████▍    | 218262/400000 [00:30<00:23, 7708.49it/s] 55%|█████▍    | 219074/400000 [00:30<00:23, 7826.78it/s] 55%|█████▍    | 219858/400000 [00:30<00:23, 7777.73it/s] 55%|█████▌    | 220641/400000 [00:31<00:23, 7792.55it/s] 55%|█████▌    | 221436/400000 [00:31<00:22, 7838.72it/s] 56%|█████▌    | 222221/400000 [00:31<00:23, 7641.11it/s] 56%|█████▌    | 222987/400000 [00:31<00:23, 7601.81it/s] 56%|█████▌    | 223749/400000 [00:31<00:23, 7558.63it/s] 56%|█████▌    | 224506/400000 [00:31<00:23, 7542.87it/s] 56%|█████▋    | 225261/400000 [00:31<00:23, 7346.53it/s] 57%|█████▋    | 226004/400000 [00:31<00:23, 7368.89it/s] 57%|█████▋    | 226746/400000 [00:31<00:23, 7382.84it/s] 57%|█████▋    | 227577/400000 [00:31<00:22, 7638.27it/s] 57%|█████▋    | 228365/400000 [00:32<00:22, 7708.97it/s] 57%|█████▋    | 229138/400000 [00:32<00:22, 7707.82it/s] 57%|█████▋    | 229911/400000 [00:32<00:22, 7566.33it/s] 58%|█████▊    | 230675/400000 [00:32<00:22, 7587.21it/s] 58%|█████▊    | 231492/400000 [00:32<00:21, 7751.11it/s] 58%|█████▊    | 232303/400000 [00:32<00:21, 7854.31it/s] 58%|█████▊    | 233140/400000 [00:32<00:20, 8001.44it/s] 58%|█████▊    | 233942/400000 [00:32<00:20, 7924.50it/s] 59%|█████▊    | 234736/400000 [00:32<00:21, 7689.90it/s] 59%|█████▉    | 235509/400000 [00:32<00:21, 7700.80it/s] 59%|█████▉    | 236286/400000 [00:33<00:21, 7720.73it/s] 59%|█████▉    | 237060/400000 [00:33<00:21, 7709.49it/s] 59%|█████▉    | 237850/400000 [00:33<00:20, 7764.87it/s] 60%|█████▉    | 238628/400000 [00:33<00:20, 7725.06it/s] 60%|█████▉    | 239402/400000 [00:33<00:21, 7592.81it/s] 60%|██████    | 240185/400000 [00:33<00:20, 7660.37it/s] 60%|██████    | 240985/400000 [00:33<00:20, 7758.12it/s] 60%|██████    | 241789/400000 [00:33<00:20, 7839.96it/s] 61%|██████    | 242586/400000 [00:33<00:19, 7878.22it/s] 61%|██████    | 243375/400000 [00:34<00:20, 7765.90it/s] 61%|██████    | 244154/400000 [00:34<00:20, 7771.91it/s] 61%|██████    | 244932/400000 [00:34<00:20, 7565.94it/s] 61%|██████▏   | 245691/400000 [00:34<00:20, 7384.39it/s] 62%|██████▏   | 246464/400000 [00:34<00:20, 7484.54it/s] 62%|██████▏   | 247215/400000 [00:34<00:20, 7379.56it/s] 62%|██████▏   | 247955/400000 [00:34<00:20, 7297.51it/s] 62%|██████▏   | 248730/400000 [00:34<00:20, 7426.99it/s] 62%|██████▏   | 249475/400000 [00:34<00:20, 7313.93it/s] 63%|██████▎   | 250210/400000 [00:34<00:20, 7323.46it/s] 63%|██████▎   | 250974/400000 [00:35<00:20, 7413.66it/s] 63%|██████▎   | 251717/400000 [00:35<00:20, 7377.74it/s] 63%|██████▎   | 252469/400000 [00:35<00:19, 7418.95it/s] 63%|██████▎   | 253260/400000 [00:35<00:19, 7558.43it/s] 64%|██████▎   | 254057/400000 [00:35<00:19, 7676.46it/s] 64%|██████▎   | 254851/400000 [00:35<00:18, 7751.70it/s] 64%|██████▍   | 255647/400000 [00:35<00:18, 7812.08it/s] 64%|██████▍   | 256430/400000 [00:35<00:18, 7782.55it/s] 64%|██████▍   | 257209/400000 [00:35<00:18, 7709.28it/s] 64%|██████▍   | 257981/400000 [00:35<00:18, 7632.56it/s] 65%|██████▍   | 258774/400000 [00:36<00:18, 7719.35it/s] 65%|██████▍   | 259581/400000 [00:36<00:17, 7820.39it/s] 65%|██████▌   | 260364/400000 [00:36<00:18, 7701.56it/s] 65%|██████▌   | 261136/400000 [00:36<00:18, 7651.04it/s] 65%|██████▌   | 261902/400000 [00:36<00:18, 7555.44it/s] 66%|██████▌   | 262674/400000 [00:36<00:18, 7601.63it/s] 66%|██████▌   | 263459/400000 [00:36<00:17, 7672.05it/s] 66%|██████▌   | 264227/400000 [00:36<00:17, 7663.30it/s] 66%|██████▌   | 264994/400000 [00:36<00:17, 7554.67it/s] 66%|██████▋   | 265756/400000 [00:36<00:17, 7572.69it/s] 67%|██████▋   | 266522/400000 [00:37<00:17, 7596.04it/s] 67%|██████▋   | 267282/400000 [00:37<00:17, 7562.13it/s] 67%|██████▋   | 268054/400000 [00:37<00:17, 7608.55it/s] 67%|██████▋   | 268816/400000 [00:37<00:17, 7591.99it/s] 67%|██████▋   | 269576/400000 [00:37<00:17, 7589.81it/s] 68%|██████▊   | 270364/400000 [00:37<00:16, 7673.16it/s] 68%|██████▊   | 271156/400000 [00:37<00:16, 7743.95it/s] 68%|██████▊   | 271931/400000 [00:37<00:16, 7744.36it/s] 68%|██████▊   | 272706/400000 [00:37<00:16, 7644.42it/s] 68%|██████▊   | 273471/400000 [00:37<00:16, 7555.96it/s] 69%|██████▊   | 274228/400000 [00:38<00:17, 7396.74it/s] 69%|██████▊   | 274969/400000 [00:38<00:17, 7169.56it/s] 69%|██████▉   | 275689/400000 [00:38<00:17, 7100.55it/s] 69%|██████▉   | 276406/400000 [00:38<00:17, 7120.62it/s] 69%|██████▉   | 277183/400000 [00:38<00:16, 7302.08it/s] 69%|██████▉   | 277916/400000 [00:38<00:16, 7300.72it/s] 70%|██████▉   | 278648/400000 [00:38<00:16, 7283.91it/s] 70%|██████▉   | 279418/400000 [00:38<00:16, 7403.37it/s] 70%|███████   | 280173/400000 [00:38<00:16, 7444.61it/s] 70%|███████   | 280933/400000 [00:38<00:15, 7488.89it/s] 70%|███████   | 281705/400000 [00:39<00:15, 7554.10it/s] 71%|███████   | 282493/400000 [00:39<00:15, 7646.96it/s] 71%|███████   | 283277/400000 [00:39<00:15, 7702.02it/s] 71%|███████   | 284048/400000 [00:39<00:15, 7633.88it/s] 71%|███████   | 284812/400000 [00:39<00:15, 7611.98it/s] 71%|███████▏  | 285598/400000 [00:39<00:14, 7684.10it/s] 72%|███████▏  | 286367/400000 [00:39<00:14, 7629.48it/s] 72%|███████▏  | 287131/400000 [00:39<00:14, 7619.62it/s] 72%|███████▏  | 287901/400000 [00:39<00:14, 7643.48it/s] 72%|███████▏  | 288706/400000 [00:40<00:14, 7759.27it/s] 72%|███████▏  | 289483/400000 [00:40<00:14, 7718.08it/s] 73%|███████▎  | 290256/400000 [00:40<00:14, 7701.62it/s] 73%|███████▎  | 291051/400000 [00:40<00:14, 7773.96it/s] 73%|███████▎  | 291829/400000 [00:40<00:13, 7772.36it/s] 73%|███████▎  | 292650/400000 [00:40<00:13, 7897.04it/s] 73%|███████▎  | 293465/400000 [00:40<00:13, 7970.44it/s] 74%|███████▎  | 294263/400000 [00:40<00:13, 7961.61it/s] 74%|███████▍  | 295060/400000 [00:40<00:13, 7904.36it/s] 74%|███████▍  | 295851/400000 [00:40<00:13, 7890.23it/s] 74%|███████▍  | 296641/400000 [00:41<00:13, 7872.07it/s] 74%|███████▍  | 297429/400000 [00:41<00:13, 7862.86it/s] 75%|███████▍  | 298216/400000 [00:41<00:13, 7812.65it/s] 75%|███████▍  | 298998/400000 [00:41<00:13, 7737.32it/s] 75%|███████▍  | 299773/400000 [00:41<00:13, 7608.52it/s] 75%|███████▌  | 300548/400000 [00:41<00:13, 7648.49it/s] 75%|███████▌  | 301315/400000 [00:41<00:12, 7654.48it/s] 76%|███████▌  | 302114/400000 [00:41<00:12, 7752.08it/s] 76%|███████▌  | 302902/400000 [00:41<00:12, 7789.54it/s] 76%|███████▌  | 303703/400000 [00:41<00:12, 7851.27it/s] 76%|███████▌  | 304489/400000 [00:42<00:12, 7789.79it/s] 76%|███████▋  | 305269/400000 [00:42<00:12, 7620.47it/s] 77%|███████▋  | 306033/400000 [00:42<00:12, 7524.55it/s] 77%|███████▋  | 306795/400000 [00:42<00:12, 7551.89it/s] 77%|███████▋  | 307551/400000 [00:42<00:12, 7523.25it/s] 77%|███████▋  | 308361/400000 [00:42<00:11, 7687.01it/s] 77%|███████▋  | 309131/400000 [00:42<00:12, 7507.45it/s] 77%|███████▋  | 309939/400000 [00:42<00:11, 7670.49it/s] 78%|███████▊  | 310709/400000 [00:42<00:11, 7637.31it/s] 78%|███████▊  | 311475/400000 [00:42<00:12, 7356.48it/s] 78%|███████▊  | 312214/400000 [00:43<00:12, 7253.29it/s] 78%|███████▊  | 312979/400000 [00:43<00:11, 7364.65it/s] 78%|███████▊  | 313798/400000 [00:43<00:11, 7591.49it/s] 79%|███████▊  | 314601/400000 [00:43<00:11, 7714.48it/s] 79%|███████▉  | 315376/400000 [00:43<00:11, 7662.33it/s] 79%|███████▉  | 316149/400000 [00:43<00:10, 7682.03it/s] 79%|███████▉  | 316946/400000 [00:43<00:10, 7765.43it/s] 79%|███████▉  | 317761/400000 [00:43<00:10, 7876.16it/s] 80%|███████▉  | 318583/400000 [00:43<00:10, 7973.93it/s] 80%|███████▉  | 319382/400000 [00:43<00:10, 7928.52it/s] 80%|████████  | 320192/400000 [00:44<00:10, 7977.00it/s] 80%|████████  | 320991/400000 [00:44<00:09, 7943.61it/s] 80%|████████  | 321786/400000 [00:44<00:09, 7900.45it/s] 81%|████████  | 322577/400000 [00:44<00:09, 7862.65it/s] 81%|████████  | 323364/400000 [00:44<00:09, 7816.44it/s] 81%|████████  | 324146/400000 [00:44<00:09, 7735.22it/s] 81%|████████  | 324920/400000 [00:44<00:09, 7731.81it/s] 81%|████████▏ | 325694/400000 [00:44<00:09, 7542.31it/s] 82%|████████▏ | 326474/400000 [00:44<00:09, 7617.07it/s] 82%|████████▏ | 327237/400000 [00:44<00:09, 7605.82it/s] 82%|████████▏ | 328030/400000 [00:45<00:09, 7698.40it/s] 82%|████████▏ | 328810/400000 [00:45<00:09, 7727.77it/s] 82%|████████▏ | 329609/400000 [00:45<00:09, 7804.13it/s] 83%|████████▎ | 330409/400000 [00:45<00:08, 7857.90it/s] 83%|████████▎ | 331198/400000 [00:45<00:08, 7866.81it/s] 83%|████████▎ | 331986/400000 [00:45<00:08, 7651.60it/s] 83%|████████▎ | 332768/400000 [00:45<00:08, 7699.54it/s] 83%|████████▎ | 333558/400000 [00:45<00:08, 7758.16it/s] 84%|████████▎ | 334355/400000 [00:45<00:08, 7816.58it/s] 84%|████████▍ | 335138/400000 [00:46<00:08, 7679.41it/s] 84%|████████▍ | 335907/400000 [00:46<00:08, 7349.20it/s] 84%|████████▍ | 336682/400000 [00:46<00:08, 7463.31it/s] 84%|████████▍ | 337432/400000 [00:46<00:08, 7425.85it/s] 85%|████████▍ | 338177/400000 [00:46<00:08, 7389.02it/s] 85%|████████▍ | 338918/400000 [00:46<00:08, 7350.87it/s] 85%|████████▍ | 339676/400000 [00:46<00:08, 7416.39it/s] 85%|████████▌ | 340419/400000 [00:46<00:08, 7412.60it/s] 85%|████████▌ | 341178/400000 [00:46<00:07, 7462.67it/s] 85%|████████▌ | 341949/400000 [00:46<00:07, 7532.25it/s] 86%|████████▌ | 342703/400000 [00:47<00:07, 7512.36it/s] 86%|████████▌ | 343463/400000 [00:47<00:07, 7535.11it/s] 86%|████████▌ | 344217/400000 [00:47<00:07, 7429.88it/s] 86%|████████▌ | 344967/400000 [00:47<00:07, 7449.21it/s] 86%|████████▋ | 345713/400000 [00:47<00:07, 7322.09it/s] 87%|████████▋ | 346446/400000 [00:47<00:07, 7287.26it/s] 87%|████████▋ | 347195/400000 [00:47<00:07, 7346.63it/s] 87%|████████▋ | 347972/400000 [00:47<00:06, 7468.30it/s] 87%|████████▋ | 348728/400000 [00:47<00:06, 7491.71it/s] 87%|████████▋ | 349479/400000 [00:47<00:06, 7493.86it/s] 88%|████████▊ | 350229/400000 [00:48<00:06, 7316.00it/s] 88%|████████▊ | 350962/400000 [00:48<00:06, 7201.09it/s] 88%|████████▊ | 351704/400000 [00:48<00:06, 7264.92it/s] 88%|████████▊ | 352461/400000 [00:48<00:06, 7351.98it/s] 88%|████████▊ | 353224/400000 [00:48<00:06, 7431.59it/s] 88%|████████▊ | 353969/400000 [00:48<00:06, 7216.81it/s] 89%|████████▊ | 354697/400000 [00:48<00:06, 7235.50it/s] 89%|████████▉ | 355451/400000 [00:48<00:06, 7321.86it/s] 89%|████████▉ | 356185/400000 [00:48<00:05, 7318.47it/s] 89%|████████▉ | 356918/400000 [00:48<00:05, 7211.43it/s] 89%|████████▉ | 357648/400000 [00:49<00:05, 7236.24it/s] 90%|████████▉ | 358408/400000 [00:49<00:05, 7340.86it/s] 90%|████████▉ | 359165/400000 [00:49<00:05, 7407.10it/s] 90%|████████▉ | 359931/400000 [00:49<00:05, 7477.60it/s] 90%|█████████ | 360680/400000 [00:49<00:05, 7448.01it/s] 90%|█████████ | 361426/400000 [00:49<00:05, 7409.96it/s] 91%|█████████ | 362168/400000 [00:49<00:05, 7406.98it/s] 91%|█████████ | 362909/400000 [00:49<00:05, 7316.42it/s] 91%|█████████ | 363642/400000 [00:49<00:05, 7154.43it/s] 91%|█████████ | 364379/400000 [00:49<00:04, 7217.26it/s] 91%|█████████▏| 365102/400000 [00:50<00:04, 7157.33it/s] 91%|█████████▏| 365885/400000 [00:50<00:04, 7346.36it/s] 92%|█████████▏| 366660/400000 [00:50<00:04, 7459.46it/s] 92%|█████████▏| 367408/400000 [00:50<00:04, 7432.32it/s] 92%|█████████▏| 368199/400000 [00:50<00:04, 7569.03it/s] 92%|█████████▏| 368958/400000 [00:50<00:04, 7308.55it/s] 92%|█████████▏| 369694/400000 [00:50<00:04, 7322.88it/s] 93%|█████████▎| 370437/400000 [00:50<00:04, 7353.10it/s] 93%|█████████▎| 371198/400000 [00:50<00:03, 7428.21it/s] 93%|█████████▎| 371987/400000 [00:51<00:03, 7558.87it/s] 93%|█████████▎| 372745/400000 [00:51<00:03, 7517.02it/s] 93%|█████████▎| 373498/400000 [00:51<00:03, 7465.99it/s] 94%|█████████▎| 374246/400000 [00:51<00:03, 7466.13it/s] 94%|█████████▍| 375006/400000 [00:51<00:03, 7504.04it/s] 94%|█████████▍| 375757/400000 [00:51<00:03, 7190.47it/s] 94%|█████████▍| 376502/400000 [00:51<00:03, 7264.80it/s] 94%|█████████▍| 377241/400000 [00:51<00:03, 7301.34it/s] 94%|█████████▍| 377980/400000 [00:51<00:03, 7326.29it/s] 95%|█████████▍| 378750/400000 [00:51<00:02, 7434.45it/s] 95%|█████████▍| 379495/400000 [00:52<00:02, 7411.66it/s] 95%|█████████▌| 380238/400000 [00:52<00:02, 7404.22it/s] 95%|█████████▌| 381012/400000 [00:52<00:02, 7500.08it/s] 95%|█████████▌| 381801/400000 [00:52<00:02, 7611.06it/s] 96%|█████████▌| 382564/400000 [00:52<00:02, 7578.36it/s] 96%|█████████▌| 383326/400000 [00:52<00:02, 7590.48it/s] 96%|█████████▌| 384086/400000 [00:52<00:02, 7538.45it/s] 96%|█████████▌| 384880/400000 [00:52<00:01, 7653.57it/s] 96%|█████████▋| 385663/400000 [00:52<00:01, 7703.45it/s] 97%|█████████▋| 386456/400000 [00:52<00:01, 7769.51it/s] 97%|█████████▋| 387271/400000 [00:53<00:01, 7879.19it/s] 97%|█████████▋| 388060/400000 [00:53<00:01, 7724.14it/s] 97%|█████████▋| 388834/400000 [00:53<00:01, 7719.96it/s] 97%|█████████▋| 389614/400000 [00:53<00:01, 7742.16it/s] 98%|█████████▊| 390408/400000 [00:53<00:01, 7798.52it/s] 98%|█████████▊| 391189/400000 [00:53<00:01, 7775.47it/s] 98%|█████████▊| 391967/400000 [00:53<00:01, 7617.06it/s] 98%|█████████▊| 392745/400000 [00:53<00:00, 7665.16it/s] 98%|█████████▊| 393534/400000 [00:53<00:00, 7727.98it/s] 99%|█████████▊| 394310/400000 [00:53<00:00, 7735.80it/s] 99%|█████████▉| 395097/400000 [00:54<00:00, 7773.38it/s] 99%|█████████▉| 395875/400000 [00:54<00:00, 7680.54it/s] 99%|█████████▉| 396644/400000 [00:54<00:00, 7583.81it/s] 99%|█████████▉| 397430/400000 [00:54<00:00, 7664.59it/s]100%|█████████▉| 398215/400000 [00:54<00:00, 7716.95it/s]100%|█████████▉| 398989/400000 [00:54<00:00, 7720.68it/s]100%|█████████▉| 399762/400000 [00:54<00:00, 7683.36it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7313.10it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fdfa9b35ba8>, <torchtext.data.dataset.TabularDataset object at 0x7fdfa9b35cf8>, <torchtext.vocab.Vocab object at 0x7fdfa9b35c18>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.97 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.97 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.97 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.27 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.27 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.48 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.48 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.18 file/s]2020-06-19 06:18:50.332552: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-19 06:18:50.336795: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-19 06:18:50.336971: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5599e9b4f6b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-19 06:18:50.336988: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist_dataset_small.npy', 'cifar10', 'test', 'train', 'mnist2', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 40%|███▉      | 3956736/9912422 [00:00<00:00, 39566454.40it/s]9920512it [00:00, 37810414.87it/s]                             
0it [00:00, ?it/s]32768it [00:00, 702412.53it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1004329.61it/s]1654784it [00:00, 12537694.64it/s]                           
0it [00:00, ?it/s]8192it [00:00, 278952.85it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

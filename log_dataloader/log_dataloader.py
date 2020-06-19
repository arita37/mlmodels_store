
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7feed74c0e18> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7feed74c0e18> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fef42a77048> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fef42a77048> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fef5c7a5e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fef5c7a5e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feef02f78c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feef02f78c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feef02f78c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 157330.93it/s] 79%|███████▉  | 7823360/9912422 [00:00<00:09, 224563.91it/s]9920512it [00:00, 44931180.70it/s]                           
0it [00:00, ?it/s]32768it [00:00, 648489.66it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1006007.98it/s]1654784it [00:00, 12560929.40it/s]                           
0it [00:00, ?it/s]8192it [00:00, 252429.83it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feeed863550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feeef2b57f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7feef02f7510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7feef02f7510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7feef02f7510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:06,  2.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:06,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:05,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:05,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:04,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:04,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:04,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:03,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:03,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:44,  3.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:44,  3.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:44,  3.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:44,  3.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:43,  3.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:43,  3.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:43,  3.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:42,  3.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:42,  3.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:42,  3.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:41,  3.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:29,  4.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:29,  4.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:29,  4.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:29,  4.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:29,  4.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:28,  4.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:28,  4.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:28,  4.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:28,  4.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:27,  4.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:27,  4.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:19,  6.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:19,  6.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:19,  6.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:19,  6.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:19,  6.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:19,  6.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:18,  6.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:18,  6.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:18,  6.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:18,  6.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:18,  6.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:13,  9.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:13,  9.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:13,  9.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:12,  9.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:12,  9.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:12,  9.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:12,  9.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:12,  9.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:12,  9.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:12,  9.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:12,  9.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:00<00:08, 12.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:08, 12.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:08, 12.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:08, 12.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:08, 12.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:08, 12.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 17.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 17.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 17.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 17.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 17.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 17.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 17.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 17.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 17.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 17.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 22.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 22.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 22.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 22.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 22.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 22.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 22.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 22.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 22.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 22.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 22.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 22.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 29.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 37.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 46.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 46.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 46.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 46.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 46.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 46.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 46.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 46.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 46.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 46.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 46.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 46.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 56.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 56.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 56.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 56.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 56.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 56.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 56.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 56.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 56.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 56.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 56.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 65.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 65.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 65.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 65.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 65.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 65.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 65.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 65.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 65.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 65.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 65.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 65.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 73.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 73.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 73.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 73.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 73.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 73.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 73.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 73.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 73.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 73.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 73.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 73.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 80.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 80.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 80.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 80.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 80.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 80.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 80.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 80.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 80.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:01<00:00, 80.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:01<00:00, 80.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:01<00:00, 80.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 87.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 87.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 87.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 87.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 87.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 87.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 87.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.05s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.05s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 87.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.05s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 87.02 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.16s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.05s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 87.02 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.16s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.16s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.93 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.16s/ url]
0 examples [00:00, ? examples/s]2020-06-19 12:08:52.449198: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-19 12:08:52.462369: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-06-19 12:08:52.463099: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b8bd6cc9e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-19 12:08:52.463118: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
29 examples [00:00, 288.51 examples/s]131 examples [00:00, 367.47 examples/s]241 examples [00:00, 459.00 examples/s]352 examples [00:00, 556.40 examples/s]463 examples [00:00, 653.90 examples/s]571 examples [00:00, 740.49 examples/s]684 examples [00:00, 825.27 examples/s]795 examples [00:00, 893.50 examples/s]897 examples [00:00, 887.99 examples/s]1006 examples [00:01, 938.60 examples/s]1117 examples [00:01, 983.14 examples/s]1228 examples [00:01, 1016.61 examples/s]1336 examples [00:01, 1033.05 examples/s]1443 examples [00:01, 1021.64 examples/s]1552 examples [00:01, 1038.67 examples/s]1658 examples [00:01, 1027.80 examples/s]1762 examples [00:01, 1010.73 examples/s]1865 examples [00:01, 1014.54 examples/s]1968 examples [00:01, 1008.31 examples/s]2077 examples [00:02, 1029.14 examples/s]2189 examples [00:02, 1053.31 examples/s]2296 examples [00:02, 1056.94 examples/s]2408 examples [00:02, 1074.92 examples/s]2522 examples [00:02, 1091.56 examples/s]2632 examples [00:02, 1093.07 examples/s]2745 examples [00:02, 1102.06 examples/s]2857 examples [00:02, 1105.61 examples/s]2969 examples [00:02, 1108.07 examples/s]3080 examples [00:02, 1107.58 examples/s]3191 examples [00:03, 1064.92 examples/s]3303 examples [00:03, 1078.61 examples/s]3414 examples [00:03, 1085.68 examples/s]3523 examples [00:03, 1072.14 examples/s]3631 examples [00:03, 1072.09 examples/s]3739 examples [00:03, 1048.14 examples/s]3845 examples [00:03, 1045.72 examples/s]3950 examples [00:03, 1044.50 examples/s]4055 examples [00:03, 1029.84 examples/s]4159 examples [00:04, 989.28 examples/s] 4267 examples [00:04, 1013.21 examples/s]4369 examples [00:04, 993.27 examples/s] 4478 examples [00:04, 1018.01 examples/s]4585 examples [00:04, 1030.60 examples/s]4691 examples [00:04, 1038.81 examples/s]4796 examples [00:04, 1020.17 examples/s]4906 examples [00:04, 1042.19 examples/s]5011 examples [00:04, 1039.49 examples/s]5118 examples [00:04, 1044.22 examples/s]5228 examples [00:05, 1059.46 examples/s]5335 examples [00:05, 1051.13 examples/s]5445 examples [00:05, 1064.60 examples/s]5553 examples [00:05, 1069.08 examples/s]5661 examples [00:05, 1071.92 examples/s]5772 examples [00:05, 1081.90 examples/s]5881 examples [00:05, 1054.27 examples/s]5988 examples [00:05, 1055.96 examples/s]6094 examples [00:05, 983.24 examples/s] 6194 examples [00:05, 949.26 examples/s]6290 examples [00:06, 948.86 examples/s]6400 examples [00:06, 988.46 examples/s]6507 examples [00:06, 1010.95 examples/s]6616 examples [00:06, 1031.08 examples/s]6726 examples [00:06, 1050.10 examples/s]6835 examples [00:06, 1060.10 examples/s]6945 examples [00:06, 1069.17 examples/s]7055 examples [00:06, 1077.26 examples/s]7166 examples [00:06, 1085.35 examples/s]7277 examples [00:06, 1089.56 examples/s]7387 examples [00:07, 1075.06 examples/s]7495 examples [00:07, 1072.21 examples/s]7603 examples [00:07, 1070.10 examples/s]7713 examples [00:07, 1076.32 examples/s]7823 examples [00:07, 1081.33 examples/s]7933 examples [00:07, 1085.24 examples/s]8044 examples [00:07, 1089.83 examples/s]8155 examples [00:07, 1093.75 examples/s]8265 examples [00:07, 1095.09 examples/s]8375 examples [00:08, 1074.98 examples/s]8484 examples [00:08, 1078.97 examples/s]8592 examples [00:08, 1054.28 examples/s]8705 examples [00:08, 1073.61 examples/s]8818 examples [00:08, 1087.22 examples/s]8927 examples [00:08, 1060.17 examples/s]9036 examples [00:08, 1068.10 examples/s]9144 examples [00:08, 1069.58 examples/s]9254 examples [00:08, 1075.87 examples/s]9362 examples [00:08, 1069.61 examples/s]9470 examples [00:09, 1057.64 examples/s]9579 examples [00:09, 1064.39 examples/s]9686 examples [00:09, 1031.84 examples/s]9797 examples [00:09, 1051.32 examples/s]9903 examples [00:09, 1049.61 examples/s]10009 examples [00:09, 990.07 examples/s]10117 examples [00:09, 1014.52 examples/s]10229 examples [00:09, 1041.37 examples/s]10341 examples [00:09, 1061.59 examples/s]10451 examples [00:09, 1071.48 examples/s]10559 examples [00:10, 1073.50 examples/s]10670 examples [00:10, 1083.87 examples/s]10781 examples [00:10, 1090.92 examples/s]10892 examples [00:10, 1096.52 examples/s]11004 examples [00:10, 1101.28 examples/s]11115 examples [00:10, 1101.68 examples/s]11226 examples [00:10, 1102.81 examples/s]11337 examples [00:10, 1097.22 examples/s]11447 examples [00:10, 1078.82 examples/s]11555 examples [00:10, 1058.97 examples/s]11667 examples [00:11, 1075.50 examples/s]11775 examples [00:11, 1052.90 examples/s]11881 examples [00:11, 1048.82 examples/s]11987 examples [00:11, 1039.29 examples/s]12099 examples [00:11, 1061.37 examples/s]12207 examples [00:11, 1064.42 examples/s]12318 examples [00:11, 1076.11 examples/s]12431 examples [00:11, 1089.13 examples/s]12544 examples [00:11, 1098.83 examples/s]12657 examples [00:12, 1106.77 examples/s]12768 examples [00:12, 1066.83 examples/s]12876 examples [00:12, 989.53 examples/s] 12977 examples [00:12, 987.51 examples/s]13077 examples [00:12, 952.34 examples/s]13174 examples [00:12, 931.66 examples/s]13268 examples [00:12, 903.31 examples/s]13360 examples [00:12, 899.21 examples/s]13456 examples [00:12, 916.40 examples/s]13549 examples [00:12, 913.09 examples/s]13641 examples [00:13, 895.01 examples/s]13738 examples [00:13, 914.56 examples/s]13849 examples [00:13, 963.45 examples/s]13961 examples [00:13, 1003.35 examples/s]14069 examples [00:13, 1024.69 examples/s]14173 examples [00:13, 987.74 examples/s] 14273 examples [00:13, 986.37 examples/s]14376 examples [00:13, 998.10 examples/s]14477 examples [00:13, 992.52 examples/s]14582 examples [00:14, 1008.31 examples/s]14693 examples [00:14, 1035.45 examples/s]14797 examples [00:14, 1019.78 examples/s]14901 examples [00:14, 1024.62 examples/s]15012 examples [00:14, 1048.68 examples/s]15118 examples [00:14, 1043.74 examples/s]15229 examples [00:14, 1062.35 examples/s]15339 examples [00:14, 1072.14 examples/s]15451 examples [00:14, 1083.92 examples/s]15562 examples [00:14, 1091.31 examples/s]15672 examples [00:15, 1093.43 examples/s]15783 examples [00:15, 1096.40 examples/s]15894 examples [00:15, 1097.87 examples/s]16005 examples [00:15, 1097.97 examples/s]16117 examples [00:15, 1102.67 examples/s]16228 examples [00:15, 1101.27 examples/s]16339 examples [00:15, 1101.88 examples/s]16450 examples [00:15, 1102.83 examples/s]16561 examples [00:15, 1078.77 examples/s]16670 examples [00:15, 1072.99 examples/s]16778 examples [00:16, 1035.98 examples/s]16882 examples [00:16, 1033.26 examples/s]16993 examples [00:16, 1053.12 examples/s]17102 examples [00:16, 1063.35 examples/s]17212 examples [00:16, 1073.53 examples/s]17323 examples [00:16, 1082.49 examples/s]17432 examples [00:16, 1084.72 examples/s]17541 examples [00:16, 1080.86 examples/s]17653 examples [00:16, 1091.09 examples/s]17765 examples [00:16, 1097.19 examples/s]17876 examples [00:17, 1100.74 examples/s]17987 examples [00:17, 1101.39 examples/s]18100 examples [00:17, 1107.36 examples/s]18211 examples [00:17, 1084.54 examples/s]18321 examples [00:17, 1089.07 examples/s]18432 examples [00:17, 1093.24 examples/s]18543 examples [00:17, 1097.63 examples/s]18655 examples [00:17, 1102.14 examples/s]18766 examples [00:17, 1098.74 examples/s]18878 examples [00:17, 1103.53 examples/s]18990 examples [00:18, 1106.50 examples/s]19102 examples [00:18, 1109.56 examples/s]19215 examples [00:18, 1114.48 examples/s]19327 examples [00:18, 1113.46 examples/s]19439 examples [00:18, 1105.76 examples/s]19550 examples [00:18, 1080.80 examples/s]19659 examples [00:18, 1055.66 examples/s]19765 examples [00:18, 1056.89 examples/s]19871 examples [00:18, 1056.06 examples/s]19979 examples [00:18, 1061.14 examples/s]20086 examples [00:19, 1012.64 examples/s]20189 examples [00:19, 1016.36 examples/s]20298 examples [00:19, 1037.22 examples/s]20409 examples [00:19, 1056.74 examples/s]20516 examples [00:19, 1043.00 examples/s]20621 examples [00:19, 1013.19 examples/s]20729 examples [00:19, 1030.97 examples/s]20837 examples [00:19, 1045.09 examples/s]20945 examples [00:19, 1054.31 examples/s]21054 examples [00:20, 1063.42 examples/s]21165 examples [00:20, 1075.19 examples/s]21275 examples [00:20, 1081.56 examples/s]21384 examples [00:20, 1075.39 examples/s]21492 examples [00:20, 1074.72 examples/s]21603 examples [00:20, 1084.23 examples/s]21714 examples [00:20, 1089.63 examples/s]21824 examples [00:20, 1090.26 examples/s]21934 examples [00:20, 1059.18 examples/s]22041 examples [00:20, 1029.60 examples/s]22151 examples [00:21, 1048.87 examples/s]22262 examples [00:21, 1066.27 examples/s]22373 examples [00:21, 1078.83 examples/s]22482 examples [00:21, 1081.60 examples/s]22593 examples [00:21, 1088.66 examples/s]22703 examples [00:21, 1089.29 examples/s]22814 examples [00:21, 1094.49 examples/s]22925 examples [00:21, 1096.78 examples/s]23035 examples [00:21, 1097.12 examples/s]23145 examples [00:21, 1097.87 examples/s]23256 examples [00:22, 1101.19 examples/s]23367 examples [00:22, 1101.44 examples/s]23478 examples [00:22, 1103.50 examples/s]23589 examples [00:22, 1102.37 examples/s]23700 examples [00:22, 1101.59 examples/s]23811 examples [00:22, 1099.21 examples/s]23921 examples [00:22, 1099.09 examples/s]24031 examples [00:22, 1098.64 examples/s]24141 examples [00:22, 1098.48 examples/s]24252 examples [00:22, 1101.09 examples/s]24363 examples [00:23, 1101.78 examples/s]24474 examples [00:23, 1103.17 examples/s]24585 examples [00:23, 1079.25 examples/s]24694 examples [00:23, 1070.45 examples/s]24802 examples [00:23, 1059.56 examples/s]24915 examples [00:23, 1077.49 examples/s]25026 examples [00:23, 1086.31 examples/s]25137 examples [00:23, 1091.57 examples/s]25248 examples [00:23, 1095.24 examples/s]25358 examples [00:23, 1095.81 examples/s]25470 examples [00:24, 1100.48 examples/s]25581 examples [00:24, 1084.01 examples/s]25694 examples [00:24, 1096.13 examples/s]25806 examples [00:24, 1101.46 examples/s]25917 examples [00:24, 1076.22 examples/s]26026 examples [00:24, 1080.30 examples/s]26136 examples [00:24, 1084.80 examples/s]26247 examples [00:24, 1090.67 examples/s]26358 examples [00:24, 1095.44 examples/s]26469 examples [00:25, 1098.52 examples/s]26581 examples [00:25, 1103.15 examples/s]26692 examples [00:25, 1104.78 examples/s]26803 examples [00:25, 1098.80 examples/s]26915 examples [00:25, 1104.91 examples/s]27026 examples [00:25, 1105.30 examples/s]27138 examples [00:25, 1106.99 examples/s]27251 examples [00:25, 1112.46 examples/s]27364 examples [00:25, 1117.08 examples/s]27476 examples [00:25, 1048.71 examples/s]27582 examples [00:26, 1039.94 examples/s]27692 examples [00:26, 1055.87 examples/s]27801 examples [00:26, 1065.64 examples/s]27908 examples [00:26, 1065.80 examples/s]28015 examples [00:26, 1044.77 examples/s]28124 examples [00:26, 1057.86 examples/s]28235 examples [00:26, 1070.96 examples/s]28346 examples [00:26, 1080.96 examples/s]28458 examples [00:26, 1090.28 examples/s]28570 examples [00:26, 1098.65 examples/s]28682 examples [00:27, 1102.32 examples/s]28793 examples [00:27, 1094.38 examples/s]28905 examples [00:27, 1099.64 examples/s]29018 examples [00:27, 1106.10 examples/s]29130 examples [00:27, 1110.04 examples/s]29243 examples [00:27, 1113.36 examples/s]29355 examples [00:27, 1110.97 examples/s]29467 examples [00:27, 1110.54 examples/s]29579 examples [00:27, 1112.61 examples/s]29691 examples [00:27, 1113.22 examples/s]29803 examples [00:28, 1087.52 examples/s]29915 examples [00:28, 1095.51 examples/s]30025 examples [00:28, 1044.02 examples/s]30137 examples [00:28, 1064.16 examples/s]30249 examples [00:28, 1079.52 examples/s]30358 examples [00:28, 1051.69 examples/s]30464 examples [00:28, 1041.81 examples/s]30574 examples [00:28, 1057.21 examples/s]30686 examples [00:28, 1074.73 examples/s]30797 examples [00:28, 1083.61 examples/s]30907 examples [00:29, 1085.99 examples/s]31016 examples [00:29, 1063.86 examples/s]31123 examples [00:29, 1023.70 examples/s]31226 examples [00:29, 1025.06 examples/s]31329 examples [00:29, 997.58 examples/s] 31439 examples [00:29, 1023.96 examples/s]31551 examples [00:29, 1049.29 examples/s]31663 examples [00:29, 1069.09 examples/s]31776 examples [00:29, 1085.39 examples/s]31886 examples [00:30, 1087.55 examples/s]31998 examples [00:30, 1095.03 examples/s]32108 examples [00:30, 1088.58 examples/s]32217 examples [00:30, 1083.47 examples/s]32326 examples [00:30, 1078.71 examples/s]32438 examples [00:30, 1089.97 examples/s]32550 examples [00:30, 1096.60 examples/s]32660 examples [00:30, 1085.24 examples/s]32770 examples [00:30, 1089.06 examples/s]32879 examples [00:30, 1048.52 examples/s]32985 examples [00:31, 1041.60 examples/s]33090 examples [00:31, 1041.19 examples/s]33200 examples [00:31, 1057.88 examples/s]33311 examples [00:31, 1072.24 examples/s]33423 examples [00:31, 1083.55 examples/s]33536 examples [00:31, 1095.46 examples/s]33646 examples [00:31, 1094.99 examples/s]33758 examples [00:31, 1101.24 examples/s]33870 examples [00:31, 1103.89 examples/s]33981 examples [00:31, 1092.41 examples/s]34091 examples [00:32, 1091.42 examples/s]34201 examples [00:32, 1057.14 examples/s]34307 examples [00:32, 1037.67 examples/s]34417 examples [00:32, 1054.40 examples/s]34528 examples [00:32, 1068.89 examples/s]34636 examples [00:32, 1040.65 examples/s]34746 examples [00:32, 1056.99 examples/s]34857 examples [00:32, 1070.99 examples/s]34969 examples [00:32, 1082.84 examples/s]35079 examples [00:33, 1085.99 examples/s]35188 examples [00:33, 1087.18 examples/s]35297 examples [00:33, 1087.84 examples/s]35406 examples [00:33, 1080.61 examples/s]35517 examples [00:33, 1088.81 examples/s]35628 examples [00:33, 1093.19 examples/s]35738 examples [00:33, 1095.03 examples/s]35850 examples [00:33, 1100.06 examples/s]35961 examples [00:33, 1100.03 examples/s]36072 examples [00:33, 1102.74 examples/s]36183 examples [00:34, 1099.66 examples/s]36294 examples [00:34, 1102.33 examples/s]36405 examples [00:34, 1069.22 examples/s]36513 examples [00:34, 1066.94 examples/s]36620 examples [00:34, 1059.76 examples/s]36727 examples [00:34, 1054.83 examples/s]36833 examples [00:34, 1027.54 examples/s]36942 examples [00:34, 1043.70 examples/s]37052 examples [00:34, 1059.20 examples/s]37159 examples [00:34, 1024.87 examples/s]37262 examples [00:35, 991.97 examples/s] 37373 examples [00:35, 1024.26 examples/s]37485 examples [00:35, 1051.10 examples/s]37597 examples [00:35, 1069.97 examples/s]37710 examples [00:35, 1084.78 examples/s]37819 examples [00:35, 1077.72 examples/s]37928 examples [00:35, 1063.28 examples/s]38040 examples [00:35, 1078.85 examples/s]38149 examples [00:35, 1048.55 examples/s]38255 examples [00:35, 1045.64 examples/s]38363 examples [00:36, 1053.01 examples/s]38472 examples [00:36, 1062.98 examples/s]38582 examples [00:36, 1071.20 examples/s]38690 examples [00:36, 1065.62 examples/s]38797 examples [00:36, 1064.59 examples/s]38905 examples [00:36, 1067.47 examples/s]39015 examples [00:36, 1075.94 examples/s]39126 examples [00:36, 1085.74 examples/s]39237 examples [00:36, 1091.28 examples/s]39348 examples [00:36, 1096.13 examples/s]39459 examples [00:37, 1099.44 examples/s]39570 examples [00:37, 1100.85 examples/s]39682 examples [00:37, 1104.30 examples/s]39794 examples [00:37, 1108.83 examples/s]39906 examples [00:37, 1110.06 examples/s]40018 examples [00:37, 1049.33 examples/s]40130 examples [00:37, 1068.68 examples/s]40241 examples [00:37, 1079.81 examples/s]40353 examples [00:37, 1090.50 examples/s]40465 examples [00:38, 1096.68 examples/s]40575 examples [00:38, 1076.90 examples/s]40687 examples [00:38, 1088.68 examples/s]40798 examples [00:38, 1092.81 examples/s]40910 examples [00:38, 1098.26 examples/s]41020 examples [00:38, 1080.49 examples/s]41129 examples [00:38, 1048.52 examples/s]41237 examples [00:38, 1057.75 examples/s]41347 examples [00:38, 1069.18 examples/s]41456 examples [00:38, 1074.26 examples/s]41564 examples [00:39, 1074.13 examples/s]41672 examples [00:39, 1061.90 examples/s]41779 examples [00:39, 1025.96 examples/s]41889 examples [00:39, 1046.86 examples/s]42000 examples [00:39, 1062.29 examples/s]42107 examples [00:39, 1010.51 examples/s]42210 examples [00:39, 1015.79 examples/s]42315 examples [00:39, 1023.51 examples/s]42424 examples [00:39, 1040.64 examples/s]42535 examples [00:39, 1058.67 examples/s]42644 examples [00:40, 1062.98 examples/s]42751 examples [00:40, 1063.08 examples/s]42861 examples [00:40, 1072.04 examples/s]42969 examples [00:40, 1068.69 examples/s]43076 examples [00:40, 1064.13 examples/s]43187 examples [00:40, 1075.77 examples/s]43299 examples [00:40, 1085.85 examples/s]43408 examples [00:40, 1085.21 examples/s]43517 examples [00:40, 1046.76 examples/s]43623 examples [00:40, 1049.66 examples/s]43731 examples [00:41, 1057.25 examples/s]43838 examples [00:41, 1059.09 examples/s]43945 examples [00:41, 1058.99 examples/s]44051 examples [00:41, 1056.17 examples/s]44163 examples [00:41, 1073.55 examples/s]44273 examples [00:41, 1080.60 examples/s]44382 examples [00:41, 1044.32 examples/s]44495 examples [00:41, 1067.95 examples/s]44607 examples [00:41, 1082.12 examples/s]44719 examples [00:42, 1090.94 examples/s]44829 examples [00:42, 1068.04 examples/s]44937 examples [00:42, 1035.37 examples/s]45045 examples [00:42, 1047.08 examples/s]45156 examples [00:42, 1063.61 examples/s]45267 examples [00:42, 1077.08 examples/s]45376 examples [00:42, 1080.83 examples/s]45486 examples [00:42, 1084.91 examples/s]45597 examples [00:42, 1091.35 examples/s]45707 examples [00:42, 1092.23 examples/s]45819 examples [00:43, 1097.43 examples/s]45929 examples [00:43, 1070.41 examples/s]46039 examples [00:43, 1077.92 examples/s]46148 examples [00:43, 1081.02 examples/s]46259 examples [00:43, 1087.55 examples/s]46368 examples [00:43, 1082.62 examples/s]46477 examples [00:43, 1072.22 examples/s]46586 examples [00:43, 1075.47 examples/s]46694 examples [00:43, 1053.48 examples/s]46800 examples [00:43, 1055.03 examples/s]46907 examples [00:44, 1059.36 examples/s]47015 examples [00:44, 1063.14 examples/s]47122 examples [00:44, 1061.32 examples/s]47230 examples [00:44, 1066.08 examples/s]47337 examples [00:44, 1028.37 examples/s]47445 examples [00:44, 1041.85 examples/s]47557 examples [00:44, 1062.89 examples/s]47664 examples [00:44, 1008.51 examples/s]47774 examples [00:44, 1033.87 examples/s]47879 examples [00:44, 1031.27 examples/s]47986 examples [00:45, 1042.26 examples/s]48096 examples [00:45, 1057.56 examples/s]48206 examples [00:45, 1068.78 examples/s]48315 examples [00:45, 1074.09 examples/s]48423 examples [00:45, 1073.55 examples/s]48531 examples [00:45, 1071.50 examples/s]48641 examples [00:45, 1077.44 examples/s]48751 examples [00:45, 1081.90 examples/s]48860 examples [00:45, 1029.18 examples/s]48964 examples [00:46, 1030.72 examples/s]49074 examples [00:46, 1049.61 examples/s]49180 examples [00:46, 1049.17 examples/s]49291 examples [00:46, 1064.27 examples/s]49402 examples [00:46, 1077.30 examples/s]49513 examples [00:46, 1084.81 examples/s]49623 examples [00:46, 1088.05 examples/s]49732 examples [00:46, 1082.78 examples/s]49841 examples [00:46, 1063.52 examples/s]49952 examples [00:46, 1076.17 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7424/50000 [00:00<00:00, 74239.19 examples/s] 41%|████      | 20264/50000 [00:00<00:00, 84994.73 examples/s] 67%|██████▋   | 33390/50000 [00:00<00:00, 95043.82 examples/s] 94%|█████████▎| 46805/50000 [00:00<00:00, 104152.13 examples/s]                                                                0 examples [00:00, ? examples/s]94 examples [00:00, 936.22 examples/s]206 examples [00:00, 983.46 examples/s]318 examples [00:00, 1020.21 examples/s]420 examples [00:00, 1019.06 examples/s]529 examples [00:00, 1038.50 examples/s]640 examples [00:00, 1057.43 examples/s]752 examples [00:00, 1072.66 examples/s]861 examples [00:00, 1077.74 examples/s]972 examples [00:00, 1086.23 examples/s]1080 examples [00:01, 1082.55 examples/s]1192 examples [00:01, 1091.46 examples/s]1301 examples [00:01, 1090.48 examples/s]1412 examples [00:01, 1093.69 examples/s]1521 examples [00:01, 1084.14 examples/s]1634 examples [00:01, 1095.40 examples/s]1748 examples [00:01, 1106.54 examples/s]1859 examples [00:01, 1104.30 examples/s]1970 examples [00:01, 1091.23 examples/s]2082 examples [00:01, 1099.25 examples/s]2193 examples [00:02, 1100.75 examples/s]2305 examples [00:02, 1104.44 examples/s]2416 examples [00:02, 1100.89 examples/s]2527 examples [00:02, 1101.05 examples/s]2638 examples [00:02, 1102.76 examples/s]2749 examples [00:02, 1078.24 examples/s]2862 examples [00:02, 1090.90 examples/s]2975 examples [00:02, 1101.98 examples/s]3088 examples [00:02, 1109.78 examples/s]3202 examples [00:02, 1116.26 examples/s]3316 examples [00:03, 1120.89 examples/s]3429 examples [00:03, 1122.93 examples/s]3542 examples [00:03, 1114.05 examples/s]3654 examples [00:03, 1056.26 examples/s]3761 examples [00:03, 1009.35 examples/s]3871 examples [00:03, 1034.30 examples/s]3977 examples [00:03, 1039.57 examples/s]4089 examples [00:03, 1060.17 examples/s]4197 examples [00:03, 1063.40 examples/s]4304 examples [00:03, 1061.22 examples/s]4411 examples [00:04, 1054.62 examples/s]4517 examples [00:04, 1033.75 examples/s]4629 examples [00:04, 1056.31 examples/s]4735 examples [00:04, 1055.25 examples/s]4848 examples [00:04, 1074.80 examples/s]4959 examples [00:04, 1082.82 examples/s]5071 examples [00:04, 1092.81 examples/s]5181 examples [00:04, 1078.82 examples/s]5290 examples [00:04, 1059.79 examples/s]5399 examples [00:04, 1067.43 examples/s]5512 examples [00:05, 1083.38 examples/s]5623 examples [00:05, 1091.16 examples/s]5736 examples [00:05, 1101.02 examples/s]5847 examples [00:05, 1101.76 examples/s]5960 examples [00:05, 1107.31 examples/s]6072 examples [00:05, 1109.21 examples/s]6184 examples [00:05, 1111.69 examples/s]6296 examples [00:05, 1109.30 examples/s]6407 examples [00:05, 1103.18 examples/s]6518 examples [00:06, 1105.16 examples/s]6630 examples [00:06, 1108.43 examples/s]6741 examples [00:06, 1106.45 examples/s]6853 examples [00:06, 1109.43 examples/s]6964 examples [00:06, 1108.87 examples/s]7075 examples [00:06, 1036.20 examples/s]7182 examples [00:06, 1041.21 examples/s]7287 examples [00:06, 995.73 examples/s] 7388 examples [00:06, 983.88 examples/s]7494 examples [00:06, 1004.72 examples/s]7605 examples [00:07, 1032.44 examples/s]7709 examples [00:07, 1030.42 examples/s]7820 examples [00:07, 1053.04 examples/s]7931 examples [00:07, 1067.19 examples/s]8042 examples [00:07, 1078.07 examples/s]8154 examples [00:07, 1087.98 examples/s]8265 examples [00:07, 1092.92 examples/s]8377 examples [00:07, 1100.38 examples/s]8489 examples [00:07, 1105.46 examples/s]8600 examples [00:07, 1102.81 examples/s]8713 examples [00:08, 1108.39 examples/s]8824 examples [00:08, 1100.95 examples/s]8935 examples [00:08, 1103.33 examples/s]9046 examples [00:08, 1104.28 examples/s]9157 examples [00:08, 1068.92 examples/s]9265 examples [00:08, 1063.08 examples/s]9372 examples [00:08, 1049.17 examples/s]9484 examples [00:08, 1067.66 examples/s]9592 examples [00:08, 1068.95 examples/s]9700 examples [00:08, 1045.34 examples/s]9805 examples [00:09, 1038.67 examples/s]9912 examples [00:09, 1047.25 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteITH0CL/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteITH0CL/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feef02f78c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feef02f78c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feef02f78c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fef3bd7e208>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fee7577d048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feef02f78c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feef02f78c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feef02f78c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feeef435be0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feeef2b5080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fee67e8a7b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fee67e8a7b8> 

  function with postional parmater data_info <function split_train_valid at 0x7fee67e8a7b8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fee67e8a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fee67e8a8c8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fee67e8a8c8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.4.5.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=236bab4778dc8701456eab8e8c573a27c14d65c7a01570c0ccdf130c70a5007a
  Stored in directory: /tmp/pip-ephem-wheel-cache-1hmh97sh/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:23:15, 11.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:13:17, 15.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:42:46, 22.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:30:29, 31.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:14:33, 45.5kB/s].vector_cache/glove.6B.zip:   1%|          | 8.56M/862M [00:01<3:39:01, 65.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:01<2:32:44, 92.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.5M/862M [00:01<1:46:21, 132kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:01<1:14:17, 189kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.1M/862M [00:01<51:46, 269kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.6M/862M [00:01<36:13, 383kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.9M/862M [00:02<25:19, 545kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<17:44, 774kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.5M/862M [00:02<12:27, 1.10MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.8M/862M [00:02<08:47, 1.55MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.1M/862M [00:02<06:12, 2.18MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:03<06:05, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<06:09, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<06:38, 2.02MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<05:07, 2.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:05<03:53, 3.44MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:07<06:47, 1.97MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<06:18, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:07<04:44, 2.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<06:15, 2.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<05:47, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.9M/862M [00:09<04:20, 3.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<06:09, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<05:31, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:11<04:36, 2.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.4M/862M [00:11<03:29, 3.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<06:15, 2.10MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<05:48, 2.26MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:13<04:25, 2.97MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<06:08, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:15<05:43, 2.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<04:17, 3.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:04, 2.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:17<05:39, 2.30MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:17<04:18, 3.01MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<06:03, 2.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<05:36, 2.30MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:19<04:16, 3.02MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<06:00, 2.15MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<05:35, 2.30MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:21<04:15, 3.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<05:58, 2.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<05:34, 2.30MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:23<04:14, 3.01MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<05:56, 2.14MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<05:32, 2.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:25<04:12, 3.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:54, 2.14MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:30, 2.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:11, 3.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:52, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:29, 2.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:10, 3.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:51, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:27, 2.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:06, 3.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:48, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:10, 2.40MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<04:09, 2.99MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<03:02, 4.07MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<51:31, 241kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<37:21, 332kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<26:26, 468kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<21:20, 578kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<16:02, 768kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<11:39, 1.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<08:16, 1.48MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<54:37, 224kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<39:32, 310kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<27:53, 438kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<19:39, 620kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<2:12:28, 92.0kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<1:34:00, 130kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<1:05:59, 184kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<48:53, 248kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<35:31, 341kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<25:04, 482kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<20:22, 592kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<15:33, 775kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<11:08, 1.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<07:56, 1.51MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<1:33:42, 128kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<1:06:53, 179kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<46:59, 254kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<35:36, 335kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<26:12, 455kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<18:34, 640kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<13:07, 902kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<1:18:43, 150kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<56:20, 210kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<39:40, 298kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<30:26, 387kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<22:34, 521kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<16:05, 730kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<13:58, 838kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<11:03, 1.06MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<08:00, 1.46MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:19, 1.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:05, 1.64MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<05:16, 2.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:23, 1.81MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:43, 2.02MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<04:16, 2.70MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:42, 2.01MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:14, 2.19MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:55, 2.93MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:26, 2.10MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<04:49, 2.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<03:59, 2.86MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<02:53, 3.92MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<42:19, 269kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<30:50, 368kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<21:50, 519kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<17:53, 631kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<13:31, 834kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<09:55, 1.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<07:02, 1.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<46:53, 239kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<34:02, 330kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<24:01, 466kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<19:24, 575kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<14:48, 754kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<10:35, 1.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<10:00, 1.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<08:13, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<06:02, 1.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:47, 1.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:56, 1.85MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<04:24, 2.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:40, 1.93MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:08, 2.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<03:50, 2.84MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:15, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:51, 2.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<03:38, 2.98MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:06, 2.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:44, 2.28MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:33, 3.03MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:59, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:39, 2.31MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:29, 3.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:58, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:38, 2.30MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:32, 3.00MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:57, 2.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:37, 2.29MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:28, 3.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:55, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:56, 2.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:19, 2.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:26, 3.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<02:35, 4.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:28, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:12, 1.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:45, 2.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<05:49, 1.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<09:03, 1.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<07:34, 1.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<05:31, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<04:02, 2.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<14:49, 698kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<10:55, 944kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:34<07:43, 1.33MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<16:10, 635kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<14:54, 689kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<12:22, 830kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<09:43, 1.05MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<07:19, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:20, 1.91MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:31, 1.56MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:19, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<03:53, 2.60MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<06:51, 1.48MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<07:13, 1.40MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:33, 1.82MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:14, 2.38MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:06, 3.24MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<16:12, 621kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<12:42, 792kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<09:10, 1.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<06:33, 1.53MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<10:48, 924kB/s] .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<08:48, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<06:25, 1.55MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:38, 2.14MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<10:01, 990kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<09:25, 1.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<07:12, 1.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:14, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:06, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:21, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:01, 2.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:03, 1.93MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:55, 1.99MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:56, 2.48MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<03:01, 3.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<02:15, 4.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<36:08, 269kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<26:38, 365kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<18:58, 511kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<13:32, 715kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<09:39, 1.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<15:14, 633kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<12:17, 784kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<08:59, 1.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:23, 1.50MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<18:02, 531kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<14:18, 669kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<10:52, 880kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<08:01, 1.19MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<05:53, 1.62MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:24, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<07:52, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<06:43, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:00, 1.89MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:37, 2.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<36:30, 259kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<26:53, 351kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<19:29, 484kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<13:51, 679kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<09:52, 951kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<12:11, 768kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<10:01, 934kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<07:16, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<05:19, 1.75MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<07:30, 1.24MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<08:21, 1.11MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<06:37, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<05:02, 1.84MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<05:17, 1.75MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<07:12, 1.28MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<05:44, 1.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<04:30, 2.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<03:18, 2.78MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<02:31, 3.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<2:39:18, 57.5kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<1:54:55, 79.7kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<1:21:05, 113kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<56:50, 161kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<39:48, 229kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<40:26, 225kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<31:38, 287kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<23:40, 384kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<16:55, 536kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<12:01, 753kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<11:01, 819kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<10:42, 843kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<08:19, 1.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<06:11, 1.46MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:28, 2.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<06:59, 1.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<07:40, 1.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<05:59, 1.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<04:30, 1.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:16, 2.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<09:46, 909kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<09:34, 928kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<07:14, 1.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<05:26, 1.63MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<03:55, 2.25MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<08:27, 1.04MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<09:00, 978kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:18<07:02, 1.25MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:04, 1.73MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<06:12, 1.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<07:26, 1.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<06:09, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<04:40, 1.87MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<03:25, 2.54MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<05:21, 1.62MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<06:18, 1.38MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<05:02, 1.72MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<03:50, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<02:46, 3.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<23:46, 362kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<19:05, 451kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<14:33, 591kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<10:54, 788kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<08:05, 1.06MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<06:02, 1.42MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:33, 1.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:35, 2.38MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<02:52, 2.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<5:51:07, 24.3kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<4:06:20, 34.6kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<2:52:20, 49.4kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<2:00:12, 70.5kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<1:36:27, 87.8kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<1:10:00, 121kB/s] .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<49:35, 171kB/s]  .vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<34:48, 242kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<24:23, 344kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<7:55:04, 17.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<5:34:39, 25.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<3:54:26, 35.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<2:44:08, 51.0kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<1:54:58, 72.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<1:20:40, 103kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<56:48, 147kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<56:23, 148kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<41:51, 199kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<31:13, 267kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<22:59, 362kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<16:38, 500kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<12:12, 681kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<09:01, 920kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<06:38, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:54, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:33<08:08, 1.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<07:11, 1.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<05:17, 1.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:54, 2.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<05:12, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<06:13, 1.32MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<05:43, 1.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<04:49, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<03:37, 2.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<02:38, 3.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<08:41, 934kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:38<08:15, 984kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<06:12, 1.30MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<04:37, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<04:47, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:40<06:03, 1.33MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:40<04:48, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:40<03:39, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<02:38, 3.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<27:52, 286kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<21:40, 368kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:42<15:49, 504kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<11:19, 703kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<08:06, 978kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<08:00, 988kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<08:14, 961kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:44<06:18, 1.25MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<04:45, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<04:39, 1.68MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<05:19, 1.47MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:46<04:09, 1.88MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<03:09, 2.47MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<03:45, 2.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<05:15, 1.48MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:48<05:16, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:48<04:36, 1.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<03:38, 2.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<02:54, 2.67MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<02:18, 3.34MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<01:51, 4.13MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<05:40, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<05:14, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:50<04:22, 1.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:50<03:16, 2.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<02:23, 3.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<10:58, 696kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<10:45, 709kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<08:12, 930kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:52<06:00, 1.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:19, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<06:44, 1.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<08:16, 915kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<08:10, 926kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:54<07:07, 1.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:54<05:42, 1.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:54<04:18, 1.75MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:12, 2.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<02:24, 3.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<15:01, 499kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<11:59, 625kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:56<08:38, 866kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<06:14, 1.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:30, 1.65MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<11:06, 668kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<10:37, 699kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:58<08:05, 918kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:58<06:01, 1.23MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<04:19, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<05:38, 1.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<06:24, 1.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<05:47, 1.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:00<04:53, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<03:48, 1.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<02:53, 2.53MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<02:09, 3.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:02<08:59, 811kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:02<11:48, 618kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:02<09:27, 770kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:02<07:20, 993kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:02<05:18, 1.37MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:50, 1.88MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:04<07:54, 913kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:04<08:08, 886kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:04<06:28, 1.11MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:04<04:55, 1.46MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<03:35, 2.00MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<04:16, 1.67MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<05:34, 1.28MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<04:52, 1.47MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:06<04:02, 1.77MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:06<03:19, 2.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<02:31, 2.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<01:52, 3.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:08<20:23, 347kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:08<15:31, 456kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:08<11:23, 621kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:08<08:23, 842kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<06:09, 1.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<04:29, 1.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<05:42, 1.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:10<05:25, 1.29MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:10<04:05, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<03:01, 2.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<03:52, 1.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:12<05:16, 1.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:12<04:42, 1.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:12<03:37, 1.91MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<02:40, 2.59MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:14<04:04, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:14<08:14, 834kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:14<06:44, 1.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:14<05:26, 1.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<04:03, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:15<02:58, 2.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:16<04:21, 1.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:16<04:14, 1.61MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<03:15, 2.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<02:22, 2.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:18<05:14, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:18<06:04, 1.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:18<04:49, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<03:31, 1.90MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<04:14, 1.57MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<05:45, 1.16MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<05:30, 1.21MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<04:40, 1.43MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:20<03:42, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<02:44, 2.42MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:03, 3.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<05:54, 1.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<05:17, 1.25MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:22<04:03, 1.62MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<02:59, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<02:32, 2.58MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:24<03:36, 1.81MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:24<05:18, 1.23MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:24<05:44, 1.14MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:24<04:49, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:24<03:40, 1.77MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<02:42, 2.40MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<02:03, 3.15MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:26<07:03, 915kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:26<06:37, 976kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:26<05:13, 1.24MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<03:48, 1.69MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<02:50, 2.26MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<02:08, 2.99MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:28<21:09, 302kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:28<19:36, 326kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:28<14:50, 430kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<10:44, 594kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<07:52, 808kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<05:52, 1.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:29<04:25, 1.43MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:29<03:33, 1.78MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:30<05:50, 1.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:30<06:11, 1.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:30<04:50, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:30<03:40, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<03:03, 2.06MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:30<02:24, 2.60MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<02:18, 2.73MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:31<02:14, 2.80MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:31<02:10, 2.88MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:31<02:10, 2.88MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:32<06:53, 908kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:32<05:45, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:32<04:24, 1.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:32<03:26, 1.81MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<02:46, 2.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<02:13, 2.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<01:54, 3.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<01:41, 3.67MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<05:00, 1.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<06:47, 911kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<05:45, 1.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:34<04:29, 1.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:34<03:29, 1.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<02:43, 2.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<02:16, 2.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<01:53, 3.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:35<01:44, 3.51MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<01:41, 3.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<10:02, 610kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<09:18, 657kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:36<06:58, 875kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:36<05:28, 1.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<04:08, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<03:27, 1.76MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<02:43, 2.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<02:27, 2.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<02:02, 2.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<01:56, 3.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<06:12, 973kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<06:25, 941kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<05:41, 1.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<04:37, 1.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<03:38, 1.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<03:00, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<02:26, 2.46MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<02:06, 2.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<01:51, 3.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<01:37, 3.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:39<01:35, 3.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<06:04, 985kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<06:20, 942kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:40<04:52, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:40<03:56, 1.51MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<03:03, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<02:36, 2.29MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<02:06, 2.82MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<01:56, 3.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<01:43, 3.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<04:54, 1.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<05:25, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<04:16, 1.38MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<03:19, 1.77MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<02:45, 2.13MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<02:14, 2.61MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<01:55, 3.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<01:43, 3.40MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<01:32, 3.77MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<05:30, 1.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<05:49, 1.00MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<04:36, 1.27MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:44<03:36, 1.61MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:44<02:51, 2.03MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<02:22, 2.45MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<02:00, 2.90MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<01:45, 3.29MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:45<01:33, 3.70MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<06:31, 885kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<06:23, 903kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<06:08, 940kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<05:21, 1.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:46<04:03, 1.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:46<03:23, 1.69MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:46<02:52, 2.00MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<02:35, 2.21MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<02:14, 2.57MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<02:10, 2.64MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<02:06, 2.72MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<01:55, 2.98MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<01:50, 3.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<01:49, 3.13MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:48<40:50, 140kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:48<29:15, 195kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:48<20:55, 272kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:48<15:01, 378kB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:48<10:59, 517kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<08:07, 698kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<06:14, 908kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<04:45, 1.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:49<03:50, 1.47MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:49<03:05, 1.83MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<07:01, 802kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<06:32, 861kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<06:09, 916kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<05:35, 1.01MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<05:15, 1.07MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<04:48, 1.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<04:29, 1.25MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<04:25, 1.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<04:24, 1.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<04:04, 1.38MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:51<03:54, 1.44MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:51<03:19, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:51<02:51, 1.97MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:51<02:27, 2.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<02:06, 2.65MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<01:57, 2.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<01:52, 2.98MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:51<01:48, 3.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:51<02:33, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:52<02:06, 2.64MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:52<02:06, 2.63MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:52<01:48, 3.06MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:52<01:55, 2.88MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:52<01:47, 3.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:52<01:43, 3.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<01:37, 3.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<01:34, 3.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<01:32, 3.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:53<01:30, 3.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:53<07:35, 724kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<07:33, 727kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:54<05:48, 945kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:54<04:40, 1.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:54<03:36, 1.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<03:06, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<02:29, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<02:18, 2.36MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<01:57, 2.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<01:55, 2.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<01:41, 3.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<01:42, 3.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<25:28, 213kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:56<18:55, 287kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:56<13:37, 398kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:56<10:05, 536kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<07:29, 721kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<05:41, 949kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<04:22, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:56<03:28, 1.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:56<02:50, 1.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<02:22, 2.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:57<02:08, 2.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:57<01:59, 2.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:57<16:52, 318kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<13:27, 398kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<10:38, 503kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<08:59, 596kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<07:06, 753kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<05:47, 923kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<04:34, 1.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<03:36, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<03:00, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<02:28, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<02:05, 2.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<01:52, 2.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<01:40, 3.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<01:29, 3.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:59<08:20, 635kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:00<06:46, 782kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:00<05:00, 1.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<03:57, 1.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<03:08, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<02:32, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<02:05, 2.52MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<01:49, 2.88MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<01:34, 3.33MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<01:24, 3.73MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:01<1:04:58, 80.4kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:01<47:06, 111kB/s]   .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<33:20, 156kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:02<23:37, 220kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:02<16:46, 310kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:02<12:03, 430kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<08:40, 597kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<06:23, 808kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:03<07:01, 735kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:03<06:19, 815kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:04<04:42, 1.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:04<03:42, 1.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<02:46, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<02:17, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:04<01:53, 2.69MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:04<01:36, 3.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:05<04:29, 1.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:05<04:17, 1.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<03:21, 1.51MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<02:32, 2.00MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<02:01, 2.49MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<01:34, 3.19MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<01:22, 3.67MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:07<08:54, 564kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:07<08:15, 608kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:08<06:18, 794kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<04:36, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:08<03:20, 1.49MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:08<02:34, 1.93MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:09<04:34, 1.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:09<04:56, 1.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:10<03:53, 1.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<02:54, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<02:11, 2.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<01:40, 2.92MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:11<05:22, 907kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:11<05:17, 921kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:11<04:05, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:12<03:03, 1.59MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<02:14, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<01:42, 2.81MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:13<05:18, 906kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:13<05:03, 952kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:13<03:48, 1.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<02:52, 1.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<02:05, 2.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<01:36, 2.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:15<31:37, 150kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:15<23:51, 199kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:15<17:37, 269kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:15<12:51, 368kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<09:24, 502kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<06:56, 679kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<05:07, 920kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<03:43, 1.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<02:47, 1.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:16<02:10, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:17<30:10, 155kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:17<22:34, 207kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<16:06, 289kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<11:19, 410kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<07:58, 578kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:19<09:56, 463kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:19<08:41, 529kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:19<06:26, 714kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<04:43, 970kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<03:26, 1.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<02:28, 1.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:21<10:25, 435kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:21<08:04, 562kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:21<06:03, 747kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:21<04:29, 1.01MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<03:18, 1.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<02:24, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:23<03:35, 1.24MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:23<04:12, 1.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:23<03:16, 1.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:24<02:38, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<02:09, 2.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<01:35, 2.78MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:25<02:40, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:25<02:33, 1.72MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<01:55, 2.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<01:26, 3.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<02:26, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<03:03, 1.42MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<02:30, 1.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<01:56, 2.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<01:26, 2.96MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<01:06, 3.82MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:29<06:06, 697kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:29<05:47, 735kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:29<04:21, 973kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<03:14, 1.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<02:18, 1.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:31<04:28, 936kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:31<04:03, 1.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:31<03:00, 1.39MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<02:13, 1.86MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<02:29, 1.65MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<02:51, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<02:23, 1.72MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:33<01:51, 2.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<01:23, 2.94MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<02:04, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<02:13, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:35<01:44, 2.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<01:16, 3.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<02:25, 1.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<02:38, 1.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:37<02:02, 1.94MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:37<01:32, 2.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<01:54, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:39<02:00, 1.94MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<01:33, 2.50MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<01:09, 3.35MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<02:13, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<02:26, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<01:55, 1.98MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<01:22, 2.75MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:43<07:57, 475kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:43<06:02, 625kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:43<04:31, 834kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:43<03:17, 1.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:21, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:45<03:13, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:45<03:02, 1.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<02:16, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<01:40, 2.18MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<02:05, 1.74MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<02:12, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:47<01:41, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<01:16, 2.83MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<01:45, 2.03MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<02:11, 1.62MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<01:57, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:49<01:31, 2.33MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:49<01:07, 3.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<01:54, 1.84MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<02:04, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:51<01:37, 2.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<01:14, 2.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<00:56, 3.67MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<02:13, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<02:20, 1.47MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<01:58, 1.74MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<01:31, 2.23MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<01:06, 3.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<02:15, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<02:17, 1.47MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<01:43, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<01:18, 2.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<01:42, 1.92MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:57<01:52, 1.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<01:25, 2.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<01:05, 3.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<01:32, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<01:43, 1.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<01:22, 2.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<00:59, 3.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<02:54, 1.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<02:56, 1.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<02:41, 1.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<02:06, 1.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<01:35, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:10, 2.64MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:03<01:52, 1.65MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:03<01:37, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:03<01:11, 2.55MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<01:34, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<01:48, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<01:36, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<01:15, 2.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:05<00:56, 3.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<01:28, 2.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<01:38, 1.80MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:07<01:17, 2.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<00:57, 3.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:08<01:36, 1.79MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:39, 1.74MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:09<01:28, 1.94MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<01:06, 2.57MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<00:48, 3.51MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:10<53:38, 52.5kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:11<37:56, 74.1kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:11<26:37, 105kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<18:34, 150kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:12<13:28, 204kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<09:55, 276kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<07:13, 378kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<05:05, 534kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<04:04, 656kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<03:20, 799kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<02:40, 995kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:15<01:55, 1.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:16<01:54, 1.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<01:52, 1.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<01:25, 1.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:17<01:04, 2.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<00:47, 3.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<03:04, 826kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<02:44, 925kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<02:09, 1.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:19<01:39, 1.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<01:13, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<01:27, 1.70MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<01:32, 1.61MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<01:11, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<00:52, 2.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:22<01:20, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:22<01:24, 1.71MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<01:05, 2.18MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:23<00:47, 2.98MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:24<01:39, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:24<01:35, 1.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<01:13, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<00:52, 2.60MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:26<01:56, 1.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:26<01:57, 1.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<01:36, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:27<01:12, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<00:52, 2.55MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:28<01:33, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:28<01:31, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:29<01:09, 1.87MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<00:50, 2.57MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<01:35, 1.33MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<01:43, 1.24MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<01:30, 1.41MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<01:09, 1.82MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:31<00:51, 2.45MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<01:05, 1.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<01:08, 1.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:32<00:53, 2.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<00:38, 3.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<02:23, 832kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<02:03, 965kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:34<01:31, 1.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<01:21, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<01:30, 1.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<01:24, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<01:07, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<00:51, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:37<00:38, 2.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:38<01:03, 1.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<00:55, 2.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:38<00:40, 2.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<00:53, 2.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<01:17, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:40<01:02, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<00:48, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<00:49, 2.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<01:02, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<01:00, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<00:47, 2.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:43<00:35, 2.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:43<00:27, 3.66MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:44<01:08, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:44<00:58, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:44<00:42, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<00:51, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<00:54, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<00:42, 2.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:30, 3.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<02:11, 686kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<01:49, 828kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<01:19, 1.12MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<00:56, 1.56MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:50<01:17, 1.12MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:50<01:10, 1.21MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<00:53, 1.60MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<00:49, 1.65MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<00:58, 1.40MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<00:55, 1.48MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:43, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<00:31, 2.53MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:39, 1.96MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:42, 1.81MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:33, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:34, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:38, 1.90MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:56<00:34, 2.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:27, 2.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:19, 3.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<02:00, 578kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<02:38, 440kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:00<02:08, 542kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:00<01:33, 738kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<01:04, 1.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:01<01:20, 821kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<01:10, 931kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:02<01:02, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:02<00:53, 1.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:02<00:45, 1.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:02<00:36, 1.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<00:26, 2.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:19, 3.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:04<01:05, 938kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:04<00:58, 1.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:04<00:43, 1.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:31, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:06<00:34, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:06<00:31, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:06<00:23, 2.35MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:08<00:26, 2.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:08<00:28, 1.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:08<00:21, 2.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<00:15, 3.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:11, 4.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:10<04:18, 190kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:10<03:08, 260kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:10<02:15, 358kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:10<01:36, 498kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:10<01:07, 699kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:46, 976kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:12<01:06, 680kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:12<00:54, 816kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:12<00:39, 1.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:27, 1.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:13<00:32, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:14<00:30, 1.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:14<00:22, 1.77MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<00:16, 2.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:15<00:21, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:16<00:21, 1.72MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:16<00:16, 2.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:11, 3.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:17<00:23, 1.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:18<00:22, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:18<00:16, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:11, 2.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:20, 1.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:20<00:19, 1.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:20<00:14, 1.96MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:10, 2.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:13, 1.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:13, 1.77MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:22<00:10, 2.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:07, 3.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:10, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:24<00:11, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:24<00:08, 2.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:05, 3.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:19, 837kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:26<00:19, 823kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:26<00:14, 1.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:26<00:09, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:27<00:08, 1.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:28<00:08, 1.42MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:28<00:05, 1.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:03, 2.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:29<00:52, 149kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:30<00:37, 205kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:30<00:23, 289kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:31<00:09, 385kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<00:07, 499kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:32<00:03, 690kB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 96/400000 [00:00<06:57, 959.00it/s]  0%|          | 955/400000 [00:00<05:05, 1307.39it/s]  0%|          | 1778/400000 [00:00<03:47, 1748.64it/s]  1%|          | 2649/400000 [00:00<02:52, 2300.13it/s]  1%|          | 3505/400000 [00:00<02:14, 2946.47it/s]  1%|          | 4370/400000 [00:00<01:47, 3672.98it/s]  1%|▏         | 5235/400000 [00:00<01:28, 4439.00it/s]  2%|▏         | 6049/400000 [00:00<01:16, 5140.10it/s]  2%|▏         | 6878/400000 [00:00<01:07, 5801.21it/s]  2%|▏         | 7752/400000 [00:01<01:00, 6451.28it/s]  2%|▏         | 8610/400000 [00:01<00:56, 6969.45it/s]  2%|▏         | 9475/400000 [00:01<00:52, 7399.64it/s]  3%|▎         | 10352/400000 [00:01<00:50, 7762.76it/s]  3%|▎         | 11226/400000 [00:01<00:48, 8031.11it/s]  3%|▎         | 12087/400000 [00:01<00:47, 8194.28it/s]  3%|▎         | 12964/400000 [00:01<00:46, 8357.74it/s]  3%|▎         | 13836/400000 [00:01<00:45, 8461.86it/s]  4%|▎         | 14703/400000 [00:01<00:45, 8522.33it/s]  4%|▍         | 15571/400000 [00:01<00:44, 8568.61it/s]  4%|▍         | 16456/400000 [00:02<00:44, 8649.08it/s]  4%|▍         | 17329/400000 [00:02<00:45, 8494.14it/s]  5%|▍         | 18204/400000 [00:02<00:44, 8568.26it/s]  5%|▍         | 19081/400000 [00:02<00:44, 8624.62it/s]  5%|▍         | 19947/400000 [00:02<00:44, 8541.27it/s]  5%|▌         | 20804/400000 [00:02<00:45, 8249.37it/s]  5%|▌         | 21661/400000 [00:02<00:45, 8342.62it/s]  6%|▌         | 22524/400000 [00:02<00:44, 8424.52it/s]  6%|▌         | 23399/400000 [00:02<00:44, 8518.00it/s]  6%|▌         | 24265/400000 [00:02<00:43, 8559.89it/s]  6%|▋         | 25146/400000 [00:03<00:43, 8630.76it/s]  7%|▋         | 26024/400000 [00:03<00:43, 8672.34it/s]  7%|▋         | 26900/400000 [00:03<00:42, 8696.55it/s]  7%|▋         | 27774/400000 [00:03<00:42, 8708.51it/s]  7%|▋         | 28647/400000 [00:03<00:42, 8713.53it/s]  7%|▋         | 29519/400000 [00:03<00:42, 8671.13it/s]  8%|▊         | 30387/400000 [00:03<00:42, 8597.71it/s]  8%|▊         | 31248/400000 [00:03<00:42, 8592.97it/s]  8%|▊         | 32131/400000 [00:03<00:42, 8659.95it/s]  8%|▊         | 32998/400000 [00:03<00:44, 8331.29it/s]  8%|▊         | 33867/400000 [00:04<00:43, 8434.35it/s]  9%|▊         | 34743/400000 [00:04<00:42, 8526.72it/s]  9%|▉         | 35616/400000 [00:04<00:42, 8584.59it/s]  9%|▉         | 36501/400000 [00:04<00:41, 8660.99it/s]  9%|▉         | 37376/400000 [00:04<00:41, 8687.40it/s] 10%|▉         | 38259/400000 [00:04<00:41, 8728.18it/s] 10%|▉         | 39144/400000 [00:04<00:41, 8763.02it/s] 10%|█         | 40026/400000 [00:04<00:41, 8778.54it/s] 10%|█         | 40905/400000 [00:04<00:40, 8781.18it/s] 10%|█         | 41787/400000 [00:04<00:40, 8790.10it/s] 11%|█         | 42668/400000 [00:05<00:40, 8794.32it/s] 11%|█         | 43559/400000 [00:05<00:40, 8826.45it/s] 11%|█         | 44442/400000 [00:05<00:40, 8817.78it/s] 11%|█▏        | 45324/400000 [00:05<00:40, 8806.82it/s] 12%|█▏        | 46205/400000 [00:05<00:40, 8796.15it/s] 12%|█▏        | 47085/400000 [00:05<00:40, 8770.22it/s] 12%|█▏        | 47966/400000 [00:05<00:40, 8780.62it/s] 12%|█▏        | 48851/400000 [00:05<00:39, 8798.99it/s] 12%|█▏        | 49731/400000 [00:05<00:40, 8558.91it/s] 13%|█▎        | 50608/400000 [00:05<00:40, 8620.55it/s] 13%|█▎        | 51486/400000 [00:06<00:40, 8665.91it/s] 13%|█▎        | 52368/400000 [00:06<00:39, 8710.60it/s] 13%|█▎        | 53240/400000 [00:06<00:39, 8699.05it/s] 14%|█▎        | 54111/400000 [00:06<00:39, 8658.38it/s] 14%|█▎        | 54991/400000 [00:06<00:39, 8699.60it/s] 14%|█▍        | 55871/400000 [00:06<00:39, 8728.25it/s] 14%|█▍        | 56756/400000 [00:06<00:39, 8761.96it/s] 14%|█▍        | 57633/400000 [00:06<00:39, 8700.15it/s] 15%|█▍        | 58504/400000 [00:06<00:39, 8633.47it/s] 15%|█▍        | 59377/400000 [00:06<00:39, 8659.46it/s] 15%|█▌        | 60252/400000 [00:07<00:39, 8683.79it/s] 15%|█▌        | 61133/400000 [00:07<00:38, 8719.27it/s] 16%|█▌        | 62008/400000 [00:07<00:38, 8727.36it/s] 16%|█▌        | 62887/400000 [00:07<00:38, 8743.64it/s] 16%|█▌        | 63762/400000 [00:07<00:38, 8699.81it/s] 16%|█▌        | 64633/400000 [00:07<00:38, 8679.43it/s] 16%|█▋        | 65502/400000 [00:07<00:38, 8604.39it/s] 17%|█▋        | 66375/400000 [00:07<00:38, 8639.27it/s] 17%|█▋        | 67244/400000 [00:07<00:38, 8652.15it/s] 17%|█▋        | 68126/400000 [00:07<00:38, 8699.32it/s] 17%|█▋        | 68997/400000 [00:08<00:38, 8684.42it/s] 17%|█▋        | 69873/400000 [00:08<00:37, 8705.75it/s] 18%|█▊        | 70749/400000 [00:08<00:37, 8721.19it/s] 18%|█▊        | 71622/400000 [00:08<00:37, 8701.26it/s] 18%|█▊        | 72493/400000 [00:08<00:37, 8689.18it/s] 18%|█▊        | 73371/400000 [00:08<00:37, 8714.17it/s] 19%|█▊        | 74247/400000 [00:08<00:37, 8727.50it/s] 19%|█▉        | 75130/400000 [00:08<00:37, 8755.70it/s] 19%|█▉        | 76006/400000 [00:08<00:37, 8717.27it/s] 19%|█▉        | 76878/400000 [00:08<00:37, 8698.61it/s] 19%|█▉        | 77758/400000 [00:09<00:36, 8726.56it/s] 20%|█▉        | 78637/400000 [00:09<00:36, 8743.30it/s] 20%|█▉        | 79517/400000 [00:09<00:36, 8758.70it/s] 20%|██        | 80393/400000 [00:09<00:36, 8721.00it/s] 20%|██        | 81272/400000 [00:09<00:36, 8740.05it/s] 21%|██        | 82148/400000 [00:09<00:36, 8745.36it/s] 21%|██        | 83029/400000 [00:09<00:36, 8763.26it/s] 21%|██        | 83906/400000 [00:09<00:36, 8764.41it/s] 21%|██        | 84783/400000 [00:09<00:36, 8641.71it/s] 21%|██▏       | 85648/400000 [00:09<00:37, 8470.61it/s] 22%|██▏       | 86497/400000 [00:10<00:38, 8233.72it/s] 22%|██▏       | 87375/400000 [00:10<00:37, 8387.80it/s] 22%|██▏       | 88245/400000 [00:10<00:36, 8476.46it/s] 22%|██▏       | 89106/400000 [00:10<00:36, 8515.98it/s] 22%|██▏       | 89977/400000 [00:10<00:36, 8572.54it/s] 23%|██▎       | 90836/400000 [00:10<00:36, 8456.18it/s] 23%|██▎       | 91711/400000 [00:10<00:36, 8539.37it/s] 23%|██▎       | 92587/400000 [00:10<00:35, 8603.95it/s] 23%|██▎       | 93449/400000 [00:10<00:35, 8574.31it/s] 24%|██▎       | 94330/400000 [00:11<00:35, 8642.18it/s] 24%|██▍       | 95210/400000 [00:11<00:35, 8686.51it/s] 24%|██▍       | 96080/400000 [00:11<00:35, 8659.71it/s] 24%|██▍       | 96947/400000 [00:11<00:35, 8657.76it/s] 24%|██▍       | 97815/400000 [00:11<00:34, 8662.25it/s] 25%|██▍       | 98697/400000 [00:11<00:34, 8707.86it/s] 25%|██▍       | 99575/400000 [00:11<00:34, 8727.82it/s] 25%|██▌       | 100455/400000 [00:11<00:34, 8747.37it/s] 25%|██▌       | 101337/400000 [00:11<00:34, 8767.16it/s] 26%|██▌       | 102214/400000 [00:11<00:34, 8752.23it/s] 26%|██▌       | 103098/400000 [00:12<00:33, 8776.15it/s] 26%|██▌       | 103976/400000 [00:12<00:33, 8775.92it/s] 26%|██▌       | 104854/400000 [00:12<00:34, 8601.14it/s] 26%|██▋       | 105730/400000 [00:12<00:34, 8647.10it/s] 27%|██▋       | 106596/400000 [00:12<00:33, 8638.16it/s] 27%|██▋       | 107468/400000 [00:12<00:33, 8662.53it/s] 27%|██▋       | 108342/400000 [00:12<00:33, 8683.27it/s] 27%|██▋       | 109218/400000 [00:12<00:33, 8705.67it/s] 28%|██▊       | 110089/400000 [00:12<00:33, 8695.74it/s] 28%|██▊       | 110959/400000 [00:12<00:33, 8641.88it/s] 28%|██▊       | 111836/400000 [00:13<00:33, 8678.06it/s] 28%|██▊       | 112715/400000 [00:13<00:32, 8711.18it/s] 28%|██▊       | 113587/400000 [00:13<00:32, 8713.08it/s] 29%|██▊       | 114463/400000 [00:13<00:32, 8727.03it/s] 29%|██▉       | 115336/400000 [00:13<00:32, 8697.34it/s] 29%|██▉       | 116206/400000 [00:13<00:32, 8675.78it/s] 29%|██▉       | 117074/400000 [00:13<00:32, 8598.48it/s] 29%|██▉       | 117951/400000 [00:13<00:32, 8646.75it/s] 30%|██▉       | 118822/400000 [00:13<00:32, 8663.91it/s] 30%|██▉       | 119693/400000 [00:13<00:32, 8676.65it/s] 30%|███       | 120576/400000 [00:14<00:32, 8719.34it/s] 30%|███       | 121457/400000 [00:14<00:31, 8745.41it/s] 31%|███       | 122345/400000 [00:14<00:31, 8782.66it/s] 31%|███       | 123224/400000 [00:14<00:31, 8777.13it/s] 31%|███       | 124102/400000 [00:14<00:31, 8748.78it/s] 31%|███       | 124988/400000 [00:14<00:31, 8779.02it/s] 31%|███▏      | 125872/400000 [00:14<00:31, 8795.56it/s] 32%|███▏      | 126756/400000 [00:14<00:31, 8808.33it/s] 32%|███▏      | 127637/400000 [00:14<00:31, 8772.72it/s] 32%|███▏      | 128517/400000 [00:14<00:30, 8777.93it/s] 32%|███▏      | 129399/400000 [00:15<00:30, 8790.00it/s] 33%|███▎      | 130282/400000 [00:15<00:30, 8801.21it/s] 33%|███▎      | 131165/400000 [00:15<00:30, 8807.42it/s] 33%|███▎      | 132046/400000 [00:15<00:31, 8618.45it/s] 33%|███▎      | 132925/400000 [00:15<00:30, 8667.71it/s] 33%|███▎      | 133793/400000 [00:15<00:30, 8637.68it/s] 34%|███▎      | 134658/400000 [00:15<00:30, 8637.30it/s] 34%|███▍      | 135533/400000 [00:15<00:30, 8669.83it/s] 34%|███▍      | 136408/400000 [00:15<00:30, 8692.84it/s] 34%|███▍      | 137278/400000 [00:15<00:30, 8681.61it/s] 35%|███▍      | 138163/400000 [00:16<00:29, 8731.37it/s] 35%|███▍      | 139043/400000 [00:16<00:29, 8737.80it/s] 35%|███▍      | 139917/400000 [00:16<00:29, 8730.79it/s] 35%|███▌      | 140791/400000 [00:16<00:29, 8712.68it/s] 35%|███▌      | 141672/400000 [00:16<00:29, 8739.14it/s] 36%|███▌      | 142565/400000 [00:16<00:29, 8795.50it/s] 36%|███▌      | 143448/400000 [00:16<00:29, 8804.04it/s] 36%|███▌      | 144334/400000 [00:16<00:28, 8818.84it/s] 36%|███▋      | 145216/400000 [00:16<00:28, 8799.82it/s] 37%|███▋      | 146099/400000 [00:16<00:28, 8808.34it/s] 37%|███▋      | 146980/400000 [00:17<00:28, 8808.61it/s] 37%|███▋      | 147864/400000 [00:17<00:28, 8816.54it/s] 37%|███▋      | 148750/400000 [00:17<00:28, 8827.69it/s] 37%|███▋      | 149633/400000 [00:17<00:28, 8798.58it/s] 38%|███▊      | 150517/400000 [00:17<00:28, 8808.79it/s] 38%|███▊      | 151400/400000 [00:17<00:28, 8813.93it/s] 38%|███▊      | 152282/400000 [00:17<00:28, 8732.62it/s] 38%|███▊      | 153156/400000 [00:17<00:28, 8597.57it/s] 39%|███▊      | 154032/400000 [00:17<00:28, 8643.25it/s] 39%|███▊      | 154897/400000 [00:17<00:28, 8641.85it/s] 39%|███▉      | 155762/400000 [00:18<00:28, 8611.47it/s] 39%|███▉      | 156632/400000 [00:18<00:28, 8636.41it/s] 39%|███▉      | 157505/400000 [00:18<00:27, 8661.69it/s] 40%|███▉      | 158385/400000 [00:18<00:27, 8701.79it/s] 40%|███▉      | 159266/400000 [00:18<00:27, 8731.76it/s] 40%|████      | 160140/400000 [00:18<00:27, 8691.01it/s] 40%|████      | 161022/400000 [00:18<00:27, 8726.95it/s] 40%|████      | 161897/400000 [00:18<00:27, 8733.18it/s] 41%|████      | 162777/400000 [00:18<00:27, 8750.46it/s] 41%|████      | 163653/400000 [00:18<00:27, 8650.40it/s] 41%|████      | 164533/400000 [00:19<00:27, 8694.64it/s] 41%|████▏     | 165410/400000 [00:19<00:26, 8715.17it/s] 42%|████▏     | 166284/400000 [00:19<00:26, 8722.54it/s] 42%|████▏     | 167171/400000 [00:19<00:26, 8765.47it/s] 42%|████▏     | 168048/400000 [00:19<00:26, 8738.01it/s] 42%|████▏     | 168922/400000 [00:19<00:26, 8696.53it/s] 42%|████▏     | 169806/400000 [00:19<00:26, 8738.24it/s] 43%|████▎     | 170684/400000 [00:19<00:26, 8750.26it/s] 43%|████▎     | 171563/400000 [00:19<00:26, 8759.62it/s] 43%|████▎     | 172440/400000 [00:19<00:26, 8660.15it/s] 43%|████▎     | 173307/400000 [00:20<00:26, 8573.10it/s] 44%|████▎     | 174190/400000 [00:20<00:26, 8648.30it/s] 44%|████▍     | 175064/400000 [00:20<00:25, 8673.24it/s] 44%|████▍     | 175950/400000 [00:20<00:25, 8727.23it/s] 44%|████▍     | 176838/400000 [00:20<00:25, 8770.59it/s] 44%|████▍     | 177716/400000 [00:20<00:25, 8760.48it/s] 45%|████▍     | 178593/400000 [00:20<00:25, 8647.16it/s] 45%|████▍     | 179459/400000 [00:20<00:25, 8606.16it/s] 45%|████▌     | 180320/400000 [00:20<00:26, 8432.79it/s] 45%|████▌     | 181203/400000 [00:20<00:25, 8545.39it/s] 46%|████▌     | 182077/400000 [00:21<00:25, 8602.34it/s] 46%|████▌     | 182958/400000 [00:21<00:25, 8661.57it/s] 46%|████▌     | 183830/400000 [00:21<00:24, 8677.10it/s] 46%|████▌     | 184712/400000 [00:21<00:24, 8717.41it/s] 46%|████▋     | 185595/400000 [00:21<00:24, 8748.86it/s] 47%|████▋     | 186471/400000 [00:21<00:24, 8704.14it/s] 47%|████▋     | 187356/400000 [00:21<00:24, 8747.13it/s] 47%|████▋     | 188231/400000 [00:21<00:24, 8732.39it/s] 47%|████▋     | 189110/400000 [00:21<00:24, 8747.93it/s] 47%|████▋     | 189997/400000 [00:21<00:23, 8782.39it/s] 48%|████▊     | 190883/400000 [00:22<00:23, 8804.52it/s] 48%|████▊     | 191765/400000 [00:22<00:23, 8808.07it/s] 48%|████▊     | 192646/400000 [00:22<00:23, 8753.95it/s] 48%|████▊     | 193536/400000 [00:22<00:23, 8795.92it/s] 49%|████▊     | 194416/400000 [00:22<00:23, 8792.81it/s] 49%|████▉     | 195296/400000 [00:22<00:24, 8310.46it/s] 49%|████▉     | 196158/400000 [00:22<00:24, 8400.93it/s] 49%|████▉     | 197040/400000 [00:22<00:23, 8522.21it/s] 49%|████▉     | 197924/400000 [00:22<00:23, 8613.63it/s] 50%|████▉     | 198810/400000 [00:23<00:23, 8683.98it/s] 50%|████▉     | 199688/400000 [00:23<00:22, 8710.54it/s] 50%|█████     | 200561/400000 [00:23<00:23, 8423.66it/s] 50%|█████     | 201407/400000 [00:23<00:23, 8422.55it/s] 51%|█████     | 202290/400000 [00:23<00:23, 8538.20it/s] 51%|█████     | 203168/400000 [00:23<00:22, 8609.08it/s] 51%|█████     | 204047/400000 [00:23<00:22, 8659.87it/s] 51%|█████     | 204922/400000 [00:23<00:22, 8685.53it/s] 51%|█████▏    | 205792/400000 [00:23<00:22, 8464.43it/s] 52%|█████▏    | 206641/400000 [00:23<00:23, 8385.71it/s] 52%|█████▏    | 207531/400000 [00:24<00:22, 8533.25it/s] 52%|█████▏    | 208386/400000 [00:24<00:22, 8447.23it/s] 52%|█████▏    | 209243/400000 [00:24<00:22, 8481.84it/s] 53%|█████▎    | 210104/400000 [00:24<00:22, 8519.67it/s] 53%|█████▎    | 210957/400000 [00:24<00:22, 8379.00it/s] 53%|█████▎    | 211841/400000 [00:24<00:22, 8512.14it/s] 53%|█████▎    | 212720/400000 [00:24<00:21, 8593.37it/s] 53%|█████▎    | 213592/400000 [00:24<00:21, 8628.31it/s] 54%|█████▎    | 214468/400000 [00:24<00:21, 8665.36it/s] 54%|█████▍    | 215336/400000 [00:24<00:21, 8620.51it/s] 54%|█████▍    | 216218/400000 [00:25<00:21, 8676.66it/s] 54%|█████▍    | 217092/400000 [00:25<00:21, 8695.06it/s] 54%|█████▍    | 217962/400000 [00:25<00:20, 8693.66it/s] 55%|█████▍    | 218833/400000 [00:25<00:20, 8697.23it/s] 55%|█████▍    | 219708/400000 [00:25<00:20, 8710.39it/s] 55%|█████▌    | 220586/400000 [00:25<00:20, 8727.53it/s] 55%|█████▌    | 221459/400000 [00:25<00:20, 8551.28it/s] 56%|█████▌    | 222326/400000 [00:25<00:20, 8586.20it/s] 56%|█████▌    | 223186/400000 [00:25<00:20, 8556.76it/s] 56%|█████▌    | 224063/400000 [00:25<00:20, 8618.75it/s] 56%|█████▌    | 224926/400000 [00:26<00:20, 8563.61it/s] 56%|█████▋    | 225796/400000 [00:26<00:20, 8602.76it/s] 57%|█████▋    | 226665/400000 [00:26<00:20, 8626.77it/s] 57%|█████▋    | 227545/400000 [00:26<00:19, 8677.64it/s] 57%|█████▋    | 228424/400000 [00:26<00:19, 8710.35it/s] 57%|█████▋    | 229296/400000 [00:26<00:19, 8712.85it/s] 58%|█████▊    | 230168/400000 [00:26<00:19, 8713.93it/s] 58%|█████▊    | 231043/400000 [00:26<00:19, 8722.43it/s] 58%|█████▊    | 231929/400000 [00:26<00:19, 8761.96it/s] 58%|█████▊    | 232812/400000 [00:26<00:19, 8780.25it/s] 58%|█████▊    | 233691/400000 [00:27<00:18, 8769.27it/s] 59%|█████▊    | 234568/400000 [00:27<00:18, 8758.25it/s] 59%|█████▉    | 235450/400000 [00:27<00:18, 8774.13it/s] 59%|█████▉    | 236336/400000 [00:27<00:18, 8796.77it/s] 59%|█████▉    | 237219/400000 [00:27<00:18, 8806.58it/s] 60%|█████▉    | 238101/400000 [00:27<00:18, 8809.94it/s] 60%|█████▉    | 238983/400000 [00:27<00:18, 8684.17it/s] 60%|█████▉    | 239852/400000 [00:27<00:18, 8675.16it/s] 60%|██████    | 240720/400000 [00:27<00:18, 8668.60it/s] 60%|██████    | 241592/400000 [00:27<00:18, 8681.31it/s] 61%|██████    | 242472/400000 [00:28<00:18, 8715.46it/s] 61%|██████    | 243344/400000 [00:28<00:18, 8702.87it/s] 61%|██████    | 244221/400000 [00:28<00:17, 8720.93it/s] 61%|██████▏   | 245100/400000 [00:28<00:17, 8740.00it/s] 61%|██████▏   | 245983/400000 [00:28<00:17, 8765.81it/s] 62%|██████▏   | 246867/400000 [00:28<00:17, 8787.09it/s] 62%|██████▏   | 247746/400000 [00:28<00:17, 8767.48it/s] 62%|██████▏   | 248626/400000 [00:28<00:17, 8777.18it/s] 62%|██████▏   | 249506/400000 [00:28<00:17, 8781.93it/s] 63%|██████▎   | 250390/400000 [00:28<00:17, 8796.61it/s] 63%|██████▎   | 251271/400000 [00:29<00:16, 8798.67it/s] 63%|██████▎   | 252151/400000 [00:29<00:16, 8796.61it/s] 63%|██████▎   | 253031/400000 [00:29<00:16, 8781.33it/s] 63%|██████▎   | 253910/400000 [00:29<00:16, 8765.18it/s] 64%|██████▎   | 254787/400000 [00:29<00:16, 8749.80it/s] 64%|██████▍   | 255672/400000 [00:29<00:16, 8779.42it/s] 64%|██████▍   | 256550/400000 [00:29<00:16, 8774.94it/s] 64%|██████▍   | 257428/400000 [00:29<00:16, 8774.45it/s] 65%|██████▍   | 258306/400000 [00:29<00:16, 8713.64it/s] 65%|██████▍   | 259178/400000 [00:29<00:16, 8712.18it/s] 65%|██████▌   | 260052/400000 [00:30<00:16, 8718.02it/s] 65%|██████▌   | 260924/400000 [00:30<00:16, 8680.53it/s] 65%|██████▌   | 261798/400000 [00:30<00:15, 8697.62it/s] 66%|██████▌   | 262677/400000 [00:30<00:15, 8723.43it/s] 66%|██████▌   | 263551/400000 [00:30<00:15, 8728.08it/s] 66%|██████▌   | 264424/400000 [00:30<00:15, 8713.72it/s] 66%|██████▋   | 265296/400000 [00:30<00:15, 8448.08it/s] 67%|██████▋   | 266160/400000 [00:30<00:15, 8503.95it/s] 67%|██████▋   | 267036/400000 [00:30<00:15, 8577.36it/s] 67%|██████▋   | 267895/400000 [00:30<00:15, 8565.87it/s] 67%|██████▋   | 268753/400000 [00:31<00:15, 8568.80it/s] 67%|██████▋   | 269613/400000 [00:31<00:15, 8576.83it/s] 68%|██████▊   | 270472/400000 [00:31<00:15, 8566.99it/s] 68%|██████▊   | 271357/400000 [00:31<00:14, 8647.36it/s] 68%|██████▊   | 272235/400000 [00:31<00:14, 8684.01it/s] 68%|██████▊   | 273112/400000 [00:31<00:14, 8707.68it/s] 68%|██████▊   | 274000/400000 [00:31<00:14, 8758.38it/s] 69%|██████▊   | 274877/400000 [00:31<00:14, 8727.08it/s] 69%|██████▉   | 275754/400000 [00:31<00:14, 8739.04it/s] 69%|██████▉   | 276630/400000 [00:31<00:14, 8743.33it/s] 69%|██████▉   | 277505/400000 [00:32<00:14, 8739.02it/s] 70%|██████▉   | 278379/400000 [00:32<00:13, 8734.49it/s] 70%|██████▉   | 279258/400000 [00:32<00:13, 8750.28it/s] 70%|███████   | 280141/400000 [00:32<00:13, 8772.35it/s] 70%|███████   | 281019/400000 [00:32<00:13, 8765.55it/s] 70%|███████   | 281896/400000 [00:32<00:13, 8740.94it/s] 71%|███████   | 282771/400000 [00:32<00:13, 8649.54it/s] 71%|███████   | 283637/400000 [00:32<00:13, 8640.16it/s] 71%|███████   | 284510/400000 [00:32<00:13, 8666.87it/s] 71%|███████▏  | 285395/400000 [00:32<00:13, 8719.19it/s] 72%|███████▏  | 286273/400000 [00:33<00:13, 8736.26it/s] 72%|███████▏  | 287147/400000 [00:33<00:13, 8671.88it/s] 72%|███████▏  | 288015/400000 [00:33<00:12, 8644.73it/s] 72%|███████▏  | 288888/400000 [00:33<00:12, 8668.27it/s] 72%|███████▏  | 289764/400000 [00:33<00:12, 8695.29it/s] 73%|███████▎  | 290634/400000 [00:33<00:12, 8580.40it/s] 73%|███████▎  | 291495/400000 [00:33<00:12, 8588.52it/s] 73%|███████▎  | 292368/400000 [00:33<00:12, 8627.83it/s] 73%|███████▎  | 293246/400000 [00:33<00:12, 8671.44it/s] 74%|███████▎  | 294114/400000 [00:34<00:12, 8348.49it/s] 74%|███████▎  | 294952/400000 [00:34<00:12, 8161.42it/s] 74%|███████▍  | 295829/400000 [00:34<00:12, 8333.22it/s] 74%|███████▍  | 296705/400000 [00:34<00:12, 8455.34it/s] 74%|███████▍  | 297579/400000 [00:34<00:11, 8537.26it/s] 75%|███████▍  | 298461/400000 [00:34<00:11, 8619.64it/s] 75%|███████▍  | 299325/400000 [00:34<00:11, 8624.65it/s] 75%|███████▌  | 300200/400000 [00:34<00:11, 8661.67it/s] 75%|███████▌  | 301083/400000 [00:34<00:11, 8711.17it/s] 75%|███████▌  | 301966/400000 [00:34<00:11, 8744.69it/s] 76%|███████▌  | 302841/400000 [00:35<00:11, 8744.04it/s] 76%|███████▌  | 303716/400000 [00:35<00:11, 8731.90it/s] 76%|███████▌  | 304593/400000 [00:35<00:10, 8740.81it/s] 76%|███████▋  | 305472/400000 [00:35<00:10, 8752.79it/s] 77%|███████▋  | 306348/400000 [00:35<00:10, 8753.65it/s] 77%|███████▋  | 307224/400000 [00:35<00:10, 8739.76it/s] 77%|███████▋  | 308099/400000 [00:35<00:10, 8464.91it/s] 77%|███████▋  | 308978/400000 [00:35<00:10, 8559.54it/s] 77%|███████▋  | 309843/400000 [00:35<00:10, 8585.71it/s] 78%|███████▊  | 310724/400000 [00:35<00:10, 8649.04it/s] 78%|███████▊  | 311606/400000 [00:36<00:10, 8699.33it/s] 78%|███████▊  | 312491/400000 [00:36<00:10, 8742.83it/s] 78%|███████▊  | 313374/400000 [00:36<00:09, 8768.10it/s] 79%|███████▊  | 314255/400000 [00:36<00:09, 8777.93it/s] 79%|███████▉  | 315134/400000 [00:36<00:09, 8741.56it/s] 79%|███████▉  | 316010/400000 [00:36<00:09, 8745.38it/s] 79%|███████▉  | 316895/400000 [00:36<00:09, 8775.99it/s] 79%|███████▉  | 317779/400000 [00:36<00:09, 8794.07it/s] 80%|███████▉  | 318660/400000 [00:36<00:09, 8796.66it/s] 80%|███████▉  | 319540/400000 [00:36<00:09, 8770.09it/s] 80%|████████  | 320418/400000 [00:37<00:09, 8749.30it/s] 80%|████████  | 321297/400000 [00:37<00:08, 8760.99it/s] 81%|████████  | 322179/400000 [00:37<00:08, 8776.27it/s] 81%|████████  | 323061/400000 [00:37<00:08, 8787.31it/s] 81%|████████  | 323940/400000 [00:37<00:08, 8771.81it/s] 81%|████████  | 324818/400000 [00:37<00:08, 8725.93it/s] 81%|████████▏ | 325697/400000 [00:37<00:08, 8743.41it/s] 82%|████████▏ | 326572/400000 [00:37<00:08, 8742.13it/s] 82%|████████▏ | 327447/400000 [00:37<00:08, 8743.33it/s] 82%|████████▏ | 328325/400000 [00:37<00:08, 8752.44it/s] 82%|████████▏ | 329201/400000 [00:38<00:08, 8718.29it/s] 83%|████████▎ | 330077/400000 [00:38<00:08, 8729.53it/s] 83%|████████▎ | 330950/400000 [00:38<00:07, 8725.22it/s] 83%|████████▎ | 331825/400000 [00:38<00:07, 8731.61it/s] 83%|████████▎ | 332707/400000 [00:38<00:07, 8755.64it/s] 83%|████████▎ | 333585/400000 [00:38<00:07, 8761.71it/s] 84%|████████▎ | 334469/400000 [00:38<00:07, 8784.03it/s] 84%|████████▍ | 335348/400000 [00:38<00:07, 8763.44it/s] 84%|████████▍ | 336225/400000 [00:38<00:07, 8575.76it/s] 84%|████████▍ | 337088/400000 [00:38<00:07, 8591.43it/s] 84%|████████▍ | 337963/400000 [00:39<00:07, 8637.38it/s] 85%|████████▍ | 338843/400000 [00:39<00:07, 8684.71it/s] 85%|████████▍ | 339712/400000 [00:39<00:06, 8677.03it/s] 85%|████████▌ | 340595/400000 [00:39<00:06, 8721.05it/s] 85%|████████▌ | 341468/400000 [00:39<00:06, 8722.17it/s] 86%|████████▌ | 342350/400000 [00:39<00:06, 8750.78it/s] 86%|████████▌ | 343231/400000 [00:39<00:06, 8768.27it/s] 86%|████████▌ | 344120/400000 [00:39<00:06, 8803.57it/s] 86%|████████▋ | 345004/400000 [00:39<00:06, 8812.81it/s] 86%|████████▋ | 345886/400000 [00:39<00:06, 8757.70it/s] 87%|████████▋ | 346762/400000 [00:40<00:06, 8755.67it/s] 87%|████████▋ | 347638/400000 [00:40<00:05, 8734.21it/s] 87%|████████▋ | 348517/400000 [00:40<00:05, 8749.31it/s] 87%|████████▋ | 349393/400000 [00:40<00:05, 8751.01it/s] 88%|████████▊ | 350269/400000 [00:40<00:05, 8674.52it/s] 88%|████████▊ | 351146/400000 [00:40<00:05, 8702.53it/s] 88%|████████▊ | 352021/400000 [00:40<00:05, 8714.72it/s] 88%|████████▊ | 352903/400000 [00:40<00:05, 8744.73it/s] 88%|████████▊ | 353778/400000 [00:40<00:05, 8524.72it/s] 89%|████████▊ | 354632/400000 [00:40<00:05, 8394.98it/s] 89%|████████▉ | 355511/400000 [00:41<00:05, 8507.74it/s] 89%|████████▉ | 356372/400000 [00:41<00:05, 8535.85it/s] 89%|████████▉ | 357254/400000 [00:41<00:04, 8617.89it/s] 90%|████████▉ | 358117/400000 [00:41<00:05, 8312.92it/s] 90%|████████▉ | 358968/400000 [00:41<00:04, 8370.06it/s] 90%|████████▉ | 359850/400000 [00:41<00:04, 8497.75it/s] 90%|█████████ | 360733/400000 [00:41<00:04, 8592.50it/s] 90%|█████████ | 361614/400000 [00:41<00:04, 8655.80it/s] 91%|█████████ | 362490/400000 [00:41<00:04, 8685.69it/s] 91%|█████████ | 363360/400000 [00:41<00:04, 8676.12it/s] 91%|█████████ | 364229/400000 [00:42<00:04, 8596.16it/s] 91%|█████████▏| 365107/400000 [00:42<00:04, 8649.10it/s] 91%|█████████▏| 365987/400000 [00:42<00:03, 8692.58it/s] 92%|█████████▏| 366869/400000 [00:42<00:03, 8727.88it/s] 92%|█████████▏| 367743/400000 [00:42<00:03, 8705.47it/s] 92%|█████████▏| 368624/400000 [00:42<00:03, 8734.71it/s] 92%|█████████▏| 369506/400000 [00:42<00:03, 8759.06it/s] 93%|█████████▎| 370383/400000 [00:42<00:03, 8687.35it/s] 93%|█████████▎| 371252/400000 [00:42<00:03, 8536.87it/s] 93%|█████████▎| 372107/400000 [00:42<00:03, 8511.90it/s] 93%|█████████▎| 372986/400000 [00:43<00:03, 8592.11it/s] 93%|█████████▎| 373851/400000 [00:43<00:03, 8608.05it/s] 94%|█████████▎| 374734/400000 [00:43<00:02, 8673.43it/s] 94%|█████████▍| 375614/400000 [00:43<00:02, 8711.01it/s] 94%|█████████▍| 376495/400000 [00:43<00:02, 8738.97it/s] 94%|█████████▍| 377384/400000 [00:43<00:02, 8783.61it/s] 95%|█████████▍| 378265/400000 [00:43<00:02, 8790.36it/s] 95%|█████████▍| 379148/400000 [00:43<00:02, 8802.20it/s] 95%|█████████▌| 380029/400000 [00:43<00:02, 8779.03it/s] 95%|█████████▌| 380907/400000 [00:44<00:02, 8453.88it/s] 95%|█████████▌| 381768/400000 [00:44<00:02, 8499.54it/s] 96%|█████████▌| 382639/400000 [00:44<00:02, 8560.46it/s] 96%|█████████▌| 383519/400000 [00:44<00:01, 8630.66it/s] 96%|█████████▌| 384395/400000 [00:44<00:01, 8669.04it/s] 96%|█████████▋| 385263/400000 [00:44<00:01, 8670.04it/s] 97%|█████████▋| 386144/400000 [00:44<00:01, 8710.52it/s] 97%|█████████▋| 387016/400000 [00:44<00:01, 8703.48it/s] 97%|█████████▋| 387897/400000 [00:44<00:01, 8734.36it/s] 97%|█████████▋| 388779/400000 [00:44<00:01, 8759.27it/s] 97%|█████████▋| 389656/400000 [00:45<00:01, 8737.44it/s] 98%|█████████▊| 390537/400000 [00:45<00:01, 8756.74it/s] 98%|█████████▊| 391420/400000 [00:45<00:00, 8777.71it/s] 98%|█████████▊| 392306/400000 [00:45<00:00, 8800.99it/s] 98%|█████████▊| 393189/400000 [00:45<00:00, 8808.62it/s] 99%|█████████▊| 394070/400000 [00:45<00:00, 8722.70it/s] 99%|█████████▊| 394947/400000 [00:45<00:00, 8735.80it/s] 99%|█████████▉| 395827/400000 [00:45<00:00, 8753.05it/s] 99%|█████████▉| 396703/400000 [00:45<00:00, 8733.66it/s] 99%|█████████▉| 397577/400000 [00:45<00:00, 8723.87it/s]100%|█████████▉| 398450/400000 [00:46<00:00, 8692.71it/s]100%|█████████▉| 399320/400000 [00:46<00:00, 8674.77it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8658.54it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fef5c5285c0>, <torchtext.data.dataset.TabularDataset object at 0x7fef5c528470>, <torchtext.vocab.Vocab object at 0x7fef5c528550>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.34 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.40 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.40 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.51 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.51 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.82 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.82 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.35 file/s]2020-06-19 12:18:02.001386: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-19 12:18:02.005785: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-06-19 12:18:02.005946: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ffef960ed0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-19 12:18:02.005961: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 454191.50it/s] 80%|███████▉  | 7905280/9912422 [00:00<00:03, 647240.89it/s]9920512it [00:00, 43463400.67it/s]                           
0it [00:00, ?it/s]32768it [00:00, 733308.90it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 999732.76it/s]1654784it [00:00, 12251017.85it/s]                          
0it [00:00, ?it/s]8192it [00:00, 227579.59it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

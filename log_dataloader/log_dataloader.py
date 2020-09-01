
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fb36e55d6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fb36e55d6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fb3d9286488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fb3d9286488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fb3f84a5ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fb3f84a5ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb3865bc488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb3865bc488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb3865bc488> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:22, 119871.33it/s] 70%|███████   | 6963200/9912422 [00:00<00:17, 171118.01it/s]9920512it [00:00, 38144448.85it/s]                           
0it [00:00, ?it/s]32768it [00:00, 581535.57it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 5052788.67it/s]          
0it [00:00, ?it/s]8192it [00:00, 83095.30it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb36d814e80>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb38475d9e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fb3865bc158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fb3865bc158> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fb3865bc158> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:47,  1.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:47,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:46,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:46,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:45,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:44,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:44,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:43,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:42,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:12,  2.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:12,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:11,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:11,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:10,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:10,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:09,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:09,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:08,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:08,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:48,  2.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:48,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:47,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:47,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:47,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:46,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:46,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:46,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:45,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:32,  4.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:32,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:32,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:31,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:31,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:31,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:31,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:30,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:30,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:30,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:21,  5.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:21,  5.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:21,  5.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:21,  5.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:21,  5.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:20,  5.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:20,  5.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:20,  5.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:20,  5.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:20,  5.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:14,  8.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:14,  8.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:14,  8.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:14,  8.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:14,  8.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:13,  8.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:13,  8.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:13,  8.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:13,  8.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:13,  8.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 15.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 15.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:06, 15.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:06, 15.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:06, 15.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 20.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 32.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 32.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 32.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 32.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 32.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 32.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 32.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 32.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 32.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 32.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 39.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 39.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 39.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 39.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 39.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 39.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 39.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 39.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 39.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 39.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 47.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 47.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 47.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 47.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 47.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 47.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:01, 47.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:01, 47.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:01, 47.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 47.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 53.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 53.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 53.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 53.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 53.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 53.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 53.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 53.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 53.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 53.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 60.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 60.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 69.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 69.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 72.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 72.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 72.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 72.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 72.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 72.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 72.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 75.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 75.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.63s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.63s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.63s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.37 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.63s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 75.37 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.25 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ url]
0 examples [00:00, ? examples/s]2020-09-01 12:07:42.803525: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-01 12:07:42.815577: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-09-01 12:07:42.815834: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d429e06ac0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-01 12:07:42.815860: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
20 examples [00:00, 199.29 examples/s]129 examples [00:00, 263.91 examples/s]236 examples [00:00, 340.94 examples/s]344 examples [00:00, 428.55 examples/s]448 examples [00:00, 519.72 examples/s]557 examples [00:00, 616.24 examples/s]666 examples [00:00, 707.61 examples/s]775 examples [00:00, 789.68 examples/s]884 examples [00:00, 860.19 examples/s]991 examples [00:01, 913.50 examples/s]1101 examples [00:01, 960.99 examples/s]1207 examples [00:01, 981.98 examples/s]1316 examples [00:01, 1010.47 examples/s]1424 examples [00:01, 1030.03 examples/s]1531 examples [00:01, 1038.64 examples/s]1641 examples [00:01, 1054.17 examples/s]1750 examples [00:01, 1062.94 examples/s]1858 examples [00:01, 1061.74 examples/s]1966 examples [00:01, 1058.81 examples/s]2074 examples [00:02, 1063.27 examples/s]2184 examples [00:02, 1071.79 examples/s]2293 examples [00:02, 1075.34 examples/s]2401 examples [00:02, 1073.94 examples/s]2510 examples [00:02, 1076.95 examples/s]2618 examples [00:02, 1077.14 examples/s]2726 examples [00:02, 1071.94 examples/s]2834 examples [00:02, 1073.47 examples/s]2942 examples [00:02, 1071.98 examples/s]3051 examples [00:02, 1075.99 examples/s]3159 examples [00:03, 1063.74 examples/s]3269 examples [00:03, 1071.89 examples/s]3377 examples [00:03, 1064.01 examples/s]3484 examples [00:03, 1056.02 examples/s]3594 examples [00:03, 1066.40 examples/s]3701 examples [00:03, 1064.01 examples/s]3808 examples [00:03, 1054.56 examples/s]3916 examples [00:03, 1061.88 examples/s]4025 examples [00:03, 1068.51 examples/s]4133 examples [00:03, 1070.44 examples/s]4241 examples [00:04, 1067.70 examples/s]4349 examples [00:04, 1069.02 examples/s]4456 examples [00:04, 1050.89 examples/s]4562 examples [00:04, 1052.24 examples/s]4669 examples [00:04, 1055.56 examples/s]4776 examples [00:04, 1058.92 examples/s]4886 examples [00:04, 1069.10 examples/s]4993 examples [00:04, 1067.30 examples/s]5100 examples [00:04, 1065.79 examples/s]5208 examples [00:04, 1069.96 examples/s]5316 examples [00:05, 1065.22 examples/s]5423 examples [00:05, 1063.51 examples/s]5533 examples [00:05, 1072.95 examples/s]5641 examples [00:05, 1058.69 examples/s]5750 examples [00:05, 1065.50 examples/s]5859 examples [00:05, 1072.35 examples/s]5969 examples [00:05, 1078.84 examples/s]6078 examples [00:05, 1081.25 examples/s]6188 examples [00:05, 1084.82 examples/s]6297 examples [00:05, 1064.59 examples/s]6405 examples [00:06, 1068.20 examples/s]6513 examples [00:06, 1068.89 examples/s]6623 examples [00:06, 1076.14 examples/s]6732 examples [00:06, 1079.18 examples/s]6840 examples [00:06, 1076.41 examples/s]6948 examples [00:06, 1076.73 examples/s]7058 examples [00:06, 1080.75 examples/s]7168 examples [00:06, 1085.16 examples/s]7277 examples [00:06, 1084.40 examples/s]7386 examples [00:06, 1084.72 examples/s]7495 examples [00:07, 1080.29 examples/s]7604 examples [00:07, 1080.75 examples/s]7713 examples [00:07, 1082.79 examples/s]7823 examples [00:07, 1085.53 examples/s]7932 examples [00:07, 1085.95 examples/s]8041 examples [00:07, 1076.97 examples/s]8150 examples [00:07, 1079.77 examples/s]8258 examples [00:07, 1065.32 examples/s]8365 examples [00:07, 1059.20 examples/s]8472 examples [00:07, 1061.54 examples/s]8579 examples [00:08, 1050.52 examples/s]8686 examples [00:08, 1056.04 examples/s]8795 examples [00:08, 1065.81 examples/s]8902 examples [00:08, 1065.30 examples/s]9012 examples [00:08, 1073.39 examples/s]9120 examples [00:08, 1073.89 examples/s]9230 examples [00:08, 1080.21 examples/s]9339 examples [00:08, 1082.29 examples/s]9449 examples [00:08, 1085.26 examples/s]9558 examples [00:08, 1086.29 examples/s]9667 examples [00:09, 1075.05 examples/s]9775 examples [00:09, 1069.44 examples/s]9882 examples [00:09, 1066.04 examples/s]9991 examples [00:09, 1071.43 examples/s]10099 examples [00:09, 1020.84 examples/s]10204 examples [00:09, 1028.23 examples/s]10313 examples [00:09, 1043.56 examples/s]10422 examples [00:09, 1054.76 examples/s]10531 examples [00:09, 1062.66 examples/s]10640 examples [00:10, 1068.47 examples/s]10747 examples [00:10, 1064.70 examples/s]10854 examples [00:10, 1064.25 examples/s]10963 examples [00:10, 1070.63 examples/s]11071 examples [00:10, 1073.31 examples/s]11180 examples [00:10, 1076.95 examples/s]11288 examples [00:10, 1070.02 examples/s]11397 examples [00:10, 1073.74 examples/s]11506 examples [00:10, 1078.47 examples/s]11615 examples [00:10, 1079.90 examples/s]11724 examples [00:11, 1073.45 examples/s]11832 examples [00:11, 1067.15 examples/s]11941 examples [00:11, 1072.02 examples/s]12049 examples [00:11, 1069.53 examples/s]12156 examples [00:11, 1068.31 examples/s]12266 examples [00:11, 1075.26 examples/s]12375 examples [00:11, 1079.15 examples/s]12485 examples [00:11, 1084.24 examples/s]12594 examples [00:11, 1082.13 examples/s]12703 examples [00:11, 1084.46 examples/s]12813 examples [00:12, 1087.43 examples/s]12922 examples [00:12, 1083.43 examples/s]13031 examples [00:12, 1083.05 examples/s]13140 examples [00:12, 1083.21 examples/s]13249 examples [00:12, 1081.20 examples/s]13358 examples [00:12, 1070.33 examples/s]13466 examples [00:12, 1070.82 examples/s]13575 examples [00:12, 1076.12 examples/s]13685 examples [00:12, 1081.38 examples/s]13794 examples [00:12, 1074.42 examples/s]13903 examples [00:13, 1078.15 examples/s]14011 examples [00:13, 1078.45 examples/s]14120 examples [00:13, 1080.69 examples/s]14229 examples [00:13, 1080.54 examples/s]14338 examples [00:13, 1082.15 examples/s]14447 examples [00:13, 1083.46 examples/s]14556 examples [00:13, 1081.71 examples/s]14665 examples [00:13, 1081.97 examples/s]14775 examples [00:13, 1084.41 examples/s]14884 examples [00:13, 1080.80 examples/s]14993 examples [00:14, 1052.99 examples/s]15102 examples [00:14, 1062.18 examples/s]15210 examples [00:14, 1064.64 examples/s]15317 examples [00:14, 1056.58 examples/s]15426 examples [00:14, 1065.74 examples/s]15536 examples [00:14, 1073.13 examples/s]15645 examples [00:14, 1076.65 examples/s]15753 examples [00:14, 1069.88 examples/s]15863 examples [00:14, 1077.65 examples/s]15971 examples [00:14, 1077.36 examples/s]16079 examples [00:15, 1076.73 examples/s]16187 examples [00:15, 1069.26 examples/s]16294 examples [00:15, 1046.82 examples/s]16404 examples [00:15, 1061.10 examples/s]16514 examples [00:15, 1069.86 examples/s]16623 examples [00:15, 1075.73 examples/s]16733 examples [00:15, 1080.11 examples/s]16842 examples [00:15, 1073.73 examples/s]16950 examples [00:15, 1073.52 examples/s]17059 examples [00:15, 1077.10 examples/s]17167 examples [00:16, 1073.93 examples/s]17276 examples [00:16, 1077.89 examples/s]17384 examples [00:16, 1069.41 examples/s]17493 examples [00:16, 1074.24 examples/s]17603 examples [00:16, 1079.50 examples/s]17712 examples [00:16, 1082.11 examples/s]17821 examples [00:16, 1081.87 examples/s]17930 examples [00:16, 1076.41 examples/s]18039 examples [00:16, 1079.94 examples/s]18148 examples [00:17, 1080.86 examples/s]18257 examples [00:17, 1079.27 examples/s]18366 examples [00:17, 1080.25 examples/s]18475 examples [00:17, 1072.46 examples/s]18583 examples [00:17, 1072.30 examples/s]18693 examples [00:17, 1078.01 examples/s]18802 examples [00:17, 1079.11 examples/s]18911 examples [00:17, 1079.86 examples/s]19019 examples [00:17, 1071.54 examples/s]19128 examples [00:17, 1074.04 examples/s]19238 examples [00:18, 1081.09 examples/s]19347 examples [00:18, 1076.81 examples/s]19456 examples [00:18, 1079.44 examples/s]19564 examples [00:18, 1071.51 examples/s]19673 examples [00:18, 1076.65 examples/s]19782 examples [00:18, 1080.24 examples/s]19892 examples [00:18, 1083.31 examples/s]20001 examples [00:18, 1032.84 examples/s]20106 examples [00:18, 1037.15 examples/s]20216 examples [00:18, 1054.93 examples/s]20322 examples [00:19, 1037.06 examples/s]20431 examples [00:19, 1051.35 examples/s]20540 examples [00:19, 1060.16 examples/s]20647 examples [00:19, 1056.58 examples/s]20755 examples [00:19, 1060.70 examples/s]20862 examples [00:19, 1059.53 examples/s]20970 examples [00:19, 1062.70 examples/s]21077 examples [00:19, 1021.92 examples/s]21184 examples [00:19, 1035.87 examples/s]21291 examples [00:19, 1043.80 examples/s]21396 examples [00:20, 1044.87 examples/s]21501 examples [00:20, 1040.59 examples/s]21606 examples [00:20, 1021.01 examples/s]21712 examples [00:20, 1029.95 examples/s]21819 examples [00:20, 1041.57 examples/s]21928 examples [00:20, 1054.22 examples/s]22036 examples [00:20, 1061.80 examples/s]22144 examples [00:20, 1067.08 examples/s]22251 examples [00:20, 1064.69 examples/s]22360 examples [00:20, 1070.39 examples/s]22469 examples [00:21, 1073.88 examples/s]22578 examples [00:21, 1076.68 examples/s]22686 examples [00:21, 1074.10 examples/s]22794 examples [00:21, 1072.35 examples/s]22902 examples [00:21, 1074.47 examples/s]23011 examples [00:21, 1077.62 examples/s]23120 examples [00:21, 1079.56 examples/s]23230 examples [00:21, 1083.67 examples/s]23339 examples [00:21, 1081.75 examples/s]23448 examples [00:21, 1082.08 examples/s]23557 examples [00:22, 1080.57 examples/s]23666 examples [00:22, 1068.44 examples/s]23775 examples [00:22, 1073.52 examples/s]23883 examples [00:22, 1066.03 examples/s]23991 examples [00:22, 1069.52 examples/s]24100 examples [00:22, 1074.93 examples/s]24210 examples [00:22, 1080.26 examples/s]24320 examples [00:22, 1083.30 examples/s]24429 examples [00:22, 1076.34 examples/s]24537 examples [00:23, 1069.87 examples/s]24646 examples [00:23, 1074.75 examples/s]24756 examples [00:23, 1079.42 examples/s]24864 examples [00:23, 1078.75 examples/s]24973 examples [00:23, 1081.43 examples/s]25082 examples [00:23, 1081.15 examples/s]25191 examples [00:23, 1081.70 examples/s]25300 examples [00:23, 1076.65 examples/s]25408 examples [00:23, 1062.58 examples/s]25517 examples [00:23, 1067.81 examples/s]25624 examples [00:24, 1050.33 examples/s]25734 examples [00:24, 1062.57 examples/s]25844 examples [00:24, 1073.13 examples/s]25952 examples [00:24, 1062.98 examples/s]26059 examples [00:24, 1041.06 examples/s]26167 examples [00:24, 1050.29 examples/s]26273 examples [00:24, 1050.81 examples/s]26382 examples [00:24, 1061.25 examples/s]26493 examples [00:24, 1072.98 examples/s]26601 examples [00:24, 1065.88 examples/s]26708 examples [00:25, 1063.51 examples/s]26815 examples [00:25, 1064.29 examples/s]26922 examples [00:25, 1040.65 examples/s]27027 examples [00:25, 1041.47 examples/s]27132 examples [00:25, 1041.28 examples/s]27240 examples [00:25, 1050.53 examples/s]27346 examples [00:25, 1051.50 examples/s]27453 examples [00:25, 1056.81 examples/s]27559 examples [00:25, 1051.34 examples/s]27665 examples [00:25, 1052.02 examples/s]27771 examples [00:26, 1050.21 examples/s]27877 examples [00:26, 1044.47 examples/s]27985 examples [00:26, 1054.09 examples/s]28091 examples [00:26, 1051.91 examples/s]28197 examples [00:26, 1053.00 examples/s]28304 examples [00:26, 1056.95 examples/s]28410 examples [00:26, 1056.21 examples/s]28519 examples [00:26, 1064.69 examples/s]28626 examples [00:26, 1039.10 examples/s]28731 examples [00:26, 1009.59 examples/s]28836 examples [00:27, 1018.74 examples/s]28943 examples [00:27, 1031.77 examples/s]29052 examples [00:27, 1047.54 examples/s]29160 examples [00:27, 1056.78 examples/s]29266 examples [00:27, 1054.23 examples/s]29374 examples [00:27, 1061.10 examples/s]29481 examples [00:27, 1059.65 examples/s]29589 examples [00:27, 1065.31 examples/s]29698 examples [00:27, 1070.59 examples/s]29806 examples [00:27, 1060.15 examples/s]29913 examples [00:28, 1056.00 examples/s]30019 examples [00:28, 1006.12 examples/s]30124 examples [00:28, 1018.22 examples/s]30231 examples [00:28, 1031.12 examples/s]30335 examples [00:28, 1032.75 examples/s]30439 examples [00:28, 1021.85 examples/s]30542 examples [00:28, 1013.80 examples/s]30649 examples [00:28, 1029.09 examples/s]30757 examples [00:28, 1043.20 examples/s]30862 examples [00:29, 1014.36 examples/s]30969 examples [00:29, 1029.27 examples/s]31075 examples [00:29, 1037.51 examples/s]31182 examples [00:29, 1044.86 examples/s]31289 examples [00:29, 1051.24 examples/s]31395 examples [00:29, 1043.08 examples/s]31504 examples [00:29, 1055.37 examples/s]31611 examples [00:29, 1059.17 examples/s]31720 examples [00:29, 1067.12 examples/s]31827 examples [00:29, 1066.70 examples/s]31935 examples [00:30, 1068.65 examples/s]32042 examples [00:30, 1062.10 examples/s]32149 examples [00:30, 1015.75 examples/s]32254 examples [00:30, 1024.32 examples/s]32357 examples [00:30, 1016.41 examples/s]32459 examples [00:30, 1001.39 examples/s]32566 examples [00:30, 1020.33 examples/s]32672 examples [00:30, 1030.47 examples/s]32781 examples [00:30, 1046.59 examples/s]32886 examples [00:30, 1036.88 examples/s]32995 examples [00:31, 1050.38 examples/s]33104 examples [00:31, 1060.36 examples/s]33211 examples [00:31, 1061.13 examples/s]33321 examples [00:31, 1069.78 examples/s]33429 examples [00:31, 1071.75 examples/s]33538 examples [00:31, 1075.90 examples/s]33648 examples [00:31, 1080.46 examples/s]33758 examples [00:31, 1086.07 examples/s]33867 examples [00:31, 1068.14 examples/s]33974 examples [00:31, 1059.43 examples/s]34081 examples [00:32, 1060.60 examples/s]34188 examples [00:32, 1060.37 examples/s]34296 examples [00:32, 1064.35 examples/s]34403 examples [00:32, 1057.60 examples/s]34512 examples [00:32, 1065.86 examples/s]34621 examples [00:32, 1069.73 examples/s]34730 examples [00:32, 1074.72 examples/s]34838 examples [00:32, 1076.01 examples/s]34948 examples [00:32, 1081.04 examples/s]35057 examples [00:32, 1082.95 examples/s]35166 examples [00:33, 1082.28 examples/s]35275 examples [00:33, 1081.72 examples/s]35384 examples [00:33, 1055.45 examples/s]35490 examples [00:33, 1055.48 examples/s]35596 examples [00:33, 1044.70 examples/s]35703 examples [00:33, 1051.07 examples/s]35812 examples [00:33, 1060.78 examples/s]35921 examples [00:33, 1057.45 examples/s]36027 examples [00:33, 1056.64 examples/s]36133 examples [00:34, 1042.60 examples/s]36242 examples [00:34, 1054.11 examples/s]36351 examples [00:34, 1062.33 examples/s]36459 examples [00:34, 1066.18 examples/s]36569 examples [00:34, 1074.12 examples/s]36677 examples [00:34, 1073.92 examples/s]36786 examples [00:34, 1076.48 examples/s]36896 examples [00:34, 1080.99 examples/s]37007 examples [00:34, 1087.40 examples/s]37116 examples [00:34, 1078.88 examples/s]37224 examples [00:35, 1078.92 examples/s]37332 examples [00:35, 1078.75 examples/s]37440 examples [00:35, 1073.15 examples/s]37548 examples [00:35, 1039.82 examples/s]37655 examples [00:35, 1047.83 examples/s]37763 examples [00:35, 1055.40 examples/s]37871 examples [00:35, 1061.09 examples/s]37980 examples [00:35, 1068.93 examples/s]38087 examples [00:35, 1068.98 examples/s]38195 examples [00:35, 1072.15 examples/s]38303 examples [00:36, 1072.38 examples/s]38411 examples [00:36, 1032.42 examples/s]38519 examples [00:36, 1045.27 examples/s]38628 examples [00:36, 1057.98 examples/s]38737 examples [00:36, 1066.87 examples/s]38846 examples [00:36, 1071.30 examples/s]38954 examples [00:36, 1061.26 examples/s]39063 examples [00:36, 1068.29 examples/s]39170 examples [00:36, 1063.21 examples/s]39278 examples [00:36, 1065.92 examples/s]39385 examples [00:37, 1066.92 examples/s]39492 examples [00:37, 1065.66 examples/s]39599 examples [00:37, 1062.14 examples/s]39707 examples [00:37, 1064.83 examples/s]39816 examples [00:37, 1069.26 examples/s]39924 examples [00:37, 1070.28 examples/s]40032 examples [00:37, 1015.47 examples/s]40138 examples [00:37, 1026.21 examples/s]40244 examples [00:37, 1035.53 examples/s]40352 examples [00:37, 1047.05 examples/s]40460 examples [00:38, 1055.52 examples/s]40566 examples [00:38, 1054.33 examples/s]40672 examples [00:38, 1048.81 examples/s]40777 examples [00:38, 1043.12 examples/s]40887 examples [00:38, 1056.57 examples/s]40994 examples [00:38, 1060.51 examples/s]41103 examples [00:38, 1068.05 examples/s]41213 examples [00:38, 1075.26 examples/s]41323 examples [00:38, 1080.22 examples/s]41432 examples [00:38, 1082.16 examples/s]41541 examples [00:39, 1064.89 examples/s]41651 examples [00:39, 1072.59 examples/s]41759 examples [00:39, 1048.30 examples/s]41868 examples [00:39, 1058.13 examples/s]41975 examples [00:39, 1059.44 examples/s]42082 examples [00:39, 1056.40 examples/s]42190 examples [00:39, 1061.39 examples/s]42298 examples [00:39, 1066.40 examples/s]42406 examples [00:39, 1070.12 examples/s]42515 examples [00:40, 1074.38 examples/s]42623 examples [00:40, 1071.72 examples/s]42731 examples [00:40, 1074.13 examples/s]42839 examples [00:40, 1064.06 examples/s]42949 examples [00:40, 1072.98 examples/s]43058 examples [00:40, 1075.64 examples/s]43166 examples [00:40, 1076.00 examples/s]43275 examples [00:40, 1078.09 examples/s]43384 examples [00:40, 1080.25 examples/s]43493 examples [00:40, 1081.81 examples/s]43602 examples [00:41, 1083.08 examples/s]43711 examples [00:41, 1071.92 examples/s]43820 examples [00:41, 1075.30 examples/s]43928 examples [00:41, 1073.42 examples/s]44036 examples [00:41, 1021.21 examples/s]44144 examples [00:41, 1036.49 examples/s]44249 examples [00:41, 1034.99 examples/s]44353 examples [00:41, 1035.46 examples/s]44462 examples [00:41, 1048.85 examples/s]44570 examples [00:41, 1055.70 examples/s]44679 examples [00:42, 1063.55 examples/s]44786 examples [00:42, 1057.16 examples/s]44895 examples [00:42, 1065.31 examples/s]45002 examples [00:42, 1064.72 examples/s]45111 examples [00:42, 1072.02 examples/s]45220 examples [00:42, 1076.60 examples/s]45328 examples [00:42, 1058.06 examples/s]45434 examples [00:42, 1054.65 examples/s]45542 examples [00:42, 1059.54 examples/s]45651 examples [00:42, 1065.96 examples/s]45760 examples [00:43, 1072.92 examples/s]45868 examples [00:43, 1069.35 examples/s]45976 examples [00:43, 1070.49 examples/s]46084 examples [00:43, 1059.82 examples/s]46193 examples [00:43, 1068.66 examples/s]46303 examples [00:43, 1076.10 examples/s]46411 examples [00:43, 1069.35 examples/s]46520 examples [00:43, 1072.78 examples/s]46628 examples [00:43, 1066.97 examples/s]46735 examples [00:43, 1066.21 examples/s]46842 examples [00:44, 1056.05 examples/s]46948 examples [00:44, 1051.09 examples/s]47054 examples [00:44, 1049.44 examples/s]47159 examples [00:44, 1045.78 examples/s]47265 examples [00:44, 1048.25 examples/s]47373 examples [00:44, 1053.61 examples/s]47479 examples [00:44, 1050.47 examples/s]47587 examples [00:44, 1058.35 examples/s]47695 examples [00:44, 1063.75 examples/s]47802 examples [00:44, 1063.65 examples/s]47911 examples [00:45, 1070.36 examples/s]48019 examples [00:45, 1070.25 examples/s]48127 examples [00:45, 1072.28 examples/s]48235 examples [00:45, 1065.58 examples/s]48343 examples [00:45, 1067.47 examples/s]48451 examples [00:45, 1070.75 examples/s]48559 examples [00:45, 1069.97 examples/s]48667 examples [00:45, 1069.52 examples/s]48775 examples [00:45, 1070.20 examples/s]48884 examples [00:46, 1075.52 examples/s]48992 examples [00:46, 1075.43 examples/s]49100 examples [00:46, 1070.78 examples/s]49208 examples [00:46, 1071.60 examples/s]49317 examples [00:46, 1075.46 examples/s]49425 examples [00:46, 1073.78 examples/s]49534 examples [00:46, 1076.85 examples/s]49642 examples [00:46, 1075.66 examples/s]49752 examples [00:46, 1080.97 examples/s]49861 examples [00:46, 1080.28 examples/s]49970 examples [00:47, 1079.63 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6245/50000 [00:00<00:00, 62443.96 examples/s] 37%|███▋      | 18607/50000 [00:00<00:00, 73330.42 examples/s] 63%|██████▎   | 31545/50000 [00:00<00:00, 84283.21 examples/s] 89%|████████▉ | 44595/50000 [00:00<00:00, 94300.68 examples/s]                                                               0 examples [00:00, ? examples/s]91 examples [00:00, 901.46 examples/s]200 examples [00:00, 950.41 examples/s]309 examples [00:00, 986.52 examples/s]420 examples [00:00, 1018.32 examples/s]530 examples [00:00, 1039.56 examples/s]641 examples [00:00, 1058.48 examples/s]750 examples [00:00, 1067.57 examples/s]857 examples [00:00, 1065.83 examples/s]967 examples [00:00, 1075.60 examples/s]1076 examples [00:01, 1078.58 examples/s]1186 examples [00:01, 1084.22 examples/s]1296 examples [00:01, 1088.17 examples/s]1405 examples [00:01, 1087.09 examples/s]1516 examples [00:01, 1092.58 examples/s]1625 examples [00:01, 1086.52 examples/s]1734 examples [00:01, 1083.15 examples/s]1844 examples [00:01, 1086.66 examples/s]1953 examples [00:01, 1037.39 examples/s]2064 examples [00:01, 1056.01 examples/s]2170 examples [00:02, 1049.96 examples/s]2279 examples [00:02, 1060.82 examples/s]2388 examples [00:02, 1066.21 examples/s]2496 examples [00:02, 1067.99 examples/s]2603 examples [00:02, 1067.93 examples/s]2711 examples [00:02, 1071.12 examples/s]2821 examples [00:02, 1078.35 examples/s]2930 examples [00:02, 1079.35 examples/s]3038 examples [00:02, 1069.59 examples/s]3146 examples [00:02, 1068.06 examples/s]3253 examples [00:03, 1065.15 examples/s]3363 examples [00:03, 1074.40 examples/s]3473 examples [00:03, 1080.12 examples/s]3582 examples [00:03, 1082.56 examples/s]3693 examples [00:03, 1088.58 examples/s]3803 examples [00:03, 1091.38 examples/s]3913 examples [00:03, 1088.68 examples/s]4024 examples [00:03, 1093.06 examples/s]4134 examples [00:03, 1088.37 examples/s]4243 examples [00:03, 1086.94 examples/s]4352 examples [00:04, 1077.14 examples/s]4461 examples [00:04, 1080.93 examples/s]4571 examples [00:04, 1086.20 examples/s]4680 examples [00:04, 1079.34 examples/s]4791 examples [00:04, 1087.19 examples/s]4901 examples [00:04, 1090.35 examples/s]5011 examples [00:04, 1091.07 examples/s]5121 examples [00:04, 1086.32 examples/s]5230 examples [00:04, 1083.31 examples/s]5340 examples [00:04, 1087.60 examples/s]5449 examples [00:05, 861.06 examples/s] 5558 examples [00:05, 918.42 examples/s]5667 examples [00:05, 962.34 examples/s]5776 examples [00:05, 996.20 examples/s]5880 examples [00:05, 1006.33 examples/s]5984 examples [00:05, 1016.03 examples/s]6092 examples [00:05, 1031.91 examples/s]6199 examples [00:05, 1041.91 examples/s]6308 examples [00:05, 1054.23 examples/s]6415 examples [00:06, 1056.59 examples/s]6522 examples [00:06, 1059.36 examples/s]6629 examples [00:06, 1062.02 examples/s]6736 examples [00:06, 1048.24 examples/s]6846 examples [00:06, 1061.52 examples/s]6955 examples [00:06, 1067.30 examples/s]7064 examples [00:06, 1073.13 examples/s]7173 examples [00:06, 1075.68 examples/s]7281 examples [00:06, 1074.15 examples/s]7391 examples [00:06, 1080.18 examples/s]7500 examples [00:07, 1080.39 examples/s]7609 examples [00:07, 1078.47 examples/s]7717 examples [00:07, 1067.63 examples/s]7825 examples [00:07, 1070.53 examples/s]7934 examples [00:07, 1073.75 examples/s]8044 examples [00:07, 1078.87 examples/s]8153 examples [00:07, 1082.05 examples/s]8263 examples [00:07, 1084.18 examples/s]8372 examples [00:07, 1081.37 examples/s]8481 examples [00:07, 1082.27 examples/s]8590 examples [00:08, 1083.76 examples/s]8699 examples [00:08, 1082.90 examples/s]8808 examples [00:08, 1080.39 examples/s]8917 examples [00:08, 1079.25 examples/s]9027 examples [00:08, 1084.03 examples/s]9137 examples [00:08, 1087.57 examples/s]9247 examples [00:08, 1089.25 examples/s]9356 examples [00:08, 1086.69 examples/s]9465 examples [00:08, 1073.00 examples/s]9575 examples [00:08, 1080.25 examples/s]9684 examples [00:09, 1070.39 examples/s]9793 examples [00:09, 1075.90 examples/s]9902 examples [00:09, 1077.34 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete7PL8AY/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete7PL8AY/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb3865bc488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb3865bc488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb3865bc488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb3846229e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb30c854630>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb3865bc488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb3865bc488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb3865bc488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb30c854fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb384760128>), {}) 

  




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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.23 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.23 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.23 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.56 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.56 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.17 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.17 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.65 file/s]2020-09-01 12:08:47.215324: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-01 12:08:47.219630: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-09-01 12:08:47.219814: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5577a96f5210 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-01 12:08:47.219829: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'mnist2', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy', 'train', 'test'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 459765.59it/s] 95%|█████████▌| 9461760/9912422 [00:00<00:00, 655434.30it/s]9920512it [00:00, 46491740.87it/s]                           
0it [00:00, ?it/s]32768it [00:00, 623925.59it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1001389.06it/s]1654784it [00:00, 11895316.11it/s]                           
0it [00:00, ?it/s]8192it [00:00, 178629.48it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

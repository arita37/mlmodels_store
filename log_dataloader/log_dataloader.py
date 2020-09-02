
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f09cc7a9620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f09cc7a9620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f0a374d2488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f0a374d2488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f0a566f1ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f0a566f1ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f09e4808488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f09e4808488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f09e4808488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 455288.84it/s] 38%|███▊      | 3743744/9912422 [00:00<00:09, 646995.28it/s]9920512it [00:00, 33831481.16it/s]                           
0it [00:00, ?it/s]32768it [00:00, 629524.07it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1040806.31it/s]1654784it [00:00, 11997737.50it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 19065.65it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f09cba60ef0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f09e29b18d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f09e4808158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f09e4808158> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f09e4808158> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:21,  1.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:21,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:20,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:20,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:19,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:19,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:18,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:18,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:17,  1.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:54,  2.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:54,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:54,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:53,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:53,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:53,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:52,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:52,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:52,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:51,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:51,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:51,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  3.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:35,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:35,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:34,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:34,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:34,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:33,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:33,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:33,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:23,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:23,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:22,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:22,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:22,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:16,  7.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:15,  7.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:15,  7.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:15,  7.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:14,  7.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:14,  7.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:10, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:09, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:09, 10.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 14.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:06, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:06, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 19.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 25.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 25.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 25.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 25.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 25.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 25.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 33.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 33.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 33.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 33.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 33.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 33.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 33.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 33.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 41.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 41.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 41.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 41.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 41.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 41.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 41.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 49.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 49.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 49.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 49.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 49.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 49.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 49.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 49.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 49.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 55.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 61.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 70.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 70.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 70.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 70.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 70.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 70.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 70.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 70.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 70.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 70.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.27s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.27s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.27s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.02 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.39s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.27s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 74.02 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.39s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.39s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.86 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.40s/ url]
0 examples [00:00, ? examples/s]2020-09-02 06:07:36.804588: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-02 06:07:36.811925: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-09-02 06:07:36.812090: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55823de0c9e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-02 06:07:36.812108: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
35 examples [00:00, 348.07 examples/s]146 examples [00:00, 437.92 examples/s]255 examples [00:00, 533.52 examples/s]361 examples [00:00, 625.92 examples/s]458 examples [00:00, 699.33 examples/s]554 examples [00:00, 760.26 examples/s]653 examples [00:00, 816.32 examples/s]749 examples [00:00, 853.89 examples/s]852 examples [00:00, 898.31 examples/s]959 examples [00:01, 942.03 examples/s]1068 examples [00:01, 981.78 examples/s]1174 examples [00:01, 1002.71 examples/s]1277 examples [00:01, 1004.76 examples/s]1381 examples [00:01, 1013.43 examples/s]1484 examples [00:01, 997.49 examples/s] 1587 examples [00:01, 1006.32 examples/s]1689 examples [00:01, 993.81 examples/s] 1793 examples [00:01, 1005.63 examples/s]1902 examples [00:01, 1027.33 examples/s]2006 examples [00:02, 1013.29 examples/s]2108 examples [00:02, 998.35 examples/s] 2216 examples [00:02, 1020.52 examples/s]2321 examples [00:02, 1027.38 examples/s]2430 examples [00:02, 1044.39 examples/s]2535 examples [00:02, 1026.13 examples/s]2639 examples [00:02, 1028.22 examples/s]2742 examples [00:02, 1022.84 examples/s]2845 examples [00:02, 1006.39 examples/s]2946 examples [00:02, 950.75 examples/s] 3049 examples [00:03, 972.14 examples/s]3147 examples [00:03, 972.63 examples/s]3247 examples [00:03, 980.12 examples/s]3350 examples [00:03, 993.48 examples/s]3450 examples [00:03, 992.56 examples/s]3550 examples [00:03, 988.26 examples/s]3649 examples [00:03, 924.74 examples/s]3756 examples [00:03, 963.30 examples/s]3864 examples [00:03, 994.32 examples/s]3966 examples [00:04, 1000.64 examples/s]4067 examples [00:04, 961.94 examples/s] 4172 examples [00:04, 985.00 examples/s]4279 examples [00:04, 1007.47 examples/s]4386 examples [00:04, 1024.40 examples/s]4489 examples [00:04, 1019.39 examples/s]4592 examples [00:04, 1009.88 examples/s]4700 examples [00:04, 1028.25 examples/s]4804 examples [00:04, 1000.88 examples/s]4905 examples [00:04, 997.33 examples/s] 5006 examples [00:05, 998.81 examples/s]5107 examples [00:05, 982.84 examples/s]5207 examples [00:05, 987.37 examples/s]5310 examples [00:05, 999.31 examples/s]5411 examples [00:05, 985.70 examples/s]5510 examples [00:05, 943.43 examples/s]5605 examples [00:05, 937.15 examples/s]5707 examples [00:05, 960.54 examples/s]5814 examples [00:05, 989.73 examples/s]5917 examples [00:05, 999.75 examples/s]6018 examples [00:06, 977.25 examples/s]6127 examples [00:06, 1007.22 examples/s]6233 examples [00:06, 1022.07 examples/s]6336 examples [00:06, 1024.35 examples/s]6447 examples [00:06, 1046.73 examples/s]6557 examples [00:06, 1062.02 examples/s]6668 examples [00:06, 1073.75 examples/s]6780 examples [00:06, 1085.75 examples/s]6889 examples [00:06, 1075.16 examples/s]7000 examples [00:06, 1083.44 examples/s]7109 examples [00:07, 1084.97 examples/s]7218 examples [00:07, 1053.60 examples/s]7330 examples [00:07, 1071.42 examples/s]7442 examples [00:07, 1083.90 examples/s]7552 examples [00:07, 1087.79 examples/s]7663 examples [00:07, 1092.85 examples/s]7774 examples [00:07, 1096.53 examples/s]7885 examples [00:07, 1099.36 examples/s]7996 examples [00:07, 1102.51 examples/s]8107 examples [00:07, 1104.69 examples/s]8219 examples [00:08, 1107.72 examples/s]8330 examples [00:08, 1105.41 examples/s]8441 examples [00:08, 1101.60 examples/s]8552 examples [00:08, 1102.54 examples/s]8664 examples [00:08, 1105.36 examples/s]8775 examples [00:08, 1103.65 examples/s]8886 examples [00:08, 1102.08 examples/s]8998 examples [00:08, 1104.57 examples/s]9109 examples [00:08, 1104.65 examples/s]9221 examples [00:09, 1108.46 examples/s]9332 examples [00:09, 1107.41 examples/s]9443 examples [00:09, 1101.58 examples/s]9554 examples [00:09, 1103.00 examples/s]9666 examples [00:09, 1105.95 examples/s]9778 examples [00:09, 1109.13 examples/s]9890 examples [00:09, 1111.03 examples/s]10002 examples [00:09, 1039.09 examples/s]10111 examples [00:09, 1052.05 examples/s]10220 examples [00:09, 1061.54 examples/s]10330 examples [00:10, 1071.92 examples/s]10440 examples [00:10, 1078.45 examples/s]10549 examples [00:10, 1075.88 examples/s]10659 examples [00:10, 1081.59 examples/s]10769 examples [00:10, 1086.07 examples/s]10880 examples [00:10, 1090.69 examples/s]10991 examples [00:10, 1094.92 examples/s]11101 examples [00:10, 1086.37 examples/s]11210 examples [00:10, 1074.13 examples/s]11320 examples [00:10, 1081.13 examples/s]11432 examples [00:11, 1090.63 examples/s]11544 examples [00:11, 1096.84 examples/s]11654 examples [00:11, 1094.18 examples/s]11764 examples [00:11, 1095.20 examples/s]11875 examples [00:11, 1097.45 examples/s]11985 examples [00:11, 1097.49 examples/s]12095 examples [00:11, 1095.56 examples/s]12205 examples [00:11, 1090.67 examples/s]12315 examples [00:11, 1090.21 examples/s]12425 examples [00:11, 1091.98 examples/s]12535 examples [00:12, 1091.37 examples/s]12646 examples [00:12, 1094.01 examples/s]12756 examples [00:12, 1090.72 examples/s]12866 examples [00:12, 1091.00 examples/s]12977 examples [00:12, 1094.80 examples/s]13087 examples [00:12, 1092.90 examples/s]13198 examples [00:12, 1094.49 examples/s]13308 examples [00:12, 1093.34 examples/s]13419 examples [00:12, 1096.68 examples/s]13529 examples [00:12, 1096.39 examples/s]13640 examples [00:13, 1098.56 examples/s]13750 examples [00:13, 1095.88 examples/s]13860 examples [00:13, 1093.42 examples/s]13971 examples [00:13, 1095.69 examples/s]14081 examples [00:13, 1085.03 examples/s]14190 examples [00:13, 1084.29 examples/s]14301 examples [00:13, 1071.49 examples/s]14409 examples [00:13, 1073.79 examples/s]14520 examples [00:13, 1083.75 examples/s]14631 examples [00:13, 1091.28 examples/s]14741 examples [00:14, 1082.43 examples/s]14851 examples [00:14, 1086.28 examples/s]14961 examples [00:14, 1088.25 examples/s]15072 examples [00:14, 1093.23 examples/s]15183 examples [00:14, 1095.80 examples/s]15293 examples [00:14, 1095.72 examples/s]15404 examples [00:14, 1098.42 examples/s]15514 examples [00:14, 1097.08 examples/s]15624 examples [00:14, 1095.91 examples/s]15735 examples [00:14, 1098.65 examples/s]15847 examples [00:15, 1103.40 examples/s]15958 examples [00:15, 1099.35 examples/s]16069 examples [00:15, 1102.11 examples/s]16180 examples [00:15, 1099.16 examples/s]16291 examples [00:15, 1100.06 examples/s]16402 examples [00:15, 1102.05 examples/s]16513 examples [00:15, 1101.95 examples/s]16624 examples [00:15, 1104.15 examples/s]16735 examples [00:15, 1103.39 examples/s]16846 examples [00:15, 1102.67 examples/s]16957 examples [00:16, 1095.49 examples/s]17069 examples [00:16, 1098.89 examples/s]17179 examples [00:16, 1088.96 examples/s]17290 examples [00:16, 1093.72 examples/s]17401 examples [00:16, 1097.25 examples/s]17513 examples [00:16, 1102.16 examples/s]17625 examples [00:16, 1104.75 examples/s]17736 examples [00:16, 1104.81 examples/s]17847 examples [00:16, 1080.48 examples/s]17959 examples [00:17, 1089.31 examples/s]18069 examples [00:17, 1092.43 examples/s]18179 examples [00:17, 1092.03 examples/s]18289 examples [00:17, 1091.42 examples/s]18401 examples [00:17, 1097.14 examples/s]18512 examples [00:17, 1098.56 examples/s]18622 examples [00:17, 1091.87 examples/s]18732 examples [00:17, 1091.42 examples/s]18842 examples [00:17, 1093.44 examples/s]18953 examples [00:17, 1096.60 examples/s]19064 examples [00:18, 1097.76 examples/s]19175 examples [00:18, 1100.74 examples/s]19286 examples [00:18, 1098.41 examples/s]19396 examples [00:18, 1098.69 examples/s]19508 examples [00:18, 1102.37 examples/s]19620 examples [00:18, 1106.88 examples/s]19731 examples [00:18, 1106.49 examples/s]19842 examples [00:18, 1103.68 examples/s]19953 examples [00:18, 1104.60 examples/s]20064 examples [00:18, 1048.27 examples/s]20173 examples [00:19, 1058.65 examples/s]20280 examples [00:19, 1039.76 examples/s]20385 examples [00:19, 1035.68 examples/s]20494 examples [00:19, 1048.50 examples/s]20600 examples [00:19, 990.02 examples/s] 20700 examples [00:19, 970.82 examples/s]20805 examples [00:19, 991.33 examples/s]20905 examples [00:19, 990.22 examples/s]21008 examples [00:19, 999.36 examples/s]21113 examples [00:19, 1012.17 examples/s]21215 examples [00:20, 996.21 examples/s] 21315 examples [00:20, 942.48 examples/s]21418 examples [00:20, 966.95 examples/s]21516 examples [00:20, 970.21 examples/s]21614 examples [00:20, 964.74 examples/s]21711 examples [00:20, 950.38 examples/s]21816 examples [00:20, 974.94 examples/s]21914 examples [00:20, 972.47 examples/s]22012 examples [00:20, 954.61 examples/s]22113 examples [00:21, 967.77 examples/s]22220 examples [00:21, 994.65 examples/s]22323 examples [00:21, 1003.72 examples/s]22427 examples [00:21, 1014.07 examples/s]22538 examples [00:21, 1039.50 examples/s]22647 examples [00:21, 1052.62 examples/s]22758 examples [00:21, 1067.30 examples/s]22868 examples [00:21, 1076.05 examples/s]22978 examples [00:21, 1080.25 examples/s]23088 examples [00:21, 1084.80 examples/s]23198 examples [00:22, 1087.00 examples/s]23309 examples [00:22, 1093.02 examples/s]23420 examples [00:22, 1094.91 examples/s]23530 examples [00:22, 1095.63 examples/s]23641 examples [00:22, 1097.82 examples/s]23752 examples [00:22, 1100.22 examples/s]23863 examples [00:22, 1099.19 examples/s]23973 examples [00:22, 1094.75 examples/s]24083 examples [00:22, 1089.76 examples/s]24192 examples [00:22, 1073.87 examples/s]24301 examples [00:23, 1078.24 examples/s]24412 examples [00:23, 1085.94 examples/s]24521 examples [00:23, 1068.44 examples/s]24628 examples [00:23, 999.26 examples/s] 24729 examples [00:23, 980.38 examples/s]24833 examples [00:23, 997.50 examples/s]24934 examples [00:23, 987.71 examples/s]25034 examples [00:23, 981.40 examples/s]25140 examples [00:23, 1002.85 examples/s]25246 examples [00:23, 1017.75 examples/s]25354 examples [00:24, 1033.26 examples/s]25463 examples [00:24, 1048.69 examples/s]25569 examples [00:24, 1045.34 examples/s]25674 examples [00:24, 1029.81 examples/s]25785 examples [00:24, 1051.17 examples/s]25891 examples [00:24, 1030.68 examples/s]25995 examples [00:24, 1025.10 examples/s]26105 examples [00:24, 1044.81 examples/s]26213 examples [00:24, 1055.11 examples/s]26321 examples [00:25, 1059.71 examples/s]26431 examples [00:25, 1071.40 examples/s]26541 examples [00:25, 1077.10 examples/s]26651 examples [00:25, 1083.01 examples/s]26760 examples [00:25, 1081.09 examples/s]26869 examples [00:25, 1078.32 examples/s]26979 examples [00:25, 1080.37 examples/s]27088 examples [00:25, 1081.31 examples/s]27198 examples [00:25, 1086.72 examples/s]27309 examples [00:25, 1093.29 examples/s]27419 examples [00:26, 1089.44 examples/s]27528 examples [00:26, 1078.52 examples/s]27636 examples [00:26, 1064.51 examples/s]27743 examples [00:26, 1035.61 examples/s]27848 examples [00:26, 1039.48 examples/s]27953 examples [00:26, 1021.96 examples/s]28056 examples [00:26, 999.36 examples/s] 28164 examples [00:26, 1021.85 examples/s]28267 examples [00:26, 1003.45 examples/s]28368 examples [00:26, 993.26 examples/s] 28473 examples [00:27, 1009.52 examples/s]28580 examples [00:27, 1026.37 examples/s]28687 examples [00:27, 1037.55 examples/s]28794 examples [00:27, 1045.33 examples/s]28899 examples [00:27, 986.30 examples/s] 29000 examples [00:27, 993.02 examples/s]29100 examples [00:27, 977.30 examples/s]29203 examples [00:27, 991.46 examples/s]29306 examples [00:27, 996.98 examples/s]29406 examples [00:28, 948.43 examples/s]29515 examples [00:28, 985.89 examples/s]29625 examples [00:28, 1016.12 examples/s]29736 examples [00:28, 1040.60 examples/s]29846 examples [00:28, 1055.81 examples/s]29956 examples [00:28, 1068.27 examples/s]30064 examples [00:28, 1017.24 examples/s]30173 examples [00:28, 1035.44 examples/s]30279 examples [00:28, 1040.91 examples/s]30384 examples [00:28, 1041.67 examples/s]30489 examples [00:29, 997.82 examples/s] 30591 examples [00:29, 1002.62 examples/s]30694 examples [00:29, 1009.91 examples/s]30796 examples [00:29, 943.45 examples/s] 30892 examples [00:29, 924.79 examples/s]30990 examples [00:29, 940.05 examples/s]31095 examples [00:29, 968.93 examples/s]31200 examples [00:29, 965.39 examples/s]31300 examples [00:29, 973.73 examples/s]31398 examples [00:30, 961.45 examples/s]31495 examples [00:30, 954.53 examples/s]31600 examples [00:30, 981.03 examples/s]31707 examples [00:30, 1004.54 examples/s]31808 examples [00:30, 968.34 examples/s] 31913 examples [00:30, 988.95 examples/s]32015 examples [00:30, 995.94 examples/s]32118 examples [00:30, 1005.27 examples/s]32225 examples [00:30, 1022.08 examples/s]32329 examples [00:30, 1026.68 examples/s]32432 examples [00:31, 1016.44 examples/s]32535 examples [00:31, 1020.23 examples/s]32638 examples [00:31, 1022.14 examples/s]32743 examples [00:31, 1029.55 examples/s]32847 examples [00:31, 1023.33 examples/s]32950 examples [00:31, 1015.45 examples/s]33052 examples [00:31, 1001.54 examples/s]33155 examples [00:31, 1009.86 examples/s]33257 examples [00:31, 1008.33 examples/s]33359 examples [00:31, 1010.81 examples/s]33465 examples [00:32, 1015.24 examples/s]33571 examples [00:32, 1025.59 examples/s]33678 examples [00:32, 1036.09 examples/s]33784 examples [00:32, 1039.78 examples/s]33889 examples [00:32, 1042.59 examples/s]33994 examples [00:32, 1015.23 examples/s]34096 examples [00:32, 1015.79 examples/s]34198 examples [00:32, 1014.83 examples/s]34301 examples [00:32, 1017.00 examples/s]34405 examples [00:32, 1021.21 examples/s]34508 examples [00:33, 1019.64 examples/s]34610 examples [00:33, 986.54 examples/s] 34714 examples [00:33, 999.40 examples/s]34819 examples [00:33, 1013.39 examples/s]34921 examples [00:33, 975.31 examples/s] 35019 examples [00:33, 972.94 examples/s]35125 examples [00:33, 995.08 examples/s]35225 examples [00:33, 982.06 examples/s]35324 examples [00:33, 971.37 examples/s]35431 examples [00:34, 996.95 examples/s]35539 examples [00:34, 1018.08 examples/s]35649 examples [00:34, 1039.86 examples/s]35754 examples [00:34, 1015.87 examples/s]35856 examples [00:34, 952.75 examples/s] 35953 examples [00:34, 924.80 examples/s]36047 examples [00:34, 889.79 examples/s]36153 examples [00:34, 932.67 examples/s]36253 examples [00:34, 949.85 examples/s]36357 examples [00:34, 974.94 examples/s]36456 examples [00:35, 969.73 examples/s]36563 examples [00:35, 997.24 examples/s]36665 examples [00:35, 1001.65 examples/s]36766 examples [00:35, 993.75 examples/s] 36866 examples [00:35, 960.40 examples/s]36963 examples [00:35, 954.59 examples/s]37059 examples [00:35, 941.73 examples/s]37155 examples [00:35, 946.75 examples/s]37253 examples [00:35, 956.22 examples/s]37349 examples [00:35, 952.14 examples/s]37447 examples [00:36, 958.12 examples/s]37554 examples [00:36, 987.21 examples/s]37659 examples [00:36, 1003.52 examples/s]37760 examples [00:36, 1002.03 examples/s]37866 examples [00:36, 1018.64 examples/s]37970 examples [00:36, 1022.79 examples/s]38079 examples [00:36, 1040.96 examples/s]38184 examples [00:36, 1037.96 examples/s]38289 examples [00:36, 1038.65 examples/s]38393 examples [00:37, 1018.33 examples/s]38497 examples [00:37, 1024.29 examples/s]38600 examples [00:37, 1001.79 examples/s]38701 examples [00:37, 989.23 examples/s] 38801 examples [00:37, 991.70 examples/s]38905 examples [00:37, 1004.79 examples/s]39006 examples [00:37, 1005.61 examples/s]39109 examples [00:37, 1010.92 examples/s]39213 examples [00:37, 1019.43 examples/s]39318 examples [00:37, 1026.36 examples/s]39421 examples [00:38, 1004.53 examples/s]39522 examples [00:38, 994.23 examples/s] 39625 examples [00:38, 1004.69 examples/s]39733 examples [00:38, 1024.00 examples/s]39843 examples [00:38, 1044.36 examples/s]39948 examples [00:38, 1037.33 examples/s]40052 examples [00:38, 998.19 examples/s] 40156 examples [00:38, 1007.70 examples/s]40258 examples [00:38, 1001.09 examples/s]40359 examples [00:38, 976.95 examples/s] 40457 examples [00:39, 965.27 examples/s]40562 examples [00:39, 988.29 examples/s]40663 examples [00:39, 994.20 examples/s]40763 examples [00:39, 994.24 examples/s]40873 examples [00:39, 1021.95 examples/s]40976 examples [00:39, 999.35 examples/s] 41078 examples [00:39, 1004.22 examples/s]41180 examples [00:39, 1008.62 examples/s]41287 examples [00:39, 1024.13 examples/s]41395 examples [00:39, 1039.97 examples/s]41500 examples [00:40, 1034.77 examples/s]41604 examples [00:40, 1024.32 examples/s]41707 examples [00:40, 1008.83 examples/s]41809 examples [00:40, 988.39 examples/s] 41909 examples [00:40, 979.15 examples/s]42010 examples [00:40, 985.18 examples/s]42109 examples [00:40, 971.52 examples/s]42216 examples [00:40, 996.83 examples/s]42321 examples [00:40, 1009.55 examples/s]42428 examples [00:41, 1026.20 examples/s]42531 examples [00:41, 1007.48 examples/s]42632 examples [00:41, 1003.55 examples/s]42738 examples [00:41, 1018.21 examples/s]42843 examples [00:41, 1026.76 examples/s]42952 examples [00:41, 1043.76 examples/s]43057 examples [00:41, 1029.14 examples/s]43164 examples [00:41, 1040.12 examples/s]43272 examples [00:41, 1050.41 examples/s]43378 examples [00:41, 1051.38 examples/s]43486 examples [00:42, 1059.77 examples/s]43596 examples [00:42, 1069.17 examples/s]43703 examples [00:42, 1065.07 examples/s]43810 examples [00:42, 1050.93 examples/s]43916 examples [00:42, 1043.82 examples/s]44021 examples [00:42, 1042.04 examples/s]44126 examples [00:42, 1042.04 examples/s]44233 examples [00:42, 1048.37 examples/s]44342 examples [00:42, 1059.01 examples/s]44449 examples [00:42, 1059.35 examples/s]44557 examples [00:43, 1062.94 examples/s]44666 examples [00:43, 1068.32 examples/s]44773 examples [00:43, 1060.46 examples/s]44880 examples [00:43, 1055.75 examples/s]44986 examples [00:43, 1044.93 examples/s]45091 examples [00:43, 1035.88 examples/s]45195 examples [00:43, 996.54 examples/s] 45300 examples [00:43, 1010.21 examples/s]45402 examples [00:43, 975.33 examples/s] 45505 examples [00:43, 988.78 examples/s]45609 examples [00:44, 1001.55 examples/s]45710 examples [00:44, 1003.14 examples/s]45815 examples [00:44, 1014.22 examples/s]45923 examples [00:44, 1031.44 examples/s]46029 examples [00:44, 1039.70 examples/s]46134 examples [00:44, 1035.21 examples/s]46238 examples [00:44, 1026.36 examples/s]46341 examples [00:44, 966.76 examples/s] 46439 examples [00:44, 947.41 examples/s]46540 examples [00:45, 964.53 examples/s]46640 examples [00:45, 974.81 examples/s]46739 examples [00:45, 977.54 examples/s]46838 examples [00:45, 968.02 examples/s]46936 examples [00:45, 947.05 examples/s]47031 examples [00:45, 920.93 examples/s]47124 examples [00:45, 890.92 examples/s]47233 examples [00:45, 940.26 examples/s]47332 examples [00:45, 954.18 examples/s]47429 examples [00:45, 955.27 examples/s]47526 examples [00:46, 938.52 examples/s]47621 examples [00:46, 935.90 examples/s]47715 examples [00:46, 875.66 examples/s]47811 examples [00:46, 897.13 examples/s]47917 examples [00:46, 940.11 examples/s]48020 examples [00:46, 964.23 examples/s]48118 examples [00:46, 946.33 examples/s]48214 examples [00:46, 945.73 examples/s]48317 examples [00:46, 967.34 examples/s]48415 examples [00:47, 954.97 examples/s]48512 examples [00:47, 958.29 examples/s]48612 examples [00:47, 969.36 examples/s]48710 examples [00:47, 953.96 examples/s]48814 examples [00:47, 978.05 examples/s]48919 examples [00:47, 995.70 examples/s]49021 examples [00:47, 1001.02 examples/s]49127 examples [00:47, 1015.98 examples/s]49229 examples [00:47, 1014.92 examples/s]49332 examples [00:47, 1017.81 examples/s]49438 examples [00:48, 1028.82 examples/s]49541 examples [00:48, 1028.03 examples/s]49644 examples [00:48, 1012.99 examples/s]49746 examples [00:48, 1004.05 examples/s]49848 examples [00:48, 1008.32 examples/s]49953 examples [00:48, 1019.27 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6203/50000 [00:00<00:00, 62028.73 examples/s] 38%|███▊      | 19007/50000 [00:00<00:00, 73377.57 examples/s] 64%|██████▍   | 31942/50000 [00:00<00:00, 84324.00 examples/s] 90%|█████████ | 45003/50000 [00:00<00:00, 94354.34 examples/s]                                                               0 examples [00:00, ? examples/s]87 examples [00:00, 862.69 examples/s]198 examples [00:00, 923.49 examples/s]310 examples [00:00, 972.54 examples/s]422 examples [00:00, 1011.24 examples/s]529 examples [00:00, 1026.61 examples/s]639 examples [00:00, 1047.18 examples/s]750 examples [00:00, 1065.12 examples/s]861 examples [00:00, 1076.87 examples/s]968 examples [00:00, 1073.34 examples/s]1073 examples [00:01, 1056.32 examples/s]1177 examples [00:01, 1028.81 examples/s]1279 examples [00:01, 1010.06 examples/s]1382 examples [00:01, 1015.13 examples/s]1483 examples [00:01, 1000.13 examples/s]1588 examples [00:01, 1012.09 examples/s]1689 examples [00:01, 984.51 examples/s] 1795 examples [00:01, 1004.36 examples/s]1899 examples [00:01, 1012.19 examples/s]2005 examples [00:01, 1024.52 examples/s]2111 examples [00:02, 1032.92 examples/s]2215 examples [00:02, 1030.08 examples/s]2319 examples [00:02, 1008.27 examples/s]2420 examples [00:02, 991.79 examples/s] 2530 examples [00:02, 1019.75 examples/s]2640 examples [00:02, 1041.84 examples/s]2748 examples [00:02, 1050.52 examples/s]2855 examples [00:02, 1055.10 examples/s]2961 examples [00:02, 1054.52 examples/s]3068 examples [00:02, 1057.59 examples/s]3174 examples [00:03, 1047.73 examples/s]3282 examples [00:03, 1054.87 examples/s]3388 examples [00:03, 1050.30 examples/s]3494 examples [00:03, 1009.09 examples/s]3596 examples [00:03, 1011.93 examples/s]3698 examples [00:03, 1007.25 examples/s]3806 examples [00:03, 1026.26 examples/s]3911 examples [00:03, 1030.74 examples/s]4016 examples [00:03, 1034.82 examples/s]4120 examples [00:04, 951.74 examples/s] 4222 examples [00:04, 970.27 examples/s]4326 examples [00:04, 988.79 examples/s]4430 examples [00:04, 1002.10 examples/s]4533 examples [00:04, 1009.73 examples/s]4639 examples [00:04, 1023.29 examples/s]4742 examples [00:04, 1012.25 examples/s]4846 examples [00:04, 1018.29 examples/s]4949 examples [00:04, 1017.84 examples/s]5051 examples [00:04, 985.64 examples/s] 5154 examples [00:05, 995.94 examples/s]5256 examples [00:05, 1000.76 examples/s]5359 examples [00:05, 1007.47 examples/s]5460 examples [00:05, 804.40 examples/s] 5560 examples [00:05, 853.52 examples/s]5669 examples [00:05, 911.62 examples/s]5776 examples [00:05, 952.42 examples/s]5880 examples [00:05, 976.40 examples/s]5981 examples [00:05, 975.75 examples/s]6081 examples [00:06, 957.08 examples/s]6187 examples [00:06, 983.71 examples/s]6287 examples [00:06, 978.03 examples/s]6386 examples [00:06, 928.95 examples/s]6494 examples [00:06, 967.29 examples/s]6598 examples [00:06, 985.63 examples/s]6698 examples [00:06, 977.68 examples/s]6806 examples [00:06, 1005.58 examples/s]6908 examples [00:06, 986.82 examples/s] 7008 examples [00:06, 983.25 examples/s]7107 examples [00:07, 894.09 examples/s]7210 examples [00:07, 929.66 examples/s]7317 examples [00:07, 967.45 examples/s]7416 examples [00:07, 961.17 examples/s]7517 examples [00:07, 974.42 examples/s]7616 examples [00:07, 969.76 examples/s]7714 examples [00:07, 918.69 examples/s]7816 examples [00:07, 945.96 examples/s]7915 examples [00:07, 957.59 examples/s]8020 examples [00:08, 981.40 examples/s]8119 examples [00:08, 974.12 examples/s]8217 examples [00:08, 914.82 examples/s]8310 examples [00:08, 912.06 examples/s]8411 examples [00:08, 937.57 examples/s]8515 examples [00:08, 964.03 examples/s]8613 examples [00:08, 936.20 examples/s]8708 examples [00:08, 882.98 examples/s]8800 examples [00:08, 893.01 examples/s]8899 examples [00:09, 918.69 examples/s]9000 examples [00:09, 941.92 examples/s]9108 examples [00:09, 978.26 examples/s]9208 examples [00:09, 983.84 examples/s]9307 examples [00:09, 976.88 examples/s]9406 examples [00:09, 976.40 examples/s]9513 examples [00:09, 1001.73 examples/s]9614 examples [00:09, 1001.98 examples/s]9715 examples [00:09, 952.49 examples/s] 9816 examples [00:09, 967.65 examples/s]9918 examples [00:10, 981.67 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteA2AHVS/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteA2AHVS/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f09e4808488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f09e4808488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f09e4808488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f09e286e748>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f09800ae2b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f09e4808488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f09e4808488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f09e4808488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f09800ae160>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f09e29c8160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.02 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.02 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.02 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.25 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.25 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.10 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.10 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.04 file/s]2020-09-02 06:08:43.638499: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-02 06:08:43.642772: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-09-02 06:08:43.642953: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561f53bb3140 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-02 06:08:43.642969: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  1%|          | 98304/9912422 [00:00<00:10, 980015.12it/s] 93%|█████████▎| 9183232/9912422 [00:00<00:00, 1393451.24it/s]9920512it [00:00, 47871030.32it/s]                            
0it [00:00, ?it/s]32768it [00:00, 655551.31it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1024069.53it/s]1654784it [00:00, 12173085.25it/s]                           
0it [00:00, ?it/s]8192it [00:00, 227233.42it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

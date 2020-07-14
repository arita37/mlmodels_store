
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f43247c50d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f43247c50d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f438fb0f1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f438fb0f1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f43ade56e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f43ade56e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f433ce3d620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f433ce3d620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f433ce3d620> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:06, 148972.40it/s] 54%|█████▍    | 5332992/9912422 [00:00<00:21, 212556.39it/s]9920512it [00:00, 38703800.52it/s]                           
0it [00:00, ?it/s]32768it [00:00, 587690.03it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 156023.19it/s]1654784it [00:00, 10863498.01it/s]                         
0it [00:00, ?it/s]8192it [00:00, 159376.86it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4323e7c908>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4323e6fba8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f433ce3d268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f433ce3d268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f433ce3d268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:21,  1.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:21,  1.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 2/162 [00:00<01:05,  2.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:05,  2.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 3/162 [00:00<00:54,  2.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:54,  2.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:01<00:47,  3.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:01<00:47,  3.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:01<00:41,  3.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<00:41,  3.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:01<00:37,  4.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<00:37,  4.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:01<00:35,  4.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<00:35,  4.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:01<00:33,  4.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:33,  4.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:02<00:32,  4.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:02<00:32,  4.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:02<00:31,  4.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:02<00:31,  4.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:02<00:31,  4.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:02<00:31,  4.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:02<00:30,  4.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:02<00:30,  4.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:02<00:30,  4.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:02<00:30,  4.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:03<00:29,  4.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:03<00:29,  4.95 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:03<00:29,  5.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:03<00:29,  5.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:03<00:28,  5.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:03<00:28,  5.08 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:03<00:28,  5.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:03<00:28,  5.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:03<00:28,  5.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:03<00:28,  5.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:04<00:28,  5.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:04<00:28,  5.07 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:04<00:28,  5.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:04<00:28,  5.05 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:04<00:28,  5.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:04<00:28,  5.02 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:04<00:28,  4.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:04<00:28,  4.99 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:04<00:27,  4.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:04<00:27,  4.97 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:05<00:28,  4.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:05<00:28,  4.93 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:05<00:27,  4.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:05<00:27,  4.93 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:05<00:28,  4.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:05<00:28,  4.85 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:05<00:27,  4.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:05<00:27,  4.82 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:05<00:27,  4.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:05<00:27,  4.81 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:06<00:28,  4.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:06<00:28,  4.75 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:06<00:28,  4.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:06<00:28,  4.71 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:06<00:27,  4.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:06<00:27,  4.72 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:06<00:27,  4.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:06<00:27,  4.77 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:06<00:26,  4.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:06<00:26,  4.83 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:07<00:26,  4.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:07<00:26,  4.89 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:07<00:25,  4.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:07<00:25,  4.92 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:07<00:25,  4.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:07<00:25,  4.89 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:07<00:25,  4.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:07<00:25,  4.88 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:07<00:25,  4.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:07<00:25,  4.90 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:08<00:25,  4.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:08<00:25,  4.88 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:08<00:25,  4.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:08<00:25,  4.86 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:08<00:25,  4.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:08<00:25,  4.84 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:08<00:25,  4.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:08<00:25,  4.76 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:08<00:25,  4.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:08<00:25,  4.68 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:09<00:25,  4.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:09<00:25,  4.61 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:09<00:25,  4.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:09<00:25,  4.54 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:09<00:25,  4.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:09<00:25,  4.48 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:09<00:25,  4.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:09<00:25,  4.43 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:10<00:25,  4.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:10<00:25,  4.46 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:10<00:25,  4.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:10<00:25,  4.51 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:10<00:24,  4.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:10<00:24,  4.54 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:10<00:24,  4.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:10<00:24,  4.55 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:11<00:24,  4.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:11<00:24,  4.54 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:11<00:24,  4.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:11<00:24,  4.44 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:11<00:24,  4.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:11<00:24,  4.46 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:11<00:23,  4.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:11<00:23,  4.50 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:11<00:23,  4.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:11<00:23,  4.54 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:12<00:23,  4.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:12<00:23,  4.55 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:12<00:22,  4.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:12<00:22,  4.56 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:12<00:22,  4.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:12<00:22,  4.58 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:12<00:22,  4.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:12<00:22,  4.59 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:12<00:22,  4.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:12<00:22,  4.59 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:13<00:21,  4.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:13<00:21,  4.60 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:13<00:21,  4.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:13<00:21,  4.60 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:13<00:21,  4.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:13<00:21,  4.60 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:13<00:21,  4.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:13<00:21,  4.60 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:14<00:20,  4.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:14<00:20,  4.60 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:14<00:20,  4.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:14<00:20,  4.61 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:14<00:20,  4.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:14<00:20,  4.61 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:14<00:20,  4.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:14<00:20,  4.58 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:14<00:20,  4.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:14<00:20,  4.53 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:15<00:20,  4.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:15<00:20,  4.49 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:15<00:20,  4.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:15<00:20,  4.45 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:15<00:20,  4.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:15<00:20,  4.45 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:15<00:19,  4.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:15<00:19,  4.42 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:16<00:19,  4.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:16<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:16<00:19,  4.37 MiB/s][A

Extraction completed...: 0 file [00:16, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:16<00:20,  4.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:16<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:16<00:20,  4.29 MiB/s][A

Extraction completed...: 0 file [00:16, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:16<00:20,  4.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:16<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:16<00:20,  4.22 MiB/s][A

Extraction completed...: 0 file [00:16, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:16<00:19,  4.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:16<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:16<00:19,  4.26 MiB/s][A

Extraction completed...: 0 file [00:16, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:17<00:19,  4.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:17<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:17<00:19,  4.34 MiB/s][A

Extraction completed...: 0 file [00:17, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:17<00:18,  4.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:17<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:17<00:18,  4.40 MiB/s][A

Extraction completed...: 0 file [00:17, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:17<00:18,  4.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:17<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:17<00:18,  4.41 MiB/s][A

Extraction completed...: 0 file [00:17, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:17<00:17,  4.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:17<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:17<00:17,  4.51 MiB/s][A

Extraction completed...: 0 file [00:17, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:17<00:17,  4.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:17<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:17<00:17,  4.49 MiB/s][A

Extraction completed...: 0 file [00:17, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:18<00:17,  4.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:18<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:18<00:17,  4.57 MiB/s][A

Extraction completed...: 0 file [00:18, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:18<00:16,  4.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:18<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:18<00:16,  4.64 MiB/s][A

Extraction completed...: 0 file [00:18, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:18<00:16,  4.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:18<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:18<00:16,  4.72 MiB/s][A

Extraction completed...: 0 file [00:18, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:18<00:15,  4.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:18<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:18<00:15,  4.77 MiB/s][A

Extraction completed...: 0 file [00:18, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:18<00:15,  4.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:18<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:18<00:15,  4.84 MiB/s][A

Extraction completed...: 0 file [00:18, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:19<00:15,  4.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:19<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:19<00:15,  4.86 MiB/s][A

Extraction completed...: 0 file [00:19, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:19<00:14,  4.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:19<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:19<00:14,  4.88 MiB/s][A

Extraction completed...: 0 file [00:19, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:19<00:14,  4.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:19<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:19<00:14,  4.85 MiB/s][A

Extraction completed...: 0 file [00:19, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:19<00:14,  4.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:19<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:19<00:14,  4.84 MiB/s][A

Extraction completed...: 0 file [00:19, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:19<00:14,  4.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:19<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:19<00:14,  4.83 MiB/s][A

Extraction completed...: 0 file [00:19, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:20<00:14,  4.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:20<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:20<00:14,  4.86 MiB/s][A

Extraction completed...: 0 file [00:20, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:20<00:13,  4.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:20<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:20<00:13,  4.82 MiB/s][A

Extraction completed...: 0 file [00:20, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:20<00:13,  4.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:20<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:20<00:13,  4.88 MiB/s][A

Extraction completed...: 0 file [00:20, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:20<00:13,  4.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:20<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:20<00:13,  4.90 MiB/s][A

Extraction completed...: 0 file [00:20, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:20<00:13,  4.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:20<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:20<00:13,  4.91 MiB/s][A

Extraction completed...: 0 file [00:20, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:21<00:12,  4.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:21<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:21<00:12,  4.98 MiB/s][A

Extraction completed...: 0 file [00:21, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:21<00:12,  5.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:21<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:21<00:12,  5.06 MiB/s][A

Extraction completed...: 0 file [00:21, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:21<00:11,  5.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:21<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:21<00:11,  5.11 MiB/s][A

Extraction completed...: 0 file [00:21, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:21<00:11,  5.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:21<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:21<00:11,  5.15 MiB/s][A

Extraction completed...: 0 file [00:21, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:21<00:11,  5.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:21<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:21<00:11,  5.19 MiB/s][A

Extraction completed...: 0 file [00:21, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:22<00:11,  5.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:22<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:22<00:11,  5.18 MiB/s][A

Extraction completed...: 0 file [00:22, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:22<00:11,  5.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:22<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:22<00:11,  5.18 MiB/s][A

Extraction completed...: 0 file [00:22, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:22<00:10,  5.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:22<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:22<00:10,  5.16 MiB/s][A

Extraction completed...: 0 file [00:22, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:22<00:10,  5.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:22<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:22<00:10,  5.16 MiB/s][A

Extraction completed...: 0 file [00:22, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:22<00:10,  5.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:22<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:22<00:10,  5.19 MiB/s][A

Extraction completed...: 0 file [00:22, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:23<00:10,  5.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:23<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:23<00:10,  5.20 MiB/s][A

Extraction completed...: 0 file [00:23, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:23<00:10,  5.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:23<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:23<00:10,  5.13 MiB/s][A

Extraction completed...: 0 file [00:23, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:23<00:10,  5.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:23<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:23<00:10,  5.09 MiB/s][A

Extraction completed...: 0 file [00:23, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:23<00:09,  5.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:23<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:23<00:09,  5.05 MiB/s][A

Extraction completed...: 0 file [00:23, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:23<00:09,  5.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:23<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:23<00:09,  5.03 MiB/s][A

Extraction completed...: 0 file [00:23, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:24<00:09,  5.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:24<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:24<00:09,  5.03 MiB/s][A

Extraction completed...: 0 file [00:24, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:24<00:09,  5.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:24<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:24<00:09,  5.01 MiB/s][A

Extraction completed...: 0 file [00:24, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:24<00:09,  5.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:24<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:24<00:09,  5.01 MiB/s][A

Extraction completed...: 0 file [00:24, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:24<00:08,  5.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:24<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:24<00:08,  5.00 MiB/s][A

Extraction completed...: 0 file [00:24, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:24<00:08,  4.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:24<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:24<00:08,  4.95 MiB/s][A

Extraction completed...: 0 file [00:24, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:25<00:08,  4.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:25<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:25<00:08,  4.95 MiB/s][A

Extraction completed...: 0 file [00:25, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:25<00:08,  4.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:25<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:25<00:08,  4.96 MiB/s][A

Extraction completed...: 0 file [00:25, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:25<00:08,  4.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:25<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:25<00:08,  4.96 MiB/s][A

Extraction completed...: 0 file [00:25, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:25<00:08,  4.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:25<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:25<00:08,  4.97 MiB/s][A

Extraction completed...: 0 file [00:25, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:25<00:07,  4.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:25<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:25<00:07,  4.96 MiB/s][A

Extraction completed...: 0 file [00:25, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:26<00:07,  4.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:26<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:26<00:07,  4.93 MiB/s][A

Extraction completed...: 0 file [00:26, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:26<00:07,  5.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:26<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:26<00:07,  5.00 MiB/s][A

Extraction completed...: 0 file [00:26, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:26<00:07,  5.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:26<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:26<00:07,  5.04 MiB/s][A

Extraction completed...: 0 file [00:26, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:26<00:06,  5.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:26<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:26<00:06,  5.14 MiB/s][A

Extraction completed...: 0 file [00:26, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:26<00:06,  5.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:26<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:26<00:06,  5.22 MiB/s][A

Extraction completed...: 0 file [00:26, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:27<00:06,  5.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:27<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:27<00:06,  5.28 MiB/s][A

Extraction completed...: 0 file [00:27, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:27<00:05,  5.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:27<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:27<00:05,  5.35 MiB/s][A

Extraction completed...: 0 file [00:27, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:27<00:05,  5.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:27<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:27<00:05,  5.36 MiB/s][A

Extraction completed...: 0 file [00:27, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:27<00:05,  5.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:27<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:27<00:05,  5.39 MiB/s][A

Extraction completed...: 0 file [00:27, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:27<00:05,  5.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:27<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:27<00:05,  5.39 MiB/s][A

Extraction completed...: 0 file [00:27, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:27<00:05,  5.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:27<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:27<00:05,  5.40 MiB/s][A

Extraction completed...: 0 file [00:27, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:28<00:05,  5.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:28<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:28<00:05,  5.40 MiB/s][A

Extraction completed...: 0 file [00:28, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:28<00:04,  5.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:28<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:28<00:04,  5.37 MiB/s][A

Extraction completed...: 0 file [00:28, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:28<00:04,  5.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:28<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:28<00:04,  5.42 MiB/s][A

Extraction completed...: 0 file [00:28, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:28<00:04,  5.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:28<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:28<00:04,  5.44 MiB/s][A

Extraction completed...: 0 file [00:28, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:28<00:04,  5.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:28<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:28<00:04,  5.43 MiB/s][A

Extraction completed...: 0 file [00:28, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:29<00:04,  5.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:29<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:29<00:04,  5.45 MiB/s][A

Extraction completed...: 0 file [00:29, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:29<00:03,  5.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:29<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:29<00:03,  5.46 MiB/s][A

Extraction completed...: 0 file [00:29, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:29<00:03,  5.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:29<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:29<00:03,  5.45 MiB/s][A

Extraction completed...: 0 file [00:29, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:29<00:03,  5.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:29<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:29<00:03,  5.28 MiB/s][A

Extraction completed...: 0 file [00:29, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:29<00:03,  5.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:29<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:29<00:03,  5.32 MiB/s][A

Extraction completed...: 0 file [00:29, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:30<00:03,  5.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:30<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:30<00:03,  5.44 MiB/s][A

Extraction completed...: 0 file [00:30, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:30<00:02,  5.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:30<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:30<00:02,  5.52 MiB/s][A

Extraction completed...: 0 file [00:30, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:30<00:02,  5.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:30<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:30<00:02,  5.48 MiB/s][A

Extraction completed...: 0 file [00:30, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:30<00:02,  5.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:30<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:30<00:02,  5.38 MiB/s][A

Extraction completed...: 0 file [00:30, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:30<00:02,  5.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:30<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:30<00:02,  5.31 MiB/s][A

Extraction completed...: 0 file [00:30, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:30<00:02,  5.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:30<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:30<00:02,  5.25 MiB/s][A

Extraction completed...: 0 file [00:30, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:31<00:02,  5.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:31<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:31<00:02,  5.20 MiB/s][A

Extraction completed...: 0 file [00:31, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:31<00:01,  5.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:31<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:31<00:01,  5.21 MiB/s][A

Extraction completed...: 0 file [00:31, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:31<00:01,  5.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:31<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:31<00:01,  5.27 MiB/s][A

Extraction completed...: 0 file [00:31, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:31<00:01,  5.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:31<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:31<00:01,  5.27 MiB/s][A

Extraction completed...: 0 file [00:31, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:31<00:01,  5.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:31<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:31<00:01,  5.36 MiB/s][A

Extraction completed...: 0 file [00:31, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:32<00:01,  5.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:32<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:32<00:01,  5.53 MiB/s][A

Extraction completed...: 0 file [00:32, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:32<00:00,  5.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:32<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:32<00:00,  5.67 MiB/s][A

Extraction completed...: 0 file [00:32, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:32<00:00,  5.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:32<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:32<00:00,  5.84 MiB/s][A

Extraction completed...: 0 file [00:32, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:32<00:00,  6.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:32<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:32<00:00,  6.02 MiB/s][A

Extraction completed...: 0 file [00:32, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:32<00:00,  6.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:32<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:32<00:00,  6.13 MiB/s][A

Extraction completed...: 0 file [00:32, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:33<00:00,  2.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:33<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:33<00:00,  2.19 MiB/s][A

Extraction completed...: 0 file [00:33, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:33<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:33<00:00,  2.19 MiB/s][A

Extraction completed...: 0 file [00:33, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:33<00:00, 33.89s/ url]Dl Completed...: 100%|██████████| 1/1 [00:33<00:00, 33.89s/ url]
Dl Size...: 100%|██████████| 162/162 [00:33<00:00,  2.19 MiB/s][A

Extraction completed...: 0 file [00:33, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:33<00:00, 33.89s/ url]
Dl Size...: 100%|██████████| 162/162 [00:33<00:00,  2.19 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:33<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:36<00:00, 36.10s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:36<00:00, 33.89s/ url]
Dl Size...: 100%|██████████| 162/162 [00:36<00:00,  2.19 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:36<00:00, 36.10s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:36<00:00, 36.10s/ file]
Dl Size...: 100%|██████████| 162/162 [00:36<00:00,  4.49 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:36<00:00, 36.10s/ url]
0 examples [00:00, ? examples/s]2020-07-14 18:10:25.344518: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-14 18:10:25.392838: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-14 18:10:25.393040: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560ecd5e5270 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-14 18:10:25.393058: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.95 examples/s]107 examples [00:00,  4.20 examples/s]209 examples [00:00,  6.00 examples/s]315 examples [00:00,  8.54 examples/s]420 examples [00:00, 12.16 examples/s]521 examples [00:00, 17.29 examples/s]610 examples [00:00, 24.49 examples/s]707 examples [00:01, 34.61 examples/s]800 examples [00:01, 48.67 examples/s]896 examples [00:01, 68.05 examples/s]1002 examples [00:01, 94.60 examples/s]1104 examples [00:01, 129.94 examples/s]1202 examples [00:01, 175.28 examples/s]1303 examples [00:01, 233.00 examples/s]1406 examples [00:01, 303.29 examples/s]1516 examples [00:01, 387.23 examples/s]1626 examples [00:01, 480.14 examples/s]1731 examples [00:02, 567.09 examples/s]1840 examples [00:02, 661.94 examples/s]1945 examples [00:02, 734.55 examples/s]2053 examples [00:02, 811.56 examples/s]2162 examples [00:02, 877.89 examples/s]2268 examples [00:02, 924.64 examples/s]2377 examples [00:02, 966.97 examples/s]2484 examples [00:02, 990.40 examples/s]2590 examples [00:02, 983.77 examples/s]2694 examples [00:02, 994.91 examples/s]2804 examples [00:03, 1024.00 examples/s]2909 examples [00:03, 1029.76 examples/s]3017 examples [00:03, 1043.50 examples/s]3123 examples [00:03, 1037.68 examples/s]3231 examples [00:03, 1048.77 examples/s]3337 examples [00:03, 1033.16 examples/s]3446 examples [00:03, 1048.36 examples/s]3552 examples [00:03, 1034.05 examples/s]3656 examples [00:03, 1031.74 examples/s]3760 examples [00:03, 1031.62 examples/s]3866 examples [00:04, 1039.80 examples/s]3974 examples [00:04, 1049.08 examples/s]4080 examples [00:04, 1050.95 examples/s]4186 examples [00:04, 1040.27 examples/s]4291 examples [00:04, 1010.51 examples/s]4393 examples [00:04, 1000.79 examples/s]4497 examples [00:04, 1011.54 examples/s]4606 examples [00:04, 1031.41 examples/s]4711 examples [00:04, 1036.73 examples/s]4815 examples [00:05, 1034.80 examples/s]4924 examples [00:05, 1049.85 examples/s]5030 examples [00:05, 1009.25 examples/s]5132 examples [00:05, 975.65 examples/s] 5240 examples [00:05, 992.15 examples/s]5350 examples [00:05, 1021.08 examples/s]5459 examples [00:05, 1038.17 examples/s]5566 examples [00:05, 1045.26 examples/s]5675 examples [00:05, 1056.24 examples/s]5781 examples [00:05, 1035.72 examples/s]5885 examples [00:06, 1034.92 examples/s]5991 examples [00:06, 1041.77 examples/s]6096 examples [00:06, 1029.00 examples/s]6206 examples [00:06, 1048.12 examples/s]6316 examples [00:06, 1062.82 examples/s]6426 examples [00:06, 1072.27 examples/s]6537 examples [00:06, 1081.05 examples/s]6646 examples [00:06, 1071.77 examples/s]6758 examples [00:06, 1084.76 examples/s]6867 examples [00:06, 1078.62 examples/s]6975 examples [00:07, 1077.74 examples/s]7085 examples [00:07, 1082.73 examples/s]7194 examples [00:07, 1081.41 examples/s]7305 examples [00:07, 1089.40 examples/s]7414 examples [00:07, 1089.35 examples/s]7523 examples [00:07, 1041.24 examples/s]7628 examples [00:07, 1043.61 examples/s]7733 examples [00:07, 1042.29 examples/s]7838 examples [00:07, 1037.31 examples/s]7948 examples [00:07, 1052.19 examples/s]8059 examples [00:08, 1067.73 examples/s]8168 examples [00:08, 1071.83 examples/s]8276 examples [00:08, 1069.89 examples/s]8385 examples [00:08, 1073.76 examples/s]8494 examples [00:08, 1077.05 examples/s]8602 examples [00:08, 1072.06 examples/s]8710 examples [00:08, 1073.89 examples/s]8818 examples [00:08, 1062.12 examples/s]8925 examples [00:08, 1062.98 examples/s]9032 examples [00:09, 1050.66 examples/s]9140 examples [00:09, 1057.00 examples/s]9246 examples [00:09, 1043.98 examples/s]9351 examples [00:09, 1020.89 examples/s]9459 examples [00:09, 1037.85 examples/s]9567 examples [00:09, 1048.91 examples/s]9677 examples [00:09, 1061.64 examples/s]9784 examples [00:09, 1014.84 examples/s]9894 examples [00:09, 1037.12 examples/s]9999 examples [00:09, 1040.36 examples/s]10104 examples [00:10, 993.18 examples/s]10211 examples [00:10, 1013.22 examples/s]10318 examples [00:10, 1027.77 examples/s]10425 examples [00:10, 1039.17 examples/s]10533 examples [00:10, 1048.85 examples/s]10639 examples [00:10, 1051.28 examples/s]10745 examples [00:10, 1016.63 examples/s]10854 examples [00:10, 1035.64 examples/s]10958 examples [00:10, 1018.49 examples/s]11061 examples [00:10, 1016.16 examples/s]11168 examples [00:11, 1029.77 examples/s]11274 examples [00:11, 1037.78 examples/s]11384 examples [00:11, 1055.69 examples/s]11490 examples [00:11, 1028.68 examples/s]11597 examples [00:11, 1040.53 examples/s]11705 examples [00:11, 1051.03 examples/s]11811 examples [00:11, 1044.39 examples/s]11921 examples [00:11, 1058.61 examples/s]12028 examples [00:11, 1059.26 examples/s]12135 examples [00:12, 1042.19 examples/s]12240 examples [00:12, 1043.55 examples/s]12345 examples [00:12, 1021.40 examples/s]12456 examples [00:12, 1044.15 examples/s]12564 examples [00:12, 1052.22 examples/s]12673 examples [00:12, 1062.88 examples/s]12784 examples [00:12, 1074.30 examples/s]12893 examples [00:12, 1076.99 examples/s]13001 examples [00:12, 1061.28 examples/s]13108 examples [00:12, 1055.31 examples/s]13217 examples [00:13, 1062.83 examples/s]13327 examples [00:13, 1073.37 examples/s]13436 examples [00:13, 1077.60 examples/s]13547 examples [00:13, 1084.60 examples/s]13656 examples [00:13, 1072.90 examples/s]13764 examples [00:13, 1070.96 examples/s]13872 examples [00:13, 1073.26 examples/s]13980 examples [00:13, 1004.37 examples/s]14092 examples [00:13, 1034.13 examples/s]14200 examples [00:13, 1047.22 examples/s]14306 examples [00:14, 1049.08 examples/s]14412 examples [00:14, 1035.20 examples/s]14518 examples [00:14, 1040.49 examples/s]14627 examples [00:14, 1052.78 examples/s]14734 examples [00:14, 1057.54 examples/s]14840 examples [00:14, 1048.99 examples/s]14946 examples [00:14, 1039.36 examples/s]15052 examples [00:14, 1044.86 examples/s]15161 examples [00:14, 1057.01 examples/s]15271 examples [00:14, 1069.15 examples/s]15380 examples [00:15, 1073.65 examples/s]15488 examples [00:15, 1073.25 examples/s]15596 examples [00:15, 1067.19 examples/s]15703 examples [00:15, 1058.89 examples/s]15811 examples [00:15, 1064.89 examples/s]15918 examples [00:15, 1065.92 examples/s]16025 examples [00:15, 1056.84 examples/s]16131 examples [00:15, 1031.37 examples/s]16236 examples [00:15, 1036.59 examples/s]16343 examples [00:15, 1045.83 examples/s]16453 examples [00:16, 1059.31 examples/s]16563 examples [00:16, 1068.47 examples/s]16670 examples [00:16, 1068.26 examples/s]16778 examples [00:16, 1068.79 examples/s]16886 examples [00:16, 1071.75 examples/s]16994 examples [00:16, 1067.67 examples/s]17103 examples [00:16, 1072.03 examples/s]17211 examples [00:16, 1032.22 examples/s]17315 examples [00:16, 1022.56 examples/s]17420 examples [00:17, 1028.87 examples/s]17524 examples [00:17, 1021.16 examples/s]17632 examples [00:17, 1037.61 examples/s]17736 examples [00:17, 1022.26 examples/s]17840 examples [00:17, 1026.59 examples/s]17949 examples [00:17, 1042.13 examples/s]18055 examples [00:17, 1047.05 examples/s]18163 examples [00:17, 1054.29 examples/s]18269 examples [00:17, 1050.75 examples/s]18376 examples [00:17, 1056.23 examples/s]18486 examples [00:18, 1067.13 examples/s]18593 examples [00:18, 1067.87 examples/s]18700 examples [00:18, 1054.78 examples/s]18808 examples [00:18, 1061.93 examples/s]18916 examples [00:18, 1064.70 examples/s]19024 examples [00:18, 1066.67 examples/s]19131 examples [00:18, 1057.58 examples/s]19239 examples [00:18, 1057.95 examples/s]19345 examples [00:18, 1045.16 examples/s]19450 examples [00:18, 1019.05 examples/s]19557 examples [00:19, 1032.55 examples/s]19661 examples [00:19, 1029.14 examples/s]19765 examples [00:19, 1029.42 examples/s]19871 examples [00:19, 1036.95 examples/s]19980 examples [00:19, 1050.11 examples/s]20086 examples [00:19, 975.86 examples/s] 20185 examples [00:19, 959.96 examples/s]20287 examples [00:19, 976.80 examples/s]20393 examples [00:19, 998.63 examples/s]20501 examples [00:19, 1019.09 examples/s]20613 examples [00:20, 1045.48 examples/s]20719 examples [00:20, 1039.70 examples/s]20828 examples [00:20, 1053.76 examples/s]20937 examples [00:20, 1061.98 examples/s]21044 examples [00:20, 1057.39 examples/s]21150 examples [00:20, 1052.04 examples/s]21256 examples [00:20, 1050.58 examples/s]21365 examples [00:20, 1061.34 examples/s]21472 examples [00:20, 1063.83 examples/s]21580 examples [00:21, 1067.00 examples/s]21690 examples [00:21, 1076.03 examples/s]21798 examples [00:21, 1074.39 examples/s]21906 examples [00:21, 1062.15 examples/s]22018 examples [00:21, 1076.20 examples/s]22128 examples [00:21, 1081.55 examples/s]22239 examples [00:21, 1089.34 examples/s]22348 examples [00:21, 1078.11 examples/s]22456 examples [00:21, 1061.54 examples/s]22567 examples [00:21, 1074.39 examples/s]22675 examples [00:22, 1060.95 examples/s]22782 examples [00:22, 1032.12 examples/s]22887 examples [00:22, 1035.14 examples/s]22996 examples [00:22, 1048.47 examples/s]23105 examples [00:22, 1058.34 examples/s]23212 examples [00:22, 1060.46 examples/s]23319 examples [00:22, 1033.15 examples/s]23427 examples [00:22, 1044.51 examples/s]23532 examples [00:22, 985.14 examples/s] 23640 examples [00:22, 1009.75 examples/s]23745 examples [00:23, 1020.30 examples/s]23848 examples [00:23, 1021.33 examples/s]23958 examples [00:23, 1041.37 examples/s]24064 examples [00:23, 1045.23 examples/s]24169 examples [00:23, 1037.88 examples/s]24276 examples [00:23, 1044.87 examples/s]24381 examples [00:23, 1042.75 examples/s]24486 examples [00:23, 1035.73 examples/s]24592 examples [00:23, 1040.41 examples/s]24697 examples [00:23, 1039.12 examples/s]24803 examples [00:24, 1044.28 examples/s]24913 examples [00:24, 1058.70 examples/s]25021 examples [00:24, 1064.14 examples/s]25130 examples [00:24, 1070.65 examples/s]25238 examples [00:24, 1034.02 examples/s]25344 examples [00:24, 1040.98 examples/s]25449 examples [00:24, 1023.11 examples/s]25553 examples [00:24, 1028.03 examples/s]25656 examples [00:24, 1004.06 examples/s]25757 examples [00:25, 1004.03 examples/s]25865 examples [00:25, 1023.84 examples/s]25968 examples [00:25, 1016.06 examples/s]26070 examples [00:25, 995.49 examples/s] 26174 examples [00:25, 1005.81 examples/s]26275 examples [00:25, 995.49 examples/s] 26382 examples [00:25, 1014.43 examples/s]26490 examples [00:25, 1030.93 examples/s]26599 examples [00:25, 1047.39 examples/s]26704 examples [00:25, 1017.29 examples/s]26807 examples [00:26, 1005.98 examples/s]26915 examples [00:26, 1024.94 examples/s]27024 examples [00:26, 1041.26 examples/s]27129 examples [00:26, 1033.98 examples/s]27238 examples [00:26, 1050.15 examples/s]27347 examples [00:26, 1059.59 examples/s]27454 examples [00:26, 1029.86 examples/s]27558 examples [00:26, 1024.78 examples/s]27663 examples [00:26, 1032.17 examples/s]27767 examples [00:26, 1032.41 examples/s]27871 examples [00:27, 1028.16 examples/s]27981 examples [00:27, 1047.10 examples/s]28090 examples [00:27, 1058.16 examples/s]28198 examples [00:27, 1062.94 examples/s]28308 examples [00:27, 1073.57 examples/s]28416 examples [00:27, 1069.01 examples/s]28523 examples [00:27, 1065.49 examples/s]28630 examples [00:27, 1059.48 examples/s]28736 examples [00:27, 1022.00 examples/s]28844 examples [00:27, 1037.78 examples/s]28953 examples [00:28, 1051.02 examples/s]29061 examples [00:28, 1059.11 examples/s]29170 examples [00:28, 1065.61 examples/s]29279 examples [00:28, 1071.34 examples/s]29387 examples [00:28, 1064.65 examples/s]29494 examples [00:28, 1057.00 examples/s]29600 examples [00:28, 1051.67 examples/s]29706 examples [00:28, 1047.18 examples/s]29811 examples [00:28, 1028.33 examples/s]29914 examples [00:29, 966.45 examples/s] 30012 examples [00:29, 932.84 examples/s]30119 examples [00:29, 968.49 examples/s]30219 examples [00:29, 975.63 examples/s]30320 examples [00:29, 984.94 examples/s]30419 examples [00:29, 979.64 examples/s]30526 examples [00:29, 1003.03 examples/s]30627 examples [00:29, 987.66 examples/s] 30736 examples [00:29, 1014.76 examples/s]30842 examples [00:29, 1027.72 examples/s]30949 examples [00:30, 1038.05 examples/s]31056 examples [00:30, 1044.89 examples/s]31161 examples [00:30, 997.78 examples/s] 31266 examples [00:30, 1010.31 examples/s]31370 examples [00:30, 1016.28 examples/s]31478 examples [00:30, 1033.26 examples/s]31584 examples [00:30, 1039.50 examples/s]31689 examples [00:30, 1008.93 examples/s]31792 examples [00:30, 1014.50 examples/s]31896 examples [00:30, 1020.39 examples/s]32001 examples [00:31, 1026.88 examples/s]32104 examples [00:31, 1027.18 examples/s]32212 examples [00:31, 1039.36 examples/s]32317 examples [00:31, 1000.93 examples/s]32422 examples [00:31, 1013.81 examples/s]32526 examples [00:31, 1019.83 examples/s]32631 examples [00:31, 1026.35 examples/s]32738 examples [00:31, 1037.54 examples/s]32843 examples [00:31, 1038.77 examples/s]32947 examples [00:32, 1019.10 examples/s]33050 examples [00:32, 997.34 examples/s] 33151 examples [00:32, 999.86 examples/s]33259 examples [00:32, 1020.71 examples/s]33363 examples [00:32, 1025.90 examples/s]33472 examples [00:32, 1041.27 examples/s]33578 examples [00:32, 1044.79 examples/s]33688 examples [00:32, 1059.75 examples/s]33795 examples [00:32, 1046.81 examples/s]33900 examples [00:32, 1044.76 examples/s]34008 examples [00:33, 1053.73 examples/s]34117 examples [00:33, 1063.35 examples/s]34224 examples [00:33, 1057.30 examples/s]34332 examples [00:33, 1061.74 examples/s]34439 examples [00:33, 1043.94 examples/s]34544 examples [00:33, 1006.23 examples/s]34650 examples [00:33, 1021.75 examples/s]34753 examples [00:33, 998.88 examples/s] 34857 examples [00:33, 1009.29 examples/s]34967 examples [00:33, 1033.59 examples/s]35075 examples [00:34, 1044.82 examples/s]35180 examples [00:34, 1031.80 examples/s]35286 examples [00:34, 1038.71 examples/s]35391 examples [00:34, 1040.14 examples/s]35499 examples [00:34, 1049.78 examples/s]35607 examples [00:34, 1057.68 examples/s]35717 examples [00:34, 1067.26 examples/s]35824 examples [00:34, 1051.81 examples/s]35931 examples [00:34, 1056.40 examples/s]36039 examples [00:34, 1061.07 examples/s]36146 examples [00:35, 1050.10 examples/s]36252 examples [00:35, 1045.12 examples/s]36357 examples [00:35, 971.03 examples/s] 36458 examples [00:35, 981.73 examples/s]36560 examples [00:35, 992.52 examples/s]36663 examples [00:35, 1000.98 examples/s]36764 examples [00:35, 996.54 examples/s] 36864 examples [00:35, 993.98 examples/s]36969 examples [00:35, 1009.14 examples/s]37072 examples [00:36, 1013.44 examples/s]37178 examples [00:36, 1024.93 examples/s]37284 examples [00:36, 1034.61 examples/s]37389 examples [00:36, 1036.80 examples/s]37494 examples [00:36, 1040.15 examples/s]37599 examples [00:36, 1037.83 examples/s]37703 examples [00:36, 1038.09 examples/s]37807 examples [00:36, 1009.21 examples/s]37909 examples [00:36, 1003.46 examples/s]38010 examples [00:36, 1000.92 examples/s]38114 examples [00:37, 1009.97 examples/s]38221 examples [00:37, 1025.51 examples/s]38331 examples [00:37, 1044.78 examples/s]38440 examples [00:37, 1057.76 examples/s]38546 examples [00:37, 1054.02 examples/s]38652 examples [00:37, 1037.82 examples/s]38757 examples [00:37, 1040.22 examples/s]38865 examples [00:37, 1049.45 examples/s]38975 examples [00:37, 1063.61 examples/s]39082 examples [00:37, 1063.56 examples/s]39189 examples [00:38, 1061.40 examples/s]39297 examples [00:38, 1064.79 examples/s]39405 examples [00:38, 1067.11 examples/s]39512 examples [00:38, 1007.63 examples/s]39618 examples [00:38, 1020.09 examples/s]39722 examples [00:38, 1025.58 examples/s]39827 examples [00:38, 1029.66 examples/s]39931 examples [00:38, 1031.71 examples/s]40035 examples [00:38, 970.35 examples/s] 40133 examples [00:38, 957.09 examples/s]40235 examples [00:39, 974.88 examples/s]40341 examples [00:39, 998.61 examples/s]40448 examples [00:39, 1018.09 examples/s]40551 examples [00:39, 1017.81 examples/s]40654 examples [00:39, 984.60 examples/s] 40760 examples [00:39, 1005.31 examples/s]40869 examples [00:39, 1026.96 examples/s]40973 examples [00:39, 1028.76 examples/s]41081 examples [00:39, 1041.28 examples/s]41186 examples [00:40, 1038.36 examples/s]41295 examples [00:40, 1052.59 examples/s]41403 examples [00:40, 1059.62 examples/s]41510 examples [00:40, 1055.67 examples/s]41618 examples [00:40, 1062.32 examples/s]41725 examples [00:40, 1062.80 examples/s]41832 examples [00:40, 1060.00 examples/s]41939 examples [00:40, 1046.92 examples/s]42048 examples [00:40, 1057.08 examples/s]42155 examples [00:40, 1058.96 examples/s]42262 examples [00:41, 1062.18 examples/s]42372 examples [00:41, 1071.06 examples/s]42481 examples [00:41, 1075.97 examples/s]42591 examples [00:41, 1082.50 examples/s]42700 examples [00:41, 1010.00 examples/s]42809 examples [00:41, 1031.78 examples/s]42918 examples [00:41, 1046.16 examples/s]43026 examples [00:41, 1051.88 examples/s]43132 examples [00:41, 1053.41 examples/s]43238 examples [00:41, 1042.96 examples/s]43345 examples [00:42, 1048.38 examples/s]43451 examples [00:42, 1047.16 examples/s]43556 examples [00:42, 1032.25 examples/s]43665 examples [00:42, 1046.67 examples/s]43770 examples [00:42, 1035.41 examples/s]43880 examples [00:42, 1051.64 examples/s]43990 examples [00:42, 1063.72 examples/s]44100 examples [00:42, 1073.85 examples/s]44208 examples [00:42, 1068.73 examples/s]44315 examples [00:42, 1045.13 examples/s]44421 examples [00:43, 1047.56 examples/s]44526 examples [00:43, 1036.43 examples/s]44630 examples [00:43, 1001.69 examples/s]44740 examples [00:43, 1028.63 examples/s]44844 examples [00:43, 1026.19 examples/s]44947 examples [00:43, 1009.11 examples/s]45051 examples [00:43, 1014.87 examples/s]45159 examples [00:43, 1031.64 examples/s]45269 examples [00:43, 1049.31 examples/s]45375 examples [00:44, 1048.41 examples/s]45482 examples [00:44, 1052.81 examples/s]45590 examples [00:44, 1058.31 examples/s]45697 examples [00:44, 1061.65 examples/s]45804 examples [00:44, 1057.93 examples/s]45910 examples [00:44, 1014.04 examples/s]46015 examples [00:44, 1022.58 examples/s]46123 examples [00:44, 1038.17 examples/s]46228 examples [00:44, 1034.60 examples/s]46339 examples [00:44, 1054.30 examples/s]46445 examples [00:45, 1044.36 examples/s]46554 examples [00:45, 1056.82 examples/s]46660 examples [00:45, 1052.66 examples/s]46766 examples [00:45, 1030.25 examples/s]46870 examples [00:45, 993.49 examples/s] 46976 examples [00:45, 1011.23 examples/s]47078 examples [00:45, 994.86 examples/s] 47185 examples [00:45, 1015.26 examples/s]47287 examples [00:45, 1001.11 examples/s]47391 examples [00:45, 1011.08 examples/s]47493 examples [00:46, 1007.02 examples/s]47601 examples [00:46, 1027.05 examples/s]47709 examples [00:46, 1041.84 examples/s]47814 examples [00:46, 1023.43 examples/s]47917 examples [00:46, 1024.46 examples/s]48023 examples [00:46, 1032.75 examples/s]48128 examples [00:46, 1033.49 examples/s]48232 examples [00:46, 1002.72 examples/s]48338 examples [00:46, 1016.82 examples/s]48440 examples [00:47, 1002.47 examples/s]48543 examples [00:47, 1009.60 examples/s]48646 examples [00:47, 1014.55 examples/s]48754 examples [00:47, 1032.53 examples/s]48858 examples [00:47, 1029.26 examples/s]48962 examples [00:47, 996.71 examples/s] 49070 examples [00:47, 1019.76 examples/s]49175 examples [00:47, 1026.27 examples/s]49278 examples [00:47, 1022.78 examples/s]49383 examples [00:47, 1028.60 examples/s]49486 examples [00:48, 998.45 examples/s] 49594 examples [00:48, 1020.80 examples/s]49697 examples [00:48, 1005.61 examples/s]49806 examples [00:48, 1027.34 examples/s]49914 examples [00:48, 1042.38 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 10%|█         | 5082/50000 [00:00<00:00, 50818.84 examples/s] 36%|███▋      | 18126/50000 [00:00<00:00, 62210.88 examples/s] 62%|██████▏   | 30777/50000 [00:00<00:00, 73402.86 examples/s] 87%|████████▋ | 43499/50000 [00:00<00:00, 84071.97 examples/s]                                                               0 examples [00:00, ? examples/s]85 examples [00:00, 847.70 examples/s]177 examples [00:00, 866.24 examples/s]285 examples [00:00, 919.10 examples/s]396 examples [00:00, 968.51 examples/s]503 examples [00:00, 995.24 examples/s]605 examples [00:00, 1002.10 examples/s]711 examples [00:00, 1017.86 examples/s]813 examples [00:00, 1018.42 examples/s]922 examples [00:00, 1034.07 examples/s]1027 examples [00:01, 1038.11 examples/s]1137 examples [00:01, 1054.67 examples/s]1242 examples [00:01, 1037.71 examples/s]1350 examples [00:01, 1048.44 examples/s]1460 examples [00:01, 1061.92 examples/s]1573 examples [00:01, 1079.13 examples/s]1681 examples [00:01, 1059.01 examples/s]1788 examples [00:01, 1060.57 examples/s]1896 examples [00:01, 1065.16 examples/s]2003 examples [00:01, 1045.45 examples/s]2110 examples [00:02, 1051.36 examples/s]2216 examples [00:02, 1025.44 examples/s]2319 examples [00:02, 999.40 examples/s] 2428 examples [00:02, 1023.69 examples/s]2532 examples [00:02, 1027.05 examples/s]2635 examples [00:02, 1007.89 examples/s]2739 examples [00:02, 1015.25 examples/s]2841 examples [00:02, 981.77 examples/s] 2940 examples [00:02, 965.74 examples/s]3049 examples [00:02, 999.13 examples/s]3154 examples [00:03, 1011.60 examples/s]3260 examples [00:03, 1023.99 examples/s]3367 examples [00:03, 1035.63 examples/s]3477 examples [00:03, 1052.14 examples/s]3586 examples [00:03, 1061.85 examples/s]3693 examples [00:03, 1049.62 examples/s]3806 examples [00:03, 1071.01 examples/s]3915 examples [00:03, 1074.06 examples/s]4028 examples [00:03, 1088.36 examples/s]4137 examples [00:03, 1071.62 examples/s]4245 examples [00:04, 1046.69 examples/s]4355 examples [00:04, 1061.79 examples/s]4466 examples [00:04, 1073.16 examples/s]4574 examples [00:04, 1049.07 examples/s]4680 examples [00:04, 1045.44 examples/s]4785 examples [00:04, 1010.64 examples/s]4895 examples [00:04, 1033.59 examples/s]5007 examples [00:04, 1056.97 examples/s]5115 examples [00:04, 1062.71 examples/s]5224 examples [00:05, 1067.85 examples/s]5331 examples [00:05, 1051.19 examples/s]5438 examples [00:05, 1055.01 examples/s]5544 examples [00:05, 1052.06 examples/s]5651 examples [00:05, 1054.77 examples/s]5757 examples [00:05, 1021.86 examples/s]5868 examples [00:05, 1044.80 examples/s]5977 examples [00:05, 1057.70 examples/s]6087 examples [00:05, 1069.20 examples/s]6195 examples [00:05, 1054.67 examples/s]6302 examples [00:06, 1056.44 examples/s]6412 examples [00:06, 1067.05 examples/s]6522 examples [00:06, 1074.68 examples/s]6630 examples [00:06, 1059.96 examples/s]6738 examples [00:06, 1064.21 examples/s]6845 examples [00:06, 1057.09 examples/s]6951 examples [00:06, 1043.39 examples/s]7056 examples [00:06, 1044.34 examples/s]7167 examples [00:06, 1061.41 examples/s]7274 examples [00:06, 1056.23 examples/s]7385 examples [00:07, 1070.74 examples/s]7494 examples [00:07, 1074.60 examples/s]7605 examples [00:07, 1083.36 examples/s]7714 examples [00:07, 1069.49 examples/s]7822 examples [00:07, 1044.96 examples/s]7932 examples [00:07, 1054.30 examples/s]8038 examples [00:07, 1005.63 examples/s]8140 examples [00:07, 996.07 examples/s] 8246 examples [00:07, 1014.29 examples/s]8350 examples [00:08, 1020.06 examples/s]8458 examples [00:08, 1036.31 examples/s]8563 examples [00:08, 1040.08 examples/s]8673 examples [00:08, 1055.74 examples/s]8781 examples [00:08, 1061.71 examples/s]8888 examples [00:08, 1032.45 examples/s]8994 examples [00:08, 1038.96 examples/s]9104 examples [00:08, 1056.43 examples/s]9214 examples [00:08, 1066.66 examples/s]9321 examples [00:08, 1037.94 examples/s]9428 examples [00:09, 1045.64 examples/s]9540 examples [00:09, 1064.98 examples/s]9647 examples [00:09, 1062.66 examples/s]9757 examples [00:09, 1072.07 examples/s]9869 examples [00:09, 1083.76 examples/s]9978 examples [00:09, 1062.76 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteGKPUKB/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteGKPUKB/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f433ce3d620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f433ce3d620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f433ce3d620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f42c62a8278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f42c62dec50>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f433ce3d620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f433ce3d620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f433ce3d620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4323e6f588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4323e6f320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f42b48c6268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f42b48c6268> 

  function with postional parmater data_info <function split_train_valid at 0x7f42b48c6268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f42b48c6378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f42b48c6378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f42b48c6378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=a1427900cf9c13b50c51262336b948f29dce24b206afff2c1b9433872928b671
  Stored in directory: /tmp/pip-ephem-wheel-cache-vb3lc9j3/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<17:11:01, 13.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<12:15:43, 19.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<8:38:15, 27.7kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:00<6:03:20, 39.5kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:13:44, 56.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.58M/862M [00:01<2:56:27, 80.5kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:01<2:03:14, 115kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:01<1:25:51, 164kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.3M/862M [00:01<59:56, 234kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 25.6M/862M [00:01<41:50, 333kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:01<29:14, 474kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.5M/862M [00:01<20:27, 675kB/s].vector_cache/glove.6B.zip:   5%|▍         | 38.8M/862M [00:01<14:20, 957kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.4M/862M [00:02<10:04, 1.35MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.5M/862M [00:02<07:24, 1.84MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.3M/862M [00:02<05:13, 2.59MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<04:09, 3.25MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<04:48, 2.79MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<07:01, 1.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<05:49, 2.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.7M/862M [00:04<04:16, 3.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:06<08:35, 1.56MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<07:33, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:06<05:36, 2.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<06:44, 1.97MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<07:33, 1.76MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<05:54, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.2M/862M [00:08<04:18, 3.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<09:48, 1.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<08:13, 1.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<06:04, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<07:18, 1.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<07:46, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<06:05, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<06:22, 2.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<07:07, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<05:33, 2.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.6M/862M [00:14<04:03, 3.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<09:47, 1.33MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<08:12, 1.59MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:16<06:03, 2.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<07:32, 1.72MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<08:09, 1.59MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<06:19, 2.05MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:18<04:37, 2.80MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<07:49, 1.65MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<06:56, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.5M/862M [00:20<05:12, 2.47MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<06:23, 2.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<05:54, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<04:26, 2.88MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<05:50, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<06:53, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<05:26, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<03:57, 3.20MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<08:49, 1.44MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:36, 1.67MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:40, 2.23MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:39, 1.89MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:26, 1.69MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:55, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<04:17, 2.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<13:34, 924kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<10:54, 1.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<07:55, 1.58MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<09:07, 1.37MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:36, 1.63MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:56, 2.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<04:20, 2.85MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:10, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:43, 1.60MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:01, 2.05MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<04:20, 2.83MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<11:07, 1.10MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<09:10, 1.34MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:42, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<07:18, 1.67MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:48, 1.56MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:02, 2.01MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<04:24, 2.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:41, 1.58MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:46, 1.79MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:01, 2.41MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:06, 1.97MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:06, 1.70MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:38, 2.13MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<04:05, 2.94MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<09:38, 1.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:06, 1.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:57, 2.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:43, 1.77MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:49, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:24, 2.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:37, 2.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:39, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:14, 2.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<03:50, 3.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:48, 1.73MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:53, 2.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:27, 2.64MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:36, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:37, 1.76MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:12, 2.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<03:46, 3.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<09:14, 1.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:46, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<05:42, 2.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:28, 1.79MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<07:04, 1.63MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:31, 2.09MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<03:59, 2.88MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<09:30, 1.21MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:00<07:58, 1.44MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:53, 1.95MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:33, 1.74MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:13, 1.58MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:36, 2.03MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<04:03, 2.81MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<09:39, 1.18MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:03, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:56, 1.91MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:33, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:53, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:26, 2.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:30, 2.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:21, 1.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:00, 2.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:38, 3.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<07:41, 1.45MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:37, 1.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:54, 2.27MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:48, 1.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<06:40, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:14, 2.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:51, 2.87MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:03, 1.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:28, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<04:06, 2.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:11, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:04, 1.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<04:46, 2.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<03:29, 3.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<07:02, 1.54MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:08, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<04:33, 2.38MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:32, 1.95MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:23, 1.69MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:59, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<03:39, 2.94MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:05, 1.76MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:28, 1.96MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<04:07, 2.59MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:09, 2.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:57, 1.79MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:41, 2.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<03:24, 3.11MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<07:49, 1.35MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:40, 1.59MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:55, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:41, 1.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:27, 1.63MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:01, 2.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<03:39, 2.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:38, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:44, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<04:17, 2.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:19, 1.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:00, 1.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:43, 2.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<03:24, 3.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<09:46, 1.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<08:01, 1.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:51, 1.76MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:17, 1.63MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:40, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:09, 1.98MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:46, 2.70MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:50, 1.74MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:15, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<03:57, 2.57MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:55, 2.05MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:30, 2.24MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<03:24, 2.95MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:39, 2.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:28, 1.83MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:20, 2.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:11, 3.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:22, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:49, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:35, 2.77MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:44, 2.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:36, 1.77MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:25, 2.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<03:13, 3.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<11:17, 871kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<08:49, 1.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<06:26, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:52, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<06:52, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:14, 1.86MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<03:47, 2.57MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<07:23, 1.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<06:16, 1.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<10:22, 934kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<07:28, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<07:10, 1.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<06:07, 1.57MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:30, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<05:14, 1.82MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<05:50, 1.64MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:33, 2.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<03:24, 2.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<04:41, 2.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:29, 2.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<03:26, 2.75MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [02:00<04:14, 2.22MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:14, 1.79MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:15, 2.21MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:06, 3.01MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:02<08:04, 1.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:43, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:55, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<05:25, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<05:59, 1.55MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:39, 1.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:25, 2.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:03, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<04:31, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:21, 2.72MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<02:28, 3.69MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<30:39, 298kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<23:23, 390kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<16:46, 544kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<11:51, 767kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<11:21, 798kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<08:45, 1.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:21, 1.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:26, 1.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<06:36, 1.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<05:04, 1.77MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:38, 2.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<07:17, 1.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<06:06, 1.46MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:29, 1.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<03:14, 2.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<33:30, 264kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<25:18, 350kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<18:09, 487kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<12:44, 690kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<27:44, 317kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<20:23, 431kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<14:27, 606kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<11:56, 730kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<09:16, 940kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<06:42, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<06:37, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<06:33, 1.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:00, 1.72MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:41, 2.33MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:44, 1.81MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:17, 2.00MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:14, 2.64MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:04, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:48, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<03:50, 2.21MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<02:47, 3.04MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<06:48, 1.24MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<05:43, 1.47MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:11, 2.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:43, 1.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:06, 1.64MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:58, 2.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:57, 2.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:15, 1.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:56, 2.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<02:57, 2.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<02:13, 3.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:31, 1.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:41, 1.44MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:16, 1.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:30, 1.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:58, 1.64MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:56, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<02:51, 2.84MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<09:58, 811kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<07:54, 1.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<05:44, 1.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:44, 1.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:43, 1.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<04:21, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<03:13, 2.48MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:19, 1.84MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:54, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:57, 2.68MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:45, 2.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:27, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:33, 2.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<02:34, 3.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<08:58, 871kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<07:10, 1.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<05:13, 1.49MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:18, 1.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:21, 1.45MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:09, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<02:59, 2.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<07:01, 1.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:46, 1.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<04:15, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:35, 1.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:58, 1.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:54, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:49, 2.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<08:59, 839kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<07:06, 1.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<05:09, 1.46MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:15, 1.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:19, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:05, 1.82MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:00, 2.47MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:00, 1.85MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:39, 2.02MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:45, 2.67MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:31, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:01, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:12, 2.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<02:19, 3.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<16:29, 440kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<12:21, 587kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<08:49, 820kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<07:41, 936kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<06:54, 1.04MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<05:09, 1.39MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:41, 1.93MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:32, 1.28MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:37, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:23, 2.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:56, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:19, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:20, 2.10MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:07<02:25, 2.88MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:53, 1.43MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:05, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:02, 2.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:13, 3.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<06:55, 999kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<06:26, 1.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:49, 1.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<03:27, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<05:10, 1.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:16, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:08, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<02:17, 2.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<07:23, 918kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<05:57, 1.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<04:19, 1.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:27, 1.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:33, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:29, 1.92MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:33, 2.61MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:49, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:24, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:32, 2.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:11, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:38, 1.80MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:50, 2.31MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<02:03, 3.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<05:13, 1.25MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:16, 1.52MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:14, 2.00MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<02:20, 2.76MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<06:36, 972kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<05:21, 1.20MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:53, 1.64MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:04, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:11, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:13, 1.97MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:20, 2.70MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<04:10, 1.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:38, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:47, 2.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:00, 3.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<08:00, 778kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<07:02, 884kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<05:15, 1.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:50, 1.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:57, 1.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:28, 1.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:34, 2.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:06, 1.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:30, 1.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:44, 2.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:01, 2.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:16, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:55, 2.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:12, 2.72MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:51, 2.08MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:18, 1.80MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:37, 2.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<01:54, 3.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<13:18, 442kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<09:59, 588kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<07:07, 820kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<06:12, 937kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<05:00, 1.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:39, 1.58MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:45, 1.52MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:18, 1.74MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:28, 2.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:55, 1.94MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:41, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:02, 2.77MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:37, 2.14MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<03:04, 1.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:27, 2.27MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<01:46, 3.12MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<06:24, 864kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<05:04, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:41, 1.49MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<03:47, 1.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:54, 1.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:02, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<02:11, 2.47MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<06:51, 787kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<05:23, 998kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:54, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:51, 1.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:56, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:00, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:10, 2.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:38, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:08, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:20, 2.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:43, 1.90MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<03:01, 1.71MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:23, 2.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<01:43, 2.98MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<12:06, 422kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<09:02, 565kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<06:26, 789kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<05:33, 908kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<05:04, 995kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:49, 1.32MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<02:43, 1.83MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<06:26, 773kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<04:56, 1.01MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:34, 1.38MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:34, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<03:35, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:44, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<01:58, 2.47MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:50, 1.26MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<03:12, 1.50MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:21, 2.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:42, 1.76MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:26, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:49, 2.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:16, 2.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<02:38, 1.78MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:05, 2.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:31, 3.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:52, 948kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:57, 1.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:52, 1.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<03:03, 1.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<03:20, 1.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:38, 1.72MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:54, 2.37MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:21<02:51, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<02:40, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:00, 2.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:28, 3.00MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<04:56, 896kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<03:52, 1.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:57, 1.49MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:11, 2.00MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:36, 2.71MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<07:31, 579kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<05:50, 745kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<04:11, 1.03MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<03:00, 1.43MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<05:42, 751kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<04:29, 953kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:16, 1.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:09, 1.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:42, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:02, 2.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:28, 2.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<27:29, 151kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<19:45, 209kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<13:53, 296kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<09:42, 421kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<11:10, 365kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<08:14, 493kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<05:49, 694kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<04:56, 811kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<04:22, 914kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:14, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:18, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:09, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<02:35, 1.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:54, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:09, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<02:24, 1.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<01:51, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:20, 2.84MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:37, 1.44MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:15, 1.67MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:40, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:57, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:12, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:43, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:14, 2.98MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:56, 1.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<02:27, 1.48MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:49, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:01, 1.77MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<02:14, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:44, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:14, 2.84MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<03:00, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:30, 1.40MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:49, 1.91MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:00, 1.72MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:48, 1.91MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:20, 2.55MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:38, 2.05MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:53, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:29, 2.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:04, 3.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:09, 1.53MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:52, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:22, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:40, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:55, 1.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:57<01:30, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:06, 2.88MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:35, 1.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:24, 2.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:04, 2.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:24, 2.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:40, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:19, 2.34MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<00:57, 3.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:53, 1.61MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:38, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:12, 2.49MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:30, 1.97MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:40, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:18, 2.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<00:56, 3.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<02:01, 1.43MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:43, 1.67MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:16, 2.25MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:30, 1.87MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:39, 1.71MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:16, 2.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<00:54, 3.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:32, 1.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:05, 1.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:31, 1.79MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:37, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:25, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:02, 2.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:18, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:31, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:11, 2.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:51, 3.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:52, 1.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:33, 1.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:08, 2.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:50, 2.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<02:36, 950kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:23, 1.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:48, 1.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:16, 1.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<03:08, 769kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:26, 982kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:45, 1.36MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:43, 1.36MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:43, 1.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:18, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:56, 2.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:25, 1.59MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:12, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:53, 2.54MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<00:38, 3.44MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<03:05, 712kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:40, 823kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:57, 1.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<01:23, 1.55MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:49, 1.17MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:30, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:06, 1.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:12, 1.71MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:18, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:01, 2.00MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:43, 2.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:18, 863kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:50, 1.08MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:19, 1.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:19, 1.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:06, 1.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:49, 2.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:57, 1.93MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:04, 1.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:49, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:36, 2.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:56, 1.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:48, 2.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:37, 2.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:26, 3.84MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:59, 865kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:47, 958kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:19, 1.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:55, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:18, 1.26MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:05, 1.50MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:48, 2.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:53, 1.78MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:58, 1.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:45, 2.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:32, 2.82MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:46, 848kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:24, 1.06MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:00, 1.47MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:59, 1.44MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:00, 1.41MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:46, 1.84MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:33, 2.51MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:48, 1.68MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:43, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:31, 2.53MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:22, 3.44MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<03:03, 425kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<02:16, 568kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:36, 793kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:20, 912kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<01:12, 1.01MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:54, 1.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:37, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:55, 1.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:46, 1.48MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:33, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:36, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:40, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:31, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:22, 2.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:15, 807kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:59, 1.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:42, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:41, 1.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:34, 1.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:25, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:28, 1.86MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:32, 1.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:24, 2.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:17, 2.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:31, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:27, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:19, 2.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:22, 1.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:20, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:14, 2.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:18, 2.14MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:22, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:17, 2.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:11, 3.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:20, 1.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:18, 1.95MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:13, 2.58MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:15, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:14, 2.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:10, 2.90MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:12, 2.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:11, 2.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:08, 3.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:10, 2.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:12, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:09, 2.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:06, 3.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:23, 826kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:18, 1.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:12, 1.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:10, 1.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:11, 1.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:08, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:04, 2.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:10, 1.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:08, 1.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:05, 1.82MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:04, 1.67MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:04, 1.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 2.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:01, 2.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 1.92MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 2.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.77MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<12:56:11,  8.59it/s]  0%|          | 821/400000 [00:00<9:02:27, 12.26it/s]  0%|          | 1691/400000 [00:00<6:19:07, 17.51it/s]  1%|          | 2572/400000 [00:00<4:25:01, 24.99it/s]  1%|          | 3440/400000 [00:00<3:05:20, 35.66it/s]  1%|          | 4299/400000 [00:00<2:09:41, 50.85it/s]  1%|▏         | 5180/400000 [00:00<1:30:48, 72.47it/s]  2%|▏         | 6051/400000 [00:00<1:03:38, 103.16it/s]  2%|▏         | 6883/400000 [00:00<44:41, 146.59it/s]    2%|▏         | 7742/400000 [00:01<31:26, 207.89it/s]  2%|▏         | 8566/400000 [00:01<22:13, 293.55it/s]  2%|▏         | 9411/400000 [00:01<15:45, 413.20it/s]  3%|▎         | 10265/400000 [00:01<11:13, 578.28it/s]  3%|▎         | 11111/400000 [00:01<08:04, 802.60it/s]  3%|▎         | 11977/400000 [00:01<05:51, 1102.77it/s]  3%|▎         | 12820/400000 [00:01<04:19, 1490.63it/s]  3%|▎         | 13660/400000 [00:01<03:15, 1975.58it/s]  4%|▎         | 14506/400000 [00:01<02:30, 2565.45it/s]  4%|▍         | 15354/400000 [00:01<01:58, 3244.14it/s]  4%|▍         | 16238/400000 [00:02<01:35, 4004.59it/s]  4%|▍         | 17111/400000 [00:02<01:20, 4780.43it/s]  5%|▍         | 18002/400000 [00:02<01:08, 5552.43it/s]  5%|▍         | 18893/400000 [00:02<01:00, 6259.97it/s]  5%|▍         | 19769/400000 [00:02<00:55, 6819.01it/s]  5%|▌         | 20682/400000 [00:02<00:51, 7378.07it/s]  5%|▌         | 21566/400000 [00:02<00:48, 7762.85it/s]  6%|▌         | 22465/400000 [00:02<00:46, 8092.44it/s]  6%|▌         | 23402/400000 [00:02<00:44, 8435.57it/s]  6%|▌         | 24305/400000 [00:02<00:44, 8505.61it/s]  6%|▋         | 25234/400000 [00:03<00:42, 8724.56it/s]  7%|▋         | 26137/400000 [00:03<00:43, 8679.87it/s]  7%|▋         | 27027/400000 [00:03<00:42, 8735.25it/s]  7%|▋         | 27943/400000 [00:03<00:42, 8857.29it/s]  7%|▋         | 28845/400000 [00:03<00:41, 8902.43it/s]  7%|▋         | 29743/400000 [00:03<00:41, 8823.74it/s]  8%|▊         | 30638/400000 [00:03<00:41, 8860.80it/s]  8%|▊         | 31531/400000 [00:03<00:41, 8878.87it/s]  8%|▊         | 32422/400000 [00:03<00:41, 8851.15it/s]  8%|▊         | 33310/400000 [00:03<00:41, 8768.41it/s]  9%|▊         | 34189/400000 [00:04<00:42, 8558.99it/s]  9%|▉         | 35048/400000 [00:04<00:43, 8473.07it/s]  9%|▉         | 35898/400000 [00:04<00:42, 8478.43it/s]  9%|▉         | 36769/400000 [00:04<00:42, 8544.36it/s]  9%|▉         | 37646/400000 [00:04<00:42, 8609.15it/s] 10%|▉         | 38509/400000 [00:04<00:41, 8612.88it/s] 10%|▉         | 39371/400000 [00:04<00:41, 8592.18it/s] 10%|█         | 40238/400000 [00:04<00:41, 8615.34it/s] 10%|█         | 41100/400000 [00:04<00:41, 8547.93it/s] 10%|█         | 41959/400000 [00:04<00:41, 8559.03it/s] 11%|█         | 42818/400000 [00:05<00:41, 8566.11it/s] 11%|█         | 43685/400000 [00:05<00:41, 8594.44it/s] 11%|█         | 44596/400000 [00:05<00:40, 8740.46it/s] 11%|█▏        | 45477/400000 [00:05<00:40, 8758.30it/s] 12%|█▏        | 46440/400000 [00:05<00:39, 9001.59it/s] 12%|█▏        | 47350/400000 [00:05<00:39, 9027.46it/s] 12%|█▏        | 48255/400000 [00:05<00:39, 8994.77it/s] 12%|█▏        | 49178/400000 [00:05<00:38, 9061.96it/s] 13%|█▎        | 50086/400000 [00:05<00:39, 8894.30it/s] 13%|█▎        | 51027/400000 [00:05<00:38, 9042.42it/s] 13%|█▎        | 51933/400000 [00:06<00:38, 9039.34it/s] 13%|█▎        | 52838/400000 [00:06<00:38, 8937.26it/s] 13%|█▎        | 53763/400000 [00:06<00:38, 9026.66it/s] 14%|█▎        | 54710/400000 [00:06<00:37, 9153.07it/s] 14%|█▍        | 55627/400000 [00:06<00:37, 9109.58it/s] 14%|█▍        | 56539/400000 [00:06<00:38, 9035.53it/s] 14%|█▍        | 57444/400000 [00:06<00:38, 8975.91it/s] 15%|█▍        | 58374/400000 [00:06<00:37, 9068.01it/s] 15%|█▍        | 59286/400000 [00:06<00:37, 9082.66it/s] 15%|█▌        | 60209/400000 [00:06<00:37, 9124.99it/s] 15%|█▌        | 61122/400000 [00:07<00:37, 8958.38it/s] 16%|█▌        | 62045/400000 [00:07<00:37, 9035.39it/s] 16%|█▌        | 62988/400000 [00:07<00:36, 9148.88it/s] 16%|█▌        | 63904/400000 [00:07<00:37, 8948.63it/s] 16%|█▌        | 64801/400000 [00:07<00:38, 8745.66it/s] 16%|█▋        | 65687/400000 [00:07<00:38, 8778.65it/s] 17%|█▋        | 66567/400000 [00:07<00:38, 8750.12it/s] 17%|█▋        | 67475/400000 [00:07<00:37, 8844.97it/s] 17%|█▋        | 68369/400000 [00:07<00:37, 8871.86it/s] 17%|█▋        | 69257/400000 [00:08<00:37, 8826.70it/s] 18%|█▊        | 70187/400000 [00:08<00:36, 8961.83it/s] 18%|█▊        | 71098/400000 [00:08<00:36, 9003.69it/s] 18%|█▊        | 72011/400000 [00:08<00:36, 9039.38it/s] 18%|█▊        | 72916/400000 [00:08<00:36, 9031.27it/s] 18%|█▊        | 73820/400000 [00:08<00:36, 8994.76it/s] 19%|█▊        | 74748/400000 [00:08<00:35, 9073.77it/s] 19%|█▉        | 75656/400000 [00:08<00:35, 9059.76it/s] 19%|█▉        | 76591/400000 [00:08<00:35, 9142.78it/s] 19%|█▉        | 77506/400000 [00:08<00:35, 9061.60it/s] 20%|█▉        | 78418/400000 [00:09<00:35, 9076.51it/s] 20%|█▉        | 79326/400000 [00:09<00:35, 9042.22it/s] 20%|██        | 80231/400000 [00:09<00:36, 8861.90it/s] 20%|██        | 81119/400000 [00:09<00:36, 8721.85it/s] 20%|██        | 81993/400000 [00:09<00:36, 8648.22it/s] 21%|██        | 82859/400000 [00:09<00:36, 8581.18it/s] 21%|██        | 83718/400000 [00:09<00:37, 8547.52it/s] 21%|██        | 84574/400000 [00:09<00:37, 8509.66it/s] 21%|██▏       | 85426/400000 [00:09<00:37, 8466.72it/s] 22%|██▏       | 86274/400000 [00:09<00:37, 8440.50it/s] 22%|██▏       | 87142/400000 [00:10<00:36, 8509.88it/s] 22%|██▏       | 88035/400000 [00:10<00:36, 8630.45it/s] 22%|██▏       | 88899/400000 [00:10<00:36, 8622.16it/s] 22%|██▏       | 89784/400000 [00:10<00:35, 8687.55it/s] 23%|██▎       | 90657/400000 [00:10<00:35, 8698.27it/s] 23%|██▎       | 91535/400000 [00:10<00:35, 8721.67it/s] 23%|██▎       | 92408/400000 [00:10<00:35, 8721.10it/s] 23%|██▎       | 93281/400000 [00:10<00:35, 8602.72it/s] 24%|██▎       | 94148/400000 [00:10<00:35, 8622.27it/s] 24%|██▍       | 95011/400000 [00:10<00:35, 8561.93it/s] 24%|██▍       | 95884/400000 [00:11<00:35, 8610.99it/s] 24%|██▍       | 96775/400000 [00:11<00:34, 8697.42it/s] 24%|██▍       | 97646/400000 [00:11<00:35, 8609.08it/s] 25%|██▍       | 98515/400000 [00:11<00:34, 8631.23it/s] 25%|██▍       | 99379/400000 [00:11<00:35, 8440.14it/s] 25%|██▌       | 100245/400000 [00:11<00:35, 8502.72it/s] 25%|██▌       | 101111/400000 [00:11<00:34, 8548.85it/s] 25%|██▌       | 101967/400000 [00:11<00:35, 8421.09it/s] 26%|██▌       | 102811/400000 [00:11<00:36, 8223.43it/s] 26%|██▌       | 103637/400000 [00:11<00:35, 8234.04it/s] 26%|██▌       | 104508/400000 [00:12<00:35, 8370.00it/s] 26%|██▋       | 105365/400000 [00:12<00:34, 8427.79it/s] 27%|██▋       | 106209/400000 [00:12<00:35, 8345.04it/s] 27%|██▋       | 107059/400000 [00:12<00:34, 8388.76it/s] 27%|██▋       | 107899/400000 [00:12<00:35, 8249.34it/s] 27%|██▋       | 108786/400000 [00:12<00:34, 8425.89it/s] 27%|██▋       | 109678/400000 [00:12<00:33, 8567.53it/s] 28%|██▊       | 110558/400000 [00:12<00:33, 8602.59it/s] 28%|██▊       | 111420/400000 [00:12<00:33, 8584.99it/s] 28%|██▊       | 112303/400000 [00:12<00:33, 8653.28it/s] 28%|██▊       | 113191/400000 [00:13<00:32, 8717.25it/s] 29%|██▊       | 114064/400000 [00:13<00:32, 8683.62it/s] 29%|██▊       | 114933/400000 [00:13<00:33, 8612.40it/s] 29%|██▉       | 115804/400000 [00:13<00:32, 8641.12it/s] 29%|██▉       | 116669/400000 [00:13<00:32, 8636.70it/s] 29%|██▉       | 117533/400000 [00:13<00:32, 8634.04it/s] 30%|██▉       | 118402/400000 [00:13<00:32, 8649.29it/s] 30%|██▉       | 119268/400000 [00:13<00:32, 8623.24it/s] 30%|███       | 120156/400000 [00:13<00:32, 8696.83it/s] 30%|███       | 121026/400000 [00:13<00:32, 8635.20it/s] 30%|███       | 121890/400000 [00:14<00:32, 8602.08it/s] 31%|███       | 122751/400000 [00:14<00:32, 8520.66it/s] 31%|███       | 123605/400000 [00:14<00:32, 8524.84it/s] 31%|███       | 124479/400000 [00:14<00:32, 8587.44it/s] 31%|███▏      | 125339/400000 [00:14<00:32, 8574.61it/s] 32%|███▏      | 126211/400000 [00:14<00:31, 8615.80it/s] 32%|███▏      | 127105/400000 [00:14<00:31, 8709.52it/s] 32%|███▏      | 127977/400000 [00:14<00:31, 8539.09it/s] 32%|███▏      | 128849/400000 [00:14<00:31, 8587.85it/s] 32%|███▏      | 129709/400000 [00:15<00:31, 8570.01it/s] 33%|███▎      | 130593/400000 [00:15<00:31, 8648.75it/s] 33%|███▎      | 131478/400000 [00:15<00:30, 8706.46it/s] 33%|███▎      | 132350/400000 [00:15<00:31, 8623.04it/s] 33%|███▎      | 133213/400000 [00:15<00:31, 8550.53it/s] 34%|███▎      | 134069/400000 [00:15<00:31, 8484.78it/s] 34%|███▎      | 134958/400000 [00:15<00:30, 8601.44it/s] 34%|███▍      | 135855/400000 [00:15<00:30, 8706.53it/s] 34%|███▍      | 136764/400000 [00:15<00:29, 8815.51it/s] 34%|███▍      | 137681/400000 [00:15<00:29, 8918.00it/s] 35%|███▍      | 138598/400000 [00:16<00:29, 8991.09it/s] 35%|███▍      | 139498/400000 [00:16<00:28, 8989.66it/s] 35%|███▌      | 140398/400000 [00:16<00:28, 8970.70it/s] 35%|███▌      | 141297/400000 [00:16<00:28, 8974.96it/s] 36%|███▌      | 142195/400000 [00:16<00:28, 8921.59it/s] 36%|███▌      | 143088/400000 [00:16<00:28, 8899.86it/s] 36%|███▌      | 143997/400000 [00:16<00:28, 8955.99it/s] 36%|███▌      | 144893/400000 [00:16<00:28, 8938.11it/s] 36%|███▋      | 145788/400000 [00:16<00:28, 8940.84it/s] 37%|███▋      | 146683/400000 [00:16<00:28, 8906.47it/s] 37%|███▋      | 147585/400000 [00:17<00:28, 8939.99it/s] 37%|███▋      | 148490/400000 [00:17<00:28, 8970.19it/s] 37%|███▋      | 149388/400000 [00:17<00:28, 8903.42it/s] 38%|███▊      | 150279/400000 [00:17<00:28, 8788.82it/s] 38%|███▊      | 151159/400000 [00:17<00:28, 8714.07it/s] 38%|███▊      | 152031/400000 [00:17<00:28, 8677.43it/s] 38%|███▊      | 152919/400000 [00:17<00:28, 8735.08it/s] 38%|███▊      | 153794/400000 [00:17<00:28, 8737.95it/s] 39%|███▊      | 154670/400000 [00:17<00:28, 8741.97it/s] 39%|███▉      | 155545/400000 [00:17<00:28, 8703.00it/s] 39%|███▉      | 156447/400000 [00:18<00:27, 8792.68it/s] 39%|███▉      | 157327/400000 [00:18<00:27, 8761.92it/s] 40%|███▉      | 158206/400000 [00:18<00:27, 8768.03it/s] 40%|███▉      | 159128/400000 [00:18<00:27, 8897.65it/s] 40%|████      | 160019/400000 [00:18<00:27, 8828.37it/s] 40%|████      | 160924/400000 [00:18<00:26, 8892.05it/s] 40%|████      | 161835/400000 [00:18<00:26, 8954.53it/s] 41%|████      | 162739/400000 [00:18<00:26, 8978.78it/s] 41%|████      | 163645/400000 [00:18<00:26, 9000.18it/s] 41%|████      | 164546/400000 [00:18<00:26, 8762.58it/s] 41%|████▏     | 165424/400000 [00:19<00:26, 8740.38it/s] 42%|████▏     | 166300/400000 [00:19<00:26, 8691.00it/s] 42%|████▏     | 167190/400000 [00:19<00:26, 8750.29it/s] 42%|████▏     | 168066/400000 [00:19<00:26, 8615.65it/s] 42%|████▏     | 168966/400000 [00:19<00:26, 8725.10it/s] 42%|████▏     | 169879/400000 [00:19<00:26, 8840.07it/s] 43%|████▎     | 170765/400000 [00:19<00:25, 8840.26it/s] 43%|████▎     | 171650/400000 [00:19<00:25, 8813.53it/s] 43%|████▎     | 172532/400000 [00:19<00:26, 8625.78it/s] 43%|████▎     | 173396/400000 [00:19<00:26, 8626.89it/s] 44%|████▎     | 174260/400000 [00:20<00:26, 8613.69it/s] 44%|████▍     | 175146/400000 [00:20<00:25, 8685.83it/s] 44%|████▍     | 176033/400000 [00:20<00:25, 8738.02it/s] 44%|████▍     | 176908/400000 [00:20<00:25, 8738.73it/s] 44%|████▍     | 177789/400000 [00:20<00:25, 8757.84it/s] 45%|████▍     | 178666/400000 [00:20<00:25, 8639.68it/s] 45%|████▍     | 179531/400000 [00:20<00:25, 8538.29it/s] 45%|████▌     | 180407/400000 [00:20<00:25, 8603.14it/s] 45%|████▌     | 181268/400000 [00:20<00:25, 8507.99it/s] 46%|████▌     | 182120/400000 [00:20<00:25, 8390.21it/s] 46%|████▌     | 182989/400000 [00:21<00:25, 8474.67it/s] 46%|████▌     | 183843/400000 [00:21<00:25, 8494.12it/s] 46%|████▌     | 184693/400000 [00:21<00:25, 8468.64it/s] 46%|████▋     | 185541/400000 [00:21<00:25, 8444.07it/s] 47%|████▋     | 186410/400000 [00:21<00:25, 8515.12it/s] 47%|████▋     | 187276/400000 [00:21<00:24, 8557.84it/s] 47%|████▋     | 188170/400000 [00:21<00:24, 8667.11it/s] 47%|████▋     | 189084/400000 [00:21<00:23, 8802.85it/s] 47%|████▋     | 189966/400000 [00:21<00:23, 8775.32it/s] 48%|████▊     | 190845/400000 [00:21<00:23, 8752.07it/s] 48%|████▊     | 191721/400000 [00:22<00:24, 8674.21it/s] 48%|████▊     | 192589/400000 [00:22<00:24, 8439.33it/s] 48%|████▊     | 193471/400000 [00:22<00:24, 8548.90it/s] 49%|████▊     | 194328/400000 [00:22<00:24, 8512.01it/s] 49%|████▉     | 195197/400000 [00:22<00:23, 8563.10it/s] 49%|████▉     | 196088/400000 [00:22<00:23, 8663.10it/s] 49%|████▉     | 196965/400000 [00:22<00:23, 8693.52it/s] 49%|████▉     | 197838/400000 [00:22<00:23, 8702.06it/s] 50%|████▉     | 198709/400000 [00:22<00:23, 8643.56it/s] 50%|████▉     | 199574/400000 [00:23<00:23, 8527.55it/s] 50%|█████     | 200428/400000 [00:23<00:24, 8297.79it/s] 50%|█████     | 201290/400000 [00:23<00:23, 8391.05it/s] 51%|█████     | 202131/400000 [00:23<00:23, 8359.84it/s] 51%|█████     | 202993/400000 [00:23<00:23, 8433.45it/s] 51%|█████     | 203866/400000 [00:23<00:23, 8519.40it/s] 51%|█████     | 204719/400000 [00:23<00:23, 8440.78it/s] 51%|█████▏    | 205586/400000 [00:23<00:22, 8505.63it/s] 52%|█████▏    | 206438/400000 [00:23<00:23, 8263.25it/s] 52%|█████▏    | 207286/400000 [00:23<00:23, 8326.33it/s] 52%|█████▏    | 208154/400000 [00:24<00:22, 8426.83it/s] 52%|█████▏    | 209043/400000 [00:24<00:22, 8560.03it/s] 52%|█████▏    | 209930/400000 [00:24<00:21, 8649.18it/s] 53%|█████▎    | 210797/400000 [00:24<00:21, 8621.60it/s] 53%|█████▎    | 211662/400000 [00:24<00:21, 8627.76it/s] 53%|█████▎    | 212528/400000 [00:24<00:21, 8635.07it/s] 53%|█████▎    | 213392/400000 [00:24<00:21, 8604.74it/s] 54%|█████▎    | 214277/400000 [00:24<00:21, 8675.91it/s] 54%|█████▍    | 215150/400000 [00:24<00:21, 8691.00it/s] 54%|█████▍    | 216020/400000 [00:24<00:21, 8536.29it/s] 54%|█████▍    | 216916/400000 [00:25<00:21, 8657.52it/s] 54%|█████▍    | 217783/400000 [00:25<00:21, 8508.76it/s] 55%|█████▍    | 218661/400000 [00:25<00:21, 8585.88it/s] 55%|█████▍    | 219521/400000 [00:25<00:21, 8555.37it/s] 55%|█████▌    | 220380/400000 [00:25<00:20, 8563.19it/s] 55%|█████▌    | 221242/400000 [00:25<00:20, 8579.56it/s] 56%|█████▌    | 222106/400000 [00:25<00:20, 8596.07it/s] 56%|█████▌    | 222966/400000 [00:25<00:20, 8593.91it/s] 56%|█████▌    | 223826/400000 [00:25<00:21, 8247.67it/s] 56%|█████▌    | 224693/400000 [00:25<00:20, 8368.36it/s] 56%|█████▋    | 225582/400000 [00:26<00:20, 8517.40it/s] 57%|█████▋    | 226437/400000 [00:26<00:20, 8522.16it/s] 57%|█████▋    | 227312/400000 [00:26<00:20, 8588.56it/s] 57%|█████▋    | 228225/400000 [00:26<00:19, 8742.73it/s] 57%|█████▋    | 229144/400000 [00:26<00:19, 8871.40it/s] 58%|█████▊    | 230054/400000 [00:26<00:19, 8936.65it/s] 58%|█████▊    | 230959/400000 [00:26<00:18, 8969.82it/s] 58%|█████▊    | 231897/400000 [00:26<00:18, 9086.47it/s] 58%|█████▊    | 232817/400000 [00:26<00:18, 9119.40it/s] 58%|█████▊    | 233759/400000 [00:26<00:18, 9207.14it/s] 59%|█████▊    | 234681/400000 [00:27<00:18, 9183.96it/s] 59%|█████▉    | 235600/400000 [00:27<00:18, 9116.08it/s] 59%|█████▉    | 236513/400000 [00:27<00:18, 8938.34it/s] 59%|█████▉    | 237408/400000 [00:27<00:18, 8898.13it/s] 60%|█████▉    | 238299/400000 [00:27<00:18, 8856.78it/s] 60%|█████▉    | 239186/400000 [00:27<00:18, 8709.99it/s] 60%|██████    | 240084/400000 [00:27<00:18, 8789.22it/s] 60%|██████    | 241002/400000 [00:27<00:17, 8901.81it/s] 60%|██████    | 241896/400000 [00:27<00:17, 8910.97it/s] 61%|██████    | 242795/400000 [00:27<00:17, 8931.96it/s] 61%|██████    | 243689/400000 [00:28<00:17, 8930.84it/s] 61%|██████    | 244583/400000 [00:28<00:17, 8853.25it/s] 61%|██████▏   | 245469/400000 [00:28<00:17, 8801.55it/s] 62%|██████▏   | 246411/400000 [00:28<00:17, 8976.74it/s] 62%|██████▏   | 247310/400000 [00:28<00:17, 8973.25it/s] 62%|██████▏   | 248209/400000 [00:28<00:16, 8977.33it/s] 62%|██████▏   | 249125/400000 [00:28<00:16, 9031.15it/s] 63%|██████▎   | 250069/400000 [00:28<00:16, 9148.28it/s] 63%|██████▎   | 250985/400000 [00:28<00:16, 9014.79it/s] 63%|██████▎   | 251888/400000 [00:29<00:16, 8973.78it/s] 63%|██████▎   | 252787/400000 [00:29<00:16, 8924.14it/s] 63%|██████▎   | 253680/400000 [00:29<00:16, 8877.97it/s] 64%|██████▎   | 254569/400000 [00:29<00:16, 8879.62it/s] 64%|██████▍   | 255498/400000 [00:29<00:16, 8998.75it/s] 64%|██████▍   | 256399/400000 [00:29<00:16, 8960.77it/s] 64%|██████▍   | 257296/400000 [00:29<00:15, 8933.72it/s] 65%|██████▍   | 258190/400000 [00:29<00:15, 8867.09it/s] 65%|██████▍   | 259090/400000 [00:29<00:15, 8906.26it/s] 65%|██████▍   | 259994/400000 [00:29<00:15, 8945.14it/s] 65%|██████▌   | 260889/400000 [00:30<00:15, 8908.78it/s] 65%|██████▌   | 261781/400000 [00:30<00:15, 8895.56it/s] 66%|██████▌   | 262671/400000 [00:30<00:15, 8836.16it/s] 66%|██████▌   | 263566/400000 [00:30<00:15, 8868.17it/s] 66%|██████▌   | 264453/400000 [00:30<00:15, 8830.29it/s] 66%|██████▋   | 265349/400000 [00:30<00:15, 8867.62it/s] 67%|██████▋   | 266236/400000 [00:30<00:15, 8820.02it/s] 67%|██████▋   | 267119/400000 [00:30<00:15, 8727.90it/s] 67%|██████▋   | 268018/400000 [00:30<00:14, 8802.80it/s] 67%|██████▋   | 268916/400000 [00:30<00:14, 8852.89it/s] 67%|██████▋   | 269817/400000 [00:31<00:14, 8897.47it/s] 68%|██████▊   | 270739/400000 [00:31<00:14, 8991.15it/s] 68%|██████▊   | 271639/400000 [00:31<00:14, 8766.00it/s] 68%|██████▊   | 272518/400000 [00:31<00:14, 8702.51it/s] 68%|██████▊   | 273428/400000 [00:31<00:14, 8816.33it/s] 69%|██████▊   | 274311/400000 [00:31<00:14, 8727.92it/s] 69%|██████▉   | 275185/400000 [00:31<00:14, 8664.63it/s] 69%|██████▉   | 276069/400000 [00:31<00:14, 8715.90it/s] 69%|██████▉   | 276943/400000 [00:31<00:14, 8722.68it/s] 69%|██████▉   | 277818/400000 [00:31<00:13, 8730.40it/s] 70%|██████▉   | 278692/400000 [00:32<00:13, 8733.22it/s] 70%|██████▉   | 279566/400000 [00:32<00:13, 8646.10it/s] 70%|███████   | 280431/400000 [00:32<00:13, 8582.72it/s] 70%|███████   | 281318/400000 [00:32<00:13, 8664.60it/s] 71%|███████   | 282189/400000 [00:32<00:13, 8678.03it/s] 71%|███████   | 283058/400000 [00:32<00:13, 8573.62it/s] 71%|███████   | 283924/400000 [00:32<00:13, 8598.91it/s] 71%|███████   | 284815/400000 [00:32<00:13, 8689.13it/s] 71%|███████▏  | 285707/400000 [00:32<00:13, 8756.53it/s] 72%|███████▏  | 286606/400000 [00:32<00:12, 8822.67it/s] 72%|███████▏  | 287524/400000 [00:33<00:12, 8925.05it/s] 72%|███████▏  | 288420/400000 [00:33<00:12, 8934.29it/s] 72%|███████▏  | 289366/400000 [00:33<00:12, 9084.42it/s] 73%|███████▎  | 290327/400000 [00:33<00:11, 9235.59it/s] 73%|███████▎  | 291252/400000 [00:33<00:11, 9238.71it/s] 73%|███████▎  | 292177/400000 [00:33<00:11, 9205.26it/s] 73%|███████▎  | 293155/400000 [00:33<00:11, 9368.94it/s] 74%|███████▎  | 294094/400000 [00:33<00:11, 9355.62it/s] 74%|███████▍  | 295031/400000 [00:33<00:11, 9211.18it/s] 74%|███████▍  | 295954/400000 [00:33<00:11, 9147.43it/s] 74%|███████▍  | 296870/400000 [00:34<00:11, 8949.16it/s] 74%|███████▍  | 297767/400000 [00:34<00:11, 8829.41it/s] 75%|███████▍  | 298654/400000 [00:34<00:11, 8840.93it/s] 75%|███████▍  | 299556/400000 [00:34<00:11, 8893.74it/s] 75%|███████▌  | 300462/400000 [00:34<00:11, 8941.28it/s] 75%|███████▌  | 301363/400000 [00:34<00:11, 8960.33it/s] 76%|███████▌  | 302299/400000 [00:34<00:10, 9076.28it/s] 76%|███████▌  | 303208/400000 [00:34<00:10, 9068.85it/s] 76%|███████▌  | 304153/400000 [00:34<00:10, 9177.63it/s] 76%|███████▋  | 305074/400000 [00:34<00:10, 9185.30it/s] 77%|███████▋  | 306021/400000 [00:35<00:10, 9268.14it/s] 77%|███████▋  | 306949/400000 [00:35<00:10, 9271.40it/s] 77%|███████▋  | 307877/400000 [00:35<00:10, 9142.11it/s] 77%|███████▋  | 308792/400000 [00:35<00:10, 9090.31it/s] 77%|███████▋  | 309702/400000 [00:35<00:10, 8967.52it/s] 78%|███████▊  | 310600/400000 [00:35<00:10, 8869.86it/s] 78%|███████▊  | 311488/400000 [00:35<00:10, 8810.45it/s] 78%|███████▊  | 312370/400000 [00:35<00:09, 8790.71it/s] 78%|███████▊  | 313250/400000 [00:35<00:10, 8535.75it/s] 79%|███████▊  | 314136/400000 [00:36<00:09, 8629.74it/s] 79%|███████▉  | 315001/400000 [00:36<00:09, 8616.21it/s] 79%|███████▉  | 315881/400000 [00:36<00:09, 8668.70it/s] 79%|███████▉  | 316751/400000 [00:36<00:09, 8676.41it/s] 79%|███████▉  | 317620/400000 [00:36<00:09, 8412.35it/s] 80%|███████▉  | 318464/400000 [00:36<00:09, 8393.59it/s] 80%|███████▉  | 319305/400000 [00:36<00:09, 8089.31it/s] 80%|████████  | 320126/400000 [00:36<00:09, 8123.83it/s] 80%|████████  | 320997/400000 [00:36<00:09, 8289.64it/s] 80%|████████  | 321879/400000 [00:36<00:09, 8440.31it/s] 81%|████████  | 322782/400000 [00:37<00:08, 8608.29it/s] 81%|████████  | 323660/400000 [00:37<00:08, 8657.98it/s] 81%|████████  | 324528/400000 [00:37<00:08, 8625.64it/s] 81%|████████▏ | 325392/400000 [00:37<00:08, 8528.70it/s] 82%|████████▏ | 326247/400000 [00:37<00:08, 8450.74it/s] 82%|████████▏ | 327129/400000 [00:37<00:08, 8557.96it/s] 82%|████████▏ | 328043/400000 [00:37<00:08, 8723.83it/s] 82%|████████▏ | 328969/400000 [00:37<00:08, 8875.88it/s] 82%|████████▏ | 329915/400000 [00:37<00:07, 9042.17it/s] 83%|████████▎ | 330822/400000 [00:37<00:07, 8934.92it/s] 83%|████████▎ | 331750/400000 [00:38<00:07, 9031.95it/s] 83%|████████▎ | 332655/400000 [00:38<00:07, 8895.60it/s] 83%|████████▎ | 333547/400000 [00:38<00:07, 8741.70it/s] 84%|████████▎ | 334423/400000 [00:38<00:07, 8616.27it/s] 84%|████████▍ | 335287/400000 [00:38<00:07, 8549.19it/s] 84%|████████▍ | 336144/400000 [00:38<00:07, 8425.22it/s] 84%|████████▍ | 337038/400000 [00:38<00:07, 8571.89it/s] 84%|████████▍ | 337902/400000 [00:38<00:07, 8590.54it/s] 85%|████████▍ | 338782/400000 [00:38<00:07, 8651.29it/s] 85%|████████▍ | 339697/400000 [00:38<00:06, 8793.26it/s] 85%|████████▌ | 340582/400000 [00:39<00:06, 8809.36it/s] 85%|████████▌ | 341464/400000 [00:39<00:06, 8791.19it/s] 86%|████████▌ | 342344/400000 [00:39<00:06, 8788.72it/s] 86%|████████▌ | 343224/400000 [00:39<00:06, 8646.01it/s] 86%|████████▌ | 344090/400000 [00:39<00:06, 8578.61it/s] 86%|████████▋ | 345025/400000 [00:39<00:06, 8795.45it/s] 86%|████████▋ | 345915/400000 [00:39<00:06, 8824.81it/s] 87%|████████▋ | 346805/400000 [00:39<00:06, 8845.33it/s] 87%|████████▋ | 347691/400000 [00:39<00:05, 8761.34it/s] 87%|████████▋ | 348568/400000 [00:39<00:05, 8756.65it/s] 87%|████████▋ | 349456/400000 [00:40<00:05, 8791.26it/s] 88%|████████▊ | 350336/400000 [00:40<00:05, 8739.12it/s] 88%|████████▊ | 351213/400000 [00:40<00:05, 8746.61it/s] 88%|████████▊ | 352088/400000 [00:40<00:05, 8661.01it/s] 88%|████████▊ | 352985/400000 [00:40<00:05, 8749.33it/s] 88%|████████▊ | 353868/400000 [00:40<00:05, 8771.59it/s] 89%|████████▊ | 354775/400000 [00:40<00:05, 8857.73it/s] 89%|████████▉ | 355686/400000 [00:40<00:04, 8931.22it/s] 89%|████████▉ | 356580/400000 [00:40<00:04, 8850.43it/s] 89%|████████▉ | 357501/400000 [00:40<00:04, 8954.23it/s] 90%|████████▉ | 358398/400000 [00:41<00:04, 8949.96it/s] 90%|████████▉ | 359294/400000 [00:41<00:04, 8909.37it/s] 90%|█████████ | 360186/400000 [00:41<00:04, 8804.66it/s] 90%|█████████ | 361067/400000 [00:41<00:04, 8647.10it/s] 90%|█████████ | 361933/400000 [00:41<00:04, 8531.95it/s] 91%|█████████ | 362788/400000 [00:41<00:04, 8410.03it/s] 91%|█████████ | 363631/400000 [00:41<00:04, 8105.93it/s] 91%|█████████ | 364500/400000 [00:41<00:04, 8271.74it/s] 91%|█████████▏| 365414/400000 [00:41<00:04, 8512.60it/s] 92%|█████████▏| 366337/400000 [00:42<00:03, 8714.75it/s] 92%|█████████▏| 367213/400000 [00:42<00:03, 8717.44it/s] 92%|█████████▏| 368110/400000 [00:42<00:03, 8789.62it/s] 92%|█████████▏| 368992/400000 [00:42<00:03, 8742.71it/s] 92%|█████████▏| 369880/400000 [00:42<00:03, 8781.45it/s] 93%|█████████▎| 370760/400000 [00:42<00:03, 8776.55it/s] 93%|█████████▎| 371669/400000 [00:42<00:03, 8867.23it/s] 93%|█████████▎| 372581/400000 [00:42<00:03, 8938.54it/s] 93%|█████████▎| 373530/400000 [00:42<00:02, 9096.09it/s] 94%|█████████▎| 374521/400000 [00:42<00:02, 9324.81it/s] 94%|█████████▍| 375456/400000 [00:43<00:02, 9246.16it/s] 94%|█████████▍| 376383/400000 [00:43<00:02, 9065.93it/s] 94%|█████████▍| 377292/400000 [00:43<00:02, 8809.70it/s] 95%|█████████▍| 378176/400000 [00:43<00:02, 8797.17it/s] 95%|█████████▍| 379088/400000 [00:43<00:02, 8890.97it/s] 95%|█████████▍| 379985/400000 [00:43<00:02, 8912.30it/s] 95%|█████████▌| 380893/400000 [00:43<00:02, 8961.62it/s] 95%|█████████▌| 381791/400000 [00:43<00:02, 8808.47it/s] 96%|█████████▌| 382677/400000 [00:43<00:01, 8822.28it/s] 96%|█████████▌| 383634/400000 [00:43<00:01, 9032.15it/s] 96%|█████████▌| 384580/400000 [00:44<00:01, 9154.68it/s] 96%|█████████▋| 385582/400000 [00:44<00:01, 9396.08it/s] 97%|█████████▋| 386525/400000 [00:44<00:01, 9377.17it/s] 97%|█████████▋| 387497/400000 [00:44<00:01, 9475.91it/s] 97%|█████████▋| 388447/400000 [00:44<00:01, 9306.44it/s] 97%|█████████▋| 389380/400000 [00:44<00:01, 9080.66it/s] 98%|█████████▊| 390291/400000 [00:44<00:01, 8632.07it/s] 98%|█████████▊| 391162/400000 [00:44<00:01, 8653.91it/s] 98%|█████████▊| 392056/400000 [00:44<00:00, 8737.41it/s] 98%|█████████▊| 392941/400000 [00:44<00:00, 8769.74it/s] 98%|█████████▊| 393821/400000 [00:45<00:00, 8732.61it/s] 99%|█████████▊| 394696/400000 [00:45<00:00, 8716.67it/s] 99%|█████████▉| 395569/400000 [00:45<00:00, 8490.01it/s] 99%|█████████▉| 396439/400000 [00:45<00:00, 8550.69it/s] 99%|█████████▉| 397304/400000 [00:45<00:00, 8579.38it/s]100%|█████████▉| 398164/400000 [00:45<00:00, 8528.84it/s]100%|█████████▉| 399018/400000 [00:45<00:00, 8349.75it/s]100%|█████████▉| 399855/400000 [00:45<00:00, 8318.36it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8728.07it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f42b48e1e80>, <torchtext.data.dataset.TabularDataset object at 0x7f42b48e1fd0>, <torchtext.vocab.Vocab object at 0x7f42b48e1ef0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.91 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.91 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.91 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.32 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.32 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.60 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.60 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.07 file/s]2020-07-14 18:19:32.564893: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-14 18:19:32.569097: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-14 18:19:32.569856: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5629842cdd60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-14 18:19:32.569872: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|███       | 3006464/9912422 [00:00<00:00, 29939963.44it/s]9920512it [00:00, 32262515.04it/s]                             
0it [00:00, ?it/s]32768it [00:00, 361768.09it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 461714.80it/s]1654784it [00:00, 11587844.87it/s]                         
0it [00:00, ?it/s]8192it [00:00, 191370.11it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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


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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f7a7e243ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f7a7e243ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f7ae95031e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f7ae95031e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f7b0784be18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f7b0784be18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7a9682c488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7a9682c488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7a9682c488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 143068.41it/s] 76%|███████▋  | 7569408/9912422 [00:00<00:11, 204214.44it/s]9920512it [00:00, 42847066.44it/s]                           
0it [00:00, ?it/s]32768it [00:00, 632429.53it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 462516.56it/s]1654784it [00:00, 11809238.14it/s]                         
0it [00:00, ?it/s]8192it [00:00, 253065.67it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7a7d958390>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7a7d940630>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f7a9682c0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f7a9682c0d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f7a9682c0d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:34,  1.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:34,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:34,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:33,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:33,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:32,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:31,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:31,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:30,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:03,  2.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:03,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:03,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:02,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:02,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:02,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:01,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:01,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:00,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:00,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:42,  3.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:42,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:42,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:41,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:41,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:41,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:41,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:40,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:40,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:40,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:39,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:27,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:27,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:27,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:27,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:27,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:26,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:26,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:26,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:26,  4.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:18,  6.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:18,  6.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:18,  6.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:18,  6.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:18,  6.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:18,  6.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:17,  6.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:17,  6.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:17,  6.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:17,  6.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:12,  9.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:12,  9.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  9.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:12,  9.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:12,  9.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:12,  9.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 22.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 22.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 22.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 22.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 22.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 22.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 22.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 22.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 22.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 22.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 28.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 28.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 28.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 28.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 28.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 28.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 28.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 28.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 28.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 28.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 35.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 35.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 35.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 35.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 35.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 35.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 35.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 35.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 35.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 35.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 43.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 43.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 43.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 43.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 43.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 43.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 43.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 43.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 43.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 43.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 51.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 51.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 51.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 51.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 51.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 51.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 51.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 51.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 51.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 51.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 51.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 59.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 73.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 73.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 73.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 76.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 76.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 76.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 76.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 76.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 76.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 80.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 80.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 80.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.42s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.42s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.42s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.39 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.63s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.42s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 80.39 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.63s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.63s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.96 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.63s/ url]
0 examples [00:00, ? examples/s]2020-06-29 00:11:37.971245: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-29 00:11:38.112680: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-29 00:11:38.113189: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556b8f8996d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-29 00:11:38.113221: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.27 examples/s]102 examples [00:00,  4.66 examples/s]205 examples [00:00,  6.64 examples/s]308 examples [00:00,  9.46 examples/s]408 examples [00:00, 13.47 examples/s]506 examples [00:00, 19.12 examples/s]601 examples [00:00, 27.08 examples/s]706 examples [00:01, 38.27 examples/s]812 examples [00:01, 53.82 examples/s]916 examples [00:01, 75.22 examples/s]1018 examples [00:01, 104.15 examples/s]1121 examples [00:01, 142.60 examples/s]1224 examples [00:01, 192.26 examples/s]1329 examples [00:01, 254.66 examples/s]1432 examples [00:01, 322.35 examples/s]1538 examples [00:01, 407.16 examples/s]1638 examples [00:01, 494.11 examples/s]1738 examples [00:02, 581.50 examples/s]1842 examples [00:02, 669.47 examples/s]1943 examples [00:02, 740.20 examples/s]2043 examples [00:02, 779.49 examples/s]2142 examples [00:02, 831.43 examples/s]2244 examples [00:02, 879.85 examples/s]2343 examples [00:02, 878.77 examples/s]2440 examples [00:02, 902.88 examples/s]2536 examples [00:02, 914.17 examples/s]2639 examples [00:02, 945.30 examples/s]2742 examples [00:03, 968.00 examples/s]2841 examples [00:03, 972.56 examples/s]2945 examples [00:03, 990.53 examples/s]3051 examples [00:03, 1009.38 examples/s]3159 examples [00:03, 1028.19 examples/s]3267 examples [00:03, 1041.12 examples/s]3374 examples [00:03, 1048.72 examples/s]3480 examples [00:03, 1046.98 examples/s]3587 examples [00:03, 1052.91 examples/s]3694 examples [00:03, 1056.76 examples/s]3800 examples [00:04, 1053.00 examples/s]3906 examples [00:04, 1035.10 examples/s]4010 examples [00:04, 1032.70 examples/s]4115 examples [00:04, 1035.71 examples/s]4219 examples [00:04, 1012.40 examples/s]4321 examples [00:04, 1004.14 examples/s]4422 examples [00:04, 931.30 examples/s] 4518 examples [00:04, 939.64 examples/s]4624 examples [00:04, 970.70 examples/s]4730 examples [00:05, 993.28 examples/s]4831 examples [00:05, 966.18 examples/s]4938 examples [00:05, 992.69 examples/s]5038 examples [00:05, 966.31 examples/s]5136 examples [00:05, 968.20 examples/s]5239 examples [00:05, 984.46 examples/s]5344 examples [00:05, 1000.77 examples/s]5449 examples [00:05, 1012.46 examples/s]5556 examples [00:05, 1027.27 examples/s]5663 examples [00:05, 1036.98 examples/s]5767 examples [00:06, 1030.25 examples/s]5871 examples [00:06, 1024.21 examples/s]5974 examples [00:06, 1020.50 examples/s]6077 examples [00:06, 1022.39 examples/s]6183 examples [00:06, 1032.48 examples/s]6290 examples [00:06, 1041.15 examples/s]6395 examples [00:06, 1032.42 examples/s]6499 examples [00:06, 977.67 examples/s] 6598 examples [00:06, 973.03 examples/s]6701 examples [00:06, 988.62 examples/s]6805 examples [00:07, 1003.11 examples/s]6910 examples [00:07, 1016.29 examples/s]7012 examples [00:07, 944.81 examples/s] 7114 examples [00:07, 964.20 examples/s]7219 examples [00:07, 987.09 examples/s]7324 examples [00:07, 1004.21 examples/s]7426 examples [00:07, 1008.81 examples/s]7528 examples [00:07, 969.42 examples/s] 7631 examples [00:07, 985.09 examples/s]7733 examples [00:08, 994.83 examples/s]7838 examples [00:08, 1008.63 examples/s]7943 examples [00:08, 1020.38 examples/s]8046 examples [00:08, 1018.35 examples/s]8149 examples [00:08, 974.24 examples/s] 8252 examples [00:08, 987.95 examples/s]8353 examples [00:08, 992.93 examples/s]8453 examples [00:08, 992.70 examples/s]8553 examples [00:08, 973.21 examples/s]8652 examples [00:08, 976.43 examples/s]8756 examples [00:09, 993.40 examples/s]8856 examples [00:09, 987.57 examples/s]8961 examples [00:09, 1004.66 examples/s]9062 examples [00:09, 997.62 examples/s] 9166 examples [00:09, 1007.59 examples/s]9272 examples [00:09, 1021.99 examples/s]9375 examples [00:09, 1024.07 examples/s]9478 examples [00:09, 980.96 examples/s] 9584 examples [00:09, 1000.77 examples/s]9685 examples [00:09, 1003.35 examples/s]9786 examples [00:10, 1004.85 examples/s]9887 examples [00:10, 1001.18 examples/s]9992 examples [00:10, 1014.58 examples/s]10094 examples [00:10, 961.56 examples/s]10194 examples [00:10, 972.49 examples/s]10300 examples [00:10, 996.78 examples/s]10406 examples [00:10, 1013.01 examples/s]10510 examples [00:10, 1020.50 examples/s]10618 examples [00:10, 1037.10 examples/s]10726 examples [00:11, 1048.72 examples/s]10834 examples [00:11, 1056.68 examples/s]10940 examples [00:11, 1056.69 examples/s]11048 examples [00:11, 1062.98 examples/s]11155 examples [00:11, 1057.43 examples/s]11261 examples [00:11, 1025.48 examples/s]11364 examples [00:11, 1020.53 examples/s]11467 examples [00:11, 1016.27 examples/s]11570 examples [00:11, 1020.06 examples/s]11673 examples [00:11, 1004.32 examples/s]11778 examples [00:12, 1016.57 examples/s]11882 examples [00:12, 1020.95 examples/s]11988 examples [00:12, 1031.14 examples/s]12097 examples [00:12, 1045.35 examples/s]12205 examples [00:12, 1054.70 examples/s]12312 examples [00:12, 1058.90 examples/s]12421 examples [00:12, 1065.76 examples/s]12528 examples [00:12, 1055.74 examples/s]12637 examples [00:12, 1064.26 examples/s]12744 examples [00:12, 1059.77 examples/s]12851 examples [00:13, 1054.29 examples/s]12957 examples [00:13, 1046.86 examples/s]13062 examples [00:13, 1037.54 examples/s]13166 examples [00:13, 1026.88 examples/s]13269 examples [00:13, 1026.72 examples/s]13374 examples [00:13, 1031.14 examples/s]13478 examples [00:13, 1007.65 examples/s]13579 examples [00:13, 990.51 examples/s] 13679 examples [00:13, 975.81 examples/s]13780 examples [00:13, 983.61 examples/s]13883 examples [00:14, 996.19 examples/s]13983 examples [00:14, 997.10 examples/s]14086 examples [00:14, 1006.53 examples/s]14189 examples [00:14, 1012.08 examples/s]14291 examples [00:14, 970.05 examples/s] 14391 examples [00:14, 977.53 examples/s]14493 examples [00:14, 989.87 examples/s]14595 examples [00:14, 997.78 examples/s]14695 examples [00:14, 954.15 examples/s]14798 examples [00:15, 974.99 examples/s]14901 examples [00:15, 990.33 examples/s]15002 examples [00:15, 992.13 examples/s]15102 examples [00:15, 994.20 examples/s]15204 examples [00:15, 1001.23 examples/s]15308 examples [00:15, 1012.14 examples/s]15410 examples [00:15, 1014.30 examples/s]15512 examples [00:15, 1005.93 examples/s]15615 examples [00:15, 1012.82 examples/s]15717 examples [00:15, 960.29 examples/s] 15823 examples [00:16, 985.79 examples/s]15923 examples [00:16, 984.63 examples/s]16026 examples [00:16, 996.80 examples/s]16127 examples [00:16, 1000.33 examples/s]16232 examples [00:16, 1013.55 examples/s]16336 examples [00:16, 1021.04 examples/s]16439 examples [00:16, 1022.34 examples/s]16542 examples [00:16, 1018.26 examples/s]16644 examples [00:16, 1016.31 examples/s]16746 examples [00:16, 954.63 examples/s] 16851 examples [00:17, 979.75 examples/s]16950 examples [00:17, 978.35 examples/s]17051 examples [00:17, 986.36 examples/s]17155 examples [00:17, 1001.67 examples/s]17256 examples [00:17, 987.54 examples/s] 17356 examples [00:17, 958.38 examples/s]17459 examples [00:17, 977.44 examples/s]17562 examples [00:17, 990.92 examples/s]17662 examples [00:17, 989.64 examples/s]17762 examples [00:18, 933.32 examples/s]17861 examples [00:18, 940.66 examples/s]17956 examples [00:18, 935.82 examples/s]18054 examples [00:18, 947.43 examples/s]18158 examples [00:18, 971.37 examples/s]18256 examples [00:18, 969.35 examples/s]18361 examples [00:18, 991.94 examples/s]18466 examples [00:18, 1008.07 examples/s]18568 examples [00:18, 1001.96 examples/s]18669 examples [00:18, 1002.70 examples/s]18770 examples [00:19, 996.06 examples/s] 18870 examples [00:19, 992.85 examples/s]18970 examples [00:19, 990.06 examples/s]19074 examples [00:19, 1002.82 examples/s]19181 examples [00:19, 1020.89 examples/s]19287 examples [00:19, 1032.00 examples/s]19391 examples [00:19, 1020.63 examples/s]19494 examples [00:19, 1020.45 examples/s]19597 examples [00:19, 1016.71 examples/s]19701 examples [00:19, 1023.11 examples/s]19804 examples [00:20, 1008.49 examples/s]19905 examples [00:20, 998.67 examples/s] 20005 examples [00:20, 946.13 examples/s]20105 examples [00:20, 959.48 examples/s]20202 examples [00:20, 955.32 examples/s]20302 examples [00:20, 966.62 examples/s]20399 examples [00:20, 947.84 examples/s]20501 examples [00:20, 966.34 examples/s]20600 examples [00:20, 971.97 examples/s]20702 examples [00:20, 959.57 examples/s]20805 examples [00:21, 976.96 examples/s]20903 examples [00:21, 974.66 examples/s]21004 examples [00:21, 984.21 examples/s]21105 examples [00:21, 991.29 examples/s]21209 examples [00:21, 1003.25 examples/s]21311 examples [00:21, 1005.93 examples/s]21415 examples [00:21, 1013.66 examples/s]21517 examples [00:21, 1011.43 examples/s]21619 examples [00:21, 1012.71 examples/s]21721 examples [00:22, 960.85 examples/s] 21819 examples [00:22, 964.38 examples/s]21916 examples [00:22, 956.78 examples/s]22015 examples [00:22, 965.35 examples/s]22119 examples [00:22, 983.76 examples/s]22218 examples [00:22, 979.34 examples/s]22317 examples [00:22, 947.92 examples/s]22420 examples [00:22, 968.90 examples/s]22522 examples [00:22, 982.72 examples/s]22624 examples [00:22, 991.05 examples/s]22724 examples [00:23, 959.74 examples/s]22821 examples [00:23, 959.05 examples/s]22921 examples [00:23, 968.63 examples/s]23023 examples [00:23, 983.15 examples/s]23125 examples [00:23, 993.31 examples/s]23230 examples [00:23, 1006.79 examples/s]23331 examples [00:23, 1002.18 examples/s]23434 examples [00:23, 1009.08 examples/s]23539 examples [00:23, 1019.83 examples/s]23643 examples [00:23, 1024.61 examples/s]23746 examples [00:24, 1003.44 examples/s]23850 examples [00:24, 1013.10 examples/s]23954 examples [00:24, 1019.30 examples/s]24057 examples [00:24, 1018.81 examples/s]24159 examples [00:24, 1015.17 examples/s]24263 examples [00:24, 1021.28 examples/s]24366 examples [00:24, 1001.66 examples/s]24467 examples [00:24, 1003.61 examples/s]24571 examples [00:24, 1014.19 examples/s]24673 examples [00:24, 1009.50 examples/s]24775 examples [00:25, 1011.78 examples/s]24877 examples [00:25, 1012.36 examples/s]24979 examples [00:25, 1011.67 examples/s]25083 examples [00:25, 1017.12 examples/s]25185 examples [00:25, 1013.07 examples/s]25287 examples [00:25, 1013.67 examples/s]25390 examples [00:25, 1017.48 examples/s]25496 examples [00:25, 1028.42 examples/s]25600 examples [00:25, 1031.43 examples/s]25704 examples [00:25, 975.04 examples/s] 25803 examples [00:26, 905.60 examples/s]25906 examples [00:26, 938.51 examples/s]26005 examples [00:26, 950.99 examples/s]26106 examples [00:26, 967.65 examples/s]26207 examples [00:26, 977.90 examples/s]26312 examples [00:26, 995.94 examples/s]26413 examples [00:26, 971.26 examples/s]26517 examples [00:26, 990.50 examples/s]26617 examples [00:26, 992.09 examples/s]26719 examples [00:27, 997.80 examples/s]26819 examples [00:27, 942.23 examples/s]26920 examples [00:27, 958.96 examples/s]27022 examples [00:27, 976.46 examples/s]27121 examples [00:27, 978.67 examples/s]27220 examples [00:27, 979.47 examples/s]27319 examples [00:27, 981.08 examples/s]27419 examples [00:27, 984.17 examples/s]27521 examples [00:27, 992.31 examples/s]27622 examples [00:27, 996.34 examples/s]27723 examples [00:28, 997.79 examples/s]27823 examples [00:28, 940.29 examples/s]27926 examples [00:28, 965.43 examples/s]28025 examples [00:28, 970.30 examples/s]28124 examples [00:28, 975.46 examples/s]28227 examples [00:28, 988.33 examples/s]28329 examples [00:28, 996.36 examples/s]28432 examples [00:28, 1003.62 examples/s]28536 examples [00:28, 1014.26 examples/s]28638 examples [00:28, 1012.31 examples/s]28740 examples [00:29, 1008.90 examples/s]28841 examples [00:29, 989.68 examples/s] 28944 examples [00:29, 1001.14 examples/s]29047 examples [00:29, 1009.08 examples/s]29149 examples [00:29, 985.14 examples/s] 29248 examples [00:29, 981.90 examples/s]29347 examples [00:29, 984.10 examples/s]29446 examples [00:29, 952.57 examples/s]29546 examples [00:29, 965.30 examples/s]29646 examples [00:30, 974.79 examples/s]29744 examples [00:30, 954.55 examples/s]29840 examples [00:30, 905.00 examples/s]29936 examples [00:30, 920.69 examples/s]30029 examples [00:30, 892.03 examples/s]30129 examples [00:30, 920.93 examples/s]30229 examples [00:30, 941.97 examples/s]30329 examples [00:30, 957.40 examples/s]30431 examples [00:30, 975.05 examples/s]30535 examples [00:30, 991.68 examples/s]30635 examples [00:31, 981.10 examples/s]30734 examples [00:31, 949.80 examples/s]30837 examples [00:31, 972.14 examples/s]30941 examples [00:31, 990.77 examples/s]31043 examples [00:31, 997.41 examples/s]31144 examples [00:31, 989.11 examples/s]31245 examples [00:31, 994.36 examples/s]31348 examples [00:31, 1003.67 examples/s]31449 examples [00:31, 1001.89 examples/s]31550 examples [00:31, 990.62 examples/s] 31650 examples [00:32, 991.70 examples/s]31750 examples [00:32, 985.92 examples/s]31855 examples [00:32, 1003.81 examples/s]31959 examples [00:32, 1013.36 examples/s]32061 examples [00:32, 1011.64 examples/s]32165 examples [00:32, 1018.78 examples/s]32269 examples [00:32, 1022.64 examples/s]32372 examples [00:32, 1023.88 examples/s]32475 examples [00:32, 1023.48 examples/s]32578 examples [00:33, 973.82 examples/s] 32676 examples [00:33, 921.75 examples/s]32770 examples [00:33, 872.05 examples/s]32865 examples [00:33, 892.44 examples/s]32965 examples [00:33, 921.87 examples/s]33061 examples [00:33, 931.74 examples/s]33161 examples [00:33, 949.31 examples/s]33257 examples [00:33, 949.86 examples/s]33362 examples [00:33, 976.79 examples/s]33465 examples [00:33, 991.58 examples/s]33566 examples [00:34, 996.05 examples/s]33666 examples [00:34, 974.84 examples/s]33764 examples [00:34, 943.04 examples/s]33860 examples [00:34, 945.94 examples/s]33955 examples [00:34, 946.34 examples/s]34050 examples [00:34, 945.49 examples/s]34154 examples [00:34, 971.74 examples/s]34258 examples [00:34, 990.92 examples/s]34358 examples [00:34, 978.88 examples/s]34462 examples [00:34, 995.55 examples/s]34562 examples [00:35, 993.21 examples/s]34664 examples [00:35, 998.22 examples/s]34767 examples [00:35, 1006.90 examples/s]34871 examples [00:35, 1014.78 examples/s]34977 examples [00:35, 1025.36 examples/s]35080 examples [00:35, 1021.06 examples/s]35183 examples [00:35, 1017.59 examples/s]35288 examples [00:35, 1024.67 examples/s]35393 examples [00:35, 1029.78 examples/s]35497 examples [00:35, 1031.27 examples/s]35601 examples [00:36, 998.73 examples/s] 35704 examples [00:36, 1005.63 examples/s]35806 examples [00:36, 1008.29 examples/s]35909 examples [00:36, 1014.02 examples/s]36013 examples [00:36, 1020.81 examples/s]36118 examples [00:36, 1028.65 examples/s]36223 examples [00:36, 1034.03 examples/s]36327 examples [00:36, 1032.36 examples/s]36431 examples [00:36, 1034.41 examples/s]36535 examples [00:37, 1024.00 examples/s]36638 examples [00:37, 1017.40 examples/s]36740 examples [00:37, 1016.78 examples/s]36842 examples [00:37, 949.61 examples/s] 36938 examples [00:37, 950.26 examples/s]37034 examples [00:37, 951.86 examples/s]37134 examples [00:37, 965.73 examples/s]37234 examples [00:37, 974.93 examples/s]37333 examples [00:37, 977.66 examples/s]37431 examples [00:37, 973.73 examples/s]37532 examples [00:38, 983.46 examples/s]37632 examples [00:38, 988.31 examples/s]37731 examples [00:38, 984.68 examples/s]37830 examples [00:38, 942.06 examples/s]37925 examples [00:38, 936.56 examples/s]38028 examples [00:38, 961.73 examples/s]38131 examples [00:38, 980.02 examples/s]38231 examples [00:38, 985.09 examples/s]38336 examples [00:38, 1003.38 examples/s]38440 examples [00:38, 1011.84 examples/s]38542 examples [00:39, 996.59 examples/s] 38642 examples [00:39, 973.15 examples/s]38747 examples [00:39, 993.90 examples/s]38851 examples [00:39, 1006.44 examples/s]38952 examples [00:39, 994.02 examples/s] 39054 examples [00:39, 1001.18 examples/s]39158 examples [00:39, 1011.14 examples/s]39261 examples [00:39, 1015.55 examples/s]39366 examples [00:39, 1024.97 examples/s]39469 examples [00:39, 1026.16 examples/s]39575 examples [00:40, 1034.65 examples/s]39680 examples [00:40, 1037.65 examples/s]39784 examples [00:40, 1030.27 examples/s]39888 examples [00:40, 1012.85 examples/s]39990 examples [00:40, 999.73 examples/s] 40091 examples [00:40, 966.27 examples/s]40194 examples [00:40, 982.09 examples/s]40298 examples [00:40, 996.82 examples/s]40403 examples [00:40, 1011.94 examples/s]40507 examples [00:41, 1018.98 examples/s]40610 examples [00:41, 1020.98 examples/s]40713 examples [00:41, 1023.13 examples/s]40816 examples [00:41, 1008.76 examples/s]40919 examples [00:41, 1014.63 examples/s]41026 examples [00:41, 1028.00 examples/s]41133 examples [00:41, 1038.84 examples/s]41237 examples [00:41, 1037.60 examples/s]41342 examples [00:41, 1040.05 examples/s]41447 examples [00:41, 1038.71 examples/s]41551 examples [00:42, 1017.97 examples/s]41653 examples [00:42, 983.64 examples/s] 41752 examples [00:42, 985.24 examples/s]41851 examples [00:42, 958.74 examples/s]41951 examples [00:42, 969.01 examples/s]42052 examples [00:42, 980.01 examples/s]42156 examples [00:42, 994.65 examples/s]42256 examples [00:42, 988.67 examples/s]42358 examples [00:42, 997.38 examples/s]42458 examples [00:42, 991.66 examples/s]42562 examples [00:43, 1005.20 examples/s]42663 examples [00:43, 999.92 examples/s] 42764 examples [00:43, 980.79 examples/s]42869 examples [00:43, 1000.45 examples/s]42973 examples [00:43, 1009.12 examples/s]43075 examples [00:43, 1005.14 examples/s]43176 examples [00:43, 1002.67 examples/s]43277 examples [00:43, 999.47 examples/s] 43381 examples [00:43, 1009.80 examples/s]43484 examples [00:43, 1014.67 examples/s]43586 examples [00:44, 1004.22 examples/s]43687 examples [00:44, 1004.78 examples/s]43791 examples [00:44, 1012.45 examples/s]43893 examples [00:44, 983.00 examples/s] 43996 examples [00:44, 995.10 examples/s]44098 examples [00:44, 1000.93 examples/s]44199 examples [00:44, 968.87 examples/s] 44304 examples [00:44, 990.12 examples/s]44408 examples [00:44, 1002.59 examples/s]44516 examples [00:45, 1023.36 examples/s]44619 examples [00:45, 1024.78 examples/s]44722 examples [00:45, 995.39 examples/s] 44826 examples [00:45, 1006.46 examples/s]44927 examples [00:45, 971.16 examples/s] 45030 examples [00:45, 985.25 examples/s]45129 examples [00:45, 985.90 examples/s]45228 examples [00:45, 986.32 examples/s]45332 examples [00:45, 1000.43 examples/s]45433 examples [00:45, 977.58 examples/s] 45537 examples [00:46, 994.92 examples/s]45637 examples [00:46, 990.19 examples/s]45737 examples [00:46, 964.65 examples/s]45841 examples [00:46, 983.38 examples/s]45943 examples [00:46, 992.38 examples/s]46048 examples [00:46, 1007.54 examples/s]46153 examples [00:46, 1017.43 examples/s]46255 examples [00:46, 1006.46 examples/s]46359 examples [00:46, 1014.16 examples/s]46463 examples [00:46, 1012.49 examples/s]46565 examples [00:47, 1006.76 examples/s]46668 examples [00:47, 1011.87 examples/s]46770 examples [00:47, 1008.78 examples/s]46871 examples [00:47, 1004.07 examples/s]46972 examples [00:47, 920.64 examples/s] 47075 examples [00:47, 949.49 examples/s]47175 examples [00:47, 962.31 examples/s]47281 examples [00:47, 989.32 examples/s]47381 examples [00:47, 976.43 examples/s]47489 examples [00:48, 1002.85 examples/s]47591 examples [00:48, 1005.14 examples/s]47695 examples [00:48, 1013.13 examples/s]47797 examples [00:48, 996.24 examples/s] 47902 examples [00:48, 1011.07 examples/s]48004 examples [00:48, 1011.91 examples/s]48111 examples [00:48, 1026.18 examples/s]48214 examples [00:48, 1026.38 examples/s]48317 examples [00:48, 973.50 examples/s] 48421 examples [00:48, 990.44 examples/s]48521 examples [00:49, 970.97 examples/s]48620 examples [00:49, 976.16 examples/s]48722 examples [00:49, 987.37 examples/s]48822 examples [00:49, 990.35 examples/s]48922 examples [00:49, 984.67 examples/s]49021 examples [00:49, 944.02 examples/s]49118 examples [00:49, 949.63 examples/s]49219 examples [00:49, 966.90 examples/s]49316 examples [00:49, 958.48 examples/s]49413 examples [00:49, 959.70 examples/s]49513 examples [00:50, 970.71 examples/s]49611 examples [00:50, 968.41 examples/s]49714 examples [00:50, 984.22 examples/s]49813 examples [00:50, 972.92 examples/s]49913 examples [00:50, 938.71 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6085/50000 [00:00<00:00, 60847.30 examples/s] 37%|███▋      | 18687/50000 [00:00<00:00, 72020.95 examples/s] 63%|██████▎   | 31376/50000 [00:00<00:00, 82755.03 examples/s] 89%|████████▉ | 44620/50000 [00:00<00:00, 93249.50 examples/s]                                                               0 examples [00:00, ? examples/s]74 examples [00:00, 736.36 examples/s]177 examples [00:00, 804.49 examples/s]276 examples [00:00, 852.21 examples/s]377 examples [00:00, 893.38 examples/s]480 examples [00:00, 927.42 examples/s]584 examples [00:00, 956.75 examples/s]689 examples [00:00, 980.41 examples/s]791 examples [00:00, 989.29 examples/s]896 examples [00:00, 1006.42 examples/s]995 examples [00:01, 961.25 examples/s] 1090 examples [00:01, 941.70 examples/s]1184 examples [00:01, 931.50 examples/s]1287 examples [00:01, 958.65 examples/s]1394 examples [00:01, 988.09 examples/s]1502 examples [00:01, 1013.00 examples/s]1604 examples [00:01, 1004.32 examples/s]1707 examples [00:01, 1009.35 examples/s]1809 examples [00:01, 998.64 examples/s] 1909 examples [00:01, 992.52 examples/s]2012 examples [00:02, 1002.19 examples/s]2113 examples [00:02, 976.36 examples/s] 2212 examples [00:02, 980.32 examples/s]2316 examples [00:02, 995.81 examples/s]2420 examples [00:02, 1007.85 examples/s]2521 examples [00:02, 978.27 examples/s] 2626 examples [00:02, 997.57 examples/s]2727 examples [00:02, 989.64 examples/s]2830 examples [00:02, 1000.72 examples/s]2936 examples [00:02, 1016.71 examples/s]3038 examples [00:03, 997.58 examples/s] 3138 examples [00:03, 994.84 examples/s]3239 examples [00:03, 998.60 examples/s]3339 examples [00:03, 988.06 examples/s]3443 examples [00:03, 1002.10 examples/s]3544 examples [00:03, 952.40 examples/s] 3650 examples [00:03, 980.19 examples/s]3758 examples [00:03, 1007.51 examples/s]3860 examples [00:03, 1009.76 examples/s]3966 examples [00:03, 1021.90 examples/s]4069 examples [00:04, 989.57 examples/s] 4174 examples [00:04, 1004.87 examples/s]4276 examples [00:04, 1007.40 examples/s]4378 examples [00:04, 1009.63 examples/s]4483 examples [00:04, 1020.83 examples/s]4586 examples [00:04, 987.35 examples/s] 4686 examples [00:04, 972.81 examples/s]4791 examples [00:04, 992.60 examples/s]4896 examples [00:04, 1006.76 examples/s]5001 examples [00:05, 1018.79 examples/s]5104 examples [00:05, 1013.67 examples/s]5206 examples [00:05, 1013.48 examples/s]5313 examples [00:05, 1028.54 examples/s]5416 examples [00:05, 1022.80 examples/s]5522 examples [00:05, 1033.28 examples/s]5626 examples [00:05, 972.53 examples/s] 5730 examples [00:05, 990.66 examples/s]5833 examples [00:05, 1001.65 examples/s]5934 examples [00:05, 990.11 examples/s] 6039 examples [00:06, 1007.33 examples/s]6144 examples [00:06, 1017.36 examples/s]6251 examples [00:06, 1029.85 examples/s]6355 examples [00:06, 1031.02 examples/s]6459 examples [00:06, 1009.03 examples/s]6563 examples [00:06, 1017.33 examples/s]6665 examples [00:06, 956.33 examples/s] 6769 examples [00:06, 974.15 examples/s]6873 examples [00:06, 991.40 examples/s]6977 examples [00:07, 1003.77 examples/s]7078 examples [00:07, 993.68 examples/s] 7178 examples [00:07, 955.31 examples/s]7278 examples [00:07, 966.57 examples/s]7379 examples [00:07, 977.23 examples/s]7478 examples [00:07, 967.36 examples/s]7575 examples [00:07, 929.46 examples/s]7681 examples [00:07, 963.42 examples/s]7785 examples [00:07, 983.80 examples/s]7886 examples [00:07, 990.66 examples/s]7991 examples [00:08, 1007.68 examples/s]8093 examples [00:08, 1000.89 examples/s]8198 examples [00:08, 1012.48 examples/s]8302 examples [00:08, 1019.83 examples/s]8405 examples [00:08, 1017.78 examples/s]8507 examples [00:08, 1007.95 examples/s]8608 examples [00:08, 948.95 examples/s] 8713 examples [00:08, 975.07 examples/s]8814 examples [00:08, 984.71 examples/s]8914 examples [00:08, 988.86 examples/s]9014 examples [00:09, 960.43 examples/s]9119 examples [00:09, 983.65 examples/s]9222 examples [00:09, 996.80 examples/s]9323 examples [00:09, 996.87 examples/s]9425 examples [00:09, 1000.21 examples/s]9527 examples [00:09, 1004.62 examples/s]9628 examples [00:09, 939.31 examples/s] 9729 examples [00:09, 957.34 examples/s]9833 examples [00:09, 979.37 examples/s]9936 examples [00:10, 992.40 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteTYHKMW/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteTYHKMW/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7a9682c488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7a9682c488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7a9682c488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7a7d9d75c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7a20a54240>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7a9682c488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7a9682c488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7a9682c488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7a20a84630>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7a7d9400b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f7a0e6021e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f7a0e6021e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f7a0e6021e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f7a0e6022f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f7a0e6022f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f7a0e6022f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=5647e354d87047c6d0e411e06d15417c3fb6dbda21d98a8a82f2d1d59ebc7497
  Stored in directory: /tmp/pip-ephem-wheel-cache-mhow5o2j/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:07:15, 11.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:19:04, 16.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:04:39, 23.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:03:46, 33.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.59M/862M [00:01<4:55:54, 48.4kB/s].vector_cache/glove.6B.zip:   1%|          | 7.84M/862M [00:01<3:26:12, 69.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.6M/862M [00:01<2:23:39, 98.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:01<1:39:58, 141kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 24.2M/862M [00:01<1:09:35, 201kB/s].vector_cache/glove.6B.zip:   3%|▎         | 30.0M/862M [00:01<48:28, 286kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.7M/862M [00:01<33:48, 408kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.3M/862M [00:02<23:35, 580kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.1M/862M [00:02<16:29, 823kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:02<11:37, 1.16MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.6M/862M [00:03<10:01, 1.34MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.2M/862M [00:04<07:55, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:04<05:46, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.7M/862M [00:05<09:47, 1.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:06<09:59, 1.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:06<07:45, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:06<05:35, 2.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.9M/862M [00:07<13:49, 964kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.3M/862M [00:08<11:05, 1.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<08:02, 1.65MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.0M/862M [00:09<08:41, 1.52MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:10<07:26, 1.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<05:32, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:11<06:59, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.3M/862M [00:11<07:35, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:12<05:52, 2.24MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:12<04:19, 3.04MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.3M/862M [00:13<07:43, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.7M/862M [00:13<06:44, 1.94MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<05:03, 2.59MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:15<06:35, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.8M/862M [00:15<05:56, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<04:29, 2.90MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:17<06:10, 2.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.7M/862M [00:17<06:58, 1.86MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:17<05:32, 2.34MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:19<05:57, 2.17MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.0M/862M [00:19<05:18, 2.43MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:19<04:03, 3.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:21<05:46, 2.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.1M/862M [00:21<05:27, 2.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:21<04:10, 3.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:23<05:42, 2.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.3M/862M [00:23<05:04, 2.51MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:23<03:48, 3.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:23<02:52, 4.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<1:36:40, 131kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<1:10:15, 181kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<49:46, 255kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<36:45, 344kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<27:01, 467kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<19:12, 656kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<16:20, 769kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<14:07, 890kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<10:25, 1.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:29<07:27, 1.68MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<10:56, 1.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<16:44, 746kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [00:33<12:41, 980kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<10:12, 1.22MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<07:26, 1.67MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<07:58, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<08:06, 1.52MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<06:17, 1.96MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<06:22, 1.93MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<05:45, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<04:20, 2.82MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<05:52, 2.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<05:21, 2.28MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<03:59, 3.05MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<05:42, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<06:30, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:10, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<03:44, 3.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<11:39:21, 17.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<8:10:31, 24.6kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<5:42:53, 35.1kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<4:02:08, 49.6kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<2:50:38, 70.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<1:59:29, 100kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<1:26:13, 139kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<1:01:21, 195kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<43:10, 276kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<32:55, 361kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<25:29, 466kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<18:26, 643kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:49<13:00, 908kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<1:33:20, 127kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<1:06:31, 177kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<46:45, 252kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<35:24, 332kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<27:10, 432kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<19:30, 601kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<13:47, 848kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<14:35, 800kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<11:32, 1.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:41, 1.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<06:33, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:09, 2.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<04:14, 2.74MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:55<03:16, 3.54MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<52:44, 220kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<41:00, 283kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<29:31, 393kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<21:20, 543kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<15:13, 759kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:57<10:51, 1.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<2:35:14, 74.3kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<1:53:49, 101kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<1:20:59, 142kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<56:55, 202kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<40:00, 287kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<32:28, 353kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<27:04, 424kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<19:56, 575kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<14:15, 802kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<12:33, 907kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<12:17, 927kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<09:21, 1.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<06:48, 1.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<07:45, 1.46MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<08:54, 1.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<07:00, 1.62MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:11, 2.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<03:49, 2.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<10:58, 1.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<11:22, 990kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<08:52, 1.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:24, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<04:44, 2.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<18:31, 604kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<16:37, 673kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<12:32, 891kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<08:59, 1.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<06:28, 1.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<33:33, 331kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<27:07, 410kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<19:52, 559kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<14:08, 784kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<12:43, 869kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<12:47, 864kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<09:51, 1.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:09, 1.54MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<07:45, 1.42MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<09:16, 1.18MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:25, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<05:26, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:32, 1.67MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<08:24, 1.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<06:43, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:57, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<06:11, 1.75MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<08:07, 1.33MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<06:27, 1.68MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:42, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:35, 3.01MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<11:14, 958kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<11:12, 961kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<08:34, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:16, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<04:33, 2.35MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<11:05, 965kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<11:26, 935kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<08:52, 1.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:24, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<04:39, 2.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<3:17:06, 54.0kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<2:21:25, 75.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<1:39:39, 107kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<1:09:59, 152kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<49:00, 216kB/s]  .vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<42:05, 251kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<32:54, 321kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<23:44, 445kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<16:53, 624kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<11:57, 879kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<16:39, 630kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<15:09, 692kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<11:26, 916kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<08:12, 1.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<08:25, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<09:18, 1.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<07:20, 1.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:22, 1.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:34, 1.58MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<07:59, 1.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:20, 1.63MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:40, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:52, 1.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<07:28, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:58, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<04:24, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<03:16, 3.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<09:49, 1.04MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<10:27, 977kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<08:02, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<05:53, 1.73MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:16, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<10:41, 950kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<11:02, 919kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<08:25, 1.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<06:08, 1.65MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<06:49, 1.48MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<08:19, 1.21MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<06:40, 1.51MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<04:52, 2.06MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<03:35, 2.79MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<11:17, 887kB/s] .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<11:25, 876kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<08:50, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<06:26, 1.55MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:59, 1.42MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<08:08, 1.22MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:22, 1.56MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<04:40, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<03:27, 2.86MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<09:16, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<09:43, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<07:37, 1.29MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<05:33, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<06:21, 1.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<07:42, 1.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<06:14, 1.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<04:33, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<03:21, 2.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<27:13, 358kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<22:15, 437kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<16:24, 593kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:50<11:38, 833kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<08:18, 1.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<20:34, 470kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<17:36, 549kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<13:06, 737kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<09:20, 1.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<09:00, 1.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<09:26, 1.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:15, 1.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<05:20, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:10, 1.54MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<07:29, 1.27MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<05:54, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<04:20, 2.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:39, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:51, 1.38MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:31, 1.71MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<04:03, 2.32MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:24, 1.74MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:51, 1.37MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:26, 1.72MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<04:00, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<02:59, 3.12MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<10:05, 923kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<10:20, 901kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<07:55, 1.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<05:44, 1.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<06:25, 1.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<07:32, 1.23MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:01, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<04:23, 2.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<03:14, 2.83MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<16:40, 550kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<14:42, 624kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<10:54, 841kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<07:52, 1.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<05:40, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<12:12, 746kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<11:32, 790kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<08:41, 1.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<06:17, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<04:34, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<11:32, 784kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<11:14, 804kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<08:36, 1.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<06:12, 1.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:42, 1.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<08:05, 1.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:22, 1.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<04:38, 1.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<03:23, 2.63MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<46:04, 193kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<35:10, 253kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<25:19, 351kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<17:52, 496kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<14:51, 595kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<13:19, 663kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<09:59, 884kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<07:11, 1.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<07:22, 1.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<08:01, 1.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:12, 1.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:31, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<03:19, 2.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<09:30, 914kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<09:30, 915kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<07:15, 1.20MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<05:28, 1.58MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<03:55, 2.20MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<09:18, 927kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<09:20, 924kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<07:06, 1.21MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<05:08, 1.67MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<03:46, 2.27MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<09:27, 905kB/s] .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<09:26, 907kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<07:17, 1.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<05:15, 1.62MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:52, 1.45MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:52, 1.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:23, 1.57MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<03:55, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<02:55, 2.88MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<08:16, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<08:32, 985kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<06:34, 1.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<04:47, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:27, 1.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<07:14, 1.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:55, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:19, 1.93MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<03:09, 2.63MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<12:16, 674kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<11:43, 706kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<08:52, 932kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<06:25, 1.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<04:40, 1.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<07:25, 1.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<08:17, 990kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<06:29, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:43, 1.73MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<03:26, 2.37MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<12:20, 660kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<11:42, 695kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<08:57, 908kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<06:27, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<06:24, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<07:18, 1.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<05:52, 1.37MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:17, 1.87MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:54, 1.63MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<06:27, 1.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:07, 1.56MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<03:44, 2.13MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<02:47, 2.84MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<06:35, 1.20MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<07:23, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:51, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<04:15, 1.86MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<04:55, 1.60MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<06:14, 1.26MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<05:04, 1.55MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<03:44, 2.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:31, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<05:43, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:34, 1.70MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:20, 2.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:31, 3.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<06:44, 1.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<07:25, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<05:45, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<04:12, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<03:03, 2.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<25:41, 298kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<20:39, 371kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<15:07, 506kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<10:44, 710kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<09:19, 814kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<09:12, 824kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<07:06, 1.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:07, 1.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<05:26, 1.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<06:16, 1.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<05:00, 1.50MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:39, 2.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<04:26, 1.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<05:44, 1.30MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:39, 1.60MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:25, 2.17MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<04:14, 1.74MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<05:22, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:23, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:14, 2.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:06, 1.78MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<05:29, 1.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:31, 1.61MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:17, 2.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<02:28, 2.93MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<06:20, 1.14MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<06:58, 1.04MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<05:22, 1.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:58, 1.82MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<02:54, 2.46MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<06:45, 1.06MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<07:14, 991kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<05:39, 1.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:06, 1.73MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:38, 1.53MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<06:25, 1.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<05:16, 1.35MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:52, 1.83MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:49, 2.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<1:03:27, 111kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<47:18, 149kB/s]  .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<33:41, 209kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<23:40, 296kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<16:38, 419kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<17:59, 387kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<15:27, 451kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<11:31, 604kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<08:11, 847kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<05:49, 1.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<11:12, 616kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<10:39, 647kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<08:04, 854kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:46, 1.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:13, 1.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:47, 1.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<06:50, 998kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:29, 1.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:58, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:53, 2.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<09:35, 705kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<09:30, 711kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<07:14, 933kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:16, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:48, 1.77MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<05:26, 1.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<06:32, 1.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<05:16, 1.27MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:49, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:48, 2.37MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<06:23, 1.04MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<10:28, 632kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<07:30, 880kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<05:20, 1.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<07:45, 845kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<08:06, 808kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<06:20, 1.03MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:33, 1.43MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:39, 1.39MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<05:54, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<04:42, 1.38MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:27, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:33, 2.51MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:57, 1.29MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<06:06, 1.05MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:55, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:34, 1.79MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:39, 2.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:55, 1.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<06:43, 943kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<05:23, 1.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:55, 1.61MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<02:50, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<48:36, 129kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<36:35, 172kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<26:14, 239kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<18:25, 339kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<12:58, 479kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<13:01, 477kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<11:39, 532kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<08:43, 711kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<06:25, 963kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:34, 1.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<05:44, 1.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<06:32, 937kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<05:12, 1.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:47, 1.61MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<04:00, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<05:17, 1.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<04:19, 1.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<03:09, 1.91MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:33, 1.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:57, 1.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<04:04, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<02:58, 2.01MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:10, 2.73MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<08:08, 728kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<08:20, 711kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<06:28, 916kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<04:38, 1.27MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:21, 1.75MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<07:04, 828kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<07:34, 773kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:55, 988kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<04:17, 1.36MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:06, 1.86MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<32:18, 179kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<25:11, 230kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<18:14, 317kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<12:50, 449kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<09:01, 635kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<18:51, 304kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<14:21, 398kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<10:20, 552kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<07:18, 777kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<07:28, 756kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<07:32, 749kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<05:48, 972kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:18, 1.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<03:06, 1.81MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:58, 1.40MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<05:04, 1.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:08, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<03:02, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<02:13, 2.49MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<33:51, 163kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<25:57, 213kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<18:38, 296kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<13:06, 418kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<09:17, 588kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<08:39, 630kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<08:29, 642kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<06:31, 834kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<04:39, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:53<03:20, 1.61MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<07:19, 734kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<07:32, 714kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<05:45, 934kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:13, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<03:02, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:10, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<05:05, 1.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:01, 1.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:54, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<02:09, 2.43MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:46, 1.39MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<04:47, 1.09MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:53, 1.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:50, 1.83MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<02:04, 2.50MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<2:18:33, 37.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<1:39:17, 52.1kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<1:09:58, 73.8kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<48:52, 105kB/s]   .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<34:09, 150kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<26:56, 189kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<21:07, 241kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<15:18, 333kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<10:48, 469kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<07:36, 662kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<38:26, 131kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<28:57, 174kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<20:45, 242kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<14:34, 343kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<11:15, 441kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<09:54, 501kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<07:26, 666kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<05:17, 933kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<03:46, 1.30MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<06:25, 762kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<06:29, 753kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<05:02, 969kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:38, 1.34MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:37, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<04:30, 1.07MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:34, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:36, 1.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<01:53, 2.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<12:11, 390kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<10:38, 447kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<07:57, 597kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<05:39, 834kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<04:01, 1.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<33:43, 139kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<25:31, 184kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<18:20, 255kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<12:53, 361kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<09:02, 511kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<36:43, 126kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<27:35, 167kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<19:41, 234kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<13:48, 332kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<09:45, 468kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<08:39, 525kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<07:55, 574kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<06:01, 754kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<04:17, 1.05MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<03:05, 1.45MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<04:23, 1.02MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<04:54, 913kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:54, 1.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:50, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<02:02, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<06:12, 710kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<06:09, 716kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:46, 922kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<03:25, 1.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<02:26, 1.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<09:53, 439kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<08:53, 489kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<06:40, 650kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<04:45, 905kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<03:22, 1.27MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<09:32, 448kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<08:35, 497kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<06:27, 660kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<04:36, 920kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<03:16, 1.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<24:39, 171kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<18:58, 222kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<13:41, 306kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<09:36, 434kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<06:49, 609kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<06:31, 635kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<06:15, 661kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:47, 860kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:26, 1.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<03:18, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<03:58, 1.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:12, 1.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:19, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:42, 2.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<03:36, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<04:09, 961kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:18, 1.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:23, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:45, 2.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:25, 1.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<04:00, 982kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:10, 1.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:17, 1.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:41, 2.29MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:28, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:53, 991kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<03:02, 1.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:14, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:39<01:37, 2.34MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:46, 1.36MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:22, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:42, 1.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:58, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:15, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:59, 1.25MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:23, 1.56MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:46, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:17, 2.83MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:45, 1.32MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:18, 1.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:39, 1.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:56, 1.86MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:24, 2.54MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<07:12, 497kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<06:18, 568kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<04:42, 759kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:21, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<03:09, 1.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<03:42, 946kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:55, 1.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:06, 1.65MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:30, 2.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<10:53, 316kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<08:55, 386kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<06:33, 525kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<04:38, 735kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<04:00, 842kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<04:14, 796kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:16, 1.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:21, 1.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:25, 1.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:51, 1.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:16, 1.45MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:39, 1.97MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:12, 2.69MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<10:37, 305kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<08:39, 374kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<06:20, 510kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<04:27, 717kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<03:09, 1.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<06:18, 502kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<05:31, 573kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<04:09, 758kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:58, 1.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:47, 1.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<03:07, 990kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<02:27, 1.25MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:47, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:58, 1.54MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:26, 1.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:59, 1.52MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:27, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:42, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:24, 1.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:57, 1.51MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:26, 2.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:03, 2.74MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<13:30, 214kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<10:44, 269kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<07:46, 371kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<05:28, 522kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<03:52, 733kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<04:15, 664kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<04:14, 664kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<03:16, 861kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<02:19, 1.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:39, 1.66MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:40, 750kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:47, 725kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:56, 931kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<02:07, 1.28MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:30, 1.77MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<18:42, 144kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<14:11, 189kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<10:10, 263kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<07:06, 373kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<04:57, 528kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<13:00, 201kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<10:10, 257kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<07:19, 356kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<05:09, 501kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<03:38, 703kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<03:36, 705kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<03:34, 713kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:42, 936kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:56, 1.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<01:24, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:04, 1.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:27, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:55, 1.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:23, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<01:01, 2.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:46, 1.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:13, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:48, 1.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<01:18, 1.82MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:25, 1.64MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:57, 1.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:33, 1.49MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:08, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:18, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:50, 1.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:30, 1.49MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:06, 2.01MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:18, 1.69MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:44, 1.26MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:23, 1.57MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:01, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<00:45, 2.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:28, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:54, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:31, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:10, 1.80MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<00:50, 2.48MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<00:38, 3.23MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<1:48:35, 19.0kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<1:16:48, 26.8kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<53:48, 38.2kB/s]  .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<37:13, 54.5kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<25:48, 77.7kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<19:10, 104kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<14:13, 140kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<10:06, 196kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<07:03, 278kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<04:55, 394kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<04:10, 462kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<03:42, 519kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:47, 686kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:58, 959kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<01:23, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<04:14, 437kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<03:47, 488kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:50, 648kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<02:00, 907kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<01:25, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<02:12, 808kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<02:21, 757kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:50, 969kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:18, 1.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:55, 1.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<04:53, 351kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<04:07, 416kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<03:01, 564kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<02:08, 790kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:30, 1.10MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:43, 952kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:53, 872kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:29, 1.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<01:03, 1.52MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<01:05, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:24, 1.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:08, 1.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:49, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:36, 2.52MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:46<01:11, 1.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:27, 1.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:10, 1.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:50, 1.74MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:36, 2.40MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<04:18, 335kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<03:35, 400kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<02:39, 540kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:51, 758kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<01:17, 1.06MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<03:40, 373kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<03:08, 437kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<02:19, 586kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:37, 819kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:24, 926kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:31, 855kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:11, 1.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:51, 1.49MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:51, 1.44MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:06, 1.12MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:52, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:37, 1.91MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:27, 2.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:48, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<01:02, 1.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:50, 1.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:36, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:26, 2.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:25, 2.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<04:55, 223kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<03:53, 281kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<02:48, 388kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:56, 548kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<01:21, 767kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:21, 753kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:24, 727kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:05, 932kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:46, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:33, 1.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:45, 1.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:57, 994kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:46, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:32, 1.69MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:02<00:23, 2.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<01:11, 750kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:13, 724kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:56, 931kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:39, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:27, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:51, 949kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:57, 846kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:45, 1.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:32, 1.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:22, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<05:12, 144kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<03:56, 190kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<02:48, 264kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<01:55, 373kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<01:25, 476kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<01:16, 531kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:56, 711kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:39, 994kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:27, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:33, 1.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:38, 950kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:30, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:21, 1.64MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:14, 2.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<01:23, 388kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<01:11, 452kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:52, 605kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:36, 846kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:29, 949kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:32, 872kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:24, 1.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:17, 1.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:12, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:18, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:22, 1.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:17, 1.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:12, 1.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:08, 2.46MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:15, 1.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:18, 1.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:14, 1.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:09, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:09, 1.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:12, 1.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:10, 1.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:09, 1.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:07, 1.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:06, 1.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:05, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 1.87MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.64MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.52MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.06MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 746/400000 [00:00<00:53, 7458.94it/s]  0%|          | 1502/400000 [00:00<00:53, 7486.43it/s]  1%|          | 2192/400000 [00:00<00:54, 7298.07it/s]  1%|          | 2944/400000 [00:00<00:53, 7361.71it/s]  1%|          | 3737/400000 [00:00<00:52, 7520.69it/s]  1%|          | 4503/400000 [00:00<00:52, 7561.78it/s]  1%|▏         | 5262/400000 [00:00<00:52, 7570.10it/s]  1%|▏         | 5969/400000 [00:00<00:53, 7412.61it/s]  2%|▏         | 6746/400000 [00:00<00:52, 7514.72it/s]  2%|▏         | 7471/400000 [00:01<00:52, 7431.39it/s]  2%|▏         | 8213/400000 [00:01<00:52, 7426.07it/s]  2%|▏         | 8989/400000 [00:01<00:51, 7521.59it/s]  2%|▏         | 9777/400000 [00:01<00:51, 7623.97it/s]  3%|▎         | 10559/400000 [00:01<00:50, 7681.43it/s]  3%|▎         | 11323/400000 [00:01<00:51, 7619.44it/s]  3%|▎         | 12119/400000 [00:01<00:50, 7716.53it/s]  3%|▎         | 12919/400000 [00:01<00:49, 7798.20it/s]  3%|▎         | 13724/400000 [00:01<00:49, 7870.23it/s]  4%|▎         | 14511/400000 [00:01<00:49, 7826.03it/s]  4%|▍         | 15294/400000 [00:02<00:51, 7524.26it/s]  4%|▍         | 16065/400000 [00:02<00:50, 7577.50it/s]  4%|▍         | 16871/400000 [00:02<00:49, 7691.41it/s]  4%|▍         | 17682/400000 [00:02<00:48, 7810.31it/s]  5%|▍         | 18465/400000 [00:02<00:48, 7813.53it/s]  5%|▍         | 19248/400000 [00:02<00:48, 7775.63it/s]  5%|▌         | 20074/400000 [00:02<00:48, 7913.07it/s]  5%|▌         | 20869/400000 [00:02<00:47, 7923.30it/s]  5%|▌         | 21663/400000 [00:02<00:47, 7923.95it/s]  6%|▌         | 22463/400000 [00:02<00:47, 7946.16it/s]  6%|▌         | 23263/400000 [00:03<00:47, 7960.00it/s]  6%|▌         | 24074/400000 [00:03<00:46, 8001.99it/s]  6%|▌         | 24875/400000 [00:03<00:47, 7980.07it/s]  6%|▋         | 25674/400000 [00:03<00:47, 7884.01it/s]  7%|▋         | 26510/400000 [00:03<00:46, 8019.72it/s]  7%|▋         | 27357/400000 [00:03<00:45, 8147.46it/s]  7%|▋         | 28187/400000 [00:03<00:45, 8192.48it/s]  7%|▋         | 29030/400000 [00:03<00:44, 8259.99it/s]  7%|▋         | 29857/400000 [00:03<00:44, 8229.04it/s]  8%|▊         | 30706/400000 [00:03<00:44, 8302.74it/s]  8%|▊         | 31561/400000 [00:04<00:43, 8374.44it/s]  8%|▊         | 32399/400000 [00:04<00:43, 8372.49it/s]  8%|▊         | 33237/400000 [00:04<00:44, 8264.24it/s]  9%|▊         | 34064/400000 [00:04<00:45, 7997.25it/s]  9%|▊         | 34889/400000 [00:04<00:45, 8070.22it/s]  9%|▉         | 35719/400000 [00:04<00:44, 8137.47it/s]  9%|▉         | 36535/400000 [00:04<00:44, 8103.64it/s]  9%|▉         | 37347/400000 [00:04<00:45, 8001.35it/s] 10%|▉         | 38149/400000 [00:04<00:45, 8005.19it/s] 10%|▉         | 38966/400000 [00:04<00:44, 8053.91it/s] 10%|▉         | 39796/400000 [00:05<00:45, 7966.19it/s] 10%|█         | 40594/400000 [00:05<00:45, 7943.80it/s] 10%|█         | 41404/400000 [00:05<00:44, 7987.67it/s] 11%|█         | 42245/400000 [00:05<00:44, 8108.49it/s] 11%|█         | 43085/400000 [00:05<00:43, 8192.42it/s] 11%|█         | 43936/400000 [00:05<00:42, 8283.09it/s] 11%|█         | 44782/400000 [00:05<00:42, 8332.94it/s] 11%|█▏        | 45617/400000 [00:05<00:42, 8337.58it/s] 12%|█▏        | 46452/400000 [00:05<00:43, 8114.79it/s] 12%|█▏        | 47266/400000 [00:05<00:44, 8003.66it/s] 12%|█▏        | 48094/400000 [00:06<00:43, 8083.32it/s] 12%|█▏        | 48942/400000 [00:06<00:42, 8197.71it/s] 12%|█▏        | 49764/400000 [00:06<00:42, 8180.33it/s] 13%|█▎        | 50607/400000 [00:06<00:42, 8252.52it/s] 13%|█▎        | 51455/400000 [00:06<00:41, 8318.17it/s] 13%|█▎        | 52305/400000 [00:06<00:41, 8370.02it/s] 13%|█▎        | 53146/400000 [00:06<00:41, 8381.42it/s] 13%|█▎        | 53989/400000 [00:06<00:41, 8393.46it/s] 14%|█▎        | 54829/400000 [00:06<00:42, 8114.77it/s] 14%|█▍        | 55643/400000 [00:06<00:42, 8030.44it/s] 14%|█▍        | 56469/400000 [00:07<00:42, 8095.18it/s] 14%|█▍        | 57315/400000 [00:07<00:41, 8199.26it/s] 15%|█▍        | 58138/400000 [00:07<00:41, 8205.56it/s] 15%|█▍        | 58979/400000 [00:07<00:41, 8263.14it/s] 15%|█▍        | 59807/400000 [00:07<00:41, 8254.06it/s] 15%|█▌        | 60657/400000 [00:07<00:40, 8325.03it/s] 15%|█▌        | 61507/400000 [00:07<00:40, 8375.10it/s] 16%|█▌        | 62345/400000 [00:07<00:40, 8371.73it/s] 16%|█▌        | 63183/400000 [00:07<00:40, 8349.68it/s] 16%|█▌        | 64033/400000 [00:07<00:40, 8391.73it/s] 16%|█▌        | 64884/400000 [00:08<00:39, 8423.45it/s] 16%|█▋        | 65728/400000 [00:08<00:39, 8427.40it/s] 17%|█▋        | 66571/400000 [00:08<00:39, 8403.30it/s] 17%|█▋        | 67412/400000 [00:08<00:39, 8351.86it/s] 17%|█▋        | 68262/400000 [00:08<00:39, 8393.76it/s] 17%|█▋        | 69111/400000 [00:08<00:39, 8420.41it/s] 17%|█▋        | 69954/400000 [00:08<00:39, 8406.92it/s] 18%|█▊        | 70798/400000 [00:08<00:39, 8416.66it/s] 18%|█▊        | 71640/400000 [00:08<00:39, 8376.58it/s] 18%|█▊        | 72478/400000 [00:09<00:39, 8219.61it/s] 18%|█▊        | 73301/400000 [00:09<00:40, 8107.99it/s] 19%|█▊        | 74113/400000 [00:09<00:41, 7911.08it/s] 19%|█▊        | 74945/400000 [00:09<00:40, 8028.44it/s] 19%|█▉        | 75750/400000 [00:09<00:40, 7914.28it/s] 19%|█▉        | 76588/400000 [00:09<00:40, 8046.72it/s] 19%|█▉        | 77423/400000 [00:09<00:39, 8133.62it/s] 20%|█▉        | 78262/400000 [00:09<00:39, 8208.26it/s] 20%|█▉        | 79084/400000 [00:09<00:39, 8166.36it/s] 20%|█▉        | 79902/400000 [00:09<00:40, 7961.28it/s] 20%|██        | 80704/400000 [00:10<00:40, 7976.34it/s] 20%|██        | 81514/400000 [00:10<00:39, 8012.83it/s] 21%|██        | 82317/400000 [00:10<00:42, 7390.59it/s] 21%|██        | 83095/400000 [00:10<00:42, 7501.40it/s] 21%|██        | 83883/400000 [00:10<00:41, 7608.54it/s] 21%|██        | 84658/400000 [00:10<00:41, 7650.29it/s] 21%|██▏       | 85428/400000 [00:10<00:42, 7488.71it/s] 22%|██▏       | 86247/400000 [00:10<00:40, 7684.65it/s] 22%|██▏       | 87038/400000 [00:10<00:40, 7748.47it/s] 22%|██▏       | 87834/400000 [00:10<00:39, 7810.25it/s] 22%|██▏       | 88618/400000 [00:11<00:39, 7807.09it/s] 22%|██▏       | 89423/400000 [00:11<00:39, 7877.34it/s] 23%|██▎       | 90212/400000 [00:11<00:42, 7363.72it/s] 23%|██▎       | 90981/400000 [00:11<00:41, 7457.59it/s] 23%|██▎       | 91798/400000 [00:11<00:40, 7657.39it/s] 23%|██▎       | 92592/400000 [00:11<00:39, 7739.82it/s] 23%|██▎       | 93400/400000 [00:11<00:39, 7836.64it/s] 24%|██▎       | 94199/400000 [00:11<00:38, 7880.46it/s] 24%|██▍       | 95011/400000 [00:11<00:38, 7949.20it/s] 24%|██▍       | 95808/400000 [00:11<00:38, 7892.34it/s] 24%|██▍       | 96608/400000 [00:12<00:38, 7922.18it/s] 24%|██▍       | 97402/400000 [00:12<00:38, 7877.35it/s] 25%|██▍       | 98191/400000 [00:12<00:39, 7575.62it/s] 25%|██▍       | 99011/400000 [00:12<00:38, 7752.00it/s] 25%|██▍       | 99838/400000 [00:12<00:37, 7900.34it/s] 25%|██▌       | 100646/400000 [00:12<00:37, 7952.69it/s] 25%|██▌       | 101444/400000 [00:12<00:37, 7894.81it/s] 26%|██▌       | 102236/400000 [00:12<00:38, 7822.10it/s] 26%|██▌       | 103022/400000 [00:12<00:37, 7831.30it/s] 26%|██▌       | 103806/400000 [00:13<00:37, 7806.21it/s] 26%|██▌       | 104629/400000 [00:13<00:37, 7928.30it/s] 26%|██▋       | 105435/400000 [00:13<00:36, 7964.69it/s] 27%|██▋       | 106233/400000 [00:13<00:38, 7678.50it/s] 27%|██▋       | 107004/400000 [00:13<00:38, 7580.38it/s] 27%|██▋       | 107848/400000 [00:13<00:37, 7817.54it/s] 27%|██▋       | 108690/400000 [00:13<00:36, 7987.67it/s] 27%|██▋       | 109493/400000 [00:13<00:36, 7892.16it/s] 28%|██▊       | 110307/400000 [00:13<00:36, 7963.77it/s] 28%|██▊       | 111106/400000 [00:13<00:36, 7931.92it/s] 28%|██▊       | 111943/400000 [00:14<00:35, 8056.31it/s] 28%|██▊       | 112751/400000 [00:14<00:36, 7953.74it/s] 28%|██▊       | 113548/400000 [00:14<00:37, 7668.84it/s] 29%|██▊       | 114332/400000 [00:14<00:37, 7718.34it/s] 29%|██▉       | 115135/400000 [00:14<00:36, 7807.80it/s] 29%|██▉       | 115950/400000 [00:14<00:35, 7893.40it/s] 29%|██▉       | 116767/400000 [00:14<00:35, 7973.35it/s] 29%|██▉       | 117566/400000 [00:14<00:36, 7774.87it/s] 30%|██▉       | 118355/400000 [00:14<00:36, 7807.96it/s] 30%|██▉       | 119138/400000 [00:14<00:36, 7679.53it/s] 30%|██▉       | 119924/400000 [00:15<00:36, 7732.22it/s] 30%|███       | 120699/400000 [00:15<00:36, 7703.15it/s] 30%|███       | 121471/400000 [00:15<00:36, 7528.11it/s] 31%|███       | 122290/400000 [00:15<00:36, 7713.74it/s] 31%|███       | 123085/400000 [00:15<00:35, 7781.29it/s] 31%|███       | 123877/400000 [00:15<00:35, 7821.53it/s] 31%|███       | 124681/400000 [00:15<00:34, 7884.72it/s] 31%|███▏      | 125493/400000 [00:15<00:34, 7953.01it/s] 32%|███▏      | 126322/400000 [00:15<00:33, 8049.40it/s] 32%|███▏      | 127132/400000 [00:15<00:33, 8061.97it/s] 32%|███▏      | 127939/400000 [00:16<00:34, 7970.35it/s] 32%|███▏      | 128737/400000 [00:16<00:34, 7832.18it/s] 32%|███▏      | 129522/400000 [00:16<00:35, 7523.65it/s] 33%|███▎      | 130325/400000 [00:16<00:35, 7666.90it/s] 33%|███▎      | 131139/400000 [00:16<00:34, 7800.92it/s] 33%|███▎      | 131956/400000 [00:16<00:33, 7907.39it/s] 33%|███▎      | 132767/400000 [00:16<00:33, 7965.65it/s] 33%|███▎      | 133566/400000 [00:16<00:33, 7861.79it/s] 34%|███▎      | 134371/400000 [00:16<00:33, 7914.93it/s] 34%|███▍      | 135174/400000 [00:17<00:33, 7948.21it/s] 34%|███▍      | 135970/400000 [00:17<00:33, 7767.78it/s] 34%|███▍      | 136795/400000 [00:17<00:33, 7906.35it/s] 34%|███▍      | 137588/400000 [00:17<00:34, 7564.32it/s] 35%|███▍      | 138395/400000 [00:17<00:33, 7706.92it/s] 35%|███▍      | 139190/400000 [00:17<00:33, 7776.23it/s] 35%|███▍      | 139989/400000 [00:17<00:33, 7836.95it/s] 35%|███▌      | 140796/400000 [00:17<00:32, 7903.79it/s] 35%|███▌      | 141639/400000 [00:17<00:32, 8053.88it/s] 36%|███▌      | 142485/400000 [00:17<00:31, 8170.19it/s] 36%|███▌      | 143320/400000 [00:18<00:31, 8222.60it/s] 36%|███▌      | 144169/400000 [00:18<00:30, 8287.14it/s] 36%|███▋      | 145002/400000 [00:18<00:30, 8298.05it/s] 36%|███▋      | 145846/400000 [00:18<00:30, 8337.64it/s] 37%|███▋      | 146681/400000 [00:18<00:30, 8218.67it/s] 37%|███▋      | 147504/400000 [00:18<00:30, 8161.77it/s] 37%|███▋      | 148322/400000 [00:18<00:30, 8165.47it/s] 37%|███▋      | 149139/400000 [00:18<00:30, 8092.91it/s] 37%|███▋      | 149962/400000 [00:18<00:30, 8131.00it/s] 38%|███▊      | 150811/400000 [00:18<00:30, 8235.16it/s] 38%|███▊      | 151644/400000 [00:19<00:30, 8262.01it/s] 38%|███▊      | 152483/400000 [00:19<00:29, 8298.87it/s] 38%|███▊      | 153316/400000 [00:19<00:29, 8307.44it/s] 39%|███▊      | 154166/400000 [00:19<00:29, 8363.99it/s] 39%|███▉      | 155012/400000 [00:19<00:29, 8391.17it/s] 39%|███▉      | 155853/400000 [00:19<00:29, 8394.90it/s] 39%|███▉      | 156700/400000 [00:19<00:28, 8414.90it/s] 39%|███▉      | 157542/400000 [00:19<00:28, 8385.37it/s] 40%|███▉      | 158393/400000 [00:19<00:28, 8420.75it/s] 40%|███▉      | 159245/400000 [00:19<00:28, 8447.52it/s] 40%|████      | 160090/400000 [00:20<00:28, 8401.59it/s] 40%|████      | 160943/400000 [00:20<00:28, 8437.39it/s] 40%|████      | 161787/400000 [00:20<00:28, 8408.88it/s] 41%|████      | 162639/400000 [00:20<00:28, 8441.08it/s] 41%|████      | 163484/400000 [00:20<00:28, 8381.32it/s] 41%|████      | 164328/400000 [00:20<00:28, 8398.35it/s] 41%|████▏     | 165174/400000 [00:20<00:27, 8414.38it/s] 42%|████▏     | 166016/400000 [00:20<00:27, 8389.64it/s] 42%|████▏     | 166856/400000 [00:20<00:27, 8380.27it/s] 42%|████▏     | 167695/400000 [00:20<00:27, 8363.91it/s] 42%|████▏     | 168532/400000 [00:21<00:28, 8235.06it/s] 42%|████▏     | 169363/400000 [00:21<00:27, 8256.65it/s] 43%|████▎     | 170192/400000 [00:21<00:27, 8264.59it/s] 43%|████▎     | 171019/400000 [00:21<00:27, 8252.04it/s] 43%|████▎     | 171864/400000 [00:21<00:27, 8309.06it/s] 43%|████▎     | 172696/400000 [00:21<00:27, 8274.41it/s] 43%|████▎     | 173524/400000 [00:21<00:27, 8228.67it/s] 44%|████▎     | 174350/400000 [00:21<00:27, 8236.16it/s] 44%|████▍     | 175181/400000 [00:21<00:27, 8257.86it/s] 44%|████▍     | 176024/400000 [00:21<00:26, 8307.91it/s] 44%|████▍     | 176862/400000 [00:22<00:26, 8328.13it/s] 44%|████▍     | 177708/400000 [00:22<00:26, 8349.79it/s] 45%|████▍     | 178556/400000 [00:22<00:26, 8386.47it/s] 45%|████▍     | 179397/400000 [00:22<00:26, 8392.65it/s] 45%|████▌     | 180237/400000 [00:22<00:26, 8381.87it/s] 45%|████▌     | 181076/400000 [00:22<00:26, 8268.10it/s] 45%|████▌     | 181904/400000 [00:22<00:26, 8188.94it/s] 46%|████▌     | 182724/400000 [00:22<00:26, 8140.55it/s] 46%|████▌     | 183539/400000 [00:22<00:26, 8090.18it/s] 46%|████▌     | 184374/400000 [00:22<00:26, 8165.62it/s] 46%|████▋     | 185191/400000 [00:23<00:26, 8134.90it/s] 47%|████▋     | 186005/400000 [00:23<00:26, 7928.21it/s] 47%|████▋     | 186839/400000 [00:23<00:26, 8046.49it/s] 47%|████▋     | 187646/400000 [00:23<00:26, 7931.16it/s] 47%|████▋     | 188441/400000 [00:23<00:27, 7682.96it/s] 47%|████▋     | 189212/400000 [00:23<00:28, 7493.23it/s] 48%|████▊     | 190035/400000 [00:23<00:27, 7698.84it/s] 48%|████▊     | 190853/400000 [00:23<00:26, 7835.80it/s] 48%|████▊     | 191688/400000 [00:23<00:26, 7981.71it/s] 48%|████▊     | 192535/400000 [00:24<00:25, 8120.65it/s] 48%|████▊     | 193350/400000 [00:24<00:25, 8012.15it/s] 49%|████▊     | 194196/400000 [00:24<00:25, 8140.46it/s] 49%|████▉     | 195043/400000 [00:24<00:24, 8236.51it/s] 49%|████▉     | 195890/400000 [00:24<00:24, 8303.26it/s] 49%|████▉     | 196723/400000 [00:24<00:24, 8308.83it/s] 49%|████▉     | 197555/400000 [00:24<00:24, 8310.33it/s] 50%|████▉     | 198407/400000 [00:24<00:24, 8370.39it/s] 50%|████▉     | 199256/400000 [00:24<00:23, 8403.90it/s] 50%|█████     | 200097/400000 [00:24<00:23, 8393.58it/s] 50%|█████     | 200937/400000 [00:25<00:23, 8353.33it/s] 50%|█████     | 201773/400000 [00:25<00:23, 8287.88it/s] 51%|█████     | 202603/400000 [00:25<00:24, 8076.85it/s] 51%|█████     | 203442/400000 [00:25<00:24, 8167.74it/s] 51%|█████     | 204260/400000 [00:25<00:25, 7829.24it/s] 51%|█████▏    | 205081/400000 [00:25<00:24, 7937.33it/s] 51%|█████▏    | 205912/400000 [00:25<00:24, 8043.01it/s] 52%|█████▏    | 206760/400000 [00:25<00:23, 8168.17it/s] 52%|█████▏    | 207605/400000 [00:25<00:23, 8248.72it/s] 52%|█████▏    | 208432/400000 [00:25<00:23, 8254.27it/s] 52%|█████▏    | 209259/400000 [00:26<00:23, 8188.74it/s] 53%|█████▎    | 210081/400000 [00:26<00:23, 8197.12it/s] 53%|█████▎    | 210902/400000 [00:26<00:23, 8154.82it/s] 53%|█████▎    | 211719/400000 [00:26<00:23, 8145.90it/s] 53%|█████▎    | 212568/400000 [00:26<00:22, 8243.99it/s] 53%|█████▎    | 213393/400000 [00:26<00:22, 8126.16it/s] 54%|█████▎    | 214214/400000 [00:26<00:22, 8149.79it/s] 54%|█████▍    | 215048/400000 [00:26<00:22, 8203.78it/s] 54%|█████▍    | 215887/400000 [00:26<00:22, 8258.14it/s] 54%|█████▍    | 216714/400000 [00:26<00:22, 8019.39it/s] 54%|█████▍    | 217518/400000 [00:27<00:23, 7903.05it/s] 55%|█████▍    | 218310/400000 [00:27<00:23, 7877.45it/s] 55%|█████▍    | 219159/400000 [00:27<00:22, 8050.11it/s] 55%|█████▌    | 220008/400000 [00:27<00:22, 8176.59it/s] 55%|█████▌    | 220846/400000 [00:27<00:21, 8235.38it/s] 55%|█████▌    | 221683/400000 [00:27<00:21, 8274.53it/s] 56%|█████▌    | 222520/400000 [00:27<00:21, 8300.97it/s] 56%|█████▌    | 223365/400000 [00:27<00:21, 8344.70it/s] 56%|█████▌    | 224217/400000 [00:27<00:20, 8394.43it/s] 56%|█████▋    | 225061/400000 [00:27<00:20, 8405.72it/s] 56%|█████▋    | 225902/400000 [00:28<00:20, 8312.70it/s] 57%|█████▋    | 226734/400000 [00:28<00:20, 8268.50it/s] 57%|█████▋    | 227562/400000 [00:28<00:20, 8259.99it/s] 57%|█████▋    | 228398/400000 [00:28<00:20, 8288.79it/s] 57%|█████▋    | 229228/400000 [00:28<00:20, 8265.31it/s] 58%|█████▊    | 230072/400000 [00:28<00:20, 8314.29it/s] 58%|█████▊    | 230904/400000 [00:28<00:20, 8304.56it/s] 58%|█████▊    | 231753/400000 [00:28<00:20, 8359.02it/s] 58%|█████▊    | 232599/400000 [00:28<00:19, 8387.19it/s] 58%|█████▊    | 233438/400000 [00:28<00:20, 8295.56it/s] 59%|█████▊    | 234268/400000 [00:29<00:20, 8223.54it/s] 59%|█████▉    | 235091/400000 [00:29<00:20, 8182.71it/s] 59%|█████▉    | 235930/400000 [00:29<00:19, 8241.41it/s] 59%|█████▉    | 236768/400000 [00:29<00:19, 8282.44it/s] 59%|█████▉    | 237618/400000 [00:29<00:19, 8344.32it/s] 60%|█████▉    | 238470/400000 [00:29<00:19, 8393.96it/s] 60%|█████▉    | 239310/400000 [00:29<00:19, 8370.01it/s] 60%|██████    | 240148/400000 [00:29<00:19, 8306.57it/s] 60%|██████    | 240979/400000 [00:29<00:19, 8295.43it/s] 60%|██████    | 241824/400000 [00:30<00:18, 8339.32it/s] 61%|██████    | 242662/400000 [00:30<00:18, 8349.26it/s] 61%|██████    | 243498/400000 [00:30<00:18, 8319.88it/s] 61%|██████    | 244331/400000 [00:30<00:18, 8306.94it/s] 61%|██████▏   | 245179/400000 [00:30<00:18, 8355.12it/s] 62%|██████▏   | 246015/400000 [00:30<00:18, 8327.04it/s] 62%|██████▏   | 246863/400000 [00:30<00:18, 8369.98it/s] 62%|██████▏   | 247701/400000 [00:30<00:18, 8239.16it/s] 62%|██████▏   | 248533/400000 [00:30<00:18, 8261.02it/s] 62%|██████▏   | 249360/400000 [00:30<00:18, 8259.02it/s] 63%|██████▎   | 250187/400000 [00:31<00:18, 8181.92it/s] 63%|██████▎   | 251032/400000 [00:31<00:18, 8258.06it/s] 63%|██████▎   | 251859/400000 [00:31<00:17, 8257.52it/s] 63%|██████▎   | 252686/400000 [00:31<00:18, 8170.61it/s] 63%|██████▎   | 253521/400000 [00:31<00:17, 8221.16it/s] 64%|██████▎   | 254369/400000 [00:31<00:17, 8295.52it/s] 64%|██████▍   | 255211/400000 [00:31<00:17, 8330.87it/s] 64%|██████▍   | 256045/400000 [00:31<00:17, 8251.30it/s] 64%|██████▍   | 256871/400000 [00:31<00:17, 8248.51it/s] 64%|██████▍   | 257697/400000 [00:31<00:17, 8226.93it/s] 65%|██████▍   | 258520/400000 [00:32<00:17, 8123.55it/s] 65%|██████▍   | 259367/400000 [00:32<00:17, 8223.93it/s] 65%|██████▌   | 260190/400000 [00:32<00:17, 8147.42it/s] 65%|██████▌   | 261016/400000 [00:32<00:16, 8179.74it/s] 65%|██████▌   | 261851/400000 [00:32<00:16, 8230.07it/s] 66%|██████▌   | 262686/400000 [00:32<00:16, 8263.26it/s] 66%|██████▌   | 263528/400000 [00:32<00:16, 8309.56it/s] 66%|██████▌   | 264360/400000 [00:32<00:16, 8258.73it/s] 66%|██████▋   | 265187/400000 [00:32<00:16, 8241.88it/s] 67%|██████▋   | 266040/400000 [00:32<00:16, 8323.52it/s] 67%|██████▋   | 266888/400000 [00:33<00:15, 8366.96it/s] 67%|██████▋   | 267725/400000 [00:33<00:15, 8331.20it/s] 67%|██████▋   | 268559/400000 [00:33<00:15, 8217.71it/s] 67%|██████▋   | 269382/400000 [00:33<00:15, 8209.08it/s] 68%|██████▊   | 270204/400000 [00:33<00:15, 8180.07it/s] 68%|██████▊   | 271023/400000 [00:33<00:15, 8097.50it/s] 68%|██████▊   | 271834/400000 [00:33<00:16, 7771.99it/s] 68%|██████▊   | 272666/400000 [00:33<00:16, 7926.29it/s] 68%|██████▊   | 273462/400000 [00:33<00:15, 7913.56it/s] 69%|██████▊   | 274293/400000 [00:33<00:15, 8028.33it/s] 69%|██████▉   | 275113/400000 [00:34<00:15, 8076.23it/s] 69%|██████▉   | 275922/400000 [00:34<00:15, 8077.83it/s] 69%|██████▉   | 276750/400000 [00:34<00:15, 8133.90it/s] 69%|██████▉   | 277565/400000 [00:34<00:15, 8064.20it/s] 70%|██████▉   | 278388/400000 [00:34<00:14, 8110.70it/s] 70%|██████▉   | 279200/400000 [00:34<00:14, 8111.08it/s] 70%|███████   | 280020/400000 [00:34<00:14, 8137.08it/s] 70%|███████   | 280865/400000 [00:34<00:14, 8226.85it/s] 70%|███████   | 281689/400000 [00:34<00:14, 8102.94it/s] 71%|███████   | 282519/400000 [00:34<00:14, 8160.59it/s] 71%|███████   | 283336/400000 [00:35<00:14, 8153.78it/s] 71%|███████   | 284177/400000 [00:35<00:14, 8228.24it/s] 71%|███████▏  | 285009/400000 [00:35<00:13, 8253.04it/s] 71%|███████▏  | 285835/400000 [00:35<00:13, 8183.75it/s] 72%|███████▏  | 286657/400000 [00:35<00:13, 8193.43it/s] 72%|███████▏  | 287499/400000 [00:35<00:13, 8257.44it/s] 72%|███████▏  | 288347/400000 [00:35<00:13, 8320.34it/s] 72%|███████▏  | 289197/400000 [00:35<00:13, 8372.69it/s] 73%|███████▎  | 290035/400000 [00:35<00:13, 8037.07it/s] 73%|███████▎  | 290858/400000 [00:35<00:13, 8092.94it/s] 73%|███████▎  | 291677/400000 [00:36<00:13, 8120.33it/s] 73%|███████▎  | 292519/400000 [00:36<00:13, 8207.57it/s] 73%|███████▎  | 293371/400000 [00:36<00:12, 8297.47it/s] 74%|███████▎  | 294202/400000 [00:36<00:12, 8299.21it/s] 74%|███████▍  | 295046/400000 [00:36<00:12, 8338.67it/s] 74%|███████▍  | 295881/400000 [00:36<00:12, 8305.04it/s] 74%|███████▍  | 296712/400000 [00:36<00:13, 7908.18it/s] 74%|███████▍  | 297551/400000 [00:36<00:12, 8046.38it/s] 75%|███████▍  | 298360/400000 [00:36<00:13, 7598.93it/s] 75%|███████▍  | 299166/400000 [00:37<00:13, 7730.29it/s] 75%|███████▌  | 300000/400000 [00:37<00:12, 7901.43it/s] 75%|███████▌  | 300837/400000 [00:37<00:12, 8034.03it/s] 75%|███████▌  | 301658/400000 [00:37<00:12, 8084.14it/s] 76%|███████▌  | 302470/400000 [00:37<00:12, 8035.38it/s] 76%|███████▌  | 303276/400000 [00:37<00:12, 7923.43it/s] 76%|███████▌  | 304122/400000 [00:37<00:11, 8076.14it/s] 76%|███████▌  | 304971/400000 [00:37<00:11, 8193.49it/s] 76%|███████▋  | 305793/400000 [00:37<00:11, 8151.43it/s] 77%|███████▋  | 306623/400000 [00:37<00:11, 8194.43it/s] 77%|███████▋  | 307444/400000 [00:38<00:11, 8190.13it/s] 77%|███████▋  | 308264/400000 [00:38<00:11, 8141.40it/s] 77%|███████▋  | 309095/400000 [00:38<00:11, 8188.73it/s] 77%|███████▋  | 309932/400000 [00:38<00:10, 8241.27it/s] 78%|███████▊  | 310764/400000 [00:38<00:10, 8262.14it/s] 78%|███████▊  | 311593/400000 [00:38<00:10, 8270.27it/s] 78%|███████▊  | 312439/400000 [00:38<00:10, 8325.78it/s] 78%|███████▊  | 313272/400000 [00:38<00:10, 8246.61it/s] 79%|███████▊  | 314097/400000 [00:38<00:10, 8028.82it/s] 79%|███████▊  | 314932/400000 [00:38<00:10, 8121.52it/s] 79%|███████▉  | 315753/400000 [00:39<00:10, 8147.33it/s] 79%|███████▉  | 316569/400000 [00:39<00:10, 8146.21it/s] 79%|███████▉  | 317385/400000 [00:39<00:10, 8136.72it/s] 80%|███████▉  | 318200/400000 [00:39<00:10, 8132.01it/s] 80%|███████▉  | 319027/400000 [00:39<00:09, 8170.14it/s] 80%|███████▉  | 319845/400000 [00:39<00:09, 8165.35it/s] 80%|████████  | 320662/400000 [00:39<00:09, 8150.94it/s] 80%|████████  | 321478/400000 [00:39<00:09, 8019.93it/s] 81%|████████  | 322281/400000 [00:39<00:10, 7735.94it/s] 81%|████████  | 323118/400000 [00:39<00:09, 7913.65it/s] 81%|████████  | 323965/400000 [00:40<00:09, 8072.61it/s] 81%|████████  | 324808/400000 [00:40<00:09, 8175.45it/s] 81%|████████▏ | 325649/400000 [00:40<00:09, 8243.96it/s] 82%|████████▏ | 326491/400000 [00:40<00:08, 8295.43it/s] 82%|████████▏ | 327323/400000 [00:40<00:08, 8302.38it/s] 82%|████████▏ | 328172/400000 [00:40<00:08, 8355.39it/s] 82%|████████▏ | 329009/400000 [00:40<00:08, 8329.89it/s] 82%|████████▏ | 329860/400000 [00:40<00:08, 8380.79it/s] 83%|████████▎ | 330699/400000 [00:40<00:08, 8228.20it/s] 83%|████████▎ | 331523/400000 [00:40<00:08, 8070.73it/s] 83%|████████▎ | 332337/400000 [00:41<00:08, 8089.47it/s] 83%|████████▎ | 333147/400000 [00:41<00:08, 8074.16it/s] 83%|████████▎ | 333956/400000 [00:41<00:08, 8033.25it/s] 84%|████████▎ | 334760/400000 [00:41<00:08, 7984.27it/s] 84%|████████▍ | 335559/400000 [00:41<00:08, 7971.70it/s] 84%|████████▍ | 336369/400000 [00:41<00:07, 8009.15it/s] 84%|████████▍ | 337171/400000 [00:41<00:07, 7905.30it/s] 85%|████████▍ | 338003/400000 [00:41<00:07, 8025.29it/s] 85%|████████▍ | 338811/400000 [00:41<00:07, 8038.92it/s] 85%|████████▍ | 339616/400000 [00:41<00:07, 8040.50it/s] 85%|████████▌ | 340439/400000 [00:42<00:07, 8094.99it/s] 85%|████████▌ | 341251/400000 [00:42<00:07, 8101.85it/s] 86%|████████▌ | 342062/400000 [00:42<00:07, 8099.57it/s] 86%|████████▌ | 342893/400000 [00:42<00:06, 8161.44it/s] 86%|████████▌ | 343710/400000 [00:42<00:06, 8133.91it/s] 86%|████████▌ | 344552/400000 [00:42<00:06, 8217.41it/s] 86%|████████▋ | 345392/400000 [00:42<00:06, 8268.36it/s] 87%|████████▋ | 346238/400000 [00:42<00:06, 8323.66it/s] 87%|████████▋ | 347071/400000 [00:42<00:06, 8268.51it/s] 87%|████████▋ | 347901/400000 [00:43<00:06, 8276.53it/s] 87%|████████▋ | 348754/400000 [00:43<00:06, 8350.26it/s] 87%|████████▋ | 349594/400000 [00:43<00:06, 8362.57it/s] 88%|████████▊ | 350431/400000 [00:43<00:05, 8353.91it/s] 88%|████████▊ | 351281/400000 [00:43<00:05, 8396.30it/s] 88%|████████▊ | 352121/400000 [00:43<00:05, 8330.87it/s] 88%|████████▊ | 352955/400000 [00:43<00:05, 8270.84it/s] 88%|████████▊ | 353783/400000 [00:43<00:05, 8202.08it/s] 89%|████████▊ | 354609/400000 [00:43<00:05, 8218.63it/s] 89%|████████▉ | 355432/400000 [00:43<00:05, 8165.60it/s] 89%|████████▉ | 356251/400000 [00:44<00:05, 8169.82it/s] 89%|████████▉ | 357086/400000 [00:44<00:05, 8220.68it/s] 89%|████████▉ | 357909/400000 [00:44<00:05, 8166.09it/s] 90%|████████▉ | 358730/400000 [00:44<00:05, 8176.38it/s] 90%|████████▉ | 359570/400000 [00:44<00:04, 8241.36it/s] 90%|█████████ | 360395/400000 [00:44<00:04, 8240.87it/s] 90%|█████████ | 361243/400000 [00:44<00:04, 8310.10it/s] 91%|█████████ | 362095/400000 [00:44<00:04, 8371.18it/s] 91%|█████████ | 362933/400000 [00:44<00:04, 8261.88it/s] 91%|█████████ | 363769/400000 [00:44<00:04, 8289.39it/s] 91%|█████████ | 364605/400000 [00:45<00:04, 8310.03it/s] 91%|█████████▏| 365454/400000 [00:45<00:04, 8363.16it/s] 92%|█████████▏| 366309/400000 [00:45<00:04, 8416.03it/s] 92%|█████████▏| 367151/400000 [00:45<00:03, 8401.16it/s] 92%|█████████▏| 367999/400000 [00:45<00:03, 8422.09it/s] 92%|█████████▏| 368843/400000 [00:45<00:03, 8425.74it/s] 92%|█████████▏| 369697/400000 [00:45<00:03, 8458.91it/s] 93%|█████████▎| 370543/400000 [00:45<00:03, 8445.02it/s] 93%|█████████▎| 371388/400000 [00:45<00:03, 8420.61it/s] 93%|█████████▎| 372237/400000 [00:45<00:03, 8439.26it/s] 93%|█████████▎| 373081/400000 [00:46<00:03, 8331.65it/s] 93%|█████████▎| 373915/400000 [00:46<00:03, 8283.05it/s] 94%|█████████▎| 374744/400000 [00:46<00:03, 8228.50it/s] 94%|█████████▍| 375568/400000 [00:46<00:03, 8129.50it/s] 94%|█████████▍| 376415/400000 [00:46<00:02, 8226.36it/s] 94%|█████████▍| 377243/400000 [00:46<00:02, 8239.79it/s] 95%|█████████▍| 378073/400000 [00:46<00:02, 8257.07it/s] 95%|█████████▍| 378900/400000 [00:46<00:02, 8211.99it/s] 95%|█████████▍| 379722/400000 [00:46<00:02, 7957.85it/s] 95%|█████████▌| 380553/400000 [00:46<00:02, 8057.74it/s] 95%|█████████▌| 381377/400000 [00:47<00:02, 8111.26it/s] 96%|█████████▌| 382216/400000 [00:47<00:02, 8192.11it/s] 96%|█████████▌| 383061/400000 [00:47<00:02, 8267.55it/s] 96%|█████████▌| 383889/400000 [00:47<00:01, 8244.49it/s] 96%|█████████▌| 384735/400000 [00:47<00:01, 8307.79it/s] 96%|█████████▋| 385567/400000 [00:47<00:01, 8309.30it/s] 97%|█████████▋| 386399/400000 [00:47<00:01, 8243.78it/s] 97%|█████████▋| 387224/400000 [00:47<00:01, 8241.13it/s] 97%|█████████▋| 388049/400000 [00:47<00:01, 8236.74it/s] 97%|█████████▋| 388894/400000 [00:47<00:01, 8298.18it/s] 97%|█████████▋| 389725/400000 [00:48<00:01, 8067.54it/s] 98%|█████████▊| 390571/400000 [00:48<00:01, 8179.36it/s] 98%|█████████▊| 391418/400000 [00:48<00:01, 8263.72it/s] 98%|█████████▊| 392250/400000 [00:48<00:00, 8279.14it/s] 98%|█████████▊| 393098/400000 [00:48<00:00, 8338.28it/s] 98%|█████████▊| 393933/400000 [00:48<00:00, 8297.60it/s] 99%|█████████▊| 394764/400000 [00:48<00:00, 8257.26it/s] 99%|█████████▉| 395594/400000 [00:48<00:00, 8269.35it/s] 99%|█████████▉| 396422/400000 [00:48<00:00, 8146.73it/s] 99%|█████████▉| 397248/400000 [00:48<00:00, 8179.70it/s]100%|█████████▉| 398072/400000 [00:49<00:00, 8197.42it/s]100%|█████████▉| 398910/400000 [00:49<00:00, 8249.41it/s]100%|█████████▉| 399755/400000 [00:49<00:00, 8307.43it/s]100%|█████████▉| 399999/400000 [00:49<00:00, 8113.19it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f7a0e6253c8>, <torchtext.data.dataset.TabularDataset object at 0x7f7a0e625518>, <torchtext.vocab.Vocab object at 0x7f7a0e625438>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  4.95 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  4.95 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  4.95 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.36 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.36 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.12 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.12 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.32 file/s]2020-06-29 00:20:53.807158: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-29 00:20:53.812005: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-29 00:20:53.812195: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558d01c70000 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-29 00:20:53.812211: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 464357.39it/s] 42%|████▏     | 4169728/9912422 [00:00<00:08, 660155.82it/s]9920512it [00:00, 34770960.24it/s]                           
0it [00:00, ?it/s]32768it [00:00, 702168.51it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1025444.85it/s]1654784it [00:00, 12162227.73it/s]                           
0it [00:00, ?it/s]8192it [00:00, 263414.12it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

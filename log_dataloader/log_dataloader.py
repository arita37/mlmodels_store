
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/415bd506b53201365532ee334feb7ea400b6dd3b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '415bd506b53201365532ee334feb7ea400b6dd3b', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/415bd506b53201365532ee334feb7ea400b6dd3b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/415bd506b53201365532ee334feb7ea400b6dd3b

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/415bd506b53201365532ee334feb7ea400b6dd3b

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f24ac48ea60> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f24ac48ea60> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f2517100488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f2517100488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f253632cea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f253632cea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f24c442e9d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f24c442e9d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f24c442e9d8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 47%|████▋     | 4620288/9912422 [00:00<00:00, 44682403.49it/s]9920512it [00:00, 39634573.12it/s]                             
0it [00:00, ?it/s]32768it [00:00, 445061.36it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 150463.36it/s]1654784it [00:00, 10924025.75it/s]                         
0it [00:00, ?it/s]8192it [00:00, 188759.69it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f24ab6cd6a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f24ab6dd940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f24c442e620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f24c442e620> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f24c442e620> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:57,  1.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:57,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:56,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:55,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:54,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:54,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:53,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:52,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:19,  1.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:19,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:18,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:18,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:17,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:17,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:16,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:16,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:15,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:14,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:52,  2.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:52,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:52,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:51,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:51,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:51,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:50,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:50,  2.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:50,  2.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:49,  2.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:01<00:35,  3.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:35,  3.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:34,  3.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:34,  3.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:34,  3.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:33,  3.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:33,  3.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:33,  3.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:33,  3.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:32,  3.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:23,  5.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:23,  5.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:23,  5.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:22,  5.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:22,  5.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:22,  5.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:22,  5.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:22,  5.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:22,  5.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:21,  5.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:21,  5.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:15,  7.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:15,  7.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:15,  7.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:15,  7.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:15,  7.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:14,  7.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:14,  7.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:14,  7.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:14,  7.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:14,  7.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:10, 10.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:10, 10.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:10, 10.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:09, 10.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:09, 10.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:09, 10.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:09, 10.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:09, 10.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:09, 10.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:06, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:06, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:06, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:06, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:06, 14.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 18.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 18.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 18.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 18.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 18.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 18.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 18.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 18.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:04, 18.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:04, 18.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 24.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 24.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 24.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 24.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 24.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 24.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 24.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:03, 24.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 24.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 24.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 30.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 30.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 30.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 30.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 30.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 30.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 30.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 30.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 30.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:02, 30.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 38.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 38.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 38.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 38.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 38.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 38.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 38.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 38.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 38.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 38.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 45.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 45.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 53.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 53.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 53.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 53.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 53.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 53.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 53.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 53.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 53.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 53.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 61.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 61.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 61.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 61.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 61.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 61.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 61.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 61.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 61.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 61.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 67.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 67.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 67.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 67.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 67.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 72.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 72.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 72.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 72.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 72.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 72.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 72.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 72.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 72.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.64s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.64s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.64s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.64 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.93s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.64s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 72.64 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.93s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.94s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.83 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.94s/ url]
0 examples [00:00, ? examples/s]2020-07-24 17:14:43.340319: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-24 17:14:43.353704: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-07-24 17:14:43.353878: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cfeeec3cf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-24 17:14:43.353897: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
7 examples [00:00, 69.97 examples/s]91 examples [00:00, 96.51 examples/s]178 examples [00:00, 131.60 examples/s]277 examples [00:00, 177.86 examples/s]367 examples [00:00, 233.73 examples/s]455 examples [00:00, 299.62 examples/s]554 examples [00:00, 378.63 examples/s]655 examples [00:00, 465.53 examples/s]756 examples [00:00, 555.20 examples/s]859 examples [00:01, 643.39 examples/s]959 examples [00:01, 719.84 examples/s]1060 examples [00:01, 787.68 examples/s]1160 examples [00:01, 839.86 examples/s]1259 examples [00:01, 869.93 examples/s]1357 examples [00:01, 896.56 examples/s]1455 examples [00:01, 919.00 examples/s]1553 examples [00:01, 934.12 examples/s]1654 examples [00:01, 955.15 examples/s]1754 examples [00:01, 968.09 examples/s]1856 examples [00:02, 981.30 examples/s]1956 examples [00:02, 986.17 examples/s]2057 examples [00:02, 991.24 examples/s]2157 examples [00:02, 992.50 examples/s]2258 examples [00:02, 996.96 examples/s]2360 examples [00:02, 1002.60 examples/s]2461 examples [00:02, 997.83 examples/s] 2561 examples [00:02, 997.22 examples/s]2661 examples [00:02, 991.12 examples/s]2761 examples [00:02, 973.75 examples/s]2859 examples [00:03, 974.19 examples/s]2957 examples [00:03, 973.11 examples/s]3055 examples [00:03, 971.98 examples/s]3155 examples [00:03, 977.66 examples/s]3257 examples [00:03, 988.19 examples/s]3357 examples [00:03, 989.88 examples/s]3457 examples [00:03, 983.05 examples/s]3557 examples [00:03, 982.07 examples/s]3656 examples [00:03, 963.34 examples/s]3753 examples [00:03, 957.79 examples/s]3851 examples [00:04, 964.14 examples/s]3953 examples [00:04, 977.91 examples/s]4054 examples [00:04, 986.57 examples/s]4153 examples [00:04, 977.00 examples/s]4255 examples [00:04, 989.23 examples/s]4357 examples [00:04, 998.06 examples/s]4460 examples [00:04, 1005.29 examples/s]4561 examples [00:04, 992.02 examples/s] 4661 examples [00:04, 987.68 examples/s]4764 examples [00:04, 998.11 examples/s]4864 examples [00:05, 991.69 examples/s]4964 examples [00:05, 993.52 examples/s]5064 examples [00:05, 985.79 examples/s]5167 examples [00:05, 996.23 examples/s]5267 examples [00:05, 974.32 examples/s]5368 examples [00:05, 983.53 examples/s]5467 examples [00:05, 976.79 examples/s]5565 examples [00:05, 961.09 examples/s]5666 examples [00:05, 973.26 examples/s]5767 examples [00:05, 982.88 examples/s]5866 examples [00:06, 951.65 examples/s]5962 examples [00:06, 944.19 examples/s]6063 examples [00:06, 960.85 examples/s]6165 examples [00:06, 977.58 examples/s]6268 examples [00:06, 992.25 examples/s]6368 examples [00:06, 941.06 examples/s]6463 examples [00:06, 915.98 examples/s]6556 examples [00:06, 914.53 examples/s]6648 examples [00:06, 909.98 examples/s]6740 examples [00:07, 901.30 examples/s]6840 examples [00:07, 926.46 examples/s]6939 examples [00:07, 943.02 examples/s]7034 examples [00:07, 944.19 examples/s]7129 examples [00:07, 938.36 examples/s]7223 examples [00:07, 936.91 examples/s]7317 examples [00:07, 932.22 examples/s]7412 examples [00:07, 936.99 examples/s]7512 examples [00:07, 954.25 examples/s]7613 examples [00:07, 968.47 examples/s]7713 examples [00:08, 977.33 examples/s]7814 examples [00:08, 985.77 examples/s]7914 examples [00:08, 987.86 examples/s]8015 examples [00:08, 993.30 examples/s]8117 examples [00:08, 1000.51 examples/s]8219 examples [00:08, 1003.76 examples/s]8320 examples [00:08, 970.72 examples/s] 8418 examples [00:08, 925.24 examples/s]8512 examples [00:08, 925.66 examples/s]8606 examples [00:08, 928.52 examples/s]8704 examples [00:09, 943.30 examples/s]8799 examples [00:09, 923.50 examples/s]8897 examples [00:09, 938.02 examples/s]8994 examples [00:09, 944.87 examples/s]9089 examples [00:09, 939.59 examples/s]9184 examples [00:09, 927.94 examples/s]9282 examples [00:09, 942.95 examples/s]9377 examples [00:09, 900.09 examples/s]9468 examples [00:09, 877.05 examples/s]9557 examples [00:10, 870.84 examples/s]9645 examples [00:10, 871.54 examples/s]9733 examples [00:10, 870.86 examples/s]9826 examples [00:10, 887.13 examples/s]9915 examples [00:10, 887.74 examples/s]10004 examples [00:10, 873.17 examples/s]10103 examples [00:10, 902.83 examples/s]10198 examples [00:10, 915.26 examples/s]10297 examples [00:10, 935.04 examples/s]10396 examples [00:10, 948.81 examples/s]10497 examples [00:11, 964.31 examples/s]10598 examples [00:11, 977.12 examples/s]10699 examples [00:11, 984.97 examples/s]10798 examples [00:11, 986.37 examples/s]10900 examples [00:11, 993.44 examples/s]11001 examples [00:11, 996.69 examples/s]11101 examples [00:11, 986.69 examples/s]11202 examples [00:11, 992.99 examples/s]11302 examples [00:11, 983.51 examples/s]11402 examples [00:11, 987.04 examples/s]11503 examples [00:12, 992.36 examples/s]11603 examples [00:12, 990.67 examples/s]11703 examples [00:12, 991.78 examples/s]11803 examples [00:12, 988.81 examples/s]11902 examples [00:12, 977.57 examples/s]12000 examples [00:12, 960.33 examples/s]12097 examples [00:12, 959.60 examples/s]12194 examples [00:12, 952.83 examples/s]12290 examples [00:12, 913.20 examples/s]12382 examples [00:12, 907.25 examples/s]12474 examples [00:13, 908.91 examples/s]12566 examples [00:13, 889.28 examples/s]12656 examples [00:13, 891.76 examples/s]12748 examples [00:13, 899.43 examples/s]12847 examples [00:13, 922.31 examples/s]12946 examples [00:13, 939.45 examples/s]13041 examples [00:13, 929.37 examples/s]13135 examples [00:13, 918.59 examples/s]13228 examples [00:13, 892.78 examples/s]13322 examples [00:14, 905.08 examples/s]13416 examples [00:14, 915.08 examples/s]13510 examples [00:14, 921.59 examples/s]13606 examples [00:14, 930.62 examples/s]13700 examples [00:14, 929.01 examples/s]13793 examples [00:14, 917.03 examples/s]13886 examples [00:14, 919.57 examples/s]13987 examples [00:14, 942.79 examples/s]14082 examples [00:14, 921.26 examples/s]14180 examples [00:14, 936.74 examples/s]14274 examples [00:15, 937.62 examples/s]14374 examples [00:15, 954.37 examples/s]14474 examples [00:15, 967.37 examples/s]14572 examples [00:15, 969.70 examples/s]14670 examples [00:15, 944.87 examples/s]14765 examples [00:15, 932.09 examples/s]14859 examples [00:15, 931.31 examples/s]14953 examples [00:15, 928.67 examples/s]15048 examples [00:15, 932.74 examples/s]15142 examples [00:15, 916.10 examples/s]15234 examples [00:16, 912.39 examples/s]15326 examples [00:16, 909.03 examples/s]15417 examples [00:16, 900.04 examples/s]15509 examples [00:16, 902.81 examples/s]15605 examples [00:16, 918.73 examples/s]15697 examples [00:16, 915.38 examples/s]15789 examples [00:16, 892.10 examples/s]15879 examples [00:16, 892.89 examples/s]15970 examples [00:16, 897.64 examples/s]16060 examples [00:16, 894.83 examples/s]16153 examples [00:17, 903.31 examples/s]16246 examples [00:17, 910.63 examples/s]16339 examples [00:17, 915.78 examples/s]16436 examples [00:17, 931.31 examples/s]16537 examples [00:17, 952.14 examples/s]16636 examples [00:17, 960.99 examples/s]16737 examples [00:17, 974.78 examples/s]16836 examples [00:17, 977.83 examples/s]16934 examples [00:17, 958.43 examples/s]17031 examples [00:18, 941.20 examples/s]17127 examples [00:18, 945.63 examples/s]17223 examples [00:18, 948.58 examples/s]17324 examples [00:18, 965.86 examples/s]17421 examples [00:18, 965.16 examples/s]17522 examples [00:18, 977.48 examples/s]17620 examples [00:18, 968.94 examples/s]17718 examples [00:18, 971.95 examples/s]17817 examples [00:18, 976.07 examples/s]17915 examples [00:18, 944.14 examples/s]18015 examples [00:19, 959.36 examples/s]18112 examples [00:19, 962.18 examples/s]18209 examples [00:19, 908.92 examples/s]18301 examples [00:19, 907.25 examples/s]18399 examples [00:19, 926.67 examples/s]18501 examples [00:19, 951.85 examples/s]18604 examples [00:19, 971.19 examples/s]18705 examples [00:19, 980.35 examples/s]18804 examples [00:19, 973.48 examples/s]18902 examples [00:19, 971.75 examples/s]19002 examples [00:20, 977.84 examples/s]19100 examples [00:20, 968.79 examples/s]19202 examples [00:20, 981.11 examples/s]19303 examples [00:20, 989.44 examples/s]19403 examples [00:20, 989.54 examples/s]19504 examples [00:20, 994.52 examples/s]19604 examples [00:20, 971.07 examples/s]19705 examples [00:20, 981.52 examples/s]19804 examples [00:20, 946.31 examples/s]19903 examples [00:20, 957.18 examples/s]20001 examples [00:21, 916.78 examples/s]20100 examples [00:21, 935.34 examples/s]20199 examples [00:21, 950.19 examples/s]20297 examples [00:21, 958.27 examples/s]20398 examples [00:21, 972.48 examples/s]20496 examples [00:21, 971.84 examples/s]20594 examples [00:21, 971.84 examples/s]20692 examples [00:21, 967.63 examples/s]20789 examples [00:21, 950.55 examples/s]20885 examples [00:22, 944.20 examples/s]20980 examples [00:22, 932.53 examples/s]21074 examples [00:22, 909.67 examples/s]21166 examples [00:22, 911.75 examples/s]21263 examples [00:22, 925.27 examples/s]21363 examples [00:22, 944.44 examples/s]21463 examples [00:22, 959.71 examples/s]21567 examples [00:22, 980.07 examples/s]21666 examples [00:22, 977.68 examples/s]21764 examples [00:22, 948.16 examples/s]21862 examples [00:23, 956.53 examples/s]21960 examples [00:23, 961.85 examples/s]22057 examples [00:23, 946.31 examples/s]22158 examples [00:23, 964.17 examples/s]22257 examples [00:23, 970.62 examples/s]22357 examples [00:23, 977.86 examples/s]22455 examples [00:23, 954.89 examples/s]22551 examples [00:23, 954.03 examples/s]22647 examples [00:23, 952.50 examples/s]22743 examples [00:23, 941.02 examples/s]22838 examples [00:24, 909.79 examples/s]22941 examples [00:24, 942.47 examples/s]23043 examples [00:24, 962.07 examples/s]23146 examples [00:24, 980.06 examples/s]23245 examples [00:24, 959.06 examples/s]23342 examples [00:24, 947.61 examples/s]23438 examples [00:24, 927.73 examples/s]23535 examples [00:24, 939.63 examples/s]23634 examples [00:24, 953.99 examples/s]23733 examples [00:25, 962.76 examples/s]23833 examples [00:25, 972.32 examples/s]23934 examples [00:25, 980.90 examples/s]24033 examples [00:25, 978.64 examples/s]24131 examples [00:25, 972.93 examples/s]24229 examples [00:25, 954.34 examples/s]24330 examples [00:25, 969.96 examples/s]24428 examples [00:25, 968.70 examples/s]24525 examples [00:25, 925.09 examples/s]24618 examples [00:25, 907.30 examples/s]24710 examples [00:26, 903.95 examples/s]24804 examples [00:26, 912.30 examples/s]24899 examples [00:26, 923.23 examples/s]25000 examples [00:26, 947.39 examples/s]25096 examples [00:26, 937.56 examples/s]25196 examples [00:26, 953.44 examples/s]25297 examples [00:26, 967.36 examples/s]25396 examples [00:26, 972.93 examples/s]25497 examples [00:26, 982.43 examples/s]25596 examples [00:26, 984.04 examples/s]25696 examples [00:27, 986.70 examples/s]25796 examples [00:27, 989.32 examples/s]25897 examples [00:27, 994.19 examples/s]25998 examples [00:27, 997.84 examples/s]26098 examples [00:27, 992.36 examples/s]26198 examples [00:27, 974.52 examples/s]26296 examples [00:27, 965.29 examples/s]26393 examples [00:27, 959.25 examples/s]26489 examples [00:27, 944.15 examples/s]26588 examples [00:27, 955.58 examples/s]26684 examples [00:28, 944.58 examples/s]26781 examples [00:28, 950.47 examples/s]26879 examples [00:28, 957.38 examples/s]26975 examples [00:28, 937.43 examples/s]27069 examples [00:28, 904.74 examples/s]27160 examples [00:28, 897.01 examples/s]27250 examples [00:28, 897.55 examples/s]27340 examples [00:28, 878.48 examples/s]27437 examples [00:28, 902.38 examples/s]27528 examples [00:29, 903.52 examples/s]27625 examples [00:29, 920.39 examples/s]27718 examples [00:29, 891.72 examples/s]27817 examples [00:29, 914.69 examples/s]27909 examples [00:29, 900.44 examples/s]28002 examples [00:29, 906.91 examples/s]28102 examples [00:29, 931.38 examples/s]28201 examples [00:29, 945.55 examples/s]28296 examples [00:29, 935.33 examples/s]28390 examples [00:29, 902.06 examples/s]28481 examples [00:30, 898.38 examples/s]28572 examples [00:30, 899.30 examples/s]28664 examples [00:30, 905.27 examples/s]28755 examples [00:30, 901.54 examples/s]28847 examples [00:30, 905.22 examples/s]28938 examples [00:30, 867.24 examples/s]29026 examples [00:30, 837.90 examples/s]29122 examples [00:30, 869.00 examples/s]29220 examples [00:30, 896.44 examples/s]29311 examples [00:30, 889.29 examples/s]29401 examples [00:31, 863.33 examples/s]29488 examples [00:31, 864.29 examples/s]29575 examples [00:31, 865.49 examples/s]29662 examples [00:31, 863.18 examples/s]29754 examples [00:31, 878.44 examples/s]29852 examples [00:31, 905.96 examples/s]29947 examples [00:31, 917.11 examples/s]30039 examples [00:31, 864.22 examples/s]30129 examples [00:31, 874.20 examples/s]30218 examples [00:32, 876.48 examples/s]30310 examples [00:32, 889.08 examples/s]30402 examples [00:32, 897.57 examples/s]30496 examples [00:32, 907.95 examples/s]30591 examples [00:32, 919.96 examples/s]30684 examples [00:32, 905.16 examples/s]30775 examples [00:32, 901.14 examples/s]30866 examples [00:32, 898.91 examples/s]30956 examples [00:32, 895.61 examples/s]31052 examples [00:32, 911.59 examples/s]31148 examples [00:33, 924.65 examples/s]31245 examples [00:33, 934.10 examples/s]31342 examples [00:33, 944.52 examples/s]31440 examples [00:33, 953.58 examples/s]31541 examples [00:33, 967.06 examples/s]31638 examples [00:33, 948.35 examples/s]31733 examples [00:33, 926.67 examples/s]31826 examples [00:33, 886.09 examples/s]31916 examples [00:33, 888.67 examples/s]32006 examples [00:33, 888.00 examples/s]32096 examples [00:34, 873.36 examples/s]32184 examples [00:34, 873.05 examples/s]32275 examples [00:34, 882.49 examples/s]32366 examples [00:34, 888.94 examples/s]32458 examples [00:34, 895.73 examples/s]32548 examples [00:34, 893.49 examples/s]32640 examples [00:34, 899.30 examples/s]32737 examples [00:34, 917.63 examples/s]32836 examples [00:34, 936.62 examples/s]32934 examples [00:34, 948.86 examples/s]33030 examples [00:35, 938.73 examples/s]33125 examples [00:35, 919.52 examples/s]33218 examples [00:35, 919.16 examples/s]33311 examples [00:35, 918.88 examples/s]33403 examples [00:35, 889.54 examples/s]33493 examples [00:35, 874.98 examples/s]33588 examples [00:35, 894.31 examples/s]33683 examples [00:35, 909.96 examples/s]33775 examples [00:35, 912.28 examples/s]33867 examples [00:36, 912.33 examples/s]33962 examples [00:36, 922.52 examples/s]34055 examples [00:36, 921.93 examples/s]34148 examples [00:36, 909.85 examples/s]34240 examples [00:36, 892.71 examples/s]34337 examples [00:36, 912.79 examples/s]34436 examples [00:36, 934.38 examples/s]34531 examples [00:36, 938.72 examples/s]34630 examples [00:36, 950.91 examples/s]34730 examples [00:36, 964.17 examples/s]34830 examples [00:37, 974.44 examples/s]34929 examples [00:37, 976.62 examples/s]35027 examples [00:37, 966.19 examples/s]35128 examples [00:37, 976.35 examples/s]35226 examples [00:37, 955.22 examples/s]35326 examples [00:37, 967.52 examples/s]35423 examples [00:37, 952.86 examples/s]35519 examples [00:37, 921.57 examples/s]35612 examples [00:37, 919.85 examples/s]35705 examples [00:37, 920.22 examples/s]35798 examples [00:38, 889.80 examples/s]35895 examples [00:38, 910.87 examples/s]35995 examples [00:38, 935.86 examples/s]36094 examples [00:38, 950.79 examples/s]36192 examples [00:38, 958.14 examples/s]36289 examples [00:38, 931.35 examples/s]36387 examples [00:38, 943.43 examples/s]36483 examples [00:38, 947.52 examples/s]36584 examples [00:38, 962.78 examples/s]36681 examples [00:39, 959.35 examples/s]36783 examples [00:39, 976.18 examples/s]36882 examples [00:39, 979.77 examples/s]36983 examples [00:39, 987.47 examples/s]37082 examples [00:39, 977.13 examples/s]37180 examples [00:39, 968.75 examples/s]37278 examples [00:39, 970.14 examples/s]37376 examples [00:39, 958.48 examples/s]37472 examples [00:39, 939.59 examples/s]37567 examples [00:39, 928.15 examples/s]37660 examples [00:40, 927.52 examples/s]37753 examples [00:40, 900.22 examples/s]37848 examples [00:40, 912.42 examples/s]37946 examples [00:40, 930.02 examples/s]38045 examples [00:40, 945.60 examples/s]38147 examples [00:40, 964.57 examples/s]38247 examples [00:40, 974.03 examples/s]38345 examples [00:40, 965.58 examples/s]38442 examples [00:40, 918.15 examples/s]38535 examples [00:40, 894.17 examples/s]38632 examples [00:41, 913.80 examples/s]38732 examples [00:41, 936.19 examples/s]38828 examples [00:41, 941.24 examples/s]38923 examples [00:41, 919.65 examples/s]39021 examples [00:41, 933.75 examples/s]39119 examples [00:41, 945.61 examples/s]39219 examples [00:41, 961.06 examples/s]39316 examples [00:41, 943.87 examples/s]39411 examples [00:41, 945.36 examples/s]39514 examples [00:42, 963.14 examples/s]39615 examples [00:42, 976.58 examples/s]39719 examples [00:42, 992.10 examples/s]39819 examples [00:42, 993.83 examples/s]39919 examples [00:42, 993.60 examples/s]40019 examples [00:42, 936.69 examples/s]40116 examples [00:42, 943.51 examples/s]40216 examples [00:42, 955.37 examples/s]40312 examples [00:42, 925.06 examples/s]40413 examples [00:42, 946.53 examples/s]40513 examples [00:43, 961.21 examples/s]40613 examples [00:43, 971.96 examples/s]40714 examples [00:43, 981.96 examples/s]40814 examples [00:43, 984.84 examples/s]40915 examples [00:43, 991.67 examples/s]41016 examples [00:43, 996.60 examples/s]41117 examples [00:43, 998.45 examples/s]41217 examples [00:43, 981.08 examples/s]41316 examples [00:43, 979.53 examples/s]41417 examples [00:43, 987.27 examples/s]41517 examples [00:44, 990.64 examples/s]41617 examples [00:44, 980.98 examples/s]41717 examples [00:44, 985.82 examples/s]41818 examples [00:44, 991.59 examples/s]41918 examples [00:44, 991.64 examples/s]42018 examples [00:44, 992.83 examples/s]42119 examples [00:44, 996.28 examples/s]42219 examples [00:44, 975.75 examples/s]42317 examples [00:44, 966.66 examples/s]42415 examples [00:44, 968.15 examples/s]42513 examples [00:45, 971.03 examples/s]42611 examples [00:45, 965.90 examples/s]42708 examples [00:45, 966.21 examples/s]42809 examples [00:45, 977.12 examples/s]42911 examples [00:45, 986.92 examples/s]43010 examples [00:45, 971.25 examples/s]43111 examples [00:45, 980.37 examples/s]43210 examples [00:45, 949.73 examples/s]43310 examples [00:45, 962.12 examples/s]43409 examples [00:45, 969.85 examples/s]43509 examples [00:46, 976.48 examples/s]43607 examples [00:46, 972.85 examples/s]43706 examples [00:46, 975.51 examples/s]43804 examples [00:46, 973.08 examples/s]43906 examples [00:46, 985.80 examples/s]44006 examples [00:46, 987.48 examples/s]44105 examples [00:46, 979.95 examples/s]44204 examples [00:46, 943.63 examples/s]44299 examples [00:46, 937.29 examples/s]44393 examples [00:47, 935.81 examples/s]44487 examples [00:47, 921.88 examples/s]44585 examples [00:47, 937.69 examples/s]44680 examples [00:47, 939.09 examples/s]44778 examples [00:47, 949.46 examples/s]44876 examples [00:47, 957.61 examples/s]44973 examples [00:47, 961.02 examples/s]45072 examples [00:47, 968.49 examples/s]45169 examples [00:47, 956.12 examples/s]45271 examples [00:47, 973.83 examples/s]45371 examples [00:48, 979.72 examples/s]45470 examples [00:48, 982.76 examples/s]45571 examples [00:48, 989.75 examples/s]45671 examples [00:48, 984.44 examples/s]45774 examples [00:48, 993.18 examples/s]45875 examples [00:48, 996.22 examples/s]45976 examples [00:48, 998.36 examples/s]46076 examples [00:48, 990.39 examples/s]46176 examples [00:48, 948.07 examples/s]46272 examples [00:48, 941.43 examples/s]46367 examples [00:49, 935.80 examples/s]46461 examples [00:49, 934.17 examples/s]46557 examples [00:49, 939.40 examples/s]46655 examples [00:49, 950.77 examples/s]46754 examples [00:49, 962.16 examples/s]46855 examples [00:49, 973.71 examples/s]46956 examples [00:49, 982.72 examples/s]47055 examples [00:49, 969.67 examples/s]47153 examples [00:49, 940.56 examples/s]47248 examples [00:49, 939.78 examples/s]47343 examples [00:50, 934.98 examples/s]47444 examples [00:50, 954.57 examples/s]47544 examples [00:50, 965.55 examples/s]47645 examples [00:50, 976.43 examples/s]47746 examples [00:50, 985.32 examples/s]47847 examples [00:50, 991.46 examples/s]47948 examples [00:50, 994.73 examples/s]48048 examples [00:50, 994.70 examples/s]48148 examples [00:50, 988.76 examples/s]48249 examples [00:50, 994.56 examples/s]48350 examples [00:51, 996.80 examples/s]48451 examples [00:51, 999.26 examples/s]48551 examples [00:51, 998.81 examples/s]48651 examples [00:51, 996.98 examples/s]48753 examples [00:51, 1001.96 examples/s]48854 examples [00:51, 1002.98 examples/s]48955 examples [00:51, 1005.03 examples/s]49056 examples [00:51, 950.87 examples/s] 49152 examples [00:51, 937.85 examples/s]49247 examples [00:52, 936.83 examples/s]49342 examples [00:52, 939.49 examples/s]49442 examples [00:52, 955.28 examples/s]49538 examples [00:52, 942.76 examples/s]49639 examples [00:52, 961.89 examples/s]49742 examples [00:52, 979.02 examples/s]49845 examples [00:52, 993.65 examples/s]49948 examples [00:52, 1002.56 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6683/50000 [00:00<00:00, 66826.56 examples/s] 35%|███▌      | 17680/50000 [00:00<00:00, 75740.50 examples/s] 58%|█████▊    | 28819/50000 [00:00<00:00, 83784.16 examples/s] 81%|████████▏ | 40737/50000 [00:00<00:00, 91978.74 examples/s]                                                               0 examples [00:00, ? examples/s]68 examples [00:00, 676.07 examples/s]163 examples [00:00, 739.59 examples/s]256 examples [00:00, 787.63 examples/s]356 examples [00:00, 839.59 examples/s]458 examples [00:00, 884.56 examples/s]558 examples [00:00, 915.98 examples/s]662 examples [00:00, 948.64 examples/s]763 examples [00:00, 964.13 examples/s]860 examples [00:00, 963.23 examples/s]961 examples [00:01, 974.95 examples/s]1058 examples [00:01, 972.93 examples/s]1159 examples [00:01, 981.48 examples/s]1260 examples [00:01, 988.26 examples/s]1359 examples [00:01, 908.82 examples/s]1451 examples [00:01, 788.96 examples/s]1546 examples [00:01, 830.39 examples/s]1644 examples [00:01, 870.00 examples/s]1748 examples [00:01, 913.07 examples/s]1853 examples [00:01, 948.17 examples/s]1955 examples [00:02, 967.47 examples/s]2060 examples [00:02, 990.83 examples/s]2164 examples [00:02, 1004.01 examples/s]2269 examples [00:02, 1014.70 examples/s]2372 examples [00:02, 995.67 examples/s] 2473 examples [00:02, 974.31 examples/s]2571 examples [00:02, 967.40 examples/s]2669 examples [00:02, 930.47 examples/s]2772 examples [00:02, 957.01 examples/s]2874 examples [00:03, 973.28 examples/s]2974 examples [00:03, 979.81 examples/s]3075 examples [00:03, 986.51 examples/s]3177 examples [00:03, 994.48 examples/s]3280 examples [00:03, 1003.54 examples/s]3382 examples [00:03, 1008.24 examples/s]3483 examples [00:03, 911.44 examples/s] 3576 examples [00:03, 885.65 examples/s]3667 examples [00:03, 890.87 examples/s]3761 examples [00:03, 904.70 examples/s]3853 examples [00:04, 906.81 examples/s]3949 examples [00:04, 921.58 examples/s]4051 examples [00:04, 947.93 examples/s]4152 examples [00:04, 965.20 examples/s]4253 examples [00:04, 975.92 examples/s]4354 examples [00:04, 983.82 examples/s]4455 examples [00:04, 990.30 examples/s]4555 examples [00:04, 974.19 examples/s]4658 examples [00:04, 989.86 examples/s]4761 examples [00:04, 1000.87 examples/s]4862 examples [00:05, 1002.84 examples/s]4963 examples [00:05, 1004.22 examples/s]5065 examples [00:05, 1008.02 examples/s]5167 examples [00:05, 1010.46 examples/s]5269 examples [00:05, 990.82 examples/s] 5369 examples [00:05, 969.66 examples/s]5467 examples [00:05, 969.86 examples/s]5567 examples [00:05, 976.71 examples/s]5667 examples [00:05, 981.38 examples/s]5767 examples [00:06, 984.65 examples/s]5866 examples [00:06, 978.71 examples/s]5965 examples [00:06, 978.57 examples/s]6064 examples [00:06, 980.17 examples/s]6164 examples [00:06, 984.55 examples/s]6263 examples [00:06, 983.89 examples/s]6362 examples [00:06, 935.00 examples/s]6461 examples [00:06, 949.09 examples/s]6561 examples [00:06, 961.19 examples/s]6658 examples [00:06, 959.66 examples/s]6760 examples [00:07, 976.41 examples/s]6860 examples [00:07, 982.83 examples/s]6962 examples [00:07, 992.58 examples/s]7062 examples [00:07, 990.87 examples/s]7162 examples [00:07, 993.53 examples/s]7265 examples [00:07, 1001.59 examples/s]7366 examples [00:07, 978.70 examples/s] 7465 examples [00:07, 972.56 examples/s]7563 examples [00:07, 965.47 examples/s]7663 examples [00:07, 974.88 examples/s]7765 examples [00:08, 986.88 examples/s]7865 examples [00:08, 990.15 examples/s]7967 examples [00:08, 998.31 examples/s]8070 examples [00:08, 1005.62 examples/s]8171 examples [00:08, 995.53 examples/s] 8271 examples [00:08, 985.56 examples/s]8370 examples [00:08, 984.88 examples/s]8469 examples [00:08, 885.63 examples/s]8562 examples [00:08, 897.06 examples/s]8659 examples [00:09, 916.92 examples/s]8757 examples [00:09, 933.09 examples/s]8856 examples [00:09, 949.17 examples/s]8957 examples [00:09, 965.40 examples/s]9057 examples [00:09, 975.29 examples/s]9160 examples [00:09, 988.34 examples/s]9261 examples [00:09, 994.12 examples/s]9363 examples [00:09, 1000.74 examples/s]9466 examples [00:09, 1007.19 examples/s]9567 examples [00:09, 1006.56 examples/s]9670 examples [00:10, 1011.16 examples/s]9772 examples [00:10, 1005.84 examples/s]9873 examples [00:10, 997.28 examples/s] 9973 examples [00:10, 981.39 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete5QV9Q7/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete5QV9Q7/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f24c442e9d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f24c442e9d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f24c442e9d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f244e7b8470>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f244e7ad8d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f24c442e9d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f24c442e9d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f24c442e9d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f24ab6dd278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f24ab6dd080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f243ce10268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f243ce10268> 

  function with postional parmater data_info <function split_train_valid at 0x7f243ce10268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f243ce10378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f243ce10378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f243ce10378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=48d4463736ade707ad39c873f8a134b21515d8266ddbed07cc96e9f9ca4e3fda
  Stored in directory: /tmp/pip-ephem-wheel-cache-_vx34nzp/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:02:44, 10.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:39:42, 15.3kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:01<11:01:08, 21.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 868k/862M [00:01<7:43:09, 31.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.85M/862M [00:01<5:23:40, 44.2kB/s].vector_cache/glove.6B.zip:   1%|          | 6.67M/862M [00:01<3:45:42, 63.2kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:01<2:37:00, 90.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.7M/862M [00:01<1:49:37, 129kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:01<1:16:18, 184kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.1M/862M [00:01<53:07, 262kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 30.0M/862M [00:01<37:12, 373kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.7M/862M [00:02<25:57, 531kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.5M/862M [00:02<18:08, 754kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.1M/862M [00:02<12:42, 1.07MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:02<09:00, 1.50MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<06:44, 1.99MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:04<11:51:54, 18.9kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<8:18:53, 26.9kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:04<5:48:25, 38.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<4:09:07, 53.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<2:57:35, 75.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<2:04:58, 107kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.1M/862M [00:06<1:27:20, 152kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<1:11:51, 185kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<51:36, 258kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:08<36:23, 365kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:10<28:32, 464kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<21:19, 621kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:10<15:13, 867kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:12<13:45, 958kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:12<12:17, 1.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<09:09, 1.44MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.6M/862M [00:12<06:32, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:14<17:56, 730kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<13:54, 942kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:14<10:02, 1.30MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:16<10:04, 1.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:16<09:39, 1.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<07:18, 1.78MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:16<05:16, 2.46MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:18<11:32, 1.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<09:24, 1.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:18<06:54, 1.87MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:20<07:49, 1.65MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<08:11, 1.57MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<06:19, 2.04MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:20<04:34, 2.81MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:22<10:51, 1.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<08:54, 1.44MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:22<06:30, 1.97MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:23<07:32, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:24<07:50, 1.63MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<06:01, 2.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<04:21, 2.91MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<12:20, 1.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<09:54, 1.28MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<07:14, 1.75MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<08:01, 1.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<08:08, 1.55MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:14, 2.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:30, 2.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<11:09, 1.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<09:05, 1.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:40, 1.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<07:35, 1.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<06:37, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<04:56, 2.52MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<06:22, 1.95MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<06:58, 1.78MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:30, 2.25MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<03:58, 3.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<1:32:15, 134kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<1:05:34, 188kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<46:07, 267kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<35:06, 350kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<26:57, 455kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<19:22, 633kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<13:41, 893kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<15:27, 790kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<12:03, 1.01MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<08:44, 1.39MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:41<08:57, 1.36MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<08:42, 1.39MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<06:37, 1.83MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<04:45, 2.54MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<16:33, 729kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<12:48, 943kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<09:12, 1.31MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<09:15, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<08:53, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<06:48, 1.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<05:17, 2.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<8:06:40, 24.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<5:41:07, 35.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:47<3:58:08, 49.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<2:52:27, 68.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<2:03:07, 96.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<1:26:41, 137kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<1:02:20, 189kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<44:47, 263kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<31:32, 373kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<24:47, 473kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<19:42, 596kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<14:17, 821kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<10:05, 1.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<16:39, 700kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<12:51, 907kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<09:16, 1.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<09:12, 1.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<08:45, 1.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:42, 1.73MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<06:33, 1.76MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:46, 2.00MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<04:16, 2.69MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<05:39, 2.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<06:15, 1.83MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<04:51, 2.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<03:32, 3.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<08:51, 1.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<07:21, 1.55MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:25, 2.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<06:26, 1.76MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<06:53, 1.64MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:23, 2.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<03:53, 2.90MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<43:29, 259kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<31:24, 358kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<22:14, 505kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<18:06, 618kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<13:47, 811kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<09:54, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<09:33, 1.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<07:48, 1.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:43, 1.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<06:37, 1.67MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<06:51, 1.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:15, 2.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:49, 2.88MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<08:23, 1.31MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<07:00, 1.57MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:07, 2.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:08, 1.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<05:23, 2.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<03:59, 2.72MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<05:23, 2.01MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<05:57, 1.82MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<04:37, 2.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:20, 3.23MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<16:50, 640kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<12:53, 836kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:20<09:16, 1.16MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<08:59, 1.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<08:26, 1.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<06:26, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<06:13, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<06:30, 1.64MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<05:00, 2.12MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<03:37, 2.92MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<09:54, 1.07MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<08:00, 1.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<05:52, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<06:32, 1.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<06:40, 1.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<05:06, 2.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<03:42, 2.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<04:52, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<7:07:23, 24.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<4:59:29, 34.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<3:28:54, 49.7kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<2:32:05, 68.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<1:48:36, 95.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<1:16:22, 136kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<53:23, 193kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<43:19, 238kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<31:21, 328kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<22:09, 463kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<17:50, 574kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<14:32, 703kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<10:41, 956kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<09:05, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<07:24, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<05:26, 1.86MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<06:08, 1.64MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<05:18, 1.90MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:55, 2.57MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:08, 1.95MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:34, 2.19MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:24, 2.93MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:45, 2.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:20, 1.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:13, 2.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:33, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:10, 2.36MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<03:10, 3.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:31, 2.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:08, 1.91MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:05, 2.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:48<02:58, 3.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<1:12:01, 135kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<51:21, 190kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<36:06, 269kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<27:27, 353kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<21:09, 458kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<15:12, 636kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<10:43, 898kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<13:33, 709kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<10:28, 918kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:33, 1.27MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<07:30, 1.27MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<07:10, 1.33MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<05:29, 1.73MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:22, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:39, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:26, 2.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:12, 2.94MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<36:20, 259kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<26:22, 357kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<18:38, 503kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<15:11, 615kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<12:30, 747kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<09:07, 1.02MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<06:29, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<10:05, 919kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<08:00, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<05:47, 1.60MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:12, 1.48MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:11, 1.48MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:47, 1.92MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<03:26, 2.66MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<8:52:13, 17.2kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<6:13:12, 24.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<4:20:43, 34.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<3:03:55, 49.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<2:09:35, 69.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<1:30:40, 99.6kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<1:05:20, 138kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<46:35, 193kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<32:44, 274kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<23:22, 382kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<6:39:54, 22.3kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<4:40:08, 31.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<3:15:15, 45.5kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<2:21:43, 62.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<1:41:04, 87.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<1:11:06, 124kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:15<49:36, 177kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<1:37:35, 90.2kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<1:09:10, 127kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<48:28, 181kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<35:50, 244kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<25:57, 336kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<18:17, 475kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<14:51, 583kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<12:11, 710kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<08:53, 973kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<06:25, 1.34MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:37, 1.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:31, 1.56MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<04:03, 2.11MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:50, 1.76MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:06, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:55, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<02:52, 2.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:21, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:36, 1.83MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:25, 2.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:21, 1.92MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:46, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:45, 2.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<03:58, 2.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<03:37, 2.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<02:44, 3.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<03:51, 2.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:21, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:27, 2.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<02:30, 3.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<12:09, 673kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<09:21, 874kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<06:42, 1.22MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<06:35, 1.23MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:15, 1.29MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:47, 1.69MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:38, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:04, 1.97MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:00, 2.66MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<03:59, 2.00MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:22, 1.82MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:23, 2.35MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<02:30, 3.17MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<04:35, 1.73MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<04:02, 1.96MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:00, 2.61MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:57, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:21, 1.80MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:27, 2.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:39, 2.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:21, 2.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:32, 3.05MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:34, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:05, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<03:14, 2.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:20, 3.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<7:24:54, 17.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<5:11:57, 24.4kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<3:37:49, 34.9kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<2:33:31, 49.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<1:48:59, 69.4kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<1:16:31, 98.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<53:17, 141kB/s]   .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<7:51:17, 15.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<5:30:21, 22.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<3:50:38, 32.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<2:42:25, 45.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<1:55:09, 64.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<1:20:50, 91.7kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<56:33, 130kB/s]   .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<5:44:43, 21.4kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<4:01:23, 30.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<2:48:03, 43.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<2:02:04, 59.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<1:27:01, 83.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<1:01:09, 119kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<42:45, 170kB/s]  .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<32:04, 225kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<23:09, 312kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<16:19, 440kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<13:04, 547kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<09:51, 725kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<07:01, 1.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:35, 1.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:03, 1.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:31, 1.56MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<03:15, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:02, 1.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:14, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:06, 2.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:48, 1.83MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:03, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:09, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<02:17, 3.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<06:21, 1.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:02, 1.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:41, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<04:10, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:18, 1.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:17, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:23, 2.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<05:22, 1.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:25, 1.52MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:15, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:50, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:01, 1.66MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:06, 2.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:14, 2.96MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<09:41, 682kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<07:27, 886kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<05:20, 1.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<05:15, 1.24MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<04:20, 1.50MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:11, 2.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:45, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<03:52, 1.67MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:02, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:08, 2.03MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<02:44, 2.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:04, 3.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:55, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:40, 2.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:01, 3.10MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:53, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:22, 1.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<02:43, 2.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:52, 2.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:38, 2.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<01:59, 3.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:49, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:12, 1.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:33, 2.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<01:50, 3.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<5:49:45, 17.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<4:05:12, 24.7kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<2:51:03, 35.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<2:00:26, 49.7kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<1:24:53, 70.5kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<59:19, 100kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<42:39, 139kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<30:27, 194kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<21:22, 276kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<16:15, 360kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<11:57, 489kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<08:29, 686kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<07:16, 796kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<06:15, 925kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<04:37, 1.25MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:43<03:16, 1.75MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<04:21, 1.31MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<3:51:51, 24.7kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<2:41:49, 35.2kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<1:53:28, 49.8kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<1:20:31, 70.2kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<56:29, 99.8kB/s]  .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<39:19, 142kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<31:16, 179kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<22:25, 249kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<15:44, 353kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<12:14, 450kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<09:40, 569kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<06:59, 786kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<04:55, 1.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<08:29, 642kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<06:29, 838kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<04:39, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<04:30, 1.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:16, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:13, 1.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<02:18, 2.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:43, 1.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:50, 1.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:47, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:10, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:16, 1.60MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:32, 2.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<01:49, 2.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<44:15, 117kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<31:27, 164kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<22:01, 233kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<16:30, 309kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<12:34, 406kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<09:02, 563kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<07:04, 711kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<06:00, 838kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<04:27, 1.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<03:08, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<20:13, 245kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<14:39, 339kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<10:18, 479kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<08:19, 589kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<06:48, 719kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<04:57, 984kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<03:30, 1.38MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<05:44, 842kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<04:29, 1.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:15, 1.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:22, 1.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:19, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:33, 1.86MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:31, 1.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:45, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:07, 2.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:31, 3.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<04:13, 1.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:25, 1.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:28, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:46, 1.64MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:51, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<02:11, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:34, 2.85MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:52, 1.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:10, 1.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<02:19, 1.92MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:38, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:44, 1.61MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:08, 2.06MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:11, 1.99MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:57, 2.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:28, 2.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:01, 2.11MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:16, 1.88MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:48, 2.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:26, 2.93MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<2:52:26, 24.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<2:00:39, 34.9kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<1:23:40, 49.8kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<1:00:36, 68.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<43:17, 95.8kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<30:24, 136kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<21:08, 194kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<16:46, 243kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<12:08, 336kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<08:31, 475kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<06:51, 584kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<05:36, 715kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<04:05, 978kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:34<02:52, 1.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<06:10, 638kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<04:43, 833kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<03:23, 1.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:15, 1.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:05, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:21, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:38<01:40, 2.28MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<14:59, 254kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<10:51, 350kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<07:37, 494kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<06:10, 605kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<04:41, 795kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<03:21, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<03:11, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:58, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:13, 1.64MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:35, 2.28MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<03:54, 919kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<03:06, 1.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:14, 1.59MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:22, 1.48MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:22, 1.48MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:50, 1.92MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:49, 1.89MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:37, 2.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:12, 2.83MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:38, 2.07MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:29, 2.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:06, 3.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:33, 2.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:25, 2.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:04, 3.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:30, 2.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:23, 2.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:02, 3.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:28, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:40, 1.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:17, 2.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<00:56, 3.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:30, 1.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:04, 1.50MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:30, 2.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:45, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:32, 1.98MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:07, 2.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:29, 1.99MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:20, 2.21MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<00:59, 2.96MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:23, 2.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:15, 2.31MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<00:56, 3.04MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:19, 2.14MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:12, 2.34MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<00:54, 3.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:17, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:10, 2.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<00:53, 3.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:15, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:25, 1.90MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:07, 2.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<00:53, 2.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<1:47:44, 24.6kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<1:15:17, 35.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<51:53, 50.0kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<37:28, 68.8kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<26:44, 96.3kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<18:45, 136kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:15<12:55, 194kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<12:59, 193kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<09:19, 268kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<06:30, 380kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<05:04, 480kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<03:47, 642kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<02:41, 896kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:24, 982kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:09, 1.09MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:37, 1.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<01:08, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<2:13:48, 17.2kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<1:33:38, 24.5kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<1:04:52, 35.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<45:12, 49.4kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<32:03, 69.6kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<22:24, 99.0kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<15:26, 141kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<12:01, 180kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<08:36, 250kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<06:00, 355kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<04:37, 453kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<03:26, 607kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:25, 849kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<02:09, 942kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:42, 1.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:13, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:18, 1.50MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:18, 1.49MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<01:00, 1.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:59, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:51, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:38, 2.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:51, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:58, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:45, 2.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:48, 2.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:44, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:33, 3.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:46, 2.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:42, 2.37MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:31, 3.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:44, 2.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:50, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:39, 2.44MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:27, 3.34MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:24, 1.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:08, 1.35MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:49, 1.84MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:54, 1.62MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:55, 1.58MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:43, 2.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<00:30, 2.81MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<1:22:16, 17.1kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<57:28, 24.4kB/s]  .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<39:31, 34.9kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<27:15, 49.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<19:18, 69.3kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<13:26, 98.5kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<09:14, 138kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<06:34, 193kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<04:32, 274kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<03:21, 358kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<02:35, 464kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<01:51, 640kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<01:15, 904kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<09:01, 126kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<06:23, 177kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<04:24, 251kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<03:03, 351kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<48:26, 22.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<33:36, 31.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<22:33, 45.1kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<16:12, 61.9kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<11:31, 86.9kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<08:00, 123kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:00<05:19, 176kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<07:07, 131kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<05:02, 184kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<03:27, 261kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<02:31, 343kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:50, 467kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<01:16, 656kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:02, 767kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:53, 898kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:38, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:26, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:52, 840kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:40, 1.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:28, 1.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:28, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:27, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:20, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:13, 2.60MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:32, 1.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:26, 1.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:18, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:19, 1.62MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:19, 1.58MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:15, 2.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:13, 1.97MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:12, 2.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:08, 2.90MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:11, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:09, 2.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:07, 3.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:09, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:07, 2.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:04, 3.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:16, 903kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:12, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:08, 1.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:07, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:03, 1.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<01:04, 104kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:43, 147kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:23, 209kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:09, 279kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:06, 368kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:03, 513kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 379/400000 [00:00<01:45, 3785.73it/s]  0%|          | 1060/400000 [00:00<01:31, 4367.22it/s]  0%|          | 1723/400000 [00:00<01:21, 4864.29it/s]  1%|          | 2466/400000 [00:00<01:13, 5426.22it/s]  1%|          | 3229/400000 [00:00<01:06, 5940.40it/s]  1%|          | 4001/400000 [00:00<01:02, 6380.64it/s]  1%|          | 4754/400000 [00:00<00:59, 6684.60it/s]  1%|▏         | 5510/400000 [00:00<00:56, 6924.47it/s]  2%|▏         | 6233/400000 [00:00<00:56, 7011.36it/s]  2%|▏         | 6993/400000 [00:01<00:54, 7175.98it/s]  2%|▏         | 7722/400000 [00:01<00:54, 7207.40it/s]  2%|▏         | 8446/400000 [00:01<00:54, 7212.82it/s]  2%|▏         | 9194/400000 [00:01<00:53, 7289.17it/s]  2%|▏         | 9968/400000 [00:01<00:52, 7417.54it/s]  3%|▎         | 10736/400000 [00:01<00:52, 7463.77it/s]  3%|▎         | 11501/400000 [00:01<00:51, 7516.57it/s]  3%|▎         | 12270/400000 [00:01<00:51, 7565.65it/s]  3%|▎         | 13028/400000 [00:01<00:52, 7441.01it/s]  3%|▎         | 13783/400000 [00:01<00:51, 7470.58it/s]  4%|▎         | 14533/400000 [00:02<00:51, 7478.38it/s]  4%|▍         | 15295/400000 [00:02<00:51, 7519.28it/s]  4%|▍         | 16048/400000 [00:02<00:51, 7471.05it/s]  4%|▍         | 16796/400000 [00:02<00:52, 7333.20it/s]  4%|▍         | 17531/400000 [00:02<00:53, 7174.61it/s]  5%|▍         | 18250/400000 [00:02<00:53, 7070.70it/s]  5%|▍         | 18959/400000 [00:02<00:54, 7009.22it/s]  5%|▍         | 19661/400000 [00:02<00:54, 6973.43it/s]  5%|▌         | 20405/400000 [00:02<00:53, 7106.84it/s]  5%|▌         | 21117/400000 [00:02<00:53, 7074.42it/s]  5%|▌         | 21826/400000 [00:03<00:53, 7020.09it/s]  6%|▌         | 22529/400000 [00:03<00:54, 6977.91it/s]  6%|▌         | 23296/400000 [00:03<00:52, 7170.24it/s]  6%|▌         | 24047/400000 [00:03<00:51, 7268.14it/s]  6%|▌         | 24797/400000 [00:03<00:51, 7333.76it/s]  6%|▋         | 25561/400000 [00:03<00:50, 7421.25it/s]  7%|▋         | 26336/400000 [00:03<00:49, 7514.54it/s]  7%|▋         | 27116/400000 [00:03<00:49, 7596.75it/s]  7%|▋         | 27877/400000 [00:03<00:48, 7595.04it/s]  7%|▋         | 28655/400000 [00:03<00:48, 7649.44it/s]  7%|▋         | 29430/400000 [00:04<00:48, 7677.98it/s]  8%|▊         | 30199/400000 [00:04<00:49, 7407.94it/s]  8%|▊         | 30943/400000 [00:04<00:51, 7229.75it/s]  8%|▊         | 31700/400000 [00:04<00:50, 7327.84it/s]  8%|▊         | 32472/400000 [00:04<00:49, 7440.94it/s]  8%|▊         | 33245/400000 [00:04<00:48, 7523.86it/s]  9%|▊         | 34011/400000 [00:04<00:48, 7563.42it/s]  9%|▊         | 34775/400000 [00:04<00:48, 7585.69it/s]  9%|▉         | 35535/400000 [00:04<00:48, 7574.07it/s]  9%|▉         | 36302/400000 [00:04<00:47, 7598.51it/s]  9%|▉         | 37077/400000 [00:05<00:47, 7641.98it/s]  9%|▉         | 37842/400000 [00:05<00:47, 7630.68it/s] 10%|▉         | 38606/400000 [00:05<00:47, 7611.51it/s] 10%|▉         | 39368/400000 [00:05<00:47, 7540.90it/s] 10%|█         | 40123/400000 [00:05<00:49, 7336.29it/s] 10%|█         | 40859/400000 [00:05<00:49, 7201.46it/s] 10%|█         | 41581/400000 [00:05<00:49, 7177.35it/s] 11%|█         | 42342/400000 [00:05<00:48, 7299.83it/s] 11%|█         | 43098/400000 [00:05<00:48, 7375.80it/s] 11%|█         | 43868/400000 [00:05<00:47, 7468.45it/s] 11%|█         | 44634/400000 [00:06<00:47, 7524.43it/s] 11%|█▏        | 45388/400000 [00:06<00:47, 7505.19it/s] 12%|█▏        | 46140/400000 [00:06<00:47, 7505.84it/s] 12%|█▏        | 46894/400000 [00:06<00:46, 7515.62it/s] 12%|█▏        | 47664/400000 [00:06<00:46, 7569.37it/s] 12%|█▏        | 48440/400000 [00:06<00:46, 7624.73it/s] 12%|█▏        | 49218/400000 [00:06<00:45, 7667.79it/s] 12%|█▏        | 49997/400000 [00:06<00:45, 7703.03it/s] 13%|█▎        | 50768/400000 [00:06<00:45, 7644.44it/s] 13%|█▎        | 51533/400000 [00:06<00:45, 7643.46it/s] 13%|█▎        | 52298/400000 [00:07<00:45, 7614.56it/s] 13%|█▎        | 53060/400000 [00:07<00:45, 7603.42it/s] 13%|█▎        | 53821/400000 [00:07<00:45, 7546.78it/s] 14%|█▎        | 54576/400000 [00:07<00:46, 7503.27it/s] 14%|█▍        | 55344/400000 [00:07<00:45, 7555.00it/s] 14%|█▍        | 56110/400000 [00:07<00:45, 7585.95it/s] 14%|█▍        | 56887/400000 [00:07<00:44, 7638.73it/s] 14%|█▍        | 57657/400000 [00:07<00:44, 7655.33it/s] 15%|█▍        | 58423/400000 [00:07<00:44, 7634.78it/s] 15%|█▍        | 59198/400000 [00:07<00:44, 7665.84it/s] 15%|█▍        | 59965/400000 [00:08<00:45, 7510.62it/s] 15%|█▌        | 60735/400000 [00:08<00:44, 7566.15it/s] 15%|█▌        | 61503/400000 [00:08<00:44, 7597.00it/s] 16%|█▌        | 62264/400000 [00:08<00:44, 7578.79it/s] 16%|█▌        | 63030/400000 [00:08<00:44, 7602.16it/s] 16%|█▌        | 63791/400000 [00:08<00:45, 7394.24it/s] 16%|█▌        | 64562/400000 [00:08<00:44, 7484.37it/s] 16%|█▋        | 65337/400000 [00:08<00:44, 7561.82it/s] 17%|█▋        | 66095/400000 [00:08<00:44, 7470.61it/s] 17%|█▋        | 66862/400000 [00:09<00:44, 7529.21it/s] 17%|█▋        | 67632/400000 [00:09<00:43, 7567.61it/s] 17%|█▋        | 68407/400000 [00:09<00:43, 7620.76it/s] 17%|█▋        | 69170/400000 [00:09<00:44, 7365.64it/s] 17%|█▋        | 69909/400000 [00:09<00:46, 7086.49it/s] 18%|█▊        | 70622/400000 [00:09<00:47, 6993.84it/s] 18%|█▊        | 71325/400000 [00:09<00:47, 6954.61it/s] 18%|█▊        | 72053/400000 [00:09<00:46, 7048.12it/s] 18%|█▊        | 72801/400000 [00:09<00:45, 7171.03it/s] 18%|█▊        | 73525/400000 [00:09<00:45, 7191.33it/s] 19%|█▊        | 74267/400000 [00:10<00:44, 7257.56it/s] 19%|█▊        | 74994/400000 [00:10<00:45, 7212.96it/s] 19%|█▉        | 75759/400000 [00:10<00:44, 7336.71it/s] 19%|█▉        | 76513/400000 [00:10<00:43, 7394.80it/s] 19%|█▉        | 77260/400000 [00:10<00:43, 7416.24it/s] 20%|█▉        | 78003/400000 [00:10<00:43, 7353.18it/s] 20%|█▉        | 78750/400000 [00:10<00:43, 7385.16it/s] 20%|█▉        | 79516/400000 [00:10<00:42, 7463.28it/s] 20%|██        | 80269/400000 [00:10<00:42, 7481.05it/s] 20%|██        | 81018/400000 [00:10<00:43, 7398.21it/s] 20%|██        | 81783/400000 [00:11<00:42, 7471.78it/s] 21%|██        | 82531/400000 [00:11<00:42, 7411.73it/s] 21%|██        | 83273/400000 [00:11<00:42, 7401.97it/s] 21%|██        | 84041/400000 [00:11<00:42, 7482.11it/s] 21%|██        | 84790/400000 [00:11<00:42, 7370.03it/s] 21%|██▏       | 85561/400000 [00:11<00:42, 7466.82it/s] 22%|██▏       | 86316/400000 [00:11<00:41, 7490.64it/s] 22%|██▏       | 87077/400000 [00:11<00:41, 7525.39it/s] 22%|██▏       | 87834/400000 [00:11<00:41, 7536.41it/s] 22%|██▏       | 88588/400000 [00:11<00:41, 7429.13it/s] 22%|██▏       | 89339/400000 [00:12<00:41, 7451.81it/s] 23%|██▎       | 90090/400000 [00:12<00:41, 7468.28it/s] 23%|██▎       | 90855/400000 [00:12<00:41, 7521.11it/s] 23%|██▎       | 91619/400000 [00:12<00:40, 7554.17it/s] 23%|██▎       | 92375/400000 [00:12<00:40, 7544.46it/s] 23%|██▎       | 93130/400000 [00:12<00:40, 7521.47it/s] 23%|██▎       | 93883/400000 [00:12<00:41, 7369.91it/s] 24%|██▎       | 94621/400000 [00:12<00:42, 7192.12it/s] 24%|██▍       | 95342/400000 [00:12<00:42, 7101.06it/s] 24%|██▍       | 96063/400000 [00:13<00:42, 7129.36it/s] 24%|██▍       | 96836/400000 [00:13<00:41, 7298.78it/s] 24%|██▍       | 97603/400000 [00:13<00:40, 7406.31it/s] 25%|██▍       | 98379/400000 [00:13<00:40, 7506.68it/s] 25%|██▍       | 99147/400000 [00:13<00:39, 7557.23it/s] 25%|██▍       | 99904/400000 [00:13<00:39, 7546.61it/s] 25%|██▌       | 100664/400000 [00:13<00:39, 7561.51it/s] 25%|██▌       | 101429/400000 [00:13<00:39, 7586.04it/s] 26%|██▌       | 102188/400000 [00:13<00:40, 7353.66it/s] 26%|██▌       | 102956/400000 [00:13<00:39, 7446.63it/s] 26%|██▌       | 103703/400000 [00:14<00:39, 7419.52it/s] 26%|██▌       | 104447/400000 [00:14<00:39, 7422.08it/s] 26%|██▋       | 105214/400000 [00:14<00:39, 7493.20it/s] 26%|██▋       | 105979/400000 [00:14<00:39, 7538.54it/s] 27%|██▋       | 106741/400000 [00:14<00:38, 7562.33it/s] 27%|██▋       | 107498/400000 [00:14<00:38, 7550.99it/s] 27%|██▋       | 108273/400000 [00:14<00:38, 7608.50it/s] 27%|██▋       | 109035/400000 [00:14<00:38, 7610.05it/s] 27%|██▋       | 109802/400000 [00:14<00:38, 7626.19it/s] 28%|██▊       | 110569/400000 [00:14<00:37, 7637.42it/s] 28%|██▊       | 111333/400000 [00:15<00:39, 7364.06it/s] 28%|██▊       | 112082/400000 [00:15<00:38, 7391.68it/s] 28%|██▊       | 112823/400000 [00:15<00:39, 7266.77it/s] 28%|██▊       | 113552/400000 [00:15<00:40, 7156.63it/s] 29%|██▊       | 114270/400000 [00:15<00:40, 7036.79it/s] 29%|██▊       | 114976/400000 [00:15<00:41, 6929.89it/s] 29%|██▉       | 115698/400000 [00:15<00:40, 7012.89it/s] 29%|██▉       | 116464/400000 [00:15<00:39, 7194.46it/s] 29%|██▉       | 117211/400000 [00:15<00:38, 7272.54it/s] 29%|██▉       | 117956/400000 [00:15<00:38, 7324.22it/s] 30%|██▉       | 118690/400000 [00:16<00:38, 7281.81it/s] 30%|██▉       | 119432/400000 [00:16<00:38, 7322.11it/s] 30%|███       | 120207/400000 [00:16<00:37, 7444.27it/s] 30%|███       | 120985/400000 [00:16<00:36, 7541.36it/s] 30%|███       | 121741/400000 [00:16<00:36, 7528.02it/s] 31%|███       | 122503/400000 [00:16<00:36, 7553.10it/s] 31%|███       | 123277/400000 [00:16<00:36, 7606.03it/s] 31%|███       | 124039/400000 [00:16<00:36, 7596.87it/s] 31%|███       | 124800/400000 [00:16<00:37, 7422.50it/s] 31%|███▏      | 125544/400000 [00:16<00:38, 7184.69it/s] 32%|███▏      | 126289/400000 [00:17<00:37, 7261.15it/s] 32%|███▏      | 127055/400000 [00:17<00:37, 7374.96it/s] 32%|███▏      | 127821/400000 [00:17<00:36, 7457.42it/s] 32%|███▏      | 128591/400000 [00:17<00:36, 7526.62it/s] 32%|███▏      | 129355/400000 [00:17<00:35, 7557.63it/s] 33%|███▎      | 130126/400000 [00:17<00:35, 7600.11it/s] 33%|███▎      | 130897/400000 [00:17<00:35, 7623.22it/s] 33%|███▎      | 131660/400000 [00:17<00:36, 7389.58it/s] 33%|███▎      | 132401/400000 [00:17<00:37, 7226.38it/s] 33%|███▎      | 133126/400000 [00:18<00:37, 7068.88it/s] 33%|███▎      | 133851/400000 [00:18<00:37, 7120.95it/s] 34%|███▎      | 134615/400000 [00:18<00:36, 7268.04it/s] 34%|███▍      | 135374/400000 [00:18<00:35, 7359.58it/s] 34%|███▍      | 136134/400000 [00:18<00:35, 7429.19it/s] 34%|███▍      | 136883/400000 [00:18<00:35, 7446.78it/s] 34%|███▍      | 137651/400000 [00:18<00:34, 7513.92it/s] 35%|███▍      | 138416/400000 [00:18<00:34, 7554.19it/s] 35%|███▍      | 139178/400000 [00:18<00:34, 7571.26it/s] 35%|███▍      | 139946/400000 [00:18<00:34, 7600.75it/s] 35%|███▌      | 140707/400000 [00:19<00:36, 7113.60it/s] 35%|███▌      | 141425/400000 [00:19<00:36, 7036.58it/s] 36%|███▌      | 142186/400000 [00:19<00:35, 7198.58it/s] 36%|███▌      | 142959/400000 [00:19<00:34, 7348.64it/s] 36%|███▌      | 143728/400000 [00:19<00:34, 7446.91it/s] 36%|███▌      | 144485/400000 [00:19<00:34, 7483.09it/s] 36%|███▋      | 145241/400000 [00:19<00:33, 7505.41it/s] 37%|███▋      | 146018/400000 [00:19<00:33, 7582.77it/s] 37%|███▋      | 146786/400000 [00:19<00:33, 7611.64it/s] 37%|███▋      | 147566/400000 [00:19<00:32, 7666.66it/s] 37%|███▋      | 148334/400000 [00:20<00:33, 7599.77it/s] 37%|███▋      | 149095/400000 [00:20<00:33, 7571.00it/s] 37%|███▋      | 149874/400000 [00:20<00:32, 7632.67it/s] 38%|███▊      | 150649/400000 [00:20<00:32, 7665.35it/s] 38%|███▊      | 151429/400000 [00:20<00:32, 7704.71it/s] 38%|███▊      | 152200/400000 [00:20<00:32, 7599.43it/s] 38%|███▊      | 152961/400000 [00:20<00:33, 7339.60it/s] 38%|███▊      | 153698/400000 [00:20<00:34, 7198.25it/s] 39%|███▊      | 154421/400000 [00:20<00:34, 7096.37it/s] 39%|███▉      | 155133/400000 [00:20<00:34, 7033.08it/s] 39%|███▉      | 155882/400000 [00:21<00:34, 7163.00it/s] 39%|███▉      | 156635/400000 [00:21<00:33, 7268.73it/s] 39%|███▉      | 157389/400000 [00:21<00:33, 7346.14it/s] 40%|███▉      | 158157/400000 [00:21<00:32, 7442.24it/s] 40%|███▉      | 158903/400000 [00:21<00:32, 7366.69it/s] 40%|███▉      | 159662/400000 [00:21<00:32, 7432.18it/s] 40%|████      | 160422/400000 [00:21<00:32, 7481.54it/s] 40%|████      | 161176/400000 [00:21<00:31, 7497.19it/s] 40%|████      | 161945/400000 [00:21<00:31, 7552.05it/s] 41%|████      | 162702/400000 [00:21<00:31, 7555.45it/s] 41%|████      | 163458/400000 [00:22<00:31, 7547.63it/s] 41%|████      | 164220/400000 [00:22<00:31, 7568.98it/s] 41%|████      | 164984/400000 [00:22<00:30, 7587.47it/s] 41%|████▏     | 165762/400000 [00:22<00:30, 7642.28it/s] 42%|████▏     | 166533/400000 [00:22<00:30, 7661.19it/s] 42%|████▏     | 167300/400000 [00:22<00:30, 7606.52it/s] 42%|████▏     | 168067/400000 [00:22<00:30, 7622.70it/s] 42%|████▏     | 168842/400000 [00:22<00:30, 7658.90it/s] 42%|████▏     | 169612/400000 [00:22<00:30, 7670.21it/s] 43%|████▎     | 170385/400000 [00:22<00:29, 7686.48it/s] 43%|████▎     | 171154/400000 [00:23<00:30, 7544.12it/s] 43%|████▎     | 171919/400000 [00:23<00:30, 7573.29it/s] 43%|████▎     | 172677/400000 [00:23<00:31, 7307.38it/s] 43%|████▎     | 173411/400000 [00:23<00:31, 7166.71it/s] 44%|████▎     | 174173/400000 [00:23<00:30, 7295.95it/s] 44%|████▎     | 174926/400000 [00:23<00:30, 7363.71it/s] 44%|████▍     | 175665/400000 [00:23<00:30, 7340.19it/s] 44%|████▍     | 176426/400000 [00:23<00:30, 7417.00it/s] 44%|████▍     | 177187/400000 [00:23<00:29, 7471.50it/s] 44%|████▍     | 177947/400000 [00:24<00:29, 7509.10it/s] 45%|████▍     | 178699/400000 [00:24<00:30, 7183.34it/s] 45%|████▍     | 179421/400000 [00:24<00:31, 6987.12it/s] 45%|████▌     | 180124/400000 [00:24<00:31, 6918.35it/s] 45%|████▌     | 180885/400000 [00:24<00:30, 7111.22it/s] 45%|████▌     | 181634/400000 [00:24<00:30, 7145.60it/s] 46%|████▌     | 182354/400000 [00:24<00:30, 7160.40it/s] 46%|████▌     | 183114/400000 [00:24<00:29, 7285.60it/s] 46%|████▌     | 183882/400000 [00:24<00:29, 7399.22it/s] 46%|████▌     | 184645/400000 [00:24<00:28, 7464.19it/s] 46%|████▋     | 185396/400000 [00:25<00:28, 7475.36it/s] 47%|████▋     | 186145/400000 [00:25<00:28, 7378.18it/s] 47%|████▋     | 186889/400000 [00:25<00:28, 7393.66it/s] 47%|████▋     | 187630/400000 [00:25<00:29, 7257.38it/s] 47%|████▋     | 188357/400000 [00:25<00:29, 7138.77it/s] 47%|████▋     | 189073/400000 [00:25<00:30, 7023.10it/s] 47%|████▋     | 189816/400000 [00:25<00:29, 7140.31it/s] 48%|████▊     | 190587/400000 [00:25<00:28, 7299.71it/s] 48%|████▊     | 191325/400000 [00:25<00:28, 7321.10it/s] 48%|████▊     | 192086/400000 [00:25<00:28, 7402.94it/s] 48%|████▊     | 192850/400000 [00:26<00:27, 7472.14it/s] 48%|████▊     | 193599/400000 [00:26<00:28, 7153.96it/s] 49%|████▊     | 194318/400000 [00:26<00:29, 7054.85it/s] 49%|████▉     | 195027/400000 [00:26<00:29, 6962.69it/s] 49%|████▉     | 195726/400000 [00:26<00:29, 6938.08it/s] 49%|████▉     | 196450/400000 [00:26<00:28, 7023.42it/s] 49%|████▉     | 197209/400000 [00:26<00:28, 7183.29it/s] 49%|████▉     | 197978/400000 [00:26<00:27, 7326.38it/s] 50%|████▉     | 198729/400000 [00:26<00:27, 7380.41it/s] 50%|████▉     | 199493/400000 [00:26<00:26, 7455.23it/s] 50%|█████     | 200240/400000 [00:27<00:26, 7445.61it/s] 50%|█████     | 200996/400000 [00:27<00:26, 7476.60it/s] 50%|█████     | 201756/400000 [00:27<00:26, 7510.56it/s] 51%|█████     | 202522/400000 [00:27<00:26, 7553.57it/s] 51%|█████     | 203278/400000 [00:27<00:26, 7540.95it/s] 51%|█████     | 204033/400000 [00:27<00:26, 7515.17it/s] 51%|█████     | 204802/400000 [00:27<00:25, 7565.15it/s] 51%|█████▏    | 205559/400000 [00:27<00:25, 7560.81it/s] 52%|█████▏    | 206323/400000 [00:27<00:25, 7580.30it/s] 52%|█████▏    | 207087/400000 [00:27<00:25, 7597.47it/s] 52%|█████▏    | 207847/400000 [00:28<00:25, 7516.87it/s] 52%|█████▏    | 208599/400000 [00:28<00:26, 7276.92it/s] 52%|█████▏    | 209329/400000 [00:28<00:26, 7141.96it/s] 53%|█████▎    | 210046/400000 [00:28<00:26, 7050.35it/s] 53%|█████▎    | 210753/400000 [00:28<00:26, 7015.22it/s] 53%|█████▎    | 211456/400000 [00:28<00:27, 6955.16it/s] 53%|█████▎    | 212215/400000 [00:28<00:26, 7132.92it/s] 53%|█████▎    | 212983/400000 [00:28<00:25, 7288.58it/s] 53%|█████▎    | 213747/400000 [00:28<00:25, 7383.17it/s] 54%|█████▎    | 214517/400000 [00:29<00:24, 7473.96it/s] 54%|█████▍    | 215266/400000 [00:29<00:25, 7330.36it/s] 54%|█████▍    | 216001/400000 [00:29<00:25, 7147.91it/s] 54%|█████▍    | 216718/400000 [00:29<00:26, 6983.19it/s] 54%|█████▍    | 217419/400000 [00:29<00:26, 6971.87it/s] 55%|█████▍    | 218138/400000 [00:29<00:25, 7035.37it/s] 55%|█████▍    | 218892/400000 [00:29<00:25, 7179.46it/s] 55%|█████▍    | 219669/400000 [00:29<00:24, 7345.61it/s] 55%|█████▌    | 220446/400000 [00:29<00:24, 7465.45it/s] 55%|█████▌    | 221224/400000 [00:29<00:23, 7555.86it/s] 55%|█████▌    | 221997/400000 [00:30<00:23, 7606.55it/s] 56%|█████▌    | 222759/400000 [00:30<00:24, 7347.12it/s] 56%|█████▌    | 223497/400000 [00:30<00:24, 7171.35it/s] 56%|█████▌    | 224217/400000 [00:30<00:24, 7089.34it/s] 56%|█████▌    | 224929/400000 [00:30<00:24, 7085.28it/s] 56%|█████▋    | 225700/400000 [00:30<00:24, 7259.69it/s] 57%|█████▋    | 226432/400000 [00:30<00:23, 7275.79it/s] 57%|█████▋    | 227192/400000 [00:30<00:23, 7367.96it/s] 57%|█████▋    | 227970/400000 [00:30<00:22, 7484.45it/s] 57%|█████▋    | 228737/400000 [00:30<00:22, 7538.63it/s] 57%|█████▋    | 229507/400000 [00:31<00:22, 7584.89it/s] 58%|█████▊    | 230267/400000 [00:31<00:23, 7194.44it/s] 58%|█████▊    | 230992/400000 [00:31<00:23, 7076.51it/s] 58%|█████▊    | 231733/400000 [00:31<00:23, 7172.19it/s] 58%|█████▊    | 232484/400000 [00:31<00:23, 7268.30it/s] 58%|█████▊    | 233250/400000 [00:31<00:22, 7379.65it/s] 59%|█████▊    | 234003/400000 [00:31<00:22, 7423.49it/s] 59%|█████▊    | 234747/400000 [00:31<00:22, 7360.02it/s] 59%|█████▉    | 235520/400000 [00:31<00:22, 7465.74it/s] 59%|█████▉    | 236279/400000 [00:31<00:21, 7500.69it/s] 59%|█████▉    | 237030/400000 [00:32<00:21, 7409.37it/s] 59%|█████▉    | 237772/400000 [00:32<00:22, 7351.52it/s] 60%|█████▉    | 238532/400000 [00:32<00:21, 7422.94it/s] 60%|█████▉    | 239291/400000 [00:32<00:21, 7470.60it/s] 60%|██████    | 240045/400000 [00:32<00:21, 7488.78it/s] 60%|██████    | 240810/400000 [00:32<00:21, 7535.63it/s] 60%|██████    | 241564/400000 [00:32<00:21, 7521.99it/s] 61%|██████    | 242327/400000 [00:32<00:20, 7551.47it/s] 61%|██████    | 243084/400000 [00:32<00:20, 7555.93it/s] 61%|██████    | 243840/400000 [00:33<00:20, 7547.31it/s] 61%|██████    | 244595/400000 [00:33<00:20, 7543.49it/s] 61%|██████▏   | 245350/400000 [00:33<00:21, 7326.47it/s] 62%|██████▏   | 246119/400000 [00:33<00:20, 7430.42it/s] 62%|██████▏   | 246883/400000 [00:33<00:20, 7492.03it/s] 62%|██████▏   | 247639/400000 [00:33<00:20, 7509.63it/s] 62%|██████▏   | 248411/400000 [00:33<00:20, 7569.51it/s] 62%|██████▏   | 249169/400000 [00:33<00:20, 7371.88it/s] 62%|██████▏   | 249908/400000 [00:33<00:21, 7122.40it/s] 63%|██████▎   | 250624/400000 [00:33<00:21, 7027.25it/s] 63%|██████▎   | 251330/400000 [00:34<00:21, 6984.66it/s] 63%|██████▎   | 252032/400000 [00:34<00:21, 6994.87it/s] 63%|██████▎   | 252742/400000 [00:34<00:20, 7023.42it/s] 63%|██████▎   | 253456/400000 [00:34<00:20, 7056.32it/s] 64%|██████▎   | 254231/400000 [00:34<00:20, 7249.44it/s] 64%|██████▎   | 254967/400000 [00:34<00:19, 7280.08it/s] 64%|██████▍   | 255697/400000 [00:34<00:19, 7266.76it/s] 64%|██████▍   | 256444/400000 [00:34<00:19, 7324.69it/s] 64%|██████▍   | 257212/400000 [00:34<00:19, 7427.30it/s] 64%|██████▍   | 257981/400000 [00:34<00:18, 7501.86it/s] 65%|██████▍   | 258745/400000 [00:35<00:18, 7540.17it/s] 65%|██████▍   | 259500/400000 [00:35<00:18, 7539.60it/s] 65%|██████▌   | 260255/400000 [00:35<00:18, 7522.79it/s] 65%|██████▌   | 261023/400000 [00:35<00:18, 7568.65it/s] 65%|██████▌   | 261781/400000 [00:35<00:18, 7552.96it/s] 66%|██████▌   | 262554/400000 [00:35<00:18, 7604.62it/s] 66%|██████▌   | 263324/400000 [00:35<00:17, 7630.91it/s] 66%|██████▌   | 264088/400000 [00:35<00:17, 7614.62it/s] 66%|██████▌   | 264859/400000 [00:35<00:17, 7640.16it/s] 66%|██████▋   | 265631/400000 [00:35<00:17, 7662.30it/s] 67%|██████▋   | 266398/400000 [00:36<00:17, 7626.52it/s] 67%|██████▋   | 267168/400000 [00:36<00:17, 7647.23it/s] 67%|██████▋   | 267933/400000 [00:36<00:17, 7584.99it/s] 67%|██████▋   | 268708/400000 [00:36<00:17, 7632.35it/s] 67%|██████▋   | 269472/400000 [00:36<00:17, 7608.33it/s] 68%|██████▊   | 270235/400000 [00:36<00:17, 7612.12it/s] 68%|██████▊   | 271002/400000 [00:36<00:16, 7629.08it/s] 68%|██████▊   | 271765/400000 [00:36<00:16, 7570.75it/s] 68%|██████▊   | 272536/400000 [00:36<00:16, 7611.35it/s] 68%|██████▊   | 273300/400000 [00:36<00:16, 7619.53it/s] 69%|██████▊   | 274063/400000 [00:37<00:16, 7614.24it/s] 69%|██████▊   | 274825/400000 [00:37<00:16, 7581.84it/s] 69%|██████▉   | 275584/400000 [00:37<00:16, 7505.13it/s] 69%|██████▉   | 276335/400000 [00:37<00:16, 7490.23it/s] 69%|██████▉   | 277108/400000 [00:37<00:16, 7559.24it/s] 69%|██████▉   | 277866/400000 [00:37<00:16, 7565.39it/s] 70%|██████▉   | 278639/400000 [00:37<00:15, 7613.28it/s] 70%|██████▉   | 279401/400000 [00:37<00:15, 7580.73it/s] 70%|███████   | 280182/400000 [00:37<00:15, 7646.62it/s] 70%|███████   | 280952/400000 [00:37<00:15, 7661.07it/s] 70%|███████   | 281732/400000 [00:38<00:15, 7701.08it/s] 71%|███████   | 282503/400000 [00:38<00:15, 7699.12it/s] 71%|███████   | 283274/400000 [00:38<00:15, 7650.03it/s] 71%|███████   | 284052/400000 [00:38<00:15, 7686.16it/s] 71%|███████   | 284821/400000 [00:38<00:14, 7683.12it/s] 71%|███████▏  | 285599/400000 [00:38<00:14, 7710.47it/s] 72%|███████▏  | 286371/400000 [00:38<00:15, 7546.47it/s] 72%|███████▏  | 287128/400000 [00:38<00:14, 7551.08it/s] 72%|███████▏  | 287903/400000 [00:38<00:14, 7608.60it/s] 72%|███████▏  | 288667/400000 [00:38<00:14, 7615.30it/s] 72%|███████▏  | 289439/400000 [00:39<00:14, 7643.73it/s] 73%|███████▎  | 290214/400000 [00:39<00:14, 7673.29it/s] 73%|███████▎  | 290982/400000 [00:39<00:14, 7652.68it/s] 73%|███████▎  | 291756/400000 [00:39<00:14, 7676.14it/s] 73%|███████▎  | 292530/400000 [00:39<00:13, 7694.90it/s] 73%|███████▎  | 293302/400000 [00:39<00:13, 7700.60it/s] 74%|███████▎  | 294073/400000 [00:39<00:13, 7702.33it/s] 74%|███████▎  | 294844/400000 [00:39<00:13, 7627.14it/s] 74%|███████▍  | 295607/400000 [00:39<00:13, 7604.19it/s] 74%|███████▍  | 296368/400000 [00:39<00:13, 7461.03it/s] 74%|███████▍  | 297115/400000 [00:40<00:14, 7232.13it/s] 74%|███████▍  | 297841/400000 [00:40<00:14, 7076.23it/s] 75%|███████▍  | 298551/400000 [00:40<00:14, 7069.04it/s] 75%|███████▍  | 299323/400000 [00:40<00:13, 7251.23it/s] 75%|███████▌  | 300093/400000 [00:40<00:13, 7378.28it/s] 75%|███████▌  | 300862/400000 [00:40<00:13, 7466.66it/s] 75%|███████▌  | 301631/400000 [00:40<00:13, 7531.63it/s] 76%|███████▌  | 302386/400000 [00:40<00:12, 7534.84it/s] 76%|███████▌  | 303156/400000 [00:40<00:12, 7582.45it/s] 76%|███████▌  | 303927/400000 [00:41<00:12, 7618.82it/s] 76%|███████▌  | 304690/400000 [00:41<00:12, 7585.66it/s] 76%|███████▋  | 305449/400000 [00:41<00:12, 7345.54it/s] 77%|███████▋  | 306186/400000 [00:41<00:13, 7145.39it/s] 77%|███████▋  | 306920/400000 [00:41<00:12, 7199.83it/s] 77%|███████▋  | 307642/400000 [00:41<00:13, 7082.27it/s] 77%|███████▋  | 308375/400000 [00:41<00:12, 7152.61it/s] 77%|███████▋  | 309140/400000 [00:41<00:12, 7292.55it/s] 77%|███████▋  | 309894/400000 [00:41<00:12, 7362.40it/s] 78%|███████▊  | 310648/400000 [00:41<00:12, 7413.78it/s] 78%|███████▊  | 311413/400000 [00:42<00:11, 7481.14it/s] 78%|███████▊  | 312178/400000 [00:42<00:11, 7530.84it/s] 78%|███████▊  | 312943/400000 [00:42<00:11, 7565.65it/s] 78%|███████▊  | 313701/400000 [00:42<00:11, 7567.97it/s] 79%|███████▊  | 314459/400000 [00:42<00:11, 7561.47it/s] 79%|███████▉  | 315217/400000 [00:42<00:11, 7564.93it/s] 79%|███████▉  | 315981/400000 [00:42<00:11, 7584.49it/s] 79%|███████▉  | 316740/400000 [00:42<00:11, 7549.86it/s] 79%|███████▉  | 317496/400000 [00:42<00:10, 7549.17it/s] 80%|███████▉  | 318252/400000 [00:42<00:10, 7536.98it/s] 80%|███████▉  | 319017/400000 [00:43<00:10, 7570.42it/s] 80%|███████▉  | 319775/400000 [00:43<00:10, 7412.12it/s] 80%|████████  | 320518/400000 [00:43<00:10, 7257.45it/s] 80%|████████  | 321264/400000 [00:43<00:10, 7316.46it/s] 81%|████████  | 322036/400000 [00:43<00:10, 7431.50it/s] 81%|████████  | 322805/400000 [00:43<00:10, 7506.42it/s] 81%|████████  | 323557/400000 [00:43<00:10, 7433.68it/s] 81%|████████  | 324318/400000 [00:43<00:10, 7481.55it/s] 81%|████████▏ | 325075/400000 [00:43<00:09, 7506.68it/s] 81%|████████▏ | 325827/400000 [00:43<00:09, 7443.16it/s] 82%|████████▏ | 326572/400000 [00:44<00:10, 7235.78it/s] 82%|████████▏ | 327298/400000 [00:44<00:10, 7086.83it/s] 82%|████████▏ | 328050/400000 [00:44<00:09, 7209.80it/s] 82%|████████▏ | 328774/400000 [00:44<00:09, 7216.10it/s] 82%|████████▏ | 329541/400000 [00:44<00:09, 7344.51it/s] 83%|████████▎ | 330296/400000 [00:44<00:09, 7402.53it/s] 83%|████████▎ | 331064/400000 [00:44<00:09, 7482.01it/s] 83%|████████▎ | 331821/400000 [00:44<00:09, 7507.84it/s] 83%|████████▎ | 332573/400000 [00:44<00:09, 7221.29it/s] 83%|████████▎ | 333336/400000 [00:44<00:09, 7336.81it/s] 84%|████████▎ | 334096/400000 [00:45<00:08, 7401.18it/s] 84%|████████▎ | 334861/400000 [00:45<00:08, 7473.96it/s] 84%|████████▍ | 335623/400000 [00:45<00:08, 7515.82it/s] 84%|████████▍ | 336376/400000 [00:45<00:08, 7409.83it/s] 84%|████████▍ | 337119/400000 [00:45<00:08, 7207.92it/s] 84%|████████▍ | 337842/400000 [00:45<00:08, 7030.98it/s] 85%|████████▍ | 338602/400000 [00:45<00:08, 7191.55it/s] 85%|████████▍ | 339372/400000 [00:45<00:08, 7336.28it/s] 85%|████████▌ | 340116/400000 [00:45<00:08, 7364.24it/s] 85%|████████▌ | 340855/400000 [00:46<00:08, 7268.25it/s] 85%|████████▌ | 341584/400000 [00:46<00:08, 7176.28it/s] 86%|████████▌ | 342336/400000 [00:46<00:07, 7275.12it/s] 86%|████████▌ | 343101/400000 [00:46<00:07, 7381.42it/s] 86%|████████▌ | 343852/400000 [00:46<00:07, 7418.48it/s] 86%|████████▌ | 344609/400000 [00:46<00:07, 7462.81it/s] 86%|████████▋ | 345375/400000 [00:46<00:07, 7518.48it/s] 87%|████████▋ | 346141/400000 [00:46<00:07, 7559.75it/s] 87%|████████▋ | 346902/400000 [00:46<00:07, 7572.26it/s] 87%|████████▋ | 347660/400000 [00:46<00:07, 7449.67it/s] 87%|████████▋ | 348428/400000 [00:47<00:06, 7514.96it/s] 87%|████████▋ | 349196/400000 [00:47<00:06, 7561.01it/s] 87%|████████▋ | 349974/400000 [00:47<00:06, 7625.36it/s] 88%|████████▊ | 350745/400000 [00:47<00:06, 7647.81it/s] 88%|████████▊ | 351511/400000 [00:47<00:06, 7615.37it/s] 88%|████████▊ | 352287/400000 [00:47<00:06, 7655.58it/s] 88%|████████▊ | 353061/400000 [00:47<00:06, 7678.97it/s] 88%|████████▊ | 353830/400000 [00:47<00:06, 7665.25it/s] 89%|████████▊ | 354601/400000 [00:47<00:05, 7676.42it/s] 89%|████████▉ | 355369/400000 [00:47<00:05, 7631.13it/s] 89%|████████▉ | 356133/400000 [00:48<00:05, 7633.64it/s] 89%|████████▉ | 356897/400000 [00:48<00:05, 7613.63it/s] 89%|████████▉ | 357659/400000 [00:48<00:05, 7604.99it/s] 90%|████████▉ | 358425/400000 [00:48<00:05, 7621.11it/s] 90%|████████▉ | 359188/400000 [00:48<00:05, 7529.52it/s] 90%|████████▉ | 359953/400000 [00:48<00:05, 7564.63it/s] 90%|█████████ | 360720/400000 [00:48<00:05, 7594.92it/s] 90%|█████████ | 361482/400000 [00:48<00:05, 7599.51it/s] 91%|█████████ | 362245/400000 [00:48<00:04, 7606.12it/s] 91%|█████████ | 363006/400000 [00:48<00:04, 7549.47it/s] 91%|█████████ | 363764/400000 [00:49<00:04, 7558.27it/s] 91%|█████████ | 364520/400000 [00:49<00:04, 7496.22it/s] 91%|█████████▏| 365272/400000 [00:49<00:04, 7501.98it/s] 92%|█████████▏| 366041/400000 [00:49<00:04, 7556.81it/s] 92%|█████████▏| 366797/400000 [00:49<00:04, 7538.01it/s] 92%|█████████▏| 367559/400000 [00:49<00:04, 7560.18it/s] 92%|█████████▏| 368317/400000 [00:49<00:04, 7563.49it/s] 92%|█████████▏| 369080/400000 [00:49<00:04, 7582.07it/s] 92%|█████████▏| 369846/400000 [00:49<00:03, 7604.29it/s] 93%|█████████▎| 370607/400000 [00:49<00:03, 7590.42it/s] 93%|█████████▎| 371367/400000 [00:50<00:03, 7417.63it/s] 93%|█████████▎| 372110/400000 [00:50<00:03, 7223.01it/s] 93%|█████████▎| 372835/400000 [00:50<00:03, 7097.46it/s] 93%|█████████▎| 373547/400000 [00:50<00:03, 7046.96it/s] 94%|█████████▎| 374267/400000 [00:50<00:03, 7091.24it/s] 94%|█████████▍| 375029/400000 [00:50<00:03, 7241.54it/s] 94%|█████████▍| 375795/400000 [00:50<00:03, 7361.51it/s] 94%|█████████▍| 376564/400000 [00:50<00:03, 7454.85it/s] 94%|█████████▍| 377331/400000 [00:50<00:03, 7515.44it/s] 95%|█████████▍| 378084/400000 [00:50<00:02, 7518.38it/s] 95%|█████████▍| 378844/400000 [00:51<00:02, 7542.46it/s] 95%|█████████▍| 379599/400000 [00:51<00:02, 7203.33it/s] 95%|█████████▌| 380365/400000 [00:51<00:02, 7332.84it/s] 95%|█████████▌| 381125/400000 [00:51<00:02, 7408.95it/s] 95%|█████████▌| 381873/400000 [00:51<00:02, 7428.37it/s] 96%|█████████▌| 382618/400000 [00:51<00:02, 7161.51it/s] 96%|█████████▌| 383338/400000 [00:51<00:02, 7077.20it/s] 96%|█████████▌| 384049/400000 [00:51<00:02, 7011.81it/s] 96%|█████████▌| 384753/400000 [00:51<00:02, 6933.88it/s] 96%|█████████▋| 385460/400000 [00:52<00:02, 6973.46it/s] 97%|█████████▋| 386193/400000 [00:52<00:01, 7074.50it/s] 97%|█████████▋| 386902/400000 [00:52<00:01, 6990.59it/s] 97%|█████████▋| 387603/400000 [00:52<00:01, 6970.94it/s] 97%|█████████▋| 388301/400000 [00:52<00:01, 6906.47it/s] 97%|█████████▋| 389068/400000 [00:52<00:01, 7116.72it/s] 97%|█████████▋| 389834/400000 [00:52<00:01, 7270.38it/s] 98%|█████████▊| 390607/400000 [00:52<00:01, 7401.97it/s] 98%|█████████▊| 391374/400000 [00:52<00:01, 7479.77it/s] 98%|█████████▊| 392137/400000 [00:52<00:01, 7521.96it/s] 98%|█████████▊| 392902/400000 [00:53<00:00, 7559.55it/s] 98%|█████████▊| 393678/400000 [00:53<00:00, 7617.40it/s] 99%|█████████▊| 394458/400000 [00:53<00:00, 7670.57it/s] 99%|█████████▉| 395229/400000 [00:53<00:00, 7681.55it/s] 99%|█████████▉| 396003/400000 [00:53<00:00, 7697.59it/s] 99%|█████████▉| 396774/400000 [00:53<00:00, 7462.96it/s] 99%|█████████▉| 397523/400000 [00:53<00:00, 7253.71it/s]100%|█████████▉| 398258/400000 [00:53<00:00, 7280.87it/s]100%|█████████▉| 399031/400000 [00:53<00:00, 7409.81it/s]100%|█████████▉| 399805/400000 [00:53<00:00, 7504.68it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7410.07it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f243ce30c50>, <torchtext.data.dataset.TabularDataset object at 0x7f243ce30da0>, <torchtext.vocab.Vocab object at 0x7f243ce30cc0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.95 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 17.39 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 17.39 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.01 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.01 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.34 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.34 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.57 file/s]2020-07-24 17:24:05.745542: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-24 17:24:05.749546: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-07-24 17:24:05.750014: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561392b974b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-24 17:24:05.750341: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'test', 'fashion-mnist_small.npy', 'mnist2', 'mnist_dataset_small.npy', 'train'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:04, 153202.24it/s] 72%|███████▏  | 7127040/9912422 [00:00<00:12, 218656.96it/s]9920512it [00:00, 43438173.06it/s]                           
0it [00:00, ?it/s]32768it [00:00, 484988.51it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 474753.61it/s]1654784it [00:00, 11714087.54it/s]                         
0it [00:00, ?it/s]8192it [00:00, 210255.47it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

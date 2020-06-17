
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/70f1a18a5306e0e1f742c0397a3d2aeed0ee5b84', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '70f1a18a5306e0e1f742c0397a3d2aeed0ee5b84', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/70f1a18a5306e0e1f742c0397a3d2aeed0ee5b84

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/70f1a18a5306e0e1f742c0397a3d2aeed0ee5b84

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/70f1a18a5306e0e1f742c0397a3d2aeed0ee5b84

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fccd7f5ff28> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fccd7f5ff28> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fcd435170d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fcd435170d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fcd5d243ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fcd5d243ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fccf0d92950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fccf0d92950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fccf0d92950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 27%|██▋       | 2703360/9912422 [00:00<00:00, 27029888.01it/s] 87%|████████▋ | 8593408/9912422 [00:00<00:00, 32223627.44it/s]9920512it [00:00, 28655318.64it/s]                             
0it [00:00, ?it/s]32768it [00:00, 633079.16it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 480597.23it/s]1654784it [00:00, 11403529.74it/s]                         
0it [00:00, ?it/s]8192it [00:00, 201099.96it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fccee3946a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fccee38a940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fccf0d92598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fccf0d92598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fccf0d92598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:14,  2.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:14,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:14,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:13,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:13,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:12,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:12,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:12,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:50,  3.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:50,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:50,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:50,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:49,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:49,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:49,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:48,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:48,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:48,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:33,  4.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:33,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:33,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:33,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:33,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:33,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:32,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:32,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:32,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:32,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:22,  5.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:22,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:22,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:21,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:21,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:21,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:21,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:14,  8.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:14,  8.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:14,  8.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:14,  8.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:14,  8.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:00<00:10, 11.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:10, 11.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 15.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 26.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 26.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 26.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 26.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 33.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 41.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 41.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 41.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 41.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 41.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 41.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 41.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 48.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 48.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 48.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 48.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 48.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 48.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 48.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 48.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 48.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 48.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 56.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 56.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 56.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 56.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 56.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 56.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 56.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 56.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 56.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 62.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 68.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 68.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 73.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 73.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 73.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 73.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 73.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 73.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 73.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 73.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 77.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 77.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 77.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 77.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 77.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 77.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 77.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 77.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 77.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 77.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 79.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 79.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 79.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 79.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 79.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 79.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 79.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 79.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 79.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 79.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 82.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 82.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.35s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.35s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.35s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.04 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.47s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.35s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 82.04 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.47s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.47s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.26 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.47s/ url]
0 examples [00:00, ? examples/s]2020-06-17 06:08:54.714753: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-17 06:08:54.768259: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-06-17 06:08:54.768651: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55dafad17e70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-17 06:08:54.768673: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:01,  1.05s/ examples]102 examples [00:01,  1.36 examples/s]198 examples [00:01,  1.95 examples/s]294 examples [00:01,  2.78 examples/s]391 examples [00:01,  3.96 examples/s]484 examples [00:01,  5.65 examples/s]581 examples [00:01,  8.06 examples/s]675 examples [00:01, 11.47 examples/s]764 examples [00:01, 16.29 examples/s]863 examples [00:01, 23.11 examples/s]963 examples [00:02, 32.68 examples/s]1063 examples [00:02, 46.04 examples/s]1163 examples [00:02, 64.49 examples/s]1261 examples [00:02, 89.60 examples/s]1361 examples [00:02, 123.25 examples/s]1459 examples [00:02, 167.03 examples/s]1560 examples [00:02, 222.67 examples/s]1661 examples [00:02, 290.52 examples/s]1761 examples [00:02, 366.71 examples/s]1863 examples [00:02, 453.42 examples/s]1963 examples [00:03, 541.76 examples/s]2062 examples [00:03, 620.09 examples/s]2160 examples [00:03, 692.79 examples/s]2257 examples [00:03, 754.82 examples/s]2357 examples [00:03, 814.44 examples/s]2455 examples [00:03, 853.91 examples/s]2553 examples [00:03, 875.06 examples/s]2655 examples [00:03, 912.82 examples/s]2754 examples [00:03, 933.31 examples/s]2855 examples [00:03, 954.27 examples/s]2956 examples [00:04, 969.61 examples/s]3059 examples [00:04, 985.32 examples/s]3161 examples [00:04, 994.30 examples/s]3262 examples [00:04, 987.25 examples/s]3362 examples [00:04, 918.44 examples/s]3458 examples [00:04, 930.16 examples/s]3555 examples [00:04, 940.54 examples/s]3650 examples [00:04, 918.84 examples/s]3743 examples [00:04, 915.23 examples/s]3846 examples [00:05, 945.05 examples/s]3942 examples [00:05, 947.07 examples/s]4042 examples [00:05, 961.77 examples/s]4139 examples [00:05, 957.09 examples/s]4237 examples [00:05, 958.43 examples/s]4334 examples [00:05, 946.10 examples/s]4429 examples [00:05, 944.38 examples/s]4530 examples [00:05, 961.82 examples/s]4629 examples [00:05, 969.87 examples/s]4729 examples [00:05, 977.95 examples/s]4833 examples [00:06, 994.40 examples/s]4933 examples [00:06, 993.33 examples/s]5035 examples [00:06, 1001.03 examples/s]5136 examples [00:06, 995.04 examples/s] 5236 examples [00:06, 977.20 examples/s]5334 examples [00:06, 975.75 examples/s]5434 examples [00:06, 980.75 examples/s]5534 examples [00:06, 985.76 examples/s]5633 examples [00:06, 974.40 examples/s]5731 examples [00:06, 974.12 examples/s]5834 examples [00:07, 988.91 examples/s]5937 examples [00:07, 999.00 examples/s]6037 examples [00:07, 993.11 examples/s]6137 examples [00:07, 986.78 examples/s]6236 examples [00:07, 986.56 examples/s]6335 examples [00:07, 987.58 examples/s]6438 examples [00:07, 997.48 examples/s]6541 examples [00:07, 1006.10 examples/s]6642 examples [00:07, 999.75 examples/s] 6743 examples [00:07, 985.35 examples/s]6842 examples [00:08, 967.35 examples/s]6942 examples [00:08, 976.16 examples/s]7041 examples [00:08, 979.10 examples/s]7139 examples [00:08, 976.59 examples/s]7241 examples [00:08, 988.19 examples/s]7340 examples [00:08, 961.40 examples/s]7440 examples [00:08, 970.14 examples/s]7538 examples [00:08, 969.02 examples/s]7636 examples [00:08, 936.84 examples/s]7735 examples [00:09, 949.53 examples/s]7831 examples [00:09, 947.65 examples/s]7926 examples [00:09, 913.75 examples/s]8018 examples [00:09, 914.64 examples/s]8110 examples [00:09, 886.47 examples/s]8200 examples [00:09, 863.49 examples/s]8287 examples [00:09, 841.79 examples/s]8372 examples [00:09, 827.36 examples/s]8459 examples [00:09, 838.71 examples/s]8544 examples [00:09, 820.52 examples/s]8637 examples [00:10, 841.17 examples/s]8735 examples [00:10, 877.58 examples/s]8833 examples [00:10, 905.29 examples/s]8930 examples [00:10, 921.75 examples/s]9028 examples [00:10, 936.14 examples/s]9128 examples [00:10, 952.86 examples/s]9226 examples [00:10, 958.49 examples/s]9327 examples [00:10, 970.47 examples/s]9425 examples [00:10, 963.14 examples/s]9522 examples [00:10, 951.32 examples/s]9618 examples [00:11, 948.12 examples/s]9719 examples [00:11, 964.98 examples/s]9816 examples [00:11, 966.45 examples/s]9913 examples [00:11, 951.01 examples/s]10009 examples [00:11, 914.42 examples/s]10104 examples [00:11, 924.04 examples/s]10205 examples [00:11, 947.58 examples/s]10306 examples [00:11, 963.03 examples/s]10403 examples [00:11, 948.97 examples/s]10500 examples [00:12, 954.92 examples/s]10605 examples [00:12, 980.92 examples/s]10704 examples [00:12, 976.75 examples/s]10802 examples [00:12, 960.77 examples/s]10902 examples [00:12, 969.46 examples/s]11000 examples [00:12, 971.82 examples/s]11101 examples [00:12, 980.55 examples/s]11203 examples [00:12, 991.52 examples/s]11304 examples [00:12, 996.14 examples/s]11404 examples [00:12, 992.16 examples/s]11504 examples [00:13, 929.81 examples/s]11603 examples [00:13, 946.48 examples/s]11704 examples [00:13, 963.89 examples/s]11804 examples [00:13, 973.38 examples/s]11906 examples [00:13, 983.99 examples/s]12005 examples [00:13, 985.34 examples/s]12107 examples [00:13, 993.63 examples/s]12208 examples [00:13, 997.67 examples/s]12308 examples [00:13, 996.54 examples/s]12410 examples [00:13, 999.15 examples/s]12510 examples [00:14, 997.20 examples/s]12610 examples [00:14, 993.52 examples/s]12714 examples [00:14, 1005.09 examples/s]12815 examples [00:14, 994.72 examples/s] 12915 examples [00:14, 993.83 examples/s]13015 examples [00:14, 983.92 examples/s]13114 examples [00:14, 975.05 examples/s]13212 examples [00:14, 966.53 examples/s]13309 examples [00:14, 951.92 examples/s]13408 examples [00:14, 960.12 examples/s]13511 examples [00:15, 978.63 examples/s]13614 examples [00:15, 992.32 examples/s]13717 examples [00:15, 1001.94 examples/s]13818 examples [00:15, 1003.86 examples/s]13919 examples [00:15, 1005.27 examples/s]14021 examples [00:15, 1007.97 examples/s]14124 examples [00:15, 1011.33 examples/s]14226 examples [00:15, 1009.82 examples/s]14328 examples [00:15, 1009.44 examples/s]14429 examples [00:15, 995.29 examples/s] 14529 examples [00:16, 983.70 examples/s]14628 examples [00:16, 978.02 examples/s]14726 examples [00:16, 944.62 examples/s]14822 examples [00:16, 949.10 examples/s]14920 examples [00:16, 957.32 examples/s]15021 examples [00:16, 972.02 examples/s]15122 examples [00:16, 981.45 examples/s]15223 examples [00:16, 987.18 examples/s]15324 examples [00:16, 991.33 examples/s]15424 examples [00:17, 982.45 examples/s]15526 examples [00:17, 991.37 examples/s]15626 examples [00:17, 981.46 examples/s]15725 examples [00:17, 975.94 examples/s]15825 examples [00:17, 982.06 examples/s]15924 examples [00:17, 975.27 examples/s]16022 examples [00:17, 959.05 examples/s]16123 examples [00:17, 973.26 examples/s]16224 examples [00:17, 981.63 examples/s]16323 examples [00:17, 977.95 examples/s]16421 examples [00:18, 929.29 examples/s]16520 examples [00:18, 946.64 examples/s]16616 examples [00:18, 950.13 examples/s]16716 examples [00:18, 964.13 examples/s]16815 examples [00:18, 969.83 examples/s]16913 examples [00:18, 968.33 examples/s]17014 examples [00:18, 980.00 examples/s]17114 examples [00:18, 984.99 examples/s]17213 examples [00:18, 974.01 examples/s]17311 examples [00:18, 962.73 examples/s]17409 examples [00:19, 966.92 examples/s]17506 examples [00:19, 945.50 examples/s]17608 examples [00:19, 965.42 examples/s]17706 examples [00:19, 968.34 examples/s]17807 examples [00:19, 979.06 examples/s]17906 examples [00:19, 966.55 examples/s]18004 examples [00:19, 970.08 examples/s]18102 examples [00:19, 966.06 examples/s]18200 examples [00:19, 968.24 examples/s]18297 examples [00:20, 920.38 examples/s]18392 examples [00:20, 928.56 examples/s]18487 examples [00:20, 934.67 examples/s]18585 examples [00:20, 945.47 examples/s]18683 examples [00:20, 954.07 examples/s]18779 examples [00:20, 952.38 examples/s]18876 examples [00:20, 955.79 examples/s]18975 examples [00:20, 964.42 examples/s]19076 examples [00:20, 977.44 examples/s]19179 examples [00:20, 990.40 examples/s]19281 examples [00:21, 997.27 examples/s]19381 examples [00:21, 992.46 examples/s]19481 examples [00:21, 967.98 examples/s]19579 examples [00:21, 969.46 examples/s]19677 examples [00:21, 963.59 examples/s]19774 examples [00:21, 956.92 examples/s]19870 examples [00:21, 947.03 examples/s]19969 examples [00:21, 958.96 examples/s]20065 examples [00:21, 928.28 examples/s]20159 examples [00:21, 919.22 examples/s]20257 examples [00:22, 934.81 examples/s]20356 examples [00:22, 948.71 examples/s]20452 examples [00:22, 950.37 examples/s]20548 examples [00:22, 942.54 examples/s]20649 examples [00:22, 959.43 examples/s]20746 examples [00:22, 942.42 examples/s]20842 examples [00:22, 946.15 examples/s]20941 examples [00:22, 958.61 examples/s]21037 examples [00:22, 950.53 examples/s]21133 examples [00:22, 947.73 examples/s]21228 examples [00:23, 946.69 examples/s]21327 examples [00:23, 956.70 examples/s]21424 examples [00:23, 959.40 examples/s]21524 examples [00:23, 970.57 examples/s]21624 examples [00:23, 978.60 examples/s]21722 examples [00:23, 973.98 examples/s]21820 examples [00:23, 975.40 examples/s]21921 examples [00:23, 984.27 examples/s]22020 examples [00:23, 982.36 examples/s]22120 examples [00:23, 986.77 examples/s]22220 examples [00:24, 989.53 examples/s]22319 examples [00:24, 987.72 examples/s]22420 examples [00:24, 992.82 examples/s]22520 examples [00:24, 989.24 examples/s]22619 examples [00:24, 986.12 examples/s]22718 examples [00:24, 953.64 examples/s]22814 examples [00:24, 945.07 examples/s]22911 examples [00:24, 949.64 examples/s]23012 examples [00:24, 966.84 examples/s]23113 examples [00:24, 977.61 examples/s]23214 examples [00:25, 983.16 examples/s]23313 examples [00:25, 953.13 examples/s]23409 examples [00:25, 912.96 examples/s]23503 examples [00:25, 920.27 examples/s]23597 examples [00:25, 925.96 examples/s]23692 examples [00:25, 930.05 examples/s]23787 examples [00:25, 935.76 examples/s]23881 examples [00:25, 921.69 examples/s]23976 examples [00:25, 928.91 examples/s]24071 examples [00:26, 934.35 examples/s]24165 examples [00:26, 934.34 examples/s]24259 examples [00:26, 916.57 examples/s]24351 examples [00:26, 891.94 examples/s]24447 examples [00:26, 909.03 examples/s]24541 examples [00:26, 917.35 examples/s]24633 examples [00:26, 917.55 examples/s]24727 examples [00:26, 921.56 examples/s]24820 examples [00:26, 919.11 examples/s]24912 examples [00:26, 908.05 examples/s]25005 examples [00:27, 912.88 examples/s]25099 examples [00:27, 918.16 examples/s]25194 examples [00:27, 925.50 examples/s]25287 examples [00:27, 924.55 examples/s]25384 examples [00:27, 937.00 examples/s]25483 examples [00:27, 952.00 examples/s]25581 examples [00:27, 958.40 examples/s]25677 examples [00:27, 944.15 examples/s]25772 examples [00:27, 926.70 examples/s]25865 examples [00:27, 919.37 examples/s]25964 examples [00:28, 938.95 examples/s]26062 examples [00:28, 949.61 examples/s]26162 examples [00:28, 962.65 examples/s]26262 examples [00:28, 971.30 examples/s]26360 examples [00:28, 971.82 examples/s]26458 examples [00:28, 964.18 examples/s]26556 examples [00:28, 966.90 examples/s]26656 examples [00:28, 975.93 examples/s]26757 examples [00:28, 983.34 examples/s]26858 examples [00:28, 990.76 examples/s]26959 examples [00:29, 994.00 examples/s]27059 examples [00:29, 986.43 examples/s]27160 examples [00:29, 993.11 examples/s]27260 examples [00:29, 962.09 examples/s]27357 examples [00:29, 927.81 examples/s]27454 examples [00:29, 939.99 examples/s]27555 examples [00:29, 957.43 examples/s]27656 examples [00:29, 971.07 examples/s]27755 examples [00:29, 974.25 examples/s]27853 examples [00:30, 970.30 examples/s]27951 examples [00:30, 953.21 examples/s]28047 examples [00:30, 953.77 examples/s]28143 examples [00:30, 920.37 examples/s]28236 examples [00:30, 917.37 examples/s]28331 examples [00:30, 926.51 examples/s]28433 examples [00:30, 951.95 examples/s]28529 examples [00:30, 948.26 examples/s]28628 examples [00:30, 958.85 examples/s]28727 examples [00:30, 966.27 examples/s]28826 examples [00:31, 970.42 examples/s]28924 examples [00:31, 938.48 examples/s]29021 examples [00:31, 945.27 examples/s]29117 examples [00:31, 946.94 examples/s]29216 examples [00:31, 959.29 examples/s]29316 examples [00:31, 970.63 examples/s]29417 examples [00:31, 979.67 examples/s]29516 examples [00:31, 978.62 examples/s]29618 examples [00:31, 989.87 examples/s]29718 examples [00:31, 952.90 examples/s]29814 examples [00:32, 947.65 examples/s]29910 examples [00:32, 949.17 examples/s]30006 examples [00:32, 903.31 examples/s]30102 examples [00:32, 919.34 examples/s]30195 examples [00:32, 908.77 examples/s]30290 examples [00:32, 918.95 examples/s]30386 examples [00:32, 929.11 examples/s]30480 examples [00:32, 918.89 examples/s]30577 examples [00:32, 933.36 examples/s]30678 examples [00:33, 954.77 examples/s]30779 examples [00:33, 970.24 examples/s]30880 examples [00:33, 981.54 examples/s]30979 examples [00:33, 977.14 examples/s]31081 examples [00:33, 987.21 examples/s]31182 examples [00:33, 993.11 examples/s]31282 examples [00:33, 938.66 examples/s]31381 examples [00:33, 953.10 examples/s]31481 examples [00:33, 965.14 examples/s]31582 examples [00:33, 975.74 examples/s]31680 examples [00:34, 973.88 examples/s]31781 examples [00:34, 983.18 examples/s]31880 examples [00:34, 981.72 examples/s]31979 examples [00:34, 959.61 examples/s]32076 examples [00:34, 951.60 examples/s]32174 examples [00:34, 959.46 examples/s]32271 examples [00:34, 953.10 examples/s]32370 examples [00:34, 961.65 examples/s]32470 examples [00:34, 971.80 examples/s]32573 examples [00:34, 986.27 examples/s]32672 examples [00:35, 986.73 examples/s]32774 examples [00:35, 994.48 examples/s]32875 examples [00:35, 996.61 examples/s]32975 examples [00:35, 978.12 examples/s]33073 examples [00:35, 915.90 examples/s]33166 examples [00:35, 914.60 examples/s]33267 examples [00:35, 940.41 examples/s]33367 examples [00:35, 956.27 examples/s]33471 examples [00:35, 978.02 examples/s]33572 examples [00:35, 987.33 examples/s]33672 examples [00:36, 958.29 examples/s]33770 examples [00:36, 963.39 examples/s]33867 examples [00:36, 956.74 examples/s]33969 examples [00:36, 971.59 examples/s]34068 examples [00:36, 976.09 examples/s]34168 examples [00:36, 980.94 examples/s]34267 examples [00:36, 977.92 examples/s]34366 examples [00:36, 979.47 examples/s]34467 examples [00:36, 986.57 examples/s]34568 examples [00:37, 993.46 examples/s]34670 examples [00:37, 999.02 examples/s]34770 examples [00:37, 999.17 examples/s]34870 examples [00:37, 984.73 examples/s]34971 examples [00:37, 989.88 examples/s]35071 examples [00:37, 975.53 examples/s]35169 examples [00:37, 964.78 examples/s]35266 examples [00:37, 949.70 examples/s]35362 examples [00:37, 944.49 examples/s]35459 examples [00:37, 949.88 examples/s]35558 examples [00:38, 959.42 examples/s]35655 examples [00:38, 960.22 examples/s]35758 examples [00:38, 978.03 examples/s]35857 examples [00:38, 979.03 examples/s]35957 examples [00:38, 982.47 examples/s]36056 examples [00:38, 972.13 examples/s]36160 examples [00:38, 989.05 examples/s]36261 examples [00:38, 995.12 examples/s]36361 examples [00:38, 983.70 examples/s]36460 examples [00:38, 978.98 examples/s]36560 examples [00:39, 984.56 examples/s]36660 examples [00:39, 988.77 examples/s]36760 examples [00:39, 990.49 examples/s]36860 examples [00:39, 958.45 examples/s]36957 examples [00:39, 942.22 examples/s]37058 examples [00:39, 959.15 examples/s]37155 examples [00:39, 957.37 examples/s]37254 examples [00:39, 965.26 examples/s]37351 examples [00:39, 961.13 examples/s]37448 examples [00:39, 962.97 examples/s]37546 examples [00:40, 964.91 examples/s]37647 examples [00:40, 975.43 examples/s]37746 examples [00:40, 978.49 examples/s]37844 examples [00:40, 971.17 examples/s]37943 examples [00:40, 973.22 examples/s]38041 examples [00:40, 960.73 examples/s]38138 examples [00:40, 916.22 examples/s]38232 examples [00:40, 922.60 examples/s]38331 examples [00:40, 941.44 examples/s]38426 examples [00:41, 923.29 examples/s]38519 examples [00:41, 922.57 examples/s]38614 examples [00:41, 929.94 examples/s]38708 examples [00:41, 930.01 examples/s]38808 examples [00:41, 947.70 examples/s]38906 examples [00:41, 956.57 examples/s]39002 examples [00:41, 946.40 examples/s]39100 examples [00:41, 954.53 examples/s]39199 examples [00:41, 964.52 examples/s]39297 examples [00:41, 967.17 examples/s]39394 examples [00:42, 952.93 examples/s]39492 examples [00:42, 958.59 examples/s]39591 examples [00:42, 967.27 examples/s]39688 examples [00:42, 927.84 examples/s]39787 examples [00:42, 945.62 examples/s]39886 examples [00:42, 958.08 examples/s]39983 examples [00:42, 945.14 examples/s]40078 examples [00:42, 895.75 examples/s]40169 examples [00:42, 895.25 examples/s]40269 examples [00:42, 923.19 examples/s]40362 examples [00:43, 918.33 examples/s]40464 examples [00:43, 945.07 examples/s]40566 examples [00:43, 965.51 examples/s]40663 examples [00:43, 964.44 examples/s]40764 examples [00:43, 975.93 examples/s]40866 examples [00:43, 986.32 examples/s]40967 examples [00:43, 992.73 examples/s]41069 examples [00:43, 999.42 examples/s]41170 examples [00:43, 992.56 examples/s]41271 examples [00:43, 996.53 examples/s]41371 examples [00:44, 994.65 examples/s]41471 examples [00:44, 994.05 examples/s]41571 examples [00:44, 975.39 examples/s]41669 examples [00:44, 943.60 examples/s]41770 examples [00:44, 960.06 examples/s]41872 examples [00:44, 976.17 examples/s]41970 examples [00:44, 965.05 examples/s]42072 examples [00:44, 978.43 examples/s]42171 examples [00:44, 978.10 examples/s]42269 examples [00:45, 968.08 examples/s]42369 examples [00:45, 976.07 examples/s]42470 examples [00:45, 984.37 examples/s]42571 examples [00:45, 990.03 examples/s]42671 examples [00:45, 982.22 examples/s]42773 examples [00:45, 990.97 examples/s]42873 examples [00:45, 970.42 examples/s]42977 examples [00:45, 989.98 examples/s]43077 examples [00:45, 982.06 examples/s]43176 examples [00:45, 960.64 examples/s]43273 examples [00:46, 951.88 examples/s]43371 examples [00:46, 959.38 examples/s]43471 examples [00:46, 968.84 examples/s]43568 examples [00:46, 965.43 examples/s]43666 examples [00:46, 968.50 examples/s]43766 examples [00:46, 976.67 examples/s]43866 examples [00:46, 982.35 examples/s]43966 examples [00:46, 987.14 examples/s]44065 examples [00:46, 983.21 examples/s]44164 examples [00:46, 976.95 examples/s]44264 examples [00:47, 981.06 examples/s]44365 examples [00:47, 987.20 examples/s]44466 examples [00:47, 991.38 examples/s]44566 examples [00:47, 991.14 examples/s]44666 examples [00:47, 987.04 examples/s]44765 examples [00:47, 986.75 examples/s]44864 examples [00:47, 936.42 examples/s]44959 examples [00:47, 906.98 examples/s]45056 examples [00:47, 923.65 examples/s]45151 examples [00:47, 929.82 examples/s]45247 examples [00:48, 936.89 examples/s]45342 examples [00:48, 938.34 examples/s]45438 examples [00:48, 943.74 examples/s]45536 examples [00:48, 951.75 examples/s]45635 examples [00:48, 962.01 examples/s]45734 examples [00:48, 967.88 examples/s]45833 examples [00:48, 972.62 examples/s]45932 examples [00:48, 975.52 examples/s]46033 examples [00:48, 984.23 examples/s]46132 examples [00:49, 971.50 examples/s]46233 examples [00:49, 981.60 examples/s]46332 examples [00:49, 956.19 examples/s]46428 examples [00:49, 932.90 examples/s]46525 examples [00:49, 942.99 examples/s]46622 examples [00:49, 949.91 examples/s]46721 examples [00:49, 959.44 examples/s]46818 examples [00:49, 958.20 examples/s]46915 examples [00:49, 960.68 examples/s]47015 examples [00:49, 970.42 examples/s]47113 examples [00:50, 972.68 examples/s]47213 examples [00:50, 980.53 examples/s]47313 examples [00:50, 986.09 examples/s]47413 examples [00:50, 989.28 examples/s]47513 examples [00:50, 991.19 examples/s]47613 examples [00:50, 983.71 examples/s]47713 examples [00:50, 988.38 examples/s]47813 examples [00:50, 989.93 examples/s]47913 examples [00:50, 965.27 examples/s]48010 examples [00:50, 939.70 examples/s]48105 examples [00:51, 920.86 examples/s]48202 examples [00:51, 934.75 examples/s]48301 examples [00:51, 950.59 examples/s]48397 examples [00:51, 947.35 examples/s]48498 examples [00:51, 962.74 examples/s]48600 examples [00:51, 977.12 examples/s]48702 examples [00:51, 989.49 examples/s]48806 examples [00:51, 1002.21 examples/s]48908 examples [00:51, 1005.17 examples/s]49009 examples [00:51, 979.70 examples/s] 49108 examples [00:52, 939.24 examples/s]49203 examples [00:52, 932.14 examples/s]49303 examples [00:52, 949.04 examples/s]49401 examples [00:52, 956.65 examples/s]49501 examples [00:52, 966.69 examples/s]49598 examples [00:52, 910.09 examples/s]49690 examples [00:52, 912.61 examples/s]49786 examples [00:52, 925.15 examples/s]49881 examples [00:52, 931.95 examples/s]49978 examples [00:53, 939.28 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 7867/50000 [00:00<00:00, 78669.70 examples/s] 41%|████      | 20252/50000 [00:00<00:00, 88337.27 examples/s] 66%|██████▌   | 32920/50000 [00:00<00:00, 97158.52 examples/s] 91%|█████████ | 45600/50000 [00:00<00:00, 104485.14 examples/s]                                                                0 examples [00:00, ? examples/s]80 examples [00:00, 791.67 examples/s]166 examples [00:00, 810.35 examples/s]257 examples [00:00, 836.64 examples/s]348 examples [00:00, 857.36 examples/s]446 examples [00:00, 888.79 examples/s]546 examples [00:00, 917.63 examples/s]643 examples [00:00, 931.04 examples/s]736 examples [00:00, 928.55 examples/s]829 examples [00:00, 926.58 examples/s]928 examples [00:01, 942.61 examples/s]1025 examples [00:01, 950.16 examples/s]1125 examples [00:01, 962.26 examples/s]1221 examples [00:01, 954.21 examples/s]1316 examples [00:01, 931.05 examples/s]1409 examples [00:01, 930.56 examples/s]1505 examples [00:01, 937.12 examples/s]1603 examples [00:01, 948.65 examples/s]1701 examples [00:01, 956.92 examples/s]1798 examples [00:01, 958.71 examples/s]1894 examples [00:02, 945.00 examples/s]1997 examples [00:02, 967.09 examples/s]2098 examples [00:02, 977.60 examples/s]2197 examples [00:02, 979.83 examples/s]2296 examples [00:02, 936.85 examples/s]2394 examples [00:02, 949.00 examples/s]2490 examples [00:02, 918.16 examples/s]2583 examples [00:02, 912.86 examples/s]2675 examples [00:02, 913.09 examples/s]2778 examples [00:02, 942.67 examples/s]2877 examples [00:03, 955.21 examples/s]2978 examples [00:03, 970.76 examples/s]3083 examples [00:03, 992.88 examples/s]3184 examples [00:03, 995.19 examples/s]3284 examples [00:03, 993.12 examples/s]3387 examples [00:03, 1003.19 examples/s]3491 examples [00:03, 1012.23 examples/s]3596 examples [00:03, 1020.33 examples/s]3699 examples [00:03, 998.01 examples/s] 3799 examples [00:03, 978.69 examples/s]3898 examples [00:04, 947.35 examples/s]4001 examples [00:04, 970.48 examples/s]4104 examples [00:04, 985.24 examples/s]4206 examples [00:04, 993.21 examples/s]4307 examples [00:04, 997.27 examples/s]4407 examples [00:04, 961.80 examples/s]4505 examples [00:04, 966.99 examples/s]4603 examples [00:04, 961.96 examples/s]4700 examples [00:04, 963.44 examples/s]4802 examples [00:05, 978.75 examples/s]4901 examples [00:05, 977.72 examples/s]4999 examples [00:05, 975.54 examples/s]5097 examples [00:05, 974.02 examples/s]5195 examples [00:05, 966.15 examples/s]5297 examples [00:05, 980.29 examples/s]5396 examples [00:05, 980.42 examples/s]5495 examples [00:05, 955.95 examples/s]5595 examples [00:05, 966.25 examples/s]5696 examples [00:05, 977.55 examples/s]5796 examples [00:06, 982.58 examples/s]5898 examples [00:06, 992.90 examples/s]5998 examples [00:06, 982.75 examples/s]6100 examples [00:06, 993.25 examples/s]6200 examples [00:06, 971.68 examples/s]6303 examples [00:06, 985.83 examples/s]6404 examples [00:06, 990.41 examples/s]6505 examples [00:06, 994.70 examples/s]6605 examples [00:06, 992.38 examples/s]6705 examples [00:06, 991.23 examples/s]6805 examples [00:07, 983.97 examples/s]6904 examples [00:07, 984.03 examples/s]7005 examples [00:07, 991.25 examples/s]7105 examples [00:07, 956.78 examples/s]7202 examples [00:07, 960.12 examples/s]7303 examples [00:07, 973.48 examples/s]7402 examples [00:07, 978.04 examples/s]7503 examples [00:07, 984.98 examples/s]7602 examples [00:07, 972.90 examples/s]7700 examples [00:07, 957.93 examples/s]7802 examples [00:08, 973.42 examples/s]7901 examples [00:08, 977.76 examples/s]8004 examples [00:08, 992.20 examples/s]8105 examples [00:08, 997.26 examples/s]8207 examples [00:08, 1003.47 examples/s]8308 examples [00:08, 1002.49 examples/s]8409 examples [00:08, 980.61 examples/s] 8508 examples [00:08, 981.66 examples/s]8613 examples [00:08, 998.78 examples/s]8714 examples [00:09, 971.31 examples/s]8812 examples [00:09, 934.40 examples/s]8911 examples [00:09, 949.03 examples/s]9009 examples [00:09, 955.74 examples/s]9108 examples [00:09, 962.99 examples/s]9205 examples [00:09, 943.47 examples/s]9308 examples [00:09, 966.62 examples/s]9409 examples [00:09, 978.49 examples/s]9511 examples [00:09, 987.93 examples/s]9613 examples [00:09, 997.25 examples/s]9714 examples [00:10, 999.28 examples/s]9815 examples [00:10, 974.93 examples/s]9913 examples [00:10, 974.89 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteWOQ33L/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteWOQ33L/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fccf0d92950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fccf0d92950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fccf0d92950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fcd3c77f390>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fcc7a20a710>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fccf0d92950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fccf0d92950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fccf0d92950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fccf0d80be0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fccee38a518>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fcc688116a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fcc688116a8> 

  function with postional parmater data_info <function split_train_valid at 0x7fcc688116a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fcc688117b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fcc688117b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fcc688117b8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=c57af0e0887bf3f059bf23fd4e55a93a8d7159d93a4762710de1abe1c6262f6c
  Stored in directory: /tmp/pip-ephem-wheel-cache-40kg9iiy/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:14:44, 13.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:00:26, 18.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:09:37, 26.1kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:25:17, 37.3kB/s].vector_cache/glove.6B.zip:   0%|          | 3.29M/862M [00:01<4:29:07, 53.2kB/s].vector_cache/glove.6B.zip:   0%|          | 3.63M/862M [00:01<3:12:08, 74.5kB/s].vector_cache/glove.6B.zip:   1%|          | 7.23M/862M [00:01<2:14:17, 106kB/s] .vector_cache/glove.6B.zip:   1%|          | 10.2M/862M [00:01<1:33:49, 151kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.7M/862M [00:01<1:05:31, 216kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.9M/862M [00:02<45:49, 307kB/s]  .vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:02<32:00, 438kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.5M/862M [00:02<22:24, 622kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.3M/862M [00:02<15:43, 883kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.0M/862M [00:02<11:01, 1.25MB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.1M/862M [00:02<07:49, 1.76MB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.7M/862M [00:02<05:37, 2.44MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.1M/862M [00:02<04:01, 3.39MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.5M/862M [00:02<02:58, 4.57MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.1M/862M [00:03<02:39, 5.09MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:03<02:01, 6.68MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.5M/862M [00:03<01:31, 8.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<01:51, 7.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:07<31:54:17, 7.01kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:07<22:35:38, 9.89kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:08<15:53:35, 14.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:08<11:10:08, 20.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.1M/862M [00:08<7:50:20, 28.5kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 58.8M/862M [00:08<5:29:26, 40.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:08<3:50:08, 58.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:09<2:50:41, 78.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:09<2:00:27, 111kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:09<1:25:07, 157kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.6M/862M [00:10<59:51, 222kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:11<45:23, 293kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.9M/862M [00:11<31:57, 415kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:11<22:32, 587kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<16:13, 811kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:14<23:33:11, 9.32kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:14<16:42:31, 13.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:15<11:44:42, 18.7kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:15<8:13:29, 26.7kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:15<5:45:28, 38.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:16<4:03:38, 53.8kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:16<2:51:49, 76.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:16<2:01:00, 108kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:17<1:24:39, 154kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:18<1:04:24, 202kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:18<46:31, 280kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:18<33:24, 390kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:19<23:38, 550kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:20<19:39, 659kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:20<15:06, 857kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:20<10:57, 1.18MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:21<07:47, 1.65MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:22<56:32, 228kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:22<40:53, 315kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:22<28:51, 446kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:24<23:09, 554kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:24<17:31, 731kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:24<12:34, 1.02MB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:26<11:47, 1.08MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:26<09:35, 1.33MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:26<07:01, 1.81MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<07:54, 1.60MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<06:49, 1.86MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:28<05:07, 2.47MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<03:46, 3.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:34<32:49, 384kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:34<36:07, 349kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:34<28:19, 445kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:34<20:33, 613kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:35<14:29, 866kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:40<39:37, 317kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:40<41:46, 300kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:40<33:54, 370kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:41<25:10, 498kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:41<18:10, 689kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:41<12:57, 965kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:41<09:18, 1.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:42<44:06, 283kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:42<32:46, 381kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:42<23:58, 520kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:42<17:12, 724kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:43<12:10, 1.02MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:44<23:35, 526kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:44<17:26, 711kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:44<12:33, 986kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:44<08:54, 1.38MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:46<30:05, 410kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:46<24:37, 501kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:46<18:44, 658kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:47<15:40, 786kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:47<11:38, 1.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:47<08:15, 1.49MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:49<12:10, 1.01MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:49<09:41, 1.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:49<07:01, 1.74MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:49<05:05, 2.39MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:51<27:01, 451kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:51<20:01, 608kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:51<14:42, 827kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:51<10:28, 1.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:53<11:26, 1.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:53<09:17, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:53<06:49, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:53<04:55, 2.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:54<16:12, 743kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:55<12:35, 956kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:55<09:04, 1.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:55<06:29, 1.84MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:56<2:26:20, 81.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:57<1:43:40, 115kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:57<1:13:11, 163kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:57<51:25, 232kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:58<38:50, 307kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:59<28:27, 418kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:59<20:51, 571kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:59<14:52, 799kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:59<10:37, 1.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [01:01<15:48, 749kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [01:01<14:32, 814kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [01:01<10:58, 1.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [01:01<08:02, 1.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [01:03<07:53, 1.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [01:03<07:18, 1.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [01:03<05:32, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:03<04:01, 2.91MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:05<10:36, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:05<09:50, 1.19MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:05<07:41, 1.52MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [01:05<05:44, 2.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:05<04:11, 2.78MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:07<12:20, 942kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:07<09:39, 1.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:07<07:15, 1.60MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:07<05:11, 2.23MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:09<21:06, 548kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:09<16:33, 698kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:09<12:15, 943kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:09<17:16, 669kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:09<12:26, 927kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:09<08:45, 1.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:10<06:13, 1.83MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:10<04:25, 2.56MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:15<32:38, 348kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:15<32:23, 350kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:15<25:30, 445kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:15<18:25, 615kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:15<13:10, 859kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:17<11:32, 977kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:17<09:23, 1.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:17<06:50, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:19<07:09, 1.56MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:19<07:31, 1.49MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:19<05:59, 1.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:19<04:29, 2.49MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:21<05:30, 2.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:21<05:00, 2.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:21<03:47, 2.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:30<17:02, 650kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:30<21:36, 513kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:31<17:49, 621kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:31<13:07, 843kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:31<09:16, 1.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:31<07:33, 1.45MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:32<05:30, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:32<04:07, 2.65MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:32<03:09, 3.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:33<31:23, 348kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:34<24:21, 448kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:34<19:11, 569kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:34<14:43, 740kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:34<11:03, 986kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:34<08:00, 1.36MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:35<08:01, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:35<06:46, 1.60MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:36<05:12, 2.08MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:36<03:50, 2.81MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:37<06:08, 1.75MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:37<05:55, 1.82MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:38<04:31, 2.38MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:38<03:18, 3.25MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:39<11:13, 954kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:39<09:13, 1.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:40<06:42, 1.59MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:40<04:53, 2.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:42<11:48, 901kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:42<18:54, 563kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:42<15:48, 673kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:43<11:55, 892kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:43<08:37, 1.23MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:43<06:16, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:43<05:50, 1.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:43<04:15, 2.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:44<03:07, 3.36MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:50<1:17:21, 136kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:50<1:05:47, 160kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:50<48:12, 218kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:50<35:03, 299kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:50<24:45, 423kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:51<17:27, 598kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:52<19:09, 545kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:52<15:05, 691kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:52<11:53, 876kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:52<09:08, 1.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:52<07:03, 1.47MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:52<05:38, 1.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:53<04:34, 2.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:53<03:24, 3.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:54<13:30, 767kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:54<10:53, 952kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:54<08:34, 1.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:54<06:30, 1.59MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:54<04:50, 2.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:54<03:42, 2.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:56<07:31, 1.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:56<07:39, 1.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:56<07:04, 1.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:56<06:28, 1.59MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:56<05:42, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:56<04:57, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:56<04:01, 2.55MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:57<03:08, 3.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:57<02:35, 3.96MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:58<07:45, 1.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:58<07:15, 1.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:58<06:08, 1.66MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:58<04:49, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:58<03:39, 2.78MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:59<02:50, 3.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [02:00<06:47, 1.50MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [02:00<06:28, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [02:00<05:34, 1.82MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [02:00<04:29, 2.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [02:00<03:25, 2.95MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [02:01<02:39, 3.80MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [02:02<06:43, 1.50MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [02:02<06:23, 1.58MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [02:02<05:51, 1.72MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [02:02<04:51, 2.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [02:02<03:45, 2.68MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [02:02<02:59, 3.35MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [02:03<02:21, 4.24MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [02:04<09:26, 1.06MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [02:04<08:43, 1.15MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [02:04<07:52, 1.27MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [02:04<06:24, 1.56MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [02:04<04:56, 2.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [02:05<03:42, 2.69MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [02:05<02:51, 3.48MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [02:06<10:44, 926kB/s] .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [02:06<09:17, 1.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [02:06<07:32, 1.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [02:06<05:39, 1.75MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [02:06<04:10, 2.37MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [02:08<05:54, 1.67MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [02:08<06:52, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [02:08<08:26, 1.17MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [02:08<06:54, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [02:08<05:38, 1.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [02:08<04:10, 2.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [02:13<08:56, 1.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [02:13<16:33, 592kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [02:13<13:45, 712kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [02:13<10:50, 904kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [02:13<09:25, 1.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [02:14<1:31:53, 107kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [02:14<1:04:12, 152kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [02:15<47:00, 207kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:15<33:13, 292kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [02:15<23:16, 415kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [02:17<22:41, 425kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [02:17<18:56, 509kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:17<14:08, 682kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:17<10:14, 940kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:17<07:15, 1.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:19<18:41, 513kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:19<13:58, 685kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:19<10:19, 926kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:19<07:18, 1.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:21<11:44, 811kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:21<09:55, 959kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:21<07:36, 1.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:21<05:35, 1.69MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:21<04:04, 2.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:23<08:04, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:23<07:09, 1.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:23<05:39, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:23<04:12, 2.24MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:25<04:54, 1.91MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:25<04:51, 1.93MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:25<04:00, 2.33MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:25<03:00, 3.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:27<04:20, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:27<04:22, 2.12MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:27<03:42, 2.50MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:27<02:48, 3.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:29<04:05, 2.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:29<06:12, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:29<05:51, 1.58MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:29<04:39, 1.98MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:29<03:26, 2.68MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:29<02:33, 3.58MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:31<21:31, 426kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:31<17:53, 512kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:31<14:01, 653kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:31<10:20, 886kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:31<07:24, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:33<07:26, 1.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:33<08:19, 1.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:33<08:16, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:33<07:07, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:33<05:21, 1.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:33<03:53, 2.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:35<06:21, 1.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:35<08:11, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:35<08:19, 1.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:35<06:41, 1.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:35<05:09, 1.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:36<05:28, 1.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:36<3:01:23, 49.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:37<2:06:50, 70.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:37<1:28:33, 101kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:38<1:07:38, 132kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:38<48:19, 185kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:38<34:18, 260kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:38<24:08, 368kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:40<19:09, 463kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:40<15:36, 567kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:40<11:56, 742kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:40<08:48, 1.00MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:40<06:27, 1.37MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:40<04:51, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:43<07:59, 1.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:43<14:43, 597kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:43<12:22, 711kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:43<09:10, 957kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:43<06:29, 1.35MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:45<09:41, 900kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:45<09:19, 935kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:45<08:12, 1.06MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:45<06:15, 1.39MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:45<04:31, 1.92MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:47<05:31, 1.57MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:47<06:20, 1.36MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:47<05:25, 1.59MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:47<04:05, 2.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:47<02:59, 2.87MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:49<05:47, 1.48MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:49<07:16, 1.18MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:49<06:36, 1.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:49<05:05, 1.68MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:49<03:57, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:49<02:55, 2.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:51<05:30, 1.54MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:51<05:43, 1.49MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:51<04:27, 1.91MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:51<03:16, 2.59MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:53<04:45, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:53<06:53, 1.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:53<07:38, 1.11MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:53<06:07, 1.38MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:53<04:32, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:53<03:23, 2.48MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:53<02:34, 3.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:55<12:49, 653kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:55<10:03, 833kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:55<07:34, 1.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:55<05:30, 1.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:59<07:47, 1.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:59<13:05, 635kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:59<13:37, 610kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:59<10:22, 800kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:59<07:47, 1.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:59<05:34, 1.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [03:01<06:22, 1.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [03:01<07:39, 1.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [03:01<08:26, 976kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [03:01<08:49, 933kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [03:01<10:05, 816kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [03:01<09:55, 830kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [03:01<09:17, 886kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [03:01<06:59, 1.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [03:02<07:20, 1.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [03:02<05:54, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [03:02<04:45, 1.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [03:02<04:00, 2.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [03:02<03:57, 2.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [03:02<05:09, 1.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [03:02<06:58, 1.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [03:02<06:39, 1.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [03:02<05:28, 1.50MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [03:02<04:34, 1.79MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [03:03<03:49, 2.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [03:04<18:10, 450kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [03:05<21:25, 381kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [03:05<17:10, 476kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [03:05<13:38, 599kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [03:05<10:18, 792kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [03:05<07:46, 1.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [03:05<06:02, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [03:05<04:49, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [03:05<04:04, 2.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [03:05<04:12, 1.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [03:06<03:36, 2.25MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [03:06<03:09, 2.57MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [03:06<02:43, 2.97MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [03:06<02:32, 3.19MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [03:06<41:27, 195kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [03:06<31:08, 260kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [03:06<24:03, 337kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [03:06<18:49, 430kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [03:06<15:13, 532kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [03:06<12:28, 649kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [03:07<10:11, 793kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [03:07<08:23, 963kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [03:07<06:33, 1.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [03:07<05:07, 1.58MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [03:07<04:10, 1.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [03:07<03:30, 2.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [03:07<02:59, 2.69MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [03:07<02:37, 3.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [03:07<02:24, 3.35MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [03:08<11:25, 703kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [03:08<09:45, 824kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [03:08<08:33, 939kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [03:08<08:22, 958kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [03:08<08:00, 1.00MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [03:08<07:19, 1.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [03:09<06:22, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [03:09<05:46, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [03:09<05:41, 1.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [03:09<04:46, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [03:09<03:59, 2.00MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [03:09<03:22, 2.37MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [03:09<04:21, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [03:10<03:34, 2.23MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [03:10<03:01, 2.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [03:10<02:36, 3.04MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [03:10<02:29, 3.19MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [03:10<02:44, 2.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [03:10<02:31, 3.14MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [03:10<02:28, 3.21MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [03:10<03:00, 2.64MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [03:10<04:11, 1.89MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [03:11<04:32, 1.74MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [03:11<04:25, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [03:11<03:43, 2.13MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [03:11<12:42, 623kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [03:11<09:40, 817kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [03:12<07:18, 1.08MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [03:12<06:02, 1.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [03:12<04:49, 1.63MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [03:12<04:09, 1.89MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [03:12<04:03, 1.94MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [03:12<03:59, 1.97MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [03:12<03:14, 2.42MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [03:12<02:54, 2.70MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [03:12<02:27, 3.20MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [03:13<10:07, 775kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [03:13<09:52, 795kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [03:14<07:39, 1.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [03:14<05:52, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [03:14<04:36, 1.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [03:14<04:26, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [03:14<04:12, 1.85MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [03:14<03:32, 2.20MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [03:14<02:54, 2.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [03:14<02:33, 3.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [03:15<05:56, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [03:15<05:56, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [03:15<04:50, 1.61MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [03:16<03:51, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [03:16<03:08, 2.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [03:16<02:42, 2.85MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [03:16<03:01, 2.57MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [03:16<03:40, 2.11MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [03:16<04:31, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [03:16<05:31, 1.40MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [03:16<05:53, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [03:16<05:43, 1.35MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [03:16<04:50, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [03:17<04:05, 1.88MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [03:17<03:20, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [03:17<08:21, 923kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [03:17<06:23, 1.20MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [03:17<04:51, 1.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [03:18<03:50, 2.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [03:18<03:03, 2.51MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [03:18<02:42, 2.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [03:18<02:53, 2.65MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [03:18<02:46, 2.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [03:18<02:23, 3.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [03:19<15:38, 488kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [03:19<12:47, 597kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [03:19<09:19, 818kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [03:20<07:01, 1.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [03:20<05:12, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [03:20<04:00, 1.90MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:20<03:31, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:20<03:13, 2.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:20<02:46, 2.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [03:21<08:02, 941kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [03:21<07:13, 1.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:21<05:31, 1.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:22<04:08, 1.82MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:22<03:07, 2.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:22<02:54, 2.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:22<02:42, 2.78MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:23<07:07, 1.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:23<07:52, 953kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:23<06:11, 1.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:24<04:32, 1.65MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:24<03:30, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:24<02:58, 2.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:24<02:41, 2.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:24<02:19, 3.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:25<07:21, 1.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:25<07:28, 993kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:25<05:49, 1.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:26<04:16, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:26<03:13, 2.29MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:26<02:55, 2.52MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:26<02:37, 2.80MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:27<07:11, 1.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [03:27<07:04, 1.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:27<05:26, 1.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:28<04:01, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:28<03:01, 2.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:28<02:30, 2.92MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:29<08:52, 822kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:29<07:56, 919kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:29<05:59, 1.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:29<04:25, 1.64MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:30<03:15, 2.23MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:31<05:07, 1.41MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:31<06:40, 1.08MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:31<05:18, 1.36MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:31<03:53, 1.85MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:32<02:53, 2.48MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:32<02:12, 3.25MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:33<2:05:04, 57.2kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:33<1:30:13, 79.3kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:33<1:03:43, 112kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:33<44:37, 160kB/s]  .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:35<32:47, 216kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:35<25:20, 280kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:35<18:30, 382kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:35<13:14, 534kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:35<09:29, 743kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:36<06:53, 1.02MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:36<05:08, 1.37MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:36<03:56, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:37<1:21:45, 85.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:37<59:13, 118kB/s]   .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:37<42:07, 166kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:37<29:45, 235kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:37<20:53, 334kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:39<16:55, 411kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:39<14:12, 489kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:39<11:01, 630kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:39<08:03, 861kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:39<05:47, 1.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:39<04:09, 1.66MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:41<17:44, 388kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:41<14:27, 476kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:41<10:43, 640kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:41<07:45, 885kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:41<05:31, 1.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:42<04:07, 1.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:43<11:48, 577kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:43<09:07, 746kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:43<06:34, 1.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:43<04:40, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:45<12:19, 547kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:45<10:18, 653kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:45<07:43, 872kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:45<05:34, 1.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:45<04:00, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:47<06:46, 985kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:47<06:06, 1.09MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:47<04:37, 1.44MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:47<03:19, 1.99MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:49<07:40, 860kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:49<06:38, 995kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:49<04:59, 1.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:49<03:37, 1.81MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:51<04:11, 1.56MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:51<04:38, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:51<03:40, 1.77MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:51<02:42, 2.40MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:51<02:00, 3.23MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:53<09:42, 666kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:53<08:29, 760kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:53<06:32, 987kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:53<04:50, 1.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:53<03:39, 1.76MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:53<02:50, 2.26MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:53<02:14, 2.86MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:55<04:50, 1.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:55<04:24, 1.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:55<03:18, 1.93MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:55<02:26, 2.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:55<01:51, 3.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:57<11:22, 556kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:57<09:09, 691kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:57<06:47, 930kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:57<04:52, 1.29MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:57<03:33, 1.76MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:59<05:52, 1.07MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:59<05:38, 1.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:59<04:16, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:59<03:11, 1.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:59<02:21, 2.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [04:01<04:46, 1.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [04:01<04:54, 1.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [04:01<04:03, 1.52MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [04:01<03:00, 2.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [04:01<02:16, 2.71MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [04:03<03:33, 1.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [04:03<04:07, 1.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [04:03<03:13, 1.89MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [04:03<02:27, 2.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [04:03<01:54, 3.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [04:03<02:02, 2.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [04:05<03:33, 1.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [04:05<04:37, 1.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [04:05<03:51, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [04:05<02:59, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [04:05<02:18, 2.61MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [04:05<01:44, 3.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [04:07<04:04, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [04:07<04:58, 1.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [04:07<03:55, 1.52MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [04:07<03:01, 1.97MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [04:07<02:19, 2.56MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [04:07<01:48, 3.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [04:07<01:26, 4.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [04:09<27:05, 218kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [04:09<20:40, 286kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [04:09<14:58, 394kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [04:09<10:47, 547kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [04:09<07:50, 751kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [04:09<05:38, 1.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [04:11<05:39, 1.03MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [04:11<05:47, 1.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [04:11<04:37, 1.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [04:11<03:36, 1.61MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [04:11<02:50, 2.05MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [04:11<02:07, 2.72MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [04:13<03:08, 1.84MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [04:13<03:11, 1.81MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [04:13<02:27, 2.34MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [04:13<01:57, 2.92MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [04:13<01:37, 3.51MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [04:13<01:14, 4.60MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [04:15<27:53, 205kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [04:15<21:20, 267kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [04:15<15:27, 369kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [04:15<11:06, 512kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [04:15<08:10, 695kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [04:15<05:54, 960kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [04:15<04:21, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [04:17<05:04, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [04:17<04:33, 1.24MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [04:17<03:25, 1.64MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [04:17<02:30, 2.24MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [04:19<03:32, 1.57MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [04:19<04:11, 1.33MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [04:19<03:21, 1.65MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [04:19<02:35, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [04:19<01:54, 2.89MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:21<03:33, 1.55MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:21<03:56, 1.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:21<03:05, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:21<02:32, 2.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:21<02:11, 2.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:21<01:54, 2.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:21<01:24, 3.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:23<10:30, 517kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:23<08:06, 669kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:23<05:55, 914kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:23<04:22, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:23<03:10, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:25<04:12, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:25<04:45, 1.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:25<04:01, 1.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:25<03:13, 1.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:25<02:25, 2.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:25<01:47, 2.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:27<03:31, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:27<03:40, 1.44MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:27<03:58, 1.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:27<04:01, 1.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:27<03:48, 1.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:27<02:59, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:27<02:19, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:27<01:45, 2.99MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:35<11:10, 467kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:35<13:29, 387kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:35<11:15, 464kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:35<08:44, 596kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:35<06:43, 774kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:35<04:55, 1.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:35<03:32, 1.46MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:37<03:53, 1.33MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:37<04:00, 1.29MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:37<04:00, 1.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:37<04:13, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:37<04:17, 1.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:37<04:15, 1.21MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:37<03:56, 1.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:37<03:04, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:38<02:16, 2.25MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:39<02:37, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:39<02:30, 2.03MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:39<02:29, 2.03MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:39<02:16, 2.22MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:39<01:53, 2.67MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:39<01:36, 3.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:39<01:39, 3.05MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:39<01:29, 3.37MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:39<01:21, 3.70MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:40<01:11, 4.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:41<03:58, 1.26MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:41<03:25, 1.46MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:41<02:56, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:41<02:29, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:41<02:12, 2.25MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:41<01:53, 2.64MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:41<01:41, 2.95MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:41<01:29, 3.31MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:41<01:09, 4.29MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:43<08:54, 555kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:43<07:35, 651kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:43<06:29, 760kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:43<05:33, 889kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:43<04:28, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:43<03:26, 1.43MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:43<02:45, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:43<02:10, 2.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:43<01:36, 3.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:45<05:26, 897kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:45<04:59, 976kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:45<04:46, 1.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:45<03:54, 1.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:45<03:06, 1.56MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:45<02:37, 1.85MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:45<02:15, 2.14MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:45<01:52, 2.57MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:45<01:29, 3.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:46<01:10, 4.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:47<21:08, 227kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:47<15:30, 309kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:47<11:17, 424kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:47<08:06, 590kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:47<05:52, 813kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:47<04:22, 1.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:47<03:19, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:47<02:29, 1.90MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:49<06:52, 688kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:49<06:05, 777kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:49<04:48, 982kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:49<03:34, 1.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:49<02:40, 1.76MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:49<02:04, 2.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:49<01:41, 2.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:49<01:23, 3.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:51<07:40, 607kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:51<06:10, 755kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:51<04:32, 1.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:51<03:21, 1.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:51<02:30, 1.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:51<02:15, 2.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:51<02:02, 2.26MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:51<01:44, 2.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:51<01:29, 3.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:53<11:05, 414kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:53<08:09, 562kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:53<05:46, 791kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:53<04:10, 1.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:53<03:03, 1.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:55<12:21, 366kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:55<09:54, 457kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:55<07:11, 627kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:55<05:11, 867kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:55<03:57, 1.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:55<03:01, 1.48MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:55<02:22, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:55<01:51, 2.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:57<05:23, 826kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:57<04:36, 967kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:57<03:28, 1.28MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:57<02:38, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:57<02:05, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:57<01:44, 2.54MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:57<01:29, 2.97MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:57<01:11, 3.70MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:59<05:25, 808kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:59<04:35, 954kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:59<03:25, 1.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:59<02:35, 1.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:59<01:58, 2.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:59<01:39, 2.62MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:59<01:21, 3.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [05:01<03:23, 1.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [05:01<03:07, 1.38MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [05:01<02:38, 1.63MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [05:01<02:04, 2.08MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [05:01<01:36, 2.67MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [05:01<01:27, 2.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [05:01<01:18, 3.25MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [05:01<01:02, 4.10MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [05:03<06:29, 655kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [05:03<05:20, 795kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [05:03<03:52, 1.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [05:03<02:48, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [05:03<02:05, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [05:05<03:27, 1.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [05:05<03:34, 1.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [05:05<02:59, 1.39MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [05:05<02:16, 1.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [05:05<01:41, 2.44MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [05:05<01:19, 3.10MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [05:06<02:34, 1.59MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [05:07<03:16, 1.25MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [05:07<02:37, 1.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [05:07<01:58, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [05:07<01:25, 2.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [05:08<04:11, 965kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [05:09<04:26, 911kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [05:09<03:51, 1.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [05:09<02:57, 1.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [05:09<02:09, 1.86MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [05:09<01:36, 2.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [05:09<01:14, 3.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [05:10<20:50, 191kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [05:11<15:52, 250kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [05:11<11:30, 345kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [05:11<08:12, 482kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [05:11<05:54, 667kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [05:11<04:12, 930kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [05:12<04:22, 892kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [05:13<03:42, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [05:13<02:42, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [05:13<02:04, 1.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [05:13<01:39, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [05:13<01:19, 2.92MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [05:14<02:36, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [05:15<02:27, 1.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [05:15<01:51, 2.05MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [05:15<01:23, 2.73MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [05:15<01:03, 3.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [05:16<03:34, 1.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [05:16<03:43, 1.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [05:17<02:52, 1.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [05:17<02:10, 1.72MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [05:17<01:37, 2.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [05:17<01:14, 2.98MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [05:18<02:53, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [05:18<03:15, 1.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [05:19<02:34, 1.43MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [05:19<01:51, 1.97MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [05:19<01:22, 2.63MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [05:20<04:22, 829kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [05:21<03:50, 941kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [05:21<02:57, 1.22MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [05:21<02:11, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [05:21<01:35, 2.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [05:23<03:30, 1.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [05:23<06:09, 578kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [05:24<05:07, 694kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [05:24<03:55, 905kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [05:24<02:50, 1.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [05:24<02:01, 1.73MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [05:25<03:29, 1.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [05:25<03:18, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [05:26<02:34, 1.35MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [05:26<01:55, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:26<01:24, 2.43MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [05:27<02:05, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [05:27<02:24, 1.42MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [05:28<01:55, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:28<01:25, 2.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:28<01:01, 3.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:29<3:05:30, 18.1kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:29<2:10:34, 25.6kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:30<1:31:17, 36.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:30<1:03:39, 52.1kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:31<44:45, 73.3kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:31<32:10, 102kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:31<22:52, 143kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:32<16:12, 202kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:32<11:25, 285kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:32<08:02, 403kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:32<05:43, 561kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:33<13:24, 239kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:33<09:53, 324kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:33<07:00, 456kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:34<04:55, 643kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:35<04:36, 682kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:35<04:07, 761kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:35<03:22, 930kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:35<02:29, 1.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:36<01:48, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:37<02:02, 1.50MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:37<02:24, 1.27MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:37<01:52, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:38<01:25, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:38<01:03, 2.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:39<01:43, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:39<02:05, 1.43MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:39<01:40, 1.79MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:40<01:15, 2.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:40<00:55, 3.16MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:41<01:55, 1.53MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:41<02:09, 1.35MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:41<01:51, 1.58MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:41<01:24, 2.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:42<01:00, 2.84MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:43<02:10, 1.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:43<02:16, 1.26MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:43<01:48, 1.59MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:43<01:20, 2.11MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:44<01:00, 2.82MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:45<01:41, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:45<02:02, 1.36MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:45<01:38, 1.70MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:45<01:15, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:46<00:54, 3.03MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:48<03:33, 767kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:48<05:13, 521kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:48<04:18, 632kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:48<03:09, 861kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:48<02:16, 1.19MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:48<01:37, 1.64MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:50<03:00, 883kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:50<02:45, 961kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:50<02:13, 1.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:50<01:37, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:50<01:12, 2.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:52<01:31, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:52<01:57, 1.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:52<02:01, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:52<01:49, 1.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:52<01:23, 1.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:52<00:59, 2.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:55<02:50, 889kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:56<04:39, 540kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:56<03:53, 646kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:56<02:51, 875kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:56<02:00, 1.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:57<02:15, 1.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:58<02:38, 928kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:58<02:53, 849kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:58<02:52, 851kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:58<02:33, 957kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:58<02:00, 1.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:58<01:29, 1.63MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:58<01:03, 2.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:59<03:04, 773kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [06:00<02:35, 918kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [06:00<02:07, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [06:00<01:37, 1.46MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [06:00<01:10, 2.00MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [06:01<01:28, 1.57MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [06:01<01:40, 1.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [06:02<01:25, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [06:02<01:04, 2.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [06:02<00:46, 2.91MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [06:03<01:34, 1.43MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [06:03<01:51, 1.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [06:04<01:41, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [06:04<01:16, 1.75MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [06:04<00:56, 2.36MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [06:04<00:41, 3.18MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [06:05<06:02, 360kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [06:05<04:57, 438kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [06:06<03:38, 595kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [06:06<02:34, 832kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [06:06<01:48, 1.17MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [06:07<04:29, 469kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [06:07<04:01, 522kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [06:08<03:30, 600kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [06:08<03:06, 676kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [06:08<02:38, 796kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [06:08<02:03, 1.02MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [06:08<01:34, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [06:08<01:08, 1.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [06:09<01:13, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [06:09<01:31, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [06:10<01:23, 1.46MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [06:10<01:04, 1.88MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [06:10<00:47, 2.53MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [06:10<00:35, 3.35MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [06:11<02:13, 884kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [06:11<02:26, 804kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [06:12<01:55, 1.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [06:12<01:23, 1.40MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [06:12<00:59, 1.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [06:13<03:47, 499kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [06:13<03:32, 536kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [06:14<02:40, 708kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [06:14<01:54, 984kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [06:14<01:21, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [06:15<01:35, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [06:15<01:57, 930kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [06:15<01:51, 985kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [06:16<01:30, 1.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [06:16<01:07, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [06:16<00:49, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [06:16<00:36, 2.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [06:18<02:31, 694kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [06:18<03:37, 484kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [06:18<02:54, 602kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [06:18<02:07, 822kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [06:18<01:30, 1.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [06:18<01:04, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [06:19<03:31, 478kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [06:20<03:05, 546kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [06:20<02:27, 686kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [06:20<01:48, 926kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [06:20<01:18, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [06:20<00:56, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [06:21<01:18, 1.23MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [06:22<01:37, 998kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [06:22<01:27, 1.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [06:22<01:08, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [06:22<00:52, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [06:22<00:38, 2.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [06:22<00:28, 3.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [06:23<02:31, 615kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [06:24<02:05, 743kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [06:24<01:35, 967kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [06:24<01:08, 1.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [06:24<00:49, 1.82MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [06:28<02:08, 691kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [06:28<02:45, 536kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [06:28<02:17, 644kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [06:29<01:40, 875kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [06:29<01:10, 1.22MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [06:29<00:50, 1.68MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [06:30<03:39, 386kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [06:30<03:12, 440kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [06:30<02:35, 543kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [06:30<01:57, 719kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [06:30<01:24, 986kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [06:31<00:59, 1.37MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [06:32<01:11, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [06:32<01:24, 950kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [06:32<01:05, 1.21MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:32<00:49, 1.61MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:33<00:35, 2.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:33<00:26, 2.95MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:34<05:22, 237kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:34<04:20, 293kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:34<03:16, 388kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:34<02:21, 536kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:34<01:39, 751kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:35<01:09, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:37<01:53, 638kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:37<02:21, 511kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:37<01:56, 619kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:37<01:31, 789kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:37<01:08, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:37<00:51, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:38<00:37, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:38<00:27, 2.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:39<01:04, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:39<00:58, 1.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:39<00:43, 1.55MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:39<00:34, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:39<00:25, 2.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:39<00:19, 3.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:41<00:53, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:41<00:50, 1.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:41<00:38, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:41<00:28, 2.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:41<00:21, 2.80MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:41<00:17, 3.54MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:43<01:08, 871kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:43<01:44, 572kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:43<01:47, 558kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:44<01:31, 654kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:44<01:08, 861kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:44<00:48, 1.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:44<00:34, 1.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:46<00:50, 1.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:46<01:24, 661kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:46<01:10, 789kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:46<00:54, 1.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:46<00:38, 1.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:46<00:27, 1.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:48<00:37, 1.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:48<00:39, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:48<00:34, 1.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:48<00:27, 1.86MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:48<00:19, 2.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:48<00:14, 3.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:50<01:19, 600kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:50<01:13, 643kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:50<00:55, 852kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:50<00:40, 1.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:50<00:28, 1.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:50<00:20, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:52<01:32, 469kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:52<01:21, 530kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:52<01:05, 655kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:52<00:48, 882kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:52<00:33, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:52<00:23, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:54<00:38, 1.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:54<00:41, 940kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:54<00:32, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:54<00:22, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:54<00:16, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:57<00:56, 619kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:57<01:10, 494kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:57<00:57, 600kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:57<00:41, 818kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:57<00:29, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:57<00:20, 1.56MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:59<00:33, 929kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:59<00:33, 920kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:59<00:32, 932kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:59<00:26, 1.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:59<00:21, 1.43MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:59<00:17, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:59<00:13, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:59<00:09, 2.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:59<00:07, 3.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [07:01<01:02, 431kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [07:01<00:53, 497kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [07:01<00:45, 588kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [07:01<00:33, 789kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [07:01<00:23, 1.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [07:01<00:16, 1.48MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [07:01<00:11, 2.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [07:03<00:34, 659kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [07:03<00:33, 662kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [07:03<00:25, 859kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [07:03<00:17, 1.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [07:03<00:11, 1.63MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [07:05<00:35, 518kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [07:05<00:32, 565kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [07:05<00:24, 745kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [07:05<00:16, 1.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [07:05<00:11, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [07:05<00:08, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [07:05<00:06, 2.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [07:07<00:14, 982kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [07:07<00:12, 1.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [07:07<00:09, 1.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [07:07<00:05, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [07:09<00:07, 1.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [07:09<00:12, 777kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [07:09<00:10, 935kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [07:09<00:08, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [07:09<00:06, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [07:09<00:04, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [07:10<00:03, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [07:10<00:02, 2.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [07:10<00:01, 3.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [07:11<00:04, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [07:11<00:03, 1.64MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [07:11<00:02, 2.20MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [07:11<00:01, 2.92MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [07:11<00:00, 3.80MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [07:18<01:58, 15.6kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [07:18<01:23, 21.8kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [07:19<00:57, 30.6kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [07:19<00:33, 43.6kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [07:19<00:15, 62.0kB/s].vector_cache/glove.6B.zip: 862MB [07:19, 1.96MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 612/400000 [00:00<01:05, 6119.48it/s]  0%|          | 1437/400000 [00:00<01:00, 6631.75it/s]  1%|          | 2291/400000 [00:00<00:55, 7106.72it/s]  1%|          | 3163/400000 [00:00<00:52, 7522.48it/s]  1%|          | 4038/400000 [00:00<00:50, 7850.97it/s]  1%|          | 4901/400000 [00:00<00:48, 8067.79it/s]  1%|▏         | 5661/400000 [00:00<00:49, 7920.88it/s]  2%|▏         | 6438/400000 [00:00<00:49, 7871.56it/s]  2%|▏         | 7273/400000 [00:00<00:49, 8008.45it/s]  2%|▏         | 8127/400000 [00:01<00:48, 8160.63it/s]  2%|▏         | 8997/400000 [00:01<00:47, 8314.97it/s]  2%|▏         | 9827/400000 [00:01<00:46, 8309.64it/s]  3%|▎         | 10665/400000 [00:01<00:46, 8330.24it/s]  3%|▎         | 11515/400000 [00:01<00:46, 8380.30it/s]  3%|▎         | 12358/400000 [00:01<00:46, 8394.97it/s]  3%|▎         | 13226/400000 [00:01<00:45, 8476.64it/s]  4%|▎         | 14073/400000 [00:01<00:46, 8383.70it/s]  4%|▎         | 14951/400000 [00:01<00:45, 8498.13it/s]  4%|▍         | 15801/400000 [00:01<00:45, 8451.32it/s]  4%|▍         | 16682/400000 [00:02<00:44, 8553.79it/s]  4%|▍         | 17540/400000 [00:02<00:44, 8560.70it/s]  5%|▍         | 18397/400000 [00:02<00:46, 8181.98it/s]  5%|▍         | 19251/400000 [00:02<00:45, 8283.94it/s]  5%|▌         | 20083/400000 [00:02<00:46, 8134.90it/s]  5%|▌         | 20911/400000 [00:02<00:46, 8177.08it/s]  5%|▌         | 21787/400000 [00:02<00:45, 8343.04it/s]  6%|▌         | 22643/400000 [00:02<00:44, 8405.32it/s]  6%|▌         | 23512/400000 [00:02<00:44, 8487.33it/s]  6%|▌         | 24363/400000 [00:02<00:44, 8469.61it/s]  6%|▋         | 25239/400000 [00:03<00:43, 8553.20it/s]  7%|▋         | 26117/400000 [00:03<00:43, 8618.94it/s]  7%|▋         | 26980/400000 [00:03<00:44, 8380.96it/s]  7%|▋         | 27835/400000 [00:03<00:44, 8430.92it/s]  7%|▋         | 28703/400000 [00:03<00:43, 8502.38it/s]  7%|▋         | 29578/400000 [00:03<00:43, 8575.01it/s]  8%|▊         | 30439/400000 [00:03<00:43, 8584.09it/s]  8%|▊         | 31299/400000 [00:03<00:43, 8537.31it/s]  8%|▊         | 32160/400000 [00:03<00:42, 8558.93it/s]  8%|▊         | 33020/400000 [00:03<00:42, 8570.85it/s]  8%|▊         | 33878/400000 [00:04<00:43, 8327.03it/s]  9%|▊         | 34713/400000 [00:04<00:44, 8258.79it/s]  9%|▉         | 35572/400000 [00:04<00:43, 8354.25it/s]  9%|▉         | 36435/400000 [00:04<00:43, 8432.50it/s]  9%|▉         | 37297/400000 [00:04<00:42, 8486.91it/s] 10%|▉         | 38186/400000 [00:04<00:42, 8601.70it/s] 10%|▉         | 39048/400000 [00:04<00:42, 8593.49it/s] 10%|▉         | 39908/400000 [00:04<00:42, 8445.67it/s] 10%|█         | 40754/400000 [00:04<00:43, 8344.91it/s] 10%|█         | 41590/400000 [00:04<00:43, 8248.36it/s] 11%|█         | 42466/400000 [00:05<00:42, 8395.02it/s] 11%|█         | 43309/400000 [00:05<00:42, 8404.68it/s] 11%|█         | 44152/400000 [00:05<00:42, 8409.61it/s] 11%|█▏        | 45004/400000 [00:05<00:42, 8440.95it/s] 11%|█▏        | 45884/400000 [00:05<00:41, 8545.22it/s] 12%|█▏        | 46772/400000 [00:05<00:40, 8640.60it/s] 12%|█▏        | 47637/400000 [00:05<00:41, 8589.40it/s] 12%|█▏        | 48497/400000 [00:05<00:42, 8368.02it/s] 12%|█▏        | 49376/400000 [00:05<00:41, 8490.19it/s] 13%|█▎        | 50257/400000 [00:05<00:40, 8582.49it/s] 13%|█▎        | 51127/400000 [00:06<00:40, 8616.68it/s] 13%|█▎        | 52007/400000 [00:06<00:40, 8670.08it/s] 13%|█▎        | 52875/400000 [00:06<00:40, 8639.96it/s] 13%|█▎        | 53768/400000 [00:06<00:39, 8723.23it/s] 14%|█▎        | 54645/400000 [00:06<00:39, 8735.35it/s] 14%|█▍        | 55524/400000 [00:06<00:39, 8750.08it/s] 14%|█▍        | 56400/400000 [00:06<00:39, 8751.18it/s] 14%|█▍        | 57276/400000 [00:06<00:39, 8695.32it/s] 15%|█▍        | 58146/400000 [00:06<00:39, 8694.23it/s] 15%|█▍        | 59019/400000 [00:06<00:39, 8703.08it/s] 15%|█▍        | 59909/400000 [00:07<00:38, 8760.47it/s] 15%|█▌        | 60786/400000 [00:07<00:39, 8602.28it/s] 15%|█▌        | 61647/400000 [00:07<00:39, 8491.67it/s] 16%|█▌        | 62498/400000 [00:07<00:40, 8327.09it/s] 16%|█▌        | 63354/400000 [00:07<00:40, 8393.98it/s] 16%|█▌        | 64241/400000 [00:07<00:39, 8530.26it/s] 16%|█▋        | 65115/400000 [00:07<00:38, 8589.66it/s] 16%|█▋        | 65975/400000 [00:07<00:39, 8506.15it/s] 17%|█▋        | 66842/400000 [00:07<00:38, 8554.00it/s] 17%|█▋        | 67699/400000 [00:08<00:38, 8524.29it/s] 17%|█▋        | 68559/400000 [00:08<00:38, 8546.50it/s] 17%|█▋        | 69415/400000 [00:08<00:38, 8519.84it/s] 18%|█▊        | 70268/400000 [00:08<00:39, 8320.83it/s] 18%|█▊        | 71141/400000 [00:08<00:38, 8438.08it/s] 18%|█▊        | 72024/400000 [00:08<00:38, 8550.97it/s] 18%|█▊        | 72907/400000 [00:08<00:37, 8630.43it/s] 18%|█▊        | 73791/400000 [00:08<00:37, 8690.45it/s] 19%|█▊        | 74661/400000 [00:08<00:37, 8662.09it/s] 19%|█▉        | 75528/400000 [00:08<00:38, 8508.35it/s] 19%|█▉        | 76380/400000 [00:09<00:38, 8326.92it/s] 19%|█▉        | 77228/400000 [00:09<00:38, 8372.16it/s] 20%|█▉        | 78084/400000 [00:09<00:38, 8426.63it/s] 20%|█▉        | 78928/400000 [00:09<00:38, 8355.93it/s] 20%|█▉        | 79791/400000 [00:09<00:37, 8435.23it/s] 20%|██        | 80671/400000 [00:09<00:37, 8540.63it/s] 20%|██        | 81526/400000 [00:09<00:37, 8408.74it/s] 21%|██        | 82379/400000 [00:09<00:37, 8442.64it/s] 21%|██        | 83225/400000 [00:09<00:37, 8424.31it/s] 21%|██        | 84068/400000 [00:09<00:37, 8390.95it/s] 21%|██        | 84908/400000 [00:10<00:38, 8242.88it/s] 21%|██▏       | 85757/400000 [00:10<00:37, 8313.63it/s] 22%|██▏       | 86590/400000 [00:10<00:38, 8226.55it/s] 22%|██▏       | 87414/400000 [00:10<00:38, 8176.18it/s] 22%|██▏       | 88255/400000 [00:10<00:37, 8244.51it/s] 22%|██▏       | 89105/400000 [00:10<00:37, 8318.12it/s] 22%|██▏       | 89938/400000 [00:10<00:37, 8210.35it/s] 23%|██▎       | 90760/400000 [00:10<00:38, 8108.31it/s] 23%|██▎       | 91605/400000 [00:10<00:37, 8205.57it/s] 23%|██▎       | 92444/400000 [00:10<00:37, 8258.03it/s] 23%|██▎       | 93318/400000 [00:11<00:36, 8396.07it/s] 24%|██▎       | 94184/400000 [00:11<00:36, 8467.01it/s] 24%|██▍       | 95053/400000 [00:11<00:35, 8530.48it/s] 24%|██▍       | 95908/400000 [00:11<00:35, 8534.32it/s] 24%|██▍       | 96762/400000 [00:11<00:35, 8445.97it/s] 24%|██▍       | 97626/400000 [00:11<00:35, 8502.07it/s] 25%|██▍       | 98486/400000 [00:11<00:35, 8530.10it/s] 25%|██▍       | 99342/400000 [00:11<00:35, 8536.86it/s] 25%|██▌       | 100196/400000 [00:11<00:35, 8526.68it/s] 25%|██▌       | 101071/400000 [00:11<00:34, 8591.69it/s] 25%|██▌       | 101952/400000 [00:12<00:34, 8654.61it/s] 26%|██▌       | 102818/400000 [00:12<00:34, 8592.00it/s] 26%|██▌       | 103678/400000 [00:12<00:34, 8473.57it/s] 26%|██▌       | 104526/400000 [00:12<00:35, 8402.66it/s] 26%|██▋       | 105404/400000 [00:12<00:34, 8510.36it/s] 27%|██▋       | 106275/400000 [00:12<00:34, 8568.83it/s] 27%|██▋       | 107143/400000 [00:12<00:34, 8601.35it/s] 27%|██▋       | 108004/400000 [00:12<00:34, 8584.46it/s] 27%|██▋       | 108863/400000 [00:12<00:34, 8509.72it/s] 27%|██▋       | 109715/400000 [00:12<00:34, 8311.49it/s] 28%|██▊       | 110578/400000 [00:13<00:34, 8402.46it/s] 28%|██▊       | 111440/400000 [00:13<00:34, 8465.88it/s] 28%|██▊       | 112288/400000 [00:13<00:34, 8346.11it/s] 28%|██▊       | 113137/400000 [00:13<00:34, 8387.40it/s] 29%|██▊       | 114005/400000 [00:13<00:33, 8470.56it/s] 29%|██▊       | 114881/400000 [00:13<00:33, 8553.71it/s] 29%|██▉       | 115738/400000 [00:13<00:33, 8534.58it/s] 29%|██▉       | 116597/400000 [00:13<00:33, 8551.00it/s] 29%|██▉       | 117453/400000 [00:13<00:33, 8448.65it/s] 30%|██▉       | 118299/400000 [00:14<00:33, 8365.56it/s] 30%|██▉       | 119161/400000 [00:14<00:33, 8438.62it/s] 30%|███       | 120032/400000 [00:14<00:32, 8515.62it/s] 30%|███       | 120885/400000 [00:14<00:32, 8481.81it/s] 30%|███       | 121734/400000 [00:14<00:33, 8408.29it/s] 31%|███       | 122606/400000 [00:14<00:32, 8495.07it/s] 31%|███       | 123465/400000 [00:14<00:32, 8521.29it/s] 31%|███       | 124327/400000 [00:14<00:32, 8548.30it/s] 31%|███▏      | 125183/400000 [00:14<00:32, 8502.93it/s] 32%|███▏      | 126034/400000 [00:14<00:32, 8481.95it/s] 32%|███▏      | 126899/400000 [00:15<00:32, 8529.24it/s] 32%|███▏      | 127753/400000 [00:15<00:32, 8418.82it/s] 32%|███▏      | 128596/400000 [00:15<00:32, 8301.91it/s] 32%|███▏      | 129461/400000 [00:15<00:32, 8403.08it/s] 33%|███▎      | 130303/400000 [00:15<00:32, 8404.43it/s] 33%|███▎      | 131155/400000 [00:15<00:31, 8436.21it/s] 33%|███▎      | 132000/400000 [00:15<00:33, 7985.74it/s] 33%|███▎      | 132842/400000 [00:15<00:32, 8108.71it/s] 33%|███▎      | 133679/400000 [00:15<00:32, 8183.23it/s] 34%|███▎      | 134523/400000 [00:15<00:32, 8257.74it/s] 34%|███▍      | 135377/400000 [00:16<00:31, 8339.44it/s] 34%|███▍      | 136240/400000 [00:16<00:31, 8422.77it/s] 34%|███▍      | 137130/400000 [00:16<00:30, 8560.37it/s] 35%|███▍      | 138016/400000 [00:16<00:30, 8646.79it/s] 35%|███▍      | 138882/400000 [00:16<00:30, 8618.29it/s] 35%|███▍      | 139767/400000 [00:16<00:29, 8683.71it/s] 35%|███▌      | 140638/400000 [00:16<00:29, 8690.27it/s] 35%|███▌      | 141518/400000 [00:16<00:29, 8721.44it/s] 36%|███▌      | 142397/400000 [00:16<00:29, 8739.61it/s] 36%|███▌      | 143272/400000 [00:16<00:29, 8663.91it/s] 36%|███▌      | 144156/400000 [00:17<00:29, 8714.15it/s] 36%|███▋      | 145028/400000 [00:17<00:29, 8677.09it/s] 36%|███▋      | 145896/400000 [00:17<00:29, 8586.95it/s] 37%|███▋      | 146756/400000 [00:17<00:29, 8505.66it/s] 37%|███▋      | 147608/400000 [00:17<00:30, 8282.57it/s] 37%|███▋      | 148438/400000 [00:17<00:30, 8259.05it/s] 37%|███▋      | 149318/400000 [00:17<00:29, 8412.81it/s] 38%|███▊      | 150201/400000 [00:17<00:29, 8533.16it/s] 38%|███▊      | 151079/400000 [00:17<00:28, 8603.75it/s] 38%|███▊      | 151941/400000 [00:17<00:29, 8539.41it/s] 38%|███▊      | 152824/400000 [00:18<00:28, 8622.69it/s] 38%|███▊      | 153688/400000 [00:18<00:28, 8567.25it/s] 39%|███▊      | 154554/400000 [00:18<00:28, 8594.34it/s] 39%|███▉      | 155417/400000 [00:18<00:28, 8603.87it/s] 39%|███▉      | 156278/400000 [00:18<00:28, 8468.32it/s] 39%|███▉      | 157130/400000 [00:18<00:28, 8482.77it/s] 39%|███▉      | 157990/400000 [00:18<00:28, 8517.57it/s] 40%|███▉      | 158871/400000 [00:18<00:28, 8600.75it/s] 40%|███▉      | 159732/400000 [00:18<00:28, 8516.77it/s] 40%|████      | 160585/400000 [00:18<00:28, 8430.84it/s] 40%|████      | 161429/400000 [00:19<00:28, 8272.67it/s] 41%|████      | 162295/400000 [00:19<00:28, 8383.01it/s] 41%|████      | 163172/400000 [00:19<00:27, 8495.46it/s] 41%|████      | 164028/400000 [00:19<00:27, 8512.67it/s] 41%|████      | 164881/400000 [00:19<00:27, 8511.17it/s] 41%|████▏     | 165761/400000 [00:19<00:27, 8593.64it/s] 42%|████▏     | 166621/400000 [00:19<00:27, 8351.37it/s] 42%|████▏     | 167473/400000 [00:19<00:27, 8400.29it/s] 42%|████▏     | 168323/400000 [00:19<00:27, 8429.78it/s] 42%|████▏     | 169180/400000 [00:19<00:27, 8469.11it/s] 43%|████▎     | 170060/400000 [00:20<00:26, 8563.38it/s] 43%|████▎     | 170918/400000 [00:20<00:26, 8521.01it/s] 43%|████▎     | 171771/400000 [00:20<00:27, 8368.49it/s] 43%|████▎     | 172609/400000 [00:20<00:27, 8233.72it/s] 43%|████▎     | 173434/400000 [00:20<00:27, 8179.49it/s] 44%|████▎     | 174253/400000 [00:20<00:27, 8154.56it/s] 44%|████▍     | 175101/400000 [00:20<00:27, 8247.20it/s] 44%|████▍     | 175927/400000 [00:20<00:27, 8018.07it/s] 44%|████▍     | 176769/400000 [00:20<00:27, 8132.61it/s] 44%|████▍     | 177591/400000 [00:21<00:27, 8156.87it/s] 45%|████▍     | 178425/400000 [00:21<00:26, 8209.69it/s] 45%|████▍     | 179266/400000 [00:21<00:26, 8266.47it/s] 45%|████▌     | 180135/400000 [00:21<00:26, 8387.07it/s] 45%|████▌     | 181015/400000 [00:21<00:25, 8506.78it/s] 45%|████▌     | 181867/400000 [00:21<00:25, 8477.09it/s] 46%|████▌     | 182716/400000 [00:21<00:25, 8413.56it/s] 46%|████▌     | 183559/400000 [00:21<00:25, 8384.63it/s] 46%|████▌     | 184424/400000 [00:21<00:25, 8460.98it/s] 46%|████▋     | 185271/400000 [00:21<00:25, 8329.60it/s] 47%|████▋     | 186132/400000 [00:22<00:25, 8409.55it/s] 47%|████▋     | 186974/400000 [00:22<00:25, 8345.92it/s] 47%|████▋     | 187810/400000 [00:22<00:26, 8050.06it/s] 47%|████▋     | 188687/400000 [00:22<00:25, 8252.31it/s] 47%|████▋     | 189571/400000 [00:22<00:25, 8417.15it/s] 48%|████▊     | 190416/400000 [00:22<00:25, 8373.23it/s] 48%|████▊     | 191274/400000 [00:22<00:24, 8433.99it/s] 48%|████▊     | 192157/400000 [00:22<00:24, 8548.39it/s] 48%|████▊     | 193034/400000 [00:22<00:24, 8610.86it/s] 48%|████▊     | 193906/400000 [00:22<00:23, 8641.37it/s] 49%|████▊     | 194771/400000 [00:23<00:23, 8608.45it/s] 49%|████▉     | 195639/400000 [00:23<00:23, 8628.31it/s] 49%|████▉     | 196503/400000 [00:23<00:24, 8407.54it/s] 49%|████▉     | 197367/400000 [00:23<00:23, 8474.22it/s] 50%|████▉     | 198237/400000 [00:23<00:23, 8537.18it/s] 50%|████▉     | 199092/400000 [00:23<00:23, 8524.94it/s] 50%|████▉     | 199946/400000 [00:23<00:23, 8461.58it/s] 50%|█████     | 200803/400000 [00:23<00:23, 8491.58it/s] 50%|█████     | 201653/400000 [00:23<00:23, 8450.97it/s] 51%|█████     | 202499/400000 [00:23<00:23, 8392.03it/s] 51%|█████     | 203340/400000 [00:24<00:23, 8396.79it/s] 51%|█████     | 204212/400000 [00:24<00:23, 8489.74it/s] 51%|█████▏    | 205062/400000 [00:24<00:23, 8349.78it/s] 51%|█████▏    | 205937/400000 [00:24<00:22, 8463.48it/s] 52%|█████▏    | 206820/400000 [00:24<00:22, 8568.44it/s] 52%|█████▏    | 207678/400000 [00:24<00:22, 8523.16it/s] 52%|█████▏    | 208532/400000 [00:24<00:22, 8514.11it/s] 52%|█████▏    | 209398/400000 [00:24<00:22, 8554.72it/s] 53%|█████▎    | 210254/400000 [00:24<00:22, 8425.51it/s] 53%|█████▎    | 211120/400000 [00:24<00:22, 8493.64it/s] 53%|█████▎    | 211971/400000 [00:25<00:22, 8413.08it/s] 53%|█████▎    | 212841/400000 [00:25<00:22, 8494.73it/s] 53%|█████▎    | 213713/400000 [00:25<00:21, 8559.07it/s] 54%|█████▎    | 214570/400000 [00:25<00:22, 8398.16it/s] 54%|█████▍    | 215411/400000 [00:25<00:22, 8033.64it/s] 54%|█████▍    | 216219/400000 [00:25<00:23, 7951.06it/s] 54%|█████▍    | 217088/400000 [00:25<00:22, 8156.90it/s] 54%|█████▍    | 217971/400000 [00:25<00:21, 8343.89it/s] 55%|█████▍    | 218809/400000 [00:25<00:21, 8354.07it/s] 55%|█████▍    | 219647/400000 [00:26<00:21, 8286.08it/s] 55%|█████▌    | 220478/400000 [00:26<00:21, 8218.64it/s] 55%|█████▌    | 221311/400000 [00:26<00:21, 8249.48it/s] 56%|█████▌    | 222137/400000 [00:26<00:22, 8054.44it/s] 56%|█████▌    | 222945/400000 [00:26<00:22, 8044.20it/s] 56%|█████▌    | 223751/400000 [00:26<00:21, 8011.57it/s] 56%|█████▌    | 224554/400000 [00:26<00:21, 7982.47it/s] 56%|█████▋    | 225416/400000 [00:26<00:21, 8163.32it/s] 57%|█████▋    | 226278/400000 [00:26<00:20, 8294.58it/s] 57%|█████▋    | 227122/400000 [00:26<00:20, 8336.42it/s] 57%|█████▋    | 227957/400000 [00:27<00:20, 8325.83it/s] 57%|█████▋    | 228791/400000 [00:27<00:21, 8030.65it/s] 57%|█████▋    | 229597/400000 [00:27<00:21, 7865.36it/s] 58%|█████▊    | 230461/400000 [00:27<00:20, 8082.16it/s] 58%|█████▊    | 231334/400000 [00:27<00:20, 8263.65it/s] 58%|█████▊    | 232178/400000 [00:27<00:20, 8315.23it/s] 58%|█████▊    | 233039/400000 [00:27<00:19, 8398.47it/s] 58%|█████▊    | 233898/400000 [00:27<00:19, 8454.58it/s] 59%|█████▊    | 234745/400000 [00:27<00:19, 8430.73it/s] 59%|█████▉    | 235590/400000 [00:27<00:19, 8436.30it/s] 59%|█████▉    | 236454/400000 [00:28<00:19, 8493.57it/s] 59%|█████▉    | 237304/400000 [00:28<00:19, 8495.18it/s] 60%|█████▉    | 238163/400000 [00:28<00:18, 8520.65it/s] 60%|█████▉    | 239016/400000 [00:28<00:19, 8234.88it/s] 60%|█████▉    | 239845/400000 [00:28<00:19, 8250.56it/s] 60%|██████    | 240703/400000 [00:28<00:19, 8345.86it/s] 60%|██████    | 241562/400000 [00:28<00:18, 8415.72it/s] 61%|██████    | 242418/400000 [00:28<00:18, 8458.20it/s] 61%|██████    | 243265/400000 [00:28<00:18, 8308.39it/s] 61%|██████    | 244098/400000 [00:28<00:19, 8084.23it/s] 61%|██████    | 244947/400000 [00:29<00:18, 8199.74it/s] 61%|██████▏   | 245818/400000 [00:29<00:18, 8343.84it/s] 62%|██████▏   | 246694/400000 [00:29<00:18, 8463.81it/s] 62%|██████▏   | 247543/400000 [00:29<00:18, 8469.77it/s] 62%|██████▏   | 248420/400000 [00:29<00:17, 8554.85it/s] 62%|██████▏   | 249277/400000 [00:29<00:18, 8271.16it/s] 63%|██████▎   | 250107/400000 [00:29<00:18, 8210.87it/s] 63%|██████▎   | 250937/400000 [00:29<00:18, 8235.52it/s] 63%|██████▎   | 251771/400000 [00:29<00:17, 8264.41it/s] 63%|██████▎   | 252647/400000 [00:30<00:17, 8406.30it/s] 63%|██████▎   | 253506/400000 [00:30<00:17, 8460.14it/s] 64%|██████▎   | 254354/400000 [00:30<00:17, 8346.54it/s] 64%|██████▍   | 255238/400000 [00:30<00:17, 8486.69it/s] 64%|██████▍   | 256088/400000 [00:30<00:16, 8487.60it/s] 64%|██████▍   | 256938/400000 [00:30<00:16, 8430.90it/s] 64%|██████▍   | 257782/400000 [00:30<00:17, 8304.52it/s] 65%|██████▍   | 258614/400000 [00:30<00:17, 8055.97it/s] 65%|██████▍   | 259452/400000 [00:30<00:17, 8149.33it/s] 65%|██████▌   | 260319/400000 [00:30<00:16, 8298.45it/s] 65%|██████▌   | 261178/400000 [00:31<00:16, 8383.17it/s] 66%|██████▌   | 262029/400000 [00:31<00:16, 8419.30it/s] 66%|██████▌   | 262895/400000 [00:31<00:16, 8489.44it/s] 66%|██████▌   | 263745/400000 [00:31<00:16, 8437.80it/s] 66%|██████▌   | 264590/400000 [00:31<00:16, 8246.25it/s] 66%|██████▋   | 265435/400000 [00:31<00:16, 8304.36it/s] 67%|██████▋   | 266267/400000 [00:31<00:16, 8278.77it/s] 67%|██████▋   | 267096/400000 [00:31<00:16, 8245.55it/s] 67%|██████▋   | 267971/400000 [00:31<00:15, 8389.85it/s] 67%|██████▋   | 268812/400000 [00:31<00:15, 8289.73it/s] 67%|██████▋   | 269642/400000 [00:32<00:15, 8216.69it/s] 68%|██████▊   | 270465/400000 [00:32<00:16, 7874.93it/s] 68%|██████▊   | 271257/400000 [00:32<00:16, 7660.69it/s] 68%|██████▊   | 272108/400000 [00:32<00:16, 7895.09it/s] 68%|██████▊   | 272976/400000 [00:32<00:15, 8112.91it/s] 68%|██████▊   | 273792/400000 [00:32<00:15, 8043.74it/s] 69%|██████▊   | 274623/400000 [00:32<00:15, 8119.61it/s] 69%|██████▉   | 275512/400000 [00:32<00:14, 8334.57it/s] 69%|██████▉   | 276403/400000 [00:32<00:14, 8497.13it/s] 69%|██████▉   | 277256/400000 [00:32<00:14, 8424.95it/s] 70%|██████▉   | 278126/400000 [00:33<00:14, 8503.07it/s] 70%|██████▉   | 278979/400000 [00:33<00:14, 8320.47it/s] 70%|██████▉   | 279842/400000 [00:33<00:14, 8410.84it/s] 70%|███████   | 280685/400000 [00:33<00:14, 8281.42it/s] 70%|███████   | 281532/400000 [00:33<00:14, 8336.53it/s] 71%|███████   | 282412/400000 [00:33<00:13, 8469.27it/s] 71%|███████   | 283261/400000 [00:33<00:13, 8435.23it/s] 71%|███████   | 284106/400000 [00:33<00:14, 8046.01it/s] 71%|███████   | 284916/400000 [00:33<00:14, 8051.78it/s] 71%|███████▏  | 285799/400000 [00:34<00:13, 8269.61it/s] 72%|███████▏  | 286630/400000 [00:34<00:13, 8266.42it/s] 72%|███████▏  | 287463/400000 [00:34<00:13, 8282.56it/s] 72%|███████▏  | 288332/400000 [00:34<00:13, 8398.18it/s] 72%|███████▏  | 289181/400000 [00:34<00:13, 8425.34it/s] 73%|███████▎  | 290065/400000 [00:34<00:12, 8545.06it/s] 73%|███████▎  | 290923/400000 [00:34<00:12, 8555.43it/s] 73%|███████▎  | 291780/400000 [00:34<00:12, 8527.32it/s] 73%|███████▎  | 292653/400000 [00:34<00:12, 8586.06it/s] 73%|███████▎  | 293513/400000 [00:34<00:12, 8505.58it/s] 74%|███████▎  | 294372/400000 [00:35<00:12, 8529.16it/s] 74%|███████▍  | 295253/400000 [00:35<00:12, 8609.15it/s] 74%|███████▍  | 296117/400000 [00:35<00:12, 8615.71it/s] 74%|███████▍  | 296979/400000 [00:35<00:12, 8549.55it/s] 74%|███████▍  | 297835/400000 [00:35<00:12, 8340.00it/s] 75%|███████▍  | 298687/400000 [00:35<00:12, 8390.49it/s] 75%|███████▍  | 299571/400000 [00:35<00:11, 8518.29it/s] 75%|███████▌  | 300425/400000 [00:35<00:11, 8333.14it/s] 75%|███████▌  | 301261/400000 [00:35<00:12, 8112.11it/s] 76%|███████▌  | 302090/400000 [00:35<00:11, 8163.54it/s] 76%|███████▌  | 302969/400000 [00:36<00:11, 8341.46it/s] 76%|███████▌  | 303833/400000 [00:36<00:11, 8427.50it/s] 76%|███████▌  | 304688/400000 [00:36<00:11, 8462.78it/s] 76%|███████▋  | 305552/400000 [00:36<00:11, 8513.67it/s] 77%|███████▋  | 306425/400000 [00:36<00:10, 8576.68it/s] 77%|███████▋  | 307304/400000 [00:36<00:10, 8637.48it/s] 77%|███████▋  | 308169/400000 [00:36<00:10, 8624.09it/s] 77%|███████▋  | 309032/400000 [00:36<00:10, 8476.72it/s] 77%|███████▋  | 309881/400000 [00:36<00:10, 8472.97it/s] 78%|███████▊  | 310729/400000 [00:36<00:10, 8435.89it/s] 78%|███████▊  | 311580/400000 [00:37<00:10, 8456.73it/s] 78%|███████▊  | 312450/400000 [00:37<00:10, 8528.00it/s] 78%|███████▊  | 313312/400000 [00:37<00:10, 8553.97it/s] 79%|███████▊  | 314191/400000 [00:37<00:09, 8623.16it/s] 79%|███████▉  | 315070/400000 [00:37<00:09, 8671.23it/s] 79%|███████▉  | 315938/400000 [00:37<00:09, 8408.22it/s] 79%|███████▉  | 316781/400000 [00:37<00:10, 8320.74it/s] 79%|███████▉  | 317615/400000 [00:37<00:09, 8291.98it/s] 80%|███████▉  | 318454/400000 [00:37<00:09, 8319.57it/s] 80%|███████▉  | 319321/400000 [00:37<00:09, 8419.75it/s] 80%|████████  | 320203/400000 [00:38<00:09, 8533.92it/s] 80%|████████  | 321082/400000 [00:38<00:09, 8607.53it/s] 80%|████████  | 321944/400000 [00:38<00:09, 8433.07it/s] 81%|████████  | 322820/400000 [00:38<00:09, 8527.36it/s] 81%|████████  | 323704/400000 [00:38<00:08, 8618.46it/s] 81%|████████  | 324589/400000 [00:38<00:08, 8686.33it/s] 81%|████████▏ | 325476/400000 [00:38<00:08, 8739.72it/s] 82%|████████▏ | 326351/400000 [00:38<00:08, 8629.93it/s] 82%|████████▏ | 327217/400000 [00:38<00:08, 8638.85it/s] 82%|████████▏ | 328094/400000 [00:38<00:08, 8677.63it/s] 82%|████████▏ | 328979/400000 [00:39<00:08, 8726.78it/s] 82%|████████▏ | 329853/400000 [00:39<00:08, 8520.90it/s] 83%|████████▎ | 330707/400000 [00:39<00:08, 8274.40it/s] 83%|████████▎ | 331578/400000 [00:39<00:08, 8400.37it/s] 83%|████████▎ | 332465/400000 [00:39<00:07, 8535.75it/s] 83%|████████▎ | 333336/400000 [00:39<00:07, 8586.97it/s] 84%|████████▎ | 334197/400000 [00:39<00:07, 8505.97it/s] 84%|████████▍ | 335056/400000 [00:39<00:07, 8529.60it/s] 84%|████████▍ | 335925/400000 [00:39<00:07, 8576.64it/s] 84%|████████▍ | 336808/400000 [00:39<00:07, 8650.45it/s] 84%|████████▍ | 337688/400000 [00:40<00:07, 8691.99it/s] 85%|████████▍ | 338568/400000 [00:40<00:07, 8721.44it/s] 85%|████████▍ | 339441/400000 [00:40<00:06, 8687.97it/s] 85%|████████▌ | 340311/400000 [00:40<00:06, 8656.23it/s] 85%|████████▌ | 341200/400000 [00:40<00:06, 8723.46it/s] 86%|████████▌ | 342085/400000 [00:40<00:06, 8759.24it/s] 86%|████████▌ | 342965/400000 [00:40<00:06, 8769.92it/s] 86%|████████▌ | 343843/400000 [00:40<00:06, 8655.08it/s] 86%|████████▌ | 344709/400000 [00:40<00:06, 8301.60it/s] 86%|████████▋ | 345543/400000 [00:41<00:06, 8234.04it/s] 87%|████████▋ | 346417/400000 [00:41<00:06, 8379.25it/s] 87%|████████▋ | 347288/400000 [00:41<00:06, 8474.07it/s] 87%|████████▋ | 348142/400000 [00:41<00:06, 8490.38it/s] 87%|████████▋ | 349025/400000 [00:41<00:05, 8589.13it/s] 87%|████████▋ | 349898/400000 [00:41<00:05, 8627.97it/s] 88%|████████▊ | 350772/400000 [00:41<00:05, 8660.25it/s] 88%|████████▊ | 351646/400000 [00:41<00:05, 8683.75it/s] 88%|████████▊ | 352515/400000 [00:41<00:05, 8534.49it/s] 88%|████████▊ | 353370/400000 [00:41<00:05, 8480.52it/s] 89%|████████▊ | 354223/400000 [00:42<00:05, 8492.99it/s] 89%|████████▉ | 355073/400000 [00:42<00:05, 8303.60it/s] 89%|████████▉ | 355905/400000 [00:42<00:05, 8268.89it/s] 89%|████████▉ | 356733/400000 [00:42<00:05, 8263.12it/s] 89%|████████▉ | 357592/400000 [00:42<00:05, 8356.21it/s] 90%|████████▉ | 358429/400000 [00:42<00:05, 8292.30it/s] 90%|████████▉ | 359307/400000 [00:42<00:04, 8431.95it/s] 90%|█████████ | 360184/400000 [00:42<00:04, 8529.81it/s] 90%|█████████ | 361038/400000 [00:42<00:04, 8525.50it/s] 90%|█████████ | 361894/400000 [00:42<00:04, 8534.50it/s] 91%|█████████ | 362749/400000 [00:43<00:04, 8537.35it/s] 91%|█████████ | 363612/400000 [00:43<00:04, 8562.07it/s] 91%|█████████ | 364492/400000 [00:43<00:04, 8631.29it/s] 91%|█████████▏| 365356/400000 [00:43<00:04, 8344.90it/s] 92%|█████████▏| 366193/400000 [00:43<00:04, 8350.38it/s] 92%|█████████▏| 367037/400000 [00:43<00:03, 8373.87it/s] 92%|█████████▏| 367918/400000 [00:43<00:03, 8498.53it/s] 92%|█████████▏| 368770/400000 [00:43<00:03, 8497.78it/s] 92%|█████████▏| 369623/400000 [00:43<00:03, 8505.63it/s] 93%|█████████▎| 370475/400000 [00:43<00:03, 8416.36it/s] 93%|█████████▎| 371329/400000 [00:44<00:03, 8451.21it/s] 93%|█████████▎| 372175/400000 [00:44<00:03, 8249.20it/s] 93%|█████████▎| 373012/400000 [00:44<00:03, 8284.77it/s] 93%|█████████▎| 373868/400000 [00:44<00:03, 8363.35it/s] 94%|█████████▎| 374752/400000 [00:44<00:02, 8498.96it/s] 94%|█████████▍| 375628/400000 [00:44<00:02, 8575.44it/s] 94%|█████████▍| 376499/400000 [00:44<00:02, 8613.98it/s] 94%|█████████▍| 377380/400000 [00:44<00:02, 8669.12it/s] 95%|█████████▍| 378248/400000 [00:44<00:02, 8560.86it/s] 95%|█████████▍| 379130/400000 [00:44<00:02, 8635.28it/s] 95%|█████████▌| 380015/400000 [00:45<00:02, 8697.06it/s] 95%|█████████▌| 380886/400000 [00:45<00:02, 8659.01it/s] 95%|█████████▌| 381764/400000 [00:45<00:02, 8693.82it/s] 96%|█████████▌| 382634/400000 [00:45<00:02, 8610.09it/s] 96%|█████████▌| 383499/400000 [00:45<00:01, 8620.62it/s] 96%|█████████▌| 384380/400000 [00:45<00:01, 8674.73it/s] 96%|█████████▋| 385261/400000 [00:45<00:01, 8712.76it/s] 97%|█████████▋| 386133/400000 [00:45<00:01, 8566.37it/s] 97%|█████████▋| 386991/400000 [00:45<00:01, 8348.35it/s] 97%|█████████▋| 387863/400000 [00:45<00:01, 8456.20it/s] 97%|█████████▋| 388711/400000 [00:46<00:01, 8356.06it/s] 97%|█████████▋| 389548/400000 [00:46<00:01, 8252.48it/s] 98%|█████████▊| 390433/400000 [00:46<00:01, 8422.20it/s] 98%|█████████▊| 391284/400000 [00:46<00:01, 8446.98it/s] 98%|█████████▊| 392163/400000 [00:46<00:00, 8545.92it/s] 98%|█████████▊| 393025/400000 [00:46<00:00, 8565.66it/s] 98%|█████████▊| 393911/400000 [00:46<00:00, 8649.88it/s] 99%|█████████▊| 394795/400000 [00:46<00:00, 8703.52it/s] 99%|█████████▉| 395666/400000 [00:46<00:00, 8626.64it/s] 99%|█████████▉| 396551/400000 [00:47<00:00, 8687.27it/s] 99%|█████████▉| 397431/400000 [00:47<00:00, 8718.39it/s]100%|█████████▉| 398311/400000 [00:47<00:00, 8740.25it/s]100%|█████████▉| 399186/400000 [00:47<00:00, 8695.48it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8437.07it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fcc6e885be0>, <torchtext.data.dataset.TabularDataset object at 0x7fcc6e885c18>, <torchtext.vocab.Vocab object at 0x7fcc6e889048>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.52 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.52 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.52 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.59 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.59 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.66 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.66 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.67 file/s]2020-06-17 06:18:56.401917: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-17 06:18:56.405651: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-06-17 06:18:56.405910: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e1944c4200 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-17 06:18:56.405931: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist_dataset_small.npy', 'test', 'fashion-mnist_small.npy', 'mnist2'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 46%|████▌     | 4546560/9912422 [00:00<00:00, 45452422.57it/s]9920512it [00:00, 33583707.90it/s]                             
0it [00:00, ?it/s]32768it [00:00, 601504.44it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 479914.78it/s]1654784it [00:00, 11989136.80it/s]                         
0it [00:00, ?it/s]8192it [00:00, 218677.73it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

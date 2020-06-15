
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/c7c470df7a126619c38e876f1b772eb299ee927c', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'c7c470df7a126619c38e876f1b772eb299ee927c', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/c7c470df7a126619c38e876f1b772eb299ee927c

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/c7c470df7a126619c38e876f1b772eb299ee927c

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/c7c470df7a126619c38e876f1b772eb299ee927c

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f37aed48ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f37aed48ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f381a3000d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f381a3000d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f383402ce18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f383402ce18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f37c7b7d950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f37c7b7d950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f37c7b7d950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 32%|███▏      | 3153920/9912422 [00:00<00:00, 31205325.73it/s]9920512it [00:00, 35759312.38it/s]                             
0it [00:00, ?it/s]32768it [00:00, 503882.36it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 475172.71it/s]1654784it [00:00, 11666555.42it/s]                         
0it [00:00, ?it/s]8192it [00:00, 212501.16it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f37c51c8748>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f37c51c1470>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f37c7b7d598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f37c7b7d598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f37c7b7d598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:21,  1.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:21,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:21,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:20,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:20,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:19,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:19,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:18,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:55,  2.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:55,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:55,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:54,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:54,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:54,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:53,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:53,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:52,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:52,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:52,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:36,  3.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:36,  3.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:36,  3.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:35,  3.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:35,  3.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:35,  3.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:35,  3.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:34,  3.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:34,  3.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:24,  5.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:24,  5.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:23,  5.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:23,  5.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:23,  5.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:22,  5.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:22,  5.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:15,  7.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:15,  7.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:15,  7.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:15,  7.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:10, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:10, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 14.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 19.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 19.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 19.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 32.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 32.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 32.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 32.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 32.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 32.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 32.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 32.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 32.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 32.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 39.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 39.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 39.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 39.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 39.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 39.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 39.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 39.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 39.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 39.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 39.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 47.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 47.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 47.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 47.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 47.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 47.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 47.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 47.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 47.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 47.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 54.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 54.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 54.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 54.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 54.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 54.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 54.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 54.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 54.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 54.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 54.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 62.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 62.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 62.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 62.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 62.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 62.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 62.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 62.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 62.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 62.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 62.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 69.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 69.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 69.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 74.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 74.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 74.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 74.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.31s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.31s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.31s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.20 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.60s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.31s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 78.20 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.60s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.60s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.23 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.60s/ url]
0 examples [00:00, ? examples/s]2020-06-15 09:29:39.254059: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-15 09:29:39.271158: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-06-15 09:29:39.271346: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563f403eb400 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-15 09:29:39.271364: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
23 examples [00:00, 228.49 examples/s]123 examples [00:00, 297.27 examples/s]229 examples [00:00, 379.04 examples/s]332 examples [00:00, 467.25 examples/s]423 examples [00:00, 547.03 examples/s]518 examples [00:00, 625.69 examples/s]610 examples [00:00, 690.46 examples/s]711 examples [00:00, 762.64 examples/s]807 examples [00:00, 812.58 examples/s]899 examples [00:01, 832.77 examples/s]993 examples [00:01, 860.65 examples/s]1085 examples [00:01, 841.67 examples/s]1182 examples [00:01, 876.25 examples/s]1281 examples [00:01, 906.37 examples/s]1378 examples [00:01, 923.24 examples/s]1474 examples [00:01, 931.44 examples/s]1572 examples [00:01, 943.52 examples/s]1668 examples [00:01, 933.00 examples/s]1762 examples [00:01, 931.52 examples/s]1856 examples [00:02, 908.79 examples/s]1948 examples [00:02, 899.90 examples/s]2047 examples [00:02, 924.13 examples/s]2154 examples [00:02, 962.54 examples/s]2252 examples [00:02, 966.43 examples/s]2353 examples [00:02, 976.32 examples/s]2451 examples [00:02, 940.47 examples/s]2546 examples [00:02, 941.83 examples/s]2655 examples [00:02, 980.58 examples/s]2754 examples [00:02, 974.65 examples/s]2852 examples [00:03, 931.46 examples/s]2951 examples [00:03, 946.45 examples/s]3054 examples [00:03, 967.58 examples/s]3153 examples [00:03, 973.69 examples/s]3257 examples [00:03, 991.84 examples/s]3362 examples [00:03, 1008.57 examples/s]3464 examples [00:03, 989.42 examples/s] 3566 examples [00:03, 997.03 examples/s]3666 examples [00:03, 944.59 examples/s]3769 examples [00:04, 967.05 examples/s]3873 examples [00:04, 985.59 examples/s]3973 examples [00:04, 970.79 examples/s]4078 examples [00:04, 991.28 examples/s]4183 examples [00:04, 1007.25 examples/s]4290 examples [00:04, 1023.30 examples/s]4393 examples [00:04, 1007.36 examples/s]4496 examples [00:04, 1012.19 examples/s]4598 examples [00:04, 993.95 examples/s] 4698 examples [00:04, 973.57 examples/s]4796 examples [00:05, 946.89 examples/s]4892 examples [00:05, 899.28 examples/s]4989 examples [00:05, 917.62 examples/s]5090 examples [00:05, 943.15 examples/s]5196 examples [00:05, 974.40 examples/s]5297 examples [00:05, 982.56 examples/s]5396 examples [00:05, 954.65 examples/s]5492 examples [00:05, 916.94 examples/s]5585 examples [00:05, 887.45 examples/s]5683 examples [00:06, 911.25 examples/s]5775 examples [00:06, 910.81 examples/s]5870 examples [00:06, 921.23 examples/s]5968 examples [00:06, 937.28 examples/s]6075 examples [00:06, 973.09 examples/s]6173 examples [00:06, 937.95 examples/s]6268 examples [00:06, 937.79 examples/s]6363 examples [00:06, 929.50 examples/s]6457 examples [00:06, 915.94 examples/s]6549 examples [00:06, 850.70 examples/s]6648 examples [00:07, 887.96 examples/s]6754 examples [00:07, 932.15 examples/s]6855 examples [00:07, 952.10 examples/s]6952 examples [00:07, 944.78 examples/s]7055 examples [00:07, 968.19 examples/s]7164 examples [00:07, 1000.35 examples/s]7269 examples [00:07, 1013.98 examples/s]7377 examples [00:07, 1031.15 examples/s]7481 examples [00:07, 1002.12 examples/s]7582 examples [00:07, 966.46 examples/s] 7685 examples [00:08, 983.13 examples/s]7784 examples [00:08, 942.78 examples/s]7881 examples [00:08, 948.50 examples/s]7984 examples [00:08, 970.62 examples/s]8089 examples [00:08, 992.80 examples/s]8195 examples [00:08, 1010.14 examples/s]8297 examples [00:08, 989.58 examples/s] 8397 examples [00:08, 981.03 examples/s]8500 examples [00:08, 995.19 examples/s]8600 examples [00:09, 993.94 examples/s]8703 examples [00:09, 1002.91 examples/s]8804 examples [00:09, 991.13 examples/s] 8904 examples [00:09, 953.24 examples/s]9001 examples [00:09, 957.91 examples/s]9107 examples [00:09, 984.16 examples/s]9208 examples [00:09, 986.94 examples/s]9307 examples [00:09, 969.52 examples/s]9405 examples [00:09, 948.38 examples/s]9510 examples [00:09, 974.45 examples/s]9608 examples [00:10, 939.43 examples/s]9710 examples [00:10, 961.86 examples/s]9810 examples [00:10, 972.06 examples/s]9908 examples [00:10, 925.26 examples/s]10002 examples [00:10, 868.70 examples/s]10102 examples [00:10, 903.44 examples/s]10207 examples [00:10, 942.30 examples/s]10303 examples [00:10, 918.24 examples/s]10396 examples [00:10, 910.38 examples/s]10498 examples [00:11, 940.02 examples/s]10593 examples [00:11, 920.59 examples/s]10686 examples [00:11, 911.11 examples/s]10778 examples [00:11, 912.87 examples/s]10877 examples [00:11, 932.59 examples/s]10983 examples [00:11, 967.34 examples/s]11081 examples [00:11, 968.22 examples/s]11182 examples [00:11, 979.92 examples/s]11284 examples [00:11, 990.49 examples/s]11387 examples [00:11, 1000.89 examples/s]11488 examples [00:12, 979.99 examples/s] 11593 examples [00:12, 998.72 examples/s]11694 examples [00:12, 999.35 examples/s]11795 examples [00:12, 995.74 examples/s]11895 examples [00:12, 966.82 examples/s]11992 examples [00:12, 954.41 examples/s]12089 examples [00:12, 957.47 examples/s]12189 examples [00:12, 968.42 examples/s]12286 examples [00:12, 942.12 examples/s]12381 examples [00:12, 943.80 examples/s]12476 examples [00:13, 937.01 examples/s]12580 examples [00:13, 963.70 examples/s]12677 examples [00:13, 962.96 examples/s]12775 examples [00:13, 965.44 examples/s]12881 examples [00:13, 989.76 examples/s]12988 examples [00:13, 1011.30 examples/s]13090 examples [00:13, 1013.47 examples/s]13192 examples [00:13, 1013.29 examples/s]13294 examples [00:13, 977.12 examples/s] 13393 examples [00:14, 945.68 examples/s]13489 examples [00:14, 924.75 examples/s]13591 examples [00:14, 949.10 examples/s]13687 examples [00:14, 942.44 examples/s]13782 examples [00:14, 943.35 examples/s]13888 examples [00:14, 975.53 examples/s]13994 examples [00:14, 998.86 examples/s]14095 examples [00:14, 978.57 examples/s]14194 examples [00:14, 978.48 examples/s]14293 examples [00:14, 936.05 examples/s]14388 examples [00:15, 937.51 examples/s]14483 examples [00:15, 935.54 examples/s]14577 examples [00:15, 907.05 examples/s]14670 examples [00:15, 913.42 examples/s]14767 examples [00:15, 929.10 examples/s]14878 examples [00:15, 975.58 examples/s]14988 examples [00:15, 1008.95 examples/s]15090 examples [00:15, 1006.58 examples/s]15192 examples [00:15, 986.01 examples/s] 15292 examples [00:15, 986.82 examples/s]15392 examples [00:16, 965.23 examples/s]15489 examples [00:16, 959.47 examples/s]15586 examples [00:16, 939.43 examples/s]15684 examples [00:16, 949.81 examples/s]15785 examples [00:16, 965.79 examples/s]15889 examples [00:16, 985.22 examples/s]15990 examples [00:16, 991.90 examples/s]16090 examples [00:16, 990.33 examples/s]16194 examples [00:16, 1003.95 examples/s]16299 examples [00:17, 1014.81 examples/s]16405 examples [00:17, 1026.25 examples/s]16508 examples [00:17, 994.60 examples/s] 16608 examples [00:17, 914.24 examples/s]16701 examples [00:17, 891.75 examples/s]16792 examples [00:17, 895.16 examples/s]16896 examples [00:17, 933.34 examples/s]17003 examples [00:17, 969.45 examples/s]17101 examples [00:17, 959.75 examples/s]17198 examples [00:17, 952.81 examples/s]17299 examples [00:18, 966.99 examples/s]17397 examples [00:18, 963.51 examples/s]17494 examples [00:18, 962.19 examples/s]17598 examples [00:18, 982.67 examples/s]17697 examples [00:18, 935.30 examples/s]17805 examples [00:18, 972.12 examples/s]17904 examples [00:18, 959.88 examples/s]18012 examples [00:18, 990.63 examples/s]18120 examples [00:18, 1014.97 examples/s]18223 examples [00:19, 1003.84 examples/s]18324 examples [00:19, 957.44 examples/s] 18421 examples [00:19, 959.51 examples/s]18529 examples [00:19, 990.66 examples/s]18634 examples [00:19, 1007.62 examples/s]18737 examples [00:19, 1013.61 examples/s]18839 examples [00:19, 990.20 examples/s] 18939 examples [00:19, 948.94 examples/s]19041 examples [00:19, 968.61 examples/s]19139 examples [00:19, 966.10 examples/s]19236 examples [00:20, 945.42 examples/s]19331 examples [00:20, 936.33 examples/s]19433 examples [00:20, 957.55 examples/s]19535 examples [00:20, 973.57 examples/s]19633 examples [00:20, 937.93 examples/s]19734 examples [00:20, 958.20 examples/s]19841 examples [00:20, 988.61 examples/s]19948 examples [00:20, 1009.14 examples/s]20050 examples [00:20, 952.18 examples/s] 20147 examples [00:21, 920.11 examples/s]20243 examples [00:21, 930.88 examples/s]20343 examples [00:21, 950.43 examples/s]20443 examples [00:21, 962.86 examples/s]20542 examples [00:21, 969.51 examples/s]20640 examples [00:21, 931.66 examples/s]20734 examples [00:21, 916.63 examples/s]20827 examples [00:21, 910.11 examples/s]20919 examples [00:21, 874.15 examples/s]21022 examples [00:21, 914.89 examples/s]21128 examples [00:22, 952.21 examples/s]21233 examples [00:22, 978.76 examples/s]21335 examples [00:22, 988.73 examples/s]21435 examples [00:22, 951.58 examples/s]21531 examples [00:22, 934.98 examples/s]21632 examples [00:22, 955.68 examples/s]21736 examples [00:22, 975.35 examples/s]21835 examples [00:22, 949.19 examples/s]21931 examples [00:22, 943.90 examples/s]22026 examples [00:23, 930.21 examples/s]22120 examples [00:23, 899.65 examples/s]22224 examples [00:23, 936.78 examples/s]22323 examples [00:23, 951.58 examples/s]22424 examples [00:23, 967.01 examples/s]22522 examples [00:23, 940.75 examples/s]22624 examples [00:23, 961.98 examples/s]22722 examples [00:23, 966.73 examples/s]22824 examples [00:23, 980.34 examples/s]22923 examples [00:23, 967.66 examples/s]23027 examples [00:24, 985.75 examples/s]23126 examples [00:24, 970.82 examples/s]23228 examples [00:24, 984.23 examples/s]23332 examples [00:24, 997.70 examples/s]23440 examples [00:24, 1020.82 examples/s]23543 examples [00:24, 1003.39 examples/s]23644 examples [00:24, 1004.19 examples/s]23745 examples [00:24, 995.31 examples/s] 23845 examples [00:24, 969.80 examples/s]23951 examples [00:24, 993.36 examples/s]24058 examples [00:25, 1012.41 examples/s]24164 examples [00:25, 1024.36 examples/s]24268 examples [00:25, 1026.78 examples/s]24374 examples [00:25, 1035.99 examples/s]24478 examples [00:25, 1026.44 examples/s]24581 examples [00:25, 1024.41 examples/s]24684 examples [00:25, 1013.74 examples/s]24786 examples [00:25, 977.25 examples/s] 24887 examples [00:25, 984.19 examples/s]24986 examples [00:25, 966.17 examples/s]25084 examples [00:26, 968.75 examples/s]25184 examples [00:26, 976.69 examples/s]25282 examples [00:26, 973.21 examples/s]25380 examples [00:26, 965.61 examples/s]25478 examples [00:26, 968.21 examples/s]25575 examples [00:26, 958.76 examples/s]25671 examples [00:26, 947.53 examples/s]25766 examples [00:26, 939.54 examples/s]25861 examples [00:26, 935.98 examples/s]25955 examples [00:27, 873.77 examples/s]26044 examples [00:27, 848.25 examples/s]26131 examples [00:27, 853.75 examples/s]26217 examples [00:27, 851.90 examples/s]26308 examples [00:27, 867.39 examples/s]26403 examples [00:27, 890.35 examples/s]26505 examples [00:27, 924.19 examples/s]26604 examples [00:27, 942.36 examples/s]26708 examples [00:27, 969.05 examples/s]26806 examples [00:27, 962.31 examples/s]26907 examples [00:28, 973.24 examples/s]27005 examples [00:28, 956.49 examples/s]27101 examples [00:28, 884.34 examples/s]27199 examples [00:28, 909.00 examples/s]27304 examples [00:28, 945.85 examples/s]27410 examples [00:28, 975.05 examples/s]27509 examples [00:28, 956.22 examples/s]27611 examples [00:28, 973.68 examples/s]27710 examples [00:28, 922.78 examples/s]27806 examples [00:29, 932.33 examples/s]27900 examples [00:29, 916.19 examples/s]27993 examples [00:29, 914.01 examples/s]28093 examples [00:29, 937.57 examples/s]28188 examples [00:29, 932.27 examples/s]28291 examples [00:29, 959.41 examples/s]28389 examples [00:29, 962.81 examples/s]28486 examples [00:29, 957.92 examples/s]28583 examples [00:29, 906.56 examples/s]28683 examples [00:29, 930.83 examples/s]28777 examples [00:30, 908.09 examples/s]28871 examples [00:30, 916.02 examples/s]28974 examples [00:30, 944.91 examples/s]29070 examples [00:30, 940.40 examples/s]29165 examples [00:30, 936.72 examples/s]29259 examples [00:30, 932.84 examples/s]29356 examples [00:30, 943.16 examples/s]29453 examples [00:30, 935.67 examples/s]29547 examples [00:30, 936.80 examples/s]29641 examples [00:30, 929.17 examples/s]29741 examples [00:31, 946.52 examples/s]29836 examples [00:31, 941.32 examples/s]29931 examples [00:31, 927.32 examples/s]30024 examples [00:31, 889.75 examples/s]30115 examples [00:31, 893.77 examples/s]30216 examples [00:31, 924.22 examples/s]30323 examples [00:31, 963.26 examples/s]30429 examples [00:31, 990.20 examples/s]30529 examples [00:31, 951.01 examples/s]30625 examples [00:32, 940.12 examples/s]30725 examples [00:32, 957.26 examples/s]30823 examples [00:32, 962.76 examples/s]30920 examples [00:32, 945.56 examples/s]31022 examples [00:32, 965.62 examples/s]31124 examples [00:32, 980.63 examples/s]31229 examples [00:32, 1000.35 examples/s]31330 examples [00:32, 989.25 examples/s] 31430 examples [00:32, 981.25 examples/s]31529 examples [00:32, 964.75 examples/s]31628 examples [00:33, 971.95 examples/s]31726 examples [00:33, 968.76 examples/s]31823 examples [00:33, 966.91 examples/s]31920 examples [00:33, 947.98 examples/s]32021 examples [00:33, 964.70 examples/s]32118 examples [00:33, 937.25 examples/s]32213 examples [00:33, 930.09 examples/s]32307 examples [00:33, 921.97 examples/s]32401 examples [00:33, 926.94 examples/s]32495 examples [00:33, 929.74 examples/s]32600 examples [00:34, 962.11 examples/s]32697 examples [00:34, 954.65 examples/s]32793 examples [00:34, 954.55 examples/s]32889 examples [00:34, 937.26 examples/s]32992 examples [00:34, 956.96 examples/s]33090 examples [00:34, 963.37 examples/s]33187 examples [00:34, 955.22 examples/s]33283 examples [00:34, 943.14 examples/s]33378 examples [00:34, 943.82 examples/s]33473 examples [00:35, 944.26 examples/s]33568 examples [00:35, 901.59 examples/s]33669 examples [00:35, 928.02 examples/s]33763 examples [00:35, 900.33 examples/s]33862 examples [00:35, 924.88 examples/s]33958 examples [00:35, 934.83 examples/s]34053 examples [00:35, 938.29 examples/s]34148 examples [00:35, 920.95 examples/s]34241 examples [00:35, 915.30 examples/s]34341 examples [00:35, 938.06 examples/s]34436 examples [00:36, 935.25 examples/s]34530 examples [00:36, 917.03 examples/s]34622 examples [00:36, 915.30 examples/s]34714 examples [00:36, 913.42 examples/s]34806 examples [00:36, 889.85 examples/s]34898 examples [00:36, 897.03 examples/s]34998 examples [00:36, 925.61 examples/s]35101 examples [00:36, 953.68 examples/s]35197 examples [00:36, 954.73 examples/s]35304 examples [00:36, 984.35 examples/s]35407 examples [00:37, 996.15 examples/s]35507 examples [00:37, 981.37 examples/s]35606 examples [00:37, 941.42 examples/s]35705 examples [00:37, 954.75 examples/s]35801 examples [00:37, 945.59 examples/s]35909 examples [00:37, 980.55 examples/s]36008 examples [00:37, 965.25 examples/s]36105 examples [00:37, 946.78 examples/s]36208 examples [00:37, 969.71 examples/s]36308 examples [00:38, 976.38 examples/s]36414 examples [00:38, 999.08 examples/s]36520 examples [00:38, 1016.21 examples/s]36622 examples [00:38, 1010.75 examples/s]36724 examples [00:38, 975.48 examples/s] 36822 examples [00:38, 949.52 examples/s]36926 examples [00:38, 973.94 examples/s]37028 examples [00:38, 986.10 examples/s]37130 examples [00:38, 994.19 examples/s]37234 examples [00:38, 1005.93 examples/s]37338 examples [00:39, 1014.12 examples/s]37440 examples [00:39, 975.00 examples/s] 37545 examples [00:39, 996.13 examples/s]37650 examples [00:39, 1009.99 examples/s]37754 examples [00:39, 1016.67 examples/s]37856 examples [00:39, 1010.10 examples/s]37958 examples [00:39, 992.80 examples/s] 38068 examples [00:39, 1020.63 examples/s]38171 examples [00:39, 1007.87 examples/s]38273 examples [00:39, 1001.43 examples/s]38377 examples [00:40, 1010.33 examples/s]38479 examples [00:40, 1003.86 examples/s]38583 examples [00:40, 1013.73 examples/s]38685 examples [00:40, 990.41 examples/s] 38785 examples [00:40, 977.36 examples/s]38885 examples [00:40, 983.26 examples/s]38990 examples [00:40, 1001.26 examples/s]39091 examples [00:40, 988.06 examples/s] 39190 examples [00:40, 910.39 examples/s]39283 examples [00:41, 877.75 examples/s]39376 examples [00:41, 891.16 examples/s]39467 examples [00:41, 896.41 examples/s]39558 examples [00:41, 880.18 examples/s]39657 examples [00:41, 908.71 examples/s]39751 examples [00:41, 917.37 examples/s]39848 examples [00:41, 929.98 examples/s]39943 examples [00:41, 927.61 examples/s]40037 examples [00:41, 831.79 examples/s]40123 examples [00:42, 828.03 examples/s]40217 examples [00:42, 858.17 examples/s]40305 examples [00:42, 814.45 examples/s]40395 examples [00:42, 836.35 examples/s]40480 examples [00:42, 811.46 examples/s]40569 examples [00:42, 828.42 examples/s]40653 examples [00:42, 808.60 examples/s]40739 examples [00:42, 822.23 examples/s]40829 examples [00:42, 842.57 examples/s]40926 examples [00:42, 876.93 examples/s]41025 examples [00:43, 908.04 examples/s]41123 examples [00:43, 927.93 examples/s]41217 examples [00:43, 927.20 examples/s]41311 examples [00:43, 910.46 examples/s]41411 examples [00:43, 934.90 examples/s]41512 examples [00:43, 954.16 examples/s]41610 examples [00:43, 960.29 examples/s]41707 examples [00:43, 937.79 examples/s]41808 examples [00:43, 958.28 examples/s]41905 examples [00:43, 949.29 examples/s]42005 examples [00:44, 961.95 examples/s]42111 examples [00:44, 987.54 examples/s]42217 examples [00:44, 1005.67 examples/s]42318 examples [00:44, 993.11 examples/s] 42418 examples [00:44, 974.08 examples/s]42516 examples [00:44, 958.38 examples/s]42613 examples [00:44, 949.85 examples/s]42710 examples [00:44, 954.15 examples/s]42806 examples [00:44, 944.67 examples/s]42901 examples [00:45, 922.17 examples/s]42994 examples [00:45, 910.94 examples/s]43096 examples [00:45, 940.43 examples/s]43191 examples [00:45, 908.72 examples/s]43293 examples [00:45, 935.08 examples/s]43394 examples [00:45, 955.69 examples/s]43495 examples [00:45, 970.39 examples/s]43600 examples [00:45, 990.82 examples/s]43705 examples [00:45, 1007.75 examples/s]43810 examples [00:45, 1020.05 examples/s]43913 examples [00:46, 1013.27 examples/s]44015 examples [00:46, 997.81 examples/s] 44116 examples [00:46, 1001.43 examples/s]44217 examples [00:46, 994.19 examples/s] 44322 examples [00:46, 1010.04 examples/s]44424 examples [00:46, 968.03 examples/s] 44522 examples [00:46, 963.97 examples/s]44619 examples [00:46, 943.73 examples/s]44714 examples [00:46, 943.79 examples/s]44810 examples [00:46, 946.12 examples/s]44909 examples [00:47, 958.40 examples/s]45014 examples [00:47, 984.10 examples/s]45113 examples [00:47, 974.69 examples/s]45211 examples [00:47, 932.22 examples/s]45305 examples [00:47, 915.58 examples/s]45403 examples [00:47, 932.37 examples/s]45497 examples [00:47, 912.03 examples/s]45595 examples [00:47, 930.97 examples/s]45694 examples [00:47, 946.98 examples/s]45791 examples [00:48, 951.85 examples/s]45892 examples [00:48, 966.45 examples/s]45989 examples [00:48, 945.51 examples/s]46084 examples [00:48, 933.37 examples/s]46180 examples [00:48, 939.87 examples/s]46276 examples [00:48, 942.29 examples/s]46371 examples [00:48, 918.75 examples/s]46464 examples [00:48, 907.67 examples/s]46555 examples [00:48, 881.48 examples/s]46648 examples [00:48, 894.67 examples/s]46743 examples [00:49, 908.49 examples/s]46836 examples [00:49, 913.85 examples/s]46928 examples [00:49, 901.82 examples/s]47025 examples [00:49, 919.63 examples/s]47130 examples [00:49, 953.51 examples/s]47227 examples [00:49, 955.79 examples/s]47330 examples [00:49, 975.37 examples/s]47429 examples [00:49, 977.57 examples/s]47527 examples [00:49, 954.50 examples/s]47624 examples [00:49, 957.74 examples/s]47720 examples [00:50, 934.50 examples/s]47816 examples [00:50, 939.67 examples/s]47911 examples [00:50, 935.57 examples/s]48005 examples [00:50, 925.74 examples/s]48099 examples [00:50, 929.00 examples/s]48195 examples [00:50, 935.95 examples/s]48289 examples [00:50, 927.38 examples/s]48382 examples [00:50, 898.00 examples/s]48473 examples [00:50, 846.78 examples/s]48559 examples [00:51, 845.24 examples/s]48652 examples [00:51, 868.08 examples/s]48755 examples [00:51, 909.52 examples/s]48856 examples [00:51, 935.70 examples/s]48951 examples [00:51, 891.00 examples/s]49044 examples [00:51, 900.72 examples/s]49135 examples [00:51, 885.04 examples/s]49237 examples [00:51, 919.45 examples/s]49335 examples [00:51, 935.44 examples/s]49439 examples [00:51, 963.24 examples/s]49540 examples [00:52, 976.42 examples/s]49640 examples [00:52, 980.49 examples/s]49744 examples [00:52, 997.41 examples/s]49845 examples [00:52, 965.15 examples/s]49942 examples [00:52, 945.70 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 10%|▉         | 4958/50000 [00:00<00:00, 49578.51 examples/s] 34%|███▍      | 17193/50000 [00:00<00:00, 60345.98 examples/s] 60%|██████    | 30089/50000 [00:00<00:00, 71807.55 examples/s] 85%|████████▍ | 42491/50000 [00:00<00:00, 82186.54 examples/s]                                                               0 examples [00:00, ? examples/s]80 examples [00:00, 789.68 examples/s]167 examples [00:00, 810.78 examples/s]271 examples [00:00, 867.11 examples/s]371 examples [00:00, 901.66 examples/s]476 examples [00:00, 941.46 examples/s]577 examples [00:00, 960.07 examples/s]680 examples [00:00, 978.53 examples/s]774 examples [00:00, 966.28 examples/s]876 examples [00:00, 980.37 examples/s]972 examples [00:01, 968.98 examples/s]1074 examples [00:01, 981.58 examples/s]1171 examples [00:01, 957.46 examples/s]1266 examples [00:01, 942.92 examples/s]1360 examples [00:01, 923.83 examples/s]1465 examples [00:01, 956.59 examples/s]1561 examples [00:01, 951.91 examples/s]1657 examples [00:01, 918.75 examples/s]1750 examples [00:01, 911.40 examples/s]1855 examples [00:01, 946.93 examples/s]1953 examples [00:02, 956.14 examples/s]2049 examples [00:02, 951.67 examples/s]2149 examples [00:02, 965.00 examples/s]2254 examples [00:02, 986.72 examples/s]2355 examples [00:02, 993.47 examples/s]2455 examples [00:02, 994.87 examples/s]2556 examples [00:02, 997.88 examples/s]2656 examples [00:02, 984.43 examples/s]2763 examples [00:02, 1007.61 examples/s]2867 examples [00:02, 1014.78 examples/s]2973 examples [00:03, 1025.20 examples/s]3076 examples [00:03, 1017.99 examples/s]3178 examples [00:03, 1014.48 examples/s]3280 examples [00:03, 993.07 examples/s] 3380 examples [00:03, 994.60 examples/s]3480 examples [00:03, 990.59 examples/s]3584 examples [00:03, 1004.42 examples/s]3685 examples [00:03, 986.17 examples/s] 3784 examples [00:03, 966.16 examples/s]3885 examples [00:03, 978.30 examples/s]3984 examples [00:04, 967.45 examples/s]4081 examples [00:04, 911.45 examples/s]4180 examples [00:04, 933.02 examples/s]4279 examples [00:04, 947.20 examples/s]4375 examples [00:04, 903.08 examples/s]4467 examples [00:04, 894.51 examples/s]4568 examples [00:04, 925.93 examples/s]4672 examples [00:04, 955.97 examples/s]4780 examples [00:04, 988.60 examples/s]4880 examples [00:05, 990.38 examples/s]4983 examples [00:05, 1000.15 examples/s]5088 examples [00:05, 1013.95 examples/s]5190 examples [00:05, 1009.91 examples/s]5294 examples [00:05, 1017.64 examples/s]5396 examples [00:05, 967.39 examples/s] 5494 examples [00:05, 945.28 examples/s]5590 examples [00:05, 934.38 examples/s]5684 examples [00:05, 930.06 examples/s]5790 examples [00:05, 965.29 examples/s]5888 examples [00:06, 925.64 examples/s]5982 examples [00:06, 887.65 examples/s]6087 examples [00:06, 930.18 examples/s]6197 examples [00:06, 971.15 examples/s]6301 examples [00:06, 990.28 examples/s]6402 examples [00:06, 993.55 examples/s]6507 examples [00:06, 1009.55 examples/s]6609 examples [00:06, 1005.48 examples/s]6710 examples [00:06, 987.54 examples/s] 6810 examples [00:07, 907.14 examples/s]6903 examples [00:07, 882.29 examples/s]6993 examples [00:07, 850.60 examples/s]7099 examples [00:07, 902.51 examples/s]7200 examples [00:07, 931.62 examples/s]7295 examples [00:07, 867.84 examples/s]7385 examples [00:07, 873.88 examples/s]7474 examples [00:07, 873.92 examples/s]7563 examples [00:07, 857.28 examples/s]7658 examples [00:08, 880.03 examples/s]7749 examples [00:08, 888.11 examples/s]7848 examples [00:08, 915.30 examples/s]7941 examples [00:08, 916.72 examples/s]8038 examples [00:08, 931.21 examples/s]8132 examples [00:08, 912.08 examples/s]8227 examples [00:08, 922.98 examples/s]8323 examples [00:08, 932.36 examples/s]8422 examples [00:08, 947.21 examples/s]8530 examples [00:08, 983.12 examples/s]8629 examples [00:09, 956.67 examples/s]8726 examples [00:09, 954.45 examples/s]8822 examples [00:09, 933.74 examples/s]8922 examples [00:09, 950.12 examples/s]9030 examples [00:09, 984.36 examples/s]9138 examples [00:09, 1009.87 examples/s]9240 examples [00:09, 912.56 examples/s] 9334 examples [00:09, 915.65 examples/s]9428 examples [00:09, 906.53 examples/s]9520 examples [00:09, 901.71 examples/s]9611 examples [00:10, 902.93 examples/s]9714 examples [00:10, 936.80 examples/s]9820 examples [00:10, 968.53 examples/s]9918 examples [00:10, 964.26 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteM0EAK8/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteM0EAK8/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f37c7b7d950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f37c7b7d950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f37c7b7d950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3750f473c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3750ff1b70>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f37c7b7d950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f37c7b7d950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f37c7b7d950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f37c51c12e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f37c51c10f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f374b019ae8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f374b019ae8> 

  function with postional parmater data_info <function split_train_valid at 0x7f374b019ae8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f374b019bf8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f374b019bf8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f374b019bf8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=41d58d43ab620ccea275f9e714bc3fe0d232e56a5109fe0e2ac0b43e2fdbbf49
  Stored in directory: /tmp/pip-ephem-wheel-cache-haakuppy/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<44:11:11, 5.42kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<31:09:40, 7.69kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<21:51:53, 11.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<15:18:26, 15.6kB/s].vector_cache/glove.6B.zip:   0%|          | 3.62M/862M [00:02<10:41:03, 22.3kB/s].vector_cache/glove.6B.zip:   1%|          | 7.69M/862M [00:02<7:26:43, 31.9kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:02<5:11:08, 45.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:02<3:36:26, 65.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:02<2:31:06, 92.8kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.1M/862M [00:02<1:45:12, 132kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:02<1:13:28, 189kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.6M/862M [00:02<51:11, 269kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<35:47, 384kB/s].vector_cache/glove.6B.zip:   5%|▍         | 43.1M/862M [00:02<24:59, 546kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.0M/862M [00:03<17:31, 775kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:03<12:17, 1.10MB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.4M/862M [00:04<10:30, 1.28MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<07:33, 1.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<06:06, 2.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:06<12:04:43, 18.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:06<8:28:26, 26.4kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<5:55:30, 37.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:08<4:11:23, 53.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:08<2:57:18, 75.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.6M/862M [00:08<2:04:12, 107kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:10<1:29:50, 148kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:10<1:05:34, 202kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:10<46:31, 285kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:10<32:34, 405kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:12<56:20, 234kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.2M/862M [00:12<40:45, 324kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.8M/862M [00:12<28:48, 457kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:13<23:10, 567kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:14<18:49, 698kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:14<13:49, 949kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:14<09:47, 1.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:15<12:41:57, 17.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:16<8:54:27, 24.4kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.0M/862M [00:16<6:13:42, 34.9kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:17<4:23:55, 49.3kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:18<3:06:00, 69.9kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.1M/862M [00:18<2:10:16, 99.5kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:19<1:33:58, 138kB/s] .vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:20<1:07:04, 193kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.3M/862M [00:20<47:10, 273kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:21<35:59, 357kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:21<26:28, 486kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:22<18:49, 682kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:23<16:10, 791kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:23<13:54, 920kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:24<10:17, 1.24MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<07:21, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:25<11:51, 1.07MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:25<09:36, 1.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:02, 1.80MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<07:53, 1.61MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<08:07, 1.56MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<06:13, 2.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<04:37, 2.73MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<06:42, 1.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<05:58, 2.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<04:29, 2.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<06:04, 2.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<06:53, 1.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<06:08, 2.04MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<04:32, 2.75MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<06:01, 2.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<05:33, 2.24MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<05:18, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<05:46, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<05:22, 2.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<04:05, 3.02MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<05:37, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<05:11, 2.37MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<03:56, 3.12MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:38, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<05:11, 2.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:39<03:54, 3.13MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<05:37, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<05:10, 2.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:41<03:52, 3.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<05:33, 2.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<06:20, 1.91MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<05:02, 2.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<05:28, 2.20MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:45<05:03, 2.38MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<03:50, 3.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<03:27, 3.46MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<8:08:13, 24.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<5:42:13, 35.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<3:58:54, 49.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<2:53:10, 68.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<2:03:49, 96.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<1:27:10, 136kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<1:01:01, 194kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<47:19, 250kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<34:07, 347kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<24:07, 490kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<17:01, 691kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<42:25, 277kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<32:05, 367kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<23:02, 510kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<17:57, 652kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<13:46, 849kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<09:52, 1.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<09:38, 1.21MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<07:55, 1.47MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<05:49, 1.99MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<06:48, 1.70MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:59<07:06, 1.63MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<05:28, 2.11MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<03:58, 2.90MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<09:15, 1.24MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<07:38, 1.50MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<05:37, 2.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:36, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<05:47, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<04:19, 2.63MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:42, 1.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:10, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<03:54, 2.90MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:23, 2.09MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:55, 2.29MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<03:43, 3.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:15, 2.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:48, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<03:36, 3.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<05:36, 1.99MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<04:05, 2.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<05:18, 2.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<04:51, 2.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<03:40, 3.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<05:09, 2.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<04:43, 2.33MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<03:34, 3.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<05:05, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<05:47, 1.89MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<04:31, 2.41MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<03:19, 3.28MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<06:51, 1.59MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:55, 1.84MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<04:22, 2.48MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:35, 1.93MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:06, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:49, 2.24MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:20<03:29, 3.08MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<10:20:01, 17.3kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<7:14:49, 24.7kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<5:03:55, 35.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<3:34:32, 49.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<2:32:18, 70.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<1:46:57, 99.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<1:14:44, 142kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<58:27, 181kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<41:58, 252kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<29:35, 357kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<23:06, 456kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<18:24, 572kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<13:25, 784kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<09:50, 1.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<6:56:24, 25.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<4:51:09, 35.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<3:24:57, 50.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<2:25:32, 71.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<1:42:12, 102kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<1:11:28, 145kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<54:20, 190kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<39:05, 264kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<27:33, 374kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<21:36, 475kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<16:09, 635kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<11:32, 887kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<10:28, 974kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<08:23, 1.22MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<06:04, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<06:39, 1.52MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:40<06:43, 1.51MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<05:09, 1.96MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:42, 2.71MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<12:00, 838kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<09:24, 1.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<06:47, 1.48MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<07:05, 1.41MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<06:59, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<05:19, 1.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<03:54, 2.55MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:50, 1.70MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:05, 1.95MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<03:48, 2.60MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:57, 1.99MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:30, 1.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<05:34, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:09, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:37, 2.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<03:28, 2.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:39, 2.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:09, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<04:05, 2.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:25, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:06, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:07, 3.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:22, 2.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:03, 2.36MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<03:02, 3.14MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<04:19, 2.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<03:59, 2.38MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:02, 3.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:21, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:58, 1.90MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<03:57, 2.38MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<02:58, 3.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:08, 1.53MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:09, 1.52MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:46, 1.96MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<04:50, 1.92MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:18, 1.75MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:06, 2.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<03:02, 3.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<04:59, 1.85MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<05:20, 1.73MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:11, 2.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:01, 3.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<10:05, 908kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<08:50, 1.04MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<06:38, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:44, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<09:21, 973kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<08:24, 1.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<06:20, 1.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<04:31, 2.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<10:48, 836kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<09:20, 966kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<06:56, 1.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<04:56, 1.82MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<22:01, 407kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<6:00:01, 24.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<4:12:05, 35.5kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<2:55:40, 50.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<2:10:12, 68.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<1:34:17, 94.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<1:06:36, 134kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<46:47, 190kB/s]  .vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<32:42, 270kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<3:07:53, 47.0kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<2:13:12, 66.3kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<1:33:32, 94.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<1:05:14, 134kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<58:06, 151kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<42:22, 207kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<30:03, 291kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<21:01, 414kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<26:30, 328kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<20:19, 428kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<14:34, 596kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<10:21, 836kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<09:40, 892kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:23<08:27, 1.02MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<06:20, 1.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:31, 1.90MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<10:44, 796kB/s] .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<09:17, 921kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<06:50, 1.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:57, 1.72MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:50, 1.46MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<05:50, 1.45MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<04:30, 1.88MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:14, 2.60MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<07:36, 1.11MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<06:58, 1.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<05:17, 1.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:51, 2.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<06:24, 1.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<08:37, 968kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<07:05, 1.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:10, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:47, 2.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<06:04, 1.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<08:19, 995kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<06:48, 1.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<05:12, 1.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:09, 1.99MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:25, 2.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<02:51, 2.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<02:31, 3.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<09:28, 867kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<09:18, 882kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<07:11, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<05:30, 1.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:18, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:28, 2.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<02:53, 2.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<02:23, 3.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<5:18:03, 25.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<3:44:55, 36.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<2:37:56, 51.5kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<1:50:47, 73.3kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<1:17:59, 104kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<54:55, 148kB/s]  .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<38:50, 208kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<27:35, 293kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<19:45, 409kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<5:32:33, 24.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<3:57:10, 34.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<2:47:07, 48.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<1:57:17, 68.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<1:22:24, 97.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<58:01, 139kB/s]   .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<40:58, 196kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<29:00, 276kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<1:36:33, 83.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<1:09:45, 115kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<49:25, 162kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<35:00, 228kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:41<25:06, 318kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<17:52, 446kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<12:56, 614kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<09:53, 803kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<07:22, 1.08MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<31:42, 250kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<25:13, 315kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<18:25, 431kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<13:26, 590kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<09:54, 799kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<07:28, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:43, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:31, 1.74MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<07:07, 1.11MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<07:46, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<06:11, 1.27MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<04:50, 1.62MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<03:50, 2.04MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<03:11, 2.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<02:43, 2.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<02:24, 3.25MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<07:15, 1.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<07:49, 997kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<06:11, 1.26MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:49, 1.61MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<03:48, 2.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<03:01, 2.56MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<02:53, 2.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<02:27, 3.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:06, 3.66MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<29:58, 258kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<24:25, 317kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<17:51, 433kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<13:09, 586kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<09:58, 773kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<07:31, 1.02MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<06:05, 1.27MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<04:51, 1.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:53, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:30, 2.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<02:55, 2.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<02:48, 2.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<14:59, 511kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<11:59, 639kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<09:00, 850kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<06:50, 1.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<05:19, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:15, 1.79MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:29, 2.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<02:57, 2.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<06:34, 1.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<07:16, 1.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<05:53, 1.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:38, 1.63MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<03:44, 2.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:03, 2.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<02:36, 2.89MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<02:18, 3.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<05:57, 1.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<06:48, 1.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<05:29, 1.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:20, 1.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:29, 2.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<02:54, 2.57MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<02:30, 2.98MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<02:12, 3.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<07:52, 947kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<07:57, 937kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<06:17, 1.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:54, 1.52MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:56, 1.89MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<03:22, 2.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<03:00, 2.47MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<02:35, 2.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<02:41, 2.75MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<02:26, 3.03MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<02:24, 3.07MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<42:53, 172kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<31:36, 234kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<22:44, 325kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<16:29, 447kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<12:02, 611kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<08:59, 818kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<06:43, 1.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:30, 1.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:16, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<08:35, 853kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<1:48:36, 67.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<1:16:46, 95.4kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<54:14, 135kB/s]   .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<38:28, 190kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<27:27, 266kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<19:44, 369kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<14:21, 507kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<10:34, 688kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:02<11:33, 628kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:02<10:49, 671kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<08:22, 866kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<06:23, 1.13MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<04:59, 1.45MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:01, 1.80MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:21, 2.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<02:52, 2.51MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<02:32, 2.83MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<06:58, 1.03MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<06:18, 1.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<05:00, 1.43MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:57, 1.81MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:18, 2.17MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<02:55, 2.45MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<02:34, 2.78MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<02:19, 3.07MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<02:09, 3.31MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<06:23, 1.11MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<07:10, 993kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<05:48, 1.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<04:38, 1.53MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:54, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:28, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:09, 2.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<02:47, 2.53MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<02:52, 2.47MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<02:38, 2.67MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<02:39, 2.66MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<02:26, 2.90MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<02:33, 2.76MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<08:27, 833kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<06:53, 1.02MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<05:29, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<04:29, 1.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<03:47, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:18, 2.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<02:53, 2.43MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<02:38, 2.65MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<02:19, 3.01MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<02:33, 2.73MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<02:28, 2.83MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<23:29, 297kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<18:04, 386kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<13:21, 523kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<09:57, 699kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<07:35, 916kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:54, 1.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:44, 1.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:54, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:11, 2.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:10, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<02:47, 2.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<17:49, 388kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<14:06, 490kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<10:33, 654kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<08:01, 861kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<06:12, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:56, 1.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:03, 1.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:26, 2.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<02:55, 2.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<02:45, 2.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<02:28, 2.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<10:32, 650kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<08:54, 769kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<06:56, 985kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<05:26, 1.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<04:23, 1.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:32, 1.92MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:08, 2.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:47, 2.43MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:25, 2.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:20, 2.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<02:07, 3.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<10:16, 660kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<08:27, 801kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<06:34, 1.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<05:13, 1.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<04:09, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:49, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:13, 2.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:54, 2.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:36, 2.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:47, 2.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:34, 2.61MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<02:34, 2.61MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<02:24, 2.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<02:26, 2.74MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<1:11:06, 94.4kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<51:30, 130kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<36:45, 182kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<26:19, 254kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<19:01, 351kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<13:50, 482kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<10:09, 657kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<08:00, 832kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<06:11, 1.08MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<04:50, 1.37MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<03:57, 1.68MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<07:32, 880kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<06:46, 980kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<05:17, 1.25MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<04:09, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<03:47, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<03:06, 2.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:38, 2.50MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:37, 2.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:17, 2.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:09, 3.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:01, 3.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:00, 3.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<2:58:36, 36.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<2:06:26, 52.0kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<1:28:59, 73.7kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<1:02:40, 105kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<44:18, 148kB/s]  .vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<31:35, 207kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<22:30, 290kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<16:24, 398kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<12:04, 541kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<08:54, 731kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<06:49, 954kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<09:01, 721kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<07:42, 844kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<06:00, 1.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<04:43, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<03:49, 1.69MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<03:07, 2.07MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<02:44, 2.36MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:29, 2.60MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:11, 2.95MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:08, 3.00MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<05:18, 1.21MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<05:06, 1.26MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:06, 1.56MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<03:15, 1.97MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<03:11, 2.01MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<02:44, 2.34MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<02:25, 2.64MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:12, 2.90MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<01:59, 3.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<01:47, 3.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<07:25, 858kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<06:33, 970kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<05:09, 1.23MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<04:04, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:20, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<02:50, 2.23MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:28, 2.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:13, 2.84MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:01, 3.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<05:49, 1.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<06:40, 945kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<05:15, 1.20MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<04:05, 1.53MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<03:31, 1.78MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<02:51, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<02:26, 2.57MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<02:17, 2.73MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<02:02, 3.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<01:52, 3.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<06:00, 1.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<06:31, 955kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<06:22, 976kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:48, 1.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<04:15, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<03:39, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<03:14, 1.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:49, 2.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:47, 2.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:37, 2.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:28, 2.49MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:22, 2.60MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<02:16, 2.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<05:25, 1.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:41, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:53, 1.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:19, 1.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<02:55, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<02:36, 2.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:39, 2.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:33, 2.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:47, 2.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:38, 2.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<02:48, 2.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:35, 2.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:49, 2.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<02:44, 2.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:40, 2.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<15:16, 399kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<11:45, 518kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<08:58, 678kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<07:00, 867kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<05:37, 1.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<04:38, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:57, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:28, 1.75MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:07, 1.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:53, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:42, 2.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:36, 2.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:20, 2.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<07:50, 767kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<06:31, 923kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<05:13, 1.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:20, 1.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<03:42, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<03:17, 1.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:54, 2.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:42, 2.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:32, 2.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:17, 2.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:29, 2.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:18, 2.59MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:27, 2.42MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<05:13, 1.14MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<04:35, 1.29MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:51, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:19, 1.79MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:56, 2.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<02:38, 2.25MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<02:28, 2.38MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<02:18, 2.57MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:12, 2.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:04, 2.83MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:06, 2.80MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:08, 2.75MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<05:29, 1.07MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<04:40, 1.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:50, 1.53MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:15, 1.80MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:51, 2.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<02:29, 2.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<02:17, 2.55MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:09, 2.70MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:03, 2.84MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<01:58, 2.94MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<01:55, 3.02MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<1:08:21, 85.1kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<49:15, 118kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<35:03, 166kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<25:01, 232kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<18:01, 321kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<13:07, 441kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<09:41, 596kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<07:14, 796kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<05:31, 1.04MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<04:26, 1.30MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<06:41, 860kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<54:57, 105kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<38:55, 148kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<27:38, 208kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<19:49, 289kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<14:20, 399kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<10:23, 551kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<07:56, 719kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<05:56, 960kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<04:47, 1.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:45, 1.52MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<09:05, 625kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<08:42, 653kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<06:41, 850kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<05:07, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<04:02, 1.40MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:08, 1.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:30, 2.25MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:17, 2.46MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<01:54, 2.95MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<05:03, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<05:22, 1.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<04:15, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<03:17, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:33, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:02, 2.72MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<01:46, 3.13MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<01:30, 3.70MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<06:52, 808kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<07:49, 709kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<06:11, 895kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<04:38, 1.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:27, 1.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:37, 2.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:06, 2.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:13, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:53<05:26, 1.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<04:20, 1.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<03:16, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:32, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<01:55, 2.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<01:36, 3.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:54, 1.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<05:29, 985kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<04:16, 1.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<03:12, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:25, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<01:51, 2.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<01:27, 3.65MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<1:05:42, 81.3kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<47:42, 112kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<33:46, 158kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<23:43, 224kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<16:39, 317kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<14:32, 363kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<13:04, 403kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<09:52, 533kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<07:04, 742kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<05:04, 1.03MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<05:21, 970kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<05:02, 1.03MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<03:49, 1.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:48, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:16, 1.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<04:36, 1.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<03:49, 1.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:51, 1.79MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:04, 2.44MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:26, 930kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:54, 858kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<04:38, 1.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:21, 1.50MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:28, 2.03MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:40, 1.36MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<05:01, 993kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<04:08, 1.21MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:00, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:13, 2.23MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:25, 1.44MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<04:36, 1.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<03:48, 1.29MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:48, 1.75MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:03, 2.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<05:33, 876kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<06:04, 800kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<04:47, 1.01MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<03:27, 1.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:30, 1.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<04:27, 1.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<05:15, 912kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<04:11, 1.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:03, 1.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:12, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<07:48, 605kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<07:24, 637kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<05:39, 834kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<04:03, 1.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:54, 1.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<06:14, 746kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<06:07, 759kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<04:41, 991kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:23, 1.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:28, 1.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:20, 1.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<04:14, 1.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:20, 1.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:28, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<01:48, 2.50MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:00, 1.50MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:48, 1.18MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:01, 1.49MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:13, 2.01MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<01:38, 2.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<03:00, 1.48MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:46, 1.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:00, 1.47MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:12, 2.00MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:38, 2.68MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:57, 1.48MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:52, 1.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:04, 1.42MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:17, 1.90MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<01:43, 2.50MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:37, 1.64MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<04:11, 1.03MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<03:32, 1.22MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:35, 1.65MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<01:55, 2.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:26, 2.96MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<11:49, 359kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<09:22, 452kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<06:46, 623kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:49, 872kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:27, 1.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<04:35, 910kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<05:17, 788kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<04:12, 989kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<03:03, 1.35MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:13, 1.85MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<04:01, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<04:51, 844kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:54, 1.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:50, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:04, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<02:54, 1.38MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<04:02, 998kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<03:18, 1.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:25, 1.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:45, 2.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<03:22, 1.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<04:08, 957kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<03:21, 1.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:27, 1.60MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:46, 2.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<03:56, 987kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<04:30, 864kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<03:30, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:44, 1.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:57, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:32, 2.50MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<04:57, 770kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<04:48, 794kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<03:39, 1.04MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:39, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:59, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:30, 2.50MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<03:13, 1.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<03:29, 1.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<02:45, 1.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:02, 1.82MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:30, 2.45MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:59, 1.86MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<1:52:38, 32.8kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<1:19:21, 46.5kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<55:28, 66.2kB/s]  .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<38:41, 94.4kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<27:02, 134kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<22:21, 162kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<17:54, 202kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<13:05, 277kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<09:15, 389kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<06:33, 547kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<04:39, 764kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<20:14, 176kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<15:11, 234kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<10:52, 326kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<07:39, 460kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<05:25, 646kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<03:53, 897kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<47:09, 74.0kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<35:26, 98.4kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<25:19, 138kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<17:46, 195kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<12:28, 277kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<08:47, 390kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<08:08, 420kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<06:40, 512kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<04:52, 698kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<03:30, 966kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:32, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:51, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<05:16, 635kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<05:31, 606kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<04:21, 767kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<03:09, 1.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:24, 1.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:44, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<04:34, 716kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<04:33, 720kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<03:31, 927kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:34, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:53, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:28, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:07, 2.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<2:31:54, 21.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<1:47:30, 29.9kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<1:15:24, 42.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<52:36, 60.5kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<36:43, 86.2kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<25:41, 123kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<20:14, 155kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<15:21, 204kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<10:59, 285kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<07:46, 401kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<05:32, 561kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<03:57, 782kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<02:52, 1.07MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<07:42, 399kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<06:28, 474kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<04:48, 638kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<03:26, 885kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:29, 1.21MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:50, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:04, 976kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<03:13, 933kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<02:31, 1.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:51, 1.61MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:23, 2.13MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:02, 2.81MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<08:20, 352kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<06:56, 422kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<05:06, 572kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<03:39, 794kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:38, 1.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:54, 1.50MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<06:13, 461kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<05:23, 532kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<04:00, 713kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:53, 982kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:06, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:32, 1.82MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<18:03, 155kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<13:36, 205kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<09:45, 286kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<06:59, 397kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<04:56, 558kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:33, 770kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<03:53, 703kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<04:03, 671kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<03:08, 865kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<02:20, 1.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:46, 1.52MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:20, 2.01MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:03, 2.54MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<00:52, 3.06MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<04:58, 535kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<04:47, 555kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<03:39, 725kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<02:40, 988kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:57, 1.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:28, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:08, 2.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<56:47, 45.6kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<41:01, 63.1kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<28:56, 89.3kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<20:15, 127kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<14:10, 180kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<09:56, 255kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<07:06, 355kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<07:36, 331kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<06:27, 390kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<04:47, 525kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<03:26, 726kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<02:29, 994kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:49, 1.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<02:29, 987kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<02:49, 865kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<02:14, 1.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<01:40, 1.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:14, 1.93MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<00:57, 2.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:46, 3.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<03:48, 626kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<03:43, 639kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<02:55, 811kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:22<02:20, 1.02MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<01:54, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:37, 1.45MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:25, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:15, 1.86MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:06, 2.10MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:08, 2.07MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:03, 2.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<00:59, 2.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<00:56, 2.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<00:51, 2.72MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<03:34, 648kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:44, 842kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:09, 1.07MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:45, 1.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:27, 1.56MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:16, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:05, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<00:56, 2.40MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<00:59, 2.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<00:53, 2.53MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:55, 2.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:50, 2.68MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<00:55, 2.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<04:08, 543kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<03:11, 702kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:30, 892kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:58, 1.13MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:36, 1.38MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:19, 1.67MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:09, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:01, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:55, 2.36MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:53, 2.49MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:49, 2.64MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:48, 2.70MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:45, 2.88MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<02:42, 804kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<02:10, 997kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:44, 1.25MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:25, 1.53MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:13, 1.76MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<01:02, 2.06MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<00:57, 2.23MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:51, 2.50MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:47, 2.69MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:45, 2.79MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:43, 2.91MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:44, 2.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:40, 3.09MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:20, 900kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:09, 978kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:43, 1.21MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:24, 1.49MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<01:10, 1.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:00, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:51, 2.39MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:47, 2.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:44, 2.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:40, 3.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:36, 1.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:35, 1.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:16, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:02, 1.94MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:58, 2.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<00:55, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:46, 2.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:46, 2.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:46, 2.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:47, 2.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:45, 2.64MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:45, 2.60MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<00:43, 2.73MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:44, 2.66MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<03:28, 567kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<02:43, 720kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:06, 933kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:39, 1.18MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:22, 1.41MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:34<01:13, 1.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<01:05, 1.77MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<01:01, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:57, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:54, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:52, 2.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:50, 2.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:49, 2.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:43, 1.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:32, 1.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:18, 1.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:06, 1.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:59, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<00:52, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<00:58, 1.94MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<00:53, 2.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<00:50, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:59, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:52, 2.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:54, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:54, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<02:10, 849kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:47, 1.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:31, 1.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:20, 1.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:09, 1.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:04, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:02, 1.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<00:57, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<00:57, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<00:53, 2.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:38<00:54, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:52, 2.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:51, 2.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:48, 2.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:54, 1.95MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:52, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:51, 2.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:46, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:52, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:50, 2.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:49, 2.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:47, 2.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:46, 2.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<00:48, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<00:47, 2.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:47, 2.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:58, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:58, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:53, 1.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:00, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:55, 1.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:59, 1.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:56, 1.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:53, 1.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:54, 1.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:51, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:59, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:00, 1.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:03, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:00, 1.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:06, 1.53MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:42<01:02, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:05, 1.55MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:00, 1.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:04, 1.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:02, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:03, 1.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<01:00, 1.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<01:01, 1.61MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<00:59, 1.66MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:56, 1.75MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:43<00:59, 1.66MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:43<00:57, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:59, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:57, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:54, 1.80MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:57, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:54, 1.81MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:57, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:53, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:44<00:56, 1.73MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:44<00:53, 1.83MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:44<00:55, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<00:51, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<00:53, 1.80MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<00:50, 1.92MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<00:52, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<00:49, 1.94MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:51, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:48, 1.98MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:45<00:50, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:45<00:47, 2.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:45<00:49, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:47, 2.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:47, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:46, 2.03MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:57, 1.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:50, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:53, 1.75MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<00:54, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<00:56, 1.66MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<00:54, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<00:54, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<00:53, 1.73MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<00:53, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<00:53, 1.73MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<00:51, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<00:50, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:49, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:47<00:49, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:47<00:47, 1.93MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:47<00:47, 1.90MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:47<00:46, 1.96MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:47<00:46, 1.96MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:45, 1.97MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:45, 1.97MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:44, 2.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:44, 2.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:43, 2.06MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<00:43, 2.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<00:42, 2.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<00:42, 2.10MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<00:41, 2.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<00:41, 2.12MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<00:41, 2.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<00:42, 2.06MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<00:42, 2.06MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<00:41, 2.12MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:49<00:40, 2.15MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:49<00:50, 1.73MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:49<00:44, 1.94MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:49<00:48, 1.79MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:49<00:50, 1.71MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:49<00:50, 1.71MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:49<00:52, 1.64MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<00:50, 1.70MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<00:52, 1.63MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<00:49, 1.73MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<00:49, 1.71MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<00:48, 1.75MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<00:48, 1.76MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<00:46, 1.84MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<00:46, 1.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<00:45, 1.88MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<00:44, 1.90MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<00:43, 1.93MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<00:42, 1.98MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:51<00:42, 1.96MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:51<00:41, 2.04MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:51<00:42, 1.95MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:51<00:40, 2.04MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:51<00:42, 1.95MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:51<00:40, 2.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<00:39, 2.08MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<00:41, 2.00MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<00:38, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:51<00:40, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:38, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:40, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:38, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:40, 2.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:37, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<00:39, 2.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<00:48, 1.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<00:42, 1.90MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<00:45, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:53<00:46, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:53<00:48, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:53<00:46, 1.71MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:53<00:46, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:53<00:45, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:53<00:47, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:53<00:44, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<00:44, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<00:42, 1.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<00:44, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:41, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:42, 1.82MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:40, 1.94MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:40, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:39, 1.97MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:40, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:38, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<00:39, 1.96MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<00:37, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<00:38, 1.98MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:55<00:36, 2.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:55<00:37, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:55<00:35, 2.14MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:55<00:38, 1.96MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:55<00:40, 1.83MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:55<00:42, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:55<00:44, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<00:45, 1.65MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<00:46, 1.59MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:45, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:46, 1.60MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:44, 1.66MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:45, 1.61MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:56<00:43, 1.68MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:56<00:44, 1.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:43, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:43, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:42, 1.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<00:41, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:57<00:42, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:57<00:40, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:57<00:42, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:57<00:40, 1.76MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:57<00:41, 1.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:57<00:40, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:57<00:40, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:57<00:39, 1.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:57<00:39, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:57<00:37, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:38, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:37, 1.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:37, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:36, 1.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:35, 1.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:35, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:35, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:41, 1.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:36, 1.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:59<00:39, 1.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:59<00:41, 1.64MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:59<00:42, 1.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:59<00:42, 1.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:59<00:41, 1.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:59<00:40, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:59<00:41, 1.64MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:59<00:40, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:59<00:39, 1.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<00:38, 1.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<00:37, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:36, 1.81MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:36, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:35, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:35, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:34, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:34, 1.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:34, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:34, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<00:33, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<00:33, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 798M/862M [06:01<00:32, 1.98MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<00:32, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<00:32, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<00:32, 1.98MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<00:31, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:01<00:31, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:01<00:31, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<00:31, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<00:30, 2.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<00:31, 1.98MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:30, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:30, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:31, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:29, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:31, 1.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:35, 1.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<00:32, 1.91MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<00:32, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<00:36, 1.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<00:36, 1.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<00:37, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<00:36, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<00:37, 1.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<00:35, 1.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<00:36, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<00:34, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:33, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:33, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:33, 1.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:31, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:32, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:31, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:31, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:30, 1.92MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:30, 1.91MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:29, 1.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<00:29, 1.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<00:29, 1.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<00:29, 1.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<00:28, 1.97MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<00:28, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:05<00:28, 2.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:05<00:28, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:05<00:27, 2.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:05<00:27, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:05<00:26, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:27, 2.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:26, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:27, 2.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:26, 2.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<00:25, 2.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<00:26, 2.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<00:26, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<00:26, 2.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:06<00:25, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:26, 2.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:26, 2.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:26, 2.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:25, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:25, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<00:24, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<00:24, 2.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<00:24, 2.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<00:23, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:07<00:23, 2.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<00:23, 2.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<00:22, 2.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<00:22, 2.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:21, 2.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:22, 2.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:21, 2.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:22, 2.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:21, 2.35MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:21, 2.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:20, 2.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<00:21, 2.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<00:20, 2.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:09<00:20, 2.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:09<00:19, 2.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:09<00:31, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:09<00:25, 1.87MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:09<00:26, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:09<00:26, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:09<00:26, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:09<00:25, 1.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:10<00:24, 1.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:24, 1.92MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:22, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:23, 1.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:22, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:22, 2.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:21, 2.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:20, 2.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:20, 2.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<00:20, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:11<00:20, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:11<00:19, 2.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:11<00:19, 2.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:11<00:19, 2.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:11<00:19, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:11<00:18, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:11<00:18, 2.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:11<00:17, 2.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:11<00:17, 2.45MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:17, 2.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:17, 2.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:17, 2.45MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:16, 2.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:16, 2.45MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:16, 2.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:17, 2.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:16, 2.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:16, 2.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:16, 2.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:13<00:18, 2.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:13<00:18, 2.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:13<00:20, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:13<00:19, 1.97MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:13<00:20, 1.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:13<00:19, 1.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:13<00:20, 1.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:13<00:19, 2.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:13<00:19, 1.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<00:18, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<00:19, 1.96MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:17, 2.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:18, 2.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:17, 2.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:17, 2.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:16, 2.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:17, 2.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:15, 2.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:16, 2.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:15, 2.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:15<00:15, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:15<00:14, 2.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:15<00:15, 2.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:15<00:14, 2.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:15<00:15, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:15<00:14, 2.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:15<00:14, 2.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:15<00:13, 2.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:15<00:13, 2.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:13, 2.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:13, 2.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:13, 2.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:16<00:12, 2.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<00:13, 2.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<00:12, 2.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<00:12, 2.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:12, 2.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:13, 2.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:12, 2.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:17<00:12, 2.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:17<00:12, 2.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:17<00:12, 2.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:17<00:11, 2.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:17<00:12, 2.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:17<00:11, 2.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:17<00:12, 2.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:17<00:11, 2.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:17<00:12, 2.39MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:17<00:11, 2.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<00:11, 2.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<00:11, 2.51MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:10, 2.59MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:10, 2.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:10, 2.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:10, 2.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<00:09, 2.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<00:10, 2.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<00:09, 2.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:19<00:10, 2.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:19<00:09, 2.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:19<01:03, 399kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:19<00:48, 521kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:19<00:36, 685kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:19<00:27, 884kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:21, 1.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:17, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:14, 1.61MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:12, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:10, 2.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:09, 2.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:20<00:08, 2.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:20<00:08, 2.58MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:21<01:38, 215kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:21<01:10, 294kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:21<00:50, 403kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:21<00:36, 548kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:21<00:27, 726kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:20, 942kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:16, 1.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:12, 1.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:10, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:09, 1.95MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:08, 2.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:22<00:06, 2.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:22<00:06, 2.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:23<00:36, 462kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:23<00:29, 567kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:23<00:21, 749kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:23<00:16, 977kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:23<00:12, 1.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:09, 1.59MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:07, 1.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<00:06, 2.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<00:05, 2.49MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:24<00:04, 2.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:24<00:04, 3.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:25<00:13, 988kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:25<00:11, 1.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:25<00:08, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:25<00:06, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:25<00:05, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:26<00:04, 2.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:26<00:03, 2.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:02, 3.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:27<00:10, 801kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:27<00:10, 844kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:27<00:07, 1.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:27<00:05, 1.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:27<00:03, 1.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:28<00:02, 2.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:01, 2.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:01, 3.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:29<00:16, 280kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:29<00:12, 364kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:29<00:07, 501kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:29<00:04, 693kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:02, 954kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 1.30MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:30<00:00, 1.70MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:31<00:01, 355kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:31<00:01, 419kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:31<00:00, 567kB/s].vector_cache/glove.6B.zip: 862MB [06:31, 2.20MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<31:17:06,  3.55it/s]  0%|          | 862/400000 [00:00<21:51:22,  5.07it/s]  0%|          | 1707/400000 [00:00<15:16:15,  7.24it/s]  1%|          | 2547/400000 [00:00<10:40:15, 10.35it/s]  1%|          | 3398/400000 [00:00<7:27:27, 14.77it/s]   1%|          | 4249/400000 [00:00<5:12:46, 21.09it/s]  1%|▏         | 5084/400000 [00:00<3:38:43, 30.09it/s]  1%|▏         | 5910/400000 [00:00<2:33:01, 42.92it/s]  2%|▏         | 6694/400000 [00:01<1:47:09, 61.17it/s]  2%|▏         | 7510/400000 [00:01<1:15:05, 87.11it/s]  2%|▏         | 8367/400000 [00:01<52:40, 123.91it/s]   2%|▏         | 9224/400000 [00:01<37:01, 175.92it/s]  3%|▎         | 10057/400000 [00:01<26:05, 249.05it/s]  3%|▎         | 10883/400000 [00:01<18:28, 351.06it/s]  3%|▎         | 11725/400000 [00:01<13:08, 492.70it/s]  3%|▎         | 12570/400000 [00:01<09:24, 686.69it/s]  3%|▎         | 13417/400000 [00:01<06:47, 948.05it/s]  4%|▎         | 14252/400000 [00:01<04:59, 1288.11it/s]  4%|▍         | 15074/400000 [00:02<03:43, 1723.97it/s]  4%|▍         | 15922/400000 [00:02<02:49, 2265.36it/s]  4%|▍         | 16769/400000 [00:02<02:11, 2903.36it/s]  4%|▍         | 17626/400000 [00:02<01:45, 3621.49it/s]  5%|▍         | 18475/400000 [00:02<01:27, 4373.35it/s]  5%|▍         | 19319/400000 [00:02<01:14, 5088.78it/s]  5%|▌         | 20168/400000 [00:02<01:05, 5783.79it/s]  5%|▌         | 21030/400000 [00:02<00:59, 6416.19it/s]  5%|▌         | 21877/400000 [00:02<00:55, 6872.19it/s]  6%|▌         | 22717/400000 [00:03<00:52, 7220.05it/s]  6%|▌         | 23570/400000 [00:03<00:49, 7567.21it/s]  6%|▌         | 24437/400000 [00:03<00:47, 7865.62it/s]  6%|▋         | 25287/400000 [00:03<00:46, 8043.39it/s]  7%|▋         | 26143/400000 [00:03<00:45, 8189.20it/s]  7%|▋         | 27011/400000 [00:03<00:44, 8329.44it/s]  7%|▋         | 27880/400000 [00:03<00:44, 8431.65it/s]  7%|▋         | 28740/400000 [00:03<00:44, 8340.52it/s]  7%|▋         | 29586/400000 [00:03<00:45, 8198.45it/s]  8%|▊         | 30415/400000 [00:03<00:46, 7921.84it/s]  8%|▊         | 31216/400000 [00:04<00:46, 7894.31it/s]  8%|▊         | 32032/400000 [00:04<00:46, 7969.56it/s]  8%|▊         | 32836/400000 [00:04<00:45, 7990.27it/s]  8%|▊         | 33663/400000 [00:04<00:45, 8071.65it/s]  9%|▊         | 34537/400000 [00:04<00:44, 8259.55it/s]  9%|▉         | 35405/400000 [00:04<00:43, 8379.68it/s]  9%|▉         | 36265/400000 [00:04<00:43, 8442.83it/s]  9%|▉         | 37134/400000 [00:04<00:42, 8513.70it/s] 10%|▉         | 38000/400000 [00:04<00:42, 8552.78it/s] 10%|▉         | 38857/400000 [00:04<00:42, 8423.94it/s] 10%|▉         | 39701/400000 [00:05<00:42, 8408.35it/s] 10%|█         | 40543/400000 [00:05<00:44, 8138.69it/s] 10%|█         | 41365/400000 [00:05<00:43, 8162.35it/s] 11%|█         | 42229/400000 [00:05<00:43, 8299.47it/s] 11%|█         | 43061/400000 [00:05<00:43, 8241.52it/s] 11%|█         | 43925/400000 [00:05<00:42, 8356.61it/s] 11%|█         | 44774/400000 [00:05<00:42, 8395.73it/s] 11%|█▏        | 45615/400000 [00:05<00:42, 8390.14it/s] 12%|█▏        | 46455/400000 [00:05<00:42, 8286.78it/s] 12%|█▏        | 47285/400000 [00:05<00:43, 8147.05it/s] 12%|█▏        | 48106/400000 [00:06<00:43, 8165.65it/s] 12%|█▏        | 48925/400000 [00:06<00:42, 8170.39it/s] 12%|█▏        | 49743/400000 [00:06<00:42, 8152.73it/s] 13%|█▎        | 50594/400000 [00:06<00:42, 8255.50it/s] 13%|█▎        | 51421/400000 [00:06<00:42, 8186.09it/s] 13%|█▎        | 52274/400000 [00:06<00:41, 8285.58it/s] 13%|█▎        | 53112/400000 [00:06<00:41, 8312.00it/s] 13%|█▎        | 53972/400000 [00:06<00:41, 8395.92it/s] 14%|█▎        | 54813/400000 [00:06<00:41, 8248.22it/s] 14%|█▍        | 55639/400000 [00:06<00:41, 8226.95it/s] 14%|█▍        | 56463/400000 [00:07<00:41, 8215.61it/s] 14%|█▍        | 57286/400000 [00:07<00:41, 8204.76it/s] 15%|█▍        | 58140/400000 [00:07<00:41, 8301.23it/s] 15%|█▍        | 58975/400000 [00:07<00:41, 8313.75it/s] 15%|█▍        | 59838/400000 [00:07<00:40, 8405.17it/s] 15%|█▌        | 60680/400000 [00:07<00:41, 8151.97it/s] 15%|█▌        | 61510/400000 [00:07<00:41, 8194.70it/s] 16%|█▌        | 62344/400000 [00:07<00:40, 8235.90it/s] 16%|█▌        | 63216/400000 [00:07<00:40, 8374.02it/s] 16%|█▌        | 64055/400000 [00:07<00:41, 8186.51it/s] 16%|█▌        | 64886/400000 [00:08<00:40, 8220.91it/s] 16%|█▋        | 65739/400000 [00:08<00:40, 8309.04it/s] 17%|█▋        | 66605/400000 [00:08<00:39, 8409.88it/s] 17%|█▋        | 67448/400000 [00:08<00:39, 8359.81it/s] 17%|█▋        | 68290/400000 [00:08<00:39, 8376.57it/s] 17%|█▋        | 69129/400000 [00:08<00:40, 8225.57it/s] 17%|█▋        | 69953/400000 [00:08<00:41, 7997.54it/s] 18%|█▊        | 70802/400000 [00:08<00:40, 8138.21it/s] 18%|█▊        | 71656/400000 [00:08<00:39, 8253.92it/s] 18%|█▊        | 72510/400000 [00:09<00:39, 8336.21it/s] 18%|█▊        | 73349/400000 [00:09<00:39, 8352.12it/s] 19%|█▊        | 74204/400000 [00:09<00:38, 8408.78it/s] 19%|█▉        | 75061/400000 [00:09<00:38, 8456.34it/s] 19%|█▉        | 75908/400000 [00:09<00:38, 8451.81it/s] 19%|█▉        | 76754/400000 [00:09<00:38, 8335.02it/s] 19%|█▉        | 77591/400000 [00:09<00:38, 8344.77it/s] 20%|█▉        | 78431/400000 [00:09<00:38, 8360.80it/s] 20%|█▉        | 79276/400000 [00:09<00:38, 8387.17it/s] 20%|██        | 80124/400000 [00:09<00:38, 8413.44it/s] 20%|██        | 80966/400000 [00:10<00:38, 8390.18it/s] 20%|██        | 81806/400000 [00:10<00:38, 8174.96it/s] 21%|██        | 82666/400000 [00:10<00:38, 8297.42it/s] 21%|██        | 83525/400000 [00:10<00:37, 8380.48it/s] 21%|██        | 84398/400000 [00:10<00:37, 8482.07it/s] 21%|██▏       | 85254/400000 [00:10<00:37, 8503.18it/s] 22%|██▏       | 86114/400000 [00:10<00:36, 8531.10it/s] 22%|██▏       | 86968/400000 [00:10<00:36, 8513.84it/s] 22%|██▏       | 87842/400000 [00:10<00:36, 8578.37it/s] 22%|██▏       | 88705/400000 [00:10<00:36, 8593.41it/s] 22%|██▏       | 89565/400000 [00:11<00:36, 8591.03it/s] 23%|██▎       | 90425/400000 [00:11<00:36, 8418.82it/s] 23%|██▎       | 91268/400000 [00:11<00:36, 8408.33it/s] 23%|██▎       | 92110/400000 [00:11<00:37, 8302.14it/s] 23%|██▎       | 92951/400000 [00:11<00:36, 8333.24it/s] 23%|██▎       | 93809/400000 [00:11<00:36, 8404.76it/s] 24%|██▎       | 94655/400000 [00:11<00:36, 8420.66it/s] 24%|██▍       | 95500/400000 [00:11<00:36, 8428.33it/s] 24%|██▍       | 96360/400000 [00:11<00:35, 8478.04it/s] 24%|██▍       | 97209/400000 [00:11<00:36, 8403.43it/s] 25%|██▍       | 98050/400000 [00:12<00:36, 8299.89it/s] 25%|██▍       | 98881/400000 [00:12<00:36, 8287.50it/s] 25%|██▍       | 99711/400000 [00:12<00:36, 8196.73it/s] 25%|██▌       | 100532/400000 [00:12<00:37, 8002.48it/s] 25%|██▌       | 101374/400000 [00:12<00:36, 8123.19it/s] 26%|██▌       | 102203/400000 [00:12<00:36, 8170.12it/s] 26%|██▌       | 103022/400000 [00:12<00:36, 8161.36it/s] 26%|██▌       | 103859/400000 [00:12<00:36, 8221.09it/s] 26%|██▌       | 104682/400000 [00:12<00:36, 8197.78it/s] 26%|██▋       | 105530/400000 [00:12<00:35, 8278.22it/s] 27%|██▋       | 106390/400000 [00:13<00:35, 8372.16it/s] 27%|██▋       | 107228/400000 [00:13<00:35, 8336.32it/s] 27%|██▋       | 108085/400000 [00:13<00:34, 8404.90it/s] 27%|██▋       | 108926/400000 [00:13<00:35, 8295.93it/s] 27%|██▋       | 109757/400000 [00:13<00:35, 8213.57it/s] 28%|██▊       | 110600/400000 [00:13<00:34, 8275.20it/s] 28%|██▊       | 111448/400000 [00:13<00:34, 8335.46it/s] 28%|██▊       | 112307/400000 [00:13<00:34, 8409.67it/s] 28%|██▊       | 113163/400000 [00:13<00:33, 8452.27it/s] 29%|██▊       | 114033/400000 [00:13<00:33, 8523.98it/s] 29%|██▊       | 114886/400000 [00:14<00:33, 8492.76it/s] 29%|██▉       | 115751/400000 [00:14<00:33, 8539.16it/s] 29%|██▉       | 116612/400000 [00:14<00:33, 8556.55it/s] 29%|██▉       | 117468/400000 [00:14<00:33, 8408.83it/s] 30%|██▉       | 118319/400000 [00:14<00:33, 8436.36it/s] 30%|██▉       | 119175/400000 [00:14<00:33, 8470.49it/s] 30%|███       | 120023/400000 [00:14<00:33, 8440.28it/s] 30%|███       | 120880/400000 [00:14<00:32, 8477.01it/s] 30%|███       | 121728/400000 [00:14<00:32, 8469.46it/s] 31%|███       | 122600/400000 [00:14<00:32, 8542.10it/s] 31%|███       | 123455/400000 [00:15<00:32, 8533.96it/s] 31%|███       | 124318/400000 [00:15<00:32, 8561.99it/s] 31%|███▏      | 125175/400000 [00:15<00:32, 8531.57it/s] 32%|███▏      | 126029/400000 [00:15<00:32, 8529.91it/s] 32%|███▏      | 126905/400000 [00:15<00:31, 8595.58it/s] 32%|███▏      | 127770/400000 [00:15<00:31, 8610.56it/s] 32%|███▏      | 128632/400000 [00:15<00:32, 8338.34it/s] 32%|███▏      | 129468/400000 [00:15<00:33, 8196.80it/s] 33%|███▎      | 130290/400000 [00:15<00:34, 7898.05it/s] 33%|███▎      | 131136/400000 [00:16<00:33, 8057.83it/s] 33%|███▎      | 131973/400000 [00:16<00:32, 8147.48it/s] 33%|███▎      | 132850/400000 [00:16<00:32, 8322.31it/s] 33%|███▎      | 133727/400000 [00:16<00:31, 8449.09it/s] 34%|███▎      | 134575/400000 [00:16<00:31, 8442.57it/s] 34%|███▍      | 135428/400000 [00:16<00:31, 8466.98it/s] 34%|███▍      | 136292/400000 [00:16<00:30, 8517.26it/s] 34%|███▍      | 137165/400000 [00:16<00:30, 8578.56it/s] 35%|███▍      | 138041/400000 [00:16<00:30, 8630.04it/s] 35%|███▍      | 138905/400000 [00:16<00:31, 8402.52it/s] 35%|███▍      | 139747/400000 [00:17<00:30, 8399.32it/s] 35%|███▌      | 140589/400000 [00:17<00:30, 8392.11it/s] 35%|███▌      | 141444/400000 [00:17<00:30, 8436.19it/s] 36%|███▌      | 142321/400000 [00:17<00:30, 8531.84it/s] 36%|███▌      | 143178/400000 [00:17<00:30, 8541.37it/s] 36%|███▌      | 144041/400000 [00:17<00:29, 8566.47it/s] 36%|███▌      | 144913/400000 [00:17<00:29, 8611.31it/s] 36%|███▋      | 145785/400000 [00:17<00:29, 8641.26it/s] 37%|███▋      | 146664/400000 [00:17<00:29, 8684.46it/s] 37%|███▋      | 147533/400000 [00:17<00:29, 8484.52it/s] 37%|███▋      | 148403/400000 [00:18<00:29, 8546.02it/s] 37%|███▋      | 149259/400000 [00:18<00:29, 8506.19it/s] 38%|███▊      | 150131/400000 [00:18<00:29, 8568.13it/s] 38%|███▊      | 150991/400000 [00:18<00:29, 8576.17it/s] 38%|███▊      | 151850/400000 [00:18<00:29, 8532.24it/s] 38%|███▊      | 152722/400000 [00:18<00:28, 8587.34it/s] 38%|███▊      | 153595/400000 [00:18<00:28, 8627.24it/s] 39%|███▊      | 154469/400000 [00:18<00:28, 8659.04it/s] 39%|███▉      | 155342/400000 [00:18<00:28, 8677.70it/s] 39%|███▉      | 156210/400000 [00:18<00:28, 8620.58it/s] 39%|███▉      | 157073/400000 [00:19<00:28, 8592.87it/s] 39%|███▉      | 157933/400000 [00:19<00:28, 8384.40it/s] 40%|███▉      | 158773/400000 [00:19<00:28, 8371.28it/s] 40%|███▉      | 159611/400000 [00:19<00:29, 8207.78it/s] 40%|████      | 160470/400000 [00:19<00:28, 8317.59it/s] 40%|████      | 161318/400000 [00:19<00:28, 8362.92it/s] 41%|████      | 162181/400000 [00:19<00:28, 8440.24it/s] 41%|████      | 163037/400000 [00:19<00:27, 8474.57it/s] 41%|████      | 163894/400000 [00:19<00:27, 8502.14it/s] 41%|████      | 164745/400000 [00:19<00:28, 8382.53it/s] 41%|████▏     | 165588/400000 [00:20<00:27, 8394.31it/s] 42%|████▏     | 166441/400000 [00:20<00:27, 8433.30it/s] 42%|████▏     | 167285/400000 [00:20<00:27, 8434.17it/s] 42%|████▏     | 168129/400000 [00:20<00:27, 8421.85it/s] 42%|████▏     | 168972/400000 [00:20<00:27, 8372.04it/s] 42%|████▏     | 169810/400000 [00:20<00:27, 8258.16it/s] 43%|████▎     | 170637/400000 [00:20<00:28, 8010.40it/s] 43%|████▎     | 171441/400000 [00:20<00:28, 8004.84it/s] 43%|████▎     | 172299/400000 [00:20<00:27, 8169.12it/s] 43%|████▎     | 173118/400000 [00:20<00:27, 8163.89it/s] 43%|████▎     | 173976/400000 [00:21<00:27, 8284.36it/s] 44%|████▎     | 174835/400000 [00:21<00:26, 8372.37it/s] 44%|████▍     | 175703/400000 [00:21<00:26, 8460.52it/s] 44%|████▍     | 176575/400000 [00:21<00:26, 8534.67it/s] 44%|████▍     | 177434/400000 [00:21<00:26, 8549.91it/s] 45%|████▍     | 178298/400000 [00:21<00:25, 8576.37it/s] 45%|████▍     | 179175/400000 [00:21<00:25, 8630.96it/s] 45%|████▌     | 180039/400000 [00:21<00:25, 8503.02it/s] 45%|████▌     | 180903/400000 [00:21<00:25, 8541.25it/s] 45%|████▌     | 181758/400000 [00:21<00:25, 8480.59it/s] 46%|████▌     | 182622/400000 [00:22<00:25, 8526.19it/s] 46%|████▌     | 183476/400000 [00:22<00:25, 8478.63it/s] 46%|████▌     | 184342/400000 [00:22<00:25, 8529.51it/s] 46%|████▋     | 185197/400000 [00:22<00:25, 8535.28it/s] 47%|████▋     | 186051/400000 [00:22<00:25, 8461.07it/s] 47%|████▋     | 186921/400000 [00:22<00:24, 8529.32it/s] 47%|████▋     | 187800/400000 [00:22<00:24, 8605.06it/s] 47%|████▋     | 188680/400000 [00:22<00:24, 8660.73it/s] 47%|████▋     | 189556/400000 [00:22<00:24, 8687.74it/s] 48%|████▊     | 190426/400000 [00:22<00:24, 8661.18it/s] 48%|████▊     | 191293/400000 [00:23<00:24, 8661.32it/s] 48%|████▊     | 192160/400000 [00:23<00:24, 8604.78it/s] 48%|████▊     | 193021/400000 [00:23<00:24, 8450.02it/s] 48%|████▊     | 193867/400000 [00:23<00:24, 8420.39it/s] 49%|████▊     | 194710/400000 [00:23<00:24, 8394.55it/s] 49%|████▉     | 195578/400000 [00:23<00:24, 8477.27it/s] 49%|████▉     | 196454/400000 [00:23<00:23, 8559.84it/s] 49%|████▉     | 197327/400000 [00:23<00:23, 8607.65it/s] 50%|████▉     | 198209/400000 [00:23<00:23, 8668.04it/s] 50%|████▉     | 199077/400000 [00:24<00:23, 8590.76it/s] 50%|████▉     | 199946/400000 [00:24<00:23, 8618.28it/s] 50%|█████     | 200815/400000 [00:24<00:23, 8638.20it/s] 50%|█████     | 201680/400000 [00:24<00:23, 8596.77it/s] 51%|█████     | 202550/400000 [00:24<00:22, 8627.11it/s] 51%|█████     | 203413/400000 [00:24<00:22, 8550.97it/s] 51%|█████     | 204269/400000 [00:24<00:23, 8451.01it/s] 51%|█████▏    | 205129/400000 [00:24<00:22, 8492.79it/s] 51%|█████▏    | 205979/400000 [00:24<00:22, 8494.28it/s] 52%|█████▏    | 206841/400000 [00:24<00:22, 8528.87it/s] 52%|█████▏    | 207695/400000 [00:25<00:22, 8497.25it/s] 52%|█████▏    | 208550/400000 [00:25<00:22, 8512.40it/s] 52%|█████▏    | 209410/400000 [00:25<00:22, 8537.05it/s] 53%|█████▎    | 210264/400000 [00:25<00:22, 8523.38it/s] 53%|█████▎    | 211142/400000 [00:25<00:21, 8596.66it/s] 53%|█████▎    | 212002/400000 [00:25<00:21, 8585.33it/s] 53%|█████▎    | 212864/400000 [00:25<00:21, 8595.48it/s] 53%|█████▎    | 213724/400000 [00:25<00:21, 8552.94it/s] 54%|█████▎    | 214592/400000 [00:25<00:21, 8590.50it/s] 54%|█████▍    | 215453/400000 [00:25<00:21, 8594.82it/s] 54%|█████▍    | 216313/400000 [00:26<00:22, 8259.04it/s] 54%|█████▍    | 217142/400000 [00:26<00:22, 8160.19it/s] 54%|█████▍    | 217961/400000 [00:26<00:22, 8097.52it/s] 55%|█████▍    | 218773/400000 [00:26<00:22, 7995.52it/s] 55%|█████▍    | 219607/400000 [00:26<00:22, 8093.62it/s] 55%|█████▌    | 220447/400000 [00:26<00:21, 8181.83it/s] 55%|█████▌    | 221285/400000 [00:26<00:21, 8238.75it/s] 56%|█████▌    | 222126/400000 [00:26<00:21, 8287.67it/s] 56%|█████▌    | 222989/400000 [00:26<00:21, 8386.08it/s] 56%|█████▌    | 223829/400000 [00:26<00:21, 8332.99it/s] 56%|█████▌    | 224663/400000 [00:27<00:21, 8304.07it/s] 56%|█████▋    | 225516/400000 [00:27<00:20, 8370.36it/s] 57%|█████▋    | 226354/400000 [00:27<00:20, 8358.72it/s] 57%|█████▋    | 227201/400000 [00:27<00:20, 8390.15it/s] 57%|█████▋    | 228057/400000 [00:27<00:20, 8439.70it/s] 57%|█████▋    | 228902/400000 [00:27<00:20, 8162.80it/s] 57%|█████▋    | 229721/400000 [00:27<00:21, 7785.39it/s] 58%|█████▊    | 230505/400000 [00:27<00:21, 7730.63it/s] 58%|█████▊    | 231293/400000 [00:27<00:21, 7774.30it/s] 58%|█████▊    | 232125/400000 [00:27<00:21, 7930.09it/s] 58%|█████▊    | 232967/400000 [00:28<00:20, 8069.37it/s] 58%|█████▊    | 233818/400000 [00:28<00:20, 8194.58it/s] 59%|█████▊    | 234674/400000 [00:28<00:19, 8300.14it/s] 59%|█████▉    | 235506/400000 [00:28<00:19, 8235.48it/s] 59%|█████▉    | 236331/400000 [00:28<00:20, 8147.93it/s] 59%|█████▉    | 237147/400000 [00:28<00:20, 7917.56it/s] 60%|█████▉    | 238006/400000 [00:28<00:19, 8106.72it/s] 60%|█████▉    | 238820/400000 [00:28<00:20, 8018.99it/s] 60%|█████▉    | 239625/400000 [00:28<00:20, 8002.45it/s] 60%|██████    | 240453/400000 [00:29<00:19, 8081.56it/s] 60%|██████    | 241263/400000 [00:29<00:20, 7870.44it/s] 61%|██████    | 242113/400000 [00:29<00:19, 8046.59it/s] 61%|██████    | 242946/400000 [00:29<00:19, 8129.32it/s] 61%|██████    | 243770/400000 [00:29<00:19, 8160.35it/s] 61%|██████    | 244626/400000 [00:29<00:18, 8274.76it/s] 61%|██████▏   | 245455/400000 [00:29<00:18, 8224.36it/s] 62%|██████▏   | 246280/400000 [00:29<00:18, 8227.59it/s] 62%|██████▏   | 247147/400000 [00:29<00:18, 8354.24it/s] 62%|██████▏   | 248021/400000 [00:29<00:17, 8464.78it/s] 62%|██████▏   | 248890/400000 [00:30<00:17, 8529.87it/s] 62%|██████▏   | 249744/400000 [00:30<00:17, 8524.39it/s] 63%|██████▎   | 250625/400000 [00:30<00:17, 8607.08it/s] 63%|██████▎   | 251487/400000 [00:30<00:17, 8573.64it/s] 63%|██████▎   | 252360/400000 [00:30<00:17, 8617.29it/s] 63%|██████▎   | 253223/400000 [00:30<00:17, 8619.22it/s] 64%|██████▎   | 254086/400000 [00:30<00:16, 8591.15it/s] 64%|██████▎   | 254946/400000 [00:30<00:17, 8374.84it/s] 64%|██████▍   | 255785/400000 [00:30<00:17, 8238.14it/s] 64%|██████▍   | 256611/400000 [00:30<00:17, 8185.38it/s] 64%|██████▍   | 257431/400000 [00:31<00:17, 8079.70it/s] 65%|██████▍   | 258282/400000 [00:31<00:17, 8201.53it/s] 65%|██████▍   | 259158/400000 [00:31<00:16, 8359.62it/s] 65%|██████▌   | 260036/400000 [00:31<00:16, 8479.05it/s] 65%|██████▌   | 260914/400000 [00:31<00:16, 8564.68it/s] 65%|██████▌   | 261793/400000 [00:31<00:16, 8629.11it/s] 66%|██████▌   | 262657/400000 [00:31<00:16, 8396.02it/s] 66%|██████▌   | 263513/400000 [00:31<00:16, 8443.19it/s] 66%|██████▌   | 264368/400000 [00:31<00:16, 8473.67it/s] 66%|██████▋   | 265217/400000 [00:31<00:15, 8425.37it/s] 67%|██████▋   | 266075/400000 [00:32<00:15, 8470.81it/s] 67%|██████▋   | 266923/400000 [00:32<00:15, 8463.39it/s] 67%|██████▋   | 267777/400000 [00:32<00:15, 8484.78it/s] 67%|██████▋   | 268648/400000 [00:32<00:15, 8548.64it/s] 67%|██████▋   | 269504/400000 [00:32<00:15, 8327.49it/s] 68%|██████▊   | 270339/400000 [00:32<00:15, 8308.65it/s] 68%|██████▊   | 271191/400000 [00:32<00:15, 8368.69it/s] 68%|██████▊   | 272060/400000 [00:32<00:15, 8460.90it/s] 68%|██████▊   | 272907/400000 [00:32<00:15, 8427.39it/s] 68%|██████▊   | 273751/400000 [00:32<00:15, 8383.26it/s] 69%|██████▊   | 274590/400000 [00:33<00:14, 8368.75it/s] 69%|██████▉   | 275457/400000 [00:33<00:14, 8454.60it/s] 69%|██████▉   | 276303/400000 [00:33<00:14, 8279.09it/s] 69%|██████▉   | 277163/400000 [00:33<00:14, 8372.25it/s] 70%|██████▉   | 278002/400000 [00:33<00:14, 8195.61it/s] 70%|██████▉   | 278824/400000 [00:33<00:14, 8100.40it/s] 70%|██████▉   | 279656/400000 [00:33<00:14, 8163.17it/s] 70%|███████   | 280475/400000 [00:33<00:14, 8170.76it/s] 70%|███████   | 281342/400000 [00:33<00:14, 8312.09it/s] 71%|███████   | 282180/400000 [00:33<00:14, 8332.27it/s] 71%|███████   | 283042/400000 [00:34<00:13, 8416.45it/s] 71%|███████   | 283906/400000 [00:34<00:13, 8479.70it/s] 71%|███████   | 284769/400000 [00:34<00:13, 8523.05it/s] 71%|███████▏  | 285642/400000 [00:34<00:13, 8583.72it/s] 72%|███████▏  | 286501/400000 [00:34<00:13, 8537.84it/s] 72%|███████▏  | 287356/400000 [00:34<00:13, 8528.92it/s] 72%|███████▏  | 288210/400000 [00:34<00:13, 8466.72it/s] 72%|███████▏  | 289064/400000 [00:34<00:13, 8486.04it/s] 72%|███████▏  | 289923/400000 [00:34<00:12, 8514.63it/s] 73%|███████▎  | 290775/400000 [00:34<00:13, 8359.79it/s] 73%|███████▎  | 291615/400000 [00:35<00:12, 8369.20it/s] 73%|███████▎  | 292461/400000 [00:35<00:12, 8393.95it/s] 73%|███████▎  | 293316/400000 [00:35<00:12, 8437.69it/s] 74%|███████▎  | 294181/400000 [00:35<00:12, 8499.91it/s] 74%|███████▍  | 295036/400000 [00:35<00:12, 8512.52it/s] 74%|███████▍  | 295888/400000 [00:35<00:12, 8480.64it/s] 74%|███████▍  | 296737/400000 [00:35<00:12, 8413.68it/s] 74%|███████▍  | 297579/400000 [00:35<00:12, 8334.12it/s] 75%|███████▍  | 298413/400000 [00:35<00:12, 8266.02it/s] 75%|███████▍  | 299285/400000 [00:36<00:11, 8395.85it/s] 75%|███████▌  | 300158/400000 [00:36<00:11, 8490.78it/s] 75%|███████▌  | 301008/400000 [00:36<00:11, 8444.26it/s] 75%|███████▌  | 301885/400000 [00:36<00:11, 8537.43it/s] 76%|███████▌  | 302758/400000 [00:36<00:11, 8593.59it/s] 76%|███████▌  | 303629/400000 [00:36<00:11, 8625.82it/s] 76%|███████▌  | 304501/400000 [00:36<00:11, 8652.35it/s] 76%|███████▋  | 305367/400000 [00:36<00:10, 8616.69it/s] 77%|███████▋  | 306238/400000 [00:36<00:10, 8641.86it/s] 77%|███████▋  | 307106/400000 [00:36<00:10, 8653.22it/s] 77%|███████▋  | 307984/400000 [00:37<00:10, 8690.84it/s] 77%|███████▋  | 308859/400000 [00:37<00:10, 8703.39it/s] 77%|███████▋  | 309730/400000 [00:37<00:10, 8622.02it/s] 78%|███████▊  | 310593/400000 [00:37<00:10, 8570.12it/s] 78%|███████▊  | 311451/400000 [00:37<00:10, 8551.84it/s] 78%|███████▊  | 312307/400000 [00:37<00:10, 8497.28it/s] 78%|███████▊  | 313157/400000 [00:37<00:10, 8447.87it/s] 79%|███████▊  | 314002/400000 [00:37<00:10, 8347.60it/s] 79%|███████▊  | 314844/400000 [00:37<00:10, 8367.65it/s] 79%|███████▉  | 315705/400000 [00:37<00:09, 8438.76it/s] 79%|███████▉  | 316577/400000 [00:38<00:09, 8519.77it/s] 79%|███████▉  | 317432/400000 [00:38<00:09, 8527.93it/s] 80%|███████▉  | 318286/400000 [00:38<00:09, 8490.16it/s] 80%|███████▉  | 319153/400000 [00:38<00:09, 8542.90it/s] 80%|████████  | 320008/400000 [00:38<00:09, 8477.72it/s] 80%|████████  | 320861/400000 [00:38<00:09, 8492.98it/s] 80%|████████  | 321728/400000 [00:38<00:09, 8544.18it/s] 81%|████████  | 322588/400000 [00:38<00:09, 8559.80it/s] 81%|████████  | 323465/400000 [00:38<00:08, 8620.07it/s] 81%|████████  | 324342/400000 [00:38<00:08, 8662.10it/s] 81%|████████▏ | 325209/400000 [00:39<00:08, 8651.32it/s] 82%|████████▏ | 326075/400000 [00:39<00:08, 8600.52it/s] 82%|████████▏ | 326936/400000 [00:39<00:08, 8579.81it/s] 82%|████████▏ | 327795/400000 [00:39<00:08, 8581.95it/s] 82%|████████▏ | 328654/400000 [00:39<00:08, 8575.83it/s] 82%|████████▏ | 329512/400000 [00:39<00:08, 8514.24it/s] 83%|████████▎ | 330364/400000 [00:39<00:08, 8506.18it/s] 83%|████████▎ | 331222/400000 [00:39<00:08, 8527.79it/s] 83%|████████▎ | 332075/400000 [00:39<00:08, 8354.74it/s] 83%|████████▎ | 332912/400000 [00:39<00:08, 8317.08it/s] 83%|████████▎ | 333753/400000 [00:40<00:07, 8342.14it/s] 84%|████████▎ | 334590/400000 [00:40<00:07, 8350.06it/s] 84%|████████▍ | 335426/400000 [00:40<00:07, 8313.31it/s] 84%|████████▍ | 336300/400000 [00:40<00:07, 8434.49it/s] 84%|████████▍ | 337178/400000 [00:40<00:07, 8534.65it/s] 85%|████████▍ | 338038/400000 [00:40<00:07, 8551.48it/s] 85%|████████▍ | 338912/400000 [00:40<00:07, 8605.29it/s] 85%|████████▍ | 339773/400000 [00:40<00:07, 8586.85it/s] 85%|████████▌ | 340632/400000 [00:40<00:06, 8578.07it/s] 85%|████████▌ | 341491/400000 [00:40<00:06, 8406.20it/s] 86%|████████▌ | 342333/400000 [00:41<00:07, 8132.56it/s] 86%|████████▌ | 343207/400000 [00:41<00:06, 8304.85it/s] 86%|████████▌ | 344041/400000 [00:41<00:06, 8313.34it/s] 86%|████████▌ | 344875/400000 [00:41<00:06, 8167.41it/s] 86%|████████▋ | 345724/400000 [00:41<00:06, 8261.55it/s] 87%|████████▋ | 346574/400000 [00:41<00:06, 8329.64it/s] 87%|████████▋ | 347444/400000 [00:41<00:06, 8437.35it/s] 87%|████████▋ | 348310/400000 [00:41<00:06, 8501.28it/s] 87%|████████▋ | 349186/400000 [00:41<00:05, 8575.27it/s] 88%|████████▊ | 350045/400000 [00:41<00:05, 8491.79it/s] 88%|████████▊ | 350895/400000 [00:42<00:05, 8472.26it/s] 88%|████████▊ | 351743/400000 [00:42<00:05, 8430.86it/s] 88%|████████▊ | 352587/400000 [00:42<00:05, 8377.43it/s] 88%|████████▊ | 353428/400000 [00:42<00:05, 8384.72it/s] 89%|████████▊ | 354282/400000 [00:42<00:05, 8428.86it/s] 89%|████████▉ | 355126/400000 [00:42<00:05, 8250.05it/s] 89%|████████▉ | 355953/400000 [00:42<00:05, 8253.21it/s] 89%|████████▉ | 356810/400000 [00:42<00:05, 8345.44it/s] 89%|████████▉ | 357646/400000 [00:42<00:05, 8197.55it/s] 90%|████████▉ | 358494/400000 [00:42<00:05, 8279.12it/s] 90%|████████▉ | 359340/400000 [00:43<00:04, 8328.02it/s] 90%|█████████ | 360211/400000 [00:43<00:04, 8436.82it/s] 90%|█████████ | 361056/400000 [00:43<00:04, 8286.97it/s] 90%|█████████ | 361886/400000 [00:43<00:04, 8131.49it/s] 91%|█████████ | 362701/400000 [00:43<00:04, 7925.97it/s] 91%|█████████ | 363551/400000 [00:43<00:04, 8089.43it/s] 91%|█████████ | 364424/400000 [00:43<00:04, 8269.66it/s] 91%|█████████▏| 365262/400000 [00:43<00:04, 8301.06it/s] 92%|█████████▏| 366094/400000 [00:43<00:04, 8155.42it/s] 92%|█████████▏| 366941/400000 [00:44<00:04, 8245.45it/s] 92%|█████████▏| 367800/400000 [00:44<00:03, 8344.60it/s] 92%|█████████▏| 368636/400000 [00:44<00:03, 8299.08it/s] 92%|█████████▏| 369467/400000 [00:44<00:03, 8171.76it/s] 93%|█████████▎| 370300/400000 [00:44<00:03, 8216.22it/s] 93%|█████████▎| 371123/400000 [00:44<00:03, 8136.88it/s] 93%|█████████▎| 371979/400000 [00:44<00:03, 8256.72it/s] 93%|█████████▎| 372840/400000 [00:44<00:03, 8359.01it/s] 93%|█████████▎| 373701/400000 [00:44<00:03, 8431.88it/s] 94%|█████████▎| 374546/400000 [00:44<00:03, 8348.74it/s] 94%|█████████▍| 375403/400000 [00:45<00:02, 8411.86it/s] 94%|█████████▍| 376267/400000 [00:45<00:02, 8478.86it/s] 94%|█████████▍| 377142/400000 [00:45<00:02, 8557.94it/s] 94%|█████████▍| 377999/400000 [00:45<00:02, 8550.30it/s] 95%|█████████▍| 378874/400000 [00:45<00:02, 8608.22it/s] 95%|█████████▍| 379745/400000 [00:45<00:02, 8637.32it/s] 95%|█████████▌| 380610/400000 [00:45<00:02, 8481.55it/s] 95%|█████████▌| 381459/400000 [00:45<00:02, 8474.50it/s] 96%|█████████▌| 382312/400000 [00:45<00:02, 8490.68it/s] 96%|█████████▌| 383162/400000 [00:45<00:01, 8451.03it/s] 96%|█████████▌| 384012/400000 [00:46<00:01, 8465.32it/s] 96%|█████████▌| 384859/400000 [00:46<00:01, 8120.93it/s] 96%|█████████▋| 385694/400000 [00:46<00:01, 8186.09it/s] 97%|█████████▋| 386553/400000 [00:46<00:01, 8302.38it/s] 97%|█████████▋| 387414/400000 [00:46<00:01, 8390.18it/s] 97%|█████████▋| 388291/400000 [00:46<00:01, 8500.07it/s] 97%|█████████▋| 389168/400000 [00:46<00:01, 8577.26it/s] 98%|█████████▊| 390045/400000 [00:46<00:01, 8632.01it/s] 98%|█████████▊| 390910/400000 [00:46<00:01, 8602.96it/s] 98%|█████████▊| 391771/400000 [00:46<00:00, 8456.49it/s] 98%|█████████▊| 392618/400000 [00:47<00:00, 8393.01it/s] 98%|█████████▊| 393466/400000 [00:47<00:00, 8417.04it/s] 99%|█████████▊| 394320/400000 [00:47<00:00, 8452.73it/s] 99%|█████████▉| 395166/400000 [00:47<00:00, 8443.80it/s] 99%|█████████▉| 396028/400000 [00:47<00:00, 8495.72it/s] 99%|█████████▉| 396878/400000 [00:47<00:00, 8327.04it/s] 99%|█████████▉| 397739/400000 [00:47<00:00, 8407.98it/s]100%|█████████▉| 398591/400000 [00:47<00:00, 8439.64it/s]100%|█████████▉| 399448/400000 [00:47<00:00, 8477.97it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8342.34it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f3833dadcc0>, <torchtext.data.dataset.TabularDataset object at 0x7f3833dad978>, <torchtext.vocab.Vocab object at 0x7f3833dad8d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 24.78 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 48.43 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 22.36 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 22.36 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.47 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.47 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.32 file/s]2020-06-15 09:38:57.674913: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-15 09:38:57.679176: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-06-15 09:38:57.679343: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561e166902b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-15 09:38:57.679358: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3751936/9912422 [00:00<00:00, 37517159.58it/s]9920512it [00:00, 35083010.41it/s]                             
0it [00:00, ?it/s]32768it [00:00, 498655.58it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 477975.38it/s]1654784it [00:00, 11645531.92it/s]                         
0it [00:00, ?it/s]8192it [00:00, 199561.72it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

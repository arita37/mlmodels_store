
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7efd85abeae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7efd85abeae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7efdf072e510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7efdf072e510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7efe0f95bea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7efe0f95bea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7efd9dad6a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7efd9dad6a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7efd9dad6a60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 32%|███▏      | 3194880/9912422 [00:00<00:00, 31905851.17it/s]9920512it [00:00, 32850150.53it/s]                             
0it [00:00, ?it/s]32768it [00:00, 652343.33it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 469580.30it/s]1654784it [00:00, 12170886.58it/s]                         
0it [00:00, ?it/s]8192it [00:00, 237795.18it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efd9dba26a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efd8598b9b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7efd9dad66a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7efd9dad66a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7efd9dad66a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:21,  1.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:21,  1.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:20,  1.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:20,  1.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:19,  1.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:18,  1.96 MiB/s][A

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
Dl Size...:   9%|▉         | 15/162 [00:00<00:53,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:37,  3.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:37,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:37,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:36,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:36,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:36,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:35,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:35,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:35,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:24,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:24,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:24,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:16,  7.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:16,  7.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:16,  7.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:11, 10.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:11, 10.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11, 10.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11, 10.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 13.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 13.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:08, 13.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 13.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 13.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 13.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 13.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 13.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 13.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 18.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 18.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 18.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 18.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 18.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 18.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 24.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 24.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 24.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 24.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 24.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 24.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 24.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 24.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 30.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 30.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 30.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 30.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 30.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 30.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 37.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 37.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 44.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 50.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 56.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 61.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 61.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 61.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 61.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 61.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 61.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 61.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 61.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 61.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 65.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 68.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 68.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 68.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 70.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 72.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.62s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.29 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.63s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 72.29 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.63s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.63s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.95 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.64s/ url]
0 examples [00:00, ? examples/s]2020-08-03 18:11:06.934278: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-03 18:11:07.041907: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-08-03 18:11:07.042098: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c5537928e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-03 18:11:07.042130: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.21 examples/s]98 examples [00:00,  3.15 examples/s]197 examples [00:00,  4.49 examples/s]291 examples [00:00,  6.41 examples/s]391 examples [00:00,  9.13 examples/s]492 examples [00:00, 12.99 examples/s]594 examples [00:01, 18.45 examples/s]691 examples [00:01, 26.15 examples/s]791 examples [00:01, 36.94 examples/s]889 examples [00:01, 51.92 examples/s]985 examples [00:01, 72.49 examples/s]1082 examples [00:01, 100.34 examples/s]1181 examples [00:01, 137.34 examples/s]1278 examples [00:01, 184.81 examples/s]1375 examples [00:01, 243.70 examples/s]1471 examples [00:01, 312.14 examples/s]1566 examples [00:02, 389.89 examples/s]1660 examples [00:02, 470.61 examples/s]1755 examples [00:02, 554.07 examples/s]1851 examples [00:02, 633.90 examples/s]1948 examples [00:02, 706.63 examples/s]2043 examples [00:02, 755.95 examples/s]2142 examples [00:02, 812.53 examples/s]2237 examples [00:02, 842.35 examples/s]2338 examples [00:02, 885.42 examples/s]2439 examples [00:02, 917.74 examples/s]2542 examples [00:03, 947.76 examples/s]2641 examples [00:03, 952.37 examples/s]2740 examples [00:03, 963.00 examples/s]2839 examples [00:03, 958.75 examples/s]2937 examples [00:03, 957.86 examples/s]3034 examples [00:03, 956.32 examples/s]3131 examples [00:03, 959.42 examples/s]3228 examples [00:03, 946.86 examples/s]3326 examples [00:03, 955.89 examples/s]3424 examples [00:04, 962.14 examples/s]3522 examples [00:04, 966.45 examples/s]3622 examples [00:04, 975.05 examples/s]3722 examples [00:04, 980.32 examples/s]3821 examples [00:04, 973.12 examples/s]3923 examples [00:04, 984.82 examples/s]4026 examples [00:04, 997.52 examples/s]4127 examples [00:04, 998.57 examples/s]4227 examples [00:04, 973.53 examples/s]4325 examples [00:04, 955.39 examples/s]4421 examples [00:05, 944.33 examples/s]4516 examples [00:05, 905.88 examples/s]4608 examples [00:05, 900.18 examples/s]4704 examples [00:05, 915.21 examples/s]4800 examples [00:05, 926.18 examples/s]4893 examples [00:05, 919.89 examples/s]4990 examples [00:05, 933.82 examples/s]5084 examples [00:05, 919.71 examples/s]5180 examples [00:05, 931.11 examples/s]5274 examples [00:05, 933.19 examples/s]5369 examples [00:06, 937.82 examples/s]5463 examples [00:06, 911.83 examples/s]5557 examples [00:06, 918.01 examples/s]5652 examples [00:06, 926.19 examples/s]5750 examples [00:06, 941.04 examples/s]5849 examples [00:06, 954.19 examples/s]5949 examples [00:06, 966.67 examples/s]6046 examples [00:06, 951.58 examples/s]6142 examples [00:06, 942.46 examples/s]6237 examples [00:06, 941.15 examples/s]6332 examples [00:07, 921.04 examples/s]6425 examples [00:07, 906.68 examples/s]6516 examples [00:07, 902.73 examples/s]6608 examples [00:07, 905.22 examples/s]6706 examples [00:07, 924.94 examples/s]6803 examples [00:07, 935.46 examples/s]6897 examples [00:07, 931.08 examples/s]6991 examples [00:07, 932.92 examples/s]7085 examples [00:07, 915.37 examples/s]7179 examples [00:08, 922.19 examples/s]7273 examples [00:08, 926.25 examples/s]7370 examples [00:08, 936.87 examples/s]7464 examples [00:08, 933.99 examples/s]7559 examples [00:08, 937.04 examples/s]7654 examples [00:08, 938.41 examples/s]7749 examples [00:08, 941.49 examples/s]7847 examples [00:08, 952.06 examples/s]7943 examples [00:08, 948.13 examples/s]8040 examples [00:08, 954.14 examples/s]8137 examples [00:09, 949.12 examples/s]8232 examples [00:09, 949.18 examples/s]8328 examples [00:09, 952.09 examples/s]8424 examples [00:09, 949.40 examples/s]8523 examples [00:09, 957.85 examples/s]8619 examples [00:09, 953.16 examples/s]8725 examples [00:09, 982.19 examples/s]8828 examples [00:09, 993.67 examples/s]8928 examples [00:09, 980.52 examples/s]9027 examples [00:09, 969.07 examples/s]9125 examples [00:10, 966.78 examples/s]9222 examples [00:10, 965.72 examples/s]9319 examples [00:10, 956.05 examples/s]9416 examples [00:10, 958.62 examples/s]9514 examples [00:10, 963.95 examples/s]9613 examples [00:10, 970.27 examples/s]9713 examples [00:10, 977.38 examples/s]9814 examples [00:10, 986.27 examples/s]9913 examples [00:10, 949.72 examples/s]10009 examples [00:10, 895.81 examples/s]10109 examples [00:11, 921.51 examples/s]10205 examples [00:11, 932.32 examples/s]10299 examples [00:11, 934.28 examples/s]10400 examples [00:11, 955.77 examples/s]10496 examples [00:11, 955.87 examples/s]10595 examples [00:11, 964.89 examples/s]10692 examples [00:11, 958.35 examples/s]10788 examples [00:11, 954.30 examples/s]10884 examples [00:11, 950.80 examples/s]10981 examples [00:11, 954.97 examples/s]11087 examples [00:12, 983.89 examples/s]11192 examples [00:12, 999.73 examples/s]11296 examples [00:12, 1009.09 examples/s]11398 examples [00:12, 991.03 examples/s] 11498 examples [00:12, 978.08 examples/s]11596 examples [00:12, 968.04 examples/s]11693 examples [00:12, 961.31 examples/s]11793 examples [00:12, 971.45 examples/s]11891 examples [00:12, 940.75 examples/s]11988 examples [00:13, 946.97 examples/s]12086 examples [00:13, 955.82 examples/s]12182 examples [00:13, 956.53 examples/s]12278 examples [00:13, 956.12 examples/s]12377 examples [00:13, 950.58 examples/s]12475 examples [00:13, 957.34 examples/s]12571 examples [00:13, 950.48 examples/s]12667 examples [00:13, 942.63 examples/s]12768 examples [00:13, 960.09 examples/s]12865 examples [00:13, 946.55 examples/s]12961 examples [00:14, 947.97 examples/s]13061 examples [00:14, 962.92 examples/s]13158 examples [00:14, 964.17 examples/s]13255 examples [00:14, 962.55 examples/s]13354 examples [00:14, 970.08 examples/s]13456 examples [00:14, 982.52 examples/s]13555 examples [00:14, 960.69 examples/s]13654 examples [00:14, 966.74 examples/s]13751 examples [00:14, 949.51 examples/s]13848 examples [00:14, 954.29 examples/s]13956 examples [00:15, 986.55 examples/s]14056 examples [00:15, 976.40 examples/s]14156 examples [00:15, 980.71 examples/s]14255 examples [00:15, 979.02 examples/s]14354 examples [00:15, 955.19 examples/s]14455 examples [00:15, 969.43 examples/s]14555 examples [00:15, 977.07 examples/s]14653 examples [00:15, 976.93 examples/s]14755 examples [00:15, 987.41 examples/s]14854 examples [00:15, 965.07 examples/s]14956 examples [00:16, 978.30 examples/s]15055 examples [00:16, 970.83 examples/s]15153 examples [00:16, 968.48 examples/s]15252 examples [00:16, 973.62 examples/s]15350 examples [00:16, 956.06 examples/s]15447 examples [00:16, 959.27 examples/s]15544 examples [00:16, 955.34 examples/s]15642 examples [00:16, 960.88 examples/s]15739 examples [00:16, 956.97 examples/s]15835 examples [00:17, 940.17 examples/s]15932 examples [00:17, 947.40 examples/s]16028 examples [00:17, 950.32 examples/s]16124 examples [00:17, 931.57 examples/s]16222 examples [00:17, 939.64 examples/s]16319 examples [00:17, 947.64 examples/s]16418 examples [00:17, 958.31 examples/s]16516 examples [00:17, 964.26 examples/s]16613 examples [00:17, 964.78 examples/s]16711 examples [00:17, 968.59 examples/s]16810 examples [00:18, 973.60 examples/s]16908 examples [00:18, 970.70 examples/s]17006 examples [00:18, 966.27 examples/s]17103 examples [00:18, 966.62 examples/s]17203 examples [00:18, 974.04 examples/s]17301 examples [00:18, 966.40 examples/s]17400 examples [00:18, 971.28 examples/s]17499 examples [00:18, 976.54 examples/s]17600 examples [00:18, 984.44 examples/s]17701 examples [00:18, 990.83 examples/s]17801 examples [00:19, 969.56 examples/s]17899 examples [00:19, 971.66 examples/s]17997 examples [00:19, 972.54 examples/s]18099 examples [00:19, 985.87 examples/s]18199 examples [00:19, 989.88 examples/s]18299 examples [00:19, 978.33 examples/s]18402 examples [00:19, 990.36 examples/s]18502 examples [00:19, 978.74 examples/s]18600 examples [00:19, 977.96 examples/s]18698 examples [00:19, 975.26 examples/s]18796 examples [00:20, 963.51 examples/s]18897 examples [00:20, 976.18 examples/s]18996 examples [00:20, 976.98 examples/s]19098 examples [00:20, 989.50 examples/s]19198 examples [00:20, 979.86 examples/s]19297 examples [00:20, 951.29 examples/s]19394 examples [00:20, 955.30 examples/s]19493 examples [00:20, 964.66 examples/s]19594 examples [00:20, 975.31 examples/s]19694 examples [00:20, 980.47 examples/s]19793 examples [00:21, 979.95 examples/s]19892 examples [00:21, 965.90 examples/s]19990 examples [00:21, 969.69 examples/s]20088 examples [00:21, 934.91 examples/s]20182 examples [00:21, 931.53 examples/s]20278 examples [00:21, 938.33 examples/s]20373 examples [00:21, 940.27 examples/s]20469 examples [00:21, 944.28 examples/s]20564 examples [00:21, 935.51 examples/s]20663 examples [00:22, 949.39 examples/s]20760 examples [00:22, 954.72 examples/s]20860 examples [00:22, 967.50 examples/s]20960 examples [00:22, 974.80 examples/s]21058 examples [00:22, 967.93 examples/s]21155 examples [00:22, 940.49 examples/s]21250 examples [00:22, 941.65 examples/s]21351 examples [00:22, 959.21 examples/s]21448 examples [00:22, 960.15 examples/s]21545 examples [00:22, 920.30 examples/s]21640 examples [00:23, 926.46 examples/s]21737 examples [00:23, 937.21 examples/s]21837 examples [00:23, 955.04 examples/s]21934 examples [00:23, 958.74 examples/s]22031 examples [00:23, 936.91 examples/s]22127 examples [00:23, 942.74 examples/s]22226 examples [00:23, 954.47 examples/s]22328 examples [00:23, 970.35 examples/s]22427 examples [00:23, 974.24 examples/s]22525 examples [00:23, 974.57 examples/s]22624 examples [00:24, 978.31 examples/s]22723 examples [00:24, 979.47 examples/s]22823 examples [00:24, 984.48 examples/s]22923 examples [00:24, 986.77 examples/s]23022 examples [00:24, 953.31 examples/s]23118 examples [00:24, 951.95 examples/s]23214 examples [00:24, 952.57 examples/s]23315 examples [00:24, 967.81 examples/s]23414 examples [00:24, 974.35 examples/s]23515 examples [00:24, 984.46 examples/s]23614 examples [00:25, 980.21 examples/s]23713 examples [00:25, 967.64 examples/s]23810 examples [00:25, 962.63 examples/s]23909 examples [00:25, 970.37 examples/s]24007 examples [00:25, 969.34 examples/s]24104 examples [00:25, 950.72 examples/s]24200 examples [00:25, 952.44 examples/s]24298 examples [00:25, 958.79 examples/s]24394 examples [00:25, 951.85 examples/s]24492 examples [00:25, 958.01 examples/s]24589 examples [00:26, 959.23 examples/s]24689 examples [00:26, 970.96 examples/s]24787 examples [00:26, 968.15 examples/s]24887 examples [00:26, 977.26 examples/s]24985 examples [00:26, 964.47 examples/s]25082 examples [00:26, 964.65 examples/s]25180 examples [00:26, 968.59 examples/s]25277 examples [00:26, 965.52 examples/s]25376 examples [00:26, 971.41 examples/s]25475 examples [00:27, 976.57 examples/s]25576 examples [00:27, 983.58 examples/s]25675 examples [00:27, 983.46 examples/s]25774 examples [00:27, 984.55 examples/s]25875 examples [00:27, 990.82 examples/s]25975 examples [00:27, 985.39 examples/s]26075 examples [00:27, 987.47 examples/s]26174 examples [00:27, 985.64 examples/s]26275 examples [00:27, 991.48 examples/s]26375 examples [00:27, 990.06 examples/s]26475 examples [00:28, 991.84 examples/s]26575 examples [00:28, 993.53 examples/s]26675 examples [00:28, 990.63 examples/s]26775 examples [00:28, 988.95 examples/s]26874 examples [00:28, 985.96 examples/s]26973 examples [00:28, 982.61 examples/s]27073 examples [00:28, 985.13 examples/s]27172 examples [00:28, 971.04 examples/s]27271 examples [00:28, 975.53 examples/s]27371 examples [00:28, 982.59 examples/s]27470 examples [00:29, 983.18 examples/s]27569 examples [00:29, 971.81 examples/s]27668 examples [00:29, 976.17 examples/s]27768 examples [00:29, 982.98 examples/s]27867 examples [00:29, 980.72 examples/s]27968 examples [00:29, 987.14 examples/s]28067 examples [00:29, 964.13 examples/s]28164 examples [00:29, 964.26 examples/s]28265 examples [00:29, 975.20 examples/s]28364 examples [00:29, 977.65 examples/s]28465 examples [00:30, 985.66 examples/s]28565 examples [00:30, 988.31 examples/s]28664 examples [00:30, 970.46 examples/s]28762 examples [00:30, 963.78 examples/s]28860 examples [00:30, 966.41 examples/s]28957 examples [00:30, 939.54 examples/s]29056 examples [00:30, 953.10 examples/s]29153 examples [00:30, 956.00 examples/s]29253 examples [00:30, 966.68 examples/s]29350 examples [00:30, 964.30 examples/s]29450 examples [00:31, 974.06 examples/s]29549 examples [00:31, 977.21 examples/s]29647 examples [00:31, 949.18 examples/s]29747 examples [00:31, 963.24 examples/s]29847 examples [00:31, 972.02 examples/s]29945 examples [00:31, 960.03 examples/s]30042 examples [00:31, 906.38 examples/s]30141 examples [00:31, 927.81 examples/s]30238 examples [00:31, 938.51 examples/s]30333 examples [00:32, 935.88 examples/s]30431 examples [00:32, 947.27 examples/s]30529 examples [00:32, 955.82 examples/s]30628 examples [00:32, 963.47 examples/s]30728 examples [00:32, 974.09 examples/s]30827 examples [00:32, 975.86 examples/s]30927 examples [00:32, 980.37 examples/s]31026 examples [00:32, 970.96 examples/s]31124 examples [00:32, 971.85 examples/s]31223 examples [00:32, 974.02 examples/s]31323 examples [00:33, 981.56 examples/s]31422 examples [00:33, 981.06 examples/s]31521 examples [00:33, 968.59 examples/s]31619 examples [00:33, 970.59 examples/s]31717 examples [00:33, 970.95 examples/s]31815 examples [00:33, 968.03 examples/s]31914 examples [00:33, 972.78 examples/s]32012 examples [00:33, 953.78 examples/s]32109 examples [00:33, 957.15 examples/s]32207 examples [00:33, 963.14 examples/s]32306 examples [00:34, 970.84 examples/s]32405 examples [00:34, 975.96 examples/s]32503 examples [00:34, 964.35 examples/s]32600 examples [00:34, 964.65 examples/s]32697 examples [00:34, 964.53 examples/s]32798 examples [00:34, 976.27 examples/s]32896 examples [00:34, 972.21 examples/s]32994 examples [00:34, 964.78 examples/s]33094 examples [00:34, 973.60 examples/s]33195 examples [00:34, 980.78 examples/s]33294 examples [00:35, 960.92 examples/s]33392 examples [00:35, 964.79 examples/s]33489 examples [00:35, 952.91 examples/s]33586 examples [00:35, 955.21 examples/s]33682 examples [00:35, 954.79 examples/s]33779 examples [00:35, 958.11 examples/s]33875 examples [00:35, 950.11 examples/s]33971 examples [00:35, 943.03 examples/s]34066 examples [00:35, 943.81 examples/s]34161 examples [00:35, 945.31 examples/s]34260 examples [00:36, 957.87 examples/s]34359 examples [00:36, 964.48 examples/s]34457 examples [00:36, 967.76 examples/s]34554 examples [00:36, 962.58 examples/s]34651 examples [00:36, 964.12 examples/s]34749 examples [00:36, 966.39 examples/s]34846 examples [00:36, 965.21 examples/s]34943 examples [00:36, 956.73 examples/s]35040 examples [00:36, 960.33 examples/s]35140 examples [00:36, 971.86 examples/s]35238 examples [00:37, 965.94 examples/s]35337 examples [00:37, 972.06 examples/s]35435 examples [00:37, 962.69 examples/s]35533 examples [00:37, 965.59 examples/s]35630 examples [00:37, 961.60 examples/s]35728 examples [00:37, 966.40 examples/s]35825 examples [00:37, 967.11 examples/s]35923 examples [00:37, 968.49 examples/s]36020 examples [00:37, 960.96 examples/s]36118 examples [00:38, 965.96 examples/s]36216 examples [00:38, 968.98 examples/s]36316 examples [00:38, 977.83 examples/s]36415 examples [00:38, 980.81 examples/s]36514 examples [00:38, 981.07 examples/s]36613 examples [00:38, 972.94 examples/s]36715 examples [00:38, 983.75 examples/s]36814 examples [00:38, 966.57 examples/s]36911 examples [00:38, 964.80 examples/s]37009 examples [00:38, 967.13 examples/s]37106 examples [00:39, 943.28 examples/s]37202 examples [00:39, 946.47 examples/s]37300 examples [00:39, 954.15 examples/s]37398 examples [00:39, 959.91 examples/s]37496 examples [00:39, 963.67 examples/s]37593 examples [00:39, 962.45 examples/s]37696 examples [00:39, 979.60 examples/s]37798 examples [00:39, 989.11 examples/s]37898 examples [00:39, 984.76 examples/s]37997 examples [00:39, 982.92 examples/s]38098 examples [00:40, 989.24 examples/s]38199 examples [00:40, 994.61 examples/s]38299 examples [00:40, 983.60 examples/s]38398 examples [00:40, 974.36 examples/s]38496 examples [00:40, 962.38 examples/s]38593 examples [00:40, 954.95 examples/s]38693 examples [00:40, 966.43 examples/s]38790 examples [00:40, 964.78 examples/s]38887 examples [00:40, 950.22 examples/s]38983 examples [00:40, 933.28 examples/s]39077 examples [00:41, 932.63 examples/s]39176 examples [00:41, 947.10 examples/s]39271 examples [00:41, 947.25 examples/s]39366 examples [00:41, 938.55 examples/s]39460 examples [00:41, 887.56 examples/s]39557 examples [00:41, 909.93 examples/s]39653 examples [00:41, 922.92 examples/s]39748 examples [00:41, 928.39 examples/s]39842 examples [00:41, 917.89 examples/s]39939 examples [00:41, 931.61 examples/s]40033 examples [00:42, 899.45 examples/s]40129 examples [00:42, 916.55 examples/s]40225 examples [00:42, 928.86 examples/s]40320 examples [00:42, 933.69 examples/s]40419 examples [00:42, 948.61 examples/s]40516 examples [00:42, 953.05 examples/s]40616 examples [00:42, 964.51 examples/s]40718 examples [00:42, 980.42 examples/s]40817 examples [00:42, 974.20 examples/s]40916 examples [00:43, 976.96 examples/s]41014 examples [00:43, 973.20 examples/s]41112 examples [00:43, 974.90 examples/s]41210 examples [00:43, 965.29 examples/s]41307 examples [00:43, 961.41 examples/s]41404 examples [00:43, 962.54 examples/s]41501 examples [00:43, 955.88 examples/s]41597 examples [00:43, 956.48 examples/s]41693 examples [00:43, 956.84 examples/s]41790 examples [00:43, 958.27 examples/s]41886 examples [00:44, 953.89 examples/s]41985 examples [00:44, 964.42 examples/s]42083 examples [00:44, 967.82 examples/s]42180 examples [00:44, 967.07 examples/s]42277 examples [00:44, 950.28 examples/s]42376 examples [00:44, 960.26 examples/s]42473 examples [00:44, 956.21 examples/s]42570 examples [00:44, 958.56 examples/s]42669 examples [00:44, 965.32 examples/s]42766 examples [00:44, 953.82 examples/s]42862 examples [00:45, 949.38 examples/s]42957 examples [00:45, 937.06 examples/s]43056 examples [00:45, 950.75 examples/s]43152 examples [00:45, 951.46 examples/s]43248 examples [00:45, 938.83 examples/s]43347 examples [00:45, 952.06 examples/s]43446 examples [00:45, 960.43 examples/s]43543 examples [00:45, 955.86 examples/s]43640 examples [00:45, 958.38 examples/s]43736 examples [00:45, 951.05 examples/s]43832 examples [00:46, 946.49 examples/s]43927 examples [00:46, 941.93 examples/s]44024 examples [00:46, 949.49 examples/s]44119 examples [00:46, 931.95 examples/s]44213 examples [00:46, 928.20 examples/s]44306 examples [00:46, 919.73 examples/s]44399 examples [00:46, 912.44 examples/s]44497 examples [00:46, 929.36 examples/s]44595 examples [00:46, 941.36 examples/s]44690 examples [00:46, 924.70 examples/s]44785 examples [00:47, 931.01 examples/s]44884 examples [00:47, 947.75 examples/s]44984 examples [00:47, 960.77 examples/s]45082 examples [00:47, 963.86 examples/s]45179 examples [00:47, 941.13 examples/s]45277 examples [00:47, 949.99 examples/s]45377 examples [00:47, 963.43 examples/s]45474 examples [00:47, 963.50 examples/s]45575 examples [00:47, 976.50 examples/s]45676 examples [00:48, 983.65 examples/s]45775 examples [00:48, 975.35 examples/s]45873 examples [00:48, 964.61 examples/s]45970 examples [00:48, 963.34 examples/s]46068 examples [00:48, 967.49 examples/s]46165 examples [00:48, 955.83 examples/s]46261 examples [00:48, 915.72 examples/s]46355 examples [00:48, 920.40 examples/s]46451 examples [00:48, 930.55 examples/s]46547 examples [00:48, 937.83 examples/s]46645 examples [00:49, 948.52 examples/s]46741 examples [00:49, 946.61 examples/s]46840 examples [00:49, 958.22 examples/s]46936 examples [00:49, 953.34 examples/s]47032 examples [00:49, 944.85 examples/s]47129 examples [00:49, 949.53 examples/s]47225 examples [00:49, 949.41 examples/s]47323 examples [00:49, 956.68 examples/s]47419 examples [00:49, 954.21 examples/s]47516 examples [00:49, 958.49 examples/s]47612 examples [00:50, 958.76 examples/s]47708 examples [00:50, 952.97 examples/s]47804 examples [00:50, 949.13 examples/s]47901 examples [00:50, 953.32 examples/s]47997 examples [00:50, 940.80 examples/s]48094 examples [00:50, 948.54 examples/s]48189 examples [00:50, 943.79 examples/s]48285 examples [00:50, 944.72 examples/s]48380 examples [00:50, 944.01 examples/s]48476 examples [00:50, 946.59 examples/s]48571 examples [00:51, 946.92 examples/s]48666 examples [00:51, 945.82 examples/s]48763 examples [00:51, 952.22 examples/s]48859 examples [00:51, 943.28 examples/s]48954 examples [00:51, 933.17 examples/s]49050 examples [00:51, 939.28 examples/s]49144 examples [00:51, 935.64 examples/s]49240 examples [00:51, 942.17 examples/s]49336 examples [00:51, 945.11 examples/s]49433 examples [00:51, 950.57 examples/s]49530 examples [00:52, 953.63 examples/s]49626 examples [00:52, 954.05 examples/s]49724 examples [00:52, 958.97 examples/s]49821 examples [00:52, 961.15 examples/s]49920 examples [00:52, 968.70 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7394/50000 [00:00<00:00, 73938.48 examples/s] 41%|████      | 20441/50000 [00:00<00:00, 84985.47 examples/s] 67%|██████▋   | 33415/50000 [00:00<00:00, 94795.12 examples/s] 93%|█████████▎| 46375/50000 [00:00<00:00, 103101.43 examples/s]                                                                0 examples [00:00, ? examples/s]85 examples [00:00, 848.97 examples/s]183 examples [00:00, 883.71 examples/s]283 examples [00:00, 914.48 examples/s]381 examples [00:00, 931.02 examples/s]480 examples [00:00, 946.24 examples/s]579 examples [00:00, 958.58 examples/s]675 examples [00:00, 957.23 examples/s]773 examples [00:00, 962.62 examples/s]872 examples [00:00, 970.35 examples/s]973 examples [00:01, 980.28 examples/s]1071 examples [00:01, 978.31 examples/s]1168 examples [00:01, 968.61 examples/s]1268 examples [00:01, 976.57 examples/s]1368 examples [00:01, 982.74 examples/s]1467 examples [00:01, 984.01 examples/s]1566 examples [00:01, 975.47 examples/s]1664 examples [00:01, 961.10 examples/s]1766 examples [00:01, 977.12 examples/s]1867 examples [00:01, 985.58 examples/s]1966 examples [00:02, 986.31 examples/s]2065 examples [00:02, 985.22 examples/s]2164 examples [00:02, 979.88 examples/s]2262 examples [00:02, 979.71 examples/s]2360 examples [00:02, 977.17 examples/s]2461 examples [00:02, 985.24 examples/s]2560 examples [00:02, 978.94 examples/s]2658 examples [00:02, 978.07 examples/s]2758 examples [00:02, 983.12 examples/s]2857 examples [00:02, 966.18 examples/s]2956 examples [00:03, 972.16 examples/s]3054 examples [00:03, 972.35 examples/s]3152 examples [00:03, 964.41 examples/s]3253 examples [00:03, 975.88 examples/s]3351 examples [00:03, 959.64 examples/s]3450 examples [00:03, 966.95 examples/s]3549 examples [00:03, 972.18 examples/s]3647 examples [00:03, 949.78 examples/s]3745 examples [00:03, 956.20 examples/s]3842 examples [00:03, 959.44 examples/s]3939 examples [00:04, 958.67 examples/s]4036 examples [00:04, 960.03 examples/s]4134 examples [00:04, 964.63 examples/s]4231 examples [00:04, 953.26 examples/s]4330 examples [00:04, 963.81 examples/s]4433 examples [00:04, 981.99 examples/s]4535 examples [00:04, 990.83 examples/s]4635 examples [00:04, 979.39 examples/s]4734 examples [00:04, 978.79 examples/s]4834 examples [00:04, 984.84 examples/s]4936 examples [00:05, 993.26 examples/s]5036 examples [00:05, 993.38 examples/s]5136 examples [00:05, 982.57 examples/s]5238 examples [00:05, 992.95 examples/s]5338 examples [00:05, 987.85 examples/s]5440 examples [00:05, 995.03 examples/s]5544 examples [00:05, 1005.37 examples/s]5645 examples [00:05, 998.89 examples/s] 5745 examples [00:05, 988.84 examples/s]5844 examples [00:05, 989.00 examples/s]5946 examples [00:06, 994.96 examples/s]6046 examples [00:06, 988.59 examples/s]6145 examples [00:06, 981.18 examples/s]6244 examples [00:06, 982.38 examples/s]6345 examples [00:06, 987.78 examples/s]6444 examples [00:06, 983.67 examples/s]6543 examples [00:06, 976.54 examples/s]6641 examples [00:06, 970.78 examples/s]6740 examples [00:06, 974.04 examples/s]6841 examples [00:07, 983.33 examples/s]6941 examples [00:07, 986.66 examples/s]7040 examples [00:07, 982.05 examples/s]7139 examples [00:07, 978.27 examples/s]7238 examples [00:07, 979.04 examples/s]7336 examples [00:07, 975.33 examples/s]7434 examples [00:07, 974.49 examples/s]7532 examples [00:07, 965.07 examples/s]7629 examples [00:07, 962.11 examples/s]7729 examples [00:07, 972.48 examples/s]7827 examples [00:08, 961.14 examples/s]7925 examples [00:08, 963.89 examples/s]8022 examples [00:08, 952.73 examples/s]8118 examples [00:08, 945.41 examples/s]8213 examples [00:08, 940.85 examples/s]8308 examples [00:08, 942.45 examples/s]8408 examples [00:08, 956.29 examples/s]8509 examples [00:08, 970.57 examples/s]8607 examples [00:08, 971.65 examples/s]8705 examples [00:08, 964.59 examples/s]8802 examples [00:09, 962.74 examples/s]8899 examples [00:09, 958.96 examples/s]8997 examples [00:09, 963.32 examples/s]9095 examples [00:09, 965.07 examples/s]9193 examples [00:09, 966.85 examples/s]9291 examples [00:09, 969.87 examples/s]9389 examples [00:09, 972.56 examples/s]9487 examples [00:09, 958.85 examples/s]9583 examples [00:09, 954.35 examples/s]9683 examples [00:09, 965.69 examples/s]9784 examples [00:10, 977.09 examples/s]9882 examples [00:10, 967.31 examples/s]9980 examples [00:10, 967.62 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteX0GHEZ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteX0GHEZ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7efd9dad6a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7efd9dad6a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7efd9dad6a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efde94430f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efd28b46ef0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7efd9dad6a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7efd9dad6a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7efd9dad6a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efd8598b588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efd8598b470>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.22 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 23.83 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.22 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.22 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.70 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.70 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.16 file/s]2020-08-03 18:12:17.582613: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-03 18:12:17.586465: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-08-03 18:12:17.586620: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560bc3569860 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-03 18:12:17.586638: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist2', 'test', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  8%|▊         | 811008/9912422 [00:00<00:01, 6369324.35it/s] 97%|█████████▋| 9625600/9912422 [00:00<00:00, 8760754.34it/s]9920512it [00:00, 18290105.79it/s]                            
0it [00:00, ?it/s]32768it [00:00, 528343.12it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 467137.29it/s]1654784it [00:00, 11835759.91it/s]                         
0it [00:00, ?it/s]8192it [00:00, 198388.73it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

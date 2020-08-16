
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f6ce25e5598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f6ce25e5598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f6d4d297400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f6d4d297400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f6d6c4b3ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f6d6c4b3ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6cfa5d20d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6cfa5d20d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f6cfa5d20d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 26%|██▌       | 2539520/9912422 [00:00<00:00, 25391410.33it/s]9920512it [00:00, 30647403.91it/s]                             
0it [00:00, ?it/s]32768it [00:00, 489242.40it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:13, 120938.36it/s]1654784it [00:00, 9189394.75it/s]                          
0it [00:00, ?it/s]8192it [00:00, 154745.71it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6cf86334e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6cf8abea20>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f6cfa5cad08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f6cfa5cad08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f6cfa5cad08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:01,  2.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:01,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:01,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:01,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:00,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:00,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:43,  3.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:43,  3.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:42,  3.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:42,  3.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:42,  3.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:42,  3.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:00<00:30,  4.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:30,  4.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:30,  4.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:29,  4.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:29,  4.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:29,  4.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:21,  6.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:21,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:21,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:21,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:21,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:20,  6.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:15,  9.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:15,  9.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:15,  9.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:15,  9.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:15,  9.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:11, 11.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:11, 11.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:11, 11.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:11, 11.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:11, 11.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:11, 11.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:01<00:08, 15.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:08, 15.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:08, 15.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:08, 15.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:08, 15.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:08, 15.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:06, 18.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:06, 18.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:06, 18.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:06, 18.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:06, 18.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:06, 18.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:05, 22.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:05, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:05, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:05, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:05, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:05, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:04, 26.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:04, 26.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:04, 26.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 26.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 26.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 26.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:03, 29.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:03, 29.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:03, 29.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:03, 29.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:03, 29.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 29.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 29.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 29.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 29.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 29.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:02, 37.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 44.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:01, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:01, 50.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:01, 50.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 50.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 50.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 50.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 50.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 50.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 50.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 50.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 56.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 62.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:00, 66.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 66.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 66.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 67.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 67.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 67.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 67.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 67.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 67.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 67.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 67.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 67.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 68.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 70.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 70.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 70.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 70.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 73.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 73.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 73.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 73.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 74.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 74.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.02s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 74.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 74.12 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.89s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  3.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 74.12 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.89s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.89s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.12 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.89s/ url]
0 examples [00:00, ? examples/s]2020-08-16 18:06:59.279893: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-16 18:06:59.298230: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095094999 Hz
2020-08-16 18:06:59.298494: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c62e3c5130 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-16 18:06:59.298515: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.68 examples/s]123 examples [00:00,  3.83 examples/s]244 examples [00:00,  5.46 examples/s]368 examples [00:00,  7.78 examples/s]493 examples [00:00, 11.09 examples/s]613 examples [00:00, 15.78 examples/s]736 examples [00:00, 22.42 examples/s]857 examples [00:01, 31.78 examples/s]980 examples [00:01, 44.90 examples/s]1105 examples [00:01, 63.16 examples/s]1224 examples [00:01, 88.22 examples/s]1343 examples [00:01, 121.88 examples/s]1462 examples [00:01, 166.76 examples/s]1579 examples [00:01, 224.34 examples/s]1702 examples [00:01, 297.11 examples/s]1821 examples [00:01, 382.67 examples/s]1943 examples [00:01, 481.54 examples/s]2063 examples [00:02, 586.34 examples/s]2188 examples [00:02, 697.34 examples/s]2311 examples [00:02, 801.38 examples/s]2433 examples [00:02, 890.21 examples/s]2554 examples [00:02, 950.79 examples/s]2681 examples [00:02, 1026.42 examples/s]2807 examples [00:02, 1084.81 examples/s]2930 examples [00:02, 1112.36 examples/s]3051 examples [00:02, 1122.01 examples/s]3173 examples [00:03, 1149.34 examples/s]3299 examples [00:03, 1179.31 examples/s]3422 examples [00:03, 1193.24 examples/s]3552 examples [00:03, 1222.98 examples/s]3680 examples [00:03, 1236.86 examples/s]3806 examples [00:03, 1235.35 examples/s]3931 examples [00:03, 1230.80 examples/s]4060 examples [00:03, 1245.54 examples/s]4186 examples [00:03, 1242.72 examples/s]4313 examples [00:03, 1247.80 examples/s]4439 examples [00:04, 1241.83 examples/s]4567 examples [00:04, 1251.70 examples/s]4697 examples [00:04, 1263.77 examples/s]4824 examples [00:04, 1261.08 examples/s]4951 examples [00:04, 1249.94 examples/s]5077 examples [00:04, 1242.57 examples/s]5203 examples [00:04, 1247.30 examples/s]5328 examples [00:04, 1243.02 examples/s]5455 examples [00:04, 1248.56 examples/s]5582 examples [00:04, 1254.41 examples/s]5708 examples [00:05, 1252.26 examples/s]5835 examples [00:05, 1255.62 examples/s]5962 examples [00:05, 1257.08 examples/s]6088 examples [00:05, 1252.53 examples/s]6214 examples [00:05, 1235.72 examples/s]6338 examples [00:05, 1215.66 examples/s]6460 examples [00:05, 1201.61 examples/s]6581 examples [00:05, 1172.74 examples/s]6699 examples [00:05, 1173.63 examples/s]6817 examples [00:05, 1171.23 examples/s]6938 examples [00:06, 1180.07 examples/s]7060 examples [00:06, 1189.12 examples/s]7181 examples [00:06, 1195.01 examples/s]7301 examples [00:06, 1193.37 examples/s]7424 examples [00:06, 1203.26 examples/s]7545 examples [00:06, 1186.54 examples/s]7664 examples [00:06, 1174.34 examples/s]7782 examples [00:06, 1168.15 examples/s]7899 examples [00:06, 1159.20 examples/s]8017 examples [00:06, 1163.12 examples/s]8134 examples [00:07, 1159.52 examples/s]8253 examples [00:07, 1165.86 examples/s]8370 examples [00:07, 1159.14 examples/s]8493 examples [00:07, 1179.43 examples/s]8612 examples [00:07, 1182.27 examples/s]8731 examples [00:07, 1175.67 examples/s]8854 examples [00:07, 1189.48 examples/s]8974 examples [00:07, 1188.07 examples/s]9093 examples [00:07, 1164.15 examples/s]9216 examples [00:07, 1181.44 examples/s]9335 examples [00:08, 1170.38 examples/s]9459 examples [00:08, 1189.85 examples/s]9583 examples [00:08, 1203.60 examples/s]9704 examples [00:08, 1201.88 examples/s]9829 examples [00:08, 1215.55 examples/s]9951 examples [00:08, 1215.94 examples/s]10073 examples [00:08, 1155.61 examples/s]10195 examples [00:08, 1172.76 examples/s]10321 examples [00:08, 1195.10 examples/s]10447 examples [00:09, 1213.19 examples/s]10572 examples [00:09, 1221.07 examples/s]10698 examples [00:09, 1231.19 examples/s]10823 examples [00:09, 1234.31 examples/s]10947 examples [00:09, 1234.12 examples/s]11072 examples [00:09, 1238.17 examples/s]11196 examples [00:09, 1235.38 examples/s]11320 examples [00:09, 1232.38 examples/s]11444 examples [00:09, 1221.83 examples/s]11569 examples [00:09, 1227.69 examples/s]11692 examples [00:10, 1224.68 examples/s]11815 examples [00:10, 1218.38 examples/s]11941 examples [00:10, 1228.06 examples/s]12068 examples [00:10, 1239.39 examples/s]12192 examples [00:10, 1234.08 examples/s]12318 examples [00:10, 1240.16 examples/s]12443 examples [00:10, 1237.91 examples/s]12568 examples [00:10, 1238.89 examples/s]12692 examples [00:10, 1237.49 examples/s]12816 examples [00:10, 1231.58 examples/s]12943 examples [00:11, 1240.44 examples/s]13068 examples [00:11, 1242.27 examples/s]13193 examples [00:11, 1235.72 examples/s]13320 examples [00:11, 1243.04 examples/s]13445 examples [00:11, 1235.96 examples/s]13569 examples [00:11, 1236.80 examples/s]13693 examples [00:11, 1224.56 examples/s]13816 examples [00:11, 1219.41 examples/s]13938 examples [00:11, 1211.46 examples/s]14060 examples [00:11, 1202.44 examples/s]14181 examples [00:12, 1197.50 examples/s]14301 examples [00:12, 1186.69 examples/s]14420 examples [00:12, 1174.30 examples/s]14541 examples [00:12, 1182.33 examples/s]14664 examples [00:12, 1194.48 examples/s]14787 examples [00:12, 1203.50 examples/s]14909 examples [00:12, 1206.20 examples/s]15032 examples [00:12, 1213.08 examples/s]15156 examples [00:12, 1220.75 examples/s]15279 examples [00:12, 1223.23 examples/s]15407 examples [00:13, 1238.44 examples/s]15531 examples [00:13, 1236.90 examples/s]15655 examples [00:13, 1219.99 examples/s]15778 examples [00:13, 1213.19 examples/s]15901 examples [00:13, 1215.60 examples/s]16023 examples [00:13, 1212.31 examples/s]16147 examples [00:13, 1217.73 examples/s]16269 examples [00:13, 1213.93 examples/s]16392 examples [00:13, 1217.99 examples/s]16516 examples [00:13, 1223.01 examples/s]16642 examples [00:14, 1231.63 examples/s]16768 examples [00:14, 1239.83 examples/s]16893 examples [00:14, 1184.71 examples/s]17015 examples [00:14, 1194.59 examples/s]17138 examples [00:14, 1202.59 examples/s]17262 examples [00:14, 1212.56 examples/s]17385 examples [00:14, 1216.93 examples/s]17507 examples [00:14, 1192.63 examples/s]17628 examples [00:14, 1195.24 examples/s]17752 examples [00:15, 1207.45 examples/s]17873 examples [00:15, 1192.89 examples/s]18001 examples [00:15, 1214.90 examples/s]18123 examples [00:15, 1206.51 examples/s]18244 examples [00:15, 1206.73 examples/s]18365 examples [00:15, 1205.37 examples/s]18487 examples [00:15, 1209.60 examples/s]18610 examples [00:15, 1214.10 examples/s]18732 examples [00:15, 1200.59 examples/s]18853 examples [00:15, 1197.21 examples/s]18973 examples [00:16, 1194.81 examples/s]19094 examples [00:16, 1197.70 examples/s]19218 examples [00:16, 1208.48 examples/s]19339 examples [00:16, 1204.13 examples/s]19461 examples [00:16, 1206.95 examples/s]19582 examples [00:16, 1190.79 examples/s]19702 examples [00:16, 1185.75 examples/s]19826 examples [00:16, 1201.36 examples/s]19947 examples [00:16, 1194.70 examples/s]20067 examples [00:16, 1137.11 examples/s]20183 examples [00:17, 1141.77 examples/s]20302 examples [00:17, 1153.48 examples/s]20418 examples [00:17, 1150.77 examples/s]20534 examples [00:17, 1141.44 examples/s]20649 examples [00:17, 1140.61 examples/s]20764 examples [00:17, 1136.29 examples/s]20885 examples [00:17, 1156.44 examples/s]21006 examples [00:17, 1171.89 examples/s]21124 examples [00:17, 1162.36 examples/s]21241 examples [00:17, 1162.84 examples/s]21363 examples [00:18, 1178.84 examples/s]21483 examples [00:18, 1182.51 examples/s]21607 examples [00:18, 1198.54 examples/s]21727 examples [00:18, 1197.84 examples/s]21847 examples [00:18, 1196.01 examples/s]21969 examples [00:18, 1200.94 examples/s]22094 examples [00:18, 1214.86 examples/s]22218 examples [00:18, 1221.23 examples/s]22341 examples [00:18, 1218.03 examples/s]22466 examples [00:18, 1225.90 examples/s]22589 examples [00:19, 1220.46 examples/s]22712 examples [00:19, 1221.19 examples/s]22837 examples [00:19, 1228.42 examples/s]22960 examples [00:19, 1216.64 examples/s]23086 examples [00:19, 1227.46 examples/s]23209 examples [00:19, 1219.71 examples/s]23333 examples [00:19, 1221.39 examples/s]23456 examples [00:19, 1217.99 examples/s]23578 examples [00:19, 1202.11 examples/s]23702 examples [00:19, 1211.46 examples/s]23826 examples [00:20, 1219.77 examples/s]23953 examples [00:20, 1232.38 examples/s]24077 examples [00:20, 1211.80 examples/s]24199 examples [00:20, 1208.96 examples/s]24321 examples [00:20, 1210.24 examples/s]24445 examples [00:20, 1218.22 examples/s]24569 examples [00:20, 1221.95 examples/s]24692 examples [00:20, 1217.25 examples/s]24814 examples [00:20, 1207.58 examples/s]24937 examples [00:21, 1213.98 examples/s]25059 examples [00:21, 1212.66 examples/s]25183 examples [00:21, 1218.28 examples/s]25307 examples [00:21, 1222.58 examples/s]25430 examples [00:21, 1208.00 examples/s]25551 examples [00:21, 1206.07 examples/s]25672 examples [00:21, 1195.49 examples/s]25792 examples [00:21, 1176.50 examples/s]25913 examples [00:21, 1184.96 examples/s]26035 examples [00:21, 1193.56 examples/s]26160 examples [00:22, 1207.73 examples/s]26281 examples [00:22, 1173.45 examples/s]26406 examples [00:22, 1194.98 examples/s]26531 examples [00:22, 1209.61 examples/s]26653 examples [00:22, 1206.59 examples/s]26774 examples [00:22, 1205.38 examples/s]26896 examples [00:22, 1207.80 examples/s]27017 examples [00:22, 1204.58 examples/s]27138 examples [00:22, 1190.12 examples/s]27258 examples [00:22, 1177.94 examples/s]27376 examples [00:23, 1169.53 examples/s]27498 examples [00:23, 1183.71 examples/s]27620 examples [00:23, 1193.49 examples/s]27742 examples [00:23, 1199.44 examples/s]27863 examples [00:23, 1190.55 examples/s]27985 examples [00:23, 1198.69 examples/s]28108 examples [00:23, 1205.60 examples/s]28230 examples [00:23, 1208.51 examples/s]28351 examples [00:23, 1206.49 examples/s]28472 examples [00:23, 1202.53 examples/s]28600 examples [00:24, 1223.32 examples/s]28725 examples [00:24, 1230.00 examples/s]28853 examples [00:24, 1242.29 examples/s]28978 examples [00:24, 1239.17 examples/s]29102 examples [00:24, 1226.77 examples/s]29225 examples [00:24, 1227.62 examples/s]29348 examples [00:24, 1223.74 examples/s]29471 examples [00:24, 1214.73 examples/s]29593 examples [00:24, 1215.53 examples/s]29715 examples [00:24, 1206.52 examples/s]29841 examples [00:25, 1220.67 examples/s]29967 examples [00:25, 1230.85 examples/s]30091 examples [00:25, 1173.39 examples/s]30213 examples [00:25, 1184.67 examples/s]30332 examples [00:25, 1182.86 examples/s]30457 examples [00:25, 1201.40 examples/s]30583 examples [00:25, 1217.55 examples/s]30706 examples [00:25, 1214.48 examples/s]30828 examples [00:25, 1211.60 examples/s]30951 examples [00:25, 1216.65 examples/s]31076 examples [00:26, 1224.42 examples/s]31200 examples [00:26, 1227.79 examples/s]31323 examples [00:26, 1216.11 examples/s]31445 examples [00:26, 1188.86 examples/s]31566 examples [00:26, 1193.29 examples/s]31691 examples [00:26, 1208.75 examples/s]31813 examples [00:26, 1203.82 examples/s]31934 examples [00:26, 1202.25 examples/s]32055 examples [00:26, 1203.71 examples/s]32176 examples [00:27, 1202.45 examples/s]32297 examples [00:27, 1185.43 examples/s]32416 examples [00:27, 1185.23 examples/s]32535 examples [00:27, 1170.22 examples/s]32660 examples [00:27, 1190.94 examples/s]32780 examples [00:27, 1184.94 examples/s]32902 examples [00:27, 1193.35 examples/s]33025 examples [00:27, 1202.98 examples/s]33150 examples [00:27, 1215.13 examples/s]33272 examples [00:27, 1215.57 examples/s]33394 examples [00:28, 1170.85 examples/s]33512 examples [00:28, 1133.71 examples/s]33627 examples [00:28, 1137.80 examples/s]33747 examples [00:28, 1153.58 examples/s]33867 examples [00:28, 1164.90 examples/s]33987 examples [00:28, 1172.45 examples/s]34111 examples [00:28, 1191.74 examples/s]34238 examples [00:28, 1212.31 examples/s]34366 examples [00:28, 1231.81 examples/s]34490 examples [00:28, 1222.79 examples/s]34613 examples [00:29, 1218.85 examples/s]34739 examples [00:29, 1229.58 examples/s]34866 examples [00:29, 1238.57 examples/s]34992 examples [00:29, 1242.98 examples/s]35117 examples [00:29, 1241.47 examples/s]35242 examples [00:29, 1212.59 examples/s]35364 examples [00:29, 1209.44 examples/s]35489 examples [00:29, 1220.77 examples/s]35612 examples [00:29, 1222.20 examples/s]35735 examples [00:29, 1209.33 examples/s]35857 examples [00:30, 1211.21 examples/s]35982 examples [00:30, 1221.51 examples/s]36108 examples [00:30, 1231.70 examples/s]36233 examples [00:30, 1235.51 examples/s]36358 examples [00:30, 1238.42 examples/s]36482 examples [00:30, 1202.87 examples/s]36606 examples [00:30, 1213.62 examples/s]36728 examples [00:30, 1209.99 examples/s]36850 examples [00:30, 1206.87 examples/s]36971 examples [00:30, 1207.06 examples/s]37092 examples [00:31, 1206.14 examples/s]37213 examples [00:31, 1206.88 examples/s]37334 examples [00:31, 1203.39 examples/s]37457 examples [00:31, 1209.51 examples/s]37580 examples [00:31, 1215.45 examples/s]37702 examples [00:31, 1204.56 examples/s]37827 examples [00:31, 1216.49 examples/s]37949 examples [00:31, 1215.64 examples/s]38071 examples [00:31, 1208.73 examples/s]38194 examples [00:32, 1214.89 examples/s]38316 examples [00:32, 1213.89 examples/s]38438 examples [00:32, 1208.28 examples/s]38561 examples [00:32, 1212.88 examples/s]38683 examples [00:32, 1194.82 examples/s]38805 examples [00:32, 1200.80 examples/s]38926 examples [00:32, 1189.66 examples/s]39052 examples [00:32, 1207.50 examples/s]39173 examples [00:32, 1207.30 examples/s]39295 examples [00:32, 1210.53 examples/s]39417 examples [00:33, 1210.83 examples/s]39539 examples [00:33, 1207.53 examples/s]39663 examples [00:33, 1215.53 examples/s]39785 examples [00:33, 1211.08 examples/s]39907 examples [00:33, 1194.68 examples/s]40027 examples [00:33, 1118.12 examples/s]40145 examples [00:33, 1134.20 examples/s]40264 examples [00:33, 1150.05 examples/s]40380 examples [00:33, 1147.67 examples/s]40503 examples [00:33, 1171.18 examples/s]40625 examples [00:34, 1184.92 examples/s]40746 examples [00:34, 1190.77 examples/s]40868 examples [00:34, 1198.96 examples/s]40991 examples [00:34, 1206.77 examples/s]41112 examples [00:34, 1202.95 examples/s]41236 examples [00:34, 1213.01 examples/s]41358 examples [00:34, 1211.05 examples/s]41482 examples [00:34, 1217.39 examples/s]41606 examples [00:34, 1223.72 examples/s]41729 examples [00:34, 1221.74 examples/s]41852 examples [00:35, 1222.39 examples/s]41975 examples [00:35, 1212.59 examples/s]42100 examples [00:35, 1221.34 examples/s]42223 examples [00:35, 1217.85 examples/s]42347 examples [00:35, 1222.25 examples/s]42470 examples [00:35, 1215.01 examples/s]42592 examples [00:35, 1203.91 examples/s]42716 examples [00:35, 1213.71 examples/s]42838 examples [00:35, 1202.35 examples/s]42959 examples [00:35, 1177.33 examples/s]43077 examples [00:36, 1168.54 examples/s]43196 examples [00:36, 1173.10 examples/s]43321 examples [00:36, 1195.02 examples/s]43446 examples [00:36, 1208.50 examples/s]43570 examples [00:36, 1214.78 examples/s]43692 examples [00:36, 1211.82 examples/s]43814 examples [00:36, 1186.52 examples/s]43933 examples [00:36, 1170.68 examples/s]44051 examples [00:36, 1167.75 examples/s]44171 examples [00:37, 1175.95 examples/s]44291 examples [00:37, 1180.82 examples/s]44410 examples [00:37, 1177.77 examples/s]44528 examples [00:37, 1174.93 examples/s]44651 examples [00:37, 1190.37 examples/s]44774 examples [00:37, 1201.46 examples/s]44895 examples [00:37, 1193.82 examples/s]45015 examples [00:37, 1191.08 examples/s]45140 examples [00:37, 1206.85 examples/s]45265 examples [00:37, 1217.40 examples/s]45388 examples [00:38, 1221.08 examples/s]45511 examples [00:38, 1218.20 examples/s]45633 examples [00:38, 1214.74 examples/s]45758 examples [00:38, 1223.84 examples/s]45881 examples [00:38, 1198.36 examples/s]46002 examples [00:38, 1200.20 examples/s]46123 examples [00:38, 1199.40 examples/s]46244 examples [00:38, 1194.35 examples/s]46364 examples [00:38, 1194.77 examples/s]46484 examples [00:38, 1185.36 examples/s]46603 examples [00:39, 1155.33 examples/s]46722 examples [00:39, 1165.05 examples/s]46841 examples [00:39, 1171.71 examples/s]46963 examples [00:39, 1185.15 examples/s]47082 examples [00:39, 1175.39 examples/s]47200 examples [00:39, 1172.26 examples/s]47318 examples [00:39, 1163.22 examples/s]47435 examples [00:39, 1151.87 examples/s]47551 examples [00:39, 1154.18 examples/s]47667 examples [00:39, 1149.85 examples/s]47783 examples [00:40, 1150.02 examples/s]47899 examples [00:40, 1145.96 examples/s]48019 examples [00:40, 1160.24 examples/s]48142 examples [00:40, 1178.74 examples/s]48263 examples [00:40, 1185.55 examples/s]48385 examples [00:40, 1194.37 examples/s]48507 examples [00:40, 1199.41 examples/s]48628 examples [00:40, 1199.14 examples/s]48749 examples [00:40, 1200.14 examples/s]48873 examples [00:40, 1210.94 examples/s]48995 examples [00:41, 1213.33 examples/s]49119 examples [00:41, 1219.39 examples/s]49241 examples [00:41, 1218.77 examples/s]49363 examples [00:41, 1218.30 examples/s]49485 examples [00:41, 1215.66 examples/s]49607 examples [00:41, 1216.04 examples/s]49729 examples [00:41, 1215.16 examples/s]49851 examples [00:41, 1210.67 examples/s]49973 examples [00:41, 1196.75 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 9098/50000 [00:00<00:00, 90978.35 examples/s] 46%|████▌     | 23034/50000 [00:00<00:00, 101554.53 examples/s] 75%|███████▌  | 37684/50000 [00:00<00:00, 111848.05 examples/s]                                                                0 examples [00:00, ? examples/s]95 examples [00:00, 949.42 examples/s]217 examples [00:00, 1016.35 examples/s]342 examples [00:00, 1075.40 examples/s]463 examples [00:00, 1111.89 examples/s]586 examples [00:00, 1143.20 examples/s]709 examples [00:00, 1166.01 examples/s]817 examples [00:00, 1137.85 examples/s]924 examples [00:00, 1093.44 examples/s]1041 examples [00:00, 1114.37 examples/s]1160 examples [00:01, 1135.17 examples/s]1272 examples [00:01, 1125.44 examples/s]1392 examples [00:01, 1146.07 examples/s]1510 examples [00:01, 1154.80 examples/s]1628 examples [00:01, 1161.92 examples/s]1746 examples [00:01, 1167.02 examples/s]1865 examples [00:01, 1172.94 examples/s]1983 examples [00:01, 1150.65 examples/s]2105 examples [00:01, 1169.07 examples/s]2224 examples [00:01, 1174.24 examples/s]2342 examples [00:02, 1174.07 examples/s]2461 examples [00:02, 1175.86 examples/s]2579 examples [00:02, 1164.61 examples/s]2704 examples [00:02, 1187.52 examples/s]2827 examples [00:02, 1197.45 examples/s]2950 examples [00:02, 1206.86 examples/s]3072 examples [00:02, 1210.74 examples/s]3194 examples [00:02, 1209.09 examples/s]3319 examples [00:02, 1219.45 examples/s]3442 examples [00:02, 1220.50 examples/s]3565 examples [00:03, 1211.28 examples/s]3689 examples [00:03, 1218.22 examples/s]3814 examples [00:03, 1226.91 examples/s]3938 examples [00:03, 1228.85 examples/s]4062 examples [00:03, 1229.99 examples/s]4186 examples [00:03, 1229.64 examples/s]4309 examples [00:03, 1226.85 examples/s]4432 examples [00:03, 1225.78 examples/s]4555 examples [00:03, 1215.07 examples/s]4679 examples [00:03, 1220.64 examples/s]4807 examples [00:04, 1235.77 examples/s]4931 examples [00:04, 1233.45 examples/s]5055 examples [00:04, 1229.86 examples/s]5179 examples [00:04, 1221.19 examples/s]5303 examples [00:04, 1224.28 examples/s]5427 examples [00:04, 1228.13 examples/s]5550 examples [00:04, 1227.66 examples/s]5676 examples [00:04, 1236.16 examples/s]5803 examples [00:04, 1244.79 examples/s]5928 examples [00:04, 1232.28 examples/s]6055 examples [00:05, 1240.83 examples/s]6180 examples [00:05, 1197.43 examples/s]6303 examples [00:05, 1205.29 examples/s]6425 examples [00:05, 1207.96 examples/s]6549 examples [00:05, 1216.81 examples/s]6675 examples [00:05, 1228.89 examples/s]6799 examples [00:05, 1221.00 examples/s]6926 examples [00:05, 1234.29 examples/s]7050 examples [00:05, 1231.67 examples/s]7174 examples [00:05, 1234.04 examples/s]7300 examples [00:06, 1240.13 examples/s]7425 examples [00:06, 1237.59 examples/s]7550 examples [00:06, 1240.70 examples/s]7676 examples [00:06, 1245.59 examples/s]7801 examples [00:06, 1232.80 examples/s]7925 examples [00:06, 1234.17 examples/s]8049 examples [00:06, 1218.62 examples/s]8173 examples [00:06, 1222.60 examples/s]8297 examples [00:06, 1226.34 examples/s]8420 examples [00:06, 1221.94 examples/s]8545 examples [00:07, 1229.83 examples/s]8669 examples [00:07, 1205.48 examples/s]8790 examples [00:07, 1198.70 examples/s]8910 examples [00:07, 1191.80 examples/s]9031 examples [00:07, 1196.24 examples/s]9157 examples [00:07, 1214.21 examples/s]9279 examples [00:07, 1213.01 examples/s]9402 examples [00:07, 1216.87 examples/s]9526 examples [00:07, 1223.19 examples/s]9652 examples [00:07, 1232.26 examples/s]9776 examples [00:08, 1222.25 examples/s]9899 examples [00:08, 1190.47 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteS3R05H/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteS3R05H/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6cfa5d20d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6cfa5d20d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f6cfa5d20d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6d45fb17b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6c808822e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6cfa5d20d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6cfa5d20d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f6cfa5d20d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6cf8abe630>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6cf8abe128>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.16 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 22.86 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.08 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.02 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.02 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.64 file/s]2020-08-16 18:07:56.220939: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-16 18:07:56.224697: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095094999 Hz
2020-08-16 18:07:56.224847: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561454e07b30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-16 18:07:56.224860: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist2', 'train', 'test', 'cifar10', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:04, 153970.11it/s] 55%|█████▍    | 5423104/9912422 [00:00<00:20, 219689.15it/s]9920512it [00:00, 38236039.19it/s]                           
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 188479.09it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 8071444.20it/s]          
0it [00:00, ?it/s]8192it [00:00, 143291.55it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
